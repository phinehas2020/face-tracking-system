"""Global identity resolution over all face evidence in an event.

The online registry is greedy: it decides at crossing time against one mean
template per person and never revisits. This module treats identity as entity
resolution instead. Every enrolled identity and every stored tracklet is a node;
pairwise face similarity becomes calibrated log-odds; and a constrained
agglomerative pass decides, for the whole event at once, which nodes are one
person.

Method, and where each piece comes from:

* Must-link / cannot-link constraints from tracking. Samples in one tracklet are
  one person; two tracklets of the same kind that overlap in time in one camera
  are two people. This is the standard video face clustering prior
  (Kalogeiton & Zisserman, "Constrained Video Face Clustering using 1NN
  Relations", BMVC 2020).
* Clustering on signed log-odds, as in correlation clustering / multicut
  tracking (Tang et al., "Multiple People Tracking by Lifted Multicut and Person
  Re-identification", CVPR 2017). Clusters are merged greedily by mean log-odds,
  highest first.
* Three-way decisions with a clerical review band (Fellegi & Sunter, 1969):
  merge automatically above ``auto_probability``, leave apart below
  ``review_probability``, and queue everything between for a person to decide.
* A complete-linkage guard: no automatic merge may contain a strongly
  contradicting pair, which stops chains of individually plausible links.

Nothing here rewrites passages or deletes identities. ``resolve`` is a pure
function of stored evidence, human decisions and the calibration; ``apply``
records its links in ``identity_links`` so they can be reviewed and reversed.
The default calibration is a stated prior, not a measurement: see
``fit_calibration``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .domain import utc_now
from .estimation import estimate_attendance
from .evidence import load_tracklets
from .face_registry import FACE_MODEL, MIN_SAMPLES
from .store import EventStore

SOURCE = "global_resolver"


@dataclass(frozen=True)
class Calibration:
    """P(same person | cosine similarity) = sigmoid(slope * similarity + intercept).

    The defaults put even odds at 0.45 and about 0.97 at the online match
    threshold of 0.60. They are a prior for ArcFace R50 embeddings, not a fit to
    this doorway; ``calibrated`` stays False until labelled pairs replace them.
    """

    slope: float = 24.0
    intercept: float = -10.8
    calibrated: bool = False
    labelled_pairs: int = 0
    model: str = FACE_MODEL
    # "prior" (stated default), "constraints" (fitted to pairs whose answer
    # tracking guarantees) or "labelled" (fitted to human-reviewed pairs).
    method: str = "prior"

    def logit(self, similarity):
        return self.slope * similarity + self.intercept

    @classmethod
    def load(cls, path: Path | None) -> Calibration:
        if path is None or not path.exists():
            return cls()
        payload = json.loads(path.read_text())
        if payload.get("model") != FACE_MODEL:
            # Scores from another recognizer are not comparable; fall back to the prior.
            return cls()
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ResolverConfig:
    auto_probability: float = 0.97
    review_probability: float = 0.30
    # An automatic merge may not contain any pair of nodes less similar than this.
    conflict_floor: float = 0.25
    # Face tracks fragment at low frame rates, often to one sample each. A lone
    # sample is admitted as evidence, but a cluster still needs three samples or
    # an enrolled identity before it counts as a person.
    min_samples: int = 1
    min_consistency: float = 0.45
    min_overlap_seconds: float = 1.0
    # Overlapping tracklets this similar are one person tracked twice, not two people.
    duplicate_similarity: float = 0.75
    max_units: int = 6000
    max_review: int = 200

    def __post_init__(self):
        if not 0 < self.review_probability < self.auto_probability < 1:
            raise ValueError("Need 0 < review_probability < auto_probability < 1")


def calibration_path(store: EventStore) -> Path:
    return store.data_root / "resolution-calibration.json"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


@dataclass
class _Unit:
    ref: str
    group: str
    vector: np.ndarray
    samples: int
    quality: float
    camera_id: str | None = None
    kind: str | None = None
    start: float | None = None
    end: float | None = None
    first_at: str | None = None


@dataclass
class Resolution:
    report: dict
    links: list[dict] = field(default_factory=list)
    # Evidence node ref -> canonical ref of the person it was assigned to.
    assignments: dict[str, str] = field(default_factory=dict)


def _block(matrix: np.ndarray, starts: np.ndarray, operation) -> np.ndarray:
    return operation.reduceat(operation.reduceat(matrix, starts, axis=0), starts, axis=1)


def _agglomerate(sum_logit, sum_sim, count, min_sim, forbid, auto_logit, floor, forced_pairs):
    """Merge clusters by mean log-odds, highest first, under the constraints.

    All matrices are symmetric over clusters and are updated in place. Sums and
    pair counts are additive under a merge; the minimum similarity and the
    cannot-link flag propagate by min / or. A heap with per-cluster version
    stamps avoids rescanning the full matrix after each merge.

    Evidence decides first. Human "same" decisions (``forced_pairs``) are then
    applied to the resulting clusters, and evidence is consulted once more. A
    confirmation can therefore only add merges: it never dilutes an average and
    silently undoes a merge the evidence had already supported.

    Returns the final clusters and the ordered merge events that produced them.
    """
    size = len(sum_logit)
    active = np.ones(size, dtype=bool)
    version = np.zeros(size, dtype=np.int64)
    owner = np.arange(size)
    members = [[index] for index in range(size)]
    events: list[dict] = []
    heap: list[tuple[float, int, int, int, int]] = []

    def push(row: int) -> None:
        mean = sum_logit[row] / count[row]
        eligible = active & ~forbid[row] & (mean >= auto_logit) & (min_sim[row] >= floor)
        eligible[row] = False
        for other in np.flatnonzero(eligible):
            first, second = (row, int(other)) if row < other else (int(other), row)
            heapq.heappush(
                heap, (-float(mean[other]), first, second, version[first], version[second])
            )

    def merge(keep: int, drop: int, forced: bool) -> None:
        events.append(
            {
                "left": list(members[keep]),
                "right": list(members[drop]),
                "logit": float(sum_logit[keep, drop] / count[keep, drop]),
                "similarity": float(sum_sim[keep, drop] / count[keep, drop]),
                "lowest": float(min_sim[keep, drop]),
                "forced": forced,
            }
        )
        for matrix in (sum_logit, sum_sim, count):
            matrix[keep] += matrix[drop]
            matrix[:, keep] = matrix[keep]
        min_sim[keep] = np.minimum(min_sim[keep], min_sim[drop])
        min_sim[:, keep] = min_sim[keep]
        forbid[keep] |= forbid[drop]
        forbid[:, keep] = forbid[keep]
        active[drop] = False
        version[keep] += 1
        owner[members[drop]] = keep
        members[keep].extend(members[drop])
        push(keep)

    def drain() -> None:
        while heap:
            _, keep, drop, keep_version, drop_version = heapq.heappop(heap)
            if (
                active[keep]
                and active[drop]
                and version[keep] == keep_version
                and version[drop] == drop_version
            ):
                merge(keep, drop, False)

    for row in range(size):
        push(row)
    drain()
    for left, right in forced_pairs:
        keep, drop = sorted((int(owner[left]), int(owner[right])))
        if keep != drop:
            merge(keep, drop, True)
    drain()
    return [(int(index), sorted(members[index])) for index in np.flatnonzero(active)], events


def resolve(
    store: EventStore,
    event_id: str,
    *,
    config: ResolverConfig | None = None,
    calibration: Calibration | None = None,
) -> Resolution:
    config = config or ResolverConfig()
    calibration = calibration or Calibration.load(calibration_path(store))
    tracklets = load_tracklets(store, event_id, FACE_MODEL)
    with store.connect() as db:
        if not db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone():
            raise LookupError(event_id)
        templates = db.execute(
            "SELECT identity_id,label,embedding,samples,quality,updated_at FROM face_templates "
            "WHERE event_id=? AND model=? ORDER BY label,identity_id",
            (event_id, FACE_MODEL),
        ).fetchall()
        variants = db.execute(
            "SELECT identity_id,embedding FROM confirmed_face_variants "
            "WHERE event_id=? AND model=?",
            (event_id, FACE_MODEL),
        ).fetchall()
        passages = db.execute(
            "SELECT id,camera_id,track_id,direction,identity_id,"
            "json_extract(evidence,'$.cross_view_rejected') AS rejected "
            "FROM passages WHERE event_id=?",
            (event_id,),
        ).fetchall()
        human = db.execute(
            "SELECT left_ref,right_ref,status FROM identity_links WHERE event_id=? "
            "AND status IN ('confirmed_same','confirmed_different') ORDER BY id",
            (event_id,),
        ).fetchall()

    # A group is a set of evidence the online system already treats as one
    # person: an enrolled identity with its tracklets, or one unenrolled tracklet.
    labels = {row["identity_id"]: row["label"] for row in templates}
    units: list[_Unit] = []
    for row in templates:
        ref = f"identity:{row['identity_id']}"
        units.append(
            _Unit(
                ref,
                ref,
                np.frombuffer(row["embedding"], dtype=np.float32),
                int(row["samples"]),
                float(row["quality"]),
                first_at=row["updated_at"],
            )
        )
    for row in variants:
        if row["identity_id"] in labels:
            ref = f"identity:{row['identity_id']}"
            units.append(
                _Unit(ref, ref, np.frombuffer(row["embedding"], dtype=np.float32), 1, 1.0)
            )
    skipped = {"too_few_samples": 0, "inconsistent": 0}
    group_of_ref = {unit.ref: unit.group for unit in units}
    tracklet_ref = {}
    for tracklet in tracklets:
        ref = f"tracklet:{tracklet.id}"
        tracklet_ref[(tracklet.camera_id, tracklet.track_key)] = ref
        if tracklet.online_identity_id in labels:
            group = f"identity:{tracklet.online_identity_id}"
        elif len(tracklet.embeddings) < config.min_samples:
            skipped["too_few_samples"] += 1
            continue
        elif tracklet.consistency < config.min_consistency:
            skipped["inconsistent"] += 1
            continue
        else:
            group = ref
        group_of_ref[ref] = group
        units.append(
            _Unit(
                ref,
                group,
                tracklet.prototype,
                len(tracklet.embeddings),
                float(tracklet.qualities.mean()),
                tracklet.camera_id,
                tracklet.kind,
                _epoch(tracklet.first_observed_at),
                _epoch(tracklet.last_observed_at),
                tracklet.first_observed_at,
            )
        )
    if len(units) > config.max_units:
        raise ValueError(
            f"{len(units)} evidence nodes exceed max_units={config.max_units}; "
            "this dense resolver needs nearest-neighbour sparsification at that scale"
        )

    group_keys = sorted(set(group_of_ref.values()))
    group_index = {key: index for index, key in enumerate(group_keys)}
    same_pairs, different_pairs = [], []
    for row in human:
        left, right = group_of_ref.get(row["left_ref"]), group_of_ref.get(row["right_ref"])
        if left is None or right is None or left == right:
            continue  # the evidence a past decision referred to is gone
        pair = (group_index[left], group_index[right])
        (same_pairs if row["status"] == "confirmed_same" else different_pairs).append(pair)

    counts_online = _passage_counts(passages, lambda row: row["identity_id"])
    report = {
        "event_id": event_id,
        "run_id": "res-" + uuid.uuid4().hex[:12],
        "created_at": utc_now(),
        "model": FACE_MODEL,
        "calibration": asdict(calibration),
        "config": asdict(config),
        "inputs": {
            "identities": len(labels),
            "tracklets": len(tracklets),
            "evidence_nodes": len(units),
            "skipped_tracklets": skipped,
        },
    }
    if not units:
        report.update(
            constraints={},
            clusters={"total": 0, "established": 0},
            suspect_identities=[],
            auto_links=[],
            review=[],
            counts={"online": counts_online, "resolved": counts_online},
            attendance=estimate_attendance(counts_online["entries"], []),
        )
        return Resolution(report)

    unit_group = np.array([group_index[unit.group] for unit in units], dtype=np.int64)
    order = np.argsort(unit_group, kind="stable")
    ordered = [units[index] for index in order]
    groups = unit_group[order]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(ordered)])
    vectors = np.stack([unit.vector for unit in ordered]).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    similarity = (vectors @ vectors.T).astype(np.float64)
    logit = calibration.logit(similarity)

    # Cannot-link: two same-kind tracklets visible together in one camera.
    cannot = np.zeros(similarity.shape, dtype=bool)
    duplicates = 0
    by_view: dict[tuple[str, str], list[int]] = {}
    for index, unit in enumerate(ordered):
        if unit.camera_id is not None:
            by_view.setdefault((unit.camera_id, unit.kind), []).append(index)
    for indices in by_view.values():
        idx = np.asarray(indices)
        begin = np.array([ordered[i].start for i in indices])
        end = np.array([ordered[i].end for i in indices])
        overlap = np.minimum.outer(end, end) - np.maximum.outer(begin, begin)
        together = overlap >= config.min_overlap_seconds
        np.fill_diagonal(together, False)
        same_person_twice = together & (similarity[np.ix_(idx, idx)] >= config.duplicate_similarity)
        duplicates += int(same_person_twice.sum()) // 2
        cannot[np.ix_(idx, idx)] = together & ~same_person_twice

    sum_logit = _block(logit, starts, np.add)
    sum_sim = _block(similarity, starts, np.add)
    min_sim = _block(similarity, starts, np.minimum)
    forbid = _block(cannot.astype(np.uint8), starts, np.maximum).astype(bool)
    count = np.outer(sizes, sizes).astype(np.float64)
    # A cannot-link inside one identity means it absorbed two people who were
    # seen together. The identity stands; a person should look at it.
    suspects = sorted(
        labels[group_keys[index].split(":", 1)[1]]
        for index in np.flatnonzero(np.diag(forbid))
        if group_keys[index].startswith("identity:")
    )
    np.fill_diagonal(forbid, False)
    for left, right in different_pairs:
        forbid[left, right] = forbid[right, left] = True

    auto_logit, review_logit = _logit(config.auto_probability), _logit(config.review_probability)
    clusters, events = _agglomerate(
        sum_logit, sum_sim, count, min_sim, forbid, auto_logit, config.conflict_floor, same_pairs
    )

    group_samples = np.add.reduceat(np.array([unit.samples for unit in ordered]), starts)
    group_quality = np.add.reduceat(
        np.array([unit.quality * unit.samples for unit in ordered]), starts
    ) / np.maximum(group_samples, 1)
    group_first = [
        min((unit.first_at for unit in ordered[start : start + size] if unit.first_at), default="")
        for start, size in zip(starts, sizes, strict=True)
    ]
    group_cameras = [
        {unit.camera_id for unit in ordered[start : start + size] if unit.camera_id}
        for start, size in zip(starts, sizes, strict=True)
    ]

    def canonical(member_groups: list[int]) -> int:
        """Prefer the lowest-labelled enrolled identity, else the earliest tracklet."""
        identities = [g for g in member_groups if group_keys[g].startswith("identity:")]
        if identities:
            return min(
                identities,
                key=lambda g: (labels[group_keys[g].split(":", 1)[1]], group_keys[g]),
            )
        return min(member_groups, key=lambda g: (group_first[g], group_keys[g]))

    def describe(group: int) -> str:
        kind, value = group_keys[group].split(":", 1)
        return labels[value] if kind == "identity" else "Unenrolled face"

    def link_row(left_groups, right_groups, mean_logit, mean_similarity, lowest, status, reason):
        left, right = canonical(left_groups), canonical(right_groups)
        probability = _sigmoid(mean_logit)
        key = "|".join([event_id, *sorted([group_keys[left], group_keys[right]])])
        samples = int(sum(group_samples[g] for g in right_groups))
        quality = sum(group_quality[g] * group_samples[g] for g in right_groups) / max(samples, 1)
        return {
            "id": "res-" + hashlib.sha1(key.encode()).hexdigest()[:20],
            "left_ref": group_keys[left],
            "right_ref": group_keys[right],
            "left_label": describe(left),
            "right_label": describe(right),
            "left_lane": ",".join(sorted(set().union(*(group_cameras[g] for g in left_groups))))
            or "gallery",
            "right_lane": ",".join(sorted(set().union(*(group_cameras[g] for g in right_groups))))
            or "gallery",
            "left_observed_at": min(filter(None, (group_first[g] for g in left_groups)),
                                    default=report["created_at"]),
            "right_observed_at": min(filter(None, (group_first[g] for g in right_groups)),
                                     default=report["created_at"]),
            "face_similarity": round(float(mean_similarity), 4),
            "lowest_similarity": round(float(lowest), 4),
            "face_quality": round(float(min(max(quality, 0), 1)), 4),
            "supporting_frames": samples,
            "probability": round(probability, 6),
            "status": status,
            "recommended_decision": "merge" if probability >= 0.5 else "split",
            "reason": reason,
        }

    # One link per evidence-driven merge, carrying the evidence as it stood when
    # the two partial clusters were joined.
    auto_links = [
        link_row(
            event["left"],
            event["right"],
            event["logit"],
            event["similarity"],
            event["lowest"],
            "auto_merged",
            "Mean face evidence between these two groups of observations is above the "
            "automatic threshold, with no contradicting pair.",
        )
        for event in events
        if not event["forced"]
    ]

    cluster_of_group = {}
    cluster_rows = []
    for position, (matrix_index, member_groups) in enumerate(clusters):
        identities = sum(group_keys[g].startswith("identity:") for g in member_groups)
        samples = int(sum(group_samples[g] for g in member_groups))
        cluster_rows.append(
            {
                "index": matrix_index,
                "groups": member_groups,
                "identities": identities,
                "tracklets": len(member_groups) - identities,
                "samples": samples,
                # Same bar as online enrollment: an identity, or three agreeing samples.
                "established": bool(identities) or samples >= MIN_SAMPLES,
            }
        )
        for group in member_groups:
            cluster_of_group[group] = position

    review = []
    active = np.array([row["index"] for row in cluster_rows])
    grid = np.ix_(active, active)
    mean_logits = sum_logit[grid] / count[grid]
    candidates = np.triu(~forbid[grid] & (mean_logits >= review_logit), 1)
    for a, b in np.argwhere(candidates):
        a, b = int(a), int(b)
        i, j = active[a], active[b]
        review.append(
            (
                a,
                b,
                link_row(
                    cluster_rows[a]["groups"],
                    cluster_rows[b]["groups"],
                    mean_logits[a, b],
                    sum_sim[i, j] / count[i, j],
                    min_sim[i, j],
                    "review",
                    "Average evidence says same person, but at least one pair of "
                    "observations disagrees strongly."
                    if mean_logits[a, b] >= auto_logit
                    else "Face evidence is between the automatic-merge and "
                    "keep-separate thresholds.",
                ),
            )
        )
    review.sort(key=lambda item: (-item[2]["probability"], item[2]["id"]))
    review = review[: config.max_review]

    def passage_cluster(row):
        if row["identity_id"] in labels:
            group = f"identity:{row['identity_id']}"
        else:
            group = group_of_ref.get(tracklet_ref.get((row["camera_id"], row["track_id"])))
        if group is None and row["rejected"]:
            # The online cross-view matcher accepted this body's appearance and
            # only lacked a face identity. Use that facing track's stored faces.
            rejected = json.loads(row["rejected"])
            if rejected.get("last_reject_reason") == "face_identity_pending":
                key = (rejected.get("facing_camera_id"), rejected.get("best_candidate"))
                group = group_of_ref.get(tracklet_ref.get(key))
        if group is None:
            return None
        position = cluster_of_group[group_index[group]]
        return position if cluster_rows[position]["established"] else None

    counts_resolved = _passage_counts(passages, passage_cluster)
    entries_per_person: dict[int, int] = {}
    for row in passages:
        position = passage_cluster(row)
        if row["direction"] == "in" and position is not None:
            entries_per_person[position] = entries_per_person.get(position, 0) + 1
    attendance = estimate_attendance(counts_resolved["entries"], list(entries_per_person.values()))

    # If every likely review merge were confirmed, how far would the count fall?
    merge_parent = {position: position for position in entries_per_person}

    def merge_find(position):
        while merge_parent[position] != position:
            position = merge_parent[position]
        return position

    for a, b, link in review:
        if link["probability"] >= 0.5 and a in merge_parent and b in merge_parent:
            merge_parent[merge_find(a)] = merge_find(b)
    components = len({merge_find(position) for position in merge_parent})
    attendance["pending_review_could_remove"] = len(merge_parent) - components

    report.update(
        constraints={
            "cannot_link_pairs": int(np.triu(cannot, 1).sum()),
            "duplicate_tracks_suspected": duplicates,
            "human_confirmed_same": len(same_pairs),
            "human_confirmed_different": len(different_pairs),
        },
        clusters={
            "total": len(cluster_rows),
            "established": sum(row["established"] for row in cluster_rows),
            "identities_merged": sum(max(row["identities"] - 1, 0) for row in cluster_rows),
            "tracklets_attached": sum(
                row["tracklets"] for row in cluster_rows if row["identities"]
            ),
            "people_from_unenrolled_faces": sum(
                row["established"] and not row["identities"] for row in cluster_rows
            ),
        },
        suspect_identities=suspects,
        auto_links=auto_links,
        review=[link for _, _, link in review],
        counts={"online": counts_online, "resolved": counts_resolved},
        attendance=attendance,
    )
    assignments = {
        ref: group_keys[canonical(cluster_rows[cluster_of_group[group_index[group]]]["groups"])]
        for ref, group in group_of_ref.items()
    }
    return Resolution(report, [*auto_links, *(link for _, _, link in review)], assignments)


def _passage_counts(passages, person_of) -> dict:
    entries = [row for row in passages if row["direction"] == "in"]
    people = [person_of(row) for row in entries]
    linked = [person for person in people if person is not None]
    return {
        "entries": len(entries),
        "exits": len(passages) - len(entries),
        "linked_entries": len(linked),
        "unique_entrants": len(set(linked)),
        "repeat_entries": len(linked) - len(set(linked)),
        "unresolved_entries": len(entries) - len(linked),
    }


def apply(store: EventStore, resolution: Resolution, actor: str) -> dict:
    """Record this run's links. Human decisions are never overwritten.

    Earlier undecided resolver links are replaced, so the review queue always
    reflects the latest evidence. Link ids are a hash of the two node refs, so a
    pair keeps its id, and any human decision on it, across runs.
    """
    if not actor:
        raise ValueError("An actor is required")
    report = resolution.report
    event_id = report["event_id"]
    written = 0
    with store.transaction() as db:
        db.execute(
            "DELETE FROM identity_links WHERE event_id=? AND status IN ('auto_merged','review') "
            "AND json_extract(evidence,'$.source')=?",
            (event_id, SOURCE),
        )
        for link in resolution.links:
            evidence = {
                "source": SOURCE,
                "run_id": report["run_id"],
                "left_label": link["left_label"],
                "right_label": link["right_label"],
                "lowest_similarity": link["lowest_similarity"],
                "calibrated": report["calibration"]["calibrated"],
            }
            written += db.execute(
                """INSERT OR IGNORE INTO identity_links
                (id,event_id,left_ref,right_ref,left_lane,right_lane,left_observed_at,
                 right_observed_at,face_similarity,body_similarity,face_quality,body_quality,
                 supporting_frames,probability,status,recommended_decision,reason,evidence,
                 created_at)
                VALUES (?,?,?,?,?,?,?,?,?,NULL,?,0,?,?,?,?,?,?,?)""",
                (
                    link["id"],
                    event_id,
                    link["left_ref"],
                    link["right_ref"],
                    link["left_lane"],
                    link["right_lane"],
                    link["left_observed_at"],
                    link["right_observed_at"],
                    link["face_similarity"],
                    link["face_quality"],
                    link["supporting_frames"],
                    link["probability"],
                    link["status"],
                    link["recommended_decision"],
                    link["reason"],
                    json.dumps(evidence),
                    report["created_at"],
                ),
            ).rowcount
        db.execute(
            "INSERT INTO audit_log "
            "(event_id,action,actor,target_type,target_id,before_state,after_state,created_at) "
            "VALUES (?,'identity_resolution',?,'event',?,?,?,?)",
            (
                event_id,
                actor,
                event_id,
                json.dumps(report["counts"]["online"]),
                json.dumps(
                    {
                        "run_id": report["run_id"],
                        "counts": report["counts"]["resolved"],
                        "attendance": report["attendance"],
                        "links_written": written,
                    }
                ),
                utc_now(),
            ),
        )
    return {"run_id": report["run_id"], "links_written": written}


def _fit_logistic(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Weighted two-parameter logistic regression by damped Newton steps."""
    design = np.column_stack([x, np.ones_like(x)])
    ridge = np.diag([1e-3, 1e-6])  # keeps separable data finite

    def loss(beta):
        z = np.clip(design @ beta, -30, 30)
        return float(np.sum(weight * (np.logaddexp(0, z) - y * z)) + 0.5 * beta @ ridge @ beta)

    beta = np.array([Calibration.slope, Calibration.intercept])
    current = loss(beta)
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        gradient = design.T @ (weight * (p - y)) + ridge @ beta
        hessian = (design * (weight * p * (1 - p))[:, None]).T @ design + ridge
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-6 and loss(beta - scale * step) > current:
            scale /= 2  # a full Newton step can overshoot far from the optimum
        beta = beta - scale * step
        updated = loss(beta)
        if current - updated < 1e-10:
            break
        current = updated
    if beta[0] <= 0:
        raise ValueError("Scores do not show similarity increasing with same-person odds")
    return float(beta[0]), float(beta[1])


def fit_calibration(
    similarities: list[float],
    same_person: list[bool],
    *,
    false_merge_weight: float = 5.0,
    method: str = "labelled",
) -> Calibration:
    """Fit P(same | similarity) by weighted logistic regression (Platt scaling).

    Different-person pairs carry extra weight because a false merge removes a
    real attendee from the count, while a false split only adds review work.
    Pairs reviewed from the queue cluster around the decision boundary, so this
    fit is most trustworthy there and should be re-checked on held-out pairs.
    """
    x = np.asarray(similarities, dtype=np.float64)
    y = np.asarray(same_person, dtype=np.float64)
    if len(x) != len(y) or len(x) < 20 or y.sum() < 5 or (1 - y).sum() < 5:
        raise ValueError("Need at least 20 labelled pairs with 5 or more of each outcome")
    slope, intercept = _fit_logistic(x, y, np.where(y == 1, 1.0, false_merge_weight))
    return Calibration(slope, intercept, method == "labelled", len(x), method=method)


def constraint_pairs(
    store: EventStore, event_id: str, impostor_scores: list[float]
) -> tuple[list[float], list[bool]]:
    """Calibration pairs whose answer is guaranteed by tracking, not by a person.

    Different people: ``impostor_scores`` from ``tools/score_survey.py``, the
    similarities of faces detected in the same frame. Same person: every pair of
    samples inside one stored tracklet.

    Two cautions. Same-tracklet pairs are close in time and pose, so they are
    easier than the cross-visit matches the resolver decides. And an impostor
    sample of a few hundred pairs says nothing about rarer look-alikes: on the
    2025 fair footage one same-frame pair in 564 scored 0.395, far outside a
    Gaussian tail, which is why a mixture model fitted to unlabelled scores was
    tried and rejected here.
    """
    genuine: list[float] = []
    for tracklet in load_tracklets(store, event_id, FACE_MODEL):
        count = len(tracklet.embeddings)
        if count >= 2:
            matrix = tracklet.embeddings @ tracklet.embeddings.T
            genuine.extend(matrix[np.triu_indices(count, 1)].tolist())
    scores = [float(score) for score in impostor_scores]
    return scores + genuine, [False] * len(scores) + [True] * len(genuine)


def labelled_pairs(store: EventStore, event_id: str) -> tuple[list[float], list[bool]]:
    """Reviewed resolver links for one event, as calibration data."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT face_similarity,status FROM identity_links WHERE event_id=? "
            "AND status IN ('confirmed_same','confirmed_different') "
            "AND face_similarity IS NOT NULL AND json_extract(evidence,'$.source')=?",
            (event_id, SOURCE),
        ).fetchall()
    return (
        [float(row["face_similarity"]) for row in rows],
        [row["status"] == "confirmed_same" for row in rows],
    )

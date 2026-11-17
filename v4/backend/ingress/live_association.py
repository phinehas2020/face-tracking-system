"""Conservative online links between facing identities and side passage events.

All times are capture times on the same host's monotonic clock. ``event_id`` is
the passage row's ID, not the station/event configuration ID. This module emits
proposals only; the caller persists them while holding the shared session lock.
Time gates restrict candidates but never constitute identity evidence.

Two kinds of evidence can link a crossing: learned body appearance, and, when
the crossing zone is enabled, the facing track's position at the door (see
``crossing_zone``). ``zone_mode`` is ``off``, ``shadow`` (decide, record and
learn, but assign nothing) or ``on``.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .crossing_zone import GATE, CrossingZone


@dataclass(frozen=True)
class AssociationProposal:
    event_id: str
    side_track_id: str
    facing_track_id: str
    identity_id: str
    appearance_score: float
    margin: float
    provenance: tuple[str, ...]
    facing_tracks: tuple[str, ...] = ()
    basis: str = "cross_view_appearance"


@dataclass
class _Track:
    last_seen: float
    first_seen: float
    samples: deque = field(default_factory=lambda: deque(maxlen=16))
    identity_id: str | None = None
    identity_source: str | None = None
    identity_conflict: bool = False
    # (time, centre x, centre y, log height) in normalised image coordinates.
    boxes: deque = field(default_factory=lambda: deque(maxlen=96))
    background_at: float = float("-inf")


@dataclass
class _Crossing:
    side_track_id: str
    captured_at: float
    deadline: float
    direction: str = "any"
    # Set when the side track already had a face identity. Such a crossing is
    # never linked here; it only checks the position evidence against a label.
    known_identity: str | None = None
    zone: dict | None = None
    identities: dict = field(default_factory=dict)
    conflict: bool = False


class LiveAssociationCoordinator:
    """Bounded evidence store. Call under the station's shared lock.

    Two distinct learned appearance observations are required in each view.
    Matching is reciprocal with a margin over *all* competing tracks, including
    tracks without a face identity. This prevents a convenient known identity
    winning merely because an equally plausible stranger has no face result.
    Thresholds are conservative starting points, not calibrated probabilities.
    """

    def __init__(
        self,
        *,
        similarity_threshold=0.85,
        margin_threshold=0.08,
        max_transition_seconds=8.0,
        retention_seconds=30.0,
        min_samples=2,
        max_tracks=256,
        max_pending=512,
        max_track_gap_seconds=None,
        sole_candidate_threshold=None,
        zone_mode="off",
        zone_min_samples=20,
        zone_margin=6.0,
    ):
        if zone_mode not in {"off", "shadow", "on"} or zone_margin <= 0:
            raise ValueError("Invalid crossing zone settings")
        self.zone_mode = zone_mode
        # Squared-distance margin; 6 is a likelihood ratio of about 20 to 1.
        self.zone_margin = zone_margin
        self.zone = CrossingZone(min_samples=zone_min_samples)
        self.zone_stats = {"decisions": {}, "label_agree": 0, "label_disagree": 0}
        self._zone_claims: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._chains_cache = None
        if not (0 < similarity_threshold <= 1 and 0 < margin_threshold <= 1):
            raise ValueError("Invalid appearance thresholds")
        if not (0 < max_transition_seconds <= retention_seconds):
            raise ValueError("Invalid transition/retention window")
        if min_samples < 2 or min_samples > 16 or max_tracks < 2 or max_pending < 1:
            raise ValueError("Invalid evidence bounds")
        if max_track_gap_seconds is not None and not (
            0 <= max_track_gap_seconds <= max_transition_seconds
        ):
            raise ValueError("Invalid overlapping-view timing window")
        if sole_candidate_threshold is not None and not (
            0 < sole_candidate_threshold <= similarity_threshold
        ):
            raise ValueError("Invalid sole-candidate threshold")
        # When exactly one person is in each view at the same moment, being the
        # only candidate is itself strong evidence, and appearance only has to
        # not contradict it. Needs the overlapping-view timing window.
        self.sole_candidate_threshold = sole_candidate_threshold
        self.max_track_gap_seconds = max_track_gap_seconds
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.max_transition_seconds = max_transition_seconds
        self.retention_seconds = retention_seconds
        self.min_samples = min_samples
        self.max_tracks = max_tracks
        self.max_pending = max_pending
        self.tracks: dict[str, OrderedDict[str, _Track]] = {
            "side": OrderedDict(),
            "facing": OrderedDict(),
        }
        self.pending: OrderedDict[str, _Crossing] = OrderedDict()
        self._emitted: OrderedDict[str, None] = OrderedDict()
        self._side_to_facing: dict[str, str] = {}
        self._facing_to_side: dict[str, str] = {}
        self._latest_time = 0.0
        self.linked_count = 0
        self.expired_count = 0
        self.last_decisions: OrderedDict[str, dict] = OrderedDict()
        # Final outcomes awaiting persistence by the session (bounded).
        self.terminal: deque = deque(maxlen=512)

    def drain_terminal(self) -> list[dict]:
        decisions = list(self.terminal)
        self.terminal.clear()
        return decisions

    def _note(self, event_id, crossing, now, reason, **evidence):
        if crossing.zone is not None:
            evidence = {**evidence, "zone": crossing.zone}
        self.last_decisions[event_id] = {
            "event_id": event_id, "side_track_id": crossing.side_track_id,
            "crossing_at": crossing.captured_at, "decision_at": now,
            "reason": reason, **evidence,
        }
        self.last_decisions.move_to_end(event_id)
        while len(self.last_decisions) > 64:
            self.last_decisions.popitem(last=False)

    def observe(
        self,
        view: str,
        track_id: str,
        captured_at: float,
        *,
        embedding=None,
        embedding_source: str | None = None,
        identity_id: str | None = None,
        identity_source: str | None = None,
        ground_position=None,
        box=None,
    ) -> None:
        if view not in self.tracks:
            raise ValueError("view must be side or facing")
        if not isfinite(captured_at):
            return
        self._prune(max(self._latest_time, captured_at))
        if captured_at < self._latest_time - self.retention_seconds:
            return
        tracks = self.tracks[view]
        track = tracks.setdefault(track_id, _Track(captured_at, captured_at))
        # Duplicate/out-of-order frames cannot manufacture multi-frame evidence.
        if captured_at < track.last_seen:
            return
        track.last_seen = captured_at
        tracks.move_to_end(track_id)
        while len(tracks) > self.max_tracks:
            self._drop_track(view, next(iter(tracks)))

        if box is not None and self.zone_mode != "off":
            self._observe_box(view, track, captured_at, box)
        self.attach_identity(view, track_id, identity_id, identity_source, captured_at)
        # Only the explicitly integrated learned encoder is admitted. Histograms,
        # time-only candidates and unknown feature types fail closed.
        if embedding_source != "osnet" or embedding is None:
            return
        try:
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all() or norm < 1e-8:
                return
        except (TypeError, ValueError):
            return
        if track.samples and track.samples[-1][0] == captured_at:
            return
        track.samples.append((captured_at, vector / norm))
        # ground_position intentionally unused until real camera calibration and
        # uncertainty are available; accepting this parameter does not imply any.

    def attach_identity(
        self,
        view: str,
        track_id: str,
        identity_id: str | None,
        identity_source: str | None,
        evidence_at: float,
    ) -> bool:
        """Apply delayed face evidence to an already observed body track.

        The caller must have associated the face to this body on the original
        frame. ``evidence_at`` is that frame's capture time, never result-arrival
        time. No track, appearance sample, last-seen time, or new lifetime is
        created here. The current coordinator watermark bounds stale results.
        """
        if view not in self.tracks:
            raise ValueError("view must be side or facing")
        if (
            not identity_id
            or identity_source not in {"face_match", "face_track_continuity"}
            or not isfinite(evidence_at)
        ):
            return False
        self._prune(self._latest_time)
        track = self.tracks[view].get(track_id)
        if (
            track is None
            or evidence_at < self._latest_time - self.retention_seconds
            or not track.first_seen <= evidence_at <= track.last_seen
        ):
            return False
        if track.identity_id and track.identity_id != identity_id:
            track.identity_conflict = True
        if track.identity_conflict:
            return False
        track.identity_id, track.identity_source = identity_id, identity_source
        return True

    def register_crossing(
        self,
        event_id: str,
        side_track_id: str,
        captured_at: float,
        *,
        direction: str = "any",
        known_identity: str | None = None,
    ) -> None:
        if not isfinite(captured_at) or event_id in self._emitted or event_id in self.pending:
            return
        if known_identity is not None and self.zone_mode == "off":
            return
        self._prune(max(self._latest_time, captured_at))
        if captured_at < self._latest_time - self.max_transition_seconds:
            self.expired_count += known_identity is None
            return
        self.pending[event_id] = _Crossing(
            side_track_id,
            captured_at,
            captured_at + self.max_transition_seconds,
            direction,
            known_identity,
        )
        while len(self.pending) > self.max_pending:
            _, dropped = self.pending.popitem(last=False)
            self.expired_count += dropped.known_identity is None

    def _time_compatible(self, left, right):
        if self.max_track_gap_seconds is None:
            return True
        gap = max(left.first_seen - right.last_seen, right.first_seen - left.last_seen, 0)
        return gap <= self.max_track_gap_seconds

    def _features(self, track: _Track, event_time: float):
        samples = [
            v for t, v in track.samples if abs(t - event_time) <= self.max_transition_seconds
        ]
        if len(samples) < self.min_samples or len({v.shape for v in samples}) != 1:
            return None
        values = np.stack(samples)
        mean = values.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            return None
        return values, mean / norm

    @staticmethod
    def _score(left, right):
        if left is None or right is None or left[1].shape != right[1].shape:
            return None
        # A high mean-vector score alone can hide contradictory frames. Require
        # pairwise multi-frame agreement as well, taking the conservative score.
        return min(float(left[1] @ right[1]), float(np.mean(left[0] @ right[0].T)))

    def resolve_crossing(self, event_id: str, now: float) -> AssociationProposal | None:
        if not isfinite(now):
            return None
        self._prune(max(self._latest_time, now))
        crossing = self.pending.get(event_id)
        if crossing is None:
            return None
        zone_linked = crossing.zone is not None and crossing.zone["decision"] == "linked"
        if crossing.known_identity is None and not (zone_linked and self.zone_mode == "on"):
            proposal = self._resolve_appearance(event_id, crossing, now)
            if proposal is not None:
                return proposal
        if self.zone_mode != "off":
            return self._resolve_zone(event_id, crossing, now)
        return None

    def _resolve_appearance(self, event_id, crossing, now) -> AssociationProposal | None:
        side = self.tracks["side"].get(crossing.side_track_id)
        if side is None:
            self._note(event_id, crossing, now, "missing_side_track")
            return None
        side_features = self._features(side, crossing.captured_at)
        evidence = {"side_samples": 0 if side_features is None else len(side_features[0]),
                    "side_first_seen": side.first_seen, "side_last_seen": side.last_seen,
                    "temporal_rejections": 0, "appearance_unavailable": 0}
        if side_features is None:
            self._note(event_id, crossing, now, "insufficient_side_appearance", **evidence)
            return None
        candidates = []
        for front_id, front in self.tracks["facing"].items():
            if not self._time_compatible(side, front):
                evidence["temporal_rejections"] += 1
                continue
            score = self._score(side_features, self._features(front, crossing.captured_at))
            if score is not None:
                candidates.append((score, front_id, front))
            else:
                evidence["appearance_unavailable"] += 1
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates:
            self._note(event_id, crossing, now, "no_eligible_facing_appearance", **evidence)
            return None
        score, front_id, front = candidates[0]
        margin = score - (candidates[1][0] if len(candidates) > 1 else -1.0)
        evidence.update(best_candidate=front_id, appearance_score=score,
                        second_score=candidates[1][0] if len(candidates) > 1 else None,
                        facing_margin=margin, reciprocal_margin=None,
                        face_ready=bool(front.identity_id and not front.identity_conflict),
                        facing_first_seen=front.first_seen, facing_last_seen=front.last_seen,
                        facing_samples=len(self._features(front, crossing.captured_at)[0]))
        side_rivals = sum(
            other_id != crossing.side_track_id and self._time_compatible(other, front)
            for other_id, other in self.tracks["side"].items()
        )
        sole = (
            self.sole_candidate_threshold is not None
            and self.max_track_gap_seconds is not None
            and len(candidates) == 1
            and evidence["appearance_unavailable"] == 0
            and side_rivals == 0
        )
        evidence["sole_candidate"] = sole
        if score < (self.sole_candidate_threshold if sole else self.similarity_threshold):
            self._note(event_id, crossing, now, "appearance_below_threshold", **evidence)
            return None
        if margin < self.margin_threshold:
            self._note(event_id, crossing, now, "ambiguous_facing_match", **evidence)
            return None
        if self._side_to_facing.get(crossing.side_track_id, front_id) != front_id:
            self._note(event_id, crossing, now, "side_already_assigned", **evidence)
            return None
        if self._facing_to_side.get(front_id, crossing.side_track_id) != crossing.side_track_id:
            self._note(event_id, crossing, now, "facing_already_assigned", **evidence)
            return None
        front_features = self._features(front, crossing.captured_at)
        for other_id, other in self.tracks["side"].items():
            if other_id == crossing.side_track_id or not self._time_compatible(other, front):
                continue
            competing_score = self._score(
                self._features(other, crossing.captured_at), front_features
            )
            if competing_score is not None:
                margin = min(margin, score - competing_score)
                if evidence["reciprocal_margin"] is None or score - competing_score < evidence["reciprocal_margin"]:
                    evidence["reciprocal_margin"] = score - competing_score
                    evidence["reciprocal_competitor"] = other_id
                    evidence["reciprocal_competitor_score"] = competing_score
        if margin < self.margin_threshold:
            self._note(event_id, crossing, now, "ambiguous_reciprocal_match", **evidence)
            return None
        # Identity is checked last, so "face_identity_pending" means every
        # appearance test passed and only a face identity is missing.
        if not front.identity_id or front.identity_conflict:
            reason = (
                "face_identity_conflict" if front.identity_conflict else "face_identity_pending"
            )
            self._note(event_id, crossing, now, reason, **evidence)
            return None
        self._note(event_id, crossing, now, "linked", **evidence)
        self.terminal.append(dict(self.last_decisions[event_id]))
        self._side_to_facing[crossing.side_track_id] = front_id
        self._facing_to_side[front_id] = crossing.side_track_id
        self.pending.pop(event_id)
        self._emitted[event_id] = None
        while len(self._emitted) > self.max_pending * 2:
            self._emitted.popitem(last=False)
        self.linked_count += 1
        return AssociationProposal(
            event_id,
            crossing.side_track_id,
            front_id,
            front.identity_id,
            score,
            margin,
            (
                "osnet_multiframe",
                "reciprocal_appearance_margin",
                "bounded_capture_time",
                *(("sole_candidate_in_both_views",) if sole else ()),
                front.identity_source,
            ),
        )

    # --- Position evidence: the crossing zone -------------------------------

    def _observe_box(self, view, track, captured_at, box) -> None:
        try:
            left, top, right, bottom = (float(value) for value in box)
        except (TypeError, ValueError):
            return
        width, height = right - left, bottom - top
        if not all(isfinite(v) for v in (left, top, right, bottom)) or width <= 0 or height <= 0:
            return
        if track.boxes and track.boxes[-1][0] >= captured_at:
            return
        entry = (captured_at, (left + right) / 2, (top + bottom) / 2, float(np.log(height)))
        track.boxes.append(entry)
        if view == "facing":
            self._chains_cache = None
            if captured_at - track.background_at >= 0.5:
                track.background_at = captured_at
                self.zone.add_background(entry[1:])

    @staticmethod
    def _state(track, at, gap):
        """Box nearest the crossing instant, plus how far in time it is from it."""
        boxes = track.boxes
        if not boxes:
            return None
        if at <= boxes[0][0]:
            offset, values = boxes[0][0] - at, boxes[0][1:]
        elif at >= boxes[-1][0]:
            offset, values = boxes[-1][0] - at, boxes[-1][1:]
        else:
            offset = 0.0
            after = next(i for i, entry in enumerate(boxes) if entry[0] >= at)
            a, b = boxes[after - 1], boxes[after]
            weight = (at - a[0]) / (b[0] - a[0])
            values = tuple(x + weight * (y - x) for x, y in zip(a[1:], b[1:], strict=True))
        if abs(offset) > gap:
            return None
        return np.array([*values, offset])

    @staticmethod
    def _moving(track, at) -> bool:
        """Someone crossing a line is in motion; someone standing in view is not."""
        near = [entry[1:] for entry in track.boxes if abs(entry[0] - at) <= 1.5]
        if len(near) < 2:
            return False
        spread = np.ptp(np.asarray(near), axis=0)
        return bool(spread[0] >= 0.03 or spread[1] >= 0.03 or spread[2] >= 0.08)

    @staticmethod
    def _continues(earlier, later, max_gap=1.0) -> bool:
        """``later`` picks up where ``earlier`` stopped: one person, tracked twice."""
        end, start = earlier.boxes[-1], later.boxes[0]
        if start[0] != later.first_seen or later.first_seen <= earlier.first_seen:
            return False
        if not -0.2 <= start[0] - end[0] <= max_gap:
            return False
        reach = 0.35 * float(np.exp(max(end[3], start[3]))) + 0.03
        return (
            abs(start[1] - end[1]) <= reach
            and abs(start[2] - end[2]) <= reach
            and abs(start[3] - end[3]) <= 0.5
        )

    def _chains(self) -> dict[str, list[str]]:
        """Facing tracks stitched where a successor is spatially continuous and unique."""
        if self._chains_cache is not None:
            return self._chains_cache
        items = [(key, track) for key, track in self.tracks["facing"].items() if track.boxes]
        options = {
            a_key: [b_key for b_key, b in items if b_key != a_key and self._continues(a, b)]
            for a_key, a in items
        }
        claimed: dict[str, list[str]] = {}
        for a_key, successors in options.items():
            for b_key in successors:
                claimed.setdefault(b_key, []).append(a_key)
        successor = {
            a_key: successors[0]
            for a_key, successors in options.items()
            if len(successors) == 1 and claimed[successors[0]] == [a_key]
        }
        chains = {}
        for key, _ in items:
            if key in successor.values():
                continue
            chain = [key]
            while chain[-1] in successor and successor[chain[-1]] not in chain:
                chain.append(successor[chain[-1]])
            for member in chain:
                chains[member] = chain
        self._chains_cache = chains
        return chains

    def _facing_people(self, at, gap) -> list[dict]:
        chains, people = self._chains(), {}
        for key, track in self.tracks["facing"].items():
            state = self._state(track, at, gap)
            if state is None:
                continue
            chain = chains[key]
            person = people.setdefault(
                chain[0], {"tracks": list(chain), "state": state, "moving": False}
            )
            if abs(state[3]) < abs(person["state"][3]):
                person["state"] = state
            person["moving"] = person["moving"] or self._moving(track, at)
        return list(people.values())

    def _zone_decide(self, crossing, gap) -> dict:
        at, direction = crossing.captured_at, crossing.direction
        side_sole = not any(
            key != crossing.side_track_id and track.first_seen - gap <= at <= track.last_seen + gap
            for key, track in self.tracks["side"].items()
        )
        people = self._facing_people(at, gap)
        movers = [person for person in people if person["moving"]]
        record = {
            "mode": self.zone_mode, "direction": direction, "side_sole": side_sole,
            "candidates": len(people), "moving_candidates": len(movers), "sample": False,
        }
        if not self.zone.ready(direction):
            record["decision"] = "zone_learning"
        elif not movers:
            record["decision"] = "no_moving_facing_track"
        else:
            scored = sorted(
                ((self.zone.distance(direction, person["state"]), index)
                 for index, person in enumerate(movers))
            )
            distance, best = scored[0][0], movers[scored[0][1]]
            second = scored[1][0] if len(scored) > 1 else None
            record.update(
                distance=round(distance, 3),
                second_distance=None if second is None else round(second, 3),
                facing_tracks=list(best["tracks"]),
            )
            rival = any(
                key in best["tracks"]
                and owner != crossing.side_track_id
                and abs(claimed_at - at) <= 3.0
                for key, (owner, claimed_at) in self._zone_claims.items()
            )
            if distance > GATE:
                record["decision"] = "no_facing_track_in_zone"
            elif second is not None and second - distance < self.zone_margin:
                record["decision"] = "ambiguous_zone_match"
            elif rival:
                record["decision"] = "facing_already_assigned"
            else:
                record["decision"] = "linked"
                for key in best["tracks"]:
                    self._zone_claims[key] = (crossing.side_track_id, at)
                while len(self._zone_claims) > self.max_pending:
                    self._zone_claims.popitem(last=False)
        # Learn only from the unambiguous case: one person in each view, moving.
        if side_sole and len(people) == 1 and movers:
            state = movers[0]["state"]
            record["sample"] = self.zone.add(direction, state)
            record["state"] = [round(float(value), 4) for value in state]
        decisions = self.zone_stats["decisions"]
        decisions[record["decision"]] = decisions.get(record["decision"], 0) + 1
        return record

    def _resolve_zone(self, event_id, crossing, now) -> AssociationProposal | None:
        gap = 1.0 if self.max_track_gap_seconds is None else self.max_track_gap_seconds
        previous = {
            key: value for key, value in self.last_decisions.get(event_id, {}).items()
            if key not in {"event_id", "side_track_id", "crossing_at", "decision_at", "zone"}
        }
        reason = previous.pop("reason", "awaiting_zone_window")
        if crossing.zone is None:
            # Wait until every facing track that starts just after the crossing is in.
            if now - crossing.captured_at < gap + 0.25:
                return None
            crossing.zone = self._zone_decide(crossing, gap)
            if crossing.zone["decision"] == "linked":
                # A face may appear anywhere along the facing track, not only here.
                crossing.deadline = crossing.captured_at + self.retention_seconds
            self._note(event_id, crossing, now, reason, **previous)
        record = crossing.zone
        if record["decision"] != "linked" or "identity" in record or crossing.conflict:
            return None
        chains, facing = self._chains(), self.tracks["facing"]
        for key in list(record["facing_tracks"]):
            for member in chains.get(key, ()):
                if member not in record["facing_tracks"]:
                    record["facing_tracks"].append(member)
        for key in record["facing_tracks"]:
            track = facing.get(key)
            if track is None:
                continue
            crossing.conflict = crossing.conflict or track.identity_conflict
            if track.identity_id:
                crossing.identities[track.identity_id] = track.identity_source
        applied = self.zone_mode == "on" and crossing.known_identity is None
        if crossing.conflict or len(crossing.identities) > 1:
            crossing.conflict = True
            record["identity_conflict"] = True
            self._note(
                event_id, crossing, now,
                "face_identity_conflict" if applied else reason, **previous,
            )
            return None
        if not crossing.identities:
            if applied and reason != "face_identity_pending":
                self._note(event_id, crossing, now, "face_identity_pending", **previous)
            return None
        (identity_id, source), = crossing.identities.items()
        record["identity"] = identity_id
        if crossing.known_identity is not None:
            agrees = identity_id == crossing.known_identity
            record["label_agrees"] = agrees
            self.zone_stats["label_agree" if agrees else "label_disagree"] += 1
        if not applied:
            self._note(event_id, crossing, now, reason, **previous)
            return None
        self._note(event_id, crossing, now, "linked", **previous)
        self.terminal.append(dict(self.last_decisions[event_id]))
        self.pending.pop(event_id)
        self._emitted[event_id] = None
        while len(self._emitted) > self.max_pending * 2:
            self._emitted.popitem(last=False)
        self.linked_count += 1
        return AssociationProposal(
            event_id,
            crossing.side_track_id,
            record["facing_tracks"][0],
            identity_id,
            float(previous.get("appearance_score") or 0.0),
            float(
                self.zone_margin if record["second_distance"] is None
                else record["second_distance"] - record["distance"]
            ),
            ("crossing_zone_position", "bounded_capture_time", source),
            tuple(record["facing_tracks"]),
            "cross_view_position",
        )

    def resolve_pending(self, now: float) -> list[AssociationProposal]:
        if not isfinite(now):
            return []
        self._prune(max(self._latest_time, now))
        proposals = []
        for event_id in list(self.pending):
            proposal = self.resolve_crossing(event_id, now)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _drop_track(self, view: str, track_id: str) -> None:
        self.tracks[view].pop(track_id, None)
        self._chains_cache = None
        if view == "side":
            paired = self._side_to_facing.pop(track_id, None)
            if paired is not None:
                self._facing_to_side.pop(paired, None)
        else:
            paired = self._facing_to_side.pop(track_id, None)
            if paired is not None:
                self._side_to_facing.pop(paired, None)

    def _prune(self, now: float) -> None:
        self._latest_time = now
        for view, tracks in self.tracks.items():
            for track_id, track in list(tracks.items()):
                if now - track.last_seen > self.retention_seconds:
                    self._drop_track(view, track_id)
        for event_id, crossing in list(self.pending.items()):
            if now > crossing.deadline:
                self.pending.pop(event_id)
                labelled = crossing.known_identity is not None
                self.expired_count += not labelled
                previous = self.last_decisions.get(event_id, {})
                self._note(
                    event_id, crossing, now,
                    "labelled_check" if labelled else "expired_unresolved",
                    **{
                        key: value for key, value in previous.items()
                        if key not in {
                            "event_id", "side_track_id", "crossing_at", "decision_at",
                            "reason", "zone",
                        }
                    },
                    last_reject_reason=previous.get("reason"),
                )
                self.terminal.append(dict(self.last_decisions[event_id]))

    def reset(self) -> None:
        """Clear transient links when counting resets or capture namespaces reset."""
        for tracks in self.tracks.values():
            tracks.clear()
        self.pending.clear()
        self._emitted.clear()
        self._side_to_facing.clear()
        self._facing_to_side.clear()
        self._zone_claims.clear()
        self._chains_cache = None
        self.linked_count = self.expired_count = 0
        self.last_decisions.clear()
        self.terminal.clear()

    def summary(self) -> dict:
        return {
            "pending": len(self.pending),
            "linked": self.linked_count,
            "expired_unresolved": self.expired_count,
            "side_tracks": len(self.tracks["side"]),
            "facing_tracks": len(self.tracks["facing"]),
            "side_appearance_ready": sum(
                self._features(track, self._latest_time) is not None
                for track in self.tracks["side"].values()
            ),
            "facing_appearance_ready": sum(
                self._features(track, self._latest_time) is not None
                for track in self.tracks["facing"].values()
            ),
            "method": "osnet_multiframe_reciprocal",
            "calibrated_geometry": False,
            "similarity_threshold": self.similarity_threshold,
            "margin_threshold": self.margin_threshold,
            "max_track_gap_seconds": self.max_track_gap_seconds,
            "sole_candidate_threshold": self.sole_candidate_threshold,
            "last_decisions": [dict(value) for value in self.last_decisions.values()],
            "zone": None
            if self.zone_mode == "off"
            else {"mode": self.zone_mode, **self.zone.summary(), **self.zone_stats},
        }

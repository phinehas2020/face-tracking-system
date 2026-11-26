"""Event-scoped face matching; embeddings never appear in API responses."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import numpy as np

from .domain import utc_now
from .evidence import save_tracklet
from .recognizer import FACE_MODEL
from .store import EventStore

MATCH_THRESHOLD = 0.60
NEW_THRESHOLD = 0.35
MIN_MARGIN = 0.08
MIN_QUALITY = 0.70
MIN_SAMPLES = 3


@dataclass
class FaceTrack:
    samples: list[np.ndarray] = field(default_factory=list, repr=False)
    identity_id: str | None = None
    label: str = "Reading face"
    state: str = "collecting"
    similarity: float | None = None
    quality: float = 0
    last_seen: float = 0
    last_face: float = 0
    resolve_after: str | None = None
    conflicts: list[np.ndarray] = field(default_factory=list, repr=False)
    qualities: list[float] = field(default_factory=list)
    required_samples: int = MIN_SAMPLES
    first_at: str | None = None
    last_at: str | None = None


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 512-dimensional face template")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError("Empty face template")
    return vector / norm


def recount(db, event_id: str) -> None:
    row = db.execute(
        """SELECT COUNT(*) AS raw,
        COUNT(DISTINCT CASE WHEN direction='in' THEN identity_id END) AS unique_count,
        SUM(CASE WHEN direction='in' AND identity_id IS NOT NULL
            THEN 1 ELSE 0 END) AS known_entries,
        SUM(CASE WHEN identity_id IS NULL THEN 1 ELSE 0 END) AS unresolved
        FROM passages WHERE event_id=?""",
        (event_id,),
    ).fetchone()
    db.execute(
        """UPDATE event_counters SET raw_passages=?,unique_attendees=?,
        repeat_passages=?,unresolved=?,updated_at=? WHERE event_id=?""",
        (
            row["raw"],
            row["unique_count"],
            (row["known_entries"] or 0) - row["unique_count"],
            row["unresolved"] or 0,
            utc_now(),
            event_id,
        ),
    )


class FaceRegistry:
    def __init__(self, store: EventStore, event_id: str, camera_id: str | None = None):
        self.store = store
        self.event_id = event_id
        # Tracklet evidence is persisted only for a named camera view.
        self.camera_id = camera_id
        self.evidence_saved = 0
        self.evidence_errors = 0
        self.tracks: dict[str, FaceTrack] = {}
        self.templates: dict[str, np.ndarray] = {}
        self.labels: dict[str, str] = {}
        self.confirmed_variants: dict[str, list[np.ndarray]] = {}
        # Visitor numbers continue past those another recognizer gave in this event.
        self.label_floor = 0
        with store.connect() as db:
            for row in db.execute(
                "SELECT label FROM face_templates WHERE event_id=? AND model!=?",
                (event_id, FACE_MODEL),
            ):
                tail = row["label"].split()[-1] if row["label"] else ""
                if row["label"].startswith("Visitor ") and tail.isdigit():
                    self.label_floor = max(self.label_floor, int(tail))
            for row in db.execute(
                "SELECT * FROM face_templates WHERE event_id=? AND model=?", (event_id, FACE_MODEL)
            ):
                self.templates[row["identity_id"]] = normalized(
                    np.frombuffer(row["embedding"], dtype=np.float32)
                )
                self.labels[row["identity_id"]] = row["label"]
            for row in db.execute(
                "SELECT identity_id,embedding FROM confirmed_face_variants "
                "WHERE event_id=? AND model=?",
                (event_id, FACE_MODEL),
            ):
                self.confirmed_variants.setdefault(row["identity_id"], []).append(
                    normalized(np.frombuffer(row["embedding"], dtype=np.float32))
                )

    def identity_similarity(self, vector: np.ndarray, identity_id: str) -> float:
        references = [self.templates[identity_id], *self.confirmed_variants.get(identity_id, [])]
        return max(float(vector @ reference) for reference in references)

    def observe(
        self,
        track_id: str,
        embedding: np.ndarray,
        quality: float,
        active_tracks: set[str],
        now: float | None = None,
        observed_at: str | None = None,
    ) -> FaceTrack:
        now = time.monotonic() if now is None else now
        state = self.tracks.setdefault(track_id, FaceTrack())
        state.last_seen = now
        if quality < 0.35:
            return state
        vector = normalized(embedding)
        stamp = observed_at or utc_now()
        state.first_at = state.first_at or stamp
        state.last_at = stamp
        occupied = {
            value.identity_id
            for key, value in self.tracks.items()
            if key != track_id and key in active_tracks and value.identity_id
        }
        if state.identity_id:
            own_score = self.identity_similarity(vector, state.identity_id)
            state.similarity = round(own_score, 3)
            state.last_face, state.quality = now, quality
            if state.identity_id in occupied:
                state.state, state.label = "conflict", "Overlapping tracks; checking"
                return state
            alternatives = sorted(
                (
                    self.identity_similarity(vector, key)
                    for key in self.templates
                    if key != state.identity_id
                ),
                reverse=True,
            )
            contradictory = own_score < 0.3 or (
                alternatives
                and alternatives[0] >= MATCH_THRESHOLD
                and alternatives[0] - own_score >= MIN_MARGIN
            )
            if own_score >= MATCH_THRESHOLD and not contradictory:
                recovering = state.state == "conflict"
                state.conflicts.clear()
                state.samples = (state.samples + [vector])[-8:]
                state.qualities = (state.qualities + [quality])[-8:]
                state.state, state.label = "matched", self.labels[state.identity_id]
                if recovering:
                    self._resolve_pending(track_id, state, observed_at)
                return state
            if not contradictory and not state.conflicts:
                # Weak pose evidence must not erase a previously confirmed identity
                # on a continuously tracked body, or contaminate its template.
                state.state, state.label = "tracked", self.labels[state.identity_id]
                return state
            if (
                not state.conflicts
                or float(vector @ normalized(np.mean(state.conflicts, axis=0))) < 0.5
            ):
                state.conflicts = [vector]
                state.resolve_after = observed_at or utc_now()
            else:
                state.conflicts.append(vector)
            state.state, state.label = "conflict", "Conflicting face; checking"
            if len(state.conflicts) < (MIN_SAMPLES if quality >= MIN_QUALITY else 6):
                return state
            state.samples = state.conflicts[:-1]
            state.qualities = [quality] * len(state.samples)
            state.conflicts = []
            state.identity_id = None
        state.last_face, state.quality = now, quality
        state.samples.append(vector)
        state.samples = state.samples[-12:]
        state.qualities = (state.qualities + [quality])[-12:]
        # One stray association/pose must not erase the entire track's evidence or
        # orphan a pending crossing. Require a dominant consistent sample cluster.
        matrix = np.stack(state.samples)
        similarities = matrix @ matrix.T
        groups = []
        for seed, row in enumerate(similarities):
            group = [seed]
            for index in np.argsort(-row):
                if index != seed and all(similarities[index, member] >= 0.5 for member in group):
                    group.append(int(index))
            groups.append(group)
        winner = max(groups, key=len)
        other = [i for i in range(len(state.samples)) if i not in winner]
        high_quality = sum(state.qualities[i] >= MIN_QUALITY for i in winner) >= MIN_SAMPLES
        state.required_samples = MIN_SAMPLES if high_quality else 6
        if len(winner) < state.required_samples or len(winner) - len(other) < 2:
            state.state, state.label = "collecting", "Reading consistent face samples"
            return state
        if max(sum(index in other for index in group) for group in groups) >= MIN_SAMPLES:
            state.state, state.label = "review", "Multiple face clusters; unresolved"
            if state.resolve_after is None:
                state.resolve_after = observed_at or utc_now()
            return state
        weights = np.asarray([state.qualities[i] for i in winner], dtype=np.float32) ** 2
        template = normalized(np.average(matrix[winner], axis=0, weights=weights))
        ranked = sorted(
            ((self.identity_similarity(template, key), key) for key in self.templates), reverse=True
        )
        score, best = ranked[0] if ranked else (0.0, None)
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        identity_id = None
        previous_identity = state.identity_id
        state.similarity = round(score, 3) if ranked else None
        match_threshold = MATCH_THRESHOLD if high_quality else 0.68
        margin = MIN_MARGIN if high_quality else 0.12
        new_threshold = NEW_THRESHOLD if high_quality else 0.25
        if best and score >= match_threshold and score - second >= margin:
            if best in occupied:
                state.state, state.label = "review", "Overlapping tracks; unresolved"
            else:
                identity_id = best
                if previous_identity != best:
                    state.state = "matched"
                state.label = self.labels[best]
        elif not ranked or score < new_threshold:
            state.samples = [state.samples[i] for i in winner]
            state.qualities = [state.qualities[i] for i in winner]
            identity_id = "face-" + uuid.uuid4().hex
            state.state = "new"
            next_number = (
                max(
                    (
                        int(label.split()[-1])
                        for label in self.labels.values()
                        if label.startswith("Visitor ") and label.split()[-1].isdigit()
                    ),
                    default=self.label_floor,
                )
                + 1
            )
            state.label = f"Visitor {next_number:03d}"
            timestamp = utc_now()
            with self.store.transaction() as db:
                db.execute(
                    """INSERT INTO identities
                    (id,event_id,created_at,updated_at,status,confidence,prototype_count)
                    VALUES (?,?,?,?,'face_observed',?,?)""",
                    (identity_id, self.event_id, timestamp, timestamp, quality, len(state.samples)),
                )
                db.execute(
                    """INSERT INTO face_templates
                    (identity_id,event_id,model,embedding,label,quality,samples,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        identity_id,
                        self.event_id,
                        FACE_MODEL,
                        template.tobytes(),
                        state.label,
                        quality,
                        len(state.samples),
                        timestamp,
                    ),
                )
            self.templates[identity_id] = template
            self.labels[identity_id] = state.label
        else:
            state.state, state.label = "review", "Uncertain face match"
        state.identity_id = identity_id
        if identity_id and previous_identity != identity_id:
            self._resolve_pending(track_id, state, observed_at)
        return state

    def _resolve_pending(self, track_id: str, state: FaceTrack, observed_at: str | None):
        # Use video time during replay, not variable CPU/wall-clock delay.
        source_now = datetime.fromisoformat(observed_at) if observed_at else datetime.now(UTC)
        cutoff = (source_now - timedelta(seconds=10)).isoformat(timespec="milliseconds")
        if state.resolve_after:
            cutoff = max(cutoff, state.resolve_after)
        with self.store.transaction() as db:
            db.execute(
                """UPDATE passages SET identity_id=?,status='face_matched',face_quality=?,
                    evidence=json_set(evidence,'$.identity_basis','later_face_match')
                WHERE event_id=? AND track_id=? AND identity_id IS NULL AND observed_at>=?""",
                (state.identity_id, state.quality, self.event_id, track_id, cutoff),
            )
            db.execute(
                "UPDATE identities SET updated_at=? WHERE id=?", (utc_now(), state.identity_id)
            )
            recount(db, self.event_id)

    def bind_face_to_body(self, body_key, face, active_bodies, now, observed_at):
        """Transfer confirmed evidence, never share mutable face/body histories."""
        if not face.identity_id or face.state in {"conflict", "review", "collecting"}:
            return
        body = self.tracks.get(body_key)
        if body and (
            body.state == "conflict" or (body.identity_id and body.identity_id != face.identity_id)
        ):
            return
        if body_key in active_bodies and any(
            key != body_key and key in active_bodies and value.identity_id == face.identity_id
            for key, value in self.tracks.items()
        ):
            return
        boundaries = [
            value for value in [face.resolve_after, body.resolve_after if body else None] if value
        ]
        bound = replace(
            face,
            samples=list(face.samples),
            qualities=list(face.qualities),
            conflicts=list(face.conflicts),
            last_seen=body.last_seen if body else now,
            resolve_after=max(boundaries) if boundaries else None,
        )
        self.tracks[body_key] = bound
        if not (
            body
            and body.identity_id == bound.identity_id
            and body.state in {"new", "matched", "tracked"}
        ):
            self._resolve_pending(body_key, bound, observed_at)

    def _persist(self, key: str, state: FaceTrack) -> None:
        """Keep a finished tracklet's samples for the global resolver.

        Evidence storage must never interrupt counting, so storage failures are
        counted and reported instead of raised.
        """
        if not self.camera_id or not state.samples or not state.first_at:
            return
        resolved = state.identity_id and state.state in {"new", "matched", "tracked"}
        try:
            saved = save_tracklet(
                self.store,
                self.event_id,
                self.camera_id,
                key,
                FACE_MODEL,
                state.samples,
                state.qualities,
                state.first_at,
                state.last_at or state.first_at,
                state.identity_id if resolved else None,
                state.state,
            )
        except (sqlite3.Error, ValueError):
            self.evidence_errors += 1
            return
        self.evidence_saved += int(saved)

    def expire(self, active_tracks: set[str], now: float) -> None:
        for key in active_tracks:
            if key in self.tracks:
                self.tracks[key].last_seen = now
        kept = {}
        for key, state in self.tracks.items():
            if now - state.last_seen <= 5:
                kept[key] = state
            else:
                self._persist(key, state)
        self.tracks = kept

    def flush_evidence(self) -> None:
        """Persist tracklets that are still active, e.g. at shutdown or replay end."""
        for key, state in self.tracks.items():
            self._persist(key, state)

    def clear_tracks(self) -> None:
        self.flush_evidence()
        self.tracks.clear()

    def summary(self) -> dict:
        with self.store.connect() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    """SELECT f.identity_id,f.label,f.quality,f.samples,
                COUNT(CASE WHEN p.direction='in' THEN 1 END) AS entries,
                MAX(p.observed_at) AS last_passage
                FROM face_templates f LEFT JOIN passages p
                ON p.identity_id=f.identity_id AND p.event_id=f.event_id
                WHERE f.event_id=? AND f.model=? GROUP BY f.identity_id ORDER BY f.label""",
                    (self.event_id, FACE_MODEL),
                )
            ]
            unresolved = db.execute(
                """SELECT COUNT(*) FROM passages
                WHERE event_id=? AND direction='in' AND identity_id IS NULL""",
                (self.event_id,),
            ).fetchone()[0]
        unique = sum(row["entries"] > 0 for row in rows)
        return {
            "known_faces": len(rows),
            "unique_attendees": unique,
            "repeat_entries": sum(max(0, row["entries"] - 1) for row in rows),
            "unresolved_entries": unresolved,
            "visitors": rows,
            "match_threshold": MATCH_THRESHOLD,
            "new_threshold": NEW_THRESHOLD,
            "minimum_samples": MIN_SAMPLES,
            "smaller_face_minimum_samples": 6,
            "calibrated": False,
        }

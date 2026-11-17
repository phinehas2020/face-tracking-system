"""Durable per-tracklet face evidence.

The online registry decides once, at crossing time, and then forgets. This table
keeps the face samples of every tracklet -- including those that never reached
enrollment -- so a later global pass can reconsider them together. Rows are
event-scoped and cascade with the event; embeddings never leave this module's
callers through an API response.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from .domain import utc_now
from .store import EventStore

EMBEDDING_SIZE = 512


@dataclass(frozen=True)
class TrackletEvidence:
    id: str
    camera_id: str
    track_key: str
    kind: str
    first_observed_at: str
    last_observed_at: str
    embeddings: np.ndarray  # (n, 512) float32, unit rows
    qualities: np.ndarray  # (n,) float32
    online_identity_id: str | None
    online_state: str

    @property
    def prototype(self) -> np.ndarray:
        """Quality-squared weighted mean, matching the online enrollment template."""
        weights = np.maximum(self.qualities, 1e-3) ** 2
        vector = np.average(self.embeddings, axis=0, weights=weights)
        return (vector / max(float(np.linalg.norm(vector)), 1e-9)).astype(np.float32)

    @property
    def consistency(self) -> float:
        """Mean pairwise cosine; low values mean the tracklet mixes different faces."""
        count = len(self.embeddings)
        if count < 2:
            return 1.0
        matrix = self.embeddings @ self.embeddings.T
        return float((matrix.sum() - np.trace(matrix)) / (count * (count - 1)))


def track_kind(track_key: str) -> str:
    return "face" if "face-track-" in track_key else "body"


def save_tracklet(
    store: EventStore,
    event_id: str,
    camera_id: str,
    track_key: str,
    model: str,
    samples: list[np.ndarray],
    qualities: list[float],
    first_observed_at: str,
    last_observed_at: str,
    online_identity_id: str | None,
    online_state: str,
) -> bool:
    """Upsert one tracklet. A later flush only replaces a row when it holds more samples."""
    count = min(len(samples), len(qualities))
    if count == 0:
        return False
    matrix = np.stack(samples[:count]).astype(np.float32)
    if matrix.shape != (count, EMBEDDING_SIZE) or not np.isfinite(matrix).all():
        raise ValueError("Tracklet evidence must be finite 512-dimensional face samples")
    quality = np.asarray(qualities[:count], dtype=np.float32)
    with store.transaction() as db:
        db.execute(
            """INSERT INTO tracklet_evidence
            (id,event_id,camera_id,track_key,kind,model,first_observed_at,last_observed_at,
             sample_count,embeddings,qualities,online_identity_id,online_state,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id,camera_id,track_key) DO UPDATE SET
              last_observed_at=excluded.last_observed_at,
              sample_count=excluded.sample_count,
              embeddings=excluded.embeddings,
              qualities=excluded.qualities,
              online_identity_id=excluded.online_identity_id,
              online_state=excluded.online_state
            WHERE excluded.sample_count >= tracklet_evidence.sample_count""",
            (
                "trk-" + uuid.uuid4().hex,
                event_id,
                camera_id,
                track_key,
                track_kind(track_key),
                model,
                first_observed_at,
                last_observed_at,
                count,
                matrix.tobytes(),
                quality.tobytes(),
                online_identity_id,
                online_state,
                utc_now(),
            ),
        )
    return True


def load_tracklets(store: EventStore, event_id: str, model: str) -> list[TrackletEvidence]:
    with store.connect() as db:
        rows = db.execute(
            "SELECT * FROM tracklet_evidence WHERE event_id=? AND model=? "
            "ORDER BY first_observed_at,id",
            (event_id, model),
        ).fetchall()
    result = []
    for row in rows:
        embeddings = np.frombuffer(row["embeddings"], dtype=np.float32).reshape(
            -1, EMBEDDING_SIZE
        )
        result.append(
            TrackletEvidence(
                id=row["id"],
                camera_id=row["camera_id"],
                track_key=row["track_key"],
                kind=row["kind"],
                first_observed_at=row["first_observed_at"],
                last_observed_at=row["last_observed_at"],
                embeddings=embeddings,
                qualities=np.frombuffer(row["qualities"], dtype=np.float32),
                online_identity_id=row["online_identity_id"],
                online_state=row["online_state"],
            )
        )
    return result

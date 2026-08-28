from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EventStatus(StrEnum):
    SETUP = "setup"
    LIVE = "live"
    CLOSED = "closed"
    PURGED = "purged"


class LinkStatus(StrEnum):
    AUTO_MERGED = "auto_merged"
    REVIEW = "review"
    CONFIRMED_SAME = "confirmed_same"
    CONFIRMED_DIFFERENT = "confirmed_different"


@dataclass(frozen=True, slots=True)
class PairEvidence:
    face_similarity: float | None
    body_similarity: float | None
    face_quality: float
    body_quality: float
    supporting_frames: int
    time_delta_seconds: float
    topology_score: float
    impossible_overlap: bool = False
    candidate_margin: float = 0.0
    camera_pair: str = "unknown"

    def as_features(self) -> dict[str, float]:
        return {
            "face_similarity": self.face_similarity or 0.0,
            "body_similarity": self.body_similarity or 0.0,
            "face_quality": self.face_quality,
            "body_quality": self.body_quality,
            "supporting_frames": min(self.supporting_frames / 20.0, 1.0),
            "time_proximity": 1.0 / (1.0 + max(self.time_delta_seconds, 0.0) / 900.0),
            "topology_score": self.topology_score,
            "candidate_margin": self.candidate_margin,
        }


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    probability: float
    status: LinkStatus
    explanation: str
    feature_contributions: dict[str, float]


JSONDict = dict[str, Any]

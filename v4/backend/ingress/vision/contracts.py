from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    sequence: int
    captured_monotonic_ns: int
    captured_utc: str
    image_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    bbox: BBox
    confidence: float
    age_frames: int

    @property
    def footpoint(self) -> tuple[float, float]:
        left, _top, right, bottom = self.bbox
        return ((left + right) / 2.0, bottom)


@dataclass(frozen=True, slots=True)
class Observation:
    track_id: str
    face_embedding: np.ndarray | None
    body_embedding: np.ndarray | None
    face_quality: float
    body_quality: float
    metadata: dict[str, object] = field(default_factory=dict)


class PersonDetector(Protocol):
    def detect(self, frame: FramePacket) -> list[Detection]: ...


class MultiObjectTracker(Protocol):
    def update(self, frame: FramePacket, detections: list[Detection]) -> list[Track]: ...


class PersonTrackProvider(Protocol):
    def track(self, frame: FramePacket) -> list[Track]: ...


class FaceEncoder(Protocol):
    def encode(
        self, image_bgr: np.ndarray, person_bbox: BBox
    ) -> tuple[np.ndarray, float] | None: ...


class BodyEncoder(Protocol):
    def encode(
        self, image_bgr: np.ndarray, person_bbox: BBox
    ) -> tuple[np.ndarray, float] | None: ...

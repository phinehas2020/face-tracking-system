from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..portal import PortalStateMachine
from .contracts import BodyEncoder, FaceEncoder, FramePacket, Observation, PersonTrackProvider


@dataclass(frozen=True, slots=True)
class PassageCandidate:
    camera_id: str
    track_id: str
    observed_at: str
    face_embedding: np.ndarray | None
    body_embedding: np.ndarray | None
    face_quality: float
    body_quality: float
    supporting_frames: int


@dataclass(slots=True)
class TrackTemplate:
    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation, maximum: int = 24) -> None:
        self.observations.append(observation)
        self.observations.sort(
            key=lambda item: item.face_quality + 0.35 * item.body_quality, reverse=True
        )
        del self.observations[maximum:]

    @staticmethod
    def _weighted_template(
        observations: list[Observation], field_name: str, quality_name: str
    ) -> tuple[np.ndarray | None, float]:
        usable = [item for item in observations if getattr(item, field_name) is not None]
        if not usable:
            return None, 0.0
        weights = np.asarray(
            [max(float(getattr(item, quality_name)), 0.01) for item in usable], dtype=np.float32
        )
        vectors = np.stack([getattr(item, field_name) for item in usable])
        template = np.average(vectors, axis=0, weights=weights)
        norm = float(np.linalg.norm(template))
        if norm > 0:
            template = template / norm
        return template.astype(np.float32), float(weights.mean())

    def summarize(self, camera_id: str, track_id: str, observed_at: str) -> PassageCandidate:
        face, face_quality = self._weighted_template(
            self.observations, "face_embedding", "face_quality"
        )
        body, body_quality = self._weighted_template(
            self.observations, "body_embedding", "body_quality"
        )
        return PassageCandidate(
            camera_id=camera_id,
            track_id=track_id,
            observed_at=observed_at,
            face_embedding=face,
            body_embedding=body,
            face_quality=face_quality,
            body_quality=body_quality,
            supporting_frames=len(self.observations),
        )


class VisionPipeline:
    """Frame-to-passage pipeline; identity resolution happens after passage emission."""

    def __init__(
        self,
        tracker: PersonTrackProvider,
        portal: PortalStateMachine,
        face_encoder: FaceEncoder | None = None,
        body_encoder: BodyEncoder | None = None,
    ) -> None:
        self.tracker = tracker
        self.portal = portal
        self.face_encoder = face_encoder
        self.body_encoder = body_encoder
        self.templates: dict[str, TrackTemplate] = {}

    def process(self, frame: FramePacket) -> list[PassageCandidate]:
        height, width = frame.image_bgr.shape[:2]
        emitted: list[PassageCandidate] = []
        for track in self.tracker.track(frame):
            face_embedding = body_embedding = None
            face_quality = body_quality = 0.0
            if self.face_encoder is not None:
                encoded = self.face_encoder.encode(frame.image_bgr, track.bbox)
                if encoded is not None:
                    face_embedding, face_quality = encoded
            if self.body_encoder is not None:
                encoded = self.body_encoder.encode(frame.image_bgr, track.bbox)
                if encoded is not None:
                    body_embedding, body_quality = encoded
            template = self.templates.setdefault(track.track_id, TrackTemplate())
            template.add(
                Observation(
                    track_id=track.track_id,
                    face_embedding=face_embedding,
                    body_embedding=body_embedding,
                    face_quality=face_quality,
                    body_quality=body_quality,
                    metadata={"sequence": frame.sequence, "confidence": track.confidence},
                )
            )
            x, y = track.footpoint
            crossed = self.portal.update(track.track_id, (x / width, y / height))
            if crossed:
                emitted.append(
                    template.summarize(frame.camera_id, track.track_id, frame.captured_utc)
                )
        return emitted

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import BBox, Detection, FramePacket, Track


class UltralyticsPersonDetector:
    """Optional YOLO26 adapter. Model installation and weights are explicit."""

    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the 'vision' extra to use Ultralytics") from exc
        self.model: Any = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: FramePacket) -> list[Detection]:
        result = self.model.predict(
            frame.image_bgr, classes=[0], conf=self.confidence, verbose=False
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            strict=True,
        ):
            detections.append(
                Detection(tuple(float(value) for value in box), float(confidence), int(class_id))
            )
        return detections


class UltralyticsPersonTracker:
    """YOLO26 + ByteTrack adapter with stable per-camera track identifiers."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.25,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the 'vision' extra to use Ultralytics") from exc
        self.model: Any = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker
        self._ages: dict[str, int] = {}

    def track(self, frame: FramePacket) -> list[Track]:
        result = self.model.track(
            frame.image_bgr,
            classes=[0],
            conf=self.confidence,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        output: list[Track] = []
        for box, confidence, raw_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.id.cpu().numpy(),
            strict=True,
        ):
            track_id = f"{frame.camera_id}:{int(raw_id)}"
            self._ages[track_id] = self._ages.get(track_id, 0) + 1
            output.append(
                Track(
                    track_id=track_id,
                    bbox=tuple(float(value) for value in box),
                    confidence=float(confidence),
                    age_frames=self._ages[track_id],
                )
            )
        return output


class InsightFaceEncoder:
    """Face encoder with the correct OpenCV BGR contract."""

    def __init__(self, model_name: str = "buffalo_l", providers: list[str] | None = None) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the 'vision' extra to use InsightFace") from exc
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def encode(self, image_bgr: np.ndarray, person_bbox: BBox) -> tuple[np.ndarray, float] | None:
        left, top, right, bottom = (int(max(value, 0)) for value in person_bbox)
        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None
        # FaceAnalysis expects an OpenCV BGR image. Do not convert to RGB here.
        faces = self.app.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda candidate: float(candidate.det_score))
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        quality = float(face.det_score) * min(float(face.bbox[2] - face.bbox[0]) / 112.0, 1.0)
        return embedding, max(0.0, min(quality, 1.0))

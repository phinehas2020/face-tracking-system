from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .accelerator import TRACKER_LOCK, mps_guard, serialized_mps
from .contracts import BBox, Detection, FramePacket, Track


def _pixel_crop(image: np.ndarray, box: BBox) -> tuple[np.ndarray, tuple[int, int]]:
    """Clip a pixel-space ROI; preserve native BGR pixels and integer origin."""
    height, width = image.shape[:2]
    left, top, right, bottom = box
    if not all(math.isfinite(float(v)) for v in box):
        raise ValueError("ROI coordinates must be finite")
    left, top = max(0, math.floor(left)), max(0, math.floor(top))
    right, bottom = min(width, math.ceil(right)), min(height, math.ceil(bottom))
    if right <= left or bottom <= top:
        raise ValueError("ROI has no pixels inside the image")
    return image[top:bottom, left:right], (left, top)


def _normalized_roi(roi: BBox | None) -> BBox | None:
    if roi is None:
        return None
    if len(roi) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in roi):
        raise ValueError("ROI must contain four finite normalized coordinates")
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("ROI right/bottom must exceed left/top")
    return tuple(float(v) for v in roi)


def _mapped_box(box, origin: tuple[int, int]) -> BBox:
    x, y = origin
    return (float(box[0]) + x, float(box[1]) + y, float(box[2]) + x, float(box[3]) + y)


class _PersonRuntime:
    """Explicit device selection and a visible, one-time MPS fallback."""

    def _configure_runtime(self, device: str, imgsz: int, roi: BBox | None) -> None:
        if device not in {"cpu", "mps", "auto"}:
            raise ValueError("Person device must be cpu, mps, or auto")
        if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz < 32 or imgsz % 32:
            raise ValueError("Person imgsz must be a positive multiple of 32")
        self.requested_device = device
        self.selected_device = "cpu"
        self._runtime_generation = 0
        self.fallback_reason: str | None = None
        self.imgsz = imgsz
        self.roi = _normalized_roi(roi)
        if device != "cpu":
            import torch

            if torch.backends.mps.is_available():
                self.selected_device = "mps"
            else:
                self.fallback_reason = "MPS unavailable; using CPU"

    def _image_roi(self, image: np.ndarray):
        if self.roi is None:
            return image, (0, 0)
        height, width = image.shape[:2]
        left, top, right, bottom = self.roi
        return _pixel_crop(image, (left * width, top * height, right * width, bottom * height))

    @serialized_mps("selected_device")
    def _infer(self, method: str, image: np.ndarray, **kwargs):
        try:
            return getattr(self.model, method)(
                image, device=self.selected_device, imgsz=self.imgsz, **kwargs
            )[0]
        except (RuntimeError, NotImplementedError) as exc:
            if self.selected_device != "mps":
                raise
            # Unsupported MPS operations should not stop counting. The caller can
            # report this fallback; a subsequent CPU failure still propagates.
            self.fallback_reason = f"MPS inference failed ({type(exc).__name__}); using CPU"
            self.selected_device = "cpu"
            self.model.to("cpu")
            predictor = getattr(self.model, "predictor", None)
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()
            self.model.predictor = None
            self._runtime_generation += 1
            if hasattr(self, "_ages"):
                self._ages.clear()
            return getattr(self.model, method)(image, device="cpu", imgsz=self.imgsz, **kwargs)[0]


class UltralyticsPersonDetector(_PersonRuntime):
    """Optional YOLO26 adapter. Model installation and weights are explicit."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.25,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        roi: BBox | None = None,
    ) -> None:
        self._configure_runtime(device, imgsz, roi)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the 'vision' extra to use Ultralytics") from exc
        with mps_guard(self.selected_device):
            self.model: Any = YOLO(model_path)
        self.confidence = confidence

    @serialized_mps("selected_device")
    def detect(self, frame: FramePacket) -> list[Detection]:
        image, origin = self._image_roi(frame.image_bgr)
        result = self._infer("predict", image, classes=[0], conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            strict=True,
        ):
            detections.append(Detection(_mapped_box(box, origin), float(confidence), int(class_id)))
        return detections


class UltralyticsPersonTracker(_PersonRuntime):
    """YOLO26 + ByteTrack adapter with stable per-camera track identifiers."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.1,
        tracker: str | None = None,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        roi: BBox | None = None,
    ) -> None:
        self._configure_runtime(device, imgsz, roi)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install the 'vision' extra to use Ultralytics") from exc
        with mps_guard(self.selected_device):
            self.model: Any = YOLO(model_path)
        self.confidence = confidence
        names = ("bytetrack", "tracktrack", "ocsort", "fasttrack")
        aliases = {name: f"{name}.yaml" for name in names}
        self.tracker = aliases.get(tracker, tracker) or str(
            Path(__file__).with_name("crowd_tracker.yaml")
        )
        self._ages: dict[str, int] = {}

    @serialized_mps("selected_device")
    def reset(self) -> None:
        """Discard identities when the source reconnects."""
        predictor = getattr(self.model, "predictor", None)
        with TRACKER_LOCK:
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()
        self._ages.clear()

    @serialized_mps("selected_device")
    def track(self, frame: FramePacket) -> list[Track]:
        image, origin = self._image_roi(frame.image_bgr)
        with TRACKER_LOCK:
            result = self._infer(
                "track",
                image,
                classes=[0],
                conf=self.confidence,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
        if result.boxes is None or result.boxes.id is None:
            return []
        output: list[Track] = []
        current_ages: dict[str, int] = {}
        for box, confidence, raw_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.id.cpu().numpy(),
            strict=True,
        ):
            suffix = (
                f"{self._runtime_generation}:{int(raw_id)}"
                if self._runtime_generation
                else str(int(raw_id))
            )
            track_id = f"{frame.camera_id}:{suffix}"
            self._ages[track_id] = self._ages.get(track_id, 0) + 1
            current_ages[track_id] = self._ages[track_id]
            output.append(
                Track(
                    track_id=track_id,
                    bbox=_mapped_box(box, origin),
                    confidence=float(confidence),
                    age_frames=self._ages[track_id],
                )
            )
        self._ages = current_ages
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


@dataclass
class DetectedFace:
    bbox: BBox
    embedding: np.ndarray
    quality: float


class LiveFaceEncoder:
    """Accept near-frontal face evidence; body crossing counts are independent."""

    MAX_YAW_DEGREES = 25.0
    MAX_PITCH_DEGREES = 20.0
    # Upper edges of the |yaw| and |pitch| buckets reported in pose_gate. Every
    # face with a valid pose is counted, accepted or not, so the gate can be set
    # from the angles real visitors actually present.
    POSE_BUCKETS = (10.0, 20.0, 30.0, 45.0, 60.0)

    def __init__(
        self,
        *,
        provider: str = "cpu",
        det_size: int = 1280,
        person_rois: bool = False,
        max_person_rois: int = 4,
        tap=None,
    ) -> None:
        import onnxruntime as ort

        if provider not in {"cpu", "coreml", "auto"}:
            raise ValueError("Face provider must be cpu, coreml, or auto")
        if (
            not isinstance(det_size, int)
            or isinstance(det_size, bool)
            or det_size < 32
            or det_size % 32
        ):
            raise ValueError("Face det_size must be a positive multiple of 32")
        if max_person_rois < 0:
            raise ValueError("max_person_rois cannot be negative")
        self.requested_provider = provider
        self.det_size = det_size
        self.person_rois = person_rois
        self.max_person_rois = max_person_rois
        # Optional observer of every face with a readable pose, accepted or not.
        self.tap = tap
        from ..recognizer import load

        # None keeps the detector's own buffalo_l template.
        self.recognizer = load()
        self.recognizer_errors = 0
        self.pose_gate = {
            "max_yaw_degrees": self.MAX_YAW_DEGREES,
            "max_pitch_degrees": self.MAX_PITCH_DEGREES,
            "accepted": 0,
            "rejected_angle": 0,
            "rejected_missing_pose": 0,
        }
        self.fallback_reason: str | None = None
        available = ort.get_available_providers()
        self.selected_provider = "CPUExecutionProvider"
        if provider != "cpu":
            if "CoreMLExecutionProvider" in available:
                self.selected_provider = "CoreMLExecutionProvider"
            else:
                self.fallback_reason = "CoreML unavailable; using CPU"
        self._prepare()

    def _prepare(self) -> None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = [self.selected_provider]
        if self.selected_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "landmark_3d_68", "recognition"],
                providers=providers,
                sess_options=options,
            )
            # InsightFace's CoreML adapter builds reusable fixed-shape sessions.
            self.app.prepare(
                ctx_id=-1 if self.selected_provider == "CPUExecutionProvider" else 0,
                det_size=(self.det_size, self.det_size),
            )
        except (RuntimeError, NotImplementedError) as exc:
            if self.selected_provider == "CPUExecutionProvider":
                raise
            self.fallback_reason = f"CoreML preparation failed ({type(exc).__name__}); using CPU"
            self.selected_provider = "CPUExecutionProvider"
            self._prepare()
            return
        self.model_providers = {
            name: list(model.session.get_providers())
            for name, model in getattr(self.app, "models", {}).items()
            if hasattr(getattr(model, "session", None), "get_providers")
        }
        if self.model_providers and not any(
            self.selected_provider in providers for providers in self.model_providers.values()
        ):
            self.fallback_reason = "Requested provider absent from model sessions; using CPU"
            self.selected_provider = "CPUExecutionProvider"

    def _get(self, image_bgr: np.ndarray):
        try:
            return self.app.get(image_bgr)
        except (RuntimeError, NotImplementedError) as exc:
            if (
                getattr(self, "selected_provider", "CPUExecutionProvider")
                != "CoreMLExecutionProvider"
            ):
                raise
            self.fallback_reason = f"CoreML inference failed ({type(exc).__name__}); using CPU"
            self.selected_provider = "CPUExecutionProvider"
            self._prepare()
            return self.app.get(image_bgr)

    def detect(
        self,
        image_bgr: np.ndarray,
        person_boxes: list[BBox] | None = None,
    ) -> list[DetectedFace]:
        # Full-frame detection covers a face even if body detection missed it.
        regions = [(image_bgr, (0, 0))]
        if getattr(self, "person_rois", False):
            for box in (person_boxes or [])[: self.max_person_rois]:
                left, top, right, bottom = box
                width, height = right - left, bottom - top
                if width <= 0 or height <= 0:
                    continue
                # Slight lateral/upper padding preserves the head at box edges.
                head_box = (
                    left - 0.1 * width,
                    top - 0.05 * height,
                    right + 0.1 * width,
                    top + 0.45 * height,
                )
                try:
                    regions.append(_pixel_crop(image_bgr, head_box))
                except ValueError:
                    continue
        result = []
        tap = getattr(self, "tap", None)
        tapped = []
        pose_counts = {"accepted": 0, "rejected_angle": 0, "rejected_missing_pose": 0}
        empty = [0] * (len(self.POSE_BUCKETS) + 1)
        histograms = {"yaw_histogram": list(empty), "pitch_histogram": list(empty)}
        for image, origin in regions:
            # InsightFace expects BGR. Crops retain native pixel resolution.
            for face in self._get(image):
                width = float(face.bbox[2] - face.bbox[0])
                quality = float(face.det_score) * min(width / 112, 1.0)
                embedding = face.normed_embedding
                if width < 40 or quality < 0.35 or embedding is None:
                    continue
                # InsightFace's 3D landmark model returns PITCH, YAW, ROLL in
                # degrees. Apply this to every size/confidence, including crops;
                # a confident detector response is not a frontal-pose guarantee.
                pose = getattr(face, "pose", None)
                pose = np.asarray(pose, dtype=np.float32) if pose is not None else np.array([])
                if pose.shape != (3,) or not np.isfinite(pose).all():
                    pose_counts["rejected_missing_pose"] += 1
                    continue
                pitch, yaw, _roll = pose
                for name, angle in (("yaw_histogram", yaw), ("pitch_histogram", pitch)):
                    bucket = int(np.searchsorted(self.POSE_BUCKETS, abs(float(angle))))
                    histograms[name][bucket] += 1
                seen = None
                if tap is not None and getattr(face, "kps", None) is not None:
                    seen = {
                        "image": image,
                        "kps": face.kps,
                        "bbox": _mapped_box(face.bbox, origin),
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "det": float(face.det_score),
                        "width": width,
                        "embedding": embedding,
                        "accepted": False,
                    }
                    tapped.append(seen)
                if abs(yaw) > self.MAX_YAW_DEGREES or abs(pitch) > self.MAX_PITCH_DEGREES:
                    pose_counts["rejected_angle"] += 1
                    continue
                if quality < 0.7:
                    # Preserve the existing small-face enrollment gate.
                    points = face.kps
                    if points is None or len(points) < 3:
                        continue
                    eye_span = float(points[1][0] - points[0][0])
                    nose_fraction = float(points[2][0] - points[0][0]) / max(eye_span, 1)
                    if face.det_score < 0.85 or eye_span < 12 or not 0.2 <= nose_fraction <= 0.8:
                        continue
                recognizer = getattr(self, "recognizer", None)
                if recognizer is not None:
                    # The selected recognizer replaces the template; a failure drops
                    # the face rather than store buffalo_l under another name.
                    try:
                        from ..recognizer import aligned_face

                        embedding = recognizer.embed(*aligned_face(image, face.kps))
                    except Exception:
                        self.recognizer_errors = getattr(self, "recognizer_errors", 0) + 1
                        continue
                embedding = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(embedding))
                if not np.isfinite(embedding).all() or norm <= 0:
                    continue
                pose_counts["accepted"] += 1
                if seen is not None:
                    seen["accepted"] = True
                result.append(
                    DetectedFace(
                        _mapped_box(face.bbox, origin),
                        embedding / norm,
                        quality,
                    )
                )
        # Publish a new dictionary atomically for the async worker's diagnostics.
        previous = getattr(self, "pose_gate", {})
        self.pose_gate = {
            "max_yaw_degrees": self.MAX_YAW_DEGREES,
            "max_pitch_degrees": self.MAX_PITCH_DEGREES,
            **{key: previous.get(key, 0) + value for key, value in pose_counts.items()},
            "pose_bucket_upper_degrees": [*self.POSE_BUCKETS, None],
            **{
                key: [old + new for old, new in zip(previous.get(key, empty), value, strict=True)]
                for key, value in histograms.items()
            },
        }
        if tapped:
            with contextlib.suppress(Exception):  # The observer must never cost a face scan.
                tap(tapped)
        return _deduplicate_faces(result)


def _deduplicate_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
    """Suppress the same face observed in overlapping full-frame/head crops."""
    from .continuity import overlap

    kept = []
    for face in sorted(faces, key=lambda candidate: candidate.quality, reverse=True):
        if any(overlap(face.bbox, old.bbox) >= 0.4 for old in kept):
            continue
        kept.append(face)
    return kept


def associate_faces(
    faces: list[DetectedFace],
    tracks: list[Track],
    identity_hints: dict[str, np.ndarray] | None = None,
) -> dict[str, DetectedFace]:
    """One-to-one head geometry, with ambiguity rejection in both directions."""
    from scipy.optimize import linear_sum_assignment

    if not faces or not tracks:
        return {}
    costs = np.full((len(faces), len(tracks) + len(faces)), 1.05, dtype=np.float64)
    costs[:, : len(tracks)] = 10000
    for index, face in enumerate(faces):
        x = (face.bbox[0] + face.bbox[2]) / 2
        y = (face.bbox[1] + face.bbox[3]) / 2
        for column, track in enumerate(tracks):
            left, top, right, bottom = track.bbox
            width, height = max(right - left, 1), max(bottom - top, 1)
            vertical = (y - top) / height
            if not left <= x <= right or not 0 <= vertical <= 0.4:
                continue
            cost = abs(x - (left + right) / 2) / (width / 2) * 0.8 + abs(vertical - 0.12) * 2
            if cost <= 1:
                hint = (identity_hints or {}).get(track.track_id)
                if hint is not None and float(face.embedding @ hint) >= 0.7:
                    cost -= 0.25
                costs[index, column] = cost
    rows, columns = linear_sum_assignment(costs)
    optimum = float(costs[rows, columns].sum())
    result = {}
    for index, column in zip(rows, columns, strict=True):
        if column >= len(tracks):
            continue
        alternative = costs.copy()
        alternative[index, column] = 10000
        other_rows, other_columns = linear_sum_assignment(alternative)
        if float(alternative[other_rows, other_columns].sum()) - optimum < 0.15:
            continue
        result[tracks[column].track_id] = faces[index]
    return result

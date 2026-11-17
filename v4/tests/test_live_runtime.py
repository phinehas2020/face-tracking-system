"""Runtime selection, native crop geometry, and conservative gap recovery."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ingress.vision.adapters import (
    DetectedFace,
    LiveFaceEncoder,
    UltralyticsPersonTracker,
    _normalized_roi,
    _pixel_crop,
)
from ingress.vision.continuity import FaceContinuity, TrackContinuity
from ingress.vision.contracts import FramePacket, Track


def vector(value=1.0):
    return np.array([value, np.sqrt(1 - value * value)], dtype=np.float32)


def tensor(value):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(value)))


@pytest.fixture
def fake_yolo(monkeypatch):
    class Model:
        def __init__(self, path):
            self.calls = []
            self.fail_mps = False
            self.predictor = None

        def track(self, image, **kwargs):
            self.calls.append((image, kwargs))
            if self.fail_mps and kwargs["device"] == "mps":
                raise NotImplementedError("unsupported test operation")
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=tensor([[10, 15, 40, 80]]),
                        conf=tensor([0.9]),
                        id=tensor([7]),
                    )
                )
            ]

        def to(self, device):
            self.moved_to = device

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        ),
    )
    return Model


def test_person_roi_maps_tracks_to_original_and_reports_mps(fake_yolo):
    tracker = UltralyticsPersonTracker(
        device="auto", imgsz=960, roi=(0.25, 0.2, 0.75, 1), tracker="tracktrack"
    )
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    frame = FramePacket("front", 1, 1, "", image)
    tracks = tracker.track(frame)
    crop, options = tracker.model.calls[0]
    assert crop.shape == (160, 200, 3)
    assert tracks[0].bbox == (110, 55, 140, 120)
    assert tracks[0].track_id == "front:7"
    assert options["device"] == tracker.selected_device == "mps"
    assert options["imgsz"] == 960
    assert options["tracker"] == "tracktrack.yaml"
    assert options["persist"] is True


def test_mps_failure_falls_back_once_and_remains_visible(fake_yolo):
    tracker = UltralyticsPersonTracker(device="mps")
    tracker.model.fail_mps = True
    frame = FramePacket("side", 1, 1, "", np.zeros((100, 100, 3), dtype=np.uint8))
    assert tracker.track(frame)[0].track_id == "side:1:7"
    assert tracker.selected_device == "cpu"
    assert "NotImplementedError" in tracker.fallback_reason
    assert tracker.model.moved_to == "cpu"
    tracker.track(frame)
    assert [kwargs["device"] for _, kwargs in tracker.model.calls] == ["mps", "cpu", "cpu"]


def test_unavailable_mps_uses_cpu_with_reason(fake_yolo, monkeypatch):
    monkeypatch.setattr(sys.modules["torch"].backends.mps, "is_available", lambda: False)
    tracker = UltralyticsPersonTracker(device="mps")
    assert tracker.selected_device == "cpu"
    assert "unavailable" in tracker.fallback_reason


@pytest.mark.parametrize("roi", [(-0.1, 0, 1, 1), (0, 0, 0, 1), (0, 0, float("nan"), 1)])
def test_invalid_normalized_roi_rejected(roi):
    with pytest.raises(ValueError):
        _normalized_roi(roi)


def test_native_crop_clips_rounds_outward_and_preserves_bgr():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 137
    crop, origin = _pixel_crop(image, (-20, 10.2, 101.5, 200))
    assert origin == (0, 10)
    assert crop.shape == (90, 102, 3)
    assert np.all(crop[:, :, 0] == 137)
    assert np.all(crop[:, :, 2] == 0)


def face(box, quality=0.95, pose=(0, 0, 0)):
    return SimpleNamespace(
        bbox=np.asarray(box), det_score=quality, normed_embedding=vector(), kps=None, pose=pose
    )


def test_head_crop_maps_and_deduplicates_face_evidence():
    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    encoder.person_rois = True
    encoder.max_person_rois = 4
    calls = []

    def detect(image):
        calls.append(image)
        if image.shape[1] == 500:
            return [face((120, 50, 232, 162), 0.90)]
        # Head crop origin is (80, 25); second observation is better.
        return [face((40, 25, 152, 137), 0.98)]

    encoder.app = SimpleNamespace(get=detect)
    image = np.zeros((600, 500, 3), dtype=np.uint8)
    image[:, :, 0] = 137
    result = encoder.detect(image, person_boxes=[(100, 50, 300, 550)])
    assert len(calls) == 2
    assert calls[1].shape == (250, 240, 3)
    assert np.all(calls[1][:, :, 0] == 137)
    assert len(result) == 1
    assert result[0].bbox == (120, 50, 232, 162)
    assert result[0].quality == 0.98


def test_roi_mode_keeps_independent_fullframe_detection():
    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    encoder.person_rois = True
    encoder.max_person_rois = 4
    encoder.app = SimpleNamespace(get=lambda image: [face((10, 10, 122, 122))])
    assert len(encoder.detect(np.zeros((200, 200, 3)), person_boxes=[])) == 1


def test_face_provider_passes_fixed_size_and_reports_session_providers(monkeypatch):
    class App:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.models = {
                "detection": SimpleNamespace(
                    session=SimpleNamespace(
                        get_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    )
                )
            }

        def prepare(self, **kwargs):
            self.prepared = kwargs

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=App))
    encoder = LiveFaceEncoder(provider="coreml", det_size=640)
    assert encoder.app.prepared == {"ctx_id": 0, "det_size": (640, 640)}
    assert encoder.selected_provider == "CoreMLExecutionProvider"
    assert encoder.model_providers["detection"][0] == "CoreMLExecutionProvider"
    assert "landmark_3d_68" in encoder.app.options["allowed_modules"]


@pytest.mark.parametrize(
    "pose,accepted,reason",
    [
        ((0, 0, 0), True, None),
        ((20, 25, 35), True, None),  # Roll is handled by face alignment.
        ((-20, -25, -35), True, None),
        ((20.1, 0, 0), False, "rejected_angle"),
        ((0, -25.1, 0), False, "rejected_angle"),
        ((24, 0, 0), False, "rejected_angle"),  # Pitch/yaw must not be swapped.
        ((0, 24, 0), True, None),
        (None, False, "rejected_missing_pose"),
        ((0, float("nan"), 0), False, "rejected_missing_pose"),
        ((0, 0), False, "rejected_missing_pose"),
    ],
)
def test_large_confident_face_cannot_bypass_pose_gate(pose, accepted, reason):
    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    encoder.app = SimpleNamespace(get=lambda image: [face((10, 10, 210, 210), 0.99, pose)])
    result = encoder.detect(np.zeros((240, 240, 3), dtype=np.uint8))
    assert bool(result) is accepted
    assert encoder.pose_gate["accepted"] == int(accepted)
    if reason:
        assert encoder.pose_gate[reason] == 1


def test_face_session_silent_cpu_fallback_is_reported(monkeypatch):
    class App:
        def __init__(self, **kwargs):
            self.models = {
                "recognition": SimpleNamespace(
                    session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
                )
            }

        def prepare(self, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=App))
    encoder = LiveFaceEncoder(provider="coreml")
    assert encoder.selected_provider == "CPUExecutionProvider"
    assert "absent" in encoder.fallback_reason


def test_face_sparse_motion_matches_strong_embedding_at_bound():
    continuity = FaceContinuity(max_motion_widths=3)
    key = continuity.update([DetectedFace((0, 0, 50, 50), vector(), 0.9)], 0)[0]
    assert continuity.update([DetectedFace((140, 0, 190, 50), vector(), 0.9)], 0.4)[0] == key
    continuity.reset()
    assert not continuity.velocities


def test_face_sparse_motion_does_not_accept_weak_or_ambiguous_match():
    continuity = FaceContinuity(max_motion_widths=3)
    key = continuity.update([DetectedFace((0, 0, 50, 50), vector(), 0.9)], 0)[0]
    assert continuity.update([DetectedFace((80, 0, 130, 50), vector(0.70), 0.9)], 0.4)[0] != key
    continuity.reset()
    key = continuity.update([DetectedFace((0, 0, 50, 50), vector(), 0.9)], 1)[0]
    results = continuity.update(
        [
            DetectedFace((80, 0, 130, 50), vector(), 0.9),
            DetectedFace((81, 0, 131, 50), vector(), 0.9),
        ],
        1.4,
    )
    assert key not in results.values()


def body(raw, left):
    return Track(raw, (left, 10, left + 60, 180), 0.9, 1)


def test_learned_body_short_gap_recovered_without_overlap():
    pytest.importorskip("cv2")
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    continuity = TrackContinuity()
    key = continuity.update([body("a", 10)], frame, 0, {"a": vector()})[0].track_id
    result = continuity.update([body("b", 100)], frame, 0.4, {"b": vector()})
    assert result[0].track_id == key
    assert continuity.handovers == 1
    assert continuity.raw_ids == {"b": key}


@pytest.mark.parametrize("positions", [[300], [100, 101]])
def test_learned_body_rejects_long_jump_and_ambiguous_candidates(positions):
    pytest.importorskip("cv2")
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    continuity = TrackContinuity()
    key = continuity.update([body("a", 10)], frame, 0, {"a": vector()})[0].track_id
    tracks = [body(str(i), x) for i, x in enumerate(positions)]
    result = continuity.update(tracks, frame, 0.4, {t.track_id: vector() for t in tracks})
    assert key not in {t.track_id for t in result}


def test_learned_body_never_reuses_currently_visible_track():
    pytest.importorskip("cv2")
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    continuity = TrackContinuity()
    key = continuity.update([body("a", 10)], frame, 0, {"a": vector()})[0].track_id
    result = continuity.update(
        [body("a", 10), body("b", 100)], frame, 0.4, {"a": vector(), "b": vector()}
    )
    assert result[0].track_id == key
    assert result[1].track_id != key


def test_recorded_face_continuity_default_remains_overlap_only():
    continuity = FaceContinuity()
    key = continuity.update([DetectedFace((0, 0, 50, 50), vector(), 0.9)], 0)[0]
    assert continuity.update([DetectedFace((80, 0, 130, 50), vector(), 0.9)], 0.4)[0] != key


def test_pose_histograms_count_rejected_and_accepted_faces():
    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    poses = iter([(5, 5, 0), (0, 40, 0), (25, 70, 0)])
    encoder.app = SimpleNamespace(get=lambda image: [face((10, 10, 210, 210), 0.99, next(poses))])
    for _ in range(3):
        encoder.detect(np.zeros((240, 240, 3), dtype=np.uint8))
    gate = encoder.pose_gate
    assert gate["pose_bucket_upper_degrees"] == [10.0, 20.0, 30.0, 45.0, 60.0, None]
    assert gate["yaw_histogram"] == [1, 0, 0, 1, 0, 1]
    assert gate["pitch_histogram"] == [2, 0, 1, 0, 0, 0]
    assert (gate["accepted"], gate["rejected_angle"]) == (1, 2)

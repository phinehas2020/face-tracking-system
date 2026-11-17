import numpy as np
import pytest
from ingress.paired_geometry import PairMatcher, ViewObservation, validate_clock_index


def observation(key, point=(0.5, 0.5), frame=1, clock=1, appearance=None):
    return ViewObservation(
        key, point, np.array([1.0, 0.0]) if appearance is None else appearance, frame, clock
    )


def test_pair_requires_fresh_frames_not_repeated_display():
    m = PairMatcher(np.eye(3))
    a, b = observation("front"), observation("side")
    assert not m.update([a], [b])
    assert not m.update([a], [b])
    assert not m.update([a], [observation("side", frame=2)])
    assert (
        m.update([observation("front", frame=2, clock=2)], [observation("side", frame=3, clock=2)])[
            0
        ]["confirmations"]
        == 2
    )


def test_no_association_from_time_alone_or_stale_view():
    m = PairMatcher(np.eye(3), confirmations=1)
    assert not m.update([observation("f")], [observation("s", (0.8, 0.8))])
    assert not m.update([observation("f")], [observation("s", clock=3)])
    assert not m.update([observation("f")], [observation("s", appearance=np.array([0.0, 1.0]))])


def test_ambiguous_crowd_stays_unlinked():
    m = PairMatcher(np.eye(3), confirmations=1)
    assert not m.update([observation("f")], [observation("s1"), observation("s2", (0.501, 0.5))])


def test_ground_projection_not_raw_image_coordinates():
    h = np.eye(3)
    h[0, 2] = 0.2
    m = PairMatcher(h, confirmations=1)
    assert m.update([observation("f", (0.3, 0.5))], [observation("s")])


def test_bad_clock_map_rejected():
    with pytest.raises(ValueError):
        validate_clock_index([{"frame": 1, "clock": 2}, {"frame": 2, "clock": 1}])
    with pytest.raises(ValueError):
        validate_clock_index([{"frame": 1, "clock": 2}, {"frame": 2, "clock": 2}])


def test_sparse_front_frames_do_not_lose_valid_history_or_link_stale_frames():
    m = PairMatcher(np.eye(3))
    a = observation("f", clock=1)
    assert not m.update([a], [observation("s", clock=1)])
    assert not m.update([a], [observation("s", frame=2, clock=2)])
    assert m.update([observation("f", frame=2, clock=2.2)], [observation("s", frame=3, clock=2.2)])


def test_nonfinite_clock_rejected():
    with pytest.raises(ValueError):
        validate_clock_index([{"frame": 0, "clock": 1}, {"frame": 1, "clock": float("nan")}])


def test_sparse_face_motion_needs_strong_distinct_embedding():
    from ingress.vision.adapters import DetectedFace
    from ingress.vision.continuity import FaceContinuity

    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    first = DetectedFace((0, 0, 50, 50), vector, 0.9)
    moved = DetectedFace((80, 0, 130, 50), vector, 0.9)
    c = FaceContinuity(max_gap=3, max_motion_widths=3)
    initial = c.update([first], 0)[0]
    assert c.update([moved], 1.5)[0] == initial
    # Two equally plausible candidates cannot both inherit the old face track.
    c = FaceContinuity(max_gap=3, max_motion_widths=3)
    initial = c.update([first], 0)[0]
    ambiguous = c.update([moved, DetectedFace((81, 0, 131, 50), vector, 0.9)], 1.5)
    assert initial not in ambiguous.values()


def test_body_evidence_can_bridge_sparse_face_track_without_weakening_identity_match():
    from ingress.vision.adapters import DetectedFace
    from ingress.vision.continuity import FaceContinuity

    a = np.zeros(512, dtype=np.float32)
    a[0] = 1
    b = a * 0.65
    b[1] = np.sqrt(1 - 0.65**2)
    c = FaceContinuity(max_gap=3, max_motion_widths=3)
    key = c.update([DetectedFace((0, 0, 50, 50), a, 0.8)], 0, {0: np.array([1.0, 0.0])})[0]
    assert (
        c.update([DetectedFace((80, 0, 130, 50), b, 0.8)], 1.5, {0: np.array([1.0, 0.0])})[0] == key
    )
    c = FaceContinuity(max_gap=3, max_motion_widths=3)
    key = c.update([DetectedFace((0, 0, 50, 50), a, 0.8)], 0, {0: np.array([1.0, 0.0])})[0]
    assert (
        c.update([DetectedFace((80, 0, 130, 50), b, 0.8)], 1.5, {0: np.array([0.0, 1.0])})[0] != key
    )

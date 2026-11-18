import numpy as np
import pytest
from ingress.gate_motion import analyze, compare_passages, strip_pixels
from ingress.live_counting import PassageCalibration


def recording(offsets):
    # The same patterned object moves past each strip at a known delay.
    clock = np.arange(200) / 20
    samples = np.zeros((len(clock), 3, 40, 3), np.uint8)
    pattern = np.random.default_rng(123).integers(45, 190, (18, 40, 3), dtype=np.uint8)
    for i, offset in enumerate(offsets):
        samples[80 + offset:98 + offset, i] = pattern
    return clock, samples


@pytest.mark.parametrize(("offsets", "entry_positive", "expected"), [
    ([0, 2, 4], True, "in"), ([4, 2, 0], True, "out"), ([0, 2, 4], False, "out"),
    ([0, 0, 0], True, None), ([0, 3, 0], True, None),
])
def test_direction_needs_both_delays(offsets, entry_positive, expected):
    windows, _ = analyze(*recording(offsets), entry_positive)
    assert len(windows) == 1
    assert windows[0]["direction"] == expected


def test_partial_motion_does_not_trigger_and_capture_gap_breaks_window():
    clock, samples = recording([0, 2, 4])
    partial = samples.copy()
    partial[:, 0] = 0
    assert not analyze(clock, partial, True)[0]
    clock[92:] += 1
    windows, _ = analyze(clock, samples, True)
    assert len(windows) == 2
    assert all(w["direction"] is None and w["reason"] == "capture boundary" for w in windows)


def test_lingering_activity_abstains():
    clock = np.arange(600) / 20
    samples = np.zeros((600, 3, 10, 3), np.uint8)
    samples[100:250] = 70
    windows, _ = analyze(clock, samples, True)
    assert [w["reason"] for w in windows] == ["sustained activity"]


def test_projection_respects_gate_orientation_and_declines_unusable_border():
    gate = PassageCalibration(axis="horizontal", segment_start=.2, segment_end=.8)
    y, x = strip_pixels(gate, 101, 101)
    assert y[:, 0].tolist() == [46, 50, 54]
    assert (x[1, 0], x[1, -1]) == (80, 20)
    with pytest.raises(ValueError, match="boundary"):
        strip_pixels(PassageCalibration(position=.01), 101, 101)


def test_comparison_keeps_multiple_conflicts_and_outside_events():
    windows = [{"start_seconds": i, "end_seconds": i + 1, "direction": "in"}
               for i in (1, 3, 5, 7)]
    events = [{"timestamp": t, "direction": d} for t, d in
              [(9, "out"), (1, "in"), (3.2, "in"), (4, "in"), (5, "out"), (0, "in")]]
    audit, outside = compare_passages(windows, events)
    assert [r["comparison"] for r in audit] == [
        "agreement", "multiple tracker passages", "opposed", "motion only"]
    assert [e["timestamp"] for e in outside] == [0, 9]

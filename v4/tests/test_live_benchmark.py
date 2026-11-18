import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ingress.live_counting import PassageCalibration
from ingress.vision.contracts import Track

SPEC = importlib.util.spec_from_file_location(
    "benchmark_live_models", Path(__file__).parents[1] / "tools" / "benchmark_live_models.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def event(timestamp, direction="in"):
    return {"timestamp": timestamp, "direction": direction}


def test_matching_tolerance_and_duplicate_predictions():
    result = benchmark.score_events([event(1.1), event(1.2), event(4)], [event(1), event(5)], .25)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == .5


def test_wrong_direction_never_matches():
    result = benchmark.score_events([event(1, "out")], [event(1, "in")])
    assert result["true_positive"] == 0
    assert result["false_positive"] == result["false_negative"] == 1


def test_matching_is_one_to_one_and_maximum_cardinality_not_nearest_greedy():
    # Choosing nearest first (1.2->1.3) would strand the later prediction.
    result = benchmark.score_events([event(1.2), event(1.4)], [event(1), event(1.3)], .25)
    assert result["true_positive"] == 2
    assert result["precision"] == result["recall"] == result["f1"] == 1
    assert len({match["truth_index"] for match in result["matches"]}) == 2


def test_unordered_events_and_inclusive_tolerance_boundary():
    result = benchmark.score_events([event(2, "out"), event(1.2)],
                                    [event(1), event(2.2, "out")], .2)
    assert result["true_positive"] == 2


def test_missing_truth_is_unavailable_not_zero():
    result = benchmark.score_events([event(1)], None)
    assert not result["truth_available"]
    assert result["precision"] is result["recall"] is result["false_negative"] is None


def test_empty_supplied_truth_and_no_predictions_have_defined_denominators_only():
    no_truth_events = benchmark.score_events([event(1)], [])
    assert no_truth_events["truth_available"]
    assert no_truth_events["precision"] == 0
    assert no_truth_events["recall"] is None
    no_predictions = benchmark.score_events([], [event(1)])
    assert no_predictions["recall"] == 0
    assert no_predictions["precision"] is None


@pytest.mark.parametrize("invalid", [event(-1), event(float("nan")), event(1, "left")])
def test_bad_event_data_is_rejected(invalid):
    with pytest.raises(ValueError):
        benchmark.score_events([invalid], [])


def test_csv_validates_headers_and_preserves_seconds(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("timestamp,direction\n0.25,in\n3.5,out\n")
    assert benchmark.read_truth(path) == [event(.25), event(3.5, "out")]
    path.write_text("time,label\n1,in\n")
    with pytest.raises(ValueError):
        benchmark.read_truth(path)


def test_video_timestamp_preserves_presentation_time_and_reports_fallback():
    assert benchmark.source_timestamp(30, 1250, 1.2, 30) == (1.25, "decoder_pts")
    assert benchmark.source_timestamp(30, 0, .9, 30) == (1, "frame_index_over_fps")
    with pytest.raises(ValueError):
        benchmark.source_timestamp(30, 0, 1.1, 30)
    with pytest.raises(ValueError):
        benchmark.source_timestamp(1, 0, 0, float("nan"))


def test_latency_statistics_are_not_empty_zero_and_interpolate():
    assert benchmark.latency_summary([])["p95_ms"] is None
    result = benchmark.latency_summary([40, 10, 30, 20])
    assert result["p50_ms"] == 25
    assert result["p95_ms"] == pytest.approx(38.5)


def test_cli_rejects_absent_model_before_loading_runtime(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--video", str(video), "--model", str(tmp_path / "missing.pt"),
                        "--calibration", str(tmp_path / "gate.json")])
    assert exc.value.code == 2


@pytest.mark.parametrize("sidecar_times,expected_count", [
    (None, 1),
    ([2, 2.1, 2.2, 2.3, 2.4, 2.5], 1),
    ([2, 2.1, 4.2, 4.3, 4.4, 4.5], 0),
])
def test_candidate_preserves_stride_timing_resets_warmup_and_scopes_truth(
    tmp_path, monkeypatch, sidecar_times, expected_count,
):
    class Capture:
        def __init__(self, _path):
            self.index = 0

        def read(self):
            if self.index >= 6:
                return False, None
            self.index += 1
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        def get(self, key):
            if key == 2:
                return 6
            return 10 if key == 0 else (self.index - 1) * 100

        def release(self):
            pass

    resets = []

    class Provider:
        selected_device = "mps"
        fallback_reason = None

        def __init__(self, *_args, **_kwargs):
            pass

        def reset(self):
            resets.append(True)

        def track(self, frame):
            x = 20 if frame.sequence == 0 else 80
            return [Track("person", (x - 5, 40, x + 5, 60), .9, 1)]

    synchronized = []
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_FPS=0, CAP_PROP_POS_MSEC=1, CAP_PROP_FRAME_COUNT=2,
    ))
    monkeypatch.setattr("ingress.vision.adapters.UltralyticsPersonTracker", Provider)
    monkeypatch.setattr(benchmark, "_validate_tracker", lambda _name: None)
    monkeypatch.setattr(benchmark, "_synchronize", synchronized.append)
    video, model = tmp_path / "video.mp4", tmp_path / "model.pt"
    video.write_bytes(b"fixture")
    model.write_bytes(b"fixture")
    sidecar = None
    if sidecar_times is not None:
        sidecar = tmp_path / "side.jsonl"
        sidecar.write_text("\n".join(json.dumps({
            "frame": i, "elapsed_seconds": t, "captured_at": 100 + t,
        }) for i, t in enumerate(sidecar_times)))
    expected_end = sidecar_times[4] if sidecar_times else .4
    result = benchmark.benchmark_candidate(
        video, model, "bytetrack", "mps", 640, PassageCalibration(), stride=2,
        max_seconds=sidecar_times[-1] if sidecar_times else .5,
        truth=[event(expected_end), event(10, "out")], warmup=2, timestamps=sidecar,
    )
    assert result["processed_frames"] == 3
    assert result["decoded_frames"] == 5
    assert result["source_end_seconds"] == expected_end
    if expected_count:
        assert result["events"][0]["timestamp"] == expected_end
    else:
        assert result["events"] == []  # Real missing time expires the previous trajectory.
    assert result["counts"] == {"in": expected_count, "out": 0}
    assert result["scoring"]["recall"] == expected_count
    assert result["timestamp_modes"] == ["capture_sidecar" if sidecar else "decoder_pts"]
    assert result["video_fps_is_nominal_only"] is bool(sidecar)
    assert result["truth_outside_evaluated_window"] == 1
    assert resets == [True]
    assert synchronized == ["mps"] * 7


@pytest.mark.parametrize("rows", [
    [{"frame": 0, "elapsed_seconds": 1}, {"frame": 2, "elapsed_seconds": 2}],
    [{"frame": 0, "elapsed_seconds": 2}, {"frame": 1, "elapsed_seconds": 1}],
    [{"frame": 0, "elapsed_seconds": 1}, {"frame": 1, "elapsed_seconds": 1}],
    [{"frame": 0}],
    [{"frame": True, "elapsed_seconds": 1}],
    [{"frame": 0, "elapsed_seconds": float("nan")}],
    [{"frame": 0, "captured_at": 101, "elapsed_seconds": 1},
     {"frame": 1, "captured_at": 102, "elapsed_seconds": 3}],
    [{"frame": 0, "captured_at": 101}],
])
def test_sidecar_rejects_missing_rows_bad_clocks_and_unknown_origin(tmp_path, rows):
    path = tmp_path / "side.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        benchmark.read_timestamps(path)


def test_captured_only_sidecar_uses_shared_capture_origin_not_first_frame(tmp_path):
    path = tmp_path / "facing.jsonl"
    path.write_text('{"frame_index":0,"captured_at":102.5}\n'
                    '{"frame_index":1,"captured_at":103.1}\n')
    (tmp_path / "capture.json").write_text('{"started_monotonic":100}')
    clock = benchmark.read_timestamps(path)
    assert clock["timestamps"] == pytest.approx([2.5, 3.1])
    assert clock["clock_modes"] == ["captured_at_minus_shared_origin"]


@pytest.mark.parametrize("frame_count", [2, 4, 0, float("nan"), 3.5])
def test_sidecar_coverage_requires_one_row_per_encoded_frame(frame_count):
    with pytest.raises(ValueError):
        benchmark.validate_timestamp_coverage([1, 2, 3], frame_count)
    benchmark.validate_timestamp_coverage([1, 2, 3], 3)


def test_motion_bursts_use_capture_time_and_ignore_tracker_labels(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"sample")
    clock = tmp_path / "side.jsonl"
    clock.write_text("\n".join(json.dumps({"frame": i, "elapsed_seconds": i / 10 + 2})
                               for i in range(61)))
    gate = PassageCalibration()
    audit = tmp_path / "audit.json"
    report = {"video_sha256": benchmark._sha256(video),
              "timestamps_sha256": benchmark._sha256(clock), "gate_endpoints": gate.endpoints,
              "entry_positive": gate.entry_positive, "source_interval": [2, 8],
              "windows": [{"start_seconds": 4, "end_seconds": 5, "direction": None}]}
    audit.write_text(json.dumps(report))
    frames, stop = benchmark.motion_burst_frames(audit, video, clock, gate, 10)
    assert frames == set(range(10, 41)) and stop == 61
    report["source_interval"] = [2, 6]
    audit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cover the evaluation"):
        benchmark.motion_burst_frames(audit, video, clock, gate, 10)
    video.write_bytes(b"other")
    with pytest.raises(ValueError, match="same recording"):
        benchmark.motion_burst_frames(audit, video, clock, gate, 10)


def test_uniform_control_uses_exact_same_budget_and_endpoints():
    adaptive = benchmark.schedule_frames({5, 6, 7, 8, 90}, 15, 4, False)
    control = benchmark.schedule_frames({5, 6, 7, 8, 90}, 15, 4, True)
    assert adaptive == {0, 4, 5, 6, 7, 8, 12}
    assert control == {0, 2, 4, 6, 8, 10, 12}

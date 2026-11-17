#!/usr/bin/env python3
"""Offline single-camera detector/tracker/passage comparison on local video.

Example (all model files must already exist):
    python tools/benchmark_live_models.py --video walk.mp4 --model yolo26n.pt \
        --tracker bytetrack --tracker tracktrack --device cpu --device mps \
        --imgsz 640 --calibration gate.json --truth passages.csv --output result.json

Truth CSV columns: timestamp,direction (seconds from video start; in or out).
For capture_validation recordings pass --timestamps side.jsonl and label truth
in shared capture elapsed seconds, including leading camera startup time.
This evaluates source-video passages only, never cross-camera identity accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from ingress.live_counting import PassageCalibration, PassageCounter, anchor_for_bbox  # noqa: E402


def _validated_events(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        timestamp = float(event["timestamp"])
        if not math.isfinite(timestamp) or timestamp < 0 or event["direction"] not in {"in", "out"}:
            raise ValueError("Events require finite nonnegative seconds and in/out direction")
        result.append({**event, "timestamp": timestamp})
    return result


def score_events(
    predicted: list[dict], truth: list[dict] | None, tolerance: float = 0.5,
) -> dict:
    """Maximum-cardinality, chronological matching per direction within tolerance.

    Every prediction and truth event is used at most once. This optimizes event
    count, not minimal timestamp error. A duplicate is a false positive; wrong
    direction is both a missed truth and an unmatched prediction.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    predicted = _validated_events(predicted)
    if truth is None:
        return {"truth_available": False, "predicted": len(predicted), "truth": None,
                "true_positive": None, "false_positive": None, "false_negative": None,
                "precision": None, "recall": None, "f1": None, "matches": [],
                "tolerance_seconds": tolerance}
    truth = _validated_events(truth)
    matches = []
    for direction in ("in", "out"):
        predictions = sorted(
            ((i, e) for i, e in enumerate(predicted) if e["direction"] == direction),
            key=lambda item: item[1]["timestamp"],
        )
        labels = sorted(((i, e) for i, e in enumerate(truth)
                         if e["direction"] == direction), key=lambda item: item[1]["timestamp"])
        p = t = 0
        while p < len(predictions) and t < len(labels):
            pi, prediction = predictions[p]
            ti, label = labels[t]
            delta = prediction["timestamp"] - label["timestamp"]
            if delta < -tolerance - 1e-9:
                p += 1
            elif delta > tolerance + 1e-9:
                t += 1
            else:
                matches.append({"prediction_index": pi, "truth_index": ti,
                                "direction": direction, "error_seconds": delta})
                p += 1
                t += 1
    tp = len(matches)
    precision = tp / len(predicted) if predicted else None
    recall = tp / len(truth) if truth else None
    f1 = (2 * tp / (len(predicted) + len(truth))) if predicted or truth else None
    return {"truth_available": True, "predicted": len(predicted), "truth": len(truth),
            "true_positive": tp, "false_positive": len(predicted) - tp,
            "false_negative": len(truth) - tp, "precision": precision, "recall": recall,
            "f1": f1, "matches": matches, "tolerance_seconds": tolerance}


def read_truth(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp", "direction"}.issubset(reader.fieldnames or []):
            raise ValueError("Truth CSV must have timestamp,direction columns")
        return _validated_events(list(reader))


def read_timestamps(path: Path) -> dict:
    """Read one authoritative source timestamp for every encoded frame.

    Never rebase to a view's first frame: elapsed_seconds already shares the
    capture run's origin. Captured-at-only sidecars need capture.json's shared
    started_monotonic; guessing a per-camera origin would destroy synchronization.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Timestamp sidecar is empty")
    origin = None
    metadata = path.parent / "capture.json"
    if metadata.is_file():
        origin = float(json.loads(metadata.read_text())["started_monotonic"])
        if not math.isfinite(origin) or origin < 0:
            raise ValueError("Capture clock origin must be finite and nonnegative")
    timestamps = []
    previous_capture = None
    modes = set()
    for index, row in enumerate(rows):
        frame_index = row.get("frame", row.get("frame_index"))
        if (type(frame_index) is not int or frame_index != index
                or ("frame_index" in row and row["frame_index"] != index)):
            raise ValueError("Timestamp rows must cover every frame contiguously from index 0")
        captured_at = row.get("captured_at")
        if captured_at is not None:
            captured_at = float(captured_at)
            if (not math.isfinite(captured_at) or captured_at < 0
                    or (previous_capture is not None and captured_at <= previous_capture)):
                raise ValueError("Capture timestamps must be finite and strictly increasing")
            previous_capture = captured_at
        elapsed = row.get("elapsed_seconds", row.get("elapsed"))
        if elapsed is not None:
            timestamp = float(elapsed)
            modes.add("shared_elapsed_seconds")
            if captured_at is not None:
                inferred_origin = captured_at - timestamp
                if origin is None:
                    origin = inferred_origin
                if abs(inferred_origin - origin) > 1e-5:
                    raise ValueError("Capture and elapsed clocks disagree on their shared origin")
        elif captured_at is not None and origin is not None:
            timestamp = captured_at - origin
            modes.add("captured_at_minus_shared_origin")
        else:
            raise ValueError("Frame needs elapsed_seconds or captured_at with shared origin")
        if (not math.isfinite(timestamp) or timestamp < 0
                or (timestamps and timestamp <= timestamps[-1])):
            raise ValueError("Elapsed times must be finite, nonnegative and strictly increasing")
        timestamps.append(timestamp)
    return {"timestamps": timestamps, "clock_modes": sorted(modes),
            "shared_started_monotonic": origin}


def validate_timestamp_coverage(timestamps: list[float], declared_frame_count: float) -> None:
    if (not math.isfinite(declared_frame_count) or declared_frame_count <= 0
            or declared_frame_count != int(declared_frame_count)):
        raise ValueError("Sidecar evaluation requires an exact declared video frame count")
    if len(timestamps) != int(declared_frame_count):
        raise ValueError("Timestamp sidecar row count does not match encoded video frame count")


def latency_summary(milliseconds: list[float]) -> dict:
    """Linear-interpolated percentiles; no unavailable-as-zero measurements."""
    ordered = sorted(milliseconds)
    if not ordered:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {"samples": len(ordered), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "mean_ms": sum(ordered) / len(ordered)}


def source_timestamp(index: int, reported_msec: float, previous: float | None, fps: float):
    """Prefer decoder presentation time; explicitly report constant-FPS fallback."""
    timestamp = reported_msec / 1000
    if math.isfinite(timestamp) and timestamp >= 0 and (previous is None or timestamp > previous):
        return timestamp, "decoder_pts"
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video has neither increasing timestamps nor valid FPS")
    timestamp = index / fps
    if previous is not None and timestamp <= previous:
        raise ValueError("Video timestamps became non-monotonic; refusing invented passage timing")
    return timestamp, "frame_index_over_fps"


def _synchronize(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _validate_tracker(name: str) -> None:
    # Built-in configs only; appearance encoders can otherwise silently download.
    import ultralytics
    import yaml

    config = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
    if not config.is_file():
        raise ValueError(f"Installed Ultralytics does not include {name}.yaml")
    value = yaml.safe_load(config.read_text())
    if value.get("with_reid", False):
        raise ValueError("This offline comparison requires built-in ReID to be disabled")


def benchmark_candidate(
    video: Path, model: Path, tracker_name: str, device: str, imgsz: int,
    calibration: PassageCalibration, *, stride: int = 1, max_seconds: float = 60,
    truth: list[dict] | None = None, tolerance: float = .5,
    roi: tuple[float, float, float, float] | None = None, confidence: float = .1,
    warmup: int = 3, timestamps: Path | None = None,
) -> dict:
    if not model.is_file() or not video.is_file():
        raise ValueError("Model and video must already exist as local files; downloads disabled")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    clock = read_timestamps(timestamps) if timestamps is not None else None
    import cv2
    from ingress.vision.adapters import UltralyticsPersonTracker
    from ingress.vision.contracts import FramePacket

    _validate_tracker(tracker_name)
    provider = UltralyticsPersonTracker(
        str(model), confidence=confidence, tracker=tracker_name,
        device=device, imgsz=imgsz, roi=roi,
    )
    capture = cv2.VideoCapture(str(video))
    try:
        if clock is not None:
            validate_timestamp_coverage(clock["timestamps"], capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, image = capture.read()
        if not ok:
            raise ValueError("Video cannot be decoded")
        for index in range(warmup):
            provider.track(FramePacket("benchmark", index, index * 100_000_000, "", image))
        _synchronize(provider.selected_device)
        provider.reset()
    finally:
        capture.release()

    capture = cv2.VideoCapture(str(video))
    counter = PassageCounter(calibration)
    events: list[dict] = []
    timing, decode_timing, pipeline_timing = [], [], []
    ids: set[str] = set()
    max_visible = invalid_boxes = track_observations = backend_resets = 0
    timestamp_modes: set[str] = set()
    index = 0
    previous = None
    first_processed = last_processed = None
    last_device = provider.selected_device
    started = time.perf_counter()
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            decoded_at = time.perf_counter()
            ok, image = capture.read()
            decode_ms = (time.perf_counter() - decoded_at) * 1000
            if not ok:
                if clock is not None and index != len(clock["timestamps"]):
                    raise ValueError("Video decoding ended before all timestamp rows were consumed")
                break
            if clock is not None:
                if index >= len(clock["timestamps"]):
                    raise ValueError("Decoded video frame is missing its timestamp sidecar row")
                timestamp, mode = clock["timestamps"][index], "capture_sidecar"
            else:
                timestamp, mode = source_timestamp(
                    index, float(capture.get(cv2.CAP_PROP_POS_MSEC)), previous, fps,
                )
            previous = timestamp
            if timestamp >= max_seconds:
                break
            timestamp_modes.add(mode)
            process = index % stride == 0
            sequence = index
            index += 1
            decode_timing.append(decode_ms)
            if not process:
                continue
            first_processed = timestamp if first_processed is None else first_processed
            last_processed = timestamp
            packet = FramePacket("benchmark", sequence, int(timestamp * 1_000_000_000), "", image)
            _synchronize(provider.selected_device)
            infer_at = time.perf_counter()
            tracks = provider.track(packet)
            _synchronize(provider.selected_device)
            timing.append((time.perf_counter() - infer_at) * 1000)
            if provider.selected_device != last_device:
                # The adapter resets tracker identities when falling back.
                # Never bridge its reused IDs into pre-fallback trajectories.
                counter.reset()
                backend_resets += 1
                last_device = provider.selected_device
            height, width = image.shape[:2]
            counter.expire(timestamp)
            max_visible = max(max_visible, len(tracks))
            track_observations += len(tracks)
            for track in tracks:
                ids.add(f"{backend_resets}:{track.track_id}")
                try:
                    point = anchor_for_bbox(track.bbox, width, height, calibration.anchor)
                except ValueError:
                    invalid_boxes += 1
                    counter.states.pop(track.track_id, None)
                    continue
                direction = counter.update(track.track_id, point, timestamp)
                if direction:
                    events.append({"timestamp": timestamp, "direction": direction,
                                   "track_id": track.track_id, "backend_epoch": backend_resets})
            pipeline_timing.append((time.perf_counter() - decoded_at) * 1000)
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not timing:
        raise ValueError("No video observations fell within the evaluation window")
    evaluated_truth = None if truth is None else [
        event for event in truth
        if first_processed is not None and first_processed <= event["timestamp"] <= last_processed
    ]
    return {
        "status": "completed", "model": str(model), "tracker": tracker_name,
        "requested_device": device, "selected_device": provider.selected_device,
        "fallback_reason": provider.fallback_reason, "backend_resets": backend_resets,
        "imgsz": imgsz, "roi": roi, "confidence": confidence, "stride": stride,
        "warmup_frames_excluded": warmup, "source_fps": fps if math.isfinite(fps) else None,
        "timestamp_modes": sorted(timestamp_modes), "decoded_frames": index,
        "timestamp_sidecar": str(timestamps.resolve()) if timestamps else None,
        "sidecar_clock_modes": clock["clock_modes"] if clock else None,
        "shared_started_monotonic": clock["shared_started_monotonic"] if clock else None,
        "video_fps_is_nominal_only": clock is not None,
        "processed_frames": len(timing), "source_start_seconds": first_processed,
        "source_end_seconds": last_processed, "wall_seconds_excluding_warmup": elapsed,
        "processed_frames_per_wall_second": len(timing) / elapsed if elapsed else None,
        "inference_and_tracking": latency_summary(timing),
        "decode_all_frames": latency_summary(decode_timing),
        "processed_frame_pipeline": latency_summary(pipeline_timing),
        "unique_track_ids": len(ids), "max_simultaneous_tracks": max_visible,
        "track_observations": track_observations, "invalid_bbox_observations": invalid_boxes,
        "events": events, "counts": {d: sum(e["direction"] == d for e in events)
                                       for d in ("in", "out")},
        "scoring": score_events(events, evaluated_truth, tolerance),
        "truth_outside_evaluated_window": (None if truth is None
                                          else len(truth) - len(evaluated_truth)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--tracker", choices=("bytetrack", "tracktrack"), action="append")
    parser.add_argument("--device", choices=("cpu", "mps"), action="append")
    parser.add_argument("--imgsz", type=int, choices=(640, 960), action="append")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--timestamps", type=Path,
                        help="Authoritative frame JSONL from capture_validation.py")
    parser.add_argument("--tolerance", type=float, default=.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument("--confidence", type=float, default=.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    required_paths = [args.video, *args.model, args.calibration]
    required_paths.extend(path for path in (args.truth, args.timestamps) if path is not None)
    for path in required_paths:
        if not path.is_file():
            parser.error(f"Local file does not exist (downloads are disabled): {path}")
    if args.stride < 1 or not 0 < args.max_seconds <= 3600 or not 0 <= args.warmup <= 100:
        parser.error("stride must be >=1; max-seconds in (0,3600]; warmup in [0,100]")
    if not 0 <= args.confidence <= 1 or not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("confidence must be in [0,1] and tolerance finite/nonnegative")
    if args.roi and (not all(math.isfinite(v) and 0 <= v <= 1 for v in args.roi)
                     or args.roi[0] >= args.roi[2] or args.roi[1] >= args.roi[3]):
        parser.error("roi must be normalized left top right bottom with positive area")
    values = json.loads(args.calibration.read_text())
    if values.get("counting_mode", "trajectory") != "trajectory":
        parser.error("Offline live-model comparison requires trajectory counting calibration")
    calibration = PassageCalibration.from_calibration(values)
    truth = read_truth(args.truth) if args.truth else None
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    results = []
    for model, tracker, device, imgsz in itertools.product(
        args.model, args.tracker or ["bytetrack"], args.device or ["cpu"], args.imgsz or [640],
    ):
        print(f"Benchmarking {model.name} / {tracker} / {device} / {imgsz}", file=sys.stderr)
        try:
            with redirect_stdout(sys.stderr):
                result = benchmark_candidate(
                    args.video.resolve(), model.resolve(), tracker, device, imgsz, calibration,
                    stride=args.stride, max_seconds=args.max_seconds, truth=truth,
                    tolerance=args.tolerance, roi=tuple(args.roi) if args.roi else None,
                    confidence=args.confidence, warmup=args.warmup, timestamps=args.timestamps,
                )
            result["model_sha256"] = _sha256(model)
        except Exception as exc:
            result = {"status": "failed", "model": str(model), "tracker": tracker,
                      "requested_device": device, "imgsz": imgsz,
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
    report = {"schema_version": 1, "video": str(args.video.resolve()),
              "video_sha256": _sha256(args.video), "calibration": values,
              "timestamps": str(args.timestamps.resolve()) if args.timestamps else None,
              "timestamps_sha256": _sha256(args.timestamps) if args.timestamps else None,
              "scope": "Single source-video passage evaluation; no cross-camera identity claims",
              "timing_note": "Warmup excluded. MPS synchronized. Offline throughput is not live "
                             "capture latency; confirmations use source-video time.",
              "truth": str(args.truth.resolve()) if args.truth else None, "results": results}
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
        print(str(args.output.resolve()))
    else:
        print(serialized)
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

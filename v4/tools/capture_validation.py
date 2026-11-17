#!/usr/bin/env python3
"""Record a bounded, local two-camera validation run with source-clock sidecars.

Run with the same Python environment as ingress:
    python tools/capture_validation.py --config /private/cameras.json \
        --output /private/walkthrough-001 --seconds 60

Every decoded input frame is written once at native resolution. MP4 playback uses
nominal --fps (default 15); it is NOT the source clock. Use each JSONL row's
captured_at / elapsed_seconds for synchronization, gaps, or passage scoring.
Images are JPEG-decoded worker frames, not lossless camera bitstreams.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HEADER = struct.Struct("!Id")
MAX_BUFFER = 16 * 1024 * 1024
BACKEND = Path(__file__).resolve().parents[1] / "backend"


class CaptureError(Exception):
    """Safe fixed diagnostic code; never copy decoder errors or credentials."""


def _private_json(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _camera_configs(path: Path) -> dict[str, dict]:
    """Read only decoder fields; do not persist or display the configuration."""
    try:
        source = json.loads(path.read_text())
        facing = source["facing_camera"]
        cameras = {}
        for role, config in (("side", source), ("facing", facing)):
            if not isinstance(config, dict) or not config.get("host"):
                raise ValueError
            cameras[role] = {
                key: config[key]
                for key in ("host", "username", "password", "channel")
                if key in config
            }
        return cameras
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CaptureError("invalid_private_camera_config") from exc


def _packets(buffer: bytearray):
    """Yield complete bounded JPEG packets, retaining an incomplete final one."""
    if len(buffer) > MAX_BUFFER:
        raise CaptureError("capture_buffer_limit")
    while len(buffer) >= HEADER.size:
        length, captured_at = HEADER.unpack_from(buffer)
        if not 0 < length <= MAX_BUFFER - HEADER.size:
            raise CaptureError("invalid_packet_length")
        if not math.isfinite(captured_at) or captured_at < 0:
            raise CaptureError("invalid_capture_timestamp")
        total = HEADER.size + length
        if len(buffer) < total:
            return
        payload = bytes(buffer[HEADER.size : total])
        del buffer[:total]
        yield captured_at, payload


def _terminate(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    for pipe in (process.stdin, process.stdout):
        if pipe is not None:
            pipe.close()


def record(config_path: Path, output: Path, seconds: float = 60, fps: float = 15) -> dict:
    """Record both decoder streams; return only metadata safe for display."""
    if not math.isfinite(seconds) or not 1 <= seconds <= 120:
        raise CaptureError("seconds_must_be_between_1_and_120")
    if not math.isfinite(fps) or not 1 <= fps <= 60:
        raise CaptureError("fps_must_be_between_1_and_60")
    cameras = _camera_configs(config_path)
    import cv2
    import numpy as np

    # Refuse existing output rather than overwriting diagnostic evidence.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output, 0o700)
    old_umask = os.umask(0o077)
    selector = selectors.DefaultSelector()
    states = {}
    started = time.monotonic()
    deadline = started + seconds
    start_utc = datetime.now(UTC).isoformat()
    interrupted = False
    overall_error = None
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
        for role, config in cameras.items():
            sidecar = (output / f"{role}.jsonl").open("x", encoding="utf-8")
            state = states[role] = {
                "process": None,
                "sidecar": sidecar,
                "writer": None,
                "buffer": bytearray(),
                "frame_count": 0,
                "size": None,
                "first_capture": None,
                "last_capture": None,
                "max_gap_seconds": 0.0,
                "gap_count": 0,
                "error": None,
                "worker_exit_code": None,
            }
            process = state["process"] = subprocess.Popen(
                [sys.executable, "-m", "ingress.capture_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            # Never place the RTSP URL/credentials in argv, stdout, or metadata.
            process.stdin.write((json.dumps(config) + "\n").encode())
            process.stdin.close()
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ, role)
        print(
            json.dumps({"status": "recording", "seconds": seconds, "views": list(cameras)}),
            flush=True,
        )
        while time.monotonic() < deadline and selector.get_map():
            for key, _ in selector.select(timeout=min(0.25, max(0, deadline - time.monotonic()))):
                role = key.data
                state = states[role]
                try:
                    chunk = os.read(key.fd, min(65536, MAX_BUFFER - len(state["buffer"])))
                    if not chunk:
                        if state["buffer"]:
                            state["error"] = "truncated_packet"
                        else:
                            state["error"] = "decoder_ended_before_deadline"
                        state["worker_exit_code"] = state["process"].poll()
                        selector.unregister(key.fileobj)
                        continue
                    state["buffer"].extend(chunk)
                    for captured_at, payload in _packets(state["buffer"]):
                        if captured_at > deadline:
                            continue
                        if captured_at < started or captured_at > time.monotonic() + 1:
                            raise CaptureError("capture_clock_out_of_range")
                        previous = state["last_capture"]
                        if previous is not None and captured_at < previous:
                            raise CaptureError("capture_clock_went_backwards")
                        image = cv2.imdecode(
                            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if image is None:
                            raise CaptureError("invalid_jpeg_frame")
                        height, width = image.shape[:2]
                        size = (width, height)
                        if state["writer"] is None:
                            writer = state["writer"] = cv2.VideoWriter(
                                str(output / f"{role}.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                size,
                            )
                            if not writer.isOpened():
                                raise CaptureError("video_writer_unavailable")
                            state["size"] = size
                            state["first_capture"] = captured_at
                        elif size != state["size"]:
                            raise CaptureError("source_resolution_changed")
                        gap = captured_at - previous if previous is not None else None
                        gap_detected = gap is not None and gap > max(0.25, 3 / fps)
                        if gap is not None:
                            state["max_gap_seconds"] = max(state["max_gap_seconds"], gap)
                        state["gap_count"] += int(gap_detected)
                        state["writer"].write(image)
                        row = {
                            "frame": state["frame_count"],
                            "captured_at": captured_at,
                            "elapsed_seconds": captured_at - started,
                            "width": width,
                            "height": height,
                            "gap_before_seconds": gap,
                            "gap_detected": gap_detected,
                        }
                        state["sidecar"].write(json.dumps(row, allow_nan=False) + "\n")
                        state["sidecar"].flush()
                        state["last_capture"] = captured_at
                        state["frame_count"] += 1
                except (CaptureError, OSError) as exc:
                    state["error"] = (
                        str(exc) if isinstance(exc, CaptureError) else "capture_io_error"
                    )
                    selector.unregister(key.fileobj)
                    _terminate(state["process"])
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        # No exception message is exposed: libraries may include a source URL.
        overall_error = f"capture_failed_{type(exc).__name__}"
    finally:
        ended = time.monotonic()
        os.umask(old_umask)
        selector.close()
        for role, state in states.items():
            try:
                if state["process"] is not None:
                    _terminate(state["process"])
            except (OSError, subprocess.TimeoutExpired):
                state["error"] = "worker_shutdown_failed"
            try:
                if state["writer"] is not None:
                    state["writer"].release()
                    check = cv2.VideoCapture(str(output / f"{role}.mp4"))
                    try:
                        if int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != state["frame_count"]:
                            state["error"] = "video_frame_count_mismatch"
                    finally:
                        check.release()
            except Exception:
                state["error"] = "video_finalize_failed"
            finally:
                state["sidecar"].close()
        for path in output.iterdir():
            if path.is_file():
                os.chmod(path, 0o600)
        metadata = {
            "schema_version": 1,
            "started_utc": start_utc,
            "started_monotonic": started,
            "requested_seconds": seconds,
            "elapsed_seconds": ended - started,
            "nominal_video_fps": fps,
            "interrupted": interrupted,
            "error": overall_error,
            "clock": "JSONL captured_at is worker capture monotonic time on this host",
            "video_timing": "Nominal playback only; no duplicated frames or hidden gap filling. "
            "Use JSONL elapsed_seconds for synchronization and passage scoring.",
            "views": {},
        }
        for role in cameras:
            state = states.get(role)
            if state is None:
                metadata["views"][role] = {"status": "unavailable", "frame_count": 0}
                continue
            first, last = state["first_capture"], state["last_capture"]
            metadata["views"][role] = {
                "status": "unavailable"
                if not state["frame_count"]
                else "partial"
                if state["error"] or interrupted or overall_error
                else "recorded",
                "frame_count": state["frame_count"],
                "native_size": state["size"],
                "first_capture": first,
                "last_capture": last,
                "captured_duration_seconds": None if first is None else last - first,
                "leading_unavailable_seconds": ended - started
                if first is None
                else first - started,
                "trailing_unavailable_seconds": None if last is None else max(0, ended - last),
                "max_gap_seconds": state["max_gap_seconds"],
                "gap_count": state["gap_count"],
                "error": state["error"],
                "worker_exit_code": state["worker_exit_code"],
                "video": f"{role}.mp4" if state["frame_count"] else None,
                "timestamps": f"{role}.jsonl",
            }
        try:
            _private_json(output / "capture.json", metadata)
        finally:
            os.umask(old_umask)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private two-camera JSON")
    parser.add_argument("--output", type=Path, required=True, help="New private output directory")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument(
        "--fps", type=float, default=15, help="Nominal MP4 FPS; source clock is JSONL"
    )
    args = parser.parse_args()
    try:
        metadata = record(args.config, args.output, args.seconds, args.fps)
    except Exception as exc:
        code = str(exc) if isinstance(exc, CaptureError) else f"capture_failed_{type(exc).__name__}"
        print(json.dumps({"status": "failed", "error": code}), file=sys.stderr)
        return 1
    complete = (
        not metadata["interrupted"]
        and not metadata["error"]
        and all(view["status"] == "recorded" for view in metadata["views"].values())
    )
    print(
        json.dumps(
            {
                "status": "complete" if complete else "partial",
                "output": str(args.output),
                "views": metadata["views"],
            }
        ),
        flush=True,
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

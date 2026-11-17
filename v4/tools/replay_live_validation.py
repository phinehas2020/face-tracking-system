#!/usr/bin/env python3
"""Replay synchronized recordings through the real asynchronous live pipeline.

No camera connections: both capture methods are replaced before workers start.
All database writes go to a new private output directory; the seed is opened
read-only and copied with SQLite's backup API. Real-time frame publication keeps
the original inter-camera gaps and the live pipeline's natural frame dropping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time
from bisect import bisect_left
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from benchmark_live_models import read_timestamps, validate_timestamp_coverage
from ingress.dual_live import DualLiveSession
from ingress.face_registry import recount
from ingress.store import EventStore


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--startup-timeout", type=float, default=60.)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) for v in (args.start, args.duration, args.startup_timeout)):
        parser.error("Times must be finite")
    if args.start < 0 or args.duration <= 0 or not 0 < args.startup_timeout <= 60:
        parser.error("Require start >= 0, duration > 0, and startup timeout in (0,60]")
    args.source, args.config, args.seed_db, args.model, args.output = (
        path.expanduser().resolve() for path in
        (args.source, args.config, args.seed_db, args.model, args.output)
    )
    for path in (args.config, args.seed_db, args.model, args.source / "capture.json"):
        if not path.is_file():
            parser.error(f"Required local file unavailable: {path.name}")
    if args.output.exists():
        parser.error("Output must be a new directory")
    metadata = json.loads((args.source / "capture.json").read_text())
    source_start = datetime.fromisoformat(metadata["started_utc"].replace("Z", "+00:00"))
    if source_start.tzinfo is None:
        raise ValueError("Capture started_utc must include a timezone")
    configuration = json.loads(args.config.read_text())
    if not configuration.get("facing_camera"):
        parser.error("The configuration must contain both camera roles")
    for role, settings in (("side", configuration), ("facing", configuration["facing_camera"])):
        if not settings.get("async_faces", False):
            parser.error(f"{role} must enable async_faces to test the actual live pipeline")
        # The output configuration preserves inference settings without copying
        # credentials; even an accidental fallback cannot address a real camera.
        settings.update(host=f"replay-{role}.invalid", username="", password="")
    timestamp_sets = {}
    for role in ("side", "facing"):
        if not (args.source / f"{role}.mp4").is_file():
            parser.error(f"Missing {role} recording")
        timestamp_sets[role] = read_timestamps(args.source / f"{role}.jsonl")["timestamps"]
        if not any(args.start <= stamp < args.start + args.duration for stamp in timestamp_sets[role]):
            parser.error(f"No {role} frames in requested interval")

    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700)
    config_copy = args.output / "replay-config.json"
    write_json(config_copy, configuration)
    seed_hash = sha256(args.seed_db)
    store = EventStore(args.output / "data")
    store.data_root.mkdir(mode=0o700)
    with sqlite3.connect(args.seed_db.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(store.db_path) as destination:
            source.backup(destination)
    store.initialize()
    with store.transaction() as db:
        db.execute("DELETE FROM passages")
        for row in db.execute("SELECT id FROM events").fetchall():
            recount(db, row[0])
        seeded_profiles = db.execute("SELECT COUNT(*) FROM face_templates").fetchone()[0]

    station = DualLiveSession(store, config_copy, str(args.model))
    sessions = {"side": station, "facing": station.facing}
    play = threading.Event()
    ready = {role: threading.Event() for role in sessions}
    done = {role: threading.Event() for role in sessions}
    origin = [None]
    failures = {}
    playback = {role: {"published": 0, "max_schedule_lateness_ms": 0.} for role in sessions}

    def capture_for(role):
        def capture(session):
            import cv2

            video = None
            try:
                stamps = timestamp_sets[role]
                video = cv2.VideoCapture(str(args.source / f"{role}.mp4"))
                if not video.isOpened():
                    raise RuntimeError("Cannot open recording")
                validate_timestamp_coverage(stamps, video.get(cv2.CAP_PROP_FRAME_COUNT))
                first = bisect_left(stamps, args.start)
                if not video.set(cv2.CAP_PROP_POS_FRAMES, first):
                    raise RuntimeError("Cannot seek recording to requested frame")
                ready[role].set()
                while not play.wait(.05):
                    if session.stop_event.is_set():
                        return
                for index in range(first, len(stamps)):
                    elapsed = stamps[index]
                    if elapsed >= args.start + args.duration or session.stop_event.is_set():
                        break
                    success, frame = video.read()
                    if not success:
                        raise RuntimeError("Recording ended before timestamp sidecar")
                    target = origin[0] + elapsed - args.start
                    if session.stop_event.wait(max(0., target - time.monotonic())):
                        break
                    late_ms = max(0., (time.monotonic() - target) * 1000)
                    with session.lock:
                        session.frame = frame
                        session.frame_time = target
                        session.frame_observed_at = (source_start + timedelta(seconds=elapsed)).isoformat(timespec="milliseconds")
                        session.frame_sequence = index + 1
                        session.generation = 1
                        session.size = [frame.shape[1], frame.shape[0]]
                    stats = playback[role]
                    stats["published"] += 1
                    stats["last_frame"] = index
                    stats["last_elapsed_seconds"] = elapsed
                    stats["max_schedule_lateness_ms"] = max(stats["max_schedule_lateness_ms"], late_ms)
            except Exception as exc:
                failures[role] = f"{type(exc).__name__}: {exc}"
            finally:
                if video is not None:
                    video.release()
                ready[role].set()
                done[role].set()
        return capture

    for role, session in sessions.items():
        session._capture = MethodType(capture_for(role), session)
        session.source_info = {"station": "A", "validation": "real_time_live_replay", "role": role}

    error = None
    snapshots = []
    started = time.monotonic()
    try:
        with (args.output / "inference.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            station.start()
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if failures:
                    raise RuntimeError("Recording initialization failed")
                for role, session in sessions.items():
                    if session.status == "error" or (session.face_worker and session.face_worker.status == "error"):
                        raise RuntimeError(f"{role} model initialization failed")
                if all(ready[role].is_set() and session.face_worker and session.face_worker.status == "ready"
                       for role, session in sessions.items()):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model initialization exceeded startup deadline")
                time.sleep(.05)
            startup_seconds = time.monotonic() - started
            origin[0] = time.monotonic() + .25
            play.set()
            # A short tail allows completion delivery; no duplicate fake camera
            # frames are created. Capture logs expose any undelivered final result.
            end = origin[0] + args.duration
            while time.monotonic() < end + 3.:
                if failures:
                    raise RuntimeError("Recording playback failed")
                snapshots.append({"elapsed_seconds": time.monotonic() - origin[0] + args.start,
                                  "snapshot": station.snapshot()})
                time.sleep(.25)
            final_snapshot = station.snapshot()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        startup_seconds = time.monotonic() - started if origin[0] is None else None
        final_snapshot = station.snapshot()
    finally:
        station.stop()

    with store.connect() as db:
        passages = [dict(row) for row in db.execute(
            "SELECT p.*, f.label FROM passages p LEFT JOIN face_templates f ON f.identity_id=p.identity_id ORDER BY observed_at")]
        for row in passages:
            row["evidence"] = json.loads(row["evidence"])
        final_profiles = db.execute("SELECT COUNT(*) FROM face_templates").fetchone()[0]
    report = {
        "schema_version": 1, "pipeline": "DualLiveSession async live, recorded capture replacement",
        "created_at": datetime.now(UTC).isoformat(), "source": str(args.source),
        "start_seconds": args.start, "duration_seconds": args.duration,
        "startup_seconds": startup_seconds, "wall_seconds": time.monotonic() - started,
        "model": str(args.model), "seed_sha256_before": seed_hash,
        "seed_sha256_after": sha256(args.seed_db), "seed_profiles": seeded_profiles,
        "final_profiles": final_profiles, "playback": playback, "capture_errors": failures,
        "error": error, "snapshot": final_snapshot, "passages": passages,
        "workers_stopped": all(not thread.is_alive() for session in sessions.values() for thread in session.threads),
    }
    write_json(args.output / "report.json", report)
    with (args.output / "snapshots.jsonl").open("w") as stream:
        for snapshot in snapshots:
            stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output / "report.json"), "counts": final_snapshot.get("counts"),
                      "error": error, "seed_unchanged": report["seed_sha256_before"] == report["seed_sha256_after"]}))
    return 1 if error or not report["workers_stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

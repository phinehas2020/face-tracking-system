#!/usr/bin/env python3
"""Continuous recording for Booth One, with HEVC compaction once raw footage passes a cap.

Each camera's main RTSP stream is copied to 10-minute MP4 segments without re-encoding
(no CPU or GPU cost to the counting service). When finished raw segments together pass
--raw-cap-gb, the oldest are re-encoded to HEVC on the Mac's media engine
(hevc_videotoolbox), checked with ffprobe, and only then is the raw segment removed.
Recording pauses, and nothing is deleted, when free disk space falls below --min-free-gb.
Camera credentials are read from cameras.json and never logged.
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
stop = threading.Event()


def log(*parts):
    print(datetime.datetime.now().isoformat(timespec="seconds"), *parts, flush=True)


def cameras(path: Path):
    cfg = json.loads(path.read_text())
    out = []
    for c in (cfg, cfg.get("facing_camera") or {}):
        if not c.get("host"):
            continue
        name = (c.get("model") or c["host"]).split("(")[-1].rstrip(")").strip() or c["host"]
        rtsp = (c.get("rtsp_path") or "/h264Preview_01_main").lstrip("/")
        user = quote(c.get("username", ""), safe="")
        secret = quote(c.get("password", ""), safe="")
        url = f"rtsp://{user}:{secret}@{c['host']}:554/{rtsp}"
        out.append((name, url))
    return out


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def record(name: str, url: str, root: Path, segment: int, min_free: float, resume_free: float):
    folder = root / "raw" / name
    folder.mkdir(parents=True, exist_ok=True)
    backoff = 5
    while not stop.is_set():
        if free_gb(root) < min_free:
            log(name, f"paused: {free_gb(root):.1f} GB free is under {min_free} GB")
            while not stop.is_set() and free_gb(root) < resume_free:
                stop.wait(60)
            continue
        command = [
            FFMPEG,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            "15000000",
            "-i",
            url,
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(segment),
            "-segment_atclocktime",
            "1",
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            "-segment_format",
            "mp4",
            "-segment_format_options",
            "movflags=+frag_keyframe+empty_moov+default_base_moof",
            str(folder / f"{name}-%Y%m%d-%H%M%S.mp4"),
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        log(name, "recording")
        while process.poll() is None and not stop.is_set():
            if free_gb(root) < min_free:
                break
            stop.wait(10)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)  # lets ffmpeg close the open segment cleanly
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        error = (process.stderr.read() or "").strip().replace(url, "<camera>")[-400:]
        if stop.is_set():
            return
        ran = time.monotonic() - started
        backoff = 5 if ran > 120 else min(backoff * 2, 120)
        log(
            name,
            f"stream ended after {ran:.0f}s (exit {process.returncode}); retry in {backoff}s",
            error,
        )
        stop.wait(backoff)


def duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


PROBE = (
    "import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); "
    "t = p / '.write-test'; t.write_bytes(b'ok'); t.unlink()"
)


def writable(folder: Path, timeout: float = 15.0) -> str | None:
    """None when FOLDER can be created and written, else the reason.

    A macOS privacy prompt for a removable volume blocks file calls until someone answers
    it on the Mac's screen, so the probe runs in a child process with a timeout.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", PROBE, str(folder)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "blocked (a macOS permission prompt may be waiting on the booth's screen)"
    if done.returncode:
        lines = done.stderr.strip().splitlines() or ["unknown error"]
        return f"not writable ({lines[-1][:160]})"
    return None


def archive_root(root: Path, external: Path | None, reported: dict) -> Path:
    """The external drive when it is mounted, writable and has room; else the internal disk."""
    internal = root / "archive"
    if external is None:
        return internal
    now = time.monotonic()
    if reported.get("state") not in (None, "external") and now - reported.get("checked", 0) < 300:
        return internal  # re-probe a failing drive every five minutes, not every pass
    reported["checked"] = now
    mount = Path(*external.parts[:3]) if str(external).startswith("/Volumes/") else None
    if mount is None or not os.path.ismount(mount):
        reason = "not mounted"
    else:
        reason = writable(external)
        if reason is None and shutil.disk_usage(external).free < 20e9:
            reason = "under 20 GB free"
    if reason is None:
        if reported.get("state") != "external":
            log("archive on", external)
            reported["state"] = "external"
        return external
    if reported.get("state") != reason:
        log("external archive", reason, "- archiving to the internal disk")
        reported["state"] = reason
    return internal


def move_to_external(root: Path, external: Path) -> None:
    """Carry footage archived internally while the drive was away over to the drive."""
    for source in sorted((root / "archive").glob("*/*.mp4")):
        if stop.is_set():
            return
        target = external / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.mp4")
        shutil.copyfile(source, partial)
        if partial.stat().st_size != source.stat().st_size:
            partial.unlink(missing_ok=True)
            log("copy to the drive failed size check; kept", source.name)
            return
        partial.replace(target)
        source.unlink()
        log("moved to the drive", source.name)


def compact(root: Path, cap_bytes: int, bitrate: str, min_age: float, external: Path | None = None):
    raw, reported = root / "raw", {}
    while not stop.is_set():
        try:
            archive = archive_root(root, external, reported)
            if external is not None and archive == external:
                move_to_external(root, external)
            segments = []
            for folder in raw.glob("*"):
                files = sorted(folder.glob("*.mp4"))
                # The newest file in each camera folder is still being written.
                segments += [f for f in files[:-1] if time.time() - f.stat().st_mtime > min_age]
            total = sum(f.stat().st_size for folder in raw.glob("*") for f in folder.glob("*.mp4"))
            if total <= cap_bytes or not segments:
                stop.wait(60)
                continue
            source = min(segments, key=lambda f: f.stat().st_mtime)
            target = archive / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".partial.mp4")
            command = [
                "nice",
                "-n",
                "10",
                FFMPEG,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-hwaccel",
                "videotoolbox",
                "-i",
                str(source),
                "-map",
                "0:v",
                "-map",
                "0:a?",
                "-c:v",
                "hevc_videotoolbox",
                "-b:v",
                bitrate,
                "-constant_bit_rate",
                "1",
                "-tag:v",
                "hvc1",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(partial),
            ]
            started = time.monotonic()
            done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            want, got = duration(source), duration(partial) if partial.exists() else 0.0
            if done.returncode == 0 and want > 0 and abs(got - want) <= max(2.0, 0.02 * want):
                partial.replace(target)
                before = source.stat().st_size
                source.unlink()
                log(
                    "compacted",
                    source.name,
                    f"{before / 1e6:.0f} MB -> {target.stat().st_size / 1e6:.0f} MB",
                    f"in {time.monotonic() - started:.0f}s",
                )
            else:
                partial.unlink(missing_ok=True)
                failed = root / "raw-unconverted" / source.parent.name
                failed.mkdir(parents=True, exist_ok=True)
                source.replace(failed / source.name)  # kept, never deleted; retried by hand
                log(
                    "compaction failed, raw kept in raw-unconverted:",
                    source.name,
                    want,
                    got,
                    done.stderr[-300:],
                )
        except Exception as exc:  # the recorder threads must outlive any compaction error
            log("compaction error", type(exc).__name__, str(exc)[:200])
            stop.wait(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=int, default=600)
    parser.add_argument("--raw-cap-gb", type=float, default=5.0)
    parser.add_argument("--hevc-bitrate", default="2500k")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="folder on an external drive for compressed footage (falls back to ROOT/archive)",
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--resume-free-gb", type=float, default=10.0)
    parser.add_argument(
        "--probe", type=int, default=0, help="record N seconds per camera to --root/probe and exit"
    )
    args = parser.parse_args()
    cams = cameras(args.cameras)
    if args.probe:
        threads = []
        for name, url in cams:
            out = args.root / "probe" / f"{name}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            command = [
                FFMPEG,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-rtsp_transport",
                "tcp",
                "-i",
                url,
                "-t",
                str(args.probe),
                "-map",
                "0:v",
                "-map",
                "0:a?",
                "-c",
                "copy",
                str(out),
            ]
            threads.append(
                (name, out, subprocess.Popen(command, stderr=subprocess.PIPE, text=True))
            )
        for name, out, process in threads:
            process.wait()
            size = out.stat().st_size if out.exists() else 0
            print(
                name,
                "exit",
                process.returncode,
                f"{size / 1e6:.1f} MB",
                f"{size * 8 / 1e6 / max(args.probe, 1):.2f} Mbit/s",
                (process.stderr.read() or "").replace(url, "<camera>")[-200:],
            )
        return
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    workers = [
        threading.Thread(
            target=record,
            args=(n, u, args.root, args.segment_seconds, args.min_free_gb, args.resume_free_gb),
            daemon=True,
        )
        for n, u in cams
    ]
    workers.append(
        threading.Thread(
            target=compact,
            args=(args.root, int(args.raw_cap_gb * 1e9), args.hevc_bitrate, 90, args.archive),
            daemon=True,
        )
    )
    for worker in workers:
        worker.start()
    log("recorder started for", ", ".join(n for n, _ in cams), f"raw cap {args.raw_cap_gb} GB")
    while not stop.is_set():
        stop.wait(5)
    for worker in workers:
        worker.join(timeout=20)


if __name__ == "__main__":
    sys.exit(main())

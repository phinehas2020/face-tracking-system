"""Compare an independent three-strip motion signal with an offline tracker replay."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from benchmark_live_models import _sha256, read_timestamps, validate_timestamp_coverage
from ingress.gate_motion import MAX_FRAME_GAP, analyze, compare_passages, strip_pixels
from ingress.live_counting import PassageCalibration


def read_strips(video, times, gate, start, stop):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        validate_timestamp_coverage(times, capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("Cannot seek to first compared frame")
        samples = []
        shape = None
        for _ in range(start, stop):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Video ended before timestamp sidecar")
            if shape is None:
                shape = frame.shape
                y, x = strip_pixels(gate, shape[1], shape[0])
            elif frame.shape != shape:
                raise ValueError("Resolution changed within recording")
            samples.append(frame[y, x])
        return np.stack(samples)
    finally:
        capture.release()


def draw_report(path, times, samples, levels, windows, outside):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    palette = ["#2e7286", "#192f3e", "#b04c2a"]
    examples = sorted(windows, key=lambda w: w.get("comparison") == "agreement")[:3]
    cols = max(1, len(examples))
    fig = plt.figure(figsize=(12, 7.0 if examples else 4.0), layout="constrained")
    fig.patch.set_facecolor("#faf8f2")
    grid = fig.add_gridspec(3, cols, height_ratios=[2.3, 1.55, .7]) if examples else (
        fig.add_gridspec(1, cols)
    )
    fig.suptitle("Doorway motion audit", x=.055, ha="left", fontsize=22, weight="bold")

    def trace(ax, mask):
        ax.set_facecolor("#faf8f2")
        clock = times[mask]
        cuts = np.flatnonzero(np.diff(clock) > MAX_FRAME_GAP) + 1
        for segment in np.split(np.flatnonzero(mask), cuts):
            for i, color in enumerate(palette):
                ax.plot(times[segment], levels[segment, i], color=color, lw=1, alpha=.88)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
        ax.grid(axis="y", color="#d8d8cc", lw=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, pad=5, labelsize=8)
        ax.set_xlabel("Capture time (seconds)", fontsize=9)

    top = fig.add_subplot(grid[0, :])
    trace(top, np.ones(len(times), dtype=bool))
    for cut in np.flatnonzero(np.diff(times) > MAX_FRAME_GAP):
        top.axvspan(times[cut], times[cut + 1], facecolor="#ddd", hatch="//", zorder=0)
    top.set_xlim(times[0] - .05, times[-1] + .05)
    top.set_ylabel("Strip changed from background", fontsize=9)
    if outside is None:
        subtitle = f"{len(windows)} motion windows  ·  Independent pixel analysis"
    else:
        agreements = sum(w["comparison"] == "agreement" for w in windows)
        motion_only = sum(w["comparison"] == "motion only" for w in windows)
        subtitle = (f"Direction agreed in {agreements}/{len(windows)} windows"
                    f"    ·    Motion only   {motion_only}    ·    Tracker only   {len(outside)}"
                    if windows else f"No motion windows    ·    Tracker only   {len(outside)}")
    top.set_title(subtitle, loc="left", fontsize=11, pad=10)
    for window in windows:
        top.axvspan(window["start_seconds"], window["end_seconds"], color="#e6c777",
                    alpha=.28, zorder=0)
    for event in outside or []:
        top.axvline(event["timestamp"], color="#af3550", ls="--", lw=1)
    handles = [Line2D([], [], color=c, label=label) for c, label in
               zip(palette, ["Negative side", "Counting line", "Positive side"], strict=True)]
    top.legend(handles=handles, loc="upper right", ncols=3, fontsize=8, frameon=False)
    for col, w in enumerate(examples):
        ax = fig.add_subplot(grid[1, col])
        lo, hi = w["start_seconds"] - .5, w["end_seconds"] + .5
        mask = (lo <= times) & (times <= hi)
        trace(ax, mask)
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        ax.set_xlim(lo, hi)
        for event in w.get("tracker_passages", []):
            ax.axvline(event["timestamp"], color="#675386", lw=2, ls=":")
        direction = {"in": "Entry", "out": "Exit", None: "Unresolved"}[w["direction"]]
        heading = f"{direction}   {w['start_seconds']:.1f}–{w['end_seconds']:.1f} s"
        if w.get("comparison") == "motion only":
            heading += "   ·   motion only"
        ax.set_title(heading,
                     loc="left", fontsize=12, weight="bold")
        strip = fig.add_subplot(grid[2, col], sharex=ax)
        strip.set_yticks([])
        strip.set_xlabel("Capture time (seconds)", fontsize=9)
        strip.tick_params(length=0, labelsize=8)
        strip.spines[:].set_visible(False)
        # Each column is the actual center line at its capture time. Never paint across gaps.
        indices = np.flatnonzero(mask)
        for block in np.split(indices, np.flatnonzero(np.diff(times[mask]) > MAX_FRAME_GAP) + 1):
            if len(block) < 2:
                continue
            clock = times[block]
            edges = np.r_[clock[0], (clock[:-1] + clock[1:]) / 2, clock[-1]]
            rgb = samples[block, 1, :, ::-1].transpose(1, 0, 2)
            strip.pcolormesh(edges, np.arange(rgb.shape[0] + 1), rgb, rasterized=True)
        strip.invert_yaxis()
    if examples:
        caption = "Bottom strips show the pixels on the counting line as time passes."
        if outside is not None:
            caption += "  Dotted lines mark tracker passages."
        fig.supxlabel(caption, fontsize=9, color="#495963")
    fig.savefig(path, dpi=140, backend="agg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, help="Output of benchmark_live_models.py")
    parser.add_argument("--run", default=0, type=int, help="Zero-based benchmark result index")
    parser.add_argument("--output", required=True, type=Path, help="New report directory")
    args = parser.parse_args()
    times = np.array(read_timestamps(args.timestamps)["timestamps"])
    gate = PassageCalibration.from_calibration(json.loads(args.calibration.read_text()))
    video_hash, clock_hash = _sha256(args.video), _sha256(args.timestamps)
    start, stop = 0, len(times)
    passages = None
    if args.benchmark:
        benchmark = json.loads(args.benchmark.read_text())
        if (benchmark["video_sha256"] != video_hash or benchmark["timestamps_sha256"] != clock_hash
                or PassageCalibration.from_calibration(benchmark["calibration"]) != gate):
            parser.error("Benchmark source, clock, and gate must match the motion audit")
        if not 0 <= args.run < len(benchmark["results"]):
            parser.error("Benchmark run index is outside the available results")
        run = benchmark["results"][args.run]
        if run["status"] != "completed":
            parser.error("Choose a completed tracker benchmark run")
        start = int(np.searchsorted(times, run["source_start_seconds"]))
        stop = int(np.searchsorted(times, run["source_end_seconds"], side="right"))
        passages = [{"timestamp": e["timestamp"], "direction": e["direction"]}
                    for e in run["events"]]
    if args.output.exists() or start >= stop:
        parser.error("Provide a new output directory and a nonempty evaluation interval")
    samples = read_strips(args.video, times, gate, start, stop)
    times = times[start:stop]
    windows, levels = analyze(times, samples, gate.entry_positive)
    outside = None
    if passages is not None:
        windows, outside = compare_passages(windows, passages)
    report = {"schema_version": 1, "video_sha256": video_hash, "timestamps_sha256": clock_hash,
              "gate_endpoints": gate.endpoints, "entry_positive": gate.entry_positive,
              "source_interval": [times[0], times[-1]], "frames_analyzed": len(times),
              "capture_gaps": [[times[i], times[i + 1]]
                               for i in np.flatnonzero(np.diff(times) > MAX_FRAME_GAP)],
              "benchmark_sha256": _sha256(args.benchmark) if args.benchmark else None,
              "benchmark_run": args.run if args.benchmark else None,
              "windows": windows, "tracker_passages_outside_windows": outside}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete private report, including when plotting fails partway.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "result"
        staged.mkdir(mode=0o700)
        (staged / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        draw_report(staged / "motion.png", times, samples, levels, windows, outside)
        os.rename(staged, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

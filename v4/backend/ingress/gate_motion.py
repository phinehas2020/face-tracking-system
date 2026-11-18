"""Offline three-strip motion audit for a stationary passage camera."""

from __future__ import annotations

import numpy as np

from .live_counting import PassageCalibration

# Distances are normalized like the live gate. Samples run along its finite segment.
STRIP_OFFSET = 0.04
SAMPLES = 80
STEP_SECONDS = 0.05
MAX_FRAME_GAP = 0.25


def strip_pixels(gate: PassageCalibration, width: int, height: int):
    start, end = np.array(gate.endpoints)
    dx, dy = end - start
    normal = np.array([dy, -dx]) / np.hypot(dx, dy)
    middle = np.linspace(start, end, SAMPLES)
    lines = np.stack([middle + shift * normal for shift in (-STRIP_OFFSET, 0, STRIP_OFFSET)])
    valid = ((lines >= 0) & (lines <= 1)).all(axis=(0, 2))
    if np.count_nonzero(valid) < SAMPLES / 2:
        raise ValueError("Gate too close to image boundary for a three-strip audit")
    xy = np.rint(lines[:, valid] * [width - 1, height - 1]).astype(int)
    return xy[:, :, 1], xy[:, :, 0]


def _nearest(times, grid):
    right = np.minimum(np.searchsorted(times, grid), len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(grid - times[left] <= times[right] - grid, left, right)


def _delay(first, second):
    # Compare the same time support for every lag, including zero and both directions.
    bound = min(10, (len(first) - 5) // 2)
    if bound < 1:
        return None
    middle = first[bound:-bound]
    scores = []
    for lag in range(-bound, bound + 1):
        shifted = second[bound + lag:len(first) - bound + lag]
        norm = np.sqrt(np.sum(middle * middle) * np.sum(shifted * shifted))
        scores.append(float(np.sum(middle * shifted) / norm) if norm else 0.)
    lag = int(np.argmax(scores)) - bound
    rival = max(scores[:bound + 1] if lag > 0 else scores[bound:])
    if lag == 0 or max(scores) < 0.15 or max(scores) - rival < 0.04:
        return None
    return round(lag * STEP_SECONDS, 2)


def _window(times, samples, first, last, entry_positive):
    start, end = times[first], times[last]
    direction = None
    delays = [None, None]
    if start - times[0] < MAX_FRAME_GAP or times[-1] - end < MAX_FRAME_GAP:
        reason = "capture boundary"
    elif end - start > 6.0:
        reason = "sustained activity"
    else:
        grid = np.arange(max(times[0], start - .6), min(times[-1], end + .6), STEP_SECONDS)
        colors = samples[_nearest(times, grid)].astype(np.float32)
        changes = np.clip(colors[2:] - colors[:-2], -60, 60)
        delays = [_delay(changes[:, i], changes[:, i + 1]) for i in (0, 1)]
        if all(d is not None for d in delays) and delays[0] * delays[1] > 0:
            direction = "in" if (delays[0] > 0) == entry_positive else "out"
            reason = "both strip pairs agree"
        else:
            reason = "strip delays inconclusive"
    return {"start_seconds": float(start), "end_seconds": float(end),
            "direction": direction, "reason": reason, "pair_delays_seconds": delays}


def analyze(times, samples, entry_positive):
    """Return review windows and per-strip foreground fractions, never a headcount."""
    times = np.asarray(times, dtype=float)
    samples = np.asarray(samples)
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Expected strictly increasing capture times")
    if (samples.ndim != 4 or samples.shape[:2] != (len(times), 3)
            or samples.shape[2] < 1 or samples.shape[3] != 3):
        raise ValueError("Expected three strips of color samples per capture")
    windows = []
    levels = np.zeros((len(times), 3))
    boundaries = np.flatnonzero(np.diff(times) > MAX_FRAME_GAP) + 1
    for frames in np.split(np.arange(len(times)), boundaries):
        clock, pixels = times[frames], samples[frames]
        reference_times = np.linspace(clock[0], clock[-1], min(401, len(clock)))
        background = np.median(pixels[_nearest(clock, reference_times)], axis=0).astype(np.int16)
        change = np.max(np.abs(pixels.astype(np.int16) - background), axis=-1)
        activity = (change > 32).mean(axis=-1)
        levels[frames] = activity
        active = np.flatnonzero(activity.min(axis=1) >= .20)
        for run in np.split(active, np.flatnonzero(np.diff(clock[active]) > .6) + 1):
            if len(run) >= 3 and clock[run[-1]] - clock[run[0]] >= .15:
                windows.append(_window(clock, pixels, run[0], run[-1], entry_positive))
    return windows, levels


def compare_passages(windows, passages):
    """Match tracker timestamps to disjoint motion intervals on the same capture clock."""
    outside = []
    ordered = sorted(passages, key=lambda p: p["timestamp"])
    comparison = []
    cursor = 0
    for window in windows:
        start, end = window["start_seconds"], window["end_seconds"]
        while cursor < len(ordered) and ordered[cursor]["timestamp"] < start:
            outside.append(ordered[cursor])
            cursor += 1
        first = cursor
        while cursor < len(ordered) and ordered[cursor]["timestamp"] <= end:
            cursor += 1
        events = ordered[first:cursor]
        if not events:
            verdict = "motion only"
        elif len(events) > 1:
            verdict = "multiple tracker passages"
        elif not window["direction"]:
            verdict = "motion direction unresolved"
        else:
            verdict = "agreement" if events[0]["direction"] == window["direction"] else "opposed"
        comparison.append({**window, "tracker_passages": events, "comparison": verdict})
    return comparison, outside + ordered[cursor:]

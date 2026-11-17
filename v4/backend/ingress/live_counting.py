"""Opt-in live passage counting from observed, normalized body trajectories.

This module does not identify people or reconnect different track IDs. Callers
must reset it on camera reconnect, pause, calibration changes, or track epochs.
Frame timestamps must be monotonic capture times, not wall clock or replay time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

Point = tuple[float, float]


def _point(value: Point) -> Point:
    if len(value) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in value):
        raise ValueError("Gate and anchor coordinates must be finite and normalized to [0, 1]")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class PassageCalibration:
    """Gate's right-hand normal defines positive travel.

    A vertical axis gate runs top to bottom (positive means right). A horizontal
    axis gate runs right to left (positive means down). Custom gate endpoints
    use that same right-hand normal: (end_y-start_y, start_x-end_x).
    """

    axis: str = "vertical"
    position: float = 0.5
    entry_positive: bool = True
    segment_start: float = 0.0
    segment_end: float = 1.0
    gate_start: Point | None = None
    gate_end: Point | None = None
    anchor: Literal["center", "footpoint"] = "center"
    hysteresis: float = 0.025
    establish_seconds: float = 0.1
    confirm_seconds: float = 0.1
    max_gap_seconds: float = 1.0
    max_traversal_seconds: float = 3.0
    min_motion: float = 0.05

    def __post_init__(self) -> None:
        if self.axis not in {"vertical", "horizontal"}:
            raise ValueError("axis must be vertical or horizontal")
        if self.anchor not in {"center", "footpoint"}:
            raise ValueError("anchor must be center or footpoint")
        if not math.isfinite(self.position) or not 0 <= self.position <= 1:
            raise ValueError("position must be normalized to [0, 1]")
        if not 0 <= self.segment_start < self.segment_end <= 1:
            raise ValueError("segment_start must precede segment_end in [0, 1]")
        for name in ("hysteresis", "min_motion"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be positive and less than 1")
        for name in ("establish_seconds", "confirm_seconds", "max_gap_seconds",
                     "max_traversal_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (self.gate_start is None) != (self.gate_end is None):
            raise ValueError("Both custom gate endpoints are required")
        if self.gate_start is not None:
            object.__setattr__(self, "gate_start", _point(self.gate_start))
            object.__setattr__(self, "gate_end", _point(self.gate_end))
        start, end = self.endpoints
        if math.dist(start, end) < 1e-6:
            raise ValueError("Gate endpoints must be distinct")

    @property
    def endpoints(self) -> tuple[Point, Point]:
        if self.gate_start is not None:
            return self.gate_start, self.gate_end
        if self.axis == "vertical":
            return (self.position, self.segment_start), (self.position, self.segment_end)
        return (self.segment_end, self.position), (self.segment_start, self.position)

    @classmethod
    def from_calibration(cls, calibration: object) -> PassageCalibration:
        """Accept LiveCalibration without importing the live runtime or changing legacy mode."""
        if isinstance(calibration, dict):
            values = calibration
        else:
            values = {item.name: getattr(calibration, item.name) for item in fields(cls)
                      if hasattr(calibration, item.name)}
        return cls(**{item.name: values[item.name] for item in fields(cls) if item.name in values})


def anchor_for_bbox(
    box: tuple[float, float, float, float], width: float, height: float,
    anchor: str = "center",
) -> Point:
    """Return normalized anchor from a pixel xyxy box; clipped feet are not inferred.

    Center is the observed box's center, suitable for explicitly configured
    torso gates. Footpoint means bottom-center of the box, not a 3D floor point.
    """
    if anchor not in {"center", "footpoint"}:
        raise ValueError("anchor must be center or footpoint")
    if not all(math.isfinite(v) for v in (*box, width, height)) or width <= 0 or height <= 0:
        raise ValueError("Image dimensions and bbox must be finite with positive image size")
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("bbox must have positive area within the image")
    return ((x1 + x2) / (2 * width), (y2 if anchor == "footpoint" else (y1 + y2) / 2)
            / height)


@dataclass
class PassageState:
    side: int = 0
    region_side: int = 0
    region_since: float = 0.0
    last_point: Point | None = None
    last_seen: float = 0.0
    origin_distance: float = 0.0
    pending_side: int = 0
    pending_since: float = 0.0
    crossing_at: float | None = None
    crosses_gate: bool = False


class PassageCounter:
    """Confirm observed gate traversals with time hysteresis and bounded history.

    The first point well outside the gate band establishes a spatial reference;
    nearer points need repeated time-separated evidence. Every destination
    requires two or more observations spanning confirm_seconds. The counter
    never confirms based on elapsed time alone or missing/predicted detections.
    """

    def __init__(self, calibration: PassageCalibration, max_tracks: int = 1024):
        if max_tracks <= 0:
            raise ValueError("max_tracks must be positive")
        self.calibration = PassageCalibration.from_calibration(calibration)
        self.max_tracks = max_tracks
        self.states: dict[str, PassageState] = {}
        self._start, self._end = self.calibration.endpoints
        self._dx = self._end[0] - self._start[0]
        self._dy = self._end[1] - self._start[1]
        self._length = math.hypot(self._dx, self._dy)

    def _distance(self, point: Point) -> float:
        return (self._dy * (point[0] - self._start[0])
                - self._dx * (point[1] - self._start[1])) / self._length

    def _intersection(self, first: Point, second: Point) -> bool | None:
        """None: no crossing; False: crossed the infinite line outside the finite gate."""
        d1, d2 = self._distance(first), self._distance(second)
        if d1 == d2 or d1 * d2 > 0:
            return None
        fraction = d1 / (d1 - d2)
        x = first[0] + fraction * (second[0] - first[0]) - self._start[0]
        y = first[1] + fraction * (second[1] - first[1]) - self._start[1]
        along = (x * self._dx + y * self._dy) / (self._length ** 2)
        return 0 <= along <= 1

    @staticmethod
    def _clear_pending(state: PassageState) -> None:
        state.pending_side = 0
        state.crossing_at = None
        state.crosses_gate = False

    def update(
        self, track_id: str, point: Point, now: float, body_anchor: Point | None = None,
    ) -> str | None:
        # body_anchor is accepted only for API compatibility. The selected
        # observed body anchor is already supplied as point by the caller.
        if not math.isfinite(now):
            raise ValueError("Observation timestamp must be finite")
        try:
            point = _point(point)
        except ValueError:
            self.states.pop(track_id, None)
            raise
        state = self.states.get(track_id)
        if state is not None and now <= state.last_seen:
            return None  # duplicate/out-of-order observations cannot confirm a crossing
        self.expire(now)
        state = self.states.get(track_id)
        if state is None:
            if len(self.states) >= self.max_tracks:
                oldest = min(self.states, key=lambda key: self.states[key].last_seen)
                del self.states[oldest]
            state = self.states[track_id] = PassageState(last_seen=now)
        previous = state.last_point
        state.last_point, state.last_seen = point, now
        distance = self._distance(point)
        side = 0 if abs(distance) <= self.calibration.hysteresis else (1 if distance > 0 else -1)

        if state.side == 0:
            if side != state.region_side:
                state.region_side, state.region_since = side, now
            clear_distance = max(2 * self.calibration.hysteresis, self.calibration.min_motion)
            if side and (abs(distance) >= clear_distance
                         or now - state.region_since >= self.calibration.establish_seconds):
                state.side, state.origin_distance = side, distance
            return None

        if (state.crossing_at is not None
                and now - state.crossing_at > self.calibration.max_traversal_seconds):
            # A long stay astride the line is ambiguous. Re-establish from
            # new observations instead of inventing a delayed passage.
            self.states[track_id] = PassageState(last_point=point, last_seen=now)
            return None
        if previous is not None:
            intersects = self._intersection(previous, point)
            if intersects is not None:
                state.crossing_at, state.crosses_gate = now, intersects

        if side == state.side:
            # Preserve the furthest observed source-side anchor. Approaching
            # the gate must not erase already observed body displacement.
            state.origin_distance = (max if side > 0 else min)(state.origin_distance, distance)
            self._clear_pending(state)
            return None
        if side == 0:
            state.pending_side = 0
            return None
        if state.pending_side != side:
            state.pending_side, state.pending_since = side, now
            return None
        if now - state.pending_since + 1e-9 < self.calibration.confirm_seconds:
            return None

        crossed = state.crossing_at is not None and state.crosses_gate
        if crossed and abs(distance - state.origin_distance) < self.calibration.min_motion:
            # Keep the observed intersection pending until displacement is
            # sufficient. Committing the destination here would lose the
            # passage permanently even if the person continues walking.
            return None
        state.side, state.origin_distance = side, distance
        self._clear_pending(state)
        if crossed:
            return "in" if (side > 0) == self.calibration.entry_positive else "out"
        return None

    def expire(self, now: float) -> None:
        self.states = {key: state for key, state in self.states.items()
                       if now - state.last_seen <= self.calibration.max_gap_seconds}

    def reset(self) -> None:
        self.states.clear()

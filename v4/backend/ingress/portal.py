from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Point = tuple[float, float]


class PortalState(StrEnum):
    OUTSIDE = "outside"
    APPROACH = "approach"
    COMMITTED = "committed"
    INSIDE = "inside"


@dataclass(slots=True)
class PortalTrack:
    track_id: str
    state: PortalState = PortalState.OUTSIDE
    stable_frames: int = 0
    last_side: float | None = None
    emitted: bool = False


class PortalStateMachine:
    """Turns a noisy track into one directional passage event."""

    def __init__(
        self,
        line_start: Point,
        line_end: Point,
        direction_sign: int = 1,
        stable_frames: int = 3,
        hysteresis: float = 0.015,
    ) -> None:
        self.line_start = line_start
        self.line_end = line_end
        self.direction_sign = 1 if direction_sign >= 0 else -1
        self.required_stable_frames = stable_frames
        self.hysteresis = hysteresis
        self.tracks: dict[str, PortalTrack] = {}

    def signed_side(self, point: Point) -> float:
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        x, y = point
        return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) * self.direction_sign

    def update(self, track_id: str, footpoint: Point) -> bool:
        track = self.tracks.setdefault(track_id, PortalTrack(track_id=track_id))
        side = self.signed_side(footpoint)

        if track.last_side is None:
            track.last_side = side
            track.state = PortalState.APPROACH if side < -self.hysteresis else PortalState.OUTSIDE
            return False

        if track.emitted:
            track.last_side = side
            return False

        if side < -self.hysteresis:
            track.state = PortalState.APPROACH
            track.stable_frames = 0
        elif side > self.hysteresis and track.state in {
            PortalState.APPROACH,
            PortalState.COMMITTED,
        }:
            track.state = PortalState.COMMITTED
            track.stable_frames += 1
            if track.stable_frames >= self.required_stable_frames:
                track.state = PortalState.INSIDE
                track.emitted = True
                track.last_side = side
                return True

        track.last_side = side
        return False

    def forget(self, track_id: str) -> None:
        self.tracks.pop(track_id, None)

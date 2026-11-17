"""Clock-aware, one-to-one association between two views of one passage area."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewObservation:
    key: str
    point: tuple[float, float]
    appearance: np.ndarray
    frame: int
    clock: float
    grounded: bool = True


class PairMatcher:
    """Require geometry, appearance, ambiguity margin and fresh-frame agreement.

    A repeated display frame never adds evidence. No matches across stale views;
    front-to-side ground-plane homography must be calibrated for this camera pair.
    """

    def __init__(
        self,
        homography,
        max_distance=0.075,
        min_appearance=0.65,
        margin=0.08,
        confirmations=2,
        max_skew=0.8,
    ):
        self.h = np.asarray(homography, dtype=float)
        if self.h.shape != (3, 3) or not np.isfinite(self.h).all():
            raise ValueError("Expected a finite 3x3 ground-plane homography")
        self.max_distance = max_distance
        self.min_appearance = min_appearance
        self.margin = margin
        self.confirmations = confirmations
        self.max_skew = max_skew
        self.history = {}

    def update(self, front, side):
        candidates = []
        for a in front:
            projected = self.h @ np.array([*a.point, 1.0])
            if not a.grounded or abs(projected[2]) < 1e-8:
                continue
            point = projected[:2] / projected[2]
            for b in side:
                skew = abs(a.clock - b.clock)
                if not b.grounded or skew > self.max_skew:
                    continue
                distance = float(np.linalg.norm(point - np.asarray(b.point)))
                similarity = float(a.appearance @ b.appearance)
                if distance > self.max_distance or similarity < self.min_appearance:
                    continue
                score = 0.7 * (1 - distance / self.max_distance) + 0.3 * similarity
                candidates.append((score, a, b, distance, similarity, skew))
        accepted, seen = [], set()
        for score, a, b, distance, similarity, skew in sorted(candidates, key=lambda c: -c[0]):
            rivals = [
                s
                for s, x, y, *_ in candidates
                if (x.key == a.key or y.key == b.key) and (x.key, y.key) != (a.key, b.key)
            ]
            if rivals and score - max(rivals) < self.margin:
                continue
            key = (a.key, b.key)
            seen.add(key)
            old = self.history.get(key)
            count = old[2] if old and a.clock - old[3] <= 3 else 0
            if not old or (a.frame != old[0] and b.frame != old[1]):
                count += 1
                self.history[key] = (a.frame, b.frame, count, a.clock)
            if count >= self.confirmations:
                accepted.append(
                    {
                        "front": a.key,
                        "side": b.key,
                        "distance": round(distance, 4),
                        "appearance": round(similarity, 4),
                        "skew": round(skew, 3),
                        "confirmations": count,
                        "front_frame": a.frame,
                        "side_frame": b.frame,
                    }
                )
        latest = max([o.clock for o in front + side], default=0)
        front_frames = {o.key: o.frame for o in front}
        self.history = {
            k: v
            for k, v in self.history.items()
            if k in seen or (front_frames.get(k[0]) == v[0] and latest - v[3] <= 3)
        }
        return accepted


def validate_clock_index(rows):
    """Reject scrambled/ambiguous timestamps instead of quietly joining wrong footage."""
    if len(rows) < 2:
        raise ValueError("A camera needs at least two timestamped frames")
    if any(
        not np.isfinite(r["clock"]) or r["frame"] < 0 or int(r["frame"]) != r["frame"] for r in rows
    ):
        raise ValueError("Clock indexes require finite times and nonnegative integer frames")
    for a, b in zip(rows, rows[1:], strict=False):
        if b["frame"] <= a["frame"] or b["clock"] <= a["clock"]:
            raise ValueError("Frame and clock indexes must increase strictly")
    return rows

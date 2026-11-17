"""Where a facing-camera track is when its person crosses the side camera's line.

The side view sees one place (the doorway); the facing view sees the room behind
it. A person crossing the line is therefore at a repeatable place in the facing
image, and often the facing track begins or ends there. That place is learned
from crossings where exactly one moving person was in each view, in the spirit
of field-of-view lines (Khan and Shah, PAMI 2003), and then used as position
evidence: the facing track that is at the door is the one that crossed.

A state is ``(centre x, centre y, log box height, time offset)``. Positions are
normalised image coordinates. The offset is how far the nearest facing sample
is from the crossing instant, zero when the track spans it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

GATE = 13.28  # chi-square 0.99, four degrees of freedom
POSITION_GATE = 11.34  # chi-square 0.99, three degrees of freedom
FLOOR = np.array([0.01, 0.01, 0.05, 0.10])


class CrossingZone:
    def __init__(self, min_samples=20, max_samples=300, max_background_rate=0.5):
        if min_samples < 8 or max_samples < min_samples or not 0 < max_background_rate <= 1:
            raise ValueError("Invalid crossing zone bounds")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_background_rate = max_background_rate
        self.samples: dict[str, deque] = {}
        self.background: deque = deque(maxlen=2000)
        self._models: dict[str, tuple | None] = {}

    def add(self, direction: str, state) -> bool:
        vector = np.asarray(state, dtype=float)
        if vector.shape != (4,) or not np.isfinite(vector).all():
            return False
        self.samples.setdefault(direction, deque(maxlen=self.max_samples)).append(vector)
        self._models.pop(direction, None)
        return True

    def add_background(self, position) -> None:
        vector = np.asarray(position, dtype=float)
        if vector.shape == (3,) and np.isfinite(vector).all():
            self.background.append(vector)

    def _model(self, direction: str):
        """Trimmed Gaussian: outliers (a bystander, a missed detection) are dropped."""
        if direction in self._models:
            return self._models[direction]
        model = None
        samples = self.samples.get(direction, ())
        if len(samples) >= self.min_samples:
            x = np.stack(samples)
            center = np.median(x, axis=0)
            scale = np.maximum(1.4826 * np.median(np.abs(x - center), axis=0), FLOOR)
            keep = (((x - center) / scale) ** 2).sum(axis=1) <= GATE
            inverse = None
            for _ in range(3):
                if keep.sum() < self.min_samples:
                    break
                center = x[keep].mean(axis=0)
                covariance = np.cov(x[keep], rowvar=False)
                # Shrink towards the diagonal; few samples estimate correlations badly.
                covariance = 0.8 * covariance + 0.2 * np.diag(np.diag(covariance))
                covariance += np.diag(FLOOR**2)
                inverse = np.linalg.inv(covariance)
                delta = x - center
                keep = np.einsum("ij,jk,ik->i", delta, inverse, delta) <= GATE
            if inverse is not None and keep.sum() >= self.min_samples:
                position_inverse = np.linalg.inv(np.linalg.inv(inverse)[:3, :3])
                model = (center, inverse, float(keep.mean()), int(keep.sum()), position_inverse)
        self._models[direction] = model
        return model

    def background_rate(self, direction: str) -> float | None:
        """Share of all facing positions that fall inside the zone. Low is good."""
        model = self._model(direction)
        if model is None or len(self.background) < 200:
            return None
        delta = np.stack(self.background) - model[0][:3]
        return float((np.einsum("ij,jk,ik->i", delta, model[4], delta) <= POSITION_GATE).mean())

    def ready(self, direction: str) -> bool:
        rate = self.background_rate(direction)
        return rate is not None and rate <= self.max_background_rate

    def distance(self, direction: str, state) -> float | None:
        model = self._model(direction)
        if model is None:
            return None
        delta = np.asarray(state, dtype=float) - model[0]
        return float(delta @ model[1] @ delta)

    def summary(self) -> dict:
        directions = {}
        for direction, samples in self.samples.items():
            model = self._model(direction)
            rate = self.background_rate(direction)
            directions[direction] = {
                "samples": len(samples),
                "inliers": model[3] if model else 0,
                "inlier_rate": round(model[2], 3) if model else None,
                "center": [round(float(v), 3) for v in model[0]] if model else None,
                "background_inside_rate": None if rate is None else round(rate, 3),
                "ready": self.ready(direction),
            }
        return {
            "min_samples": self.min_samples,
            "background_positions": len(self.background),
            "directions": directions,
        }

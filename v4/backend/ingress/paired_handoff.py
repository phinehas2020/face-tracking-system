"""Short same-view identity handoff across tracker ID breaks, with audit evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


class IdentityHandoff:
    def __init__(self, max_gap=1.5, min_similarity=0.85, max_distance=0.10, margin=0.10):
        self.max_gap = max_gap
        self.min_similarity = min_similarity
        self.max_distance = max_distance
        self.margin = margin
        self.history = {}

    def update(self, observations, registry, clock, observed_at):
        active = {o.key for o in observations}
        candidates = []
        out = []
        for o in observations:
            state = registry.tracks.get(o.key)
            if state and (state.identity_id or state.state == "conflict"):
                continue
            for old_key, (old, face) in self.history.items():
                if old_key in active or not 0 < clock - old.clock <= self.max_gap:
                    continue
                if face.identity_id not in registry.templates:
                    continue
                distance = float(np.linalg.norm(np.asarray(o.point) - old.point))
                similarity = float(o.appearance @ old.appearance)
                if distance <= self.max_distance and similarity >= self.min_similarity:
                    candidates.append((similarity, o, old_key, old, face, distance))
        for score, o, key, old, face, distance in sorted(candidates, key=lambda x: -x[0]):
            rivals = [
                s
                for s, t, k, *_ in candidates
                if (t.key == o.key or k == key) and (t.key, k) != (o.key, key)
            ]
            if rivals and score - max(rivals) < self.margin:
                continue
            registry.bind_face_to_body(
                o.key, replace(face, state="tracked"), active, clock, observed_at
            )
            after = registry.tracks.get(o.key)
            if after and after.identity_id == face.identity_id:
                out.append(
                    {
                        "from": key,
                        "to": o.key,
                        "similarity": round(score, 4),
                        "distance": round(distance, 4),
                        "gap": round(clock - old.clock, 3),
                        "source_frame": old.frame,
                        "target_frame": o.frame,
                        "identity_id": after.identity_id,
                        "clock": clock,
                    }
                )
        self.history = {k: v for k, v in self.history.items() if clock - v[0].clock <= self.max_gap}
        for o in observations:
            state = registry.tracks.get(o.key)
            if (
                state
                and state.identity_id
                and state.state not in {"conflict", "review", "collecting"}
            ):
                self.history[o.key] = (
                    o,
                    replace(
                        state,
                        samples=list(state.samples),
                        qualities=list(state.qualities),
                        conflicts=list(state.conflicts),
                    ),
                )
        return out

"""One-to-one, repeated person-appearance correspondence on a common clock."""

from __future__ import annotations


class AppearancePairMatcher:
    def __init__(self, min_similarity=0.55, margin=0.08, confirmations=2, max_skew=0.8):
        self.min_similarity = min_similarity
        self.margin = margin
        self.confirmations = confirmations
        self.max_skew = max_skew
        self.history = {}

    def update(self, front, side):
        eligible = []
        for a in front:
            ranked = []
            for b in side:
                if abs(a.clock - b.clock) > self.max_skew:
                    continue
                score = float(a.appearance @ b.appearance)
                ranked.append((score, b))
            ranked.sort(key=lambda v: -v[0])
            if not ranked:
                continue
            score, b = ranked[0]
            separation = score - ranked[1][0] if len(ranked) > 1 else score
            # Compare every contemporaneous alternative BEFORE applying passage-zone gates.
            # Otherwise removing a plausible rival can artificially inflate confidence.
            if score < self.min_similarity or separation < self.margin:
                continue
            if a.point[1] < 0.4 or b.point[1] < 0.5:
                continue
            eligible.append((score, a, b, separation))
        accepted = []
        seen = set()
        for score, a, b, separation in eligible:
            rivals = [s for s, x, y, _ in eligible if y.key == b.key and x.key != a.key]
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
                        "appearance": round(score, 4),
                        "margin": round(separation, 4),
                        "skew": round(abs(a.clock - b.clock), 3),
                        "confirmations": count,
                        "front_frame": a.frame,
                        "side_frame": b.frame,
                        "kind": "repeated_person_appearance",
                    }
                )
        latest = max([o.clock for o in front + side], default=0)
        frames = {o.key: o.frame for o in front}
        self.history = {
            k: v
            for k, v in self.history.items()
            if k in seen or (frames.get(k[0]) == v[0] and latest - v[3] <= 3)
        }
        return accepted

"""Conservative short-gap handoff for fragmented body tracker identifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Track


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / max(union, 1)


@dataclass
class BodyHistory:
    bbox: tuple
    appearance: np.ndarray
    seen: float
    age: int = 1
    learned_appearance: np.ndarray | None = None
    velocity: tuple[float, float] = (0.0, 0.0)


class TrackContinuity:
    def __init__(self, max_gap=1.2):
        self.max_gap = max_gap
        self.histories: dict[str, BodyHistory] = {}
        self.raw_ids: dict[str, str] = {}
        self.next_id = 0
        self.handovers = 0

    def reset(self):
        self.histories.clear()
        self.raw_ids.clear()

    @staticmethod
    def appearance(frame, box):
        import cv2

        left, top, right, bottom = box
        width, height = right - left, bottom - top
        crop = frame[
            max(0, int(top + 0.2 * height)) : max(0, int(top + 0.75 * height)),
            max(0, int(left + 0.2 * width)) : max(0, int(right - 0.2 * width)),
        ]
        if not crop.size:
            return np.zeros(32, dtype=np.float32)
        hsv = cv2.cvtColor(cv2.resize(crop, (24, 32)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
        return np.sqrt(hist / max(float(hist.sum()), 1))

    def update(
        self,
        tracks: list[Track],
        frame,
        now: float,
        learned_appearance: dict[str, np.ndarray] | None = None,
    ) -> list[Track]:
        learned_appearance = learned_appearance or {}
        self.histories = {key: h for key, h in self.histories.items() if now - h.seen <= 5}
        self.raw_ids = {raw: key for raw, key in self.raw_ids.items() if key in self.histories}
        occupied = {self.raw_ids[t.track_id] for t in tracks if t.track_id in self.raw_ids}
        descriptors = {t.track_id: self.appearance(frame, t.bbox) for t in tracks}
        candidates = []
        for track in tracks:
            if track.track_id in self.raw_ids:
                continue
            for key, old in self.histories.items():
                if key in occupied or now - old.seen > self.max_gap:
                    continue
                iou = overlap(track.bbox, old.bbox)
                similarity = float(descriptors[track.track_id] @ old.appearance)
                learned = learned_appearance.get(track.track_id)
                old_learned = old.learned_appearance
                if learned is not None and old_learned is not None:
                    learned_similarity = float(learned @ old_learned)
                    a, b = track.bbox, old.bbox
                    distance = float(
                        np.hypot(
                            (a[0] + a[2] - b[0] - b[2]) / 2,
                            (a[1] + a[3] - b[1] - b[3]) / 2,
                        )
                    )
                    radius = 2 * max(a[2] - a[0], b[2] - b[0], 1)
                    size_ratio = max(a[2] - a[0], b[2] - b[0]) / max(
                        min(a[2] - a[0], b[2] - b[0]), 1
                    )
                    if learned_similarity >= 0.85 and distance <= radius and size_ratio <= 2:
                        score = 0.85 * learned_similarity + 0.15 * (1 - distance / radius)
                        candidates.append((score, track.track_id, key))
                        continue
                    # Short fast motion can lose all overlap while a partial crop
                    # weakens ReID. Require three independent cues and a measured
                    # trajectory; clothing color alone never reconnects a person.
                    gap = now - old.seen
                    velocity = np.asarray(old.velocity)
                    displacement = np.asarray(
                        [(a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2]
                    )
                    speed = float(np.linalg.norm(velocity))
                    tolerance = 0.6 * max(a[2] - a[0], b[2] - b[0], 1)
                    prediction_error = float(np.linalg.norm(displacement - velocity * gap))
                    agreement = float(displacement @ velocity) / max(distance * speed, 1e-8)
                    if (
                        old.age >= 3
                        and 0 < gap <= 0.65
                        and learned_similarity >= 0.65
                        and similarity >= 0.9
                        and size_ratio <= 2
                        and speed >= 0.1 * max(b[2] - b[0], 1)
                        and agreement >= 0.8
                        and distance <= 3 * speed * gap
                        and prediction_error <= tolerance
                    ):
                        score = (
                            0.5 * similarity
                            + 0.3 * learned_similarity
                            + 0.2 * (1 - prediction_error / tolerance)
                        )
                        candidates.append((score, track.track_id, key))
                        continue
                    # Contradictory learned evidence must not be bypassed by
                    # matching clothing color alone.
                    if learned_similarity < 0.65:
                        continue
                if iou >= 0.4 and similarity >= 0.85:
                    score = 0.65 * iou + 0.35 * similarity - 0.05 * (now - old.seen)
                    candidates.append((score, track.track_id, key))
        # Mutual best matches with a margin; crossing/overlapping candidates stay separate.
        for score, raw, key in sorted(candidates, reverse=True):
            if raw in self.raw_ids or key in occupied or score < 0.68:
                continue
            rivals = [
                s for s, r, k in candidates if (r == raw or k == key) and (r, k) != (raw, key)
            ]
            if rivals and score - max(rivals) < 0.12:
                continue
            self.raw_ids = {r: k for r, k in self.raw_ids.items() if k != key}
            self.raw_ids[raw] = key
            occupied.add(key)
            self.handovers += 1
        result = []
        for track in tracks:
            key = self.raw_ids.get(track.track_id)
            if key is None:
                self.next_id += 1
                key = f"track-{self.next_id}"
                self.raw_ids[track.track_id] = key
            previous = self.histories.get(key)
            appearance = descriptors[track.track_id]
            age = previous.age + 1 if previous else 1
            learned = learned_appearance.get(track.track_id)
            if learned is None and previous is not None:
                learned = previous.learned_appearance
            velocity = (0.0, 0.0)
            if previous and now > previous.seen:
                a, b = track.bbox, previous.bbox
                measured = np.asarray(
                    [(a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2]
                ) / (now - previous.seen)
                velocity = tuple(
                    measured
                    if previous.age == 1
                    else 0.5 * measured + 0.5 * np.asarray(previous.velocity)
                )
            self.histories[key] = BodyHistory(track.bbox, appearance, now, age, learned, velocity)
            result.append(Track(key, track.bbox, track.confidence, age))
        return result


class FaceContinuity:
    """Keep short face tracklets independently of occluded body boxes."""

    def __init__(self, max_gap=1.5, max_motion_widths=0):
        self.max_motion_widths = max_motion_widths
        self.max_gap = max_gap
        self.history = {}
        self.velocities = {}
        self.next_id = 0

    def reset(self):
        self.history.clear()
        self.velocities.clear()

    def update(self, faces, now, body_descriptors=None):
        body_descriptors = body_descriptors or {}
        self.history = {
            key: value for key, value in self.history.items() if now - value[2] <= self.max_gap
        }
        self.velocities = {
            key: value for key, value in self.velocities.items() if key in self.history
        }
        candidates = []
        for index, face in enumerate(faces):
            for key, (box, embedding, _seen, previous_body) in self.history.items():
                similarity = float(face.embedding @ embedding)
                iou = overlap(face.bbox, box)
                body = body_descriptors.get(index)
                same_body = (
                    body is not None
                    and previous_body is not None
                    and float(body @ previous_body) >= 0.8
                )
                centre_distance = np.linalg.norm(
                    np.array([(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2])
                    - np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                )
                motion_bound = max(face.bbox[2] - face.bbox[0], box[2] - box[0]) * 3
                if same_body and similarity >= 0.60 and centre_distance <= motion_bound:
                    candidates.append(
                        (0.5 * similarity + 0.5 * float(body @ previous_body), index, key)
                    )
                elif similarity >= 0.60 and iou >= 0.15:
                    candidates.append((0.5 * iou + 0.5 * similarity, index, key))
                elif self.max_motion_widths and similarity >= 0.75:
                    # Sparse recordings can move more than one face width between frames.
                    a = np.array(
                        [(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2]
                    )
                    b = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    radius = self.max_motion_widths * max(
                        face.bbox[2] - face.bbox[0], box[2] - box[0]
                    )
                    distance = float(np.linalg.norm(a - b))
                    # The embedding remains strongly gated (>= .75). Geometry
                    # ranks plausible matches instead of nullifying a strong
                    # match simply because sparse frames have no overlap.
                    widths = (face.bbox[2] - face.bbox[0], box[2] - box[0])
                    size_ratio = max(widths) / max(min(widths), 1)
                    if radius > 0 and distance <= radius and size_ratio <= 2:
                        velocity = self.velocities.get(key, np.zeros(2))
                        predicted = b + velocity * min(max(now - _seen, 0), 0.5)
                        predicted_distance = min(float(np.linalg.norm(a - predicted)), radius)
                        candidates.append(
                            (0.8 * similarity + 0.2 * (1 - predicted_distance / radius), index, key)
                        )
        assignments = {}
        used = set()
        for score, index, key in sorted(candidates, reverse=True):
            if score < 0.55 or index in assignments or key in used:
                continue
            rivals = [
                s for s, i, k in candidates if (i == index or k == key) and (i, k) != (index, key)
            ]
            if rivals and score - max(rivals) < 0.08:
                continue
            assignments[index] = key
            used.add(key)
        for index, face in enumerate(faces):
            if index not in assignments:
                self.next_id += 1
                assignments[index] = f"face-track-{self.next_id}"
            key = assignments[index]
            previous = self.history.get(key)
            if previous is not None and now > previous[2]:
                centre = np.array(
                    [(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2]
                )
                box = previous[0]
                old_centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                velocity = (centre - old_centre) / (now - previous[2])
                # Limit extrapolation for abrupt detector-box jumps.
                max_speed = max(face.bbox[2] - face.bbox[0], 1) * 8
                speed = float(np.linalg.norm(velocity))
                self.velocities[key] = velocity * min(1, max_speed / max(speed, 1))
            self.history[key] = (
                face.bbox,
                face.embedding,
                now,
                body_descriptors.get(index),
            )
        return assignments

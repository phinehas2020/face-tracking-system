"""Same-frame RF-DETR person proposals for the counting camera."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..gate_motion import strip_pixels
from ..live_counting import PassageCalibration
from .continuity import overlap
from .contracts import Detection


class RFPersonDetector:
    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(path), sess_options=options,
                                         providers=["CPUExecutionProvider"])
        inp, = self.model.get_inputs()
        self.input_name = inp.name
        if inp.shape != [1, 3, 704, 704] or not {"dets", "labels"}.issubset(
            {o.name for o in self.model.get_outputs()}
        ):
            raise ValueError("Expected a static 704px RF-DETR-Large detection export")
        self.mean = np.array([.485, .456, .406], dtype=np.float32)
        self.std = np.array([.229, .224, .225], dtype=np.float32)

    def detect(self, image):
        import cv2

        h, w = image.shape[:2]
        rgb = cv2.resize(image, (704, 704), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        norm = (rgb.astype(np.float32) / 255 - self.mean) / self.std
        boxes, logits = self.model.run(["dets", "labels"], {
            self.input_name: np.ascontiguousarray(norm.transpose(2, 0, 1)[None])
        })
        # COCO person is index 1 in this checkpoint. YOLO uses index 0.
        scores = 1 / (1 + np.exp(-np.clip(logits[0, :, 1], -60, 60)))
        out = []
        for idx in np.flatnonzero(scores >= .25):
            cx, cy, bw, bh = boxes[0, idx]
            xyxy = np.clip([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], 0, 1) * [w, h, w, h]
            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                out.append(Detection(tuple(map(float, xyxy)), float(scores[idx]), 0))
        return sorted(out, key=lambda d: -d.confidence)


def same_person_box(a, b):
    if overlap(a, b) >= .4:
        return True
    # Strong containment also occurs when one model sees only the upper body.
    # Require aligned centers horizontally to keep adjacent walkers distinct.
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aw, bw = a[2] - a[0], b[2] - b[0]
    area = min(aw * (a[3] - a[1]), bw * (b[3] - b[1]))
    return (area > 0 and iw * ih >= .82 * area
            and abs((a[0]+a[2]) - (b[0]+b[2])) / 2 <= .25 * min(aw, bw))


class SecondLook:
    def __init__(self, model, name):
        self.model = model
        self.metrics = {"model": Path(name).name, "device": "cpu", "completed": 0, "errors": 0,
                        "additional_boxes": 0, "strengthened_boxes": 0,
                        "confirmed_low_score": 0, "last_reason": None, "last_ms": None}
        self.reset()

    def reset(self):
        self.prev = self.pixel_key = self.coords = None
        self.weak = []
        self.last_run = -float("inf")
        self.last_count = self.followups = 0

    def _motion(self, frame, gate):
        if gate is None:
            return False
        h, w = frame.shape[:2]
        key = (w, h, gate)
        if self.pixel_key != key:
            self.pixel_key, self.prev = key, None
            try:
                self.coords = strip_pixels(PassageCalibration.from_calibration(gate), w, h)
            except ValueError:
                self.coords = None
        if self.coords is None:
            return False
        pixels = frame[self.coords].astype(np.int16)
        prev, self.prev = self.prev, pixels
        if prev is None:
            return False
        changed = (np.abs(pixels - prev).max(axis=2) > 32).mean(axis=1)
        return np.count_nonzero(changed >= .2) >= 2

    def review(self, frame, detections, now, gate, roi):
        moving = self._motion(frame, gate)
        dropped = len(detections) < self.last_count
        self.last_count = len(detections)
        elapsed = now - self.last_run
        if self.followups and elapsed >= .1:
            reason = "confirm additional box"
        elif (moving or dropped or len(detections) >= 2) and elapsed >= .6:
            reason = "gate motion" if moving else "crowded view"
            if dropped and not moving:
                reason = "detection count fell"
        elif elapsed >= (2 if detections else 5):
            reason = "periodic check"
        else:
            return detections
        self.last_run = now
        started = time.monotonic()
        try:
            candidates = self.model.detect(frame)
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = type(exc).__name__
            self.followups = 0
            return detections
        # All boxes are from this exact frame. Only novel proposals reach association.
        out = list(detections)
        previous_weak = self.weak if elapsed <= 1.2 else []
        self.weak = []
        h, w = frame.shape[:2]
        for candidate in candidates:
            a = candidate.bbox
            if roi and not (roi[0] <= (a[0]+a[2])/(2*w) <= roi[2] and
                            roi[1] <= (a[1]+a[3])/(2*h) <= roi[3]):
                continue
            matches = [i for i, existing in enumerate(out) if same_person_box(a, existing.bbox)]
            if candidate.confidence < .5:
                if matches and any(out[i].confidence >= candidate.confidence for i in matches):
                    continue
                self.weak.append(a)
                corroboration = [b for b in previous_weak if same_person_box(a, b)]
                if len(corroboration) != 1:
                    continue
                previous_weak.remove(corroboration[0])
                self.metrics["confirmed_low_score"] += 1
            if not matches:
                out.append(candidate)
            elif len(matches) == 1 and out[matches[0]].confidence < .35:
                # Use the independent RF score with its own box for a weak YOLO observation.
                out[matches[0]] = candidate
                self.metrics["strengthened_boxes"] += 1
        added = len(out) - len(detections)
        self.followups = max(0, self.followups-1) if reason == "confirm additional box" else (
            2 if added or self.weak else 0
        )
        self.metrics["completed"] += 1
        self.metrics["additional_boxes"] += added
        self.metrics.update(last_reason=reason, last_ms=round((time.monotonic()-started)*1000, 1))
        self.metrics.pop("last_error", None)
        return out

"""Bounded asynchronous face analysis. Counting never waits for face inference."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaceRequest:
    frame: Any
    tracks: list
    hints: dict
    sequence: int
    generation: int
    captured_at: float
    observed_at: str


@dataclass(frozen=True)
class FaceResult:
    request: FaceRequest
    detections: list
    faces: dict
    face_keys: dict
    latency_ms: float
    epoch: int = 0


class AsyncFaceWorker:
    def __init__(self, encoder_factory, max_age=2.5):
        self.encoder_factory = encoder_factory
        self.max_age = max_age
        self.condition = threading.Condition()
        self.pending = None
        self.results = deque(maxlen=12)
        self.metrics = {
            "submitted": 0,
            "completed": 0,
            "accepted": 0,
            "expired_before_inference": 0,
            "expired_on_delivery": 0,
            "overflow": 0,
        }
        self.stopped = False
        self.epoch = 0
        self.status = "loading"
        self.error = None
        self.provider = None
        self.fallback_reason = None
        self.pose_gate = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="live-face-analysis")

    def start(self):
        self.thread.start()

    def submit(self, request):
        with self.condition:
            self.metrics["submitted"] += 1
            self.pending = (request, self.epoch)  # One latest frame; no stale backlog.
            self.condition.notify()

    def take(self, generation, now):
        with self.condition:
            while self.results:
                result = self.results.popleft()
                if result.epoch != self.epoch or result.request.generation != generation:
                    continue
                if now - result.request.captured_at > self.max_age:
                    self.metrics["expired_on_delivery"] += 1
                    continue
                self.metrics["accepted"] += 1
                return result
        return None

    def reset(self):
        with self.condition:
            self.epoch += 1
            self.pending = None
            self.results.clear()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def _run(self):
        from .vision.adapters import associate_faces
        from .vision.continuity import FaceContinuity

        try:
            encoder = self.encoder_factory()
            self.provider = getattr(encoder, "selected_provider", "cpu")
            self.fallback_reason = getattr(encoder, "fallback_reason", None)
            self.pose_gate = getattr(encoder, "pose_gate", None)
            self.status = "ready"
            continuity = FaceContinuity(max_motion_widths=3)
            generation = None
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.stopped)
                    if self.stopped:
                        return
                    (request, epoch), self.pending = self.pending, None
                if time.monotonic() - request.captured_at > self.max_age:
                    self.metrics["expired_before_inference"] += 1
                    continue
                if generation != (request.generation, epoch):
                    continuity.reset()
                    generation = (request.generation, epoch)
                started = time.monotonic()
                detections = encoder.detect(
                    request.frame, person_boxes=[track.bbox for track in request.tracks]
                )
                self.provider = getattr(encoder, "selected_provider", "cpu")
                self.fallback_reason = getattr(encoder, "fallback_reason", None)
                self.pose_gate = getattr(encoder, "pose_gate", None)
                faces = associate_faces(detections, request.tracks, request.hints)
                keys = continuity.update(detections, request.captured_at)
                result = FaceResult(
                    request, detections, faces, keys, (time.monotonic() - started) * 1000, epoch
                )
                with self.condition:
                    if not self.stopped and epoch == self.epoch:
                        self.metrics["completed"] += 1
                        if len(self.results) == self.results.maxlen:
                            self.metrics["overflow"] += 1
                        self.results.append(result)
        except Exception as exc:
            self.status = "error"
            self.error = f"Face analysis unavailable ({type(exc).__name__})"

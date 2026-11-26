"""Aligned face crops for a shadow recognizer, saved beside the event.

The live recognizer hands every face with a readable pose to ``offer``, including
the turned faces its own pose gate rejects. A separate process embeds the crops with
other recognizers. Nothing here may slow or stop live recognition: offers are
rate-limited per face, the queue and the folder are bounded, and errors are counted.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

FOLDER = "face-shadow"
LINK_IOU = 0.35
LINK_SECONDS = 1.0


def _iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class FaceSpool:
    def __init__(
        self,
        directory: Path,
        camera: str,
        max_bytes: int,
        *,
        per_track: int = 24,
        min_gap: float = 0.4,
        flush_faces: int = 48,
        flush_seconds: float = 20.0,
        start_writer: bool = True,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.max_bytes = max_bytes
        self.per_track, self.min_gap = per_track, min_gap
        self.flush_faces, self.flush_seconds = flush_faces, flush_seconds
        self.boot = f"{int(time.time()):x}"
        self.calls = self.next_track = 0
        self.tracks: list[dict] = []
        self.pending: deque = deque(maxlen=512)
        self.condition = threading.Condition()
        self.stopped = False
        self.metrics = {
            "saved": 0,
            "skipped_rate_limit": 0,
            "skipped_full": 0,
            "errors": 0,
            # The cap covers every camera's crops in this folder.
            "bytes": sum(path.stat().st_size for path in self.directory.glob("chunk-*.npz")),
        }
        self.thread = threading.Thread(target=self._write, daemon=True, name="face-spool")
        if start_writer:
            self.thread.start()

    def offer(self, candidates: list[dict], now: float | None = None) -> None:
        """One call per analysed frame. Never raises."""
        try:
            self._offer(candidates, time.time() if now is None else now)
        except Exception:
            self.metrics["errors"] += 1

    def _offer(self, candidates: list[dict], now: float) -> None:
        from .recognizer import aligned_face

        self.calls += 1
        kept: list[dict] = []
        # The same face can arrive from the full frame and from a head crop.
        for item in sorted(candidates, key=lambda c: c["det"] * c["width"], reverse=True):
            twin = next((old for old in kept if _iou(item["bbox"], old["bbox"]) >= 0.4), None)
            if twin is None:
                kept.append(item)
            elif item["accepted"]:
                twin["accepted"] = True
        live = [track for track in self.tracks if now - track["t"] <= LINK_SECONDS]
        current, taken = [], set()
        for item in kept:
            best, best_iou = None, 0.0
            for index, track in enumerate(live):
                score = _iou(item["bbox"], track["box"]) if index not in taken else 0.0
                if score > best_iou:
                    best, best_iou = index, score
            if best is not None and best_iou >= LINK_IOU:
                taken.add(best)
                track = live[best]
            else:
                track = {"id": f"{self.camera}-{self.boot}-{self.next_track}", "n": 0, "saved": 0.0}
                self.next_track += 1
            track.update(box=item["bbox"], t=now)
            current.append(track)
            if track["n"] >= self.per_track or now - track["saved"] < self.min_gap:
                self.metrics["skipped_rate_limit"] += 1
                continue
            if self.metrics["bytes"] >= self.max_bytes or len(self.pending) == self.pending.maxlen:
                self.metrics["skipped_full"] += 1
                continue
            crop, points = aligned_face(item["image"], item["kps"])
            record = {
                "aligned": crop,
                "kps": points,
                "embedding": np.asarray(item["embedding"], dtype=np.float32),
                "meta": {
                    "camera": self.camera,
                    "track": track["id"],
                    "call": self.calls,
                    "t": round(now, 3),
                    "yaw": round(float(item["yaw"]), 2),
                    "pitch": round(float(item["pitch"]), 2),
                    "det": round(float(item["det"]), 4),
                    "width": round(float(item["width"]), 1),
                    "accepted": bool(item["accepted"]),
                },
            }
            track["n"] += 1
            track["saved"] = now
            with self.condition:
                self.pending.append(record)
                self.condition.notify()
        self.tracks = current + [track for track in live if all(track is not c for c in current)]

    def flush(self) -> None:
        with self.condition:
            records = list(self.pending)
            self.pending.clear()
        if not records:
            return
        try:
            name = f"chunk-{self.camera}-{int(records[0]['meta']['t'] * 1000)}"
            temporary = self.directory / f".{name}.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    aligned=np.stack([r["aligned"] for r in records]).astype(np.uint8),
                    kps=np.stack([r["kps"] for r in records]),
                    embedding=np.stack([r["embedding"] for r in records]),
                    meta=json.dumps([r["meta"] for r in records]),
                )
            size = temporary.stat().st_size
            temporary.replace(self.directory / f"{name}.npz")
            self.metrics["bytes"] += size
            self.metrics["saved"] += len(records)
        except Exception:
            self.metrics["errors"] += 1

    def _write(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.stopped)
                if not self.stopped:
                    # Gather a batch, but never hold a face longer than flush_seconds.
                    self.condition.wait_for(
                        lambda: self.stopped or len(self.pending) >= self.flush_faces,
                        timeout=self.flush_seconds,
                    )
                stopped = self.stopped
            self.flush()
            if stopped:
                return

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify()
        if self.thread.is_alive():
            self.thread.join(timeout=5)

"""TrackTrack association with two independent geometry checks on short ID handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .accelerator import TRACKER_LOCK


@dataclass(frozen=True)
class PairHistory:
    public_id: int
    raw_id: int
    seen: float
    observations: int
    color: np.ndarray


class Handoff:
    """An existing primary ID wins. Only a new ID can inherit an unoccupied history."""

    def __init__(self):
        self.next_id = 0
        self.handovers = 0
        self.primary = {}
        self.history = {}

    def update(self, tracks, witnesses, now, colors):
        self.primary = {k: v for k, v in self.primary.items() if now - v[1] <= 5}
        self.history = {k: v for k, v in self.history.items() if now - v.seen <= 1.2}
        current = {int(t[4]) for t in tracks}
        occupied = {self.primary[i][0] for i in current if i in self.primary}
        out = tracks.copy()
        for t in out:
            raw, idx = int(t[4]), int(t[-1])
            pair = tuple(w.get(idx) for w in witnesses)
            old = self.history.get(pair) if None not in pair else None
            color = colors[idx]
            if raw in self.primary:
                public = self.primary[raw][0]
            elif (old and old.public_id not in occupied and old.raw_id not in current
                  and old.observations >= 3
                  and self.primary.get(old.raw_id) == (old.public_id, old.seen)
                  and 0 < now - old.seen <= 1.2 and float(color @ old.color) >= .90):
                public = old.public_id
                self.primary.pop(old.raw_id, None)
                self.handovers += 1
            else:
                self.next_id += 1
                public = self.next_id
            self.primary[raw] = public, now
            occupied.add(public)
            if None not in pair:
                continuing = old and old.raw_id == raw and now - old.seen <= .25
                streak = old.observations + 1 if continuing else 1
                self.history[pair] = PairHistory(public, raw, now, streak, color)
            t[4] = public
        return out


class CrowdConsensus:
    def __init__(self, verify_handoffs=True):
        import ultralytics
        import yaml
        from ultralytics.trackers.track import TRACKER_MAP

        folder = Path(ultralytics.__file__).parent / "cfg" / "trackers"
        live = Path(__file__).with_name("live_tracker.yaml")
        self.trackers = []
        paths = [live]
        if verify_handoffs:
            paths.extend([folder / "bytetrack.yaml", folder / "ocsort.yaml"])
        from ultralytics.trackers.basetrack import BaseTrack

        # These constructors reset Ultralytics' process-global allocator. Each live
        # camera owns an allocator, including when its peer reconnects mid-track.
        with TRACKER_LOCK:
            saved = BaseTrack._count
            try:
                for path in paths:
                    cfg = yaml.safe_load(path.read_text())
                    cls = TRACKER_MAP[cfg["tracker_type"]]
                    self.trackers.append(cls(args=SimpleNamespace(**cfg)))
            finally:
                BaseTrack._count = saved
        self._last_id = 0
        self.handoff = Handoff()

    def update(self, boxes, now, colors):
        from ultralytics.trackers.basetrack import BaseTrack

        with TRACKER_LOCK:
            saved = BaseTrack._count
            BaseTrack._count = self._last_id
            try:
                primary, *checks = [tracker.update(boxes) for tracker in self.trackers]
            finally:
                self._last_id = BaseTrack._count
                BaseTrack._count = saved
        if not checks:
            return primary
        witnesses = [{int(t[-1]): int(t[4]) for t in result if t[-1] >= 0} for result in checks]
        return self.handoff.update(np.asarray(primary), witnesses, now, colors)


class CrowdPersonTracker:
    """Use one detector pass for all association methods on each frame."""

    def __init__(self, model, tracker, *, second_look=None, **kwargs):
        from ingress.vision.adapters import UltralyticsPersonDetector

        self.detector = UltralyticsPersonDetector(model, **kwargs)
        self.verify_handoffs = tracker == "crowd-consensus"
        self.second_look = second_look
        self.gate = None
        self.total_handovers = 0
        self.reset()

    @property
    def selected_device(self):
        return self.detector.selected_device

    @property
    def fallback_reason(self):
        return self.detector.fallback_reason

    def reset(self):
        self.association = CrowdConsensus(self.verify_handoffs)
        self.device = self.selected_device
        self.ages = {}
        if self.second_look:
            self.second_look.reset()

    def track(self, packet):
        from ultralytics.engine.results import Boxes

        from ingress.vision.continuity import TrackContinuity
        from ingress.vision.contracts import Track

        detections = self.detector.detect(packet)
        if self.device != self.selected_device:
            self.reset()
        now = packet.captured_monotonic_ns / 1e9
        if self.second_look is not None:
            detections = self.second_look.review(packet.image_bgr, detections, now, self.gate,
                                                 self.detector.roi)
        data = np.asarray([[*d.bbox, d.confidence, 0] for d in detections],
                          dtype=np.float32).reshape(-1, 6)
        boxes = Boxes(data, packet.image_bgr.shape[:2])
        colors = ([TrackContinuity.appearance(packet.image_bgr, d.bbox) for d in detections]
                  if self.verify_handoffs else [])
        before = self.association.handoff.handovers
        rows = self.association.update(boxes, now, colors)
        self.total_handovers += self.association.handoff.handovers - before
        ages = {int(t[4]): self.ages.get(int(t[4]), 0) + 1 for t in rows}
        self.ages = ages
        return [
            Track(f"{packet.camera_id}:{int(t[4])}", tuple(map(float, t[:4])), float(t[5]),
                  ages[int(t[4])]) for t in rows
        ]

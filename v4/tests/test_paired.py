import json
import time

import numpy as np
import pytest
from ingress.paired import PairedSession
from ingress.store import EventStore
from ingress.vision.contracts import Track


def test_both_streams_advance_but_only_side_counts(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from ingress.vision import adapters

    path = tmp_path / "fixture.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    w.release()
    rows = [{"frame": i, "clock": 1 + i / 10} for i in range(10)]
    cfg = tmp_path / "paired.json"
    cfg.write_text(
        json.dumps(
            {
                "paired_replay": {
                    "views": [
                        {"role": r, "path": str(path), "clock_index": rows}
                        for r in ["front", "side"]
                    ],
                    "start_clock": 1,
                    "duration_seconds": 0.9,
                    "front_to_side": np.eye(3).tolist(),
                    "synchronization_note": "Synthetic synchronized fixtures",
                    "calibration": {
                        "axis": "horizontal",
                        "position": 0.5,
                        "require_body_motion": False,
                    },
                }
            }
        )
    )

    class Tracker:
        def __init__(self, *args):
            pass

        def track(self, packet):
            y = 16 if packet.sequence < 5 else 38
            return [Track("one", (20, 0, 40, y), 0.9, packet.sequence + 1)]

    class Encoder:
        def detect(self, frame):
            return []

    monkeypatch.setattr(adapters, "UltralyticsPersonTracker", Tracker)
    monkeypatch.setattr(adapters, "LiveFaceEncoder", Encoder)
    store = EventStore(tmp_path / "data")
    store.initialize()
    session = PairedSession(store, cfg, "unused.pt")
    session.start()
    try:
        deadline = time.monotonic() + 8
        while not session.replay_complete and not session.error and time.monotonic() < deadline:
            time.sleep(0.02)
        assert session.error is None
        assert session.replay_complete
        d = session.snapshot()
        assert d["frames_processed"] == d["frames_expected"] == 20
        assert d["views"]["front"]["frame"] == d["views"]["side"]["frame"] == 9
        assert d["counts"] == {"in": 1, "out": 0}
        assert d["comparison"]["side_only_entries"] == 1
        assert d["comparison"]["cross_view_assisted_entries"] == 0
        assert session.report_path.exists()
    finally:
        session.stop()

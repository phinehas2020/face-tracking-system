import json
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ingress.live import LiveCalibration, live_router
from ingress.recorded import RecordedSession, ReplayClip
from ingress.store import EventStore


def make_session(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    config = tmp_path / "replay.json"
    config.write_text(
        json.dumps(
            {
                "replay_clips": [
                    {
                        "path": str(path),
                        "label": "Fixture crowd",
                        "camera": "station-01-counter",
                        "station": "Station 1",
                        "start_seconds": 0.5,
                        "duration_seconds": 0.4,
                    }
                ],
                "playback_speed": 4,
            }
        )
    )
    store = EventStore(tmp_path / "data")
    store.initialize()
    return RecordedSession(store, config, "unused.pt")


def test_replay_seek_eof_and_source_isolation(tmp_path, monkeypatch):
    session = make_session(tmp_path)

    # Exercise actual encoded input and pacing, without models or camera access.
    def consume(self):
        self.face_status = "ready"
        while not self.stop_event.wait(0.005):
            with self.lock:
                if self.frame_sequence > self.processed_sequence:
                    self.processed_sequence = self.frame_sequence
                    self.frames_processed += 1
                    self.processed_time = self.frame_time
                    self.jpeg = b"final-frame"

    monkeypatch.setattr(RecordedSession, "_process", consume)
    session.start()
    deadline = time.monotonic() + 4
    while not session.replay_complete and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        assert session.replay_complete
        assert session.frame_sequence == 4
        assert 0.5 <= session.source_info["position_seconds"] <= 0.9
        assert session.event_id != "single-camera-test"
        assert session.snapshot()["status"] == "complete"
        assert session.snapshot()["clips"][0]["station"] == "Station 1"
        assert (
            session.store.data_root / "events" / session.event_id / "replay-report.json"
        ).is_file()
        app = FastAPI()
        session.calibrate(LiveCalibration(position=0.6, require_body_motion=True))
        saved = json.loads(session.config_path.read_text())
        assert saved["replay_clips"][0]["calibration"]["require_body_motion"] is True
        app.state.live = session
        app.include_router(live_router())
        session.processed_time -= 30
        with TestClient(app) as client:
            assert client.get("/api/live/frame.jpg").status_code == 200
            restarted = client.post("/api/live/replay/start")
            assert restarted.status_code == 200
            assert restarted.json()["event_id"] != session.event_id
            assert app.state.live.clips[0].calibration.position == 0.6
            assert client.get("/api/live/status").json()["counts"] == {"in": 0, "out": 0}
            app.state.live.stop()
    finally:
        session.stop()
    fresh = RecordedSession(session.store, session.config_path, "unused.pt")
    assert fresh.event_id != session.event_id


def test_replay_validation(tmp_path):
    with pytest.raises(ValueError):
        ReplayClip(
            path=tmp_path / "x",
            label="x",
            camera="station-01",
            station="1",
            start_seconds=-1,
            duration_seconds=30,
        )
    with pytest.raises(ValueError):
        ReplayClip(
            path=tmp_path / "x",
            label="x",
            camera="station-01",
            station="1",
            start_seconds=0,
            duration_seconds=5000,
        )

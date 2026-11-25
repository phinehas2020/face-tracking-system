import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ingress.face_spool import FOLDER, FaceSpool
from ingress.live import LiveSession, live_router
from ingress.store import EventStore
from ingress.vision.adapters import LiveFaceEncoder

pytest.importorskip("insightface")

KPS = np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]], dtype=np.float32)


def candidate(left=100.0, accepted=True, det=0.9):
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1
    return {
        "image": np.full((400, 600, 3), 90, dtype=np.uint8),
        "kps": KPS + (left, 100),
        "bbox": (left, 100.0, left + 112, 212.0),
        "yaw": 40.0,
        "pitch": 5.0,
        "det": det,
        "width": 112.0,
        "embedding": embedding,
        "accepted": accepted,
    }


def test_spool_follows_a_face_limits_its_rate_and_writes_a_readable_chunk(tmp_path):
    spool = FaceSpool(tmp_path, "live-facing", 10**9, start_writer=False)
    # Two reports of one face in one scan are one face; a second face is separate.
    spool.offer(
        [candidate(det=0.9, accepted=False), candidate(101, det=0.8), candidate(400)], now=10.0
    )
    spool.offer([candidate(110)], now=10.1)  # same face, too soon
    spool.offer([candidate(120)], now=10.6)  # same face, later
    spool.offer([candidate(120)], now=20.0)  # after a gap it is a new followed face
    spool.flush()
    assert spool.metrics["saved"] == 4 and spool.metrics["skipped_rate_limit"] == 1
    assert spool.metrics["errors"] == 0 and spool.metrics["bytes"] > 0
    (path,) = tmp_path.glob("chunk-live-facing-*.npz")
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    assert data["aligned"].shape == (4, 112, 112, 3) and data["aligned"].dtype == np.uint8
    assert data["kps"].shape == (4, 5, 2) and data["embedding"].shape == (4, 512)
    assert np.allclose(data["kps"][0], KPS / 112, atol=0.02)  # already in the aligned position
    tracks = [row["track"] for row in meta]
    assert tracks[0] == tracks[2] and len(set(tracks)) == 3
    assert meta[0]["accepted"] is True  # either report passing the live gate counts
    assert meta[0]["call"] == meta[1]["call"] == 1 and meta[0]["yaw"] == 40.0
    assert not list(tmp_path.glob(".*"))


def test_spool_stops_at_its_size_cap_and_survives_bad_input(tmp_path):
    spool = FaceSpool(tmp_path, "live-side", 1, start_writer=False)
    spool.offer([candidate()], now=1.0)
    spool.flush()
    spool.offer([candidate()], now=5.0)
    assert spool.metrics["saved"] == 1 and spool.metrics["skipped_full"] == 1
    spool.offer([{"bbox": None}], now=6.0)
    assert spool.metrics["errors"] == 1
    # A restart counts what is already on disk against the cap.
    assert FaceSpool(tmp_path, "live-side", 1, start_writer=False).metrics["bytes"] > 0


def test_encoder_taps_turned_faces_the_pose_gate_rejects():
    def face(box, pose):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1
        return SimpleNamespace(
            bbox=np.asarray(box), det_score=0.95, normed_embedding=embedding, kps=KPS, pose=pose
        )

    seen = []
    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    encoder.tap = seen.append
    encoder.app = SimpleNamespace(
        get=lambda image: [
            face((10, 10, 130, 130), (0, 0, 0)),
            face((200, 10, 320, 130), (0, 50, 0)),
        ]
    )
    assert len(encoder.detect(np.zeros((240, 400, 3), dtype=np.uint8))) == 1
    (batch,) = seen
    assert [item["accepted"] for item in batch] == [True, False]
    assert batch[1]["yaw"] == 50 and batch[1]["bbox"] == (200.0, 10.0, 320.0, 130.0)

    def broken(_batch):
        raise RuntimeError("observer failed")

    encoder.tap = broken
    assert len(encoder.detect(np.zeros((240, 400, 3), dtype=np.uint8))) == 1


def test_shadow_endpoint_serves_the_report_only_when_present(tmp_path):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"host": "camera.invalid", "face_shadow": True}))
    store = EventStore(tmp_path / "data")
    store.initialize()
    session = LiveSession(store, config, "unused")
    session.start(start_workers=False)
    app = FastAPI()
    app.state.live = session
    app.include_router(live_router())
    client = TestClient(app)
    assert client.get("/api/live/shadow").json() == {"enabled": True, "report": None}
    folder = store._event_directory(session.event_id) / FOLDER
    (folder / "report.json").write_text(json.dumps({"faces": 3}))
    assert client.get("/api/live/shadow").json()["report"] == {"faces": 3}
    session.stop()

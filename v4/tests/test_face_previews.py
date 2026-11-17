import json

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ingress.live import LiveSession, live_router
from ingress.store import EventStore


def setup_session(tmp_path):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"host": "camera.invalid"}))
    store = EventStore(tmp_path / "data")
    store.initialize()
    session = LiveSession(store, config, "unused")
    session.start(start_workers=False)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1
    for _ in range(3):
        face = session.face_registry.observe("person", embedding, 0.95, {"person"})
    return session, config, face.identity_id


def jpeg(value):
    return cv2.imencode(".jpg", np.full((112, 112, 3), value, dtype=np.uint8))[1].tobytes()


def test_thumbnail_survives_restart_and_is_served(tmp_path):
    session, config, identity = setup_session(tmp_path)
    data = jpeg(120)
    session._save_face_preview(identity, data, 0.9)
    restored = LiveSession(session.store, config, "unused")
    restored.start(start_workers=False)
    assert restored.face_previews[identity] == data
    assert restored.snapshot()["recognition"]["visitors"][0]["has_preview"]
    app = FastAPI()
    app.state.live = restored
    app.include_router(live_router())
    with TestClient(app) as client:
        result = client.get(f"/api/live/faces/{identity}/preview.jpg")
        assert result.status_code == 200 and result.content == data
        assert result.headers["content-type"] == "image/jpeg"
    assert session.store.db_path.stat().st_mode & 0o777 == 0o600


def test_best_preview_is_retained_and_event_purge_removes_it(tmp_path):
    session, _, identity = setup_session(tmp_path)
    session._save_face_preview(identity, jpeg(100), 0.8)
    session._save_face_preview(identity, jpeg(200), 0.6)
    assert session.store.load_face_previews(session.event_id)[0][identity] == jpeg(100)
    session._save_face_preview(identity, jpeg(220), 0.95)
    assert session.store.load_face_previews(session.event_id)[0][identity] == jpeg(220)
    assert session.store.purge_plan(session.event_id)["database_records"]["face_previews"] == 1
    session.store.purge_event(session.event_id, f"PURGE {session.event_id}", "pytest")
    assert session.store.load_face_previews(session.event_id) == ({}, {})


def test_rejects_unrelated_identity_and_invalid_images(tmp_path):
    session, _, identity = setup_session(tmp_path)
    with pytest.raises(ValueError):
        session.store.save_face_preview("other-event", identity, jpeg(100), 0.8)
    with pytest.raises(ValueError):
        session.store.save_face_preview(session.event_id, identity, b"invalid", 0.8)

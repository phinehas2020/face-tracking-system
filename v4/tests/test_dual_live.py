import json
import subprocess
import sys
import threading
import time

from ingress.dual_live import DualLiveSession
from ingress.store import EventStore


def station(tmp_path):
    config = tmp_path / "camera.json"
    config.write_text(
        json.dumps({"host": "side.invalid", "facing_camera": {"host": "front.invalid"}})
    )
    store = EventStore(tmp_path / "data")
    store.initialize()
    return DualLiveSession(store, config, "unused")


def test_roles_shared_templates_and_no_facing_passages(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    session = station(tmp_path)
    session.start()
    assert session.facing.face_registry.templates is session.face_registry.templates
    assert session.facing.face_registry.tracks is not session.face_registry.tracks
    session.facing._record("front-person", "in")
    session._record("side-person", "in")
    state = session.snapshot()
    assert state["counts"]["in"] == 1
    assert state["facing"]["count_passages"] is False
    assert state["station_status"] == "degraded"
    with session.store.connect() as db:
        assert (
            db.execute("SELECT role FROM cameras WHERE id='live-facing'").fetchone()[0]
            == "face_identity"
        )


def test_blocked_decoder_is_killed_and_replaced(tmp_path, monkeypatch):
    import ingress.live as live

    session = station(tmp_path)
    real_popen = subprocess.Popen
    children = []

    def stalled_worker(*args, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(live.subprocess, "Popen", stalled_worker)
    monkeypatch.setattr(live, "CAPTURE_STALL_SECONDS", 0.1)
    thread = threading.Thread(target=session._capture)
    thread.start()
    deadline = time.monotonic() + 4
    try:
        while len(children) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(children) >= 2
        assert children[0].poll() is not None
    finally:
        session.stop_event.set()
        thread.join(4)
    assert not thread.is_alive()
    assert all(child.poll() is not None for child in children)


def test_frames_and_pause_address_both_views(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ingress.live import live_router

    session = station(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda self: None)
        session.start()
    session.jpeg, session.facing.jpeg = b"side", b"facing"
    session.processed_time = session.facing.processed_time = time.monotonic()
    app = FastAPI()
    app.state.live = session
    app.include_router(live_router())
    client = TestClient(app)
    assert client.get("/api/live/frame.jpg").content == b"side"
    assert client.get("/api/live/frame.jpg?camera=facing").content == b"facing"
    assert client.post("/api/live/control", json={"paused": True}).status_code == 200
    assert session.paused and session.facing.paused


def test_cross_camera_proposal_persists_without_changing_passage_count(tmp_path, monkeypatch):
    import numpy as np

    session = station(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda self: None)
        session.start()
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    for _ in range(3):
        face = session.facing.face_registry.observe("front-face", vector, 0.95, {"front-face"})
    for stamp in [100.0, 100.1]:
        session.association.observe(
            "side", "side-body", stamp, embedding=vector, embedding_source="osnet"
        )
        session.association.observe(
            "facing", "front-body", stamp, embedding=vector, embedding_source="osnet"
        )
    session._record("side-body", "in", captured_at=100.2)
    assert session.snapshot()["recognition"]["unresolved_entries"] == 1
    assert session.association.attach_identity(
        "facing", "front-body", face.identity_id, "face_match", 100.0
    )
    session._resolve_cross_view(100.3)
    state = session.snapshot()
    assert state["counts"]["in"] == 1
    assert state["recognition"]["unresolved_entries"] == 0
    assert state["recent"][0]["identity_basis"] == "cross_view_appearance"
    session._resolve_cross_view(100.4)
    assert session.snapshot()["counts"]["in"] == 1


def test_unlinked_crossing_keeps_its_cross_view_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    session = station(tmp_path)
    session.start()
    for at in (10.0, 10.25):
        for view, key in (("side", "s1"), ("facing", "f1")):
            session.association.observe(
                view, key, at, embedding=(1.0, 0.0, 0.0), embedding_source="osnet"
            )
    session._record("s1", "in", captured_at=10.25)
    session._resolve_cross_view(11.0)
    session._resolve_cross_view(30.0)
    with session.store.connect() as db:
        evidence = json.loads(db.execute("SELECT evidence FROM passages").fetchone()[0])
    rejected = evidence["cross_view_rejected"]
    assert rejected["last_reject_reason"] == "face_identity_pending"
    assert rejected["best_candidate"] == "f1"
    assert rejected["facing_camera_id"] == "live-facing"
    assert rejected["appearance_score"] > 0.99
    assert evidence["identity_resolution"] is False

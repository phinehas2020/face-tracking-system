"""Side entry tally and per-station face gates (KP-RPE stations open them wider)."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from ingress.face_registry import FaceRegistry
from ingress.live import LiveCalibration, LiveSession, SideEntryGate
from ingress.store import EventStore
from ingress.vision.adapters import LiveFaceEncoder


def make_session(tmp_path, monkeypatch, **extra):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"host": "192.0.2.1", "username": "u", "password": "p", **extra}))
    store = EventStore(tmp_path / "data")
    store.initialize()
    monkeypatch.setattr(LiveSession, "_capture", lambda self: None)
    monkeypatch.setattr(LiveSession, "_process", lambda self: None)
    session = LiveSession(store, config, "unused.pt")
    session.start()
    return session


def walk(session, key, xs, y, start):
    now = datetime.now(UTC).isoformat(timespec="milliseconds")
    for index, x in enumerate(xs):
        session._update_side_entry(key, key, x, y, start + index * 0.1, now)


def test_side_entries_count_only_entry_direction_inside_the_band(tmp_path, monkeypatch):
    session = make_session(
        tmp_path, monkeypatch, side_entry_gate={"position": 0.5, "y_min": 0.6, "y_max": 1.0}
    )
    across = [0.2] * 3 + [0.8] * 4
    walk(session, "entering", across, 0.8, 10)
    walk(session, "leaving", list(reversed(across)), 0.8, 10)
    walk(session, "behind-the-booth", across, 0.52, 10)
    walk(session, "feet-below-frame", across, 1.0, 20)
    snapshot = session.snapshot()
    assert snapshot["side_entries"]["count"] == 2
    assert snapshot["side_entries"]["last_hour"] == 2
    assert snapshot["side_entries"]["gate"]["y_min"] == 0.6
    # The main passage count and its line are untouched.
    assert snapshot["counts"] == {"in": 0, "out": 0}
    session.calibrate(LiveCalibration(position=0.3))
    assert session.side_entry_counter.calibration.position == 0.5
    walk(session, "after-recalibration", across, 0.9, 40)
    assert session.snapshot()["side_entries"]["count"] == 3


def test_side_entries_are_off_unless_configured(tmp_path, monkeypatch):
    session = make_session(tmp_path, monkeypatch)
    assert session.side_entry_counter is None
    assert "side_entries" not in session.snapshot()


def test_side_entry_band_must_have_height():
    with pytest.raises(ValueError):
        SideEntryGate(y_min=0.7, y_max=0.6)


def unit(seed):
    vector = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def face(yaw=0.0, pitch=0.0, width=80.0, det=0.95):
    points = np.array([[20, 20], [44, 20], [32, 30], [23, 45], [41, 45]], dtype=np.float32)
    return SimpleNamespace(
        bbox=np.array([0, 0, width, width]),
        det_score=det,
        kps=points,
        normed_embedding=unit(1),
        pose=np.array([pitch, yaw, 0.0]),
    )


def encoder(candidate, **gates):
    value = LiveFaceEncoder.__new__(LiveFaceEncoder)
    for name, setting in gates.items():
        setattr(value, name, setting)
    value.app = SimpleNamespace(get=lambda image: [candidate])
    return value


def test_pose_gate_defaults_hold_and_a_station_can_widen_them():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    turned = face(yaw=30, pitch=25)
    assert not encoder(turned).detect(image)
    widened = encoder(turned, max_yaw_degrees=35.0, max_pitch_degrees=30.0)
    assert len(widened.detect(image)) == 1
    assert widened.pose_gate["max_yaw_degrees"] == 35.0
    assert not encoder(face(yaw=40), max_yaw_degrees=35.0, max_pitch_degrees=30.0).detect(image)


def test_small_face_detector_floor_is_configurable_and_counted():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    small = face(width=60, det=0.7)  # quality 0.375: the small-face branch applies
    default = encoder(small)
    assert not default.detect(image)
    assert default.pose_gate["rejected_small_face"] == 1
    assert len(encoder(small, small_face_min_det=0.6).detect(image)) == 1


def test_encoder_rejects_impossible_gates():
    with pytest.raises(ValueError):
        LiveFaceEncoder(max_yaw_degrees=0)


def correlated(similarity, count):
    common = unit(100)
    samples = []
    for index in range(count):
        noise = unit(200 + index)
        noise -= (noise @ common) * common
        noise /= np.linalg.norm(noise)
        samples.append(np.sqrt(similarity) * common + np.sqrt(1 - similarity) * noise)
    return samples


@pytest.mark.parametrize(("link", "state"), [(0.5, "collecting"), (0.4, "new")])
def test_cluster_link_decides_whether_moderately_similar_samples_enroll(
    tmp_path, monkeypatch, link, state
):
    session = make_session(tmp_path, monkeypatch)
    registry = FaceRegistry(session.store, session.event_id, cluster_link=link)
    result = None
    for index, sample in enumerate(correlated(0.45, 3)):
        result = registry.observe("track", sample, 0.8, {"track"}, now=10 + index)
    assert result.state == state


def test_cluster_link_must_be_a_similarity(tmp_path, monkeypatch):
    session = make_session(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        FaceRegistry(session.store, session.event_id, cluster_link=1.5)


def add_passages(session, directions):
    with session.store.transaction() as db:
        for index, direction in enumerate(directions):
            db.execute(
                "INSERT INTO passages (id,event_id,camera_id,lane,track_id,observed_at,direction,"
                "status) VALUES (?,?,?,?,?,?,?,?)",
                (f"p{index}", session.event_id, session.camera_id, "live", f"t{index}",
                 datetime.now(UTC).isoformat(timespec="milliseconds"), direction, "unresolved"),
            )


def test_entry_scale_publishes_scaled_entries_and_keeps_the_raw_count(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ingress.live import live_router

    session = make_session(
        tmp_path, monkeypatch, entry_count_scale=2.4,
        side_entry_gate={"position": 0.5, "y_min": 0.6, "y_max": 1.0},
    )
    add_passages(session, ["in"] * 10 + ["out"] * 3)
    walk(session, "entering", [0.2] * 3 + [0.8] * 4, 0.8, 10)
    snapshot = session.snapshot()
    assert snapshot["counts"] == {"in": 24, "out": 3}
    assert snapshot["counts_raw"] == {"in": 10, "out": 3}
    assert snapshot["side_entries"]["count"] == 2
    assert snapshot["side_entries"]["count_raw"] == 1
    app = FastAPI()
    app.include_router(live_router())
    app.state.live = session
    buckets = TestClient(app).get("/api/live/history").json()["buckets"]
    assert sum(bucket["in"] for bucket in buckets) == 24
    assert sum(bucket["out"] for bucket in buckets) == 3


def test_entry_scale_defaults_to_raw_counts(tmp_path, monkeypatch):
    session = make_session(tmp_path, monkeypatch)
    add_passages(session, ["in"] * 5)
    snapshot = session.snapshot()
    assert snapshot["counts"] == {"in": 5, "out": 0}
    assert "counts_raw" not in snapshot


@pytest.mark.parametrize("scale", [0.5, 11, float("nan"), "2"])
def test_entry_scale_must_be_a_sane_multiplier(tmp_path, monkeypatch, scale):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"host": "192.0.2.1", "entry_count_scale": scale}))
    store = EventStore(tmp_path / "data")
    store.initialize()
    with pytest.raises(ValueError, match="entry_count_scale"):
        LiveSession(store, config, "unused.pt")

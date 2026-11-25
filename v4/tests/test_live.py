import json
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ingress.face_registry import FaceRegistry
from ingress.live import CameraConfig, CrossingCounter, LiveCalibration, LiveSession, live_router
from ingress.store import EventStore
from ingress.vision.adapters import DetectedFace, associate_faces
from ingress.vision.contracts import Track


def test_crossings_jitter_return_trips_and_expiration():
    counter = CrossingCounter(LiveCalibration())
    emitted = []
    points = [0.2] * 3 + [0.49, 0.51] * 4 + [0.8] * 5 + [0.2] * 4 + [0.8] * 4
    for index, x in enumerate(points):
        direction = counter.update("person", (x, 0.7), 10 + index * 0.1)
        if direction:
            emitted.append(direction)
    assert emitted == ["in", "out", "in"]
    counter.expire(100)
    assert not counter.states
    assert counter.update("person", (0.2, 0.7), 100) is None


def test_horizontal_reversed_direction_and_two_people():
    counter = CrossingCounter(LiveCalibration(axis="horizontal", entry_positive=False))
    emitted = []
    for i, y in enumerate([0.8] * 3 + [0.2] * 3):
        for name in ("a", "b"):
            result = counter.update(name, (0.2, y), 10 + i * 0.1)
            if result:
                emitted.append((name, result))
    assert emitted == [("a", "in"), ("b", "in")]


def test_never_count_first_appearance_or_a_single_noisy_frame():
    counter = CrossingCounter(LiveCalibration())
    for index, x in enumerate([0.8] * 3 + [0.2] + [0.8] * 3):
        assert counter.update("person", (x, 0.5), 10 + index * 0.1) is None
    # A gap must not bridge missing footage into an invented passage.
    for index in range(3):
        assert counter.update("person", (0.2, 0.5), 30 + index * 0.1) is None


@pytest.fixture
def live(tmp_path, monkeypatch):
    config = tmp_path / "camera.json"
    config.write_text(
        json.dumps({"host": "192.0.2.1", "username": "test-user", "password": "test-password"})
    )
    store = EventStore(tmp_path / "data")
    store.initialize()
    # No real camera, model, or network can be reached in these tests.
    monkeypatch.setattr(LiveSession, "_capture", lambda self: None)
    monkeypatch.setattr(LiveSession, "_process", lambda self: None)
    value = LiveSession(store, config, "unused.pt")
    value.start()
    return value


def test_passage_persistence_and_no_identity_claims(live, monkeypatch):
    timestamps = iter([f"2026-09-15T17:00:0{i}+00:00" for i in range(3)])
    monkeypatch.setattr("ingress.live.utc_now", lambda: next(timestamps))
    live._record("track-1", "in")
    live._record("track-1", "out")
    live._record("track-1", "in")
    snapshot = live.snapshot()
    assert snapshot["counts"] == {"in": 2, "out": 1}
    assert snapshot["identity_resolution"] is False
    store = EventStore(live.store.data_root)
    summary = store.get_summary(live.event_id)
    assert summary["raw_passages"] == 3
    assert summary["unique_attendees"] == 0
    assert summary["unresolved"] == 3
    assert "test-password" not in json.dumps(snapshot)
    assert "test-password" not in json.dumps(store.get_cameras(live.event_id))
    assert "test-password" not in repr(live.config)


def test_live_api_freshness_calibration_and_pause(live):
    app = FastAPI()
    app.state.live = live
    app.include_router(live_router())
    with TestClient(app) as client:
        assert client.get("/api/live/frame.jpg").status_code == 503
        live.jpeg = b"jpeg-fixture"
        live.processed_time = time.monotonic()
        live.status = "live"
        response = client.get("/api/live/frame.jpg")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        live.processed_time -= 4
        assert client.get("/api/live/frame.jpg").status_code == 503
        assert client.get("/api/live/status").json()["status"] == "disconnected"
        assert client.put("/api/live/calibration", json={"position": 2}).status_code == 422
        response = client.put(
            "/api/live/calibration",
            json={
                "axis": "horizontal",
                "position": 0.6,
                "entry_positive": False,
            },
        )
        assert response.status_code == 200
        saved = live.store.get_cameras(live.event_id)[0]["configuration"]
        assert saved["position"] == 0.6
        assert client.post("/api/live/control", json={"paused": True}).json()["paused"]
        assert live.paused is True
        assert not live.counter.states


def test_history_buckets_recent_passages_and_ignores_older_ones(live, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    stamps = iter(
        [
            (now - timedelta(hours=30)).isoformat(),
            (now - timedelta(minutes=25)).isoformat(),
            (now - timedelta(seconds=2)).isoformat(),
            (now - timedelta(seconds=1)).isoformat(),
        ]
    )
    monkeypatch.setattr("ingress.live.utc_now", lambda: next(stamps))
    live._record("track-old", "in")
    live._record("track-1", "in")
    live._record("track-2", "in")
    live._record("track-2", "out")
    app = FastAPI()
    app.state.live = live
    app.include_router(live_router())
    with TestClient(app) as client:
        body = client.get("/api/live/history?hours=2&bucket_minutes=10").json()
        assert body["bucket_seconds"] == 600
        assert len(body["buckets"]) == 12
        starts = [bucket["start"] for bucket in body["buckets"]]
        assert starts == sorted(starts)
        assert sum(bucket["in"] for bucket in body["buckets"]) == 2
        assert sum(bucket["out"] for bucket in body["buckets"]) == 1
        assert sum(bucket["linked"] for bucket in body["buckets"]) == 0
        assert client.get("/api/live/history?hours=0").status_code == 422
        assert client.get("/api/live/history?bucket_minutes=0").status_code == 422


def test_camera_url_encodes_credentials():
    config = CameraConfig("192.0.2.1", "user@name", "a!b:c#d")
    assert config.url == "rtsp://user%40name:a%21b%3Ac%23d@192.0.2.1:554/Streaming/Channels/101"


def vector(index=0):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def read_face(registry, key, embedding, active=None, quality=0.95):
    for _ in range(3):
        result = registry.observe(key, embedding, quality, active or {key})
    return result


def test_face_identity_survives_new_tracks_and_restart(live, monkeypatch):
    registry = live.face_registry
    first = read_face(registry, "visit-a", vector())
    assert first.identity_id
    assert registry.summary()["known_faces"] == 1
    assert registry.summary()["unique_attendees"] == 0
    timestamps = iter([f"2026-09-15T18:00:0{i}+00:00" for i in range(4)])
    monkeypatch.setattr("ingress.live.utc_now", lambda: next(timestamps))
    live._record("visit-a", "in")
    live._record("visit-a", "out")
    registry.clear_tracks()
    returning = read_face(registry, "visit-b", vector())
    assert returning.identity_id == first.identity_id
    assert returning.state == "matched"
    live._record("visit-b", "in")
    assert registry.summary()["unique_attendees"] == 1
    assert registry.summary()["repeat_entries"] == 1
    restored = FaceRegistry(live.store, live.event_id)
    again = read_face(restored, "after-restart", vector())
    assert again.identity_id == first.identity_id
    assert restored.summary()["known_faces"] == 1
    assert restored.summary()["repeat_entries"] == 1
    assert live.store.get_summary(live.event_id)["unique_attendees"] == 1


def test_distinct_people_and_uncertain_faces(live):
    registry = live.face_registry
    first = read_face(registry, "a", vector())
    ambiguous = 0.5 * vector() + np.sqrt(0.75) * vector(1)
    unknown = read_face(registry, "uncertain", ambiguous)
    assert unknown.state == "review"
    assert unknown.identity_id is None
    assert registry.summary()["known_faces"] == 1
    second = read_face(registry, "b", vector(1))
    assert second.identity_id != first.identity_id
    assert registry.summary()["known_faces"] == 2
    occupied = read_face(registry, "other-body", vector(), {"a", "other-body"})
    assert occupied.state == "review"
    assert occupied.identity_id is None
    assert registry.summary()["known_faces"] == 2


def test_face_quality_and_consistency_gate(live):
    registry = live.face_registry
    poor = read_face(registry, "a", vector(), quality=0.3)
    assert not poor.identity_id
    assert not poor.samples
    for i in range(8):
        mixed = registry.observe("mixed", vector(i % 2), 0.9, {"mixed"})
        assert mixed.identity_id is None
    assert registry.summary()["known_faces"] == 0
    with pytest.raises(ValueError):
        registry.observe("invalid", np.zeros(512), 0.9, {"invalid"})


def test_face_after_crossing_resolves_only_recent_same_track(live):
    live._record("pending", "in")
    live._record("unrelated", "in")
    assert live.face_registry.summary()["unresolved_entries"] == 2
    read_face(live.face_registry, "pending", vector())
    summary = live.face_registry.summary()
    assert summary["unique_attendees"] == 1
    assert summary["unresolved_entries"] == 1
    assert summary["repeat_entries"] == 0
    assert live.store.purge_plan(live.event_id)["database_records"]["face_templates"] == 1
    live.store.purge_event(live.event_id, f"PURGE {live.event_id}", "pytest")
    with live.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM face_templates").fetchone()[0] == 0


def test_face_body_association_rejects_overlap_and_multiple_faces():
    face = DetectedFace((10, 10, 30, 30), vector(), 0.9)
    body = Track("a", (0, 0, 100, 200), 0.9, 5)
    assert associate_faces([face], [body]) == {"a": face}
    other_body = Track("b", (0, 0, 100, 200), 0.9, 5)
    assert not associate_faces([face], [body, other_body])
    assert not associate_faces([face, face], [body])


def test_weak_pose_retains_confirmed_track_without_new_match(live):
    registry = live.face_registry
    original = read_face(registry, "body-a", vector()).identity_id
    weak = 0.45 * vector() + np.sqrt(1 - 0.45**2) * vector(1)
    state = registry.observe("body-a", weak, 0.9, {"body-a"})
    assert state.identity_id == original
    assert state.state == "tracked"
    assert len(state.samples) == 3
    live._record("body-a", "in")
    assert registry.summary()["unique_attendees"] == 1
    assert registry.summary()["known_faces"] == 1


def test_conflicting_face_quarantines_crossing_then_requires_consensus(live):
    registry = live.face_registry
    original = read_face(registry, "body-a", vector()).identity_id
    stamp = "2026-09-16T17:00:00+00:00"
    state = registry.observe("body-a", vector(1), 0.9, {"body-a"}, observed_at=stamp)
    assert state.state == "conflict"
    live._record("body-a", "in", stamp)
    assert registry.summary()["unresolved_entries"] == 1
    for _ in range(2):
        state = registry.observe("body-a", vector(1), 0.9, {"body-a"}, observed_at=stamp)
    assert state.identity_id and state.identity_id != original
    with live.store.connect() as db:
        row = db.execute("SELECT identity_id FROM passages").fetchone()
        assert row[0] == state.identity_id


def test_head_geometry_resolves_face_inside_two_overlapping_body_boxes():
    face = DetectedFace((45, 15, 65, 35), vector(), 0.9)
    owner = Track("owner", (25, 0, 85, 220), 0.9, 4)
    behind = Track("behind", (0, -100, 100, 200), 0.9, 4)
    assert associate_faces([face], [owner, behind]) == {"owner": face}


def test_short_gap_handover_and_simultaneous_tracks():
    pytest.importorskip("cv2")
    from ingress.vision.continuity import TrackContinuity

    continuity = TrackContinuity()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    first = continuity.update([Track("raw-a", (20, 10, 80, 180), 0.9, 1)], frame, 0)[0]
    returned = continuity.update([Track("raw-b", (22, 10, 82, 180), 0.9, 1)], frame, 0.2)[0]
    assert first.track_id == returned.track_id
    assert continuity.handovers == 1
    simultaneous = continuity.update(
        [
            Track("raw-b", (22, 10, 82, 180), 0.9, 1),
            Track("raw-c", (24, 10, 84, 180), 0.9, 1),
        ],
        frame,
        0.3,
    )
    assert len({t.track_id for t in simultaneous}) == 2
    far = continuity.update([Track("raw-d", (120, 10, 180, 180), 0.9, 1)], frame, 0.4)[0]
    assert far.track_id not in {t.track_id for t in simultaneous}


def test_smaller_faces_require_six_samples_and_stricter_match_margin(live):
    registry = live.face_registry
    for _ in range(5):
        state = registry.observe("small", vector(), 0.5, {"small"})
        assert state.identity_id is None
        assert state.required_samples == 6
    state = registry.observe("small", vector(), 0.5, {"small"})
    known = state.identity_id
    assert known
    registry.clear_tracks()
    borderline = 0.64 * vector() + np.sqrt(1 - 0.64**2) * vector(1)
    for _ in range(6):
        state = registry.observe("borderline", borderline, 0.5, {"borderline"})
    assert state.identity_id is None
    assert state.state == "review"
    assert registry.summary()["known_faces"] == 1
    for _ in range(6):
        state = registry.observe("returning", vector(), 0.5, {"returning"})
    assert state.identity_id == known


def test_pending_link_uses_video_clock_not_processing_delay(live):
    registry = live.face_registry
    source_time = "2025-11-28T10:00:00+00:00"
    live._record("delayed", "in", source_time)
    for i in range(3):
        registry.observe(
            "delayed", vector(), 0.95, {"delayed"}, observed_at=f"2025-11-28T10:00:0{i + 1}+00:00"
        )
    assert registry.summary()["unique_attendees"] == 1
    assert registry.summary()["unresolved_entries"] == 0


def test_isolated_bad_sample_does_not_orphan_an_entry(live):
    registry = live.face_registry
    live._record("entrant", "in", "2025-11-28T10:00:00+00:00")
    for i, embedding in enumerate([vector(), vector(1), vector(), vector()]):
        registry.observe(
            "entrant", embedding, 0.95, {"entrant"}, observed_at=f"2025-11-28T10:00:0{i + 1}+00:00"
        )
    assert registry.summary()["unique_attendees"] == 1
    assert registry.summary()["unresolved_entries"] == 0


def test_small_face_gate_requires_frontal_confident_landmarks():
    from types import SimpleNamespace

    from ingress.vision.adapters import LiveFaceEncoder

    encoder = LiveFaceEncoder.__new__(LiveFaceEncoder)
    points = np.array([[20, 20], [44, 20], [32, 30], [23, 45], [41, 45]])
    face = SimpleNamespace(
        bbox=np.array([0, 0, 80, 80]),
        det_score=0.95,
        kps=points,
        normed_embedding=vector(),
        pose=np.zeros(3),
    )
    encoder.app = SimpleNamespace(get=lambda image: [face])
    assert len(encoder.detect(np.zeros((100, 100, 3), dtype=np.uint8))) == 1
    face.kps = points.copy()
    face.kps[2, 0] = 46
    assert not encoder.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_box_height_jitter_does_not_count_without_body_motion():
    counter = CrossingCounter(LiveCalibration(axis="horizontal", require_body_motion=True))
    for i, y in enumerate([0.3] * 3 + [0.8] * 6 + [0.3] * 6):
        assert counter.update("body", (0.5, y), i * 0.1, body_anchor=(0.5, 0.1)) is None
    results = []
    for i in range(6):
        result = counter.update("body", (0.5, 0.8), 2 + i * 0.1, body_anchor=(0.5, 0.2))
        if result:
            results.append(result)
    assert results == ["in"]


def test_face_evidence_survives_temporary_missing_body(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from ingress.live import LiveSession
    from ingress.vision import adapters

    class BodyModel:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def track(self, frame):
            return [Track("raw", (0, 0, 60, 120), 0.9, 1)] if frame.sequence in [1, 4] else []

    class FaceModel:
        def detect(self, frame):
            return [DetectedFace((10, 10, 30, 30), vector(), 0.95)]

    monkeypatch.setattr(adapters, "UltralyticsPersonTracker", BodyModel)
    monkeypatch.setattr(adapters, "LiveFaceEncoder", FaceModel)
    monkeypatch.setattr(LiveSession, "_capture", lambda self: None)
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps({"host": "192.0.2.1", "username": "fixture", "password": "fixture"})
    )
    store = EventStore(tmp_path / "data")
    store.initialize()
    session = LiveSession(store, config, "unused")
    session.is_replay = True
    session.start()
    try:
        for sequence in range(1, 5):
            with session.lock:
                session.frame = np.zeros((200, 200, 3), dtype=np.uint8)
                session.frame_sequence = sequence
                session.frame_time = time.monotonic()
                session.analysis_time = sequence * 0.4
                session.frame_observed_at = f"2025-11-28T10:00:0{sequence}+00:00"
                session.generation = 1
            deadline = time.monotonic() + 3
            while session.processed_sequence < sequence and time.monotonic() < deadline:
                time.sleep(0.01)
            assert session.processed_sequence == sequence, session.error
            if sequence == 1:
                key = next(k for k in session.face_registry.tracks if ":track-" in k)
                session._record(key, "in", session.frame_observed_at)
        assert session.face_registry.summary()["known_faces"] == 1
        assert session.face_registry.summary()["unique_attendees"] == 1
        assert session.face_registry.summary()["unresolved_entries"] == 0
    finally:
        session.stop()


def test_global_face_assignment_resolves_jointly_clear_pairing():
    pytest.importorskip("scipy")
    faces = [
        DetectedFace((51, 20, 57, 28), vector(), 0.9),
        DetectedFace((59, 20, 65, 28), vector(1), 0.9),
    ]
    bodies = [Track("a", (0, 0, 100, 200), 0.9, 1), Track("b", (10, 0, 110, 200), 0.9, 1)]
    result = associate_faces(faces, bodies)
    assert result == {"a": faces[0], "b": faces[1]}


def test_face_body_binding_does_not_share_mutable_track_state(live):
    registry = live.face_registry
    face = read_face(registry, "face-a", vector())
    original = face.identity_id
    registry.bind_face_to_body("body-a", face, {"body-a"}, time.monotonic(), None)
    body = registry.tracks["body-a"]
    assert body is not face
    assert body.samples is not face.samples
    face.identity_id = None
    face.samples.clear()
    face.label = "Changed face track"
    assert body.identity_id == original
    assert len(body.samples) == 3
    assert body.label != "Changed face track"


def test_stronger_alternative_is_not_ignored_by_latched_identity(live):
    registry = live.face_registry
    first = read_face(registry, "a", vector()).identity_id
    second = read_face(registry, "b", vector(1)).identity_id
    query = (0.7 * vector() + 0.9 * vector(1)) / np.sqrt(0.7**2 + 0.9**2)
    state = registry.observe("a", query, 0.95, {"a"})
    assert state.identity_id == first
    assert state.state == "conflict"
    for _ in range(2):
        state = registry.observe("a", query, 0.95, {"a"})
    assert state.identity_id == second

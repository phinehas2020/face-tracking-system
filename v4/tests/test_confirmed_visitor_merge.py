import json

import numpy as np
import pytest
from ingress.face_registry import FaceRegistry
from ingress.live import LiveSession
from ingress.store import EventStore


def fixture(tmp_path):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"host": "fixture.invalid"}))
    store = EventStore(tmp_path / "data")
    store.initialize()
    live = LiveSession(store, config, "unused")
    live.start(start_workers=False)
    vectors = np.eye(512, dtype=np.float32)
    ids = []
    for i in range(3):
        for _ in range(3):
            face = live.face_registry.observe(f"track-{i}", vectors[i], 0.95, {f"track-{i}"})
        ids.append(face.identity_id)
    return store, live, ids, vectors


def test_confirmed_merge_preserves_views_and_counts_after_restart(tmp_path):
    store, live, ids, vectors = fixture(tmp_path)
    live._record("track-1", "in", observed_at="2026-10-08T12:00:00+00:00")
    live._record("track-2", "in", observed_at="2026-10-08T12:00:01+00:00")
    live._record("track-2", "out", observed_at="2026-10-08T12:00:02+00:00")
    result = store.merge_confirmed_visitors(
        live.event_id, ids[1], ids[2], "test human confirmation"
    )
    assert result["moved_passages"] == 1
    registry = FaceRegistry(store, live.event_id)
    assert len(registry.templates) == 2
    assert registry.identity_similarity(vectors[1], ids[2]) == pytest.approx(1)
    assert registry.identity_similarity(vectors[2], ids[2]) == pytest.approx(1)
    for _ in range(3):
        match = registry.observe("return", vectors[1], 0.95, {"return"})
    assert match.identity_id == ids[2]
    summary = registry.summary()
    assert summary["unique_attendees"] == 1
    assert summary["repeat_entries"] == 1
    assert live.snapshot()["counts"] == {"in": 2, "out": 1}
    assert store.purge_plan(live.event_id)["database_records"]["confirmed_face_variants"] == 1
    store.purge_event(live.event_id, f"PURGE {live.event_id}", "pytest")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM confirmed_face_variants").fetchone()[0] == 0


def test_labels_are_not_reused_after_a_merge_and_invalid_merge_is_atomic(tmp_path):
    store, live, ids, vectors = fixture(tmp_path)
    with pytest.raises(ValueError):
        store.merge_confirmed_visitors("other-event", ids[1], ids[2], "test")
    assert len(live.face_registry.summary()["visitors"]) == 3
    store.merge_confirmed_visitors(live.event_id, ids[1], ids[2], "test")
    registry = FaceRegistry(store, live.event_id)
    for _ in range(3):
        face = registry.observe("new", vectors[3], 0.95, {"new"})
    assert face.label == "Visitor 004"

"""Activity windows: capture, budgets, labelling and the counting scorecard."""

import json

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ingress import activity
from ingress.activity import ActivityRecorder, FrameRing, GateMotion
from ingress.live import LiveCalibration, live_router
from ingress.store import EventStore

EVENT = "activity-test"


def make_store(tmp_path) -> EventStore:
    store = EventStore(tmp_path)
    store.initialize()
    with store.transaction() as db:
        db.execute(
            "INSERT INTO events (id,title,event_date,status,source_label,created_at) "
            "VALUES (?, 'Test', '2026-11-18', 'live', 'test', '2026-11-18T12:00:00+00:00')",
            (EVENT,),
        )
    return store


def jpeg(value: int, size=(360, 640)) -> bytes:
    image = np.full((*size, 3), value, dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def recorder(store, **options):
    rings = {"side": FrameRing(seconds=30), "facing": FrameRing(seconds=30)}
    for step in range(120):  # twelve seconds at ten frames a second
        for view, ring in rings.items():
            ring.push(100 + step / 10, jpeg(step * 2 if view == "side" else 255 - step * 2))
    options = {"min_free_bytes": 0, "start_writer": False, **options}
    return ActivityRecorder(store, EVENT, "live-side", rings, **options), rings


def flush(rec):
    saved = []
    while not rec.jobs.empty():
        saved.append(rec.save(*rec.jobs.get()))
    return saved


def rows(store):
    with store.connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM activity_windows ORDER BY rowid")]


def test_ring_thins_to_its_rate_and_forgets_old_frames():
    ring = FrameRing(seconds=2, max_fps=5)
    for step in range(100):
        ring.push(step / 20, b"x")
    held = ring.window(0, 10)
    assert 10 <= len(held) <= 11
    assert held[0][0] >= 4.95 - 2 - 0.2
    assert all(b - a >= 0.199 for (a, _), (b, _) in zip(held, held[1:], strict=False))


def test_counted_window_keeps_a_strip_from_each_camera(tmp_path):
    store = make_store(tmp_path)
    rec, _ = recorder(store)
    rec.update(103.0, "2026-11-18T12:00:03.000+00:00", 1, False)
    rec.passage("p1", "in", 103.6)
    rec.update(104.0, None, 1, False)
    rec.passage("p2", "out", 104.2)
    rec.update(104.3, None, 0, False)
    assert rec.jobs.empty()  # still inside the quiet period
    rec.update(105.4, None, 0, False)
    (window_id,) = flush(rec)
    (row,) = rows(store)
    assert (row["cause"], row["system_in"], row["system_out"]) == ("passage", 1, 1)
    assert [p["direction"] for p in json.loads(row["passages"])] == ["in", "out"]
    strips = json.loads(row["strips"])
    assert set(strips) == {"side", "facing"}
    for view in strips:
        path = activity.strip_path(store, EVENT, window_id, view)
        sheet = cv2.imread(str(path))
        layout = strips[view]
        # 1.5 s before the first activity to 1 s after it went quiet, at six frames a second.
        assert 20 <= layout["frames"] <= 30 and len(layout["times"]) == layout["frames"]
        assert sheet.shape[1] == layout["columns"] * layout["cell"][0]
        assert 0 <= layout["times"][0] < 0.3
    assert activity.strip_path(store, EVENT, window_id, "../../ingress.db") is None
    assert rec.metrics["saved"] == 1 and rec.metrics["bytes"] > 0


def test_someone_standing_in_view_is_one_uncounted_window(tmp_path):
    store = make_store(tmp_path)
    rec, _ = recorder(store, max_seconds=3)
    for step in range(100):  # ten seconds of one person in view
        now = 101 + step / 10
        rec.update(now, None, 1, False)
        if step == 40:
            rec.passage("late", "in", now)  # they finally walk through
    rec.update(112.5, None, 0, False)
    assert len(flush(rec)) == 2
    assert rec.metrics["skipped_rate_limit"] >= 1
    first, second = rows(store)
    assert (first["cause"], first["system_in"]) == ("people", 0)
    # The crossing is kept even though it falls in a later stretch of the same visit.
    assert (second["cause"], second["system_in"]) == ("passage", 1)


def test_uncounted_windows_are_rate_limited_and_motion_alone_is_recorded(tmp_path):
    store = make_store(tmp_path)
    rec, _ = recorder(store, uncounted_per_hour=2)
    for burst in range(4):
        start = 101 + burst * 2.5
        rec.update(start, None, 0, True)
        rec.update(start + 1.2, None, 0, False)
    assert len(flush(rec)) == 2 and rec.metrics["skipped_rate_limit"] == 2
    assert {row["cause"] for row in rows(store)} == {"motion"}


def test_low_disk_skips_and_the_byte_budget_prunes_strips_but_keeps_labels(tmp_path):
    store = make_store(tmp_path)
    rec, _ = recorder(store, min_free_bytes=10**18)
    rec.update(103, None, 1, False)
    rec.passage("p", "in", 103.2)
    rec.update(104.5, None, 0, False)
    assert flush(rec) == [None] and rec.metrics["skipped_low_disk"] == 1 and not rows(store)

    rec, _ = recorder(store, max_bytes=1)
    ids = []
    for burst in range(3):
        start = 102 + burst * 3
        rec.update(start, f"2026-11-18T12:00:0{burst}.000+00:00", 1, False)
        rec.passage(f"p{burst}", "in", start + 0.2)
        rec.update(start + 1.5, None, 0, False)
        ids += flush(rec)
    assert len(rows(store)) == 3 and rec.metrics["pruned"] >= 4
    assert activity.strip_path(store, EVENT, ids[0], "side") is None
    assert activity.label_window(store, EVENT, ids[0], 1, 0, "tester")


def test_gate_motion_sees_change_on_the_counting_line():
    calibration = LiveCalibration(counting_mode="trajectory", axis="vertical", position=0.5)
    motion = GateMotion()
    still = np.zeros((360, 640, 3), dtype=np.uint8)
    moved = still.copy()
    moved[:, 280:360] = 200
    assert motion.update(still, calibration) is False
    assert motion.update(still, calibration) is False
    assert motion.update(moved, calibration) is True
    assert motion.update(moved, calibration) is False


def insert(store, window_id, cause, system, label=None):
    with store.transaction() as db:
        db.execute(
            "INSERT INTO activity_windows (id,event_id,camera_id,started_at,duration_seconds,"
            "cause,system_in,system_out,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (window_id, EVENT, "live-side", f"2026-11-18T12:00:{len(window_id):02d}+00:00", 2.0,
             cause, *system, "2026-11-18T12:00:00+00:00"),
        )
    if label:
        assert activity.label_window(store, EVENT, window_id, *label, "tester")


def test_scorecard_counts_misses_extras_and_unlabelled_windows(tmp_path):
    store = make_store(tmp_path)
    insert(store, "a", "passage", (1, 0), (1, 0))       # right
    insert(store, "bb", "passage", (2, 1), (1, 1))      # one extra entry
    insert(store, "ccc", "people", (0, 0), (0, 1))      # one missed exit
    insert(store, "dddd", "motion", (0, 0), (0, 0))     # door swing, correctly ignored
    insert(store, "eeeee", "passage", (1, 0), (0, 1))   # wrong direction: an extra and a miss
    insert(store, "ffffff", "passage", (1, 0))          # not reviewed
    card = activity.scorecard(store, EVENT)
    assert (card["windows"], card["labelled"], card["unlabelled"]) == (6, 5, 1)
    assert card["exact_windows"] == 2
    assert card["directions"]["in"] == {"counted": 4, "true": 2, "missed": 0, "extra": 2}
    assert card["directions"]["out"] == {"counted": 1, "true": 3, "missed": 2, "extra": 0}
    assert (card["counted"], card["true"], card["missed"], card["extra"]) == (5, 5, 2, 2)
    assert card["recall"] == 0.6 and card["precision"] == 0.6 and card["count_error"] == 0
    low, high = card["recall_interval_95"]
    assert low < 0.6 < high and low >= 0 and high <= 1
    assert card["by_cause"]["people"] == {"windows": 1, "counted": 0, "true": 1}
    assert activity.scorecard(make_store(tmp_path / "empty"), EVENT)["recall"] is None


class FakeSession:
    def __init__(self, store):
        self.store, self.event_id, self.activity = store, EVENT, None


def test_review_api_lists_serves_and_labels(tmp_path):
    store = make_store(tmp_path)
    rec, _ = recorder(store)
    rec.update(103, None, 1, False)
    rec.passage("p", "in", 103.2)
    rec.update(104.5, None, 0, False)
    (window_id,) = flush(rec)
    app = FastAPI()
    app.state.live = FakeSession(store)
    app.include_router(live_router())
    with TestClient(app) as client:
        listed = client.get("/api/live/activity").json()
        assert [w["id"] for w in listed["windows"]] == [window_id]
        assert "file" not in listed["windows"][0]["strips"]["side"]
        assert client.get(f"/api/live/activity/{window_id}/side.jpg").content[:2] == b"\xff\xd8"
        assert client.get(f"/api/live/activity/{window_id}/top.jpg").status_code == 404
        assert client.get("/api/live/activity?state=nope").status_code == 422
        bad = client.post(f"/api/live/activity/{window_id}/label", json={"entries": -1, "exits": 0})
        assert bad.status_code == 422
        card = client.post(
            f"/api/live/activity/{window_id}/label", json={"entries": 1, "exits": 0}
        ).json()
        assert card["labelled"] == 1 and card["recall"] == 1.0
        assert client.post("/api/live/activity/nope/label", json={"entries": 0, "exits": 0}
                           ).status_code == 404
        assert client.get("/api/live/activity").json()["windows"] == []
        assert client.get("/api/live/scorecard").json()["exact_windows"] == 1
        assert "Door review" in client.get("/api/live/review").text

from ingress.seed import DEMO_EVENT_ID, seed_demo
from ingress.store import EventStore


def test_seed_review_calibration_and_secure_purge(tmp_path):
    store = EventStore(tmp_path)
    store.initialize()
    seed_demo(store)

    summary = store.get_summary(DEMO_EVENT_ID)
    assert summary["unique_attendees"] == 3841
    assert len(summary["cameras"]) == 3

    candidate = store.resolve_candidate(DEMO_EVENT_ID, "candidate-001", "same", "pytest")
    assert candidate["status"] == "confirmed_same"
    assert store.get_summary(DEMO_EVENT_ID)["unresolved"] == 2

    camera = store.update_camera_calibration(
        DEMO_EVENT_ID,
        "cam-lane-a",
        [(0.1, 0.1), (0.9, 0.1), (0.8, 0.7)],
        ((0.2, 0.7), (0.8, 0.7)),
        "in",
    )
    assert camera["configuration"]["direction"] == "in"

    event_files = tmp_path / "events" / DEMO_EVENT_ID
    event_files.mkdir(parents=True)
    (event_files / "embedding.bin").write_bytes(b"ephemeral-biometric-template")
    result = store.purge_event(DEMO_EVENT_ID, f"PURGE {DEMO_EVENT_ID}", "pytest")
    assert result["status"] == "purged"
    assert not store.event_exists(DEMO_EVENT_ID)
    assert not event_files.exists()
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event_id = ?", (DEMO_EVENT_ID,)
        ).fetchone()[0] == 0

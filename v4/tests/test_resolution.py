import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from ingress import resolution
from ingress.app import create_app
from ingress.clustering_metrics import clustering_metrics
from ingress.config import Settings
from ingress.estimation import estimate_attendance
from ingress.evidence import load_tracklets, save_tracklet
from ingress.face_registry import FACE_MODEL, FaceRegistry
from ingress.store import EventStore

EVENT = "resolution-test"
CAMERA = "cam"
BASE = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


def at(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def unit(rng) -> np.ndarray:
    vector = rng.normal(size=512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def near(rng, base: np.ndarray, cosine: float) -> np.ndarray:
    """A unit vector with exactly the requested cosine to ``base``."""
    noise = unit(rng).astype(np.float64)
    noise -= (noise @ base) * base
    noise /= np.linalg.norm(noise)
    return (cosine * base + np.sqrt(1 - cosine**2) * noise).astype(np.float32)


def make_store(tmp_path, event=EVENT) -> EventStore:
    store = EventStore(tmp_path)
    store.initialize()
    with store.transaction() as db:
        db.execute(
            "INSERT INTO events (id,title,event_date,status,source_label,created_at) "
            "VALUES (?, 'Test', '2026-11-14', 'live', 'test', ?)",
            (event, at(0)),
        )
        db.execute(
            "INSERT INTO event_counters (event_id,unresolved,updated_at) VALUES (?,7,?)",
            (event, at(0)),
        )
        db.execute(
            "INSERT INTO cameras (id,event_id,name,role,source,status) "
            "VALUES (?,?,?,'passage_counter','test','live')",
            (CAMERA + event, event, CAMERA),
        )
    return store


def add_tracklet(store, rng, key, prototype, start, *, samples=3, length=2.0, event=EVENT):
    vectors = [near(rng, prototype, 0.97) for _ in range(samples)]
    save_tracklet(
        store, event, CAMERA + event, key, FACE_MODEL, vectors, [0.9] * samples,
        at(start), at(start + length), None, "collecting",
    )


def add_identity(store, identity, label, vector):
    with store.transaction() as db:
        db.execute(
            "INSERT INTO identities (id,event_id,created_at,updated_at,status,confidence,"
            "prototype_count) VALUES (?,?,?,?,'face_observed',0.9,3)",
            (identity, EVENT, at(0), at(0)),
        )
        db.execute(
            "INSERT INTO face_templates (identity_id,event_id,model,embedding,label,quality,"
            "samples,updated_at) VALUES (?,?,?,?,?,0.9,3,?)",
            (identity, EVENT, FACE_MODEL, vector.tobytes(), label, at(0)),
        )


def add_passage(store, name, track, identity=None, direction="in"):
    with store.transaction() as db:
        db.execute(
            "INSERT INTO passages (id,event_id,camera_id,lane,track_id,observed_at,direction,"
            "identity_id,status) VALUES (?,?,?,'A',?,?,?,?,'unresolved')",
            (name, EVENT, CAMERA + EVENT, track, at(len(name)), direction, identity),
        )


def refs_by_key(store, event=EVENT):
    return {t.track_key: f"tracklet:{t.id}" for t in load_tracklets(store, event, FACE_MODEL)}


def test_fragmented_crowd_with_lookalikes_and_a_human_review_loop(tmp_path):
    """No false merges unaided; exact count once a person clears the review band."""
    rng = np.random.default_rng(7)
    store = make_store(tmp_path)
    people = [unit(rng) for _ in range(40)]
    # Five sibling pairs: different people whose faces are genuinely similar.
    for index in range(5):
        people.append(near(rng, people[index], 0.7))
    truth_by_key = {}
    clock = 0.0
    for person, centre in enumerate(people):
        for fragment in range(int(rng.integers(1, 4))):
            key = f"ns:track-{person}-{fragment}"
            # Pose and lighting move each tracklet away from the person's centre,
            # so two tracklets of one person score between about 0.52 and 0.81.
            add_tracklet(store, rng, key, near(rng, centre, rng.uniform(0.72, 0.9)), clock)
            truth_by_key[key] = person
            clock += 10
    refs = refs_by_key(store)
    truth = {refs[key]: person for key, person in truth_by_key.items()}

    first = resolution.resolve(store, EVENT)
    unaided = clustering_metrics(first.assignments, truth)
    assert unaided["false_merge_pairs"] == 0
    assert unaided["pairwise_precision"] == 1.0
    assert unaided["true_people"] == 45
    # Fewer hypotheses than raw tracklets, but never fewer than the real crowd.
    assert 45 <= unaided["predicted_people"] < len(truth)
    assert first.report["calibration"]["calibrated"] is False

    resolution.apply(store, first, "pytest")
    queue = store.get_candidates(EVENT, status="review")
    assert queue and all(item["evidence"]["source"] == "global_resolver" for item in queue)
    for item in queue:
        same = truth[item["left_ref"]] == truth[item["right_ref"]]
        store.resolve_candidate(EVENT, item["id"], "same" if same else "different", "oracle")
    # Reviewing a derived link must not hand-edit the stored counters.
    assert store.get_summary(EVENT)["unresolved"] == 7

    second = resolution.resolve(store, EVENT)
    reviewed = clustering_metrics(second.assignments, truth)
    assert reviewed["count_error"] == 0
    assert reviewed["false_merge_pairs"] == reviewed["false_split_pairs"] == 0
    assert second.report["review"] == []
    assert second.report["constraints"]["human_confirmed_different"] > 0

    # The reviews are now labelled calibration pairs, of both outcomes.
    scores, labels = resolution.labelled_pairs(store, EVENT)
    assert len(scores) == len(queue) and True in labels and False in labels


def test_people_seen_together_are_never_merged(tmp_path):
    rng = np.random.default_rng(3)
    first = unit(rng)
    lookalike = near(rng, first, 0.68)

    together = make_store(tmp_path / "together")
    add_tracklet(together, rng, "ns:track-1", first, 0, length=5)
    add_tracklet(together, rng, "ns:track-2", lookalike, 2, length=5)
    result = resolution.resolve(together, EVENT)
    assert result.report["constraints"]["cannot_link_pairs"] == 1
    assert result.report["clusters"]["total"] == 2
    assert result.report["review"] == []

    apart = make_store(tmp_path / "apart")
    add_tracklet(apart, rng, "ns:track-1", first, 0, length=5)
    add_tracklet(apart, rng, "ns:track-2", lookalike, 60, length=5)
    assert resolution.resolve(apart, EVENT).report["clusters"]["total"] == 1

    # Near-identical overlapping tracklets are one person tracked twice.
    doubled = make_store(tmp_path / "doubled")
    add_tracklet(doubled, rng, "ns:track-1", first, 0, length=5)
    add_tracklet(doubled, rng, "ns:track-2", near(rng, first, 0.95), 2, length=5)
    report = resolution.resolve(doubled, EVENT).report
    assert report["constraints"]["duplicate_tracks_suspected"] == 1
    assert report["clusters"]["total"] == 1


def test_duplicate_identities_merge_and_unresolved_passages_attach(tmp_path):
    rng = np.random.default_rng(11)
    store = make_store(tmp_path)
    person = unit(rng)
    add_identity(store, "face-a", "Visitor 001", person)
    add_identity(store, "face-b", "Visitor 002", near(rng, person, 0.75))
    add_tracklet(store, rng, "ns:track-3", near(rng, person, 0.75), 100, samples=2)
    add_tracklet(store, rng, "ns:track-4", unit(rng), 200, samples=2)
    add_passage(store, "p1", "ns:track-1", "face-a")
    add_passage(store, "p22", "ns:track-2", "face-b")
    add_passage(store, "p333", "ns:track-3")
    add_passage(store, "p4444", "ns:track-4")
    add_passage(store, "p55555", "ns:track-1", "face-a", direction="out")

    result = resolution.resolve(store, EVENT)
    counts = result.report["counts"]
    assert counts["online"] == {
        "entries": 4, "exits": 1, "linked_entries": 2, "unique_entrants": 2,
        "repeat_entries": 0, "unresolved_entries": 2,
    }
    # One person entered three times. The stranger has only two samples, below
    # the enrollment bar, so that entry stays unresolved rather than invented.
    assert counts["resolved"] == {
        "entries": 4, "exits": 1, "linked_entries": 3, "unique_entrants": 1,
        "repeat_entries": 2, "unresolved_entries": 1,
    }
    assert result.report["clusters"]["identities_merged"] == 1
    assert result.report["clusters"]["tracklets_attached"] == 1
    assert result.report["attendance"]["lower_bound"] == 1
    assert result.report["attendance"]["upper_bound"] == 2
    assert "embedding" not in json.dumps(result.report)

    applied = resolution.apply(store, result, "pytest")
    assert applied["links_written"] == 2
    merge = next(
        item for item in store.get_candidates(EVENT)
        if {item["left_ref"], item["right_ref"]} == {"identity:face-a", "identity:face-b"}
    )
    assert merge["status"] == "auto_merged"
    store.resolve_candidate(EVENT, merge["id"], "different", "operator")
    overruled = resolution.resolve(store, EVENT)
    assert overruled.report["clusters"]["identities_merged"] == 0
    assert overruled.report["counts"]["resolved"]["unique_entrants"] == 2
    # Re-applying keeps the human decision.
    resolution.apply(store, overruled, "pytest")
    kept = next(item for item in store.get_candidates(EVENT) if item["id"] == merge["id"])
    assert kept["status"] == "confirmed_different" and kept["reviewed_by"] == "operator"


def test_registry_persists_every_finished_tracklet_and_purge_removes_it(tmp_path):
    rng = np.random.default_rng(5)
    store = make_store(tmp_path)
    registry = FaceRegistry(store, EVENT, CAMERA)
    glimpse, enrolled = unit(rng), unit(rng)
    for step in range(2):
        registry.observe("ns:face-track-1", near(rng, glimpse, 0.97), 0.9, set(), step, at(step))
    for step in range(3):
        registry.observe("ns:track-2", near(rng, enrolled, 0.97), 0.9, set(), step, at(step))
    assert load_tracklets(store, EVENT, FACE_MODEL) == []
    registry.expire(set(), 100)

    stored = {t.track_key: t for t in load_tracklets(store, EVENT, FACE_MODEL)}
    assert registry.evidence_saved == 2 and registry.evidence_errors == 0
    assert stored["ns:face-track-1"].kind == "face"
    assert stored["ns:face-track-1"].online_identity_id is None
    assert len(stored["ns:face-track-1"].embeddings) == 2
    assert stored["ns:track-2"].online_identity_id == registry.tracks.get(
        "ns:track-2", stored["ns:track-2"]
    ).online_identity_id
    assert stored["ns:track-2"].online_identity_id.startswith("face-")
    assert stored["ns:track-2"].first_observed_at == at(0)

    # A registry without a camera view (the paired controls) stores nothing.
    silent = FaceRegistry(store, EVENT)
    silent.observe("ns:track-9", near(rng, glimpse, 0.97), 0.9, set(), 0, at(0))
    silent.clear_tracks()
    assert len(load_tracklets(store, EVENT, FACE_MODEL)) == 2

    assert store.purge_plan(EVENT)["database_records"]["tracklet_evidence"] == 2
    store.purge_event(EVENT, f"PURGE {EVENT}", "pytest")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM tracklet_evidence").fetchone()[0] == 0


def test_attendance_estimate_recovers_a_simulated_crowd():
    rng = np.random.default_rng(21)
    true_people = 400
    visits = rng.choice([1, 2, 3], size=true_people, p=[0.7, 0.22, 0.08])
    resolved = rng.binomial(visits, 0.6)
    result = estimate_attendance(int(visits.sum()), resolved.tolist())
    assert result["lower_bound"] <= true_people <= result["upper_bound"]
    assert result["interval_90"][0] <= result["estimate"] <= result["interval_90"][1]
    assert abs(result["estimate"] - true_people) / true_people < 0.12
    # Better than either naive reading of the same data.
    assert abs(result["estimate"] - true_people) < true_people - result["lower_bound"]
    assert abs(result["estimate"] - true_people) < result["upper_bound"] - true_people


def test_attendance_reports_only_bounds_when_too_little_is_resolved():
    sparse = estimate_attendance(24, [1, 1, 1])
    assert (sparse["lower_bound"], sparse["upper_bound"]) == (3, 24)
    assert sparse["estimate"] is None and sparse["note"]
    exact = estimate_attendance(5, [2, 2, 1])
    assert exact["estimate"] == 3 and exact["method"] == "exact"
    assert estimate_attendance(0, [])["upper_bound"] == 0
    with pytest.raises(ValueError):
        estimate_attendance(1, [2])


def test_calibration_fit_and_persistence(tmp_path):
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.2, 0.8, size=400)
    same = rng.uniform(size=400) < 1 / (1 + np.exp(-(30 * scores - 15)))
    fitted = resolution.fit_calibration(scores.tolist(), same.tolist(), false_merge_weight=1.0)
    # Even odds near 0.5, as generated.
    assert abs(-fitted.intercept / fitted.slope - 0.5) < 0.03
    # Weighting false merges moves the boundary toward caution.
    cautious = resolution.fit_calibration(scores.tolist(), same.tolist())
    assert -cautious.intercept / cautious.slope > -fitted.intercept / fitted.slope
    path = tmp_path / "calibration.json"
    fitted.save(path)
    assert resolution.Calibration.load(path) == fitted
    assert resolution.Calibration.load(tmp_path / "missing.json").calibrated is False
    with pytest.raises(ValueError):
        resolution.fit_calibration([0.5] * 10, [True] * 10)


def test_clustering_metrics_on_a_known_case():
    truth = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 3}
    predicted = {"a": "x", "b": "x", "c": "x", "d": "y", "e": "z"}
    metrics = clustering_metrics(predicted, truth)
    assert metrics["false_merge_pairs"] == 2 and metrics["false_split_pairs"] == 1
    assert metrics["pairwise_precision"] == round(1 / 3, 4)
    assert metrics["pairwise_recall"] == 0.5
    assert metrics["count_error"] == 0
    assert clustering_metrics(truth, truth)["bcubed_f1"] == 1.0


def test_resolution_api(tmp_path):
    rng = np.random.default_rng(13)
    data_root = tmp_path / "data"
    store = make_store(data_root)
    person = unit(rng)
    add_tracklet(store, rng, "ns:track-1", person, 0)
    add_tracklet(store, rng, "ns:track-2", near(rng, person, 0.8), 60)
    settings = Settings(
        data_root=data_root, host="127.0.0.1", port=8000, demo_mode=False,
        demo_tick_seconds=0.01, allowed_origins=(), web_dist=tmp_path / "missing",
    )
    with TestClient(create_app(settings)) as client:
        report = client.get(f"/api/events/{EVENT}/resolution").json()
        assert report["clusters"]["total"] == 1 and len(report["auto_links"]) == 1
        assert client.get("/api/events/missing/resolution").status_code == 404
        applied = client.post(
            f"/api/events/{EVENT}/resolution/apply", json={"actor": "pytest"}
        ).json()
        assert applied["applied"]["links_written"] == 1
        assert client.post(
            f"/api/events/{EVENT}/resolution/apply", json={"actor": ""}
        ).status_code == 422


def test_pending_cross_view_match_attaches_a_passage_to_facing_faces(tmp_path):
    """A side crossing whose body matched a facing track uses that track's faces."""
    rng = np.random.default_rng(17)
    store = make_store(tmp_path)
    person = unit(rng)
    facing_camera = CAMERA + EVENT
    # The facing camera fragmented this face into single-sample tracks.
    add_tracklet(store, rng, "f:track-1", near(rng, person, 0.9), 0, samples=1)
    add_tracklet(store, rng, "f:face-track-1", near(rng, person, 0.9), 0, samples=1)
    add_tracklet(store, rng, "f:face-track-2", near(rng, person, 0.9), 30, samples=1)
    for name, reason in (("p1", "face_identity_pending"), ("p22", "appearance_below_threshold")):
        add_passage(store, name, f"side:{name}")
        with store.transaction() as db:
            db.execute(
                "UPDATE passages SET evidence=json_set(evidence,'$.cross_view_rejected',json(?)) "
                "WHERE id=?",
                (
                    json.dumps(
                        {
                            "last_reject_reason": reason,
                            "best_candidate": "f:track-1",
                            "facing_camera_id": facing_camera,
                        }
                    ),
                    name,
                ),
            )
    report = resolution.resolve(store, EVENT).report
    # Three one-sample tracklets agree, which meets the three-sample bar.
    assert report["clusters"]["people_from_unenrolled_faces"] == 1
    assert report["counts"]["online"]["linked_entries"] == 0
    # Only the crossing whose appearance match was accepted is attached.
    assert report["counts"]["resolved"]["linked_entries"] == 1
    assert report["counts"]["resolved"]["unresolved_entries"] == 1


def test_calibration_from_tracking_constraints(tmp_path):
    rng = np.random.default_rng(4)
    store = make_store(tmp_path)
    for index in range(12):
        prototype = unit(rng)
        vectors = [near(rng, prototype, rng.uniform(0.75, 0.95)) for _ in range(4)]
        save_tracklet(
            store, EVENT, CAMERA + EVENT, f"ns:track-{index}", FACE_MODEL, vectors, [0.9] * 4,
            at(index * 10), at(index * 10 + 2), None, "collecting",
        )
    impostors = rng.normal(0.05, 0.07, size=300).tolist()
    scores, labels = resolution.constraint_pairs(store, EVENT, impostors)
    assert labels.count(False) == 300 and labels.count(True) == 12 * 6
    fitted = resolution.fit_calibration(scores, labels, method="constraints")
    assert fitted.method == "constraints" and fitted.calibrated is False
    boundary = -fitted.intercept / fitted.slope
    assert max(impostors) < boundary + 0.15 and 0.15 < boundary < 0.6
    # A labelled fit is the only kind reported as calibrated.
    assert resolution.fit_calibration(scores, labels).calibrated is True

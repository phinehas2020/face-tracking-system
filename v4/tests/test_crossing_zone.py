"""Position evidence for the cross-camera link: the learned crossing zone."""

import json
import threading
from math import exp

import numpy as np
from ingress import resolution
from ingress.crossing_zone import CrossingZone
from ingress.dual_live import DualLiveSession
from ingress.live_association import LiveAssociationCoordinator
from ingress.store import EventStore


def door_box(offset, lateral=0.0, jitter=(0.0, 0.0)):
    """Facing-view box of someone ``offset`` seconds from the doorway."""
    height = 0.6 * exp(0.5 * offset)
    x, y = 0.5 + 0.05 * offset + lateral + jitter[0], 0.55 + 0.04 * offset + jitter[1]
    return (x - 0.15 * height, y - height / 2, x + 0.15 * height, y + height / 2)


class Scene:
    """Feeds time-ordered observations to a coordinator, resolving as it goes."""

    def __init__(self, coordinator, seed=0):
        self.coordinator, self.events, self.proposals = coordinator, [], []
        self.rng = np.random.default_rng(seed)

    def side(self, key, start, end):
        for at in np.arange(start, end, 0.1):
            self.events.append((float(at), "side", key, (0.3, 0.1, 0.7, 0.9), None))

    def facing(self, key, cross_at, start, end, *, lateral=0.0, identity=None, identity_from=None):
        for at in np.arange(cross_at + start, cross_at + end, 0.1):
            known = identity if identity_from is None or at >= cross_at + identity_from else None
            box = door_box(float(at) - cross_at, lateral, self.rng.normal(0, 0.004, 2))
            self.events.append((float(at), "facing", key, box, known))

    def standing(self, key, start, end, where=(0.2, 0.5), identity=None):
        box = (where[0] - 0.05, where[1] - 0.2, where[0] + 0.05, where[1] + 0.2)
        for at in np.arange(start, end, 0.1):
            self.events.append((float(at), "facing", key, box, identity))

    def crossing(self, passage, key, at, direction="in", known=None):
        self.events.append((at, "crossing", passage, (key, direction, known), None))

    def run(self, until):
        # The live loop resolves on every frame, whether or not anyone is in view.
        first = min((event[0] for event in self.events), default=until)
        ticks = [(float(at), "tick", None, None, None) for at in np.arange(first, until, 0.25)]
        for at, kind, key, payload, identity in sorted(self.events + ticks, key=lambda e: e[0]):
            if kind == "tick":
                pass
            elif kind == "crossing":
                side_key, direction, known = payload
                self.coordinator.register_crossing(
                    key, side_key, at, direction=direction, known_identity=known
                )
            else:
                self.coordinator.observe(
                    kind, key, at, box=payload, identity_id=identity,
                    identity_source="face_match" if identity else None,
                )
            self.proposals += self.coordinator.resolve_pending(at)
        self.events = []
        self.proposals += self.coordinator.resolve_pending(until)
        return self.proposals


def trained(mode="on", crossings=36):
    """A coordinator that has watched lone people walk through the door."""
    coordinator = LiveAssociationCoordinator(
        zone_mode=mode, max_track_gap_seconds=1.0, similarity_threshold=0.65
    )
    scene = Scene(coordinator, seed=1)
    for index in range(crossings):
        at = 100.0 + 40 * index
        scene.side(f"ts{index}", at - 1, at + 0.5)
        # As on the real door: the facing track sometimes ends just before the
        # crossing or starts just after it.
        start, end = [(-3, 1.2), (-3, -0.25), (0.25, 3)][index % 3]
        scene.facing(f"tf{index}", at, start, end)
        scene.crossing(f"train-{index}", f"ts{index}", at)
    assert scene.run(100.0 + 40 * crossings) == []
    assert coordinator.zone.ready("in")
    coordinator.drain_terminal()
    coordinator.expired_count = 0
    coordinator.zone_stats["decisions"].clear()
    return coordinator, 100.0 + 40 * crossings + 100


def zone_of(coordinator, passage):
    return coordinator.last_decisions[passage]["zone"]


def test_zone_trims_outliers_and_measures_how_selective_it_is():
    rng = np.random.default_rng(3)
    zone = CrossingZone(min_samples=20)
    door = np.array([0.5, 0.55, -0.5, 0.0])
    for _ in range(40):
        zone.add("in", door + rng.normal(0, [0.01, 0.01, 0.05, 0.1]))
    for _ in range(6):  # someone else, far from the door, on six occasions
        zone.add("in", [0.15, 0.4, -1.4, 0.0])
    assert not zone.ready("in")  # no idea yet how much of the view the zone covers
    for _ in range(300):
        zone.add_background(rng.uniform([0, 0, -2], [1, 1, 0]))
    summary = zone.summary()["directions"]["in"]
    assert summary["ready"] and summary["inliers"] == 40
    assert summary["background_inside_rate"] < 0.1
    assert zone.distance("in", door) < 1
    assert zone.distance("in", [0.15, 0.4, -1.4, 0.0]) > 100
    assert zone.distance("out", door) is None
    assert not zone.add("in", [0.5, float("nan"), 0, 0])


def test_time_alone_never_links_while_the_zone_is_still_learning():
    coordinator = LiveAssociationCoordinator(zone_mode="on", max_track_gap_seconds=1.0)
    scene = Scene(coordinator)
    scene.side("s", 9, 10.5)
    scene.facing("f", 10, -3, 1.2, identity="person-1")
    scene.crossing("p", "s", 10.0)
    assert scene.run(12.0) == []
    record = zone_of(coordinator, "p")
    assert record["decision"] == "zone_learning" and record["sample"] is True
    assert len(record["state"]) == 4


def test_position_links_without_any_appearance_evidence():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("f", at, -3, 1.2, identity="person-1")
    scene.crossing("p", "s", at)
    (proposal,) = scene.run(at + 3)
    assert proposal.identity_id == "person-1" and proposal.basis == "cross_view_position"
    assert proposal.facing_tracks == ("f",)
    assert "crossing_zone_position" in proposal.provenance
    summary = coordinator.summary()["zone"]
    assert summary["mode"] == "on" and summary["decisions"]["linked"] == 1
    assert summary["directions"]["in"]["background_inside_rate"] < 0.5


def test_shadow_mode_decides_and_records_but_assigns_nothing():
    coordinator, at = trained("shadow")
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("f", at, -3, 1.2, identity="person-1")
    scene.crossing("p", "s", at)
    assert scene.run(at + 40) == []
    (decision,) = coordinator.drain_terminal()
    assert decision["reason"] == "expired_unresolved"
    assert decision["zone"]["decision"] == "linked"
    assert decision["zone"]["identity"] == "person-1" and decision["zone"]["mode"] == "shadow"


def test_a_known_bystander_does_not_lend_their_identity_to_the_person_crossing():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.standing("staff", at - 20, at + 20, identity="staff-member")
    scene.side("s", at - 1, at + 0.5)
    scene.facing("visitor", at, -3, 1.2)
    scene.crossing("p", "s", at)
    assert scene.run(at + 40) == []
    (decision,) = coordinator.drain_terminal()
    assert decision["zone"]["facing_tracks"] == ["visitor"]
    assert decision["last_reject_reason"] == "face_identity_pending"
    assert decision["zone"]["sample"] is False  # two people in view: not a training case


def test_someone_standing_in_view_is_not_the_person_who_crossed():
    coordinator, at = trained()
    scene = Scene(coordinator)
    # Standing exactly where crossers are seen, but not moving; the real crosser
    # was missed by the facing camera.
    scene.standing("staff", at - 20, at + 20, where=(0.5, 0.55), identity="staff-member")
    scene.side("s", at - 1, at + 0.5)
    scene.crossing("p", "s", at)
    assert scene.run(at + 40) == []
    assert zone_of(coordinator, "p")["decision"] == "no_moving_facing_track"


def test_two_people_at_the_door_together_stay_unresolved():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("f1", at, -3, 1.2, identity="person-1")
    scene.facing("f2", at, -3, 1.2, lateral=0.01, identity="person-2")
    scene.crossing("p", "s", at)
    assert scene.run(at + 40) == []
    assert zone_of(coordinator, "p")["decision"] == "ambiguous_zone_match"


def test_someone_well_behind_does_not_block_the_link():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("front", at, -3, 1.2, identity="person-1")
    scene.facing("behind", at + 2.5, -3, 1.2, identity="person-2")
    scene.crossing("p", "s", at)
    (proposal,) = scene.run(at + 3)
    assert proposal.identity_id == "person-1"


def test_face_seen_on_an_earlier_fragment_of_the_facing_track_is_used():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("a", at, -3, -1.25, identity="person-1")  # face read far from the door
    scene.facing("b", at, -1.0, -0.35)  # lost behind the door frame
    scene.facing("c", at, 0.2, 1.2)  # picked up again after the crossing
    scene.crossing("p", "s", at)
    (proposal,) = scene.run(at + 3)
    assert proposal.identity_id == "person-1"
    assert set(proposal.facing_tracks) == {"a", "b", "c"}


def test_face_read_long_after_the_crossing_still_resolves_it():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("f", at, -1, 20, identity="person-1", identity_from=15)
    scene.crossing("p", "s", at)
    proposals = scene.run(at + 21)
    assert [p.identity_id for p in proposals] == ["person-1"]
    # Without a zone link the crossing would have expired after eight seconds.
    assert coordinator.expired_count == 0


def test_two_faces_on_one_linked_person_fail_closed():
    coordinator, at = trained()
    scene = Scene(coordinator)
    scene.side("s", at - 1, at + 0.5)
    scene.facing("a", at, -3, -0.35, identity="person-1")
    scene.facing("b", at, 0.0, 1.2, identity="person-2")
    scene.crossing("p", "s", at)
    assert scene.run(at + 40) == []
    (decision,) = coordinator.drain_terminal()
    assert decision["zone"]["identity_conflict"] is True


def test_crossings_that_already_have_a_face_score_the_position_evidence():
    coordinator, at = trained("shadow")
    scene = Scene(coordinator)
    for index, facing_identity in enumerate(["person-1", "someone-else"]):
        when = at + 60 * index
        scene.side(f"s{index}", when - 1, when + 0.5)
        scene.facing(f"f{index}", when, -3, 1.2, identity=facing_identity)
        scene.crossing(f"p{index}", f"s{index}", when, known="person-1")
    assert scene.run(at + 200) == []
    summary = coordinator.summary()
    assert summary["zone"]["label_agree"] == 1 and summary["zone"]["label_disagree"] == 1
    assert summary["expired_unresolved"] == 0  # labelled checks are not failures
    assert {d["reason"] for d in coordinator.drain_terminal()} == {"labelled_check"}


def test_labelled_crossings_are_ignored_when_the_zone_is_off():
    coordinator = LiveAssociationCoordinator()
    coordinator.register_crossing("p", "s", 10.0, direction="in", known_identity="person-1")
    assert not coordinator.pending and coordinator.summary()["zone"] is None


def station(tmp_path, mode):
    config = tmp_path / "camera.json"
    config.write_text(
        json.dumps(
            {
                "host": "side.invalid",
                "facing_camera": {"host": "front.invalid"},
                "association_overlap_seconds": 1.0,
                "association_zone": mode,
                "association_zone_min_samples": 8,
            }
        )
    )
    store = EventStore(tmp_path / "data")
    store.initialize()
    return DualLiveSession(store, config, "unused")


def test_session_stores_the_zone_record_and_relearns_from_it(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    session = station(tmp_path, "shadow")
    session.start()
    for index in range(10):
        at = 100.0 + 40 * index
        scene = Scene(session.association, seed=index)
        scene.side(f"s{index}", at - 1, at + 0.5)
        scene.facing(f"f{index}", at, -3, 1.2)
        scene.run(at - 0.01)
        session._record(f"s{index}", "in", captured_at=at)
        scene.facing(f"f{index}", at, 1.2, 2.0)
        scene.run(at + 2)
        session._resolve_cross_view(at + 30)
    with session.store.connect() as db:
        rows = [json.loads(row[0]) for row in db.execute("SELECT evidence FROM passages")]
    assert len(rows) == 10
    assert all(row["cross_view_zone"]["sample"] is True for row in rows)
    assert rows[0]["cross_view_zone"]["facing_camera_id"] == "live-facing"
    assert rows[0]["identity_resolution"] is False
    assert "side_first_offset" in rows[0]["cross_view_rejected"]

    again = DualLiveSession(session.store, tmp_path / "camera.json", "unused")
    again.start()
    assert again.snapshot()["cross_camera_association"]["zone"]["directions"]["in"]["samples"] == 10


def test_resolver_follows_a_position_link_to_the_facing_faces(tmp_path):
    from test_resolution import CAMERA, EVENT, add_passage, add_tracklet, make_store, near, unit

    rng = np.random.default_rng(23)
    store = make_store(tmp_path)
    person, facing_camera = unit(rng), CAMERA + EVENT
    # One person, tracked in three pieces by the facing camera.
    for index, key in enumerate(("f:track-1", "f:track-2", "f:track-3")):
        add_tracklet(store, rng, key, near(rng, person, 0.9), 10 * index, samples=1)
    for name, mode in (("p-on", "on"), ("p-shadow", "shadow")):
        add_passage(store, name, f"side:{name}")
        record = {
            "mode": mode, "decision": "linked", "facing_camera_id": facing_camera,
            "facing_tracks": ["f:track-1", "f:track-2", "f:track-3"],
        }
        with store.transaction() as db:
            db.execute(
                "UPDATE passages SET evidence=json_set(evidence,'$.cross_view_zone',json(?)) "
                "WHERE id=?",
                (json.dumps(record), name),
            )
    counts = resolution.resolve(store, EVENT).report["counts"]
    assert counts["online"]["linked_entries"] == 0
    # A shadow decision is recorded for study and is not counted.
    assert counts["resolved"]["linked_entries"] == 1
    assert counts["resolved"]["unresolved_entries"] == 1

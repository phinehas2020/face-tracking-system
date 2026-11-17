import numpy as np
from ingress.face_registry import FaceRegistry
from ingress.paired_identity import PairedIdentityFusion
from ingress.store import EventStore


def registry(store, name):
    with store.transaction() as db:
        db.execute(
            "INSERT INTO events (id,title,event_date,status,source_label,created_at) "
            "VALUES (?,?,?,'live',?,?)",
            (name, name, "2025-11-28", "fixture", "2025-11-28T00:00:00+00:00"),
        )
    return FaceRegistry(store, name)


def observe(r, key, v):
    for i in range(3):
        state = r.observe(key, v, 0.95, {key}, float(i), "2025-11-28T00:00:02+00:00")
    return state


def test_front_gallery_cannot_suppress_independent_side_match(tmp_path):
    store = EventStore(tmp_path)
    store.initialize()
    combined = registry(store, "combined")
    side = registry(store, "side")
    front = registry(store, "front")
    a = np.zeros(512, dtype=np.float32)
    a[0] = 1
    b = a * 0.5
    b[1] = np.sqrt(0.75)
    front_state = observe(front, "front:1", b)
    side_state = observe(side, "side:1", a)
    f = PairedIdentityFusion(combined)
    f.adopt("front", front, front_state)
    f.sync_local("side", side, {"side:1"}, 3, "2025-11-28T00:00:03+00:00")
    result = combined.tracks["side:1"]
    assert result.identity_id is not None
    assert result.state in {"new", "matched"}
    assert side.tracks["side:1"].identity_id == side_state.identity_id
    # No mutable histories shared across the local/control/combined decision paths.
    result.samples.clear()
    assert len(side_state.samples) == 3
    assert len(combined.templates) == 2
    f.sync_local("side", side, {"side:1"}, 4, "2025-11-28T00:00:04+00:00")
    assert combined.tracks["side:1"].identity_id == result.identity_id
    assert len(combined.templates) == 2


def test_local_confirmation_takes_priority_over_inferred_other_camera(tmp_path):
    store = EventStore(tmp_path)
    store.initialize()
    combined = registry(store, "combined")
    side = registry(store, "side")
    front = registry(store, "front")
    a = np.zeros(512, dtype=np.float32)
    a[0] = 1
    b = np.zeros(512, dtype=np.float32)
    b[1] = 1
    f = PairedIdentityFusion(combined)
    guessed = f.adopt("front", front, observe(front, "front:1", b))
    combined.tracks["side:1"] = guessed
    confirmed = observe(side, "side:1", a)
    f.sync_local("side", side, {"side:1"}, 3, "2025-11-28T00:00:03+00:00")
    assert combined.tracks["side:1"].identity_id != guessed.identity_id
    assert f.aliases[("side", confirmed.identity_id)] == combined.tracks["side:1"].identity_id


def test_independent_ids_can_merge_with_strong_repeated_support(tmp_path):
    store = EventStore(tmp_path)
    store.initialize()
    combined = registry(store, "combined")
    side = registry(store, "side")
    front = registry(store, "front")
    a = np.zeros(512, dtype=np.float32)
    a[0] = 1
    b = a * 0.6
    b[1] = 0.8
    f = PairedIdentityFusion(combined)
    fs = observe(front, "front:1", b)
    ss = observe(side, "side:1", a)
    fc = f.adopt("front", front, fs)
    sc = f.adopt("side", side, ss)
    combined.tracks["front:1"] = fc
    combined.tracks["side:1"] = sc
    assert not f.merge_confirmed_pair(
        fc, sc, {"confirmations": 2, "appearance": 0.9, "margin": 0.2}
    )
    assert f.merge_confirmed_pair(fc, sc, {"confirmations": 3, "appearance": 0.8, "margin": 0.2})
    assert f.aliases[("front", fs.identity_id)] == sc.identity_id
    assert len(combined.templates) == 1
    assert f.sources[sc.identity_id] == {"front", "side"}
    assert len(front.templates) == len(side.templates) == 1


def test_actual_face_conflict_is_not_merged(tmp_path):
    store = EventStore(tmp_path)
    store.initialize()
    combined = registry(store, "combined")
    side = registry(store, "side")
    front = registry(store, "front")
    a = np.zeros(512, dtype=np.float32)
    a[0] = 1
    b = np.zeros(512, dtype=np.float32)
    b[1] = 1
    f = PairedIdentityFusion(combined)
    fc = f.adopt("front", front, observe(front, "front:1", b))
    sc = f.adopt("side", side, observe(side, "side:1", a))
    assert not f.merge_confirmed_pair(
        fc, sc, {"confirmations": 8, "appearance": 0.99, "margin": 0.2}
    )
    assert len(combined.templates) == 2


def test_short_identity_handoff_never_duplicates_active_person(tmp_path):
    from ingress.paired_geometry import ViewObservation
    from ingress.paired_handoff import IdentityHandoff

    store = EventStore(tmp_path)
    store.initialize()
    r = registry(store, "combined")
    v = np.zeros(512, dtype=np.float32)
    v[0] = 1
    state = observe(r, "side:old", v)
    h = IdentityHandoff()
    old = ViewObservation("side:old", (0.5, 0.7), np.array([1.0, 0.0]), 10, 3.0)
    new = ViewObservation("side:new", (0.52, 0.7), np.array([1.0, 0.0]), 11, 3.5)
    assert not h.update([old], r, 3.0, "2025-11-28T00:00:03+00:00")
    assert not h.update([old, new], r, 3.5, "2025-11-28T00:00:03.500+00:00")
    assert h.update([new], r, 3.5, "2025-11-28T00:00:03.500+00:00")
    assert r.tracks["side:new"].identity_id == state.identity_id
    assert r.tracks["side:new"] is not state


def test_stale_identity_handoff_is_rejected(tmp_path):
    from ingress.paired_geometry import ViewObservation
    from ingress.paired_handoff import IdentityHandoff

    store = EventStore(tmp_path)
    store.initialize()
    r = registry(store, "combined")
    v = np.zeros(512, dtype=np.float32)
    v[0] = 1
    observe(r, "side:old", v)
    h = IdentityHandoff()
    h.update(
        [ViewObservation("side:old", (0.5, 0.7), np.array([1.0, 0.0]), 10, 3.0)],
        r,
        3.0,
        "2025-11-28T00:00:03+00:00",
    )
    assert not h.update(
        [ViewObservation("side:new", (0.5, 0.7), np.array([1.0, 0.0]), 20, 6.0)],
        r,
        6.0,
        "2025-11-28T00:00:06+00:00",
    )

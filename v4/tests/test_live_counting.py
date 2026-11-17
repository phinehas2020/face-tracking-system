import pytest
from ingress.live_counting import PassageCalibration, PassageCounter, anchor_for_bbox


def run(counter, observations, track="person"):
    return [event for now, point in observations
            if (event := counter.update(track, point, now))]


def test_sparse_fast_crossing_does_not_require_three_destination_frames():
    counter = PassageCounter(PassageCalibration())
    assert run(counter, [(0, (.2, .6)), (.25, (.8, .6)), (.5, (.9, .6))]) == ["in"]
    assert run(counter, [(.75, (.9, .6)), (1, (.8, .6))]) == []


def test_exit_and_reentry_on_same_track_and_reverse_configuration():
    points = [(0, (.2, .6)), (.2, (.8, .6)), (.4, (.8, .6)),
              (.6, (.2, .6)), (.8, (.2, .6)), (1, (.8, .6)), (1.2, (.8, .6))]
    assert run(PassageCounter(PassageCalibration()), points) == ["in", "out", "in"]
    assert run(PassageCounter(PassageCalibration(entry_positive=False)), points) == [
        "out", "in", "out"]


def test_confirmation_uses_elapsed_observed_time_and_never_empty_expiry():
    counter = PassageCounter(PassageCalibration(confirm_seconds=.2))
    assert run(counter, [(0, (.2, .6)), (.1, (.8, .6)), (.12, (.8, .6)),
                         (.15, (.8, .6)), (.19, (.8, .6))]) == []
    counter.expire(.3)
    assert counter.states["person"].side == -1
    assert counter.update("person", (.8, .6), .31) == "in"


def test_finite_gate_rejects_crossings_outside_endpoints_and_can_rearm():
    counter = PassageCounter(PassageCalibration(segment_start=.3, segment_end=.7))
    assert run(counter, [(0, (.2, .9)), (.2, (.8, .9)), (.4, (.8, .9))]) == []
    assert run(counter, [(.6, (.8, .5)), (.8, (.2, .5)), (1, (.2, .5))]) == ["out"]


def test_gate_intersection_uses_trajectory_not_just_destination_projection():
    counter = PassageCounter(PassageCalibration(segment_start=.4, segment_end=.6))
    # Destination projects within gate but crossing happened above its endpoint.
    assert run(counter, [(0, (.4, .1)), (.2, (.8, .5)), (.4, (.8, .5))]) == []


def test_hysteresis_and_turnback_reject_jitter_and_short_excursions():
    counter = PassageCounter(PassageCalibration())
    assert run(counter, [(0, (.2, .5)), (.1, (.49, .5)), (.2, (.51, .5)),
                         (.3, (.49, .5)), (.4, (.54, .5)), (.45, (.45, .5)),
                         (.6, (.3, .5))]) == []


def test_trajectory_through_neutral_band_keeps_real_intersection():
    counter = PassageCounter(PassageCalibration())
    assert run(counter, [(0, (.3, .5)), (.1, (.49, .5)), (.2, (.51, .5)),
                         (.3, (.6, .5)), (.4, (.7, .5))]) == ["in"]


def test_near_gate_first_sighting_needs_stable_region():
    counter = PassageCounter(PassageCalibration())
    assert counter.update("person", (.47, .5), 0) is None
    assert counter.states["person"].side == 0
    assert counter.update("person", (.47, .5), .11) is None
    assert counter.states["person"].side == -1


def test_motion_evidence_survives_approach_toward_gate():
    counter = PassageCounter(PassageCalibration(hysteresis=.01, min_motion=.15))
    assert run(counter, [(0, (.2, .5)), (.1, (.45, .5)), (.2, (.53, .5)),
                         (.4, (.54, .5)), (.6, (.7, .5)), (.8, (.8, .5))]) == ["in"]


def test_confirmed_destination_waits_for_sufficient_body_displacement():
    counter = PassageCounter(PassageCalibration(hysteresis=.01, min_motion=.15))
    assert run(counter, [(0, (.46, .5)), (.11, (.46, .5)), (.2, (.53, .5)),
                         (.4, (.54, .5))]) == []
    assert counter.states["person"].side == -1
    assert counter.update("person", (.7, .5), .6) == "in"
    assert counter.update("person", (.8, .5), .8) is None


def test_stale_gap_reset_and_different_track_id_never_bridge():
    counter = PassageCounter(PassageCalibration(max_gap_seconds=.5))
    assert run(counter, [(0, (.2, .5)), (.7, (.8, .5)), (.9, (.8, .5))]) == []
    counter.reset()
    assert counter.update("old", (.2, .5), 1) is None
    assert run(counter, [(1.1, (.8, .5)), (1.3, (.8, .5))], "new") == []
    counter.expire(2)
    assert not counter.states


def test_max_traversal_rejects_long_ambiguous_gate_dwell():
    counter = PassageCounter(PassageCalibration(max_traversal_seconds=.5))
    assert run(counter, [(0, (.2, .5)), (.1, (.51, .5)), (.3, (.51, .5)),
                         (.5, (.51, .5)), (.7, (.8, .5)), (.9, (.8, .5))]) == []


def test_duplicate_and_out_of_order_time_cannot_supply_confirmation():
    counter = PassageCounter(PassageCalibration())
    assert run(counter, [(0, (.2, .5)), (.2, (.8, .5)), (.2, (.8, .5)),
                         (.1, (.1, .5))]) == []
    assert counter.update("person", (.8, .5), .31) == "in"


def test_horizontal_and_custom_diagonal_direction():
    horizontal = PassageCounter(PassageCalibration(axis="horizontal"))
    assert run(horizontal, [(0, (.5, .2)), (.2, (.5, .8)), (.4, (.5, .8))]) == ["in"]
    diagonal = PassageCounter(PassageCalibration(gate_start=(.2, .2), gate_end=(.8, .8)))
    assert run(diagonal, [(0, (.3, .7)), (.2, (.7, .3)), (.4, (.8, .2))]) == ["in"]


def test_bbox_anchor_explicitly_distinguishes_center_and_bottom():
    assert anchor_for_bbox((20, 20, 60, 100), 100, 100, "center") == (.4, .6)
    assert anchor_for_bbox((20, 20, 60, 100), 100, 100, "footpoint") == (.4, 1)
    with pytest.raises(ValueError):
        anchor_for_bbox((20, 20, 60, 101), 100, 100)


@pytest.mark.parametrize("values", [
    {"gate_start": (.1, .1)}, {"gate_start": (.1, .1), "gate_end": (.1, .1)},
    {"gate_start": (-.1, .1), "gate_end": (.8, .8)}, {"position": float("nan")},
    {"segment_start": .8, "segment_end": .2}, {"max_gap_seconds": 0},
    {"confirm_seconds": float("inf")}, {"anchor": "guess"},
])
def test_invalid_calibration_is_rejected(values):
    with pytest.raises(ValueError):
        PassageCalibration(**values)


def test_state_is_bounded_and_invalid_anchor_drops_history():
    counter = PassageCounter(PassageCalibration(), max_tracks=2)
    for i in range(3):
        counter.update(str(i), (.2, .5), i * .1)
    assert set(counter.states) == {"1", "2"}
    with pytest.raises(ValueError):
        counter.update("2", (float("nan"), .5), .4)
    assert "2" not in counter.states


def test_legacy_shaped_configuration_can_be_adapted_without_mutation():
    class Legacy:
        axis = "horizontal"
        position = .4
        entry_positive = False
        require_body_motion = True

    calibration = PassageCalibration.from_calibration(Legacy())
    assert calibration.axis == "horizontal"
    assert calibration.position == .4
    assert calibration.anchor == "center"
    assert not calibration.entry_positive

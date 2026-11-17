import numpy as np
from ingress.live_counting import PassageCalibration, PassageCounter
from ingress.vision.continuity import TrackContinuity
from ingress.vision.contracts import Track


def test_real_fast_entry_geometry_survives_raw_id_break_with_three_cues():
    frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    continuity = TrackContinuity()
    counter = PassageCounter(PassageCalibration(position=0.55, anchor="center"))
    a = np.array([1.0, 0.0])
    b = np.array([0.6901, np.sqrt(1 - 0.6901**2)])
    samples = [
        (23.719, "a", (0.252, 0.542, 0.448, 0.998), a),
        (23.877, "a", (0.320, 0.541, 0.505, 0.999), a),
        (24.019, "a", (0.335, 0.559, 0.582, 0.999), a),
        (24.359, "b", (0.595, 0.532, 0.786, 0.998), b),
        (24.480, "b", (0.624, 0.543, 0.827, 1.0), b),
    ]
    ids, events = [], []
    for stamp, raw, box, feature in samples:
        pixels = tuple(v * (1920 if i % 2 == 0 else 1080) for i, v in enumerate(box))
        track = continuity.update([Track(raw, pixels, 0.9, 1)], frame, stamp, {raw: feature})[0]
        ids.append(track.track_id)
        point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        event = counter.update(track.track_id, point, stamp)
        if event:
            events.append(event)
    assert len(set(ids)) == 1
    assert events == ["in"]


def test_similar_clothes_do_not_override_opposing_motion():
    frame = np.full((1000, 1000, 3), 100, dtype=np.uint8)
    continuity = TrackContinuity()
    a = np.array([1.0, 0.0])
    b = np.array([0.69, np.sqrt(1 - 0.69**2)])
    for now, x in [(0, 300), (0.15, 400), (0.3, 500)]:
        old = continuity.update(
            [Track("old", (x, 100, x + 150, 500), 0.9, 1)], frame, now, {"old": a}
        )[0]
    new = continuity.update([Track("new", (250, 100, 400, 500), 0.9, 1)], frame, 0.5, {"new": b})[0]
    assert new.track_id != old.track_id

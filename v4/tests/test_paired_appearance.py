import numpy as np
from ingress.paired_appearance import AppearancePairMatcher
from ingress.paired_geometry import ViewObservation


def obs(key, vector=(1.0, 0.0), frame=1, clock=1, y=0.6):
    return ViewObservation(key, (0.5, y), np.asarray(vector), frame, clock)


def test_repeated_display_frame_is_not_confirmation():
    m = AppearancePairMatcher()
    a, b = obs("front"), obs("side")
    assert not m.update([a], [b])
    assert not m.update([a], [obs("side", frame=2)])
    out = m.update([obs("front", frame=2, clock=2)], [obs("side", frame=3, clock=2)])
    assert len(out) == 1 and out[0]["confirmations"] == 2


def test_stale_or_ambiguous_match_is_rejected():
    m = AppearancePairMatcher(confirmations=1)
    assert not m.update([obs("f")], [obs("s", clock=3)])
    assert not m.update([obs("f")], [obs("s"), obs("other")])
    assert not m.update([obs("f"), obs("other")], [obs("s")])


def test_region_filter_does_not_inflate_similarity_margin():
    m = AppearancePairMatcher(confirmations=1)
    assert not m.update([obs("f")], [obs("s", (0.8, 0.6)), obs("rival", (0.77, 0.638), y=0.1)])


def test_unique_one_to_one_matches():
    m = AppearancePairMatcher(confirmations=1)
    out = m.update([obs("f1"), obs("f2", (0.0, 1.0))], [obs("s1"), obs("s2", (0.0, 1.0))])
    assert {(m["front"], m["side"]) for m in out} == {("f1", "s1"), ("f2", "s2")}

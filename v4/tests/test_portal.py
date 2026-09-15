from ingress.portal import PortalState, PortalStateMachine


def test_portal_emits_one_passage_after_stable_crossing():
    portal = PortalStateMachine((0.0, 0.5), (1.0, 0.5), stable_frames=2, hysteresis=0.01)
    assert not portal.update("track-1", (0.5, 0.35))
    assert not portal.update("track-1", (0.5, 0.52))
    assert portal.update("track-1", (0.5, 0.61))
    assert not portal.update("track-1", (0.5, 0.72))
    assert portal.tracks["track-1"].state == PortalState.INSIDE


def test_jitter_near_line_does_not_count():
    portal = PortalStateMachine((0.0, 0.5), (1.0, 0.5), stable_frames=3, hysteresis=0.03)
    for y in (0.49, 0.51, 0.48, 0.52, 0.5):
        assert not portal.update("jitter", (0.4, y))

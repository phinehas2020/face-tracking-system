import numpy as np
from ingress.vision.crowd import Handoff


def rows(*ids):
    return np.array([[10, 10, 60, 90, key, .8, 0, i] for i, key in enumerate(ids)])


def established():
    h = Handoff()
    for now in (0, .1, .2):
        h.update(rows(10), [{0: 20}, {0: 30}], now, [np.array([1., 0.])])
    return h


def test_short_gap_requires_two_witnesses_and_matching_color():
    for checks, color, expected in [
        ([{0: 20}, {0: 30}], [1., 0.], 1),
        ([{0: 20}, {0: 31}], [1., 0.], 2),
        ([{0: 20}, {0: 30}], [0., 1.], 2),
    ]:
        out = established().update(rows(11), checks, 1.0, [np.array(color)])
        assert out[0, 4] == expected


def test_stale_evidence_and_occupied_history_cannot_steal_an_id():
    h = established()
    h.update(rows(10), [{0: 21}, {0: 31}], .3, [np.array([1., 0.])])
    assert h.update(rows(11), [{0: 20}, {0: 30}], .5, [np.array([1., 0.])])[0, 4] == 2
    h = established()
    out = h.update(rows(11, 10), [{0: 20, 1: 21}, {0: 30, 1: 31}], .5,
                   [np.array([1., 0.]), np.array([1., 0.])])
    assert list(out[:, 4]) == [2, 1]


def test_old_raw_id_returning_after_handoff_stays_distinct():
    h = established()
    color = np.array([1., 0.])
    h.update(rows(11), [{0: 20}, {0: 30}], 1.0, [color])
    out = h.update(rows(10, 11), [{0: 21, 1: 20}, {0: 31, 1: 30}], 1.1, [color, color])
    assert list(out[:, 4]) == [2, 1]

from ingress.domain import LinkStatus, PairEvidence
from ingress.identity import DisjointSet, FusionModel


def evidence(**overrides):
    values = {
        "face_similarity": 0.98,
        "body_similarity": 0.91,
        "face_quality": 0.96,
        "body_quality": 0.86,
        "supporting_frames": 20,
        "time_delta_seconds": 3.0,
        "topology_score": 0.98,
        "candidate_margin": 0.28,
    }
    values.update(overrides)
    return PairEvidence(**values)


def test_fusion_auto_merges_only_over_high_threshold():
    decision = FusionModel().score(evidence())
    assert decision.status == LinkStatus.AUTO_MERGED
    assert decision.probability >= 0.985
    assert "face similarity" in decision.explanation


def test_impossible_overlap_is_a_hard_constraint():
    decision = FusionModel().score(evidence(impossible_overlap=True))
    assert decision.status == LinkStatus.CONFIRMED_DIFFERENT
    assert decision.probability == 0


def test_disjoint_set_is_deterministic():
    clusters = DisjointSet(["p3", "p1", "p2"])
    clusters.union("p3", "p2")
    clusters.union("p2", "p1")
    assert clusters.clusters() == {"p1": ["p3", "p1", "p2"]}

"""Cross-camera identity must have learned, unambiguous multi-frame evidence."""

import numpy as np

from ingress.live_association import LiveAssociationCoordinator


def observe_pair(
    coordinator,
    *,
    side="s1",
    facing="f1",
    feature=(1.0, 0.0, 0.0),
    at=10.0,
    identity="person-1",
    source="osnet",
):
    for timestamp in (at, at + 0.25):
        coordinator.observe("side", side, timestamp, embedding=feature, embedding_source=source)
        coordinator.observe(
            "facing",
            facing,
            timestamp,
            embedding=feature,
            embedding_source=source,
            identity_id=identity,
            identity_source="face_match",
        )


def test_resolves_real_multiframe_appearance_once_with_provenance():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    coordinator.register_crossing("passage-1", "s1", 10.25)
    proposal = coordinator.resolve_crossing("passage-1", 10.3)
    assert proposal.identity_id == "person-1"
    assert proposal.appearance_score > 0.99
    assert "osnet_multiframe" in proposal.provenance
    assert "face_match" in proposal.provenance
    assert coordinator.resolve_pending(10.4) == []
    coordinator.register_crossing("passage-1", "s1", 10.25)
    assert coordinator.resolve_pending(10.5) == []


def test_face_can_arrive_after_crossing_without_losing_event():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    coordinator.register_crossing("passage-1", "s1", 10.25)
    assert coordinator.resolve_crossing("passage-1", 10.3) is None
    assert coordinator.summary()["pending"] == 1
    coordinator.observe("facing", "f1", 11.0, identity_id="person-1", identity_source="face_match")
    assert coordinator.resolve_crossing("passage-1", 11.1).identity_id == "person-1"


def test_unknown_lookalike_competes_with_known_identity():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    for timestamp in (10.0, 10.25):
        coordinator.observe(
            "facing", "stranger", timestamp, embedding=(0.999, 0.01, 0), embedding_source="osnet"
        )
    coordinator.register_crossing("p", "s1", 10.25)
    assert coordinator.resolve_crossing("p", 10.3) is None


def test_two_side_tracks_cannot_claim_one_facing_identity():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    for timestamp in (10.0, 10.25):
        coordinator.observe(
            "side", "s2", timestamp, embedding=(1.0, 0, 0), embedding_source="osnet"
        )
    coordinator.register_crossing("p1", "s1", 10.25)
    coordinator.register_crossing("p2", "s2", 10.25)
    assert coordinator.resolve_pending(10.3) == []


def test_distinct_concurrent_tracks_resolve_one_to_one():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    observe_pair(coordinator, side="s2", facing="f2", feature=(0.0, 1.0, 0.0), identity="person-2")
    coordinator.register_crossing("p1", "s1", 10.25)
    coordinator.register_crossing("p2", "s2", 10.25)
    proposals = coordinator.resolve_pending(10.3)
    assert {(p.event_id, p.identity_id) for p in proposals} == {
        ("p1", "person-1"),
        ("p2", "person-2"),
    }


def test_time_and_histogram_features_alone_never_match():
    for source in (None, "histogram", "color", "unverified_encoder"):
        coordinator = LiveAssociationCoordinator()
        observe_pair(coordinator, source=source)
        coordinator.register_crossing("p", "s1", 10.25)
        assert coordinator.resolve_pending(10.3) == []


def test_single_frame_or_duplicate_frame_is_insufficient():
    coordinator = LiveAssociationCoordinator()
    for _ in range(3):
        for view in ("side", "facing"):
            coordinator.observe(
                view,
                view,
                10.0,
                embedding=(1.0, 0),
                embedding_source="osnet",
                identity_id="person-1",
                identity_source="face_match",
            )
    coordinator.register_crossing("p", "side", 10.0)
    assert coordinator.resolve_pending(10.1) == []


def test_wrong_time_and_expiry_leave_unresolved():
    coordinator = LiveAssociationCoordinator(max_transition_seconds=2.0)
    observe_pair(coordinator, at=10.0)
    coordinator.register_crossing("p", "s1", 14.0)
    assert coordinator.resolve_pending(14.0) == []
    assert coordinator.resolve_pending(16.1) == []
    assert coordinator.summary()["expired_unresolved"] == 1


def test_conflicting_identity_evidence_fails_closed():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    coordinator.observe("facing", "f1", 10.4, identity_id="person-2", identity_source="face_match")
    coordinator.register_crossing("p", "s1", 10.4)
    assert coordinator.resolve_pending(10.5) == []


def test_unverified_identity_and_invalid_embeddings_fail_closed():
    for feature in ((float("nan"), 1), (0.0, 0.0), (float("inf"), 1), [[1.0, 0.0]]):
        coordinator = LiveAssociationCoordinator()
        observe_pair(coordinator, feature=feature)
        coordinator.register_crossing("p", "s1", 10.25)
        assert coordinator.resolve_pending(10.3) == []
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    coordinator.observe("facing", "f1", 10.4, identity_id="guess", identity_source="time_proximity")
    coordinator.register_crossing("p", "s1", 10.4)
    assert coordinator.resolve_pending(10.5) == []


def test_divergent_frames_cannot_hide_behind_mean_embedding():
    coordinator = LiveAssociationCoordinator()
    for view in ("side", "facing"):
        for timestamp, feature in [(10.0, (1.0, 0)), (10.25, (0.0, 1))]:
            coordinator.observe(
                view,
                view,
                timestamp,
                embedding=feature,
                embedding_source="osnet",
                identity_id="p1",
                identity_source="face_match",
            )
    coordinator.register_crossing("p", "side", 10.25)
    assert coordinator.resolve_pending(10.3) == []


def test_bounds_expiration_reset_and_normalization():
    coordinator = LiveAssociationCoordinator(max_tracks=2, max_pending=2)
    observe_pair(coordinator, feature=np.array([100.0, 0.0, 0.0]))
    coordinator.register_crossing("p1", "s1", 10.25)
    assert coordinator.resolve_pending(10.3)[0].appearance_score > 0.99
    for i in range(5):
        coordinator.observe("side", str(i), 11.0 + i)
        coordinator.register_crossing("event" + str(i), str(i), 11.0 + i)
    assert coordinator.summary()["pending"] == 2
    assert coordinator.summary()["side_tracks"] == 2
    coordinator.resolve_pending(50.0)
    assert coordinator.summary()["facing_tracks"] == 0
    coordinator.reset()
    assert coordinator.summary()["linked"] == 0
    assert coordinator.summary()["expired_unresolved"] == 0


def test_track_pair_claim_is_not_reassigned_to_another_side_track():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    coordinator.register_crossing("p1", "s1", 10.25)
    assert coordinator.resolve_pending(10.3)
    # The first body's appearance evidence has aged outside this event's window,
    # but its one-to-one association is still reserved while the track is alive.
    observe_pair(coordinator, side="s2", at=20.0)
    coordinator.register_crossing("p2", "s2", 20.25)
    assert coordinator.resolve_pending(20.3) == []


def test_same_track_can_link_separate_entry_and_exit_passages():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    coordinator.register_crossing("entry", "s1", 10.25)
    assert coordinator.resolve_pending(10.3)
    coordinator.register_crossing("exit", "s1", 11.0)
    assert coordinator.resolve_pending(11.1)[0].identity_id == "person-1"


def test_delayed_identity_after_body_left_uses_original_frame_without_new_observation():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    coordinator.register_crossing("entry", "s1", 10.25)
    assert coordinator.resolve_pending(12.0) == []
    front = coordinator.tracks["facing"]["f1"]
    seen, sample_count = front.last_seen, len(front.samples)
    assert coordinator.attach_identity("facing", "f1", "person-1", "face_match", 10.0)
    assert front.last_seen == seen
    assert len(front.samples) == sample_count
    assert coordinator.resolve_pending(12.1)[0].identity_id == "person-1"


def test_delayed_identity_cannot_resurrect_expired_or_create_missing_track():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    coordinator.resolve_pending(41.0)
    assert not coordinator.attach_identity("facing", "f1", "person-1", "face_match", 10.0)
    assert not coordinator.attach_identity("facing", "missing", "person-1", "face_match", 41.0)
    assert coordinator.summary()["facing_tracks"] == 0


def test_delayed_identity_rejects_outside_track_lifetime_and_untrusted_source():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    for evidence_at in (9.0, 10.5, float("nan"), float("inf")):
        assert not coordinator.attach_identity(
            "facing", "f1", "person-1", "face_match", evidence_at
        )
    assert not coordinator.attach_identity("facing", "f1", "person-1", "time_only", 10.0)
    assert coordinator.tracks["facing"]["f1"].identity_id is None


def test_delayed_identity_conflict_remains_sticky():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    assert not coordinator.attach_identity("facing", "f1", "person-2", "face_match", 10.0)
    assert not coordinator.attach_identity("facing", "f1", "person-1", "face_match", 10.25)
    coordinator.register_crossing("entry", "s1", 10.25)
    assert coordinator.resolve_pending(11.0) == []


def test_overlapping_views_exclude_old_visit_with_similar_clothes():
    coordinator = LiveAssociationCoordinator(max_track_gap_seconds=1.0)
    for stamp in [1.0, 1.25]:
        coordinator.observe(
            "facing", "old-visit", stamp, embedding=(1.0, 0.0, 0.0), embedding_source="osnet"
        )
    observe_pair(coordinator, at=6.0)
    coordinator.register_crossing("current-passage", "s1", 6.25)
    assert coordinator.resolve_crossing("current-passage", 6.3).facing_track_id == "f1"


def test_decision_diagnostics_distinguish_identity_wait_and_expiration():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    coordinator.register_crossing('p', 's1', 10.25)
    coordinator.resolve_pending(11.)
    decision = coordinator.summary()['last_decisions'][-1]
    assert decision['reason'] == 'face_identity_pending'
    assert decision['appearance_score'] > .99
    assert decision['face_ready'] is False
    coordinator.resolve_pending(19.)
    decision = coordinator.summary()['last_decisions'][-1]
    assert decision['reason'] == 'expired_unresolved'
    assert decision['last_reject_reason'] == 'face_identity_pending'
    assert 'embedding' not in str(decision)


def test_decision_diagnostics_report_reciprocal_ambiguity():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    for timestamp in (10., 10.25):
        coordinator.observe('side', 'rival', timestamp, embedding=(1., 0., 0.), embedding_source='osnet')
    coordinator.register_crossing('p', 's1', 10.25)
    coordinator.resolve_pending(11.)
    decision = coordinator.summary()['last_decisions'][-1]
    assert decision['reason'] == 'ambiguous_reciprocal_match'
    assert decision['reciprocal_margin'] == 0
    assert decision['reciprocal_competitor'] == 'rival'


def test_decision_diagnostics_are_bounded_and_resettable():
    coordinator = LiveAssociationCoordinator()
    for i in range(80):
        coordinator.register_crossing(str(i), 'missing', 10.)
        coordinator.resolve_crossing(str(i), 10.)
    decisions = coordinator.summary()['last_decisions']
    assert len(decisions) == 64
    assert decisions[-1]['reason'] == 'missing_side_track'
    coordinator.reset()
    assert coordinator.summary()['last_decisions'] == []


def test_terminal_decisions_are_drained_once_for_persistence():
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator)
    coordinator.register_crossing('linked', 's1', 10.25)
    assert coordinator.resolve_pending(11.)
    observe_pair(coordinator, side='s2', facing='f2', feature=(0.0, 1.0, 0.0), identity=None)
    coordinator.register_crossing('waiting', 's2', 10.25)
    coordinator.resolve_pending(11.)
    # A crossing still inside its window is not final.
    assert [d['reason'] for d in coordinator.drain_terminal()] == ['linked']
    coordinator.resolve_pending(19.)
    final = coordinator.drain_terminal()
    assert [(d['event_id'], d['reason'], d['last_reject_reason']) for d in final] == [
        ('waiting', 'expired_unresolved', 'face_identity_pending')
    ]
    assert final[0]['best_candidate'] == 'f2'
    assert coordinator.drain_terminal() == []


def test_identity_is_checked_after_every_appearance_test():
    """'face_identity_pending' must mean the appearance match itself was accepted."""
    coordinator = LiveAssociationCoordinator()
    observe_pair(coordinator, identity=None)
    # A second side track looks the same, so the match is reciprocally ambiguous.
    for timestamp in (10.0, 10.25):
        coordinator.observe(
            'side', 's2', timestamp, embedding=(1.0, 0.0, 0.0), embedding_source='osnet'
        )
    coordinator.register_crossing('p', 's1', 10.25)
    coordinator.resolve_pending(11.)
    assert coordinator.summary()['last_decisions'][-1]['reason'] == 'ambiguous_reciprocal_match'


def _sole_setup(**extra):
    coordinator = LiveAssociationCoordinator(
        similarity_threshold=0.65, margin_threshold=0.15, max_track_gap_seconds=1.0, **extra
    )
    for timestamp in (10.0, 10.25):
        coordinator.observe(
            'side', 's1', timestamp, embedding=(1.0, 0.0, 0.0), embedding_source='osnet'
        )
        # Cosine 0.6 to the side view: a typical genuine cross-view score here.
        coordinator.observe(
            'facing', 'f1', timestamp, embedding=(0.6, 0.8, 0.0), embedding_source='osnet',
            identity_id='person-1', identity_source='face_match',
        )
    coordinator.register_crossing('p', 's1', 10.25)
    return coordinator


def test_sole_candidate_in_both_views_links_at_the_lower_threshold():
    assert _sole_setup().resolve_pending(11.) == []
    proposal, = _sole_setup(sole_candidate_threshold=0.5).resolve_pending(11.)
    assert proposal.identity_id == 'person-1'
    assert 'sole_candidate_in_both_views' in proposal.provenance


def test_a_second_person_in_either_view_restores_the_full_threshold():
    for view, key in (('facing', 'f2'), ('side', 's2')):
        coordinator = _sole_setup(sole_candidate_threshold=0.5)
        for timestamp in (10.0, 10.25):
            coordinator.observe(view, key, timestamp)  # present, even without appearance
        assert coordinator.resolve_pending(11.) == []
        decision = coordinator.summary()['last_decisions'][-1]
        assert decision['reason'] == 'appearance_below_threshold'
        assert decision['sole_candidate'] is False
    # Someone who was in view at another time does not count as a rival.
    coordinator = _sole_setup(sole_candidate_threshold=0.5)
    coordinator.observe('facing', 'old', 2.0)
    assert len(coordinator.resolve_pending(11.)) == 1

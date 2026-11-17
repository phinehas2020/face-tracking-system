"""Conservative online links between facing identities and side passage events.

All times are capture times on the same host's monotonic clock. ``event_id`` is
the passage row's ID, not the station/event configuration ID. This module emits
proposals only; the caller persists them while holding the shared session lock.
Time gates restrict candidates but never constitute identity evidence.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class AssociationProposal:
    event_id: str
    side_track_id: str
    facing_track_id: str
    identity_id: str
    appearance_score: float
    margin: float
    provenance: tuple[str, ...]


@dataclass
class _Track:
    last_seen: float
    first_seen: float
    samples: deque = field(default_factory=lambda: deque(maxlen=16))
    identity_id: str | None = None
    identity_source: str | None = None
    identity_conflict: bool = False


@dataclass(frozen=True)
class _Crossing:
    side_track_id: str
    captured_at: float


class LiveAssociationCoordinator:
    """Bounded evidence store. Call under the station's shared lock.

    Two distinct learned appearance observations are required in each view.
    Matching is reciprocal with a margin over *all* competing tracks, including
    tracks without a face identity. This prevents a convenient known identity
    winning merely because an equally plausible stranger has no face result.
    Thresholds are conservative starting points, not calibrated probabilities.
    """

    def __init__(
        self,
        *,
        similarity_threshold=0.85,
        margin_threshold=0.08,
        max_transition_seconds=8.0,
        retention_seconds=30.0,
        min_samples=2,
        max_tracks=256,
        max_pending=512,
        max_track_gap_seconds=None,
        sole_candidate_threshold=None,
    ):
        if not (0 < similarity_threshold <= 1 and 0 < margin_threshold <= 1):
            raise ValueError("Invalid appearance thresholds")
        if not (0 < max_transition_seconds <= retention_seconds):
            raise ValueError("Invalid transition/retention window")
        if min_samples < 2 or min_samples > 16 or max_tracks < 2 or max_pending < 1:
            raise ValueError("Invalid evidence bounds")
        if max_track_gap_seconds is not None and not (
            0 <= max_track_gap_seconds <= max_transition_seconds
        ):
            raise ValueError("Invalid overlapping-view timing window")
        if sole_candidate_threshold is not None and not (
            0 < sole_candidate_threshold <= similarity_threshold
        ):
            raise ValueError("Invalid sole-candidate threshold")
        # When exactly one person is in each view at the same moment, being the
        # only candidate is itself strong evidence, and appearance only has to
        # not contradict it. Needs the overlapping-view timing window.
        self.sole_candidate_threshold = sole_candidate_threshold
        self.max_track_gap_seconds = max_track_gap_seconds
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.max_transition_seconds = max_transition_seconds
        self.retention_seconds = retention_seconds
        self.min_samples = min_samples
        self.max_tracks = max_tracks
        self.max_pending = max_pending
        self.tracks: dict[str, OrderedDict[str, _Track]] = {
            "side": OrderedDict(),
            "facing": OrderedDict(),
        }
        self.pending: OrderedDict[str, _Crossing] = OrderedDict()
        self._emitted: OrderedDict[str, None] = OrderedDict()
        self._side_to_facing: dict[str, str] = {}
        self._facing_to_side: dict[str, str] = {}
        self._latest_time = 0.0
        self.linked_count = 0
        self.expired_count = 0
        self.last_decisions: OrderedDict[str, dict] = OrderedDict()
        # Final outcomes awaiting persistence by the session (bounded).
        self.terminal: deque = deque(maxlen=512)

    def drain_terminal(self) -> list[dict]:
        decisions = list(self.terminal)
        self.terminal.clear()
        return decisions

    def _note(self, event_id, crossing, now, reason, **evidence):
        self.last_decisions[event_id] = {
            "event_id": event_id, "side_track_id": crossing.side_track_id,
            "crossing_at": crossing.captured_at, "decision_at": now,
            "reason": reason, **evidence,
        }
        self.last_decisions.move_to_end(event_id)
        while len(self.last_decisions) > 64:
            self.last_decisions.popitem(last=False)

    def observe(
        self,
        view: str,
        track_id: str,
        captured_at: float,
        *,
        embedding=None,
        embedding_source: str | None = None,
        identity_id: str | None = None,
        identity_source: str | None = None,
        ground_position=None,
    ) -> None:
        if view not in self.tracks:
            raise ValueError("view must be side or facing")
        if not isfinite(captured_at):
            return
        self._prune(max(self._latest_time, captured_at))
        if captured_at < self._latest_time - self.retention_seconds:
            return
        tracks = self.tracks[view]
        track = tracks.setdefault(track_id, _Track(captured_at, captured_at))
        # Duplicate/out-of-order frames cannot manufacture multi-frame evidence.
        if captured_at < track.last_seen:
            return
        track.last_seen = captured_at
        tracks.move_to_end(track_id)
        while len(tracks) > self.max_tracks:
            self._drop_track(view, next(iter(tracks)))

        self.attach_identity(view, track_id, identity_id, identity_source, captured_at)
        # Only the explicitly integrated learned encoder is admitted. Histograms,
        # time-only candidates and unknown feature types fail closed.
        if embedding_source != "osnet" or embedding is None:
            return
        try:
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all() or norm < 1e-8:
                return
        except (TypeError, ValueError):
            return
        if track.samples and track.samples[-1][0] == captured_at:
            return
        track.samples.append((captured_at, vector / norm))
        # ground_position intentionally unused until real camera calibration and
        # uncertainty are available; accepting this parameter does not imply any.

    def attach_identity(
        self,
        view: str,
        track_id: str,
        identity_id: str | None,
        identity_source: str | None,
        evidence_at: float,
    ) -> bool:
        """Apply delayed face evidence to an already observed body track.

        The caller must have associated the face to this body on the original
        frame. ``evidence_at`` is that frame's capture time, never result-arrival
        time. No track, appearance sample, last-seen time, or new lifetime is
        created here. The current coordinator watermark bounds stale results.
        """
        if view not in self.tracks:
            raise ValueError("view must be side or facing")
        if (
            not identity_id
            or identity_source not in {"face_match", "face_track_continuity"}
            or not isfinite(evidence_at)
        ):
            return False
        self._prune(self._latest_time)
        track = self.tracks[view].get(track_id)
        if (
            track is None
            or evidence_at < self._latest_time - self.retention_seconds
            or not track.first_seen <= evidence_at <= track.last_seen
        ):
            return False
        if track.identity_id and track.identity_id != identity_id:
            track.identity_conflict = True
        if track.identity_conflict:
            return False
        track.identity_id, track.identity_source = identity_id, identity_source
        return True

    def register_crossing(self, event_id: str, side_track_id: str, captured_at: float) -> None:
        if not isfinite(captured_at) or event_id in self._emitted or event_id in self.pending:
            return
        self._prune(max(self._latest_time, captured_at))
        if captured_at < self._latest_time - self.max_transition_seconds:
            self.expired_count += 1
            return
        self.pending[event_id] = _Crossing(side_track_id, captured_at)
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
            self.expired_count += 1

    def _time_compatible(self, left, right):
        if self.max_track_gap_seconds is None:
            return True
        gap = max(left.first_seen - right.last_seen, right.first_seen - left.last_seen, 0)
        return gap <= self.max_track_gap_seconds

    def _features(self, track: _Track, event_time: float):
        samples = [
            v for t, v in track.samples if abs(t - event_time) <= self.max_transition_seconds
        ]
        if len(samples) < self.min_samples or len({v.shape for v in samples}) != 1:
            return None
        values = np.stack(samples)
        mean = values.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            return None
        return values, mean / norm

    @staticmethod
    def _score(left, right):
        if left is None or right is None or left[1].shape != right[1].shape:
            return None
        # A high mean-vector score alone can hide contradictory frames. Require
        # pairwise multi-frame agreement as well, taking the conservative score.
        return min(float(left[1] @ right[1]), float(np.mean(left[0] @ right[0].T)))

    def resolve_crossing(self, event_id: str, now: float) -> AssociationProposal | None:
        if not isfinite(now):
            return None
        self._prune(max(self._latest_time, now))
        crossing = self.pending.get(event_id)
        if crossing is None:
            return None
        side = self.tracks["side"].get(crossing.side_track_id)
        if side is None:
            self._note(event_id, crossing, now, "missing_side_track")
            return None
        side_features = self._features(side, crossing.captured_at)
        evidence = {"side_samples": 0 if side_features is None else len(side_features[0]),
                    "side_first_seen": side.first_seen, "side_last_seen": side.last_seen,
                    "temporal_rejections": 0, "appearance_unavailable": 0}
        if side_features is None:
            self._note(event_id, crossing, now, "insufficient_side_appearance", **evidence)
            return None
        candidates = []
        for front_id, front in self.tracks["facing"].items():
            if not self._time_compatible(side, front):
                evidence["temporal_rejections"] += 1
                continue
            score = self._score(side_features, self._features(front, crossing.captured_at))
            if score is not None:
                candidates.append((score, front_id, front))
            else:
                evidence["appearance_unavailable"] += 1
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates:
            self._note(event_id, crossing, now, "no_eligible_facing_appearance", **evidence)
            return None
        score, front_id, front = candidates[0]
        margin = score - (candidates[1][0] if len(candidates) > 1 else -1.0)
        evidence.update(best_candidate=front_id, appearance_score=score,
                        second_score=candidates[1][0] if len(candidates) > 1 else None,
                        facing_margin=margin, reciprocal_margin=None,
                        face_ready=bool(front.identity_id and not front.identity_conflict),
                        facing_first_seen=front.first_seen, facing_last_seen=front.last_seen,
                        facing_samples=len(self._features(front, crossing.captured_at)[0]))
        side_rivals = sum(
            other_id != crossing.side_track_id and self._time_compatible(other, front)
            for other_id, other in self.tracks["side"].items()
        )
        sole = (
            self.sole_candidate_threshold is not None
            and self.max_track_gap_seconds is not None
            and len(candidates) == 1
            and evidence["appearance_unavailable"] == 0
            and side_rivals == 0
        )
        evidence["sole_candidate"] = sole
        if score < (self.sole_candidate_threshold if sole else self.similarity_threshold):
            self._note(event_id, crossing, now, "appearance_below_threshold", **evidence)
            return None
        if margin < self.margin_threshold:
            self._note(event_id, crossing, now, "ambiguous_facing_match", **evidence)
            return None
        if self._side_to_facing.get(crossing.side_track_id, front_id) != front_id:
            self._note(event_id, crossing, now, "side_already_assigned", **evidence)
            return None
        if self._facing_to_side.get(front_id, crossing.side_track_id) != crossing.side_track_id:
            self._note(event_id, crossing, now, "facing_already_assigned", **evidence)
            return None
        front_features = self._features(front, crossing.captured_at)
        for other_id, other in self.tracks["side"].items():
            if other_id == crossing.side_track_id or not self._time_compatible(other, front):
                continue
            competing_score = self._score(
                self._features(other, crossing.captured_at), front_features
            )
            if competing_score is not None:
                margin = min(margin, score - competing_score)
                if evidence["reciprocal_margin"] is None or score - competing_score < evidence["reciprocal_margin"]:
                    evidence["reciprocal_margin"] = score - competing_score
                    evidence["reciprocal_competitor"] = other_id
                    evidence["reciprocal_competitor_score"] = competing_score
        if margin < self.margin_threshold:
            self._note(event_id, crossing, now, "ambiguous_reciprocal_match", **evidence)
            return None
        # Identity is checked last, so "face_identity_pending" means every
        # appearance test passed and only a face identity is missing.
        if not front.identity_id or front.identity_conflict:
            reason = (
                "face_identity_conflict" if front.identity_conflict else "face_identity_pending"
            )
            self._note(event_id, crossing, now, reason, **evidence)
            return None
        self._note(event_id, crossing, now, "linked", **evidence)
        self.terminal.append(dict(self.last_decisions[event_id]))
        self._side_to_facing[crossing.side_track_id] = front_id
        self._facing_to_side[front_id] = crossing.side_track_id
        self.pending.pop(event_id)
        self._emitted[event_id] = None
        while len(self._emitted) > self.max_pending * 2:
            self._emitted.popitem(last=False)
        self.linked_count += 1
        return AssociationProposal(
            event_id,
            crossing.side_track_id,
            front_id,
            front.identity_id,
            score,
            margin,
            (
                "osnet_multiframe",
                "reciprocal_appearance_margin",
                "bounded_capture_time",
                *(("sole_candidate_in_both_views",) if sole else ()),
                front.identity_source,
            ),
        )

    def resolve_pending(self, now: float) -> list[AssociationProposal]:
        if not isfinite(now):
            return []
        self._prune(max(self._latest_time, now))
        proposals = []
        for event_id in list(self.pending):
            proposal = self.resolve_crossing(event_id, now)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _drop_track(self, view: str, track_id: str) -> None:
        self.tracks[view].pop(track_id, None)
        if view == "side":
            paired = self._side_to_facing.pop(track_id, None)
            if paired is not None:
                self._facing_to_side.pop(paired, None)
        else:
            paired = self._facing_to_side.pop(track_id, None)
            if paired is not None:
                self._side_to_facing.pop(paired, None)

    def _prune(self, now: float) -> None:
        self._latest_time = now
        for view, tracks in self.tracks.items():
            for track_id, track in list(tracks.items()):
                if now - track.last_seen > self.retention_seconds:
                    self._drop_track(view, track_id)
        for event_id, crossing in list(self.pending.items()):
            if now - crossing.captured_at > self.max_transition_seconds:
                self.pending.pop(event_id)
                self.expired_count += 1
                previous = self.last_decisions.get(event_id, {})
                self._note(event_id, crossing, now, "expired_unresolved", **{
                    key: value for key, value in previous.items()
                    if key not in {"event_id", "side_track_id", "crossing_at", "decision_at", "reason"}
                }, last_reject_reason=previous.get("reason"))
                self.terminal.append(dict(self.last_decisions[event_id]))

    def reset(self) -> None:
        """Clear transient links when counting resets or capture namespaces reset."""
        for tracks in self.tracks.values():
            tracks.clear()
        self.pending.clear()
        self._emitted.clear()
        self._side_to_facing.clear()
        self._facing_to_side.clear()
        self.linked_count = self.expired_count = 0
        self.last_decisions.clear()
        self.terminal.clear()

    def summary(self) -> dict:
        return {
            "pending": len(self.pending),
            "linked": self.linked_count,
            "expired_unresolved": self.expired_count,
            "side_tracks": len(self.tracks["side"]),
            "facing_tracks": len(self.tracks["facing"]),
            "side_appearance_ready": sum(
                self._features(track, self._latest_time) is not None
                for track in self.tracks["side"].values()
            ),
            "facing_appearance_ready": sum(
                self._features(track, self._latest_time) is not None
                for track in self.tracks["facing"].values()
            ),
            "method": "osnet_multiframe_reciprocal",
            "calibrated_geometry": False,
            "similarity_threshold": self.similarity_threshold,
            "margin_threshold": self.margin_threshold,
            "max_track_gap_seconds": self.max_track_gap_seconds,
            "sole_candidate_threshold": self.sole_candidate_threshold,
            "last_decisions": [dict(value) for value in self.last_decisions.values()],
        }

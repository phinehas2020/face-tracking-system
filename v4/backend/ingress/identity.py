from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .domain import IdentityDecision, LinkStatus, PairEvidence

DEFAULT_WEIGHTS = {
    "bias": -7.25,
    "face_similarity": 6.2,
    "body_similarity": 2.35,
    "face_quality": 1.15,
    "body_quality": 0.55,
    "supporting_frames": 0.9,
    "time_proximity": 0.65,
    "topology_score": 1.3,
    "candidate_margin": 1.5,
}


@dataclass(slots=True)
class FusionModel:
    """Small, inspectable same-person model designed for per-event calibration."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    auto_merge_threshold: float = 0.985
    auto_new_threshold: float = 0.12

    @classmethod
    def load(cls, path: Path | None) -> FusionModel:
        if path is None or not path.exists():
            return cls()
        payload = json.loads(path.read_text())
        return cls(
            weights={**DEFAULT_WEIGHTS, **payload.get("weights", {})},
            auto_merge_threshold=float(payload.get("auto_merge_threshold", 0.985)),
            auto_new_threshold=float(payload.get("auto_new_threshold", 0.12)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "auto_merge_threshold": self.auto_merge_threshold,
                    "auto_new_threshold": self.auto_new_threshold,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def score(self, evidence: PairEvidence) -> IdentityDecision:
        if evidence.impossible_overlap:
            return IdentityDecision(
                probability=0.0,
                status=LinkStatus.CONFIRMED_DIFFERENT,
                explanation="The observations overlap in time and cannot be the same person.",
                feature_contributions={"impossible_overlap": -1.0},
            )

        features = evidence.as_features()
        contributions = {
            name: value * self.weights.get(name, 0.0) for name, value in features.items()
        }
        logit = self.weights["bias"] + sum(contributions.values())
        probability = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))

        if probability >= self.auto_merge_threshold:
            status = LinkStatus.AUTO_MERGED
        elif probability <= self.auto_new_threshold:
            status = LinkStatus.CONFIRMED_DIFFERENT
        else:
            status = LinkStatus.REVIEW

        strongest = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        explanation = ", ".join(name.replace("_", " ") for name, _ in strongest)
        return IdentityDecision(
            probability=round(probability, 6),
            status=status,
            explanation=f"Decision driven by {explanation}.",
            feature_contributions={name: round(value, 4) for name, value in contributions.items()},
        )


class DisjointSet:
    """Reversible decisions live in storage; this rebuilds clusters deterministically."""

    def __init__(self, members: list[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def clusters(self) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for member in self.parent:
            output.setdefault(self.find(member), []).append(member)
        return output

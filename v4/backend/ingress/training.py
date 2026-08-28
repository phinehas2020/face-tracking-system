from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .domain import PairEvidence
from .identity import DEFAULT_WEIGHTS, FusionModel

FEATURES = tuple(name for name in DEFAULT_WEIGHTS if name != "bias")


@dataclass(frozen=True, slots=True)
class LabeledPair:
    evidence: PairEvidence
    same_person: bool

    @classmethod
    def load_csv(cls, path: Path) -> list[LabeledPair]:
        pairs: list[LabeledPair] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pairs.append(
                    cls(
                        evidence=PairEvidence(
                            face_similarity=float(row["face_similarity"]),
                            body_similarity=float(row["body_similarity"]),
                            face_quality=float(row["face_quality"]),
                            body_quality=float(row["body_quality"]),
                            supporting_frames=int(row["supporting_frames"]),
                            time_delta_seconds=float(row["time_delta_seconds"]),
                            topology_score=float(row["topology_score"]),
                            candidate_margin=float(row["candidate_margin"]),
                            impossible_overlap=row.get("impossible_overlap", "false").lower()
                            in {"1", "true", "yes"},
                        ),
                        same_person=row["same_person"].lower() in {"1", "true", "yes", "same"},
                    )
                )
        if len(pairs) < 20:
            raise ValueError("at least 20 reviewed pairs are required")
        return pairs


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: FusionModel
    metrics: dict[str, float | int]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def train_fusion_model(
    pairs: list[LabeledPair], *, seed: int = 41, epochs: int = 800, learning_rate: float = 0.04
) -> TrainingResult:
    """Fit a deterministic logistic fusion layer with a false-merge-heavy loss.

    The large negative-class weight is deliberate: merging two people is more damaging
    to an event-wide unique count than sending an uncertain pair to review.
    """

    if len(pairs) < 20:
        raise ValueError("at least 20 reviewed pairs are required")
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8))
    training = shuffled[:split]
    validation = shuffled[split:] or shuffled[-1:]
    weights = dict(DEFAULT_WEIGHTS)

    for _ in range(epochs):
        gradients = {name: 0.0 for name in weights}
        for pair in training:
            features = pair.evidence.as_features()
            target = 1.0 if pair.same_person and not pair.evidence.impossible_overlap else 0.0
            probability = _sigmoid(
                weights["bias"]
                + sum(weights[name] * float(features[name]) for name in FEATURES)
            )
            example_weight = 5.0 if target == 0.0 else 1.0
            error = (probability - target) * example_weight
            gradients["bias"] += error
            for name in FEATURES:
                gradients[name] += error * float(features[name])
        scale = learning_rate / len(training)
        for name in weights:
            regularizer = 0.0005 * weights[name] if name != "bias" else 0.0
            weights[name] -= scale * (gradients[name] + regularizer)

    probabilities = []
    for pair in validation:
        features = pair.evidence.as_features()
        probability = _sigmoid(
            weights["bias"] + sum(weights[name] * float(features[name]) for name in FEATURES)
        )
        probabilities.append(
            (probability, pair.same_person and not pair.evidence.impossible_overlap)
        )

    # Select an automatic-merge threshold with zero validation false merges when possible.
    negatives = [score for score, label in probabilities if not label]
    auto_merge = min(0.999, max(0.95, (max(negatives) + 0.01) if negatives else 0.985))
    false_merges = sum(score >= auto_merge and not label for score, label in probabilities)
    false_splits = sum(score <= 0.12 and label for score, label in probabilities)
    reviewed = sum(0.12 < score < auto_merge for score, _ in probabilities)
    correct = sum((score >= 0.5) == label for score, label in probabilities)
    model = FusionModel(weights=weights, auto_merge_threshold=auto_merge, auto_new_threshold=0.12)
    return TrainingResult(
        model=model,
        metrics={
            "training_pairs": len(training),
            "validation_pairs": len(validation),
            "validation_accuracy": round(correct / len(validation), 4),
            "false_merges": false_merges,
            "false_splits": false_splits,
            "sent_to_review": reviewed,
            "auto_merge_threshold": round(auto_merge, 6),
        },
    )

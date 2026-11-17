"""Identity clustering quality against labelled truth.

Pairwise precision is the false-merge measure and pairwise recall the
false-split measure. B-cubed averages per item, so one large wrong cluster
cannot dominate. Count error is what the attendance figure inherits.
"""

from __future__ import annotations

from collections import Counter


def clustering_metrics(predicted: dict[str, object], truth: dict[str, object]) -> dict:
    items = sorted(set(predicted) & set(truth))
    if not items:
        raise ValueError("No items have both a prediction and a truth label")
    joint = Counter((predicted[item], truth[item]) for item in items)
    by_cluster = Counter(predicted[item] for item in items)
    by_person = Counter(truth[item] for item in items)

    def pairs(count: int) -> int:
        return count * (count - 1) // 2

    together = sum(pairs(count) for count in joint.values())
    predicted_pairs = sum(pairs(count) for count in by_cluster.values())
    true_pairs = sum(pairs(count) for count in by_person.values())
    precision = together / predicted_pairs if predicted_pairs else 1.0
    recall = together / true_pairs if true_pairs else 1.0
    b3_precision = sum(count * count / by_cluster[c] for (c, _), count in joint.items())
    b3_recall = sum(count * count / by_person[p] for (_, p), count in joint.items())
    b3_precision, b3_recall = b3_precision / len(items), b3_recall / len(items)

    def f1(a: float, b: float) -> float:
        return 2 * a * b / (a + b) if a + b else 0.0

    return {
        "items": len(items),
        "true_people": len(by_person),
        "predicted_people": len(by_cluster),
        "count_error": len(by_cluster) - len(by_person),
        "false_merge_pairs": predicted_pairs - together,
        "false_split_pairs": true_pairs - together,
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_f1": round(f1(precision, recall), 4),
        "bcubed_precision": round(b3_precision, 4),
        "bcubed_recall": round(b3_recall, 4),
        "bcubed_f1": round(f1(b3_precision, b3_recall), 4),
    }

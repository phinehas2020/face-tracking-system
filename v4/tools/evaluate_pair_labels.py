"""Evaluate reviewed, frame-bounded counterpart labels without fitting on validation."""

import argparse
import json
from pathlib import Path

import numpy as np
from ingress.paired_appearance import AppearancePairMatcher
from ingress.paired_geometry import ViewObservation

p = argparse.ArgumentParser()
p.add_argument("--observations", type=Path, required=True)
p.add_argument("--labels", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
scenes = json.loads(a.observations.read_text())
labels = json.loads(a.labels.read_text())["cases"]
labels = [e for e in labels if "-s" not in e["case"]]
results = []
for e in labels:
    views = []
    for row in scenes[e["scene"]]:
        w, h = row["size"]
        views.append(
            [
                ViewObservation(
                    row["role"] + ":" + t["id"],
                    ((t["bbox"][0] + t["bbox"][2]) / 2 / w, t["bbox"][3] / h),
                    np.asarray(t["reid"]),
                    row["frame"],
                    row["clock"],
                )
                for t in row["tracks"]
            ]
        )
    # Single-frame candidate evaluation; deployed linking separately requires 2 fresh frames.
    candidates = AppearancePairMatcher(confirmations=1).update(*views)
    actual = next(
        (
            m["side"].split(":", 1)[1]
            for m in candidates
            if m["front"] == "front:" + e["views"]["front"]["track"]
        ),
        None,
    )
    truth = e["views"]["side"]["track"]
    results.append(
        {
            "case": e["case"],
            "split": e["split"],
            "expected": truth,
            "actual": actual,
            "result": "correct" if actual == truth else "unresolved" if actual is None else "wrong",
        }
    )
summary = {
    split: {
        outcome: sum(r["split"] == split and r["result"] == outcome for r in results)
        for outcome in ["correct", "wrong", "unresolved"]
    }
    for split in ["train", "validation"]
}
report = {
    "method": "OSNet x1.0 MSMT17; independent threshold/margin selected on training cases",
    "parameters": {
        "min_similarity": 0.55,
        "margin": 0.08,
        "max_skew": 0.8,
        "deployed_confirmations": 2,
    },
    "spatial_projection_enabled": False,
    "cases": results,
    "summary": summary,
    "limitation": (
        "Small reviewed counterpart-label set, not a population accuracy estimate "
        "or ground-truth attendance tally."
    ),
}
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(summary))

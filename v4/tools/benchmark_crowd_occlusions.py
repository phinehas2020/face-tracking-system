"""Stress-test tracker association by hiding a moving person's cached boxes for eight frames.

Input JSONL rows have role, frame, clock, size [width,height], and boxes
[[source_id,x1,y1,x2,y2,score],...]. Source IDs label the test and never reach the tracker.
These are conditional recovery scores on previously tracked people.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def cases(rows):
    # Keep each trial inside its own temporal half. Use moving people with a nearby person.
    for split, section in zip(("development", "later"), np.array_split(np.arange(len(rows)), 2),
                              strict=True):
        for i in range(int(section[0]) + 8, int(section[-1]) - 11, 24):
            clip = rows[i-8:i+12]
            gaps = np.diff([r["clock"] for r in clip])
            if np.any(gaps <= 0) or np.max(gaps) > .2:
                continue
            visible = [{b[0]: np.asarray(b[1:5]) for b in r["boxes"]} for r in clip]
            for key in sorted(set.intersection(*(set(frame) for frame in visible))):
                a, b = visible[7][key], visible[16][key]
                width = max(a[2] - a[0], 1)
                shift = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
                neighbors = [box for k, box in visible[7].items() if k != key and
                             np.all(np.minimum(a[2:], box[2:]) > np.maximum(a[:2], box[:2]))]
                if shift >= .15 * width and neighbors:
                    yield split, clip, key


def replay(clip, key, tracker_class, config, hide):
    from ingress.vision.crowd import CrowdConsensus
    from ultralytics.engine.results import Boxes

    consensus = tracker_class is CrowdConsensus
    tracker = CrowdConsensus() if consensus else tracker_class(args=SimpleNamespace(**config))
    previous = None
    recovered = set()
    other_claim = False
    for i, row in enumerate(clip):
        obs = [b for b in row["boxes"] if not (hide and 8 <= i < 16 and b[0] == key)]
        data = np.asarray([[*b[1:], 0] for b in obs], dtype=np.float32).reshape(-1, 6)
        boxes = Boxes(data, tuple(reversed(row["size"])))
        if consensus:
            colors = [np.asarray(row["colors"][b[0]], dtype=np.float32) for b in obs]
            results = tracker.update(boxes, row["clock"], colors)
        else:
            results = tracker.update(boxes)
        assigned = {obs[int(t[-1])][0]: int(t[4]) for t in results if 0 <= t[-1] < len(obs)}
        if i == 7:
            previous = assigned.get(key)
        if i > 7 and previous is not None:
            if i >= 16 and key in assigned:
                recovered.add(assigned[key])
            other_claim |= any(t == previous and k != key for k, t in assigned.items())
    if previous is None:
        return "unestablished"
    if other_claim:
        return "prior_id_on_neighbor"
    if recovered == {previous}:
        return "retained"
    return "fragmented" if recovered else "missing"


def main():
    import ultralytics
    import yaml
    from ingress.vision.crowd import CrowdConsensus
    from ultralytics.trackers.track import TRACKER_MAP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tracker", action="append", required=True,
                        help="Ultralytics tracker name or local YAML")
    args = parser.parse_args()
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rows = [r for r in rows if r["role"] == "side"]
    trials = list(cases(rows))
    if not trials:
        parser.error("No eligible moving, overlapping tracks")
    report = {}
    for name in args.tracker:
        path = Path(name)
        if not path.is_file() and name != "consensus":
            path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
        cfg = ({"tracker_type": "consensus"} if name == "consensus"
               else yaml.safe_load(path.read_text()))
        if cfg.get("with_reid"):
            parser.error("Cached boxes have no image features for ReID")
        cls = CrowdConsensus if name == "consensus" else TRACKER_MAP[cfg["tracker_type"]]
        results = []
        for split, clip, key in trials:
            clear = replay(clip, key, cls, cfg, False)
            masked = replay(clip, key, cls, cfg, True)
            results.append({"split": split, "frame": clip[8]["frame"], "track": key,
                            "clear": clear, "masked": masked})
        report[name] = {"config": cfg, "trials": results}
        for split in ("development", "later"):
            subset = [r for r in results if r["split"] == split]
            print(name, split, len(subset), dict(Counter(r["masked"] for r in subset)), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

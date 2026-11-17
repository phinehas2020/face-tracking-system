"""Re-evaluate line placement from frozen face/body evidence, without re-running models."""

from __future__ import annotations

import json
from pathlib import Path

from .live import CrossingCounter, LiveCalibration


def evaluate_counting(path: Path, position: float, require_body_motion: bool) -> dict:
    counter = None
    current_clip = None
    passages = []
    frames = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        frames += 1
        if row["clip"] != current_clip:
            current_clip = row["clip"]
            counter = CrossingCounter(
                LiveCalibration(
                    axis="horizontal",
                    position=position,
                    require_body_motion=require_body_motion,
                )
            )
        for passage in passages:
            if passage["identity"] or row["clock"] - passage["clock"] > 10:
                continue
            link = row["links"].get(passage["key"])
            if link and (not link["after"] or passage["at"] >= link["after"]):
                passage["identity"] = link["id"]
        for key, x, y, head_y in row["points"]:
            direction = counter.update(key, (x, y), row["clock"], body_anchor=(x, head_y))
            if direction:
                passages.append(
                    {
                        "key": key,
                        "at": row["at"],
                        "clock": row["clock"],
                        "direction": direction,
                        "identity": row["links"].get(key, {}).get("id"),
                        "view": row["source"]["label"],
                    }
                )
        counter.expire(row["clock"])
    entries = [p for p in passages if p["direction"] == "in"]
    linked = [p for p in entries if p["identity"]]
    unique = len({p["identity"] for p in linked})
    return {
        "position": position,
        "require_body_motion": require_body_motion,
        "frames": frames,
        "entries": len(entries),
        "exits": len(passages) - len(entries),
        "linked_entries": len(linked),
        "unresolved_entries": len(entries) - len(linked),
        "face_linked_entrants": unique,
        "repeat_entries": len(linked) - unique,
        "views": {
            view: {
                "entries": sum(p["view"] == view for p in entries),
                "linked_entries": sum(p["view"] == view for p in linked),
            }
            for view in sorted({p["view"] for p in passages})
        },
    }

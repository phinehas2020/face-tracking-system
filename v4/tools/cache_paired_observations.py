"""Cache local model observations for repeatable paired-camera calibration.

Contains face templates: keep this file on the recording Mac, mode 0600.
No network uploads. Original recordings are never changed.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import cv2
from ingress.paired import PairedPlan
from ingress.vision.adapters import LiveFaceEncoder, UltralyticsPersonTracker
from ingress.vision.continuity import TrackContinuity
from ingress.vision.contracts import FramePacket

parser = argparse.ArgumentParser()
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
os.umask(0o077)
plan = PairedPlan.model_validate(json.loads(a.plan.read_text())["paired_replay"])
a.output.mkdir(parents=True, exist_ok=True)
caps = {v.role: cv2.VideoCapture(str(v.path)) for v in plan.views}
trackers = {r: UltralyticsPersonTracker(a.model) for r in caps}
continuity = {r: TrackContinuity(max_gap=3 if r == "front" else 1.2) for r in caps}
encoder = LiveFaceEncoder()
last_scan = {r: -1e9 for r in caps}
events = sorted(
    (x["clock"], v.role, x["frame"])
    for v in plan.views
    for x in v.clock_index
    if plan.start_clock <= x["clock"] <= plan.start_clock + plan.duration_seconds
)
metadata = []
with gzip.open(a.output / "observations.jsonl.gz", "wt") as output:
    for i, (clock, role, idx) in enumerate(events):
        cap = caps[role]
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"decode failed {role} {idx}")
        tracks = continuity[role].update(
            trackers[role].track(FramePacket(role, idx, 0, "", im)), im, clock
        )
        scan = role == "front" or clock - last_scan[role] >= 0.2 - 1e-8
        faces = encoder.detect(im) if scan else []
        if scan:
            last_scan[role] = clock
        row = {
            "role": role,
            "frame": idx,
            "clock": clock,
            "size": [im.shape[1], im.shape[0]],
            "scan": scan,
            "tracks": [
                {
                    "id": t.track_id,
                    "bbox": list(t.bbox),
                    "confidence": t.confidence,
                    "age": t.age_frames,
                    "appearance": continuity[role].appearance(im, t.bbox).tolist(),
                }
                for t in tracks
            ],
            "faces": [
                {"bbox": list(f.bbox), "quality": f.quality, "embedding": f.embedding.tolist()}
                for f in faces
            ],
        }
        output.write(json.dumps(row) + "\n")
        metadata.append(
            {**row, "faces": [{"bbox": f["bbox"], "quality": f["quality"]} for f in row["faces"]]}
        )
        if i % 100 == 0:
            print(f"{i}/{len(events)}", flush=True)
for cap in caps.values():
    cap.release()
(a.output / "metadata.json").write_text(json.dumps(metadata))
(a.output / "manifest.json").write_text(
    json.dumps(
        {
            "views": [
                {"role": v.role, "path": str(v.path), "bytes": v.path.stat().st_size}
                for v in plan.views
            ],
            "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
            "face_encoder": "buffalo_l/w600k_r50",
        },
        indent=2,
    )
)
print("complete", len(events), flush=True)

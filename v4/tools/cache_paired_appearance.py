"""Add person appearance vectors to an existing local observation cache."""

import argparse
import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ingress.vision.contracts import Track
from ingress.vision.reid import AppearanceEncoder

p = argparse.ArgumentParser()
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--weights", type=Path, required=True)
a = p.parse_args()
os.umask(0o077)
plan = json.loads(a.plan.read_text())["paired_replay"]
caps = {v["role"]: cv2.VideoCapture(v["path"]) for v in plan["views"]}
model = AppearanceEncoder(a.weights)
keys = []
values = []
with gzip.open(a.cache / "observations.jsonl.gz", "rt") as inp:
    for i, line in enumerate(inp):
        row = json.loads(line)
        cap = caps[row["role"]]
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != row["frame"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ok, im = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode appearance source")
        tracks = [
            Track(t["id"], tuple(t["bbox"]), t["confidence"], t["age"]) for t in row["tracks"]
        ]
        vectors = model.encode(im, tracks)
        for key, value in vectors.items():
            keys.append(f"{row['role']}:{row['frame']}:{key}")
            values.append(value)
        if i % 100 == 0:
            print(i, flush=True)
np.savez_compressed(
    a.cache / "appearance.npz", keys=np.asarray(keys), features=np.asarray(values, dtype=np.float32)
)
print("complete", len(keys), flush=True)

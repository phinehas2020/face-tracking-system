#!/usr/bin/env python3
"""Who came back: people seen by face on separate occasions during one event.

A station's "repeat entries" tile counts line crossings that the live matcher tied
to an enrolled face. Where few crossings get a face, that tile says almost nothing
about how many people returned. This tool answers the wider question offline, from
a copy of the station's data, without touching the station:

    returning_visitors.py --data-root COPY --event EVENT --recognizer kprpe_wf12m \\
        --shadow-db COPY/face-shadow/shadow.db --shadow-model kprpe_wf12m-scrfd

1. Fit the resolver's score calibration to this station: different people are faces
   in one frame (from the shadow recognizer's database), the same person is two
   samples of one stored tracklet.
2. Resolve the whole event (ingress/resolution.py) with that calibration.
3. Split each person's sightings into occasions wherever they were not seen for
   --gap-minutes, and report who has two or more.

"Seen again" is not "entered again": someone who sat in view all evening and looked
up twice counts. Nothing here is checked against labels. --sheets DIR writes face
strips to look at instead; that needs the spooled crops (--spool) and the recognizer
(--models-root), and only uses a crop that is provably the stored sample (cosine
at least 0.9 under the same recognizer).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def same_frame_scores(path: Path, model: str, max_yaw: float, max_pitch: float) -> list[float]:
    """Similarities of two different faces in one analysed frame: certainly two people."""
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT f.track,f.call,v.vector FROM faces f JOIN vectors v ON v.face_id=f.id "
        "WHERE v.model=? AND ABS(f.yaw)<=? AND ABS(f.pitch)<=?", (model, max_yaw, max_pitch)).fetchall()
    frames = defaultdict(list)
    for track, call, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        frames[(track.rsplit("-", 1)[0], call)].append((track, vector / np.linalg.norm(vector)))
    return [float(a @ b) for faces in frames.values() for i, (left, a) in enumerate(faces)
            for right, b in faces[i + 1:] if left != right]


def occasions(tracklets, gap: float) -> list[list]:
    """One person's tracklets in time order, split where nothing was seen for gap seconds."""
    stamp = lambda text: datetime.fromisoformat(text).timestamp()  # noqa: E731
    ordered = sorted(tracklets, key=lambda t: t.first_observed_at)
    groups, last = [[ordered[0]]], stamp(ordered[0].last_observed_at)
    for tracklet in ordered[1:]:
        if stamp(tracklet.first_observed_at) - last > gap:
            groups.append([])
        groups[-1].append(tracklet)
        last = max(last, stamp(tracklet.last_observed_at))
    return groups


def template(tracklets) -> tuple[np.ndarray, int]:
    samples = np.unique(np.concatenate([t.embeddings for t in tracklets]), axis=0)
    mean = samples.mean(0)
    return mean / max(float(np.linalg.norm(mean)), 1e-9), len(samples)


def crop_index(shadow_db: Path, spool: Path, models_root: Path, model: str, cache: Path):
    """KP-RPE (or other) templates of the crops the live pose gate accepted, by face id."""
    db = sqlite3.connect(shadow_db)
    meta = {row[0]: row[1:] for row in db.execute(
        "SELECT id,chunk,position,camera,t,width,det FROM faces WHERE accepted=1")}
    if cache.exists():
        saved = np.load(cache)
        return meta, saved["ids"], saved["vectors"]
    sys.path.insert(0, str(ROOT / "tools"))
    import torch
    from benchmark_face_recognizers import Embedder

    embedder = Embedder(model, models_root, "mps" if torch.backends.mps.is_available() else "cpu")
    by_chunk = defaultdict(list)
    for face_id, row in meta.items():
        by_chunk[row[0]].append((face_id, row[1]))
    ids, vectors = [], []
    for chunk, members in sorted(by_chunk.items()):
        if not (spool / chunk).exists():
            continue
        with np.load(spool / chunk) as data:
            positions = [position for _id, position in members]
            vectors.append(embedder(data["aligned"][positions], data["kps"][positions]))
        ids += [face_id for face_id, _position in members]
    ids, vectors = np.asarray(ids), np.concatenate(vectors)
    np.savez(cache, ids=ids, vectors=vectors)
    return meta, ids, vectors


def write_sheets(people, tracklets, args, out: Path) -> dict:
    from PIL import Image, ImageDraw

    spool = Path(args.spool)
    meta, ids, vectors = crop_index(Path(args.shadow_db), spool, Path(args.models_root),
                                    args.shadow_model, out / "accepted-crop-templates.npz")
    samples = np.unique(np.concatenate([t.embeddings for t in tracklets]), axis=0)
    crop_of = {}
    for start in range(0, len(samples), 2000):
        scores = samples[start:start + 2000] @ vectors.T
        for sample, best, score in zip(samples[start:start + 2000], scores.argmax(1), scores.max(1), strict=True):
            if score >= 0.9:
                crop_of[sample.tobytes()] = int(ids[best])
    rows = []
    for person in people:
        shots = []
        for group in person["groups"]:
            found = {crop_of[s.tobytes()] for t in group for s in t.embeddings if s.tobytes() in crop_of}
            if found:
                shots.append((max(found, key=lambda i: meta[i][4] * meta[i][5]), group[0].first_observed_at))
        if len(shots) >= 2:
            rows.append((person, shots))
    rows.sort(key=lambda row: row[0]["lowest_similarity"])
    chunks: dict[str, np.ndarray] = {}

    def crop(face_id: int):
        chunk, position = meta[face_id][0], meta[face_id][1]
        if chunk not in chunks:
            chunks[chunk] = np.load(spool / chunk)["aligned"]
        return Image.fromarray(chunks[chunk][position][:, :, ::-1])

    chosen = {"weakest": rows[:args.sheet_rows],
              "random": random.Random(0).sample(rows, min(args.sheet_rows, len(rows)))}
    for name, picked in chosen.items():
        if not picked:
            continue
        columns = min(6, max(len(shots) for _person, shots in picked))
        sheet = Image.new("RGB", (84 + columns * 116, len(picked) * 130), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (person, shots) in enumerate(picked):
            y = index * 130
            draw.text((2, y + 30), f"#{index + 1}\n{len(person['groups'])} times\n"
                      f"min {person['lowest_similarity']:.2f}", fill="black")
            for column, (face_id, seen) in enumerate(shots[:columns]):
                sheet.paste(crop(face_id), (84 + column * 116, y))
                local = datetime.fromisoformat(seen).astimezone().strftime("%H:%M")
                draw.text((86 + column * 116, y + 114), f"{local} {meta[face_id][2][5:]}", fill="black")
        sheet.save(out / f"seen-again-{name}.jpg", quality=88)
    return {"samples_with_a_proven_crop": len(crop_of), "samples": len(samples),
            "people_with_crops_on_two_occasions": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder holding a copy of the station's ingress.db")
    parser.add_argument("--event", required=True)
    parser.add_argument("--recognizer", default="buffalo_l", help="the station's INGRESS_FACE_RECOGNIZER")
    parser.add_argument("--shadow-db", required=True)
    parser.add_argument("--shadow-model", required=True, help="the same recognizer's name in the shadow database")
    parser.add_argument("--gap-minutes", type=float, default=15.0)
    parser.add_argument("--max-yaw", type=float, default=35.0, help="same-frame faces used as different people")
    parser.add_argument("--max-pitch", type=float, default=25.0)
    parser.add_argument("--max-units", type=int, default=20000, help="dense: about 6 n^2 x 8 bytes of memory")
    parser.add_argument("--sheets", help="folder for face strips of the people found")
    parser.add_argument("--spool", help="folder of chunk-*.npz crops (for --sheets)")
    parser.add_argument("--models-root", help="recognizer repositories (for --sheets)")
    parser.add_argument("--sheet-rows", type=int, default=16)
    parser.add_argument("--save-calibration", action="store_true",
                        help="write resolution-calibration.json into --data-root")
    args = parser.parse_args()

    os.environ["INGRESS_FACE_RECOGNIZER"] = args.recognizer  # read when ingress is imported
    sys.path.insert(0, str(ROOT / "backend"))
    from ingress import resolution
    from ingress.evidence import load_tracklets
    from ingress.face_registry import MIN_SAMPLES
    from ingress.store import EventStore

    store = EventStore(Path(args.data_root))
    impostors = same_frame_scores(Path(args.shadow_db), args.shadow_model, args.max_yaw, args.max_pitch)
    calibration = resolution.fit_calibration(
        *resolution.constraint_pairs(store, args.event, impostors), method="constraints")
    if args.save_calibration:
        calibration.save(resolution.calibration_path(store))
    at = lambda p: round((resolution._logit(p) - calibration.intercept) / calibration.slope, 3)  # noqa: E731
    config = resolution.ResolverConfig(max_units=args.max_units)
    resolved = resolution.resolve(store, args.event, config=config, calibration=calibration)

    tracklets = load_tracklets(store, args.event, resolution.FACE_MODEL)
    by_person = defaultdict(list)
    for tracklet in tracklets:
        person = resolved.assignments.get(f"tracklet:{tracklet.id}")
        if person:
            by_person[person].append(tracklet)
    established = {key: members for key, members in by_person.items()
                   if key.startswith("identity:") or template(members)[1] >= MIN_SAMPLES}
    people, strong = [], 0
    for key, members in established.items():
        groups = occasions(members, args.gap_minutes * 60)
        if len(groups) < 2:
            continue
        templates = [template(group) for group in groups]
        pairs = [float(a[0] @ b[0]) for i, a in enumerate(templates) for b in templates[i + 1:]]
        solid = [vector for vector, count in templates if count >= MIN_SAMPLES]
        strong += len(solid) >= 2 and min(
            float(a @ b) for i, a in enumerate(solid) for b in solid[i + 1:]) >= 0.6
        people.append({"person": key, "groups": groups, "lowest_similarity": min(pairs)})

    scores = np.asarray(impostors)
    report = {
        "event_id": args.event,
        "model": resolution.FACE_MODEL,
        "verified_against_labels": False,
        "calibration": {
            "different_people_pairs": len(impostors),
            "different_people_highest": round(float(scores.max()), 3),
            "different_people_p999": round(float(np.quantile(scores, 0.999)), 3),
            "even_odds_at_similarity": at(0.5),
            "automatic_merge_at_similarity": at(config.auto_probability),
        },
        "inputs": resolved.report["inputs"],
        "clusters": resolved.report["clusters"],
        "entries": resolved.report["counts"],
        "people_with_three_or_more_faces": len(established),
        "gap_minutes": args.gap_minutes,
        "seen_on_two_or_more_occasions": len(people),
        # Both occasions have three or more faces and their templates agree at 0.6 or more.
        "of_those_with_strong_evidence_each_time": int(strong),
        "occasions_per_person": {
            str(count): sum(len(p["groups"]) == count for p in people)
            for count in sorted({len(p["groups"]) for p in people})
        },
    }
    if args.sheets:
        if not (args.spool and args.models_root):
            parser.error("--sheets needs --spool and --models-root")
        out = Path(args.sheets)
        out.mkdir(parents=True, exist_ok=True)
        report["sheets"] = write_sheets(people, tracklets, args, out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

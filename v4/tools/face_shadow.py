#!/usr/bin/env python3
"""Run other face recognizers beside the live one, on the faces the station saw.

The live service saves an aligned crop of every face with a readable pose, turned
or not, into the event's face-shadow folder (ingress/face_spool.py). This process
embeds those crops with the recognizers named in --models and writes report.json
there, which the station serves at /api/live/shadow. It is a separate, low-priority,
CPU-only process: it shares no memory, no GPU and no lock with counting, and it can
be stopped or restarted at any time without touching the station.

    face_shadow.py --spool DATA/events/EVENT/face-shadow --models-root MODELS \\
                   --models adaface_ir101_wf12m,kprpe_wf12m-scrfd

The report scores each recognizer the way benchmark_face_recognizers.py does: two
crops of one followed face must match (genuine), two faces in one frame must not
(impostor). Nothing here changes who the live station recognizes.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_face_recognizers import YAW_EDGES, YAW_NAMES, Embedder, score_arrays  # noqa: E402

LIVE = "buffalo_l"
# Evidence needed before a number is shown, in distinct pairs of people seen in one picture
# (two people standing together give many crop pairs but are one piece of evidence).
MIN_PEOPLE_PAIRS = 100       # match rates
MIN_PEOPLE_PAIRS_COUNT = 300  # the people estimate, whose threshold sits in the tail
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (name TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY, chunk TEXT NOT NULL, position INTEGER NOT NULL, camera TEXT, track TEXT,
    call INTEGER, t REAL, yaw REAL, pitch REAL, width REAL, det REAL, accepted INTEGER);
CREATE TABLE IF NOT EXISTS vectors (
    face_id INTEGER NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (face_id, model));
"""


def ingest(db: sqlite3.Connection, spool: Path) -> int:
    """Register new chunks and keep the live recognizer's own template for each face."""
    done = {row[0] for row in db.execute("SELECT name FROM chunks")}
    added = 0
    for path in sorted(spool.glob("chunk-*.npz")):
        if path.name in done:
            continue
        with np.load(path) as data:
            meta, live = json.loads(str(data["meta"])), data["embedding"].astype(np.float32)
        with db:
            for position, (row, vector) in enumerate(zip(meta, live, strict=True)):
                face_id = db.execute(
                    "INSERT INTO faces (chunk,position,camera,track,call,t,yaw,pitch,width,det,accepted) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (path.name, position, row["camera"], row["track"], row["call"], row["t"], row["yaw"],
                     row["pitch"], row["width"], row["det"], int(row["accepted"]))).lastrowid
                db.execute("INSERT INTO vectors VALUES (?,?,?)", (face_id, LIVE, vector.tobytes()))
            db.execute("INSERT INTO chunks VALUES (?)", (path.name,))
        added += len(meta)
    return added


def embed_missing(db: sqlite3.Connection, spool: Path, embedder: Embedder, timing: dict) -> int:
    rows = db.execute(
        "SELECT id,chunk,position FROM faces WHERE NOT EXISTS "
        "(SELECT 1 FROM vectors WHERE face_id=faces.id AND model=?) ORDER BY chunk,position",
        (embedder.name,)).fetchall()
    by_chunk: dict[str, list] = {}
    for face_id, chunk, position in rows:
        by_chunk.setdefault(chunk, []).append((face_id, position))
    count = 0
    for chunk, members in by_chunk.items():
        path = spool / chunk
        if not path.exists():
            continue
        with np.load(path) as data:
            positions = [position for _id, position in members]
            aligned, kps = data["aligned"][positions], data["kps"][positions]
        started = time.monotonic()
        vectors = embedder(aligned, kps)
        timing["seconds"] = timing.get("seconds", 0.0) + time.monotonic() - started
        timing["faces"] = timing.get("faces", 0) + len(members)
        with db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                           [(face_id, embedder.name, vector.tobytes())
                            for (face_id, _p), vector in zip(members, vectors, strict=True)])
        count += len(members)
    return count


def people(track, vectors, mask, threshold: float) -> dict:
    """Distinct people if followed faces were merged at this threshold (greedy, in time order)."""
    centres, used = [], 0
    for name in dict.fromkeys(track[mask]):          # first-seen order
        member = vectors[mask & (track == name)]
        if len(member) < 3:
            continue
        used += 1
        mean = member.mean(0)
        mean /= max(float(np.linalg.norm(mean)), 1e-9)
        if not centres or max(float(mean @ centre) for centre in centres) < threshold:
            centres.append(mean)
    return {"people": len(centres), "tracks": used}


def report(db: sqlite3.Connection, models: list[str], window: int, timings: dict, status: dict) -> dict:
    faces = db.execute(
        "SELECT id,camera,track,call,t,yaw,width,accepted FROM faces ORDER BY id DESC LIMIT ?", (window,)).fetchall()[::-1]
    total = db.execute("SELECT COUNT(*), COUNT(DISTINCT track) FROM faces").fetchone()
    out = {"updated_at": datetime.now(UTC).isoformat(), "live_model": LIVE, "shadow_models": models,
           "faces": total[0], "tracks": total[1], "scored_faces": len(faces), "min_people_pairs": MIN_PEOPLE_PAIRS,
           "people_pairs": 0, "rated": False,
           "status": status, "models": {}}
    if not faces:
        return out
    ids = np.array([row[0] for row in faces])
    track = np.array([row[2] for row in faces])
    yaw = np.array([row[5] for row in faces], dtype=float)
    accepted = np.array([bool(row[7]) for row in faces])
    # Faces in one frame share a camera, a service run and a scan number.
    sessions = {name: index for index, name in enumerate(dict.fromkeys(t.rsplit("-", 1)[0] for t in track))}
    video = np.array([sessions[t.rsplit("-", 1)[0]] for t in track])
    out["accepted_by_live_gate"] = int(accepted.sum())
    out["by_yaw"] = {name: int(((np.abs(yaw) >= lo) & (np.abs(yaw) < hi)).sum())
                     for name, lo, hi in zip(YAW_NAMES, YAW_EDGES, YAW_EDGES[1:], strict=False)}
    embeddings = {}
    for model in [LIVE, *models]:
        found = dict(db.execute(
            "SELECT face_id,vector FROM vectors WHERE model=? AND face_id>=?", (model, int(ids[0]))))
        if all(int(face_id) in found for face_id in ids):
            embeddings[model] = np.stack([np.frombuffer(found[int(face_id)], dtype=np.float32) for face_id in ids])
    scored = score_arrays(yaw, track, [row[4] for row in faces], video, [row[3] for row in faces],
                          [row[6] for row in faces], embeddings)
    out.update(genuine_pairs=scored["genuine_pairs"], impostor_pairs=scored["impostor_pairs"],
               people_pairs=scored["impostor_track_pairs"],
               rated=scored["impostor_track_pairs"] >= MIN_PEOPLE_PAIRS)
    everyone = np.ones(len(faces), dtype=bool)
    for model, entry in scored["models"].items():
        vectors = embeddings[model]
        row = {"impostor": entry["impostor"], "summary": entry["summary"], "buckets": entry["buckets"]}
        timing = timings.get(model)
        if timing and timing.get("faces"):
            row["ms_per_face"] = round(1000 * timing["seconds"] / timing["faces"], 1)
        if scored["impostor_track_pairs"] >= MIN_PEOPLE_PAIRS_COUNT:
            # An estimate. Merge threshold from this station's own faces: 1 in 1000
            # different-person pairs scores higher.
            pairs = entry["impostor"]
            index = np.flatnonzero(np.diff(video.astype(np.int64) * 10_000_000 + np.array([r[3] for r in faces])) == 0)
            threshold = float(np.quantile((vectors[index] * vectors[index + 1]).sum(1), 0.999)) if len(index) else pairs["max"]
            row["people"] = {"threshold": round(threshold, 3),
                             "frontal_only": people(track, vectors, accepted, threshold),
                             "all_angles": people(track, vectors, everyone, threshold)}
        out["models"][model] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--models", default="adaface_ir101_wf12m")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=20000, help="most recent faces scored")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    try:
        os.nice(10)
    except OSError:
        pass
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(spool / "shadow.db")
    db.executescript(SCHEMA)
    names = [name for name in args.models.split(",") if name]
    embedders, errors, timings = {}, {}, {}
    for name in names:
        try:
            embedders[name] = Embedder(name, Path(args.models_root), "cpu")
        except Exception as exc:                                  # one bad model must not stop the others
            errors[name] = f"{type(exc).__name__}: {exc}"[:200]
            print(f"{name}: failed to load: {errors[name]}", flush=True)
    print(f"loaded {list(embedders)}; watching {spool}", flush=True)
    changed = True
    while True:
        try:
            added = ingest(db, spool)
            embedded = sum(embed_missing(db, spool, embedder, timings.setdefault(name, {}))
                           for name, embedder in embedders.items())
            changed = changed or bool(added or embedded)
            if added or embedded:
                print(f"{datetime.now().strftime('%H:%M:%S')} +{added} faces, {embedded} templates", flush=True)
            status = {"state": "running", "load_errors": errors}
            if changed:
                latest = report(db, list(embedders), args.window, timings, status)
                changed = False
            else:                                                 # heartbeat only
                latest.update(updated_at=datetime.now(UTC).isoformat(), status=status)
            temporary = spool / ".report.json.tmp"
            temporary.write_text(json.dumps(latest))
            temporary.replace(spool / "report.json")
        except Exception as exc:
            print(f"cycle failed: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

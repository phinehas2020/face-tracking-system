#!/usr/bin/env python3
"""Compare face recognizers on booth footage, by head angle, without identity labels.

Three stages, each its own invocation so models with clashing module names
never share a process:

    extract  --video a.mkv [b.mkv ...] --out run/          detect, pose, track, crop
    embed    --run run/ --model NAME --models-root DIR      one recognizer
    score    --run run/                                     table per recognizer

Ground truth without labels:
  * genuine  two faces of one short face track (same head followed frame to
             frame), at least --min-gap seconds apart;
  * impostor two faces in the same frame (certainly different people).
Pairs are grouped by the yaw of the more turned face. The "frontal to turned"
genuine pairs (one face within 20 degrees) are the live use case: a turned face
must match the person's frontal template.

Recognizers: buffalo_l (w600k_r50), antelopev2 (glintr100), adaface_ir101_wf12m,
kprpe_wf12m and kprpe_wf4m (ViT-B + KeyPoint RPE, Kim et al., CVPR 2024), each
KP-RPE model with SCRFD keypoints ("-scrfd") and with its own DFA aligner ("-dfa").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

YAW_EDGES = (0, 20, 35, 50, 70, 181)
YAW_NAMES = ("0-20", "20-35", "35-50", "50-70", "70+")


def bucket(yaw: float) -> int:
    return int(np.searchsorted(YAW_EDGES, abs(yaw), side="right") - 1)


# ---------------------------------------------------------------- extract
def extract(args) -> None:
    import cv2
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l", root=str(Path(args.models_root) / "insightface"),
                       allowed_modules=["detection", "landmark_3d_68"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=0.5)

    rows, aligned, loose, loose_kps = [], [], [], []
    next_track = 0
    for video_index, video in enumerate(args.video):
        cap = cv2.VideoCapture(video)
        native = cap.get(cv2.CAP_PROP_FPS) or 20.0
        if not 1 <= native <= 120:
            native = 20.0
        step = max(1, round(native / args.fps))
        live: list[dict] = []   # tracks seen in the previous sampled frame
        index = -1
        while True:
            ok = cap.grab()
            if not ok:
                break
            index += 1
            if index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            t = index / native
            faces = [f for f in app.get(frame) if (f.bbox[2] - f.bbox[0]) >= args.min_width]
            current = []
            taken = set()
            for face in faces:
                x0, y0, x1, y1 = face.bbox
                best, best_iou = None, 0.0
                for k, tr in enumerate(live):
                    if k in taken:
                        continue
                    a0, b0, a1, b1 = tr["box"]
                    iw = max(0.0, min(x1, a1) - max(x0, a0)); ih = max(0.0, min(y1, b1) - max(y0, b0))
                    inter = iw * ih
                    union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best, best_iou = k, iou
                if best is not None and best_iou >= args.link_iou:
                    taken.add(best)
                    track = live[best]["id"]
                else:
                    track = next_track
                    next_track += 1
                current.append({"id": track, "box": face.bbox.copy()})
                pitch, yaw, roll = (float(v) for v in face.pose)
                side = 1.5 * max(x1 - x0, y1 - y0)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                M = np.array([[160 / side, 0, 80 - cx * 160 / side], [0, 160 / side, 80 - cy * 160 / side]], dtype=np.float32)
                loose.append(cv2.warpAffine(frame, M, (160, 160), borderValue=0))
                loose_kps.append((face.kps @ M[:, :2].T + M[:, 2]).astype(np.float32) / 160.0)
                aligned.append(face_align.norm_crop(frame, face.kps, 112))
                A = face_align.estimate_norm(face.kps, 112)
                kps_al = (face.kps @ A[:, :2].T + A[:, 2]) / 112.0
                rows.append({"video": video_index, "frame": index, "t": round(t, 3), "track": track,
                             "width": float(x1 - x0), "det": float(face.det_score),
                             "pitch": pitch, "yaw": yaw, "roll": roll,
                             "kps_aligned": kps_al.astype(float).round(4).tolist()})
            live = current
        cap.release()
        print(f"{Path(video).name}: {len(rows)} faces so far, {next_track} tracks", flush=True)
    np.savez_compressed(out / "faces.npz", aligned=np.asarray(aligned, dtype=np.uint8),
                        loose=np.asarray(loose, dtype=np.uint8), loose_kps=np.asarray(loose_kps, dtype=np.float32))
    (out / "faces.json").write_text(json.dumps({"videos": [str(v) for v in args.video], "rows": rows}))
    yaws = np.abs([r["yaw"] for r in rows]) if rows else np.zeros(0)
    print(json.dumps({"faces": len(rows), "tracks": next_track,
                      "by_yaw": {n: int(((yaws >= lo) & (yaws < hi)).sum())
                                 for n, lo, hi in zip(YAW_NAMES, YAW_EDGES, YAW_EDGES[1:])}}))


# ------------------------------------------------------------------ embed
def _onnx_embed(path: Path, aligned_bgr: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = []
    for i in range(0, len(aligned_bgr), 64):
        x = aligned_bgr[i:i + 64][..., ::-1].astype(np.float32)   # RGB
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        out.append(session.run(None, {name: np.ascontiguousarray(x)})[0])
    return np.concatenate(out)


def _load_cvl(path: Path):
    from transformers import AutoModel
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, str(path))
    try:
        return AutoModel.from_pretrained(str(path), trust_remote_code=True).eval()
    finally:
        os.chdir(cwd)
        sys.path.pop(0)


def embed(args) -> None:
    import torch
    run = Path(args.run)
    data = np.load(run / "faces.npz")
    aligned, loose, loose_kps = data["aligned"], data["loose"], data["loose_kps"]
    rows = json.loads((run / "faces.json").read_text())["rows"]
    root = Path(args.models_root)
    name = args.model
    if name == "buffalo_l":
        emb = _onnx_embed(root / "insightface/models/buffalo_l/w600k_r50.onnx", aligned)
    elif name == "antelopev2":
        hits = list((root / "insightface/models/antelopev2").rglob("glintr100.onnx"))
        emb = _onnx_embed(hits[0], aligned)
    else:
        device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
        base, _, variant = name.partition("-")
        repo = {"adaface_ir101_wf12m": "cvlface_adaface_ir101_webface12m",
                "kprpe_wf12m": "cvlface_adaface_vit_base_kprpe_webface12m",
                "kprpe_wf4m": "cvlface_adaface_vit_base_kprpe_webface4m"}[base]
        to_tensor = lambda a: ((torch.from_numpy(np.ascontiguousarray(a[..., ::-1])).float() / 255 - 0.5) / 0.5).permute(0, 3, 1, 2)
        kps_al = np.asarray([r["kps_aligned"] for r in rows], dtype=np.float32)
        if variant == "dfa":
            aligner = _load_cvl(root / "cvlface_DFA_mobilenet").to(device)
            for mod in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
                del sys.modules[mod]
        model = _load_cvl(root / repo).to(device)
        out = []
        with torch.no_grad():
            for i in range(0, len(aligned), 32):
                if variant == "dfa":
                    ax, _orig, al, *_ = aligner(to_tensor(loose[i:i + 32]).to(device))
                    y = model(ax, al)
                elif base.startswith("kprpe"):
                    y = model(to_tensor(aligned[i:i + 32]).to(device), torch.from_numpy(kps_al[i:i + 32]).to(device))
                else:
                    y = model(to_tensor(aligned[i:i + 32]).to(device))
                if isinstance(y, (tuple, list)):
                    y = y[0]
                out.append(y.float().cpu().numpy())
        emb = np.concatenate(out)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    np.save(run / f"emb-{name}.npy", emb.astype(np.float32))
    print(name, emb.shape, flush=True)


# ------------------------------------------------------------------ score
def score(args) -> None:
    run = Path(args.run)
    rows = json.loads((run / "faces.json").read_text())["rows"]
    n = len(rows)
    yaw = np.array([r["yaw"] for r in rows]); track = np.array([r["track"] for r in rows])
    t = np.array([r["t"] for r in rows]); video = np.array([r["video"] for r in rows])
    frame = np.array([r["frame"] for r in rows]); width = np.array([r["width"] for r in rows])
    keep = width >= args.min_width
    rng = np.random.default_rng(0)
    gen, imp = [], []
    for tr in np.unique(track[keep]):
        idx = np.flatnonzero((track == tr) & keep)
        if len(idx) > args.max_per_track:
            idx = np.sort(rng.choice(idx, args.max_per_track, replace=False))
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                if abs(t[a] - t[b]) >= args.min_gap:
                    gen.append((a, b))
    key = video.astype(np.int64) * 10_000_000 + frame
    for k in np.unique(key[keep]):
        idx = np.flatnonzero((key == k) & keep)
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                imp.append((a, b))
    gen = np.array(gen).reshape(-1, 2); imp = np.array(imp).reshape(-1, 2)
    ayaw = np.abs(yaw)
    def groups(pairs):
        hi = np.maximum(ayaw[pairs[:, 0]], ayaw[pairs[:, 1]]); lo = np.minimum(ayaw[pairs[:, 0]], ayaw[pairs[:, 1]])
        return np.array([bucket(v) for v in hi]), lo
    gb, glo = groups(gen) if len(gen) else (np.zeros(0, int), np.zeros(0))
    ib, _ = groups(imp) if len(imp) else (np.zeros(0, int), np.zeros(0))
    report = {"faces": int(keep.sum()), "tracks": int(len(np.unique(track[keep]))),
              "genuine_pairs": len(gen), "impostor_pairs": len(imp), "models": {}}
    for path in sorted(run.glob("emb-*.npy")):
        name = path.stem[4:]
        e = np.load(path)
        if len(e) != n:
            continue
        g = (e[gen[:, 0]] * e[gen[:, 1]]).sum(1) if len(gen) else np.zeros(0)
        i = (e[imp[:, 0]] * e[imp[:, 1]]).sum(1) if len(imp) else np.zeros(0)
        entry = {"impostor": {"n": len(i)}, "buckets": {}}
        if len(i):
            entry["impostor"].update(p50=round(float(np.percentile(i, 50)), 3), p99=round(float(np.percentile(i, 99)), 3),
                                     max=round(float(i.max()), 3))
        for b, label in enumerate(YAW_NAMES):
            row = {}
            ib_scores = i[ib == b]
            pool = ib_scores if len(ib_scores) >= args.min_impostors else i
            row["impostor_n"] = int(len(ib_scores)); row["impostor_basis"] = "bucket" if pool is ib_scores else "all"
            for label2, mask in (("any", gb == b), ("frontal_to_turned", (gb == b) & (glo <= 20))):
                s = g[mask]
                cell = {"n": int(len(s))}
                if len(s):
                    cell["median"] = round(float(np.median(s)), 3)
                    if len(pool):
                        for far in (0.01, 0.001):
                            thr = float(np.quantile(pool, 1 - far))
                            cell[f"tar@far{far}"] = round(float((s > thr).mean()), 3)
                        cell["auc"] = round(float((s[:, None] > pool[None, :: max(1, len(pool) // 2000)]).mean()), 4)
                row[label2] = cell
            entry["buckets"][label] = row
        report["models"][name] = entry
    (run / "score.json").write_text(json.dumps(report, indent=1))
    print(f"faces {report['faces']}  tracks {report['tracks']}  genuine pairs {len(gen)}  same-frame impostor pairs {len(imp)}")
    for name, entry in report["models"].items():
        print(f"\n{name}   impostor {entry['impostor']}")
        print(f"  {'yaw':>6} {'gen n':>7} {'median':>7} {'TAR@1%':>7} {'TAR@.1%':>8} {'AUC':>7} | {'f2t n':>6} {'median':>7} {'TAR@1%':>7} {'AUC':>7} | imp n ({'basis'})")
        for label, row in entry["buckets"].items():
            a, f = row["any"], row["frontal_to_turned"]
            fmt = lambda c, k: f"{c.get(k, float('nan')):7.3f}" if k in c else "      -"
            print(f"  {label:>6} {a['n']:7d} {fmt(a,'median')} {fmt(a,'tar@far0.01')} {fmt(a,'tar@far0.001'):>8} {fmt(a,'auc')} | "
                  f"{f['n']:6d} {fmt(f,'median')} {fmt(f,'tar@far0.01')} {fmt(f,'auc')} | {row['impostor_n']} ({row['impostor_basis']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--video", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--models-root", required=True); p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--det-size", type=int, default=1280); p.add_argument("--min-width", type=float, default=32.0)
    p.add_argument("--link-iou", type=float, default=0.35)
    p.set_defaults(func=extract)
    p = sub.add_parser("embed")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--models-root", required=True); p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=embed)
    p = sub.add_parser("score")
    p.add_argument("--run", required=True); p.add_argument("--min-gap", type=float, default=0.4)
    p.add_argument("--max-per-track", type=int, default=40); p.add_argument("--min-width", type=float, default=32.0)
    p.add_argument("--min-impostors", type=int, default=300)
    p.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

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


def _load_cvl(path: Path, aligner: bool = False):
    """Build a CVLFace model from its repository directly (its transformers wrapper
    breaks on current transformers releases)."""
    import yaml
    from omegaconf import OmegaConf
    for mod in [m for m in sys.modules if m.split(".")[0] in ("models", "aligners")]:
        del sys.modules[mod]
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, str(path))
    try:
        conf = OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml")))
        if aligner:
            from aligners import get_aligner
            model = get_aligner(conf)
        else:
            from models import get_model
            model = get_model(conf)
        model.load_state_dict_from_path(str(Path(path).resolve() / "pretrained_model" / "model.pt"))
        return model.eval()
    finally:
        os.chdir(cwd)


CVL_REPOS = {"adaface_ir101_wf12m": "cvlface_adaface_ir101_webface12m",
             "kprpe_wf12m": "cvlface_adaface_vit_base_kprpe_webface12m",
             "kprpe_wf4m": "cvlface_adaface_vit_base_kprpe_webface4m"}


class Embedder:
    """One recognizer, loaded once: aligned BGR crops in, unit-length templates out."""

    def __init__(self, name: str, root: Path, device: str = "cpu"):
        self.name, self.device = name, device
        self.base, _, self.variant = name.partition("-")
        root = Path(root).resolve()
        self.onnx = self.aligner = self.model = None
        if name == "buffalo_l":
            self.onnx = root / "insightface/models/buffalo_l/w600k_r50.onnx"
        elif name == "antelopev2":
            self.onnx = list((root / "insightface/models/antelopev2").rglob("glintr100.onnx"))[0]
        else:
            if self.variant == "dfa":
                self.aligner = _load_cvl(root / "cvlface_DFA_mobilenet", aligner=True).to(device)
            self.model = _load_cvl(root / CVL_REPOS[self.base]).to(device)

    def __call__(self, aligned: np.ndarray, kps_aligned: np.ndarray, loose: np.ndarray | None = None) -> np.ndarray:
        if self.onnx is not None:
            emb = _onnx_embed(self.onnx, aligned)
        else:
            import torch
            to_tensor = lambda a: ((torch.from_numpy(np.ascontiguousarray(a[..., ::-1])).float() / 255 - 0.5) / 0.5).permute(0, 3, 1, 2).contiguous()
            kps_aligned = np.asarray(kps_aligned, dtype=np.float32)
            out = []
            with torch.no_grad():
                for i in range(0, len(aligned), 32):
                    if self.variant == "dfa":
                        ax, _orig, al, *_ = self.aligner(to_tensor(loose[i:i + 32]).to(self.device))
                        y = self.model(ax, al)
                    elif self.base.startswith("kprpe"):
                        y = self.model(to_tensor(aligned[i:i + 32]).to(self.device), torch.from_numpy(kps_aligned[i:i + 32]).to(self.device))
                    else:
                        y = self.model(to_tensor(aligned[i:i + 32]).to(self.device))
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    out.append(y.float().cpu().numpy())
            emb = np.concatenate(out)
        return (emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)


def embed(args) -> None:
    import torch
    run = Path(args.run)
    data = np.load(run / "faces.npz")
    rows = json.loads((run / "faces.json").read_text())["rows"]
    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    emb = Embedder(args.model, Path(args.models_root), device)(
        data["aligned"], np.asarray([r["kps_aligned"] for r in rows], dtype=np.float32), data["loose"])
    np.save(run / f"emb-{args.model}.npy", emb)
    print(args.model, emb.shape, flush=True)


# ------------------------------------------------------------------ score
def score_arrays(yaw, track, t, video, frame, width, embeddings: dict, *, min_gap: float = 0.4,
                 max_per_track: int = 40, min_width: float = 32.0, min_impostors: int = 300) -> dict:
    """Score recognizers on faces described by parallel arrays; embeddings maps name -> (n, 512)."""
    yaw, track, t = np.asarray(yaw, dtype=float), np.asarray(track), np.asarray(t, dtype=float)
    video, frame, width = np.asarray(video), np.asarray(frame), np.asarray(width, dtype=float)
    keep = width >= min_width
    rng = np.random.default_rng(0)
    gen, imp = [], []
    for tr in np.unique(track[keep]):
        idx = np.flatnonzero((track == tr) & keep)
        if len(idx) > max_per_track:
            idx = np.sort(rng.choice(idx, max_per_track, replace=False))
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                if abs(t[a] - t[b]) >= min_gap:
                    gen.append((a, b))
    key = video.astype(np.int64) * 10_000_000 + frame
    order = np.argsort(key, kind="stable")
    order = order[keep[order]]
    for idx in np.split(order, np.flatnonzero(np.diff(key[order])) + 1) if len(order) else []:
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                imp.append((a, b))
    gen = np.array(gen, dtype=int).reshape(-1, 2); imp = np.array(imp, dtype=int).reshape(-1, 2)
    ayaw = np.abs(yaw)
    def groups(pairs):
        hi = np.maximum(ayaw[pairs[:, 0]], ayaw[pairs[:, 1]]); lo = np.minimum(ayaw[pairs[:, 0]], ayaw[pairs[:, 1]])
        return np.array([bucket(v) for v in hi]), lo
    gb, glo = groups(gen) if len(gen) else (np.zeros(0, int), np.zeros(0))
    ib, _ = groups(imp) if len(imp) else (np.zeros(0, int), np.zeros(0))

    def cell(s, pool):
        out = {"n": int(len(s))}
        if len(s):
            out["median"] = round(float(np.median(s)), 3)
            if len(pool):
                for far in (0.01, 0.001):
                    out[f"tar@far{far}"] = round(float((s > float(np.quantile(pool, 1 - far))).mean()), 3)
                ranked = np.sort(pool[:: max(1, len(pool) // 2000)])
                out["auc"] = round(float(np.searchsorted(ranked, s, side="left").mean() / len(ranked)), 4)
        return out

    report = {"faces": int(keep.sum()), "tracks": int(len(np.unique(track[keep]))),
              "genuine_pairs": len(gen), "impostor_pairs": len(imp), "models": {}}
    for name, e in embeddings.items():
        g = (e[gen[:, 0]] * e[gen[:, 1]]).sum(1) if len(gen) else np.zeros(0)
        i = (e[imp[:, 0]] * e[imp[:, 1]]).sum(1) if len(imp) else np.zeros(0)
        entry = {"impostor": {"n": len(i)}, "buckets": {}}
        if len(i):
            entry["impostor"].update(p50=round(float(np.percentile(i, 50)), 3), p99=round(float(np.percentile(i, 99)), 3),
                                     max=round(float(i.max()), 3))
        for b, label in enumerate(YAW_NAMES):
            ib_scores = i[ib == b]
            pool = ib_scores if len(ib_scores) >= min_impostors else i
            entry["buckets"][label] = {
                "impostor_n": int(len(ib_scores)), "impostor_basis": "bucket" if pool is ib_scores else "all",
                "any": cell(g[gb == b], pool), "frontal_to_turned": cell(g[(gb == b) & (glo <= 20)], pool)}
        # The live question in two numbers: both faces frontal, and a frontal face against a turned one.
        entry["summary"] = {"frontal": cell(g[gb == 0], i), "turned": cell(g[(gb >= 1) & (glo <= 20)], i)}
        report["models"][name] = entry
    return report


def score(args) -> None:
    run = Path(args.run)
    rows = json.loads((run / "faces.json").read_text())["rows"]
    embeddings = {}
    for path in sorted(run.glob("emb-*.npy")):
        e = np.load(path)
        if len(e) == len(rows):
            embeddings[path.stem[4:]] = e
    column = lambda name: np.array([r[name] for r in rows])
    report = score_arrays(column("yaw"), column("track"), column("t"), column("video"), column("frame"),
                          column("width"), embeddings, min_gap=args.min_gap, max_per_track=args.max_per_track,
                          min_width=args.min_width, min_impostors=args.min_impostors)
    (run / "score.json").write_text(json.dumps(report, indent=1))
    print(f"faces {report['faces']}  tracks {report['tracks']}  genuine pairs {report['genuine_pairs']}  same-frame impostor pairs {report['impostor_pairs']}")
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

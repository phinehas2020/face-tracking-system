#!/usr/bin/env python3
"""Measure the different-person similarity distribution of a camera, without labels.

Two faces detected in the same frame are two different people. Scanning footage
and scoring every same-frame pair therefore gives an exact sample of impostor
scores for this camera, crowd and recognizer, using the same face acceptance
gates as the live pipeline. Only scalar scores are written; no image or
embedding leaves memory.

    python tools/score_survey.py --video crowd.mp4 --start 300 --duration 600 \
        --fps 2 --output impostor-scores.json

With --pose-study the pose gate is lifted for the scan and every score is
grouped by how far the less frontal face of the pair is turned. Same-frame pairs
give the impostor side. A face that stays in place between consecutive sampled
frames gives the genuine side. Together they show whether faces the live gate
rejects still carry identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)


def summarize(scores: list[float], frames: int, faces: int, crowded_frames: int) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    summary = {"frames_scanned": frames, "faces": faces, "frames_with_two_or_more": crowded_frames,
               "same_frame_pairs": len(values)}
    if len(values):
        counts, edges = np.histogram(values, bins=np.arange(-0.3, 1.0001, 0.05))
        summary.update(
            mean=round(float(values.mean()), 4),
            sd=round(float(values.std()), 4),
            percentiles={str(q): round(float(np.percentile(values, q)), 4)
                         for q in (50, 90, 99, 99.9)},
            maximum=round(float(values.max()), 4),
            pairs_at_or_above={f"{t:.2f}": int((values >= t).sum()) for t in THRESHOLDS},
            histogram=[[round(float(edge), 2), int(count)]
                       for edge, count in zip(edges, counts, strict=False) if count],
            # Scalar scores only, for `ingress calibrate-resolution --impostor-scores`.
            scores=[round(float(value), 4) for value in values],
        )
    return summary


POSE_CLASSES = ("inside gate", "up to 35 deg", "35 to 50 deg", "beyond 50 deg")


def pose_class(pitch: float, yaw: float, max_pitch: float, max_yaw: float) -> int:
    if abs(yaw) <= max_yaw and abs(pitch) <= max_pitch:
        return 0
    worst = max(abs(yaw), abs(pitch))
    return 1 if worst <= 35 else 2 if worst <= 50 else 3


def overlap(a, b) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / max(union, 1e-9)


def pose_study(encoder, frames_iter) -> dict:
    """Impostor and genuine scores by the pose of the less frontal face."""
    max_pitch, max_yaw = encoder.MAX_PITCH_DEGREES, encoder.MAX_YAW_DEGREES
    encoder.MAX_PITCH_DEGREES = encoder.MAX_YAW_DEGREES = 180.0  # this instance only
    raw: list = []
    original = encoder.app.get

    def capture(image):
        found = original(image)
        raw[:] = found
        return found

    encoder.app.get = capture
    impostor = {index: [] for index in range(len(POSE_CLASSES))}
    genuine = {index: [] for index in range(len(POSE_CLASSES))}
    faces_by_class = [0] * len(POSE_CLASSES)
    previous: list = []
    frames = 0
    for frame in frames_iter:
        frames += 1
        current = []
        for face in encoder.detect(frame):
            source = next((r for r in raw if tuple(map(float, r.bbox)) == face.bbox), None)
            if source is None:
                continue
            pitch, yaw = float(source.pose[0]), float(source.pose[1])
            group = pose_class(pitch, yaw, max_pitch, max_yaw)
            faces_by_class[group] += 1
            current.append((face.bbox, face.embedding, group))
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                worse = max(current[i][2], current[j][2])
                impostor[worse].append(float(current[i][1] @ current[j][1]))
        # A face in nearly the same place half a second later is the same person,
        # provided the match is unambiguous in both directions.
        for box, embedding, group in current:
            matches = [old for old in previous if overlap(box, old[0]) >= 0.4]
            if len(matches) != 1:
                continue
            old = matches[0]
            if sum(overlap(old[0], other[0]) >= 0.4 for other in current) != 1:
                continue
            # Only pairs anchored by one gate-passing face say how an off-angle
            # face compares with the kind of sample enrollment uses.
            if min(group, old[2]) == 0:
                genuine[max(group, old[2])].append(float(embedding @ old[1]))
        previous = current

    def describe(values, genuine_side):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return {"pairs": 0}
        result = {"pairs": len(values), "median": round(float(np.median(values)), 3)}
        if genuine_side:
            result.update(p5=round(float(np.percentile(values, 5)), 3),
                          p25=round(float(np.percentile(values, 25)), 3),
                          share_below_0_40=round(float((values < 0.40).mean()), 3),
                          share_at_least_0_60=round(float((values >= 0.60).mean()), 3))
        else:
            result.update(p99=round(float(np.percentile(values, 99)), 3),
                          maximum=round(float(values.max()), 3),
                          at_or_above_0_30=int((values >= 0.30).sum()),
                          at_or_above_0_40=int((values >= 0.40).sum()))
        return result

    return {
        "frames_scanned": frames,
        "gate": {"max_yaw_degrees": max_yaw, "max_pitch_degrees": max_pitch},
        "faces_by_pose": dict(zip(POSE_CLASSES, faces_by_class, strict=True)),
        "different_people_same_frame": {
            POSE_CLASSES[k]: describe(v, False) for k, v in impostor.items()},
        "same_person_consecutive_frames": {
            POSE_CLASSES[k]: describe(v, True) for k, v in genuine.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--provider", default="cpu", choices=("cpu", "coreml", "auto"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-study", action="store_true")
    args = parser.parse_args(argv)

    import cv2
    from ingress.vision.adapters import LiveFaceEncoder

    encoder = LiveFaceEncoder(provider=args.provider)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        parser.error("Cannot open video")
    native = video.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(native / args.fps))
    first, last = round(args.start * native), round((args.start + args.duration) * native)
    video.set(cv2.CAP_PROP_POS_FRAMES, first)
    def sampled_frames():
        for index in range(first, last):
            if not video.grab():
                break
            if (index - first) % step:
                continue
            ok, frame = video.retrieve()
            if not ok:
                break
            yield frame

    if args.pose_study:
        summary = {"video": args.video.name, "start_seconds": args.start,
                   "duration_seconds": args.duration, "sampled_fps": round(native / step, 2),
                   "provider": encoder.selected_provider, **pose_study(encoder, sampled_frames())}
        args.output.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary))
        return 0

    scores: list[float] = []
    frames = faces = crowded = 0
    for frame in sampled_frames():
        detected = encoder.detect(frame)
        frames += 1
        faces += len(detected)
        if len(detected) >= 2:
            crowded += 1
            matrix = np.stack([face.embedding for face in detected])
            upper = np.triu_indices(len(detected), 1)
            scores.extend((matrix @ matrix.T)[upper].tolist())
    summary = {"video": args.video.name, "start_seconds": args.start,
               "duration_seconds": args.duration, "sampled_fps": round(native / step, 2),
               "provider": encoder.selected_provider, "pose_gate": encoder.pose_gate,
               **summarize(scores, frames, faces, crowded)}
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    hidden = ("histogram", "pose_gate", "scores")
    print(json.dumps({k: v for k, v in summary.items() if k not in hidden}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

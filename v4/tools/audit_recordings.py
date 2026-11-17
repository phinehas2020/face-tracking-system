"""Sample recordings without changing originals; produce a visual catalog and crowd ranking."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "thumbs").mkdir(exist_ok=True)
model = YOLO(args.model)
files = sorted(
    p
    for p in args.root.rglob("*")
    if p.is_file()
    and not p.name.startswith(".")
    and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
)
rows = []
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
for index, path in enumerate(files):
    row = {
        "index": index,
        "path": str(path),
        "relative": str(path.relative_to(args.root)),
        "bytes": path.stat().st_size,
        "samples": [],
    }
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    row.update(
        fps=fps,
        duration_seconds=count / fps if fps else 0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    best = None
    for fraction in [0.15, 0.5, 0.85]:
        if not cap.isOpened() or not fps:
            break
        seconds = row["duration_seconds"] * fraction
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, classes=[0], conf=0.3, imgsz=640, verbose=False)[0]
        people = len(result.boxes) if result.boxes is not None else 0
        row["samples"].append({"seconds": round(seconds, 3), "people": people})
        if best is None or people > best[0]:
            best = (people, frame, seconds)
    cap.release()
    row["status"] = "readable" if best else "unreadable"
    row["max_people"] = best[0] if best else 0
    if best:
        rgb = cv2.cvtColor(cv2.resize(best[1], (480, 270)), cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(args.output / "thumbs" / f"{index:03d}.jpg")
    with path.open("rb") as handle:
        first = handle.read(1024 * 1024)
        handle.seek(max(0, row["bytes"] - 1024 * 1024))
        last = handle.read(1024 * 1024)
    row["head_tail_sha256"] = hashlib.sha256(first + last).hexdigest()
    rows.append(row)
    (args.output / "audit.json").write_text(json.dumps(rows, indent=2))
    if index % 15 == 0:
        print(f"Scanned {index + 1}/{len(files)}", flush=True)
for page in range((len(rows) + 11) // 12):
    sheet = Image.new("RGB", (1440, 4 * 310), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows[page * 12 : (page + 1) * 12]):
        x = (j % 3) * 480
        y = (j // 3) * 310
        thumb = args.output / "thumbs" / f"{row['index']:03d}.jpg"
        if thumb.exists():
            sheet.paste(Image.open(thumb), (x, y))
        draw.text(
            (x + 4, y + 271), f"#{row['index']:03d} {row['relative']}", font=font, fill="black"
        )
        draw.text(
            (x + 4, y + 290),
            f"{row['status']} | {row['duration_seconds']:.0f}s | "
            f"peak sampled people {row['max_people']}",
            font=font,
            fill="black",
        )
    sheet.save(args.output / f"catalog-{page + 1:02d}.jpg")
print(
    json.dumps(
        {
            "videos": len(rows),
            "readable": sum(r["status"] == "readable" for r in rows),
            "unreadable": sum(r["status"] == "unreadable" for r in rows),
            "top_crowds": [
                (r["index"], r["relative"], r["max_people"])
                for r in sorted(rows, key=lambda r: r["max_people"], reverse=True)[:12]
            ],
        }
    ),
    flush=True,
)

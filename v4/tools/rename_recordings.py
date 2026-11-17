"""Apply a reviewed recording catalog, with collision checks and a reversible journal."""

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    with path.open("rb") as handle:
        first = handle.read(1024 * 1024)
        handle.seek(max(0, path.stat().st_size - 1024 * 1024))
        last = handle.read(1024 * 1024)
    return hashlib.sha256(first + last).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--audit", type=Path, required=True)
parser.add_argument("--catalog", type=Path, required=True)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--undo", action="store_true")
args = parser.parse_args()
args.catalog.mkdir(exist_ok=True)
journal = args.catalog / "rename-map.json"
if args.undo:
    plan = json.loads(journal.read_text())
    for row in plan:
        old, new = Path(row["original_path"]), Path(row["renamed_path"])
        if old.exists() or not new.exists() or fingerprint(new) != row["head_tail_sha256"]:
            raise SystemExit("Undo preflight failed; no files changed")
    for row in reversed(plan):
        old, new = Path(row["original_path"]), Path(row["renamed_path"])
        new.rename(old)
        sidecar = new.with_name("._" + new.name)
        if sidecar.exists():
            sidecar.rename(old.with_name("._" + old.name))
    print("Original names restored")
    raise SystemExit
if journal.exists():
    raise SystemExit("A rename journal already exists; refusing to overwrite it")
plan = []
for row in json.loads(args.audit.read_text()):
    old = Path(row["path"])
    folder = old.parent.name
    station = (
        "station-01"
        if "Station one" in folder
        else "station-unverified"
        if folder == "2025-11-29"
        else "station-02"
    )
    role = (
        "approach-view"
        if old.name.startswith("entry_")
        else "counter-view"
        if station == "station-01"
        else "wide-view"
    )
    time = old.stem.rsplit("_", 1)[-1]
    suffix = "__unreadable" if row["status"] == "unreadable" else ""
    new = old.with_name(f"{station}__{role}__{folder[:10]}__{time}{suffix}{old.suffix}")
    plan.append(
        {
            **row,
            "station": station,
            "view": role,
            "original_path": str(old),
            "renamed_path": str(new),
            "station_evidence": "folder label + sampled scene"
            if station == "station-01"
            else "unverified; unreadable source"
            if station == "station-unverified"
            else "sampled scene and continuous camera sequence",
        }
    )
for row in plan:
    old, new = Path(row["original_path"]), Path(row["renamed_path"])
    if (
        not old.exists()
        or new.exists()
        or new.with_name("._" + new.name).exists()
        or fingerprint(old) != row["head_tail_sha256"]
    ):
        raise SystemExit("Rename preflight failed; no files changed")
if len({row["renamed_path"] for row in plan}) != len(plan):
    raise SystemExit("Duplicate destinations")
journal.write_text(json.dumps(plan, indent=2))
if not args.apply:
    print("Plan written; no video renamed")
    raise SystemExit
completed = []
try:
    for row in plan:
        old, new = Path(row["original_path"]), Path(row["renamed_path"])
        old.rename(new)
        completed.append((old, new))
        sidecar = old.with_name("._" + old.name)
        if sidecar.exists():
            target = new.with_name("._" + new.name)
            sidecar.rename(target)
            completed.append((sidecar, target))
    for row in plan:
        path = Path(row["renamed_path"])
        if path.stat().st_size != row["bytes"] or fingerprint(path) != row["head_tail_sha256"]:
            raise RuntimeError("Post-rename integrity check failed")
except Exception:
    for old, new in reversed(completed):
        new.rename(old)
    raise
print(
    json.dumps(
        {
            "renamed": len(plan),
            "readable": sum(r["status"] == "readable" for r in plan),
            "unreadable": sum(r["status"] == "unreadable" for r in plan),
            "station_counts": {
                s: sum(r["station"] == s for r in plan)
                for s in sorted({r["station"] for r in plan})
            },
            "verified": "sizes and first/last 1 MiB fingerprints unchanged",
        }
    )
)

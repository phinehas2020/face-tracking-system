#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

BACKUP_DIR="backups"
LIVE_DB="live_people_tracking.db"
LIVE_THUMBS="live_thumbnails"

display_backups() {
    printf "\nAvailable backups:\n"
    ls -t "$BACKUP_DIR"/backup_*.db 2>/dev/null | nl -w2 -s'. '
}

select_backup() {
    display_backups
    printf "\nEnter the number of the backup to activate: "
    read -r choice
    local selected
    selected="$(ls -t "$BACKUP_DIR"/backup_*.db | sed -n "${choice}p")"
    if [ -z "$selected" ]; then
        echo "Invalid choice" >&2
        exit 1
    fi
    BACKUP_DB="$selected"
    local base
    base="$(basename "$selected" .db)"
    local suffix="${base#backup_}"
    BACKUP_THUMBS="$BACKUP_DIR/thumbnails_${suffix}"
    if [ ! -d "$BACKUP_THUMBS" ]; then
        echo "Matching thumbnails folder not found: $BACKUP_THUMBS" >&2
        exit 1
    fi
}

swap_to_backup() {
    if [ -f people_tracking.db ] && [ ! -f "$LIVE_DB" ]; then
        mv people_tracking.db "$LIVE_DB"
    fi
    if [ -d thumbnails ] && [ ! -d "$LIVE_THUMBS" ]; then
        mv thumbnails "$LIVE_THUMBS"
    fi
    cp "$BACKUP_DB" people_tracking.db
    rm -rf thumbnails
    cp -R "$BACKUP_THUMBS" thumbnails
}

restore_live() {
    if [ ! -f "$LIVE_DB" ]; then
        echo "No saved live database found." >&2
        exit 1
    fi
    rm -f people_tracking.db
    rm -rf thumbnails
    mv "$LIVE_DB" people_tracking.db
    if [ -d "$LIVE_THUMBS" ]; then
        mv "$LIVE_THUMBS" thumbnails
    else
        mkdir -p thumbnails
    fi
    echo "Live data restored. Snapshot remains in $BACKUP_DB."
}

adjust_threshold() {
    echo ""
    read -r -p "Tune FACE_MATCH_THRESHOLD? (u)p / (d)own / (s)kip: " resp
    resp="$(printf '%s' "$resp" | tr '[:upper:]' '[:lower:]')"
    case "$resp" in
        u|up) direction=1 ;;
        d|down) direction=-1 ;;
        *) echo "Skipping threshold adjustment."; return ;;
    esac

    read -r -p "Adjustment amount (default 0.02): " step
    step="${step:-0.02}"

    python3 - "$direction" "$step" <<'PY'
import sys
from pathlib import Path
import re

sign = float(sys.argv[1])
step = float(sys.argv[2])
delta = sign * step

path = Path("main_dual_camera.py")
text = path.read_text()
match = re.search(r"(FACE_MATCH_THRESHOLD\s*=\s*)([0-9.]+)", text)
if not match:
    raise SystemExit("FACE_MATCH_THRESHOLD not found")

value = float(match.group(2))
value = round(min(max(value + delta, 0.1), 0.6), 2)
new_text = text[:match.start(2)] + f"{value}" + text[match.end(2):]
path.write_text(new_text)
print(f"FACE_MATCH_THRESHOLD updated to {value}")
PY
}

case "${1:-}" in
    --restore)
        restore_live
        exit 0
        ;;
esac

select_backup
swap_to_backup

cat <<MSG

Snapshot activated from: $BACKUP_DB
Matching thumbnails: $BACKUP_THUMBS
(Live data saved as $LIVE_DB / $LIVE_THUMBS)

Next steps:
  1) source venv/bin/activate
  2) python merge_duplicates.py  [--apply]
  3) ./start_all.sh (optional)

Run './swap_backup.sh --restore' later to bring your live DB back.
MSG

adjust_threshold

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f swap_backup.sh ]; then
    echo "swap_backup.sh not found. Run from project root." >&2
    exit 1
fi

./swap_backup.sh

if [ ! -d venv ]; then
    echo "Virtualenv not found. Run setup.sh first." >&2
    exit 1
fi

source venv/bin/activate

echo "\nRunning duplicate scan (dry run)..."
python merge_duplicates.py

read -r -p "Apply merges to this snapshot? (y/N): " apply
apply_lc=$(printf '%s' "$apply" | tr '[:upper:]' '[:lower:]')
if [[ "$apply_lc" =~ ^y ]]; then
    python merge_duplicates.py --apply
fi

read -r -p "Launch start_all.sh with this snapshot? (y/N): " launch
launch_lc=$(printf '%s' "$launch" | tr '[:upper:]' '[:lower:]')
if [[ "$launch_lc" =~ ^y ]]; then
    echo "Starting system. Press Ctrl+C when done testing."
    ./start_all.sh
fi

read -r -p "Restore live data now? (y/N): " restore
restore_lc=$(printf '%s' "$restore" | tr '[:upper:]' '[:lower:]')
if [[ "$restore_lc" =~ ^y ]]; then
    ./swap_backup.sh --restore
else
    echo "Backup snapshot remains active. Run './swap_backup.sh --restore' later to switch back." 
fi

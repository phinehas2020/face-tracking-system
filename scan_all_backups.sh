#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

backups=(backups/backup_*.db)
if [ ! -e "${backups[0]}" ]; then
    echo "No backups found." >&2
    exit 1
fi

for db in $(ls -t backups/backup_*.db); do
    echo "========================================"
    echo "Scanning $db"
    total=$(python3 - "$db" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
count = conn.execute("SELECT COUNT(DISTINCT person_id) FROM persons").fetchone()[0]
print(count or 0)
PY
)
    echo "Total people: $total"
    found=""
    pairs=0
    for threshold in $(seq 45 -1 20); do
        t=$(awk "BEGIN { printf \"%.2f\", $threshold/100 }")
        output=$(python merge_duplicates.py --db "$db" --threshold "$t")
        if [[ "$output" != *"No duplicates found"* ]]; then
            found="$t"
            pairs=$(printf "%s\n" "$output" | grep -c ' -> ' || true)
            echo "First duplicate threshold: $t"
            echo "$output"
            break
        fi
    done
    if [ -z "$found" ]; then
        echo "No duplicates flagged down to threshold 0.20"
    else
        echo "Duplicate pairs at $found: $pairs"
    fi
    echo ""
done

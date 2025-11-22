#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

list_backups() {
    ls -t backups/backup_*.db 2>/dev/null
}

select_db() {
    local backups_list backups_count
    IFS=$'\n' backups_list=($(list_backups))
    backups_count=${#backups_list[@]}
    if [ "$backups_count" -eq 0 ]; then
        echo "No backups found in ./backups" >&2
        exit 1
    fi

    echo "Available backups:"
    local idx=1
    for f in "${backups_list[@]}"; do
        echo " $idx) $f"
        idx=$((idx+1))
    done

    read -r -p "Enter number or path (default 1): " choice
    if [ -z "$choice" ]; then
        echo "${backups_list[0]}"
    elif [[ "$choice" =~ ^[0-9]+$ ]]; then
        local pos=$((choice-1))
        if [ $pos -ge 0 ] && [ $pos -lt "$backups_count" ]; then
            echo "${backups_list[$pos]}"
        else
            echo "Invalid selection" >&2
            exit 1
        fi
    else
        echo "$choice"
    fi
}

choose_db() {
    local input
    input="$(select_db)"
    input="$(echo -n "$input" | tr -d '[:space:]')"
    if [ -f "$input" ]; then
        DB_PATH="$input"
        return
    fi
    if [ -f "./$input" ]; then
        DB_PATH="./$input"
        return
    fi
    echo "Database not found: $input" >&2
    exit 1
}

if [ $# -gt 0 ]; then
    DB_PATH="$1"
    if [ ! -f "$DB_PATH" ]; then
        echo "Database not found: $DB_PATH" >&2
        exit 1
    fi
else
    choose_db
fi

printf "Scanning %s for duplicate faces\n" "$DB_PATH"

total_people=$(python3 - "$DB_PATH" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
count = conn.execute("SELECT COUNT(DISTINCT person_id) FROM persons").fetchone()[0]
print(count or 0)
PY
)

found_threshold=""
found_output=""
pairs=0
for threshold in $(seq 45 -1 20); do
    t=$(awk "BEGIN { printf \"%.2f\", $threshold/100 }")
    echo "Checking threshold $t ..."
    output=$(python merge_duplicates.py --db "$DB_PATH" --threshold "$t")
    if [[ "$output" != *"No duplicates found"* ]]; then
        found_threshold="$t"
        found_output="$output"
        pairs=$(printf "%s\n" "$output" | grep -c ' -> ' || true)
        break
    fi
    sleep 0.1
done

if [ -z "$found_threshold" ]; then
    echo "No duplicates found down to threshold 0.20"
    echo "Total people in DB: $total_people"
    exit 0
fi

printf "\nDuplicates detected at threshold %s\n" "$found_threshold"
echo "$found_output"
cat <<SUMMARY

Summary:
  Total people in DB : $total_people
  Duplicate pairs    : $pairs

Run this command to merge them:
  python merge_duplicates.py --db '$DB_PATH' --threshold $found_threshold --apply
SUMMARY

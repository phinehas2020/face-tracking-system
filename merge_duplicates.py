#!/usr/bin/env python3
"""
Utility to find and merge duplicate people (same face with different IDs).
Usage:
    python merge_duplicates.py --db people_tracking.db --apply
Default mode is dry-run (prints candidates). Use --apply to perform merges.
"""

import argparse
import sqlite3
import pickle
import numpy as np
from pathlib import Path

MERGE_THRESHOLD = 0.26  # cosine similarity threshold for merging


def load_embeddings(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.person_id, p.created_at, f.embedding
        FROM persons p
        JOIN faces f ON p.person_id = f.person_id
        """
    )
    persons = {}
    for person_id, created_at, blob in cur.fetchall():
        emb = pickle.loads(blob)
        emb = emb / np.linalg.norm(emb)
        entry = persons.setdefault(person_id, {"created_at": created_at, "embeddings": []})
        entry["embeddings"].append(emb)
    return persons


def person_similarity(embs_a, embs_b):
    best = -1.0
    for ea in embs_a:
        for eb in embs_b:
            best = max(best, float(np.dot(ea, eb)))
    return best


def merge_person(conn, src_id, dst_id):
    cur = conn.cursor()
    cur.execute("UPDATE faces SET person_id=? WHERE person_id=?", (dst_id, src_id))
    cur.execute("UPDATE crossings SET person_id=? WHERE person_id=?", (dst_id, src_id))
    cur.execute("DELETE FROM persons WHERE person_id=?", (src_id,))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Merge duplicate people by embedding similarity")
    parser.add_argument('--db', default='people_tracking.db', help='Path to SQLite database')
    parser.add_argument('--threshold', type=float, default=MERGE_THRESHOLD,
                        help='Cosine similarity threshold used to treat two IDs as duplicates')
    parser.add_argument('--apply', action='store_true',
                        help='Actually merge duplicates (default: dry run)')
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    persons = load_embeddings(conn)

    ids = list(persons.keys())
    merges = []
    for i, pid in enumerate(ids):
        for other in ids[i+1:]:
            sim = person_similarity(persons[pid]['embeddings'], persons[other]['embeddings'])
            if sim >= args.threshold:
                # Keep the older record by created_at, merge the newer into it
                keep = pid
                drop = other
                if persons[other]['created_at'] and persons[pid]['created_at']:
                    if persons[other]['created_at'] < persons[pid]['created_at']:
                        keep, drop = other, pid
                merges.append((drop, keep, sim))

    if not merges:
        print("No duplicates found above the threshold.")
        return

    merges.sort(key=lambda x: -x[2])
    print("Potential duplicates (drop -> keep | similarity):")
    for drop, keep, sim in merges:
        print(f"  {drop} -> {keep} (sim={sim:.3f})")

    if args.apply:
        for drop, keep, sim in merges:
            print(f"Merging {drop} into {keep} (sim={sim:.3f})")
            merge_person(conn, drop, keep)
        print("Done. You may want to rerun to confirm no duplicates remain.")
    else:
        print("Dry run only. Re-run with --apply to perform merges (database is unchanged).")


if __name__ == '__main__':
    main()

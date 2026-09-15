from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .domain import EventStatus, utc_now


class EventNotFoundError(LookupError):
    pass


class EventStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.db_path = data_root / "ingress.db"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as connection:
            connection.executescript(schema_path.read_text())

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    def event_exists(self, event_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone()
        return row is not None

    def list_events(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = self._rows(
                connection.execute(
                    """
                    SELECT e.*, c.raw_passages, c.unique_attendees, c.unresolved
                    FROM events e
                    JOIN event_counters c ON c.event_id = e.id
                    ORDER BY e.event_date DESC, e.created_at DESC
                    """
                )
            )
        return rows

    def get_summary(self, event_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT e.*, c.raw_passages, c.unique_attendees, c.repeat_passages,
                       c.unresolved, c.rehearsal_progress, c.ground_truth, c.updated_at
                FROM events e
                JOIN event_counters c ON c.event_id = e.id
                WHERE e.id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise EventNotFoundError(event_id)
            cameras = self._rows(
                connection.execute(
                    "SELECT * FROM cameras WHERE event_id = ? ORDER BY name", (event_id,)
                )
            )
            run = connection.execute(
                """
                SELECT * FROM replay_runs
                WHERE event_id = ?
                ORDER BY CASE name WHEN 'Fusion v4' THEN 0 ELSE 1 END, created_at DESC
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        payload = dict(row)
        payload["cameras"] = cameras
        payload["active_run"] = dict(run) if run else None
        payload["deduplication_rate"] = round(
            payload["unique_attendees"] / max(payload["raw_passages"], 1), 4
        )
        return payload

    def get_replay_runs(self, event_id: str) -> list[dict[str, Any]]:
        self._assert_event(event_id)
        with self.connect() as connection:
            return self._rows(
                connection.execute(
                    "SELECT * FROM replay_runs WHERE event_id = ? ORDER BY created_at, name",
                    (event_id,),
                )
            )

    def get_cameras(self, event_id: str) -> list[dict[str, Any]]:
        self._assert_event(event_id)
        with self.connect() as connection:
            rows = self._rows(
                connection.execute(
                    "SELECT * FROM cameras WHERE event_id = ? ORDER BY name", (event_id,)
                )
            )
        for row in rows:
            row["configuration"] = json.loads(row["configuration"] or "{}")
        return rows

    def update_camera_calibration(
        self,
        event_id: str,
        camera_id: str,
        approach_zone: list[tuple[float, float]],
        commit_line: tuple[tuple[float, float], tuple[float, float]],
        direction: str,
    ) -> dict[str, Any]:
        if len(approach_zone) < 3:
            raise ValueError("approach_zone must contain at least three points")
        values = [coordinate for point in [*approach_zone, *commit_line] for coordinate in point]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("calibration coordinates must be normalized between 0 and 1")
        configuration = {
            "approach_zone": approach_zone,
            "commit_line": commit_line,
            "direction": direction,
        }
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ? AND event_id = ?", (camera_id, event_id)
            ).fetchone()
            if row is None:
                raise EventNotFoundError(camera_id)
            before = dict(row)
            connection.execute(
                "UPDATE cameras SET configuration = ? WHERE id = ? AND event_id = ?",
                (json.dumps(configuration), camera_id, event_id),
            )
            connection.execute(
                """
                INSERT INTO audit_log
                    (event_id, action, actor, target_type, target_id,
                     before_state, after_state, created_at)
                VALUES (?, 'camera_calibration', 'operator', 'camera', ?, ?, ?, ?)
                """,
                (
                    event_id,
                    camera_id,
                    json.dumps(before, default=str),
                    json.dumps(configuration),
                    utc_now(),
                ),
            )
        return next(camera for camera in self.get_cameras(event_id) if camera["id"] == camera_id)

    def get_timeline(self, event_id: str) -> list[dict[str, Any]]:
        self._assert_event(event_id)
        with self.connect() as connection:
            return self._rows(
                connection.execute(
                    """
                    SELECT * FROM timeline_samples
                    WHERE event_id = ? ORDER BY sample_index
                    """,
                    (event_id,),
                )
            )

    def get_candidates(self, event_id: str, status: str | None = None) -> list[dict[str, Any]]:
        self._assert_event(event_id)
        query = "SELECT * FROM identity_links WHERE event_id = ?"
        params: list[Any] = [event_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY CASE status WHEN 'review' THEN 0 ELSE 1 END, probability DESC"
        with self.connect() as connection:
            rows = self._rows(connection.execute(query, params))
        for row in rows:
            row["evidence"] = json.loads(row["evidence"] or "{}")
        return rows

    def resolve_candidate(
        self, event_id: str, candidate_id: str, decision: str, actor: str
    ) -> dict[str, Any]:
        if decision not in {"same", "different", "review"}:
            raise ValueError("decision must be same, different, or review")
        status_by_decision = {
            "same": "confirmed_same",
            "different": "confirmed_different",
            "review": "review",
        }
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM identity_links WHERE id = ? AND event_id = ?",
                (candidate_id, event_id),
            ).fetchone()
            if row is None:
                raise EventNotFoundError(candidate_id)
            before = dict(row)
            after_status = status_by_decision[decision]
            connection.execute(
                """
                UPDATE identity_links
                SET status = ?, reviewed_by = ?, reviewed_at = ?
                WHERE id = ? AND event_id = ?
                """,
                (after_status, actor, utc_now(), candidate_id, event_id),
            )
            if before["status"] == "review" and after_status != "review":
                unique_delta = -1 if after_status == "confirmed_same" else 0
                connection.execute(
                    """
                    UPDATE event_counters
                    SET unresolved = MAX(unresolved - 1, 0),
                        unique_attendees = MAX(unique_attendees + ?, 0),
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (unique_delta, utc_now(), event_id),
                )
            connection.execute(
                """
                INSERT INTO audit_log
                    (event_id, action, actor, target_type, target_id,
                     before_state, after_state, created_at)
                VALUES (?, 'identity_decision', ?, 'identity_link', ?, ?, ?, ?)
                """,
                (
                    event_id,
                    actor,
                    candidate_id,
                    json.dumps(before, default=str),
                    json.dumps({"status": after_status}),
                    utc_now(),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM identity_links WHERE id = ?", (candidate_id,)
            ).fetchone()
        result = dict(updated)
        result["evidence"] = json.loads(result["evidence"] or "{}")
        return result

    def purge_plan(self, event_id: str) -> dict[str, Any]:
        self._assert_event(event_id)
        with self.connect() as connection:
            counts = {}
            for table in (
                "cameras",
                "identities",
                "passages",
                "identity_links",
                "replay_runs",
                "timeline_samples",
                "audit_log",
            ):
                counts[table] = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE event_id = ?", (event_id,)
                ).fetchone()[0]
        event_dir = self._event_directory(event_id)
        bytes_on_disk = (
            sum(path.stat().st_size for path in event_dir.rglob("*") if path.is_file())
            if event_dir.exists()
            else 0
        )
        return {
            "event_id": event_id,
            "database_records": counts,
            "event_files": str(event_dir),
            "bytes_on_disk": bytes_on_disk,
            "confirmation": f"PURGE {event_id}",
        }

    def purge_event(self, event_id: str, confirmation: str, actor: str) -> dict[str, Any]:
        expected = f"PURGE {event_id}"
        if confirmation != expected:
            raise ValueError(f"confirmation must exactly equal {expected!r}")
        plan = self.purge_plan(event_id)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO audit_log
                    (event_id, action, actor, target_type, target_id, before_state, created_at)
                VALUES (?, 'event_purge', ?, 'event', ?, ?, ?)
                """,
                (event_id, actor, event_id, json.dumps(plan), utc_now()),
            )
            connection.execute("DELETE FROM events WHERE id = ?", (event_id,))
            connection.execute("DELETE FROM audit_log WHERE event_id = ?", (event_id,))
        event_dir = self._event_directory(event_id)
        if event_dir.exists():
            shutil.rmtree(event_dir)
        # Secure-delete clears row contents; checkpoint + vacuum removes WAL and free pages.
        with self._lock, self.connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
        return {"event_id": event_id, "status": EventStatus.PURGED, "deleted": plan}

    def _assert_event(self, event_id: str) -> None:
        if not self.event_exists(event_id):
            raise EventNotFoundError(event_id)

    def _event_directory(self, event_id: str) -> Path:
        events_root = (self.data_root / "events").resolve()
        event_dir = (events_root / event_id).resolve()
        if event_dir.parent != events_root:
            raise ValueError("invalid event id for event-scoped file access")
        return event_dir

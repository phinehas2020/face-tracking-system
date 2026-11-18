"""Activity windows: what the cameras saw each time something moved at the door.

Counts and identity links have never been checked against the truth, because a
live passage keeps no picture. An activity window is a short film strip from
every camera, saved whenever people or gate motion were present on the counting
view, whether or not a passage was counted. A person then states how many
people really entered and left. The labelled windows are the ground truth the
scorecard is computed from, and windows with no counted passage are how missed
people are found.

Strips are small contact sheets (one JPEG per camera), event-scoped, and
bounded by a byte budget and a free-disk floor. Labels stay in the database
after a strip is pruned.
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .domain import utc_now
from .store import EventStore

CELL_WIDTH = 480
COLUMNS = 8
MAX_FRAMES = 40


class FrameRing:
    """The last few seconds of encoded camera frames, thinned to a fixed rate."""

    def __init__(self, seconds: float = 16.0, max_fps: float = 6.0):
        self.seconds = seconds
        self.interval = 1 / max_fps
        self.frames: deque[tuple[float, bytes]] = deque()
        self.lock = threading.Lock()

    def push(self, captured_at: float, jpeg: bytes) -> None:
        with self.lock:
            if self.frames and captured_at - self.frames[-1][0] < self.interval - 1e-3:
                return
            self.frames.append((captured_at, jpeg))
            while self.frames and captured_at - self.frames[0][0] > self.seconds:
                self.frames.popleft()

    def window(self, start: float, end: float) -> list[tuple[float, bytes]]:
        with self.lock:
            return [item for item in self.frames if start <= item[0] <= end]


class GateMotion:
    """Frame-to-frame change on three pixel strips at the counting line."""

    def __init__(self):
        self.key = self.coords = self.previous = None

    def update(self, frame, calibration) -> bool:
        from .gate_motion import strip_pixels
        from .live_counting import PassageCalibration

        height, width = frame.shape[:2]
        key = (width, height, id(calibration))
        if key != self.key:
            # Keep the object so its id cannot be reused by a later calibration.
            self.key, self.previous, self.calibration = key, None, calibration
            try:
                self.coords = strip_pixels(
                    PassageCalibration.from_calibration(calibration), width, height
                )
            except (ValueError, TypeError):
                self.coords = None
        if self.coords is None:
            return False
        pixels = frame[self.coords].astype(np.int16)
        previous, self.previous = self.previous, pixels
        if previous is None:
            return False
        changed = (np.abs(pixels - previous).max(axis=2) > 32).mean(axis=1)
        return int(np.count_nonzero(changed >= 0.2)) >= 2


@dataclass
class _Window:
    start: float
    started_at: str
    last_active: float
    cause: str
    pre: float = 0.0
    peak_people: int = 0
    passages: list[tuple[str, str, float]] = field(default_factory=list)
    continuation: bool = False
    moved: bool = False


def build_strip(frames: list[tuple[float, bytes]], start: float):
    """Tile up to MAX_FRAMES evenly spaced frames into one JPEG contact sheet."""
    import cv2

    if len(frames) > MAX_FRAMES:
        picks = np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)
        frames = [frames[i] for i in picks]
    cells, times = [], []
    for captured_at, jpeg in frames:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if image is None:
            continue
        height, width = image.shape[:2]
        cell_height = max(2, round(height * CELL_WIDTH / width))
        cells.append(cv2.resize(image, (CELL_WIDTH, cell_height), interpolation=cv2.INTER_AREA))
        times.append(round(captured_at - start, 3))
    if not cells:
        return None
    cell_height = cells[0].shape[0]
    cells = [c for c in cells if c.shape[0] == cell_height]
    times = times[: len(cells)]
    rows = math.ceil(len(cells) / COLUMNS)
    sheet = np.zeros((rows * cell_height, COLUMNS * CELL_WIDTH, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        top, left = (index // COLUMNS) * cell_height, (index % COLUMNS) * CELL_WIDTH
        sheet[top : top + cell_height, left : left + CELL_WIDTH] = cell
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return encoded.tobytes(), {
        "frames": len(cells),
        "columns": COLUMNS,
        "cell": [CELL_WIDTH, cell_height],
        "times": times,
    }


class ActivityRecorder:
    """Open a window while the counting view is active; save it when it goes quiet."""

    def __init__(
        self,
        store: EventStore,
        event_id: str,
        camera_id: str,
        rings: dict[str, FrameRing],
        *,
        max_bytes: int = 600 * 1024 * 1024,
        min_free_bytes: int = 2 * 1024**3,
        quiet_seconds: float = 1.0,
        max_seconds: float = 10.0,
        pre_seconds: float = 1.0,
        post_seconds: float = 1.0,
        uncounted_per_hour: int = 30,
        start_writer: bool = True,
    ):
        self.store = store
        self.event_id = event_id
        self.camera_id = camera_id
        self.rings = rings
        self.root = store._event_directory(event_id) / "activity"
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.quiet_seconds = quiet_seconds
        self.max_seconds = max_seconds
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.uncounted_per_hour = uncounted_per_hour
        self.lock = threading.Lock()
        self.window: _Window | None = None
        self.previous_forced = False
        self.uncounted_times: deque[float] = deque()
        self.jobs: queue.Queue = queue.Queue()
        self.motion = GateMotion()
        self.metrics = {
            "saved": 0,
            "saved_without_passage": 0,
            "skipped_rate_limit": 0,
            "skipped_low_disk": 0,
            "skipped_no_frames": 0,
            "errors": 0,
            "bytes": 0,
            "pruned": 0,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.metrics["bytes"] = sum(p.stat().st_size for p in self.root.rglob("*.jpg"))
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="activity-strips")
        if start_writer:
            self.thread.start()

    # -- called from the counting loop ------------------------------------

    def observe(self, now: float, observed_at: str | None, people: int, frame, calibration):
        try:
            moving = self.motion.update(frame, calibration)
        except Exception:
            moving = False
        self.update(now, observed_at, people, moving)

    def update(self, now: float, observed_at: str | None, people: int, moving: bool) -> None:
        with self.lock:
            window = self.window
            if people or moving:
                if window is None:
                    # A window cut at the length limit and the one that continues
                    # it share no frames, so no crossing is shown to a reviewer twice.
                    window = self.window = _Window(
                        now,
                        observed_at or utc_now(),
                        now,
                        "people" if people else "motion",
                        pre=0.0 if self.previous_forced else self.pre_seconds,
                        continuation=self.previous_forced,
                    )
                window.last_active = now
                window.moved = window.moved or moving
                window.peak_people = max(window.peak_people, people)
                if people and window.cause == "motion":
                    window.cause = "people"
            self.previous_forced = False
            if window is None:
                return
            forced = now - window.start >= self.max_seconds
            if forced or now - window.last_active >= self.quiet_seconds:
                self._close(window, min(now, window.last_active + self.quiet_seconds), forced)
                self.previous_forced = forced

    def passage(self, passage_id: str, direction: str, now: float) -> None:
        with self.lock:
            if self.window is None:
                self.window = _Window(now, utc_now(), now, "passage", pre=self.pre_seconds)
            self.window.passages.append((passage_id, direction, now))
            self.window.last_active = now

    def _close(self, window: _Window, end: float, forced: bool = False) -> None:
        self.window = None
        if not window.passages:
            # Someone standing still in view is one window, not one every ten
            # seconds. A later stretch is kept only if something moved at the line.
            while self.uncounted_times and end - self.uncounted_times[0] > 3600:
                self.uncounted_times.popleft()
            if (window.continuation and not window.moved) or (
                len(self.uncounted_times) >= self.uncounted_per_hour
            ):
                self.metrics["skipped_rate_limit"] += 1
                return
            self.uncounted_times.append(end)
        self.jobs.put((window, end, 0.0 if forced else self.post_seconds))

    # -- writer thread ------------------------------------------------------

    def _run(self) -> None:
        while not self.stopped.is_set():
            try:
                window, end, post = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            # Let the cameras deliver the frames that follow the window.
            self.stopped.wait(max(0.0, end + post + 0.4 - time.monotonic()))
            try:
                self.save(window, end, post)
            except Exception:
                self.metrics["errors"] += 1

    def save(self, window: _Window, end: float, post: float | None = None) -> str | None:
        if shutil.disk_usage(self.root).free < self.min_free_bytes:
            self.metrics["skipped_low_disk"] += 1
            return None
        post = self.post_seconds if post is None else post
        start = window.start - window.pre
        # The part of the strip the label is about; the rest is context.
        span = [round(window.pre, 3), round(window.pre + end - window.start, 3)]
        window_id = "act-" + uuid.uuid4().hex
        folder = self.root / window.started_at[:10]
        strips, written = {}, 0
        for view, ring in self.rings.items():
            built = build_strip(ring.window(start, end + post), start)
            if built is None:
                continue
            data, layout = built
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{window_id}-{view}.jpg"
            path.write_bytes(data)
            path.chmod(0o600)
            written += len(data)
            strips[view] = {
                **layout,
                "span": span,
                "file": str(path.relative_to(self.root)),
                "bytes": len(data),
            }
        if not strips:
            self.metrics["skipped_no_frames"] += 1
            return None
        directions = [direction for _id, direction, _at in window.passages]
        with self.store.transaction() as db:
            db.execute(
                """INSERT INTO activity_windows
                (id,event_id,camera_id,started_at,duration_seconds,cause,system_in,system_out,
                 passages,peak_people,strips,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    window_id,
                    self.event_id,
                    self.camera_id,
                    window.started_at,
                    round(end - window.start, 3),
                    "passage" if window.passages else window.cause,
                    directions.count("in"),
                    directions.count("out"),
                    json.dumps(
                        [
                            {"id": pid, "direction": direction, "offset": round(at - start, 3)}
                            for pid, direction, at in window.passages
                        ]
                    ),
                    window.peak_people,
                    json.dumps(strips),
                    utc_now(),
                ),
            )
        self.metrics["saved"] += 1
        self.metrics["saved_without_passage"] += not window.passages
        self.metrics["bytes"] += written
        self._prune()
        return window_id

    def _prune(self) -> None:
        """Drop the oldest strips beyond the byte budget. Rows and labels remain."""
        if self.metrics["bytes"] <= self.max_bytes:
            return
        files = sorted(self.root.rglob("*.jpg"), key=lambda p: (p.parent.name, p.stat().st_mtime))
        for path in files:
            if self.metrics["bytes"] <= self.max_bytes * 0.9:
                break
            self.metrics["bytes"] -= path.stat().st_size
            path.unlink()
            self.metrics["pruned"] += 1

    def stop(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)


# -- review and scoring -------------------------------------------------------


def _row(row) -> dict:
    item = dict(row)
    item["passages"] = json.loads(item["passages"])
    item["strips"] = {
        view: {key: value for key, value in strip.items() if key != "file"}
        for view, strip in json.loads(item["strips"]).items()
    }
    return item


def list_windows(
    store: EventStore,
    event_id: str,
    state: str = "unlabelled",
    limit: int = 50,
    since: str | None = None,
):
    clause = {
        "unlabelled": "AND labelled_at IS NULL",
        "labelled": "AND labelled_at IS NOT NULL",
        "all": "",
    }[state]
    with store.connect() as db:
        rows = db.execute(
            f"SELECT * FROM activity_windows WHERE event_id=? AND started_at>=? {clause} "
            "ORDER BY started_at DESC LIMIT ?",
            (event_id, since or "", limit),
        ).fetchall()
    return [_row(row) for row in rows]


def strip_path(store: EventStore, event_id: str, window_id: str, view: str) -> Path | None:
    with store.connect() as db:
        row = db.execute(
            "SELECT strips FROM activity_windows WHERE id=? AND event_id=?", (window_id, event_id)
        ).fetchone()
    strip = json.loads(row["strips"]).get(view) if row else None
    if not strip:
        return None
    root = (store._event_directory(event_id) / "activity").resolve()
    path = (root / strip["file"]).resolve()
    return path if root in path.parents and path.is_file() else None


def label_window(
    store: EventStore,
    event_id: str,
    window_id: str,
    entries: int,
    exits: int,
    actor: str,
    note: str | None = None,
) -> bool:
    with store.transaction() as db:
        changed = db.execute(
            "UPDATE activity_windows SET label_in=?,label_out=?,label_note=?,labelled_by=?,"
            "labelled_at=? WHERE id=? AND event_id=?",
            (entries, exits, note, actor, utc_now(), window_id, event_id),
        ).rowcount
    return bool(changed)


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    return [
        round(max(0.0, (centre - spread) / denominator), 4),
        round(min(1.0, (centre + spread) / denominator), 4),
    ]


def scorecard(store: EventStore, event_id: str, since: str | None = None) -> dict:
    """Counting accuracy over the windows a person has labelled.

    Misses and extras are counted per window and direction, so an extra and a
    miss of the same direction inside one window cancel. Windows are short, so
    this is rare, but the figures are a floor on both error types. ``since``
    (an ISO time) keeps one configuration's windows apart from an earlier one's.
    """
    with store.connect() as db:
        rows = db.execute(
            "SELECT cause,system_in,system_out,label_in,label_out,labelled_at "
            "FROM activity_windows WHERE event_id=? AND started_at>=?",
            (event_id, since or ""),
        ).fetchall()
    labelled = [row for row in rows if row["labelled_at"]]
    result = {
        "event_id": event_id,
        "since": since,
        "windows": len(rows),
        "labelled": len(labelled),
        "unlabelled": len(rows) - len(labelled),
        "exact_windows": sum(
            row["system_in"] == row["label_in"] and row["system_out"] == row["label_out"]
            for row in labelled
        ),
        "directions": {},
        "by_cause": {},
    }
    truth_total = system_total = missed_total = extra_total = 0
    for name, system_key, label_key in (
        ("in", "system_in", "label_in"),
        ("out", "system_out", "label_out"),
    ):
        system = sum(row[system_key] for row in labelled)
        truth = sum(row[label_key] for row in labelled)
        missed = sum(max(0, row[label_key] - row[system_key]) for row in labelled)
        extra = sum(max(0, row[system_key] - row[label_key]) for row in labelled)
        result["directions"][name] = {
            "counted": system,
            "true": truth,
            "missed": missed,
            "extra": extra,
        }
        truth_total += truth
        system_total += system
        missed_total += missed
        extra_total += extra
    result.update(
        counted=system_total,
        true=truth_total,
        missed=missed_total,
        extra=extra_total,
        recall=round((truth_total - missed_total) / truth_total, 4) if truth_total else None,
        recall_interval_95=wilson(truth_total - missed_total, truth_total),
        precision=round((system_total - extra_total) / system_total, 4) if system_total else None,
        precision_interval_95=wilson(system_total - extra_total, system_total),
        count_error=system_total - truth_total,
    )
    for row in labelled:
        cause = result["by_cause"].setdefault(
            row["cause"], {"windows": 0, "counted": 0, "true": 0}
        )
        cause["windows"] += 1
        cause["counted"] += row["system_in"] + row["system_out"]
        cause["true"] += row["label_in"] + row["label_out"]
    return result

"""Local, single-camera rehearsal against a real RTSP feed.

Frames stay in RAM. Event-scoped face templates resolve new/returning attendees;
directional passages confirm entry. Uncertain identities remain unresolved.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

from .domain import utc_now
from .face_registry import FaceRegistry, recount
from .recognizer import FACE_MODEL
from .recognizer import load as load_recognizer
from .store import EventStore

CAPTURE_STALL_SECONDS = 10


class LiveCalibration(BaseModel):
    axis: str = Field(default="vertical", pattern="^(vertical|horizontal)$")
    position: float = Field(default=0.5, ge=0.05, le=0.95)
    entry_positive: bool = True
    require_body_motion: bool = False
    counting_mode: str = Field(default="legacy", pattern="^(legacy|trajectory)$")
    anchor: str = Field(default="footpoint", pattern="^(footpoint|center)$")
    segment_start: float = Field(default=0.0, ge=0, le=1)
    segment_end: float = Field(default=1.0, ge=0, le=1)
    hysteresis: float = Field(default=0.025, ge=0.005, le=0.15)
    establish_seconds: float = Field(default=0.1, gt=0, le=2)
    confirm_seconds: float = Field(default=0.1, gt=0, le=2)
    max_gap_seconds: float = Field(default=1.0, gt=0, le=3)
    min_motion: float = Field(default=0.05, gt=0, le=0.5)
    confirm_distance: float | None = Field(default=None, gt=0, le=0.5)
    gate_start: tuple[float, float] | None = None
    gate_end: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_segment(self):
        if self.segment_start >= self.segment_end:
            raise ValueError("Counting segment must have positive length")
        if self.counting_mode == "trajectory":
            from .live_counting import PassageCalibration

            PassageCalibration.from_calibration(self)
        return self


class LiveControl(BaseModel):
    paused: bool


class SideEntryGate(BaseModel):
    """A plain entry tally: one direction across a vertical line, inside a band of rows.

    It shares the tracker with the main counter but keeps its own line, so moving the
    main counting line never changes it. Rows outside [y_min, y_max] (people walking
    behind a booth or out in the field) are not counted.
    """

    position: float = Field(default=0.5, ge=0.05, le=0.95)
    y_min: float = Field(default=0.0, ge=0, le=1)
    y_max: float = Field(default=1.0, ge=0, le=1)
    entry_positive: bool = True

    @model_validator(mode="after")
    def validate_band(self):
        if self.y_min >= self.y_max:
            raise ValueError("Side entry band must have positive height")
        return self


class ActivityLabel(BaseModel):
    entries: int = Field(ge=0, le=50)
    exits: int = Field(ge=0, le=50)
    actor: str = Field(default="operator", min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=300)


@dataclass
class CrossingState:
    side: int = 0
    pending_side: int = 0
    stable: int = 0
    last_seen: float = 0
    motion_origin: float | None = None


class CrossingCounter:
    """Count stable side changes, including later return trips on the same track."""

    def __init__(self, calibration: LiveCalibration, stable_frames: int = 3):
        self.calibration = calibration
        self.stable_frames = stable_frames
        self.states: dict[str, CrossingState] = {}

    def update(
        self,
        track_id: str,
        point: tuple[float, float],
        now: float,
        body_anchor: tuple[float, float] | None = None,
    ) -> str | None:
        state = self.states.setdefault(track_id, CrossingState())
        if now - state.last_seen > 3:
            state.side = state.pending_side = state.stable = 0
            state.motion_origin = None
        state.last_seen = now
        coordinate = point[0 if self.calibration.axis == "vertical" else 1]
        anchor = (
            body_anchor[0 if self.calibration.axis == "vertical" else 1] if body_anchor else None
        )
        if state.motion_origin is None:
            state.motion_origin = anchor
        distance = coordinate - self.calibration.position
        if abs(distance) < 0.025:
            state.pending_side = state.stable = 0
            return None
        side = 1 if distance > 0 else -1
        if anchor is not None and state.side == side:
            state.motion_origin = (max if side > 0 else min)(state.motion_origin, anchor)
        if side != state.pending_side:
            state.pending_side, state.stable = side, 1
        else:
            state.stable += 1
        if state.stable < self.stable_frames or side == state.side:
            return None
        if (
            state.side
            and self.calibration.require_body_motion
            and (
                anchor is None
                or state.motion_origin is None
                or side * (anchor - state.motion_origin) < 0.015
            )
        ):
            return None
        before, state.side = state.side, side
        state.motion_origin = anchor
        if before == 0:
            return None
        return "in" if (side > 0) == self.calibration.entry_positive else "out"

    def expire(self, now: float) -> None:
        self.states = {
            key: state for key, state in self.states.items() if now - state.last_seen <= 3
        }


@dataclass
class CameraConfig:
    host: str = ""
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    channel: str = "101"
    model: str = "IP camera"
    replay_clips: list[dict] = field(default_factory=list)
    playback_speed: float = 1.0
    paired_replay: dict | None = None
    event_id: str = "single-camera-test"
    camera_id: str = "single-camera-test-annke"
    count_passages: bool = True
    facing_camera: dict | None = field(default=None, repr=False)
    detection_device: str = "cpu"
    detector_imgsz: int = 640
    detector_roi: tuple[float, float, float, float] | None = None
    tracker_name: str | None = None
    second_look_model: str | None = None
    face_provider: str = "cpu"
    face_det_size: int = 1280
    face_person_rois: bool = False
    async_faces: bool = False
    # Face gates; the defaults were set for buffalo_l. Recognizers that hold up at wider
    # angles and smaller faces (e.g. KP-RPE) can open them per station.
    face_max_yaw: float = 25.0
    face_max_pitch: float = 20.0
    face_small_min_det: float = 0.85
    # Pairwise similarity that ties a track's face samples into one consistent cluster.
    face_cluster_link: float = 0.5
    # Optional plain entry tally on the counting camera (see SideEntryGate).
    side_entry_gate: dict | None = None
    # Published entries = counted entries x this. Set from a recount of the station's own
    # recordings when it is known to miss crossings; raw counts stay in the database and
    # under "counts_raw". Exits are not scaled. When the station's miss rate changes (e.g.
    # it is made faster), entry_count_scale_until (ISO time) ends that period and entries
    # observed from then on are multiplied by entry_count_scale_after instead.
    entry_count_scale: float = 1.0
    entry_count_scale_until: str | None = None
    entry_count_scale_after: float = 1.0
    # A dated list of {"from": ISO time with zone, "scale": x} that takes over from its first
    # "from" on: each entry is weighted by the latest scale whose "from" has passed, so a whole
    # day can be corrected period by period from an audit of the station's recordings. Raw
    # counts stay in the database. A scale of 0 leaves out set-up and test crossings.
    entry_count_schedule: list[dict] | None = None
    # False runs a counting-only station: no face detection or recognition, which frees
    # the GPU for the person detector.
    faces: bool = True
    # False leaves the facing camera out of the live service (it stays in this file for
    # other readers, e.g. the recorder).
    facing_live: bool = True
    # Frames wider than this are shrunk in the capture process before they are passed on.
    # Re-encoding and decoding every full 2560x1920 frame held Booth One near 8 fps, and the
    # person detector only looks at about 960 pixels. None keeps the camera's resolution.
    capture_max_width: int | None = None
    appearance_model: str | None = None
    target_fps: float = 10.0
    association_threshold: float = 0.85
    association_margin: float = 0.08
    association_min_samples: int = 2
    association_overlap_seconds: float | None = None
    association_sole_candidate_threshold: float | None = None
    # Position evidence for the cross-camera link: "off", "shadow" or "on".
    association_zone: str = "off"
    association_zone_min_samples: int = 20
    # Save a short film strip of every active moment at the door for labelling.
    activity_capture: bool = False
    activity_capture_max_mb: int = 600
    # Save aligned face crops for a shadow recognizer running in its own process.
    face_shadow: bool = False
    face_shadow_max_mb: int = 500
    rtsp_path: str | None = None

    @property
    def url(self) -> str:
        path = self.rtsp_path or f"/Streaming/Channels/{self.channel}"
        return (
            f"rtsp://{quote(self.username, safe='')}:{quote(self.password, safe='')}"
            f"@{self.host}:554/{path.lstrip('/')}"
        )


class LiveSession:
    def __init__(
        self,
        store: EventStore,
        config_path: Path,
        model_path: str,
        camera_config: CameraConfig | None = None,
    ):
        self.store = store
        self.config_path = config_path
        self.config = camera_config or CameraConfig(**json.loads(config_path.read_text()))
        self.model_path = model_path
        self.event_id = self.config.event_id
        self.camera_id = self.config.camera_id
        for name in ("entry_count_scale", "entry_count_scale_after"):
            scale = getattr(self.config, name)
            if not (isinstance(scale, int | float) and math.isfinite(scale) and 1 <= scale <= 10):
                raise ValueError(f"{name} must be between 1 and 10")
        self.scale_cutoff = None
        if self.config.entry_count_scale_until:
            cutoff = datetime.fromisoformat(self.config.entry_count_scale_until)
            if cutoff.tzinfo is None:
                raise ValueError("entry_count_scale_until must include a timezone")
            self.scale_cutoff = cutoff.astimezone(UTC).isoformat(timespec="milliseconds")
        self.scale_schedule: list[tuple[str, float]] = []
        for item in self.config.entry_count_schedule or []:
            start, scale = datetime.fromisoformat(item["from"]), item["scale"]
            if start.tzinfo is None:
                raise ValueError("entry_count_schedule times must include a timezone")
            if not (isinstance(scale, int | float) and math.isfinite(scale) and 0 <= scale <= 10):
                raise ValueError("entry_count_schedule scales must be between 0 and 10")
            self.scale_schedule.append(
                (start.astimezone(UTC).isoformat(timespec="milliseconds"), float(scale))
            )
        self.scale_schedule.sort()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.calibration = LiveCalibration()
        self.side_entry_gate = (
            SideEntryGate(**self.config.side_entry_gate)
            if self.config.side_entry_gate and self.config.count_passages
            else None
        )
        self.counter = self._new_counter()
        self.paused = False
        self.status = "starting"
        self.error: str | None = None
        self.jpeg: bytes | None = None
        self.frame = None
        self.frame_sequence = 0
        self.processed_sequence = 0
        self.frame_time = 0.0
        self.processed_time = 0.0
        self.generation = 0
        self.frames_processed = 0
        self.people_visible = 0
        self.fps = 0.0
        self.size = [0, 0]
        self.threads: list[threading.Thread] = []
        self.face_registry: FaceRegistry | None = None
        self.face_status = "loading"
        self.face_samples_processed = 0
        self.body_face_samples = 0
        self.face_detections = 0
        self.face_scans = 0
        self.peak_people_visible = 0
        self.track_handovers = 0
        self.tracking_history = deque(maxlen=240)
        self.runtime = {}
        self.face_worker = None
        self.capture_restarts = -1
        self._fps_times = deque(maxlen=60)
        self._last_association_error = None
        self.face_previews: dict[str, bytes] = {}
        self.face_preview_quality: dict[str, float] = {}
        self.active_faces: list[dict] = []
        self.frame_ring = None
        self.activity = None
        self.face_spool = None
        self._stage_ms: dict[str, dict[str, float]] = {"idle": {}, "with_people": {}}
        self._busy_frames = 0
        self._busy_seconds = 0.0

    def _time_stages(self, busy: bool, period: float | None, **stages: float) -> None:
        """Running mean of where a frame's time goes, split by whether anyone is in view."""
        bucket = self._stage_ms["with_people" if busy else "idle"]
        for name, seconds in stages.items():
            value = seconds * 1000
            bucket[name] = round(bucket[name] * 0.95 + value * 0.05, 1) if name in bucket else value
        if busy and period is not None:
            self._busy_frames += 1
            self._busy_seconds += min(period, 1.0)
        self.runtime["stage_ms"] = {key: dict(value) for key, value in self._stage_ms.items()}
        if self._busy_seconds:
            self.runtime["fps_with_people"] = round(self._busy_frames / self._busy_seconds, 1)
            self.runtime["frames_with_people"] = self._busy_frames

    def _new_counter(self):
        # Every reset of the main counter (reconnect, recalibration) also drops the side
        # entry tally's in-flight crossings; recorded entries are kept.
        gate = getattr(self, "side_entry_gate", None)
        self.side_entry_counter = (
            CrossingCounter(
                LiveCalibration(
                    axis="vertical", position=gate.position, entry_positive=gate.entry_positive
                )
            )
            if gate
            else None
        )
        if self.calibration.counting_mode == "trajectory":
            from .live_counting import PassageCalibration, PassageCounter

            return PassageCounter(PassageCalibration.from_calibration(self.calibration))
        return CrossingCounter(self.calibration)

    def start(self, start_workers: bool = True) -> None:
        self.store.data_root.chmod(0o700)
        self.store.db_path.chmod(0o600)
        now = utc_now()
        with self.store.transaction() as db:
            db.execute(
                """INSERT OR IGNORE INTO events
                (id,title,event_date,status,source_label,started_at,created_at)
                VALUES (?, 'Single camera test', ?, 'live', 'Live RTSP camera', ?, ?)""",
                (self.event_id, now[:10], now, now),
            )
            db.execute(
                "INSERT OR IGNORE INTO event_counters (event_id,updated_at) VALUES (?,?)",
                (self.event_id, now),
            )
            db.execute(
                """INSERT OR IGNORE INTO cameras
                (id,event_id,name,role,lane,source,status,configuration)
                VALUES (?,?,?,?,'A',?,'starting',?)""",
                (
                    self.camera_id,
                    self.event_id,
                    self.camera_id,
                    "passage_counter" if self.config.count_passages else "face_identity",
                    f"rtsp://{self.config.host}",
                    self.calibration.model_dump_json(),
                ),
            )
            saved = db.execute(
                "SELECT configuration FROM cameras WHERE id=?", (self.camera_id,)
            ).fetchone()
            self.calibration = LiveCalibration.model_validate_json(saved[0])
            self.counter = self._new_counter()
        self.face_registry = FaceRegistry(
            self.store, self.event_id, self.camera_id, cluster_link=self.config.face_cluster_link
        )
        if start_workers:
            # Loaded here, before any camera thread exists; a failure stops the start.
            load_recognizer()
        self.face_previews, self.face_preview_quality = self.store.load_face_previews(self.event_id)
        if self.config.activity_capture and not getattr(self, "is_replay", False):
            from .activity import ActivityRecorder, FrameRing

            self.frame_ring = FrameRing()
            if self.config.count_passages:
                self.activity = ActivityRecorder(
                    self.store,
                    self.event_id,
                    self.camera_id,
                    {"side": self.frame_ring},
                    max_bytes=self.config.activity_capture_max_mb * 1024 * 1024,
                )
        if self.config.face_shadow and not getattr(self, "is_replay", False):
            from .face_spool import FOLDER, FaceSpool

            self.face_spool = FaceSpool(
                self.store._event_directory(self.event_id) / FOLDER,
                self.camera_id,
                self.config.face_shadow_max_mb * 1024 * 1024,
            )
        if start_workers:
            self._start_workers()

    def _start_workers(self):
        for target in (self._capture, self._process):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        if self.face_worker:
            self.face_worker.stop()
        if self.activity:
            self.activity.stop()
        if self.face_spool:
            self.face_spool.stop()
        for thread in self.threads:
            thread.join(timeout=7)
        if self.face_registry:
            with self.lock:
                self.face_registry.flush_evidence()

    def _capture(self) -> None:
        # Separate decoder from inference libraries; kill and replace a blocked decoder.
        import cv2
        import numpy as np

        while not self.stop_event.is_set():
            self.capture_restarts += 1
            process = subprocess.Popen(
                [sys.executable, "-m", "ingress.capture_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            selector = selectors.DefaultSelector()
            try:
                process.stdin.write((json.dumps(asdict(self.config)) + "\n").encode())
                process.stdin.close()
                selector.register(process.stdout, selectors.EVENT_READ)
                buffer = bytearray()
                last_frame = time.monotonic()
                with self.lock:
                    self.generation += 1
                while not self.stop_event.is_set():
                    if time.monotonic() - last_frame > CAPTURE_STALL_SECONDS:
                        break
                    if not selector.select(timeout=0.25):
                        continue
                    chunk = os.read(process.stdout.fileno(), 1024 * 1024)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while len(buffer) >= 12:
                        length, captured_at = struct.unpack("!Id", buffer[:12])
                        if length > 16 * 1024 * 1024:
                            raise ValueError("Invalid capture packet")
                        if len(buffer) < length + 12:
                            break
                        encoded = bytes(buffer[12 : 12 + length])
                        frame = cv2.imdecode(
                            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        del buffer[: 12 + length]
                        if frame is None:
                            continue
                        if self.frame_ring:
                            self.frame_ring.push(captured_at, encoded)
                        last_frame = time.monotonic()
                        with self.lock:
                            self.frame = frame
                            self.frame_time = captured_at
                            self.frame_observed_at = (
                                datetime.now(UTC)
                                - timedelta(seconds=max(0, time.monotonic() - captured_at))
                            ).isoformat(timespec="milliseconds")
                            self.frame_sequence += 1
                            self.size = [frame.shape[1], frame.shape[0]]
            except Exception:
                pass  # Never log credential-bearing decoder exceptions.
            finally:
                selector.close()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
                with self.lock:
                    self.status = "disconnected"
                    self.frame = None
                    self.jpeg = None
            self.stop_event.wait(1)

    def _record(
        self,
        track_id: str,
        direction: str,
        observed_at: str | None = None,
        captured_at: float | None = None,
    ) -> None:
        if not self.config.count_passages:
            return
        now = observed_at or utc_now()
        passage_id = str(uuid.uuid4())
        face = self.face_registry.tracks.get(track_id) if self.face_registry else None
        identity_id = (
            face.identity_id if face and face.state in {"new", "matched", "tracked"} else None
        )
        with self.store.transaction() as db:
            db.execute(
                """INSERT INTO passages
                (id,event_id,camera_id,lane,track_id,observed_at,direction,status,evidence,
                 identity_id,face_quality)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    passage_id,
                    self.event_id,
                    self.camera_id,
                    getattr(self, "source_info", {}).get("station", "A"),
                    track_id,
                    now,
                    direction,
                    "face_matched" if identity_id else "unresolved",
                    json.dumps(
                        {
                            "source": "live",
                            "identity_resolution": bool(identity_id),
                            "face_samples": len(face.samples) if face else 0,
                            "face_state": face.state if face else "no_face",
                            "face_similarity": face.similarity if face else None,
                            "identity_basis": (
                                "continuous_track"
                                if identity_id and face.state == "tracked"
                                else "face_match"
                                if identity_id
                                else "unresolved"
                            ),
                            "calibration": self.calibration.model_dump(),
                            "recording": getattr(self, "source_info", None),
                        }
                    ),
                    identity_id,
                    face.quality if face else 0,
                ),
            )
            recount(db, self.event_id)
        if self.activity:
            self.activity.passage(
                passage_id, direction, captured_at if captured_at is not None else time.monotonic()
            )
        if hasattr(self, "association"):
            # A crossing that already has a face is registered only as a labelled
            # check of the position evidence; the coordinator never links it.
            self.association.register_crossing(
                passage_id,
                track_id,
                captured_at if captured_at is not None else time.monotonic(),
                direction=direction,
                known_identity=identity_id,
            )

    def _update_side_entry(self, key, track_id, x, y, now, observed_at):
        """Count a left-to-right (entry side) crossing of the side entry line in its band."""
        gate = self.side_entry_gate
        direction = self.side_entry_counter.update(track_id, (x, y), now)
        if direction != "in" or not gate.y_min <= y <= gate.y_max:
            return
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO side_entries (id,event_id,camera_id,track_id,observed_at,y) "
                "VALUES (?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    self.event_id,
                    self.camera_id,
                    key,
                    observed_at or utc_now(),
                    round(y, 4),
                ),
            )

    def _resolve_cross_view(self, now):
        if not hasattr(self, "association"):
            return
        for proposal in self.association.resolve_pending(now):
            with self.store.transaction() as db:
                row = db.execute(
                    "SELECT evidence FROM passages "
                    "WHERE id=? AND event_id=? AND identity_id IS NULL",
                    (proposal.event_id, self.event_id),
                ).fetchone()
                if row is None:
                    continue
                valid = db.execute(
                    "SELECT 1 FROM face_templates WHERE identity_id=? AND event_id=?",
                    (proposal.identity_id, self.event_id),
                ).fetchone()
                if not valid:
                    continue
                evidence = json.loads(row["evidence"])
                evidence.update(
                    identity_resolution=True,
                    identity_basis=proposal.basis,
                    cross_view={
                        "facing_track": proposal.facing_track_id,
                        "facing_tracks": list(proposal.facing_tracks),
                        "side_track": proposal.side_track_id,
                        "appearance_score": proposal.appearance_score,
                        "margin": proposal.margin,
                        "provenance": proposal.provenance,
                    },
                )
                db.execute(
                    "UPDATE passages SET identity_id=?,status='face_matched',evidence=? "
                    "WHERE id=? AND identity_id IS NULL",
                    (proposal.identity_id, json.dumps(evidence), proposal.event_id),
                )
                recount(db, self.event_id)
        # Keep the final reason and scores for every crossing that was not
        # linked. Thresholds can then be calibrated from real decisions, and the
        # global resolver can use an appearance match that only lacked a face.
        terminal = self.association.drain_terminal()
        rejected = [d for d in terminal if d["reason"] == "expired_unresolved"]
        zoned = [d for d in terminal if d.get("zone")]
        if rejected or zoned:
            facing_camera = getattr(self, "facing", self).camera_id
            with self.store.transaction() as db:
                # What the position evidence decided, kept for every crossing: the
                # resolver reads the facing tracks, and the samples retrain the zone.
                for decision in zoned:
                    record = {**decision["zone"], "facing_camera_id": facing_camera}
                    db.execute(
                        "UPDATE passages SET evidence=json_set(evidence,"
                        "'$.cross_view_zone',json(?)) WHERE id=? AND event_id=?",
                        (json.dumps(record, default=float), decision["event_id"], self.event_id),
                    )
                for decision in rejected:
                    record = {
                        key: decision.get(key)
                        for key in (
                            "last_reject_reason", "best_candidate", "appearance_score",
                            "second_score", "facing_margin", "reciprocal_margin",
                            "side_samples", "facing_samples", "temporal_rejections",
                            "appearance_unavailable", "face_ready", "sole_candidate",
                        )
                    }
                    record["facing_camera_id"] = facing_camera
                    # Seconds relative to the crossing; a track that starts after it
                    # or ends before it was not visible at the crossing instant.
                    for key in ("side_first_seen", "side_last_seen",
                                "facing_first_seen", "facing_last_seen"):
                        if decision.get(key) is not None:
                            record[key.replace("_seen", "_offset")] = round(
                                decision[key] - decision["crossing_at"], 3
                            )
                    db.execute(
                        "UPDATE passages SET evidence=json_set(evidence,"
                        "'$.cross_view_rejected',json(?)) WHERE id=? AND event_id=?",
                        (json.dumps(record, default=float), decision["event_id"], self.event_id),
                    )

    def _save_face_preview(self, identity_id: str, jpeg: bytes, quality: float):
        if quality <= self.face_preview_quality.get(identity_id, -1) + 0.02:
            return
        if self.store.save_face_preview(self.event_id, identity_id, jpeg, quality):
            self.face_previews[identity_id] = jpeg
            self.face_preview_quality[identity_id] = quality

    def _apply_face_evidence(
        self,
        registry,
        namespace,
        active_keys,
        body_face_refs,
        detections,
        faces,
        face_keys,
        face_time,
        face_observed_at,
        face_frame,
        tracking_time,
    ):
        import cv2

        height, width = face_frame.shape[:2]
        self.face_scans += 1
        self.face_detections += len(detections)
        face_boxes = []
        body_for_face = {id(detected): f"{namespace}:{key}" for key, detected in faces.items()}
        active_face_keys = {f"{namespace}:{key}" for key in face_keys.values()}
        for index, detected in enumerate(detections):
            face_key = f"{namespace}:{face_keys[index]}"
            body_key = body_for_face.get(id(detected))
            face = registry.observe(
                face_key,
                detected.embedding,
                detected.quality,
                active_face_keys,
                face_time,
                observed_at=face_observed_at,
            )
            if body_key:
                registry.observe(
                    body_key,
                    detected.embedding,
                    detected.quality,
                    active_keys,
                    face_time,
                    observed_at=face_observed_at,
                )
                body_face_refs[body_key] = (face_key, face_time)
                self.body_face_samples += 1
            body_face_refs = {
                key: ref for key, ref in body_face_refs.items() if tracking_time - ref[1] <= 1.5
            }
            for owner, (reference, reference_time) in body_face_refs.items():
                if reference == face_key:
                    registry.bind_face_to_body(
                        owner, face, active_keys, face_time, face_observed_at
                    )
                    bound = registry.tracks.get(owner)
                    if (
                        hasattr(self, "association")
                        and bound
                        and bound.identity_id
                        and bound.state in {"new", "matched", "tracked"}
                    ):
                        self.association.attach_identity(
                            self.association_view,
                            owner,
                            bound.identity_id,
                            "face_match"
                            if reference_time == face_time
                            else "face_track_continuity",
                            reference_time,
                        )
            self.face_samples_processed += 1
            face_boxes.append(detected.bbox)
            # A visitor with no photo yet takes the first accepted sighting, so no card
            # stays blank; later photos must still resemble the stored profile closely.
            if face.identity_id and (
                (
                    face.identity_id not in self.face_previews
                    and face.state in {"new", "matched", "tracked"}
                )
                or (
                    face.state in {"new", "matched"}
                    and registry.identity_similarity(detected.embedding, face.identity_id) >= 0.6
                )
            ):
                left, top, right, bottom = map(int, detected.bbox)
                crop = face_frame[
                    max(0, top) : min(height, bottom),
                    max(0, left) : min(width, right),
                ]
                if crop.size:
                    crop = cv2.resize(crop, (112, 112))
                    ok, jpeg = cv2.imencode(".jpg", crop)
                    if ok:
                        self._save_face_preview(face.identity_id, jpeg.tobytes(), detected.quality)
        if detections:
            # Empty scans used to flush every real track out of this short history.
            self.tracking_history.append(
                {
                    "source": "faces",
                    "time": face_observed_at,
                    "tracks": [],
                    "faces": [
                        {
                            "quality": round(face.quality, 3),
                            "width": round(face.bbox[2] - face.bbox[0]),
                            "track": face_keys.get(index),
                        }
                        for index, face in enumerate(detections)
                    ],
                }
            )
        return body_face_refs, face_boxes

    @staticmethod
    def _valid_box(box, width, height):
        return (
            len(box) == 4
            and all(math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width
            and 0 <= box[1] < box[3] <= height
        )

    def _process(self) -> None:
        try:
            import cv2

            from .vision.adapters import LiveFaceEncoder, UltralyticsPersonTracker, associate_faces
            from .vision.continuity import FaceContinuity, TrackContinuity
            from .vision.contracts import FramePacket

            # Keep low-score detections for ByteTrack's recovery stage; the tracker
            # applies its own stricter threshold when creating a new person track.
            replay = getattr(self, "is_replay", False)
            advanced = self.config.async_faces and not replay
            crowd_mode = self.config.tracker_name == "crowd-consensus"
            second_look = None
            if self.config.second_look_model:
                from .vision.second_look import RFPersonDetector, SecondLook

                if not crowd_mode:
                    raise ValueError("Second look requires crowd-consensus tracking")
                path = self.config.second_look_model
                second_look = SecondLook(RFPersonDetector(path), path)
            if crowd_mode:
                from .vision.crowd import CrowdPersonTracker

                tracker = CrowdPersonTracker(
                    self.model_path, "crowd-consensus", confidence=.1,
                    device=self.config.detection_device, imgsz=self.config.detector_imgsz,
                    roi=self.config.detector_roi, second_look=second_look,
                )
            elif advanced:
                tracker = UltralyticsPersonTracker(
                    self.model_path, device=self.config.detection_device,
                    imgsz=self.config.detector_imgsz, roi=self.config.detector_roi,
                    tracker=self.config.tracker_name,
                )
            else:
                tracker = UltralyticsPersonTracker(self.model_path)
            self.runtime.update(
                detector_device=getattr(tracker, "selected_device", "cpu"),
                tracker=self.config.tracker_name or "bytetrack",
                detector_model=Path(self.model_path).name,
                detector_imgsz=self.config.detector_imgsz,
            )
            appearance_encoder = None
            if advanced and self.config.appearance_model:
                try:
                    from .vision.reid import AppearanceEncoder

                    appearance_encoder = AppearanceEncoder(self.config.appearance_model)
                    self.runtime["appearance"] = "osnet"
                except Exception as exc:
                    self.runtime["appearance"] = f"unavailable ({type(exc).__name__})"
            last_appearance = -1.0
            last_detector_device = getattr(tracker, "selected_device", "cpu")
            continuity = TrackContinuity()
            face_continuity = FaceContinuity()
            body_face_refs = {}
            faces_on = self.config.faces
            if advanced and faces_on:
                from .live_faces import AsyncFaceWorker, FaceRequest

                self.face_worker = AsyncFaceWorker(
                    lambda: LiveFaceEncoder(
                        provider=self.config.face_provider,
                        det_size=self.config.face_det_size,
                        person_rois=self.config.face_person_rois,
                        tap=self.face_spool.offer if self.face_spool else None,
                        max_yaw_degrees=self.config.face_max_yaw,
                        max_pitch_degrees=self.config.face_max_pitch,
                        small_face_min_det=self.config.face_small_min_det,
                    )
                )
                self.face_worker.start()
                face_encoder = None
            elif not faces_on:
                face_encoder = None
                with self.lock:
                    self.face_status = "off"
            else:
                face_encoder = (
                    LiveFaceEncoder(tap=self.face_spool.offer)
                    if self.face_spool
                    else LiveFaceEncoder()
                )
                with self.lock:
                    self.face_status = "ready"
            registry = self.face_registry
            assert registry is not None
            last_sequence = last_generation = -1
            last_finished = 0.0
            last_face_scan = -1.0
            last_face_submit = last_face_seen = -math.inf
            face_boxes = []
            namespace = uuid.uuid4().hex[:8]
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    frame = self.frame
                    sequence, generation = self.frame_sequence, self.generation
                    frame_time, paused = self.frame_time, self.paused
                    tracking_time = getattr(self, "analysis_time", frame_time or started)
                    observed_at = getattr(self, "frame_observed_at", None)
                replay = getattr(self, "is_replay", False)
                if (
                    frame is None
                    or sequence == last_sequence
                    or (started - frame_time > 3 and not replay)
                    or (replay and paused)
                ):
                    # Poll finely: a 30 ms wait for the next frame held a 25 fps camera
                    # whose frames take 30 ms to analyse near 12 fps.
                    self.stop_event.wait(0.005)
                    continue
                last_sequence = sequence
                if generation != last_generation:
                    tracker.reset()
                    continuity.reset()
                    face_continuity.reset()
                    body_face_refs.clear()
                    if self.face_worker:
                        self.face_worker.reset()
                    namespace = uuid.uuid4().hex[:8]
                    with self.lock:
                        self.counter = self._new_counter()
                        registry.clear_tracks()
                        face_boxes = []
                    last_generation = generation
                packet = FramePacket(
                    self.camera_id,
                    sequence,
                    int(frame_time * 1_000_000_000),
                    observed_at or utc_now(),
                    frame,
                )
                if second_look:
                    tracker.gate = self.calibration
                tracks = [] if paused else tracker.track(packet)
                frame_height, frame_width = frame.shape[:2]
                tracks = [
                    track
                    for track in tracks
                    if self._valid_box(track.bbox, frame_width, frame_height)
                ]
                tracked_at = time.monotonic()
                if self.activity and not paused:
                    self.activity.observe(
                        tracking_time, observed_at, len(tracks), frame, self.calibration
                    )
                detector_device = getattr(tracker, "selected_device", "cpu")
                if detector_device != last_detector_device:
                    # A backend fallback restarts model tracking and may reuse raw IDs.
                    continuity.reset()
                    face_continuity.reset()
                    body_face_refs.clear()
                    namespace = uuid.uuid4().hex[:8]
                    if self.face_worker:
                        self.face_worker.reset()
                    with self.lock:
                        self.counter = self._new_counter()
                        registry.clear_tracks()
                        face_boxes = []
                        if hasattr(self, "association"):
                            self.association.reset()
                    last_detector_device = detector_device
                learned = {}
                if not paused:
                    if appearance_encoder and tracks and tracking_time - last_appearance >= 0.2:
                        try:
                            learned = appearance_encoder.encode(frame, tracks)
                            last_appearance = tracking_time
                        except Exception as exc:
                            self.runtime["appearance"] = f"unavailable ({type(exc).__name__})"
                            appearance_encoder = None
                    raw_features = learned
                    tracks = (
                        continuity.update(
                            tracks, frame, tracking_time, learned_appearance=raw_features
                        )
                        if advanced
                        else continuity.update(tracks, frame, tracking_time)
                    )
                    learned = {
                        continuity.raw_ids[raw]: vector
                        for raw, vector in raw_features.items()
                        if raw in continuity.raw_ids
                    }
                appearance_at = time.monotonic()
                face_batches = []
                faces = None
                detections = []
                face_keys = {}
                face_time, face_observed_at, face_frame = tracking_time, observed_at, frame
                # Don't miss a brief readable face just before/after a crossing.
                scan_interval = 0.4
                for candidate in tracks:
                    state = registry.tracks.get(f"{namespace}:{candidate.track_id}")
                    if state is None or state.identity_id is None or state.state == "conflict":
                        scan_interval = 1 / 7.5
                        break
                if advanced and self.face_worker:
                    self.face_status = self.face_worker.status
                    self.runtime.update(
                        face_provider=self.face_worker.provider,
                        face_model=FACE_MODEL,
                        face_error=self.face_worker.error,
                        face_fallback=self.face_worker.fallback_reason,
                        face_pose_gate=self.face_worker.pose_gate,
                    )
                    if not paused:
                        hints = {
                            track.track_id: registry.templates[state.identity_id]
                            for track in tracks
                            if (state := registry.tracks.get(f"{namespace}:{track.track_id}"))
                            and state.identity_id
                            and state.state != "conflict"
                        }
                        # An empty counting view is scanned twice a second, not on every
                        # frame: it had run 325,891 scans for one face.
                        # The recognition camera keeps scanning every frame.
                        if (
                            not self.config.count_passages
                            or tracks
                            or tracking_time - last_face_seen <= 2.0
                            or tracking_time - last_face_submit >= 0.5
                        ):
                            last_face_submit = tracking_time
                            self.face_worker.submit(
                                FaceRequest(
                                    frame,
                                    tracks,
                                    hints,
                                    sequence,
                                    generation,
                                    frame_time,
                                    observed_at or utc_now(),
                                )
                            )
                        while result := self.face_worker.take(generation, time.monotonic()):
                            face_batches.append(result)
                            if result.detections:
                                last_face_seen = tracking_time
                            self.runtime["face_inference_ms"] = round(result.latency_ms, 1)
                        self.runtime["face_delivery"] = dict(self.face_worker.metrics)
                elif (
                    face_encoder is not None
                    and not paused
                    and tracking_time - last_face_scan >= scan_interval - 1e-8
                ):
                    detections = face_encoder.detect(frame)
                    hints = {}
                    for track in tracks:
                        previous = registry.tracks.get(f"{namespace}:{track.track_id}")
                        if previous and previous.identity_id and previous.state != "conflict":
                            hints[track.track_id] = registry.templates[previous.identity_id]
                    faces = associate_faces(detections, tracks, hints)
                    face_keys = face_continuity.update(detections, tracking_time)
                    last_face_scan = tracking_time
                faces_at = time.monotonic()
                display = frame.copy()
                height, width = display.shape[:2]
                with self.lock:
                    # Discard inference that straddled a pause/reconnect or became stale.
                    if (
                        (paused != self.paused and not replay)
                        or generation != self.generation
                        or (time.monotonic() - frame_time > 3 and not replay)
                    ):
                        if replay:
                            last_sequence = -1
                        continue
                    if tracks or detections:
                        self.tracking_history.append(
                            {
                                "time": observed_at or utc_now(),
                                "tracks": [
                                    {
                                        "id": track.track_id,
                                        "point": [
                                            round(track.footpoint[0] / width, 3),
                                            round(track.footpoint[1] / height, 3),
                                        ],
                                        "box": [
                                            round(v / (width if i % 2 == 0 else height), 3)
                                            for i, v in enumerate(track.bbox)
                                        ],
                                        "confidence": round(track.confidence, 3),
                                    }
                                    for track in tracks
                                ],
                                "faces": [
                                    {
                                        "quality": round(face.quality, 3),
                                        "width": round(face.bbox[2] - face.bbox[0]),
                                        "track": face_keys.get(i),
                                    }
                                    for i, face in enumerate(detections)
                                ],
                            }
                        )
                    active_keys = {f"{namespace}:{track.track_id}" for track in tracks}
                    registry.expire(active_keys, tracking_time)
                    if advanced:
                        for result in face_batches:
                            if result.epoch != self.face_worker.epoch:
                                continue
                            body_face_refs, face_boxes = self._apply_face_evidence(
                                registry,
                                namespace,
                                active_keys,
                                body_face_refs,
                                result.detections,
                                result.faces,
                                result.face_keys,
                                result.request.captured_at,
                                result.request.observed_at,
                                result.request.frame,
                                tracking_time,
                            )
                            last_face_scan = result.request.captured_at
                    elif faces is not None:
                        body_face_refs, face_boxes = self._apply_face_evidence(
                            registry,
                            namespace,
                            active_keys,
                            body_face_refs,
                            detections,
                            faces,
                            face_keys,
                            face_time,
                            face_observed_at,
                            face_frame,
                            tracking_time,
                        )
                    self.active_faces = []
                    for track in tracks:
                        key = f"{namespace}:{track.track_id}"
                        face = registry.tracks.get(key)
                        if self.calibration.anchor == "center":
                            x = (track.bbox[0] + track.bbox[2]) / 2
                            y = (track.bbox[1] + track.bbox[3]) / 2
                        else:
                            x, y = track.footpoint
                        if hasattr(self, "association"):
                            vector = learned.get(track.track_id)
                            self.association.observe(
                                self.association_view,
                                key,
                                tracking_time,
                                embedding=vector,
                                embedding_source="osnet" if vector is not None else None,
                                identity_id=face.identity_id
                                if face and face.state in {"new", "matched", "tracked"}
                                else None,
                                identity_source=(
                                    "face_track_continuity"
                                    if face.state == "tracked"
                                    else "face_match"
                                )
                                if face
                                and face.identity_id
                                and face.state in {"new", "matched", "tracked"}
                                else None,
                                box=(
                                    track.bbox[0] / width, track.bbox[1] / height,
                                    track.bbox[2] / width, track.bbox[3] / height,
                                ),
                            )
                        if self.config.count_passages:
                            cv2.circle(display, (int(x), int(y)), 5, (60, 140, 240), -1)
                        if self.activity:
                            self.activity.track(
                                tracking_time, track.track_id, x / width, y / height
                            )
                        direction = self.counter.update(
                            track.track_id,
                            (x / width, y / height),
                            tracking_time,
                            body_anchor=(x / width, track.bbox[1] / height),
                        )
                        if direction and self.config.count_passages:
                            self._record(key, direction, observed_at, captured_at=tracking_time)
                        if self.side_entry_counter is not None:
                            self._update_side_entry(key, track.track_id, x / width, y / height,
                                                    tracking_time, observed_at)
                        left, top, right, bottom = map(int, track.bbox)
                        cv2.rectangle(display, (left, top), (right, bottom), (150, 210, 40), 2)
                        label = (
                            face.label
                            if face and face.identity_id and face.state != "conflict"
                            else ""
                        )
                        if face:
                            self.active_faces.append(
                                {
                                    "track": track.track_id,
                                    "label": face.label,
                                    "state": face.state,
                                    "similarity": face.similarity,
                                    "samples": len(face.samples),
                                    "required_samples": face.required_samples,
                                    "identity_id": face.identity_id,
                                }
                            )
                        cv2.putText(
                            display,
                            label,
                            (left, max(top - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * width / 1280,
                            (150, 210, 40),
                            max(2, round(2 * width / 1280)),
                        )
                    if not paused and tracking_time - last_face_scan < 0.6:
                        for box in face_boxes:
                            left, top, right, bottom = map(int, box)
                            cv2.rectangle(display, (left, top), (right, bottom), (220, 160, 100), 2)
                    self._resolve_cross_view(tracking_time)
                    self.counter.expire(tracking_time)
                    if self.side_entry_counter is not None:
                        self.side_entry_counter.expire(tracking_time)
                    if replay and hasattr(self, "append_evidence"):
                        self.append_evidence(
                            tracks, registry, namespace, observed_at, tracking_time, width, height
                        )
                    config = self.calibration
                    if config.gate_start and config.gate_end:
                        start = (
                            int(width * config.gate_start[0]),
                            int(height * config.gate_start[1]),
                        )
                        end = (int(width * config.gate_end[0]), int(height * config.gate_end[1]))
                    elif config.axis == "vertical":
                        start = (int(width * config.position), int(height * config.segment_start))
                        end = (start[0], int(height * config.segment_end))
                    else:
                        start = (int(width * config.segment_start), int(height * config.position))
                        end = (int(width * config.segment_end), start[1])
                    if self.side_entry_gate is not None:
                        gate = self.side_entry_gate
                        band_x = int(width * gate.position)
                        cv2.line(
                            display,
                            (band_x, int(height * gate.y_min)),
                            (band_x, int(height * gate.y_max)),
                            (80, 200, 60),
                            8,
                        )
                    if self.config.count_passages:
                        cv2.line(display, start, end, (60, 140, 240), 2)
                        cv2.circle(display, start, 5, (60, 140, 240), -1)
                        cv2.circle(display, end, 5, (60, 140, 240), -1)

                        dx, dy = end[0] - start[0], end[1] - start[1]
                        length = max(math.hypot(dx, dy), 1)
                        sign = 1 if config.entry_positive else -1
                        # Axis horizontal gates are drawn left-to-right above; positive means down.
                        if config.axis == "horizontal" and not config.gate_start:
                            sign *= -1
                        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                        tip = (
                            int(center[0] + sign * dy / length * 65),
                            int(center[1] - sign * dx / length * 65),
                        )
                        cv2.arrowedLine(display, center, tip, (60, 140, 240), 3, tipLength=0.3)
                    if width > 1280:
                        display = cv2.resize(display, (1280, round(height * 1280 / width)))
                    ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.jpeg = jpeg.tobytes()
                    self.processed_time = frame_time
                    self.processed_sequence = sequence
                    self.people_visible = len(tracks)
                    self.peak_people_visible = max(self.peak_people_visible, len(tracks))
                    self.track_handovers = continuity.handovers
                    self.frames_processed += 1
                    finished = time.monotonic()
                    self.fps = (
                        round(1 / max(finished - last_finished, 0.001), 1)
                        if last_finished and not paused
                        else 0.0
                    )
                    self._fps_times.append(finished)
                    if len(self._fps_times) > 1:
                        self.fps = round(
                            (len(self._fps_times) - 1)
                            / max(self._fps_times[-1] - self._fps_times[0], 0.001),
                            1,
                        )
                    self._time_stages(
                        bool(tracks),
                        finished - last_finished if last_finished else None,
                        detect_track=tracked_at - started,
                        appearance=appearance_at - tracked_at,
                        faces=faces_at - appearance_at,
                        apply_render=finished - faces_at,
                        total=finished - started,
                        # Time between frames that is not model work: the pacing
                        # sleep plus waiting for the camera to deliver a frame.
                        between_frames=started - last_finished if last_finished else 0.0,
                    )
                    if self.activity:
                        self.runtime["activity_capture"] = dict(self.activity.metrics)
                    if self.face_spool:
                        self.runtime["face_shadow"] = dict(self.face_spool.metrics)
                    if second_look:
                        self.runtime["second_look"] = dict(second_look.metrics)
                    if crowd_mode:
                        self.runtime["crowd_handovers"] = tracker.total_handovers
                    self.runtime["detector_device"] = getattr(tracker, "selected_device", "cpu")
                    last_finished = finished
                    self.status = (
                        "complete"
                        if getattr(self, "replay_complete", False)
                        else ("paused" if paused else "live")
                    )
                self.stop_event.wait(
                    max(0, 1 / max(self.config.target_fps, 1) - (time.monotonic() - started))
                )
        except Exception as exc:
            # Exception messages from camera libraries may contain the stream URL.
            with self.lock:
                self.error = f"Processing stopped ({type(exc).__name__}); check model installation."
                self.status = "error"
                self.face_status = "error"

    def snapshot(self) -> dict:
        with self.lock:
            age = (
                round((time.monotonic() - self.processed_time) * 1000)
                if self.processed_time
                else None
            )
            status = self.status
            if status in {"live", "paused"} and (age is None or age > 3000):
                status = "disconnected"
            payload = {
                "enabled": True,
                "event_id": self.event_id,
                "camera": self.config.host,
                "model": self.config.model,
                "status": status,
                "error": self.error,
                "paused": self.paused,
                "frame_age_ms": age,
                "frames_processed": self.frames_processed,
                "frames_received": self.frame_sequence,
                "processing_fps": self.fps,
                "people_visible": self.people_visible,
                "resolution": self.size,
                "calibration": self.calibration.model_dump(),
                "identity_resolution": self.face_status == "ready",
                "count_passages": self.config.count_passages,
                "face_status": self.face_status,
                "face_samples_processed": self.face_samples_processed,
                "body_face_samples": self.body_face_samples,
                "face_detections": self.face_detections,
                "face_scans": self.face_scans,
                "peak_people_visible": self.peak_people_visible,
                "track_handovers": self.track_handovers,
                "active_faces": self.active_faces,
                "tracking_history": list(self.tracking_history),
                "runtime": dict(self.runtime),
                "capture_restarts": max(0, self.capture_restarts),
                "recognition": self.face_registry.summary() if self.face_registry else None,
            }
            if payload["recognition"]:
                for visitor in payload["recognition"]["visitors"]:
                    visitor["has_preview"] = visitor["identity_id"] in self.face_previews
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT direction,COUNT(*) FROM passages WHERE event_id=? GROUP BY direction",
                (self.event_id,),
            ).fetchall()
            raw = {"in": 0, "out": 0, **{row[0]: row[1] for row in rows}}
            payload["counts"] = {
                **raw, "in": self.weighted_entries(db, "passages", "AND direction='in'")
            }
            if self._scaling:
                payload["counts_raw"] = raw
            if self.side_entry_gate is not None:
                hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat(
                    timespec="milliseconds"
                )
                total, last_hour = db.execute(
                    "SELECT COUNT(*), COUNT(CASE WHEN observed_at>=? THEN 1 END) "
                    "FROM side_entries WHERE event_id=? AND camera_id=?",
                    (hour_ago, self.event_id, self.camera_id),
                ).fetchone()
                camera = ("AND camera_id=?", (self.camera_id,))
                payload["side_entries"] = {
                    "count": self.weighted_entries(db, "side_entries", *camera),
                    "last_hour": self.weighted_entries(
                        db, "side_entries", camera[0] + " AND observed_at>=?",
                        (self.camera_id, hour_ago),
                    ),
                    "count_raw": total,
                    "gate": self.side_entry_gate.model_dump(),
                }
            payload["recent"] = [
                dict(row)
                for row in db.execute(
                    "SELECT p.observed_at,p.direction,p.track_id,p.status,p.evidence,f.label "
                    "FROM passages p LEFT JOIN face_templates f ON p.identity_id=f.identity_id "
                    "WHERE p.event_id=? ORDER BY p.observed_at DESC LIMIT 12",
                    (self.event_id,),
                ).fetchall()
            ]
        for passage in payload["recent"]:
            evidence = json.loads(passage.pop("evidence"))
            passage["recording"] = evidence.get("recording")
            passage["identity_basis"] = evidence.get("identity_basis")
            passage["reviewed_replay"] = bool(evidence.get("reviewed_replay"))
        return payload

    @property
    def _scaling(self) -> bool:
        return (
            self.config.entry_count_scale != 1
            or self.config.entry_count_scale_after != 1
            or bool(self.scale_schedule)
        )

    def _entries_after_cutoff(self, db, table: str, where: str, params: tuple) -> int:
        if self.scale_cutoff is None:
            return 0
        return db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE event_id=? AND observed_at>=? {where}",
            (self.event_id, self.scale_cutoff, *params),
        ).fetchone()[0]

    def entry_weight(self, observed_at: str) -> float:
        for start, scale in reversed(self.scale_schedule):
            if observed_at >= start:
                return scale
        if self.scale_cutoff is not None and observed_at >= self.scale_cutoff:
            return self.config.entry_count_scale_after
        return self.config.entry_count_scale

    def scaled_entries(self, count: int, after_cutoff: int = 0) -> int:
        before = count - after_cutoff
        return round(
            before * self.config.entry_count_scale
            + after_cutoff * self.config.entry_count_scale_after
        )

    def weighted_entries(self, db, table: str, where: str = "", params: tuple = ()) -> int:
        """Entries in TABLE matching WHERE, each weighted by the scale in force when it was seen."""

        def count(low: str | None = None, high: str | None = None) -> int:
            sql, args = f"SELECT COUNT(*) FROM {table} WHERE event_id=? {where}", [self.event_id, *params]
            if low is not None:
                sql, args = sql + " AND observed_at>=?", [*args, low]
            if high is not None:
                sql, args = sql + " AND observed_at<?", [*args, high]
            return db.execute(sql, args).fetchone()[0]

        first = self.scale_schedule[0][0] if self.scale_schedule else None
        before = count(high=first)
        after_cutoff = count(self.scale_cutoff, first) if self.scale_cutoff is not None else 0
        total = (
            (before - after_cutoff) * self.config.entry_count_scale
            + after_cutoff * self.config.entry_count_scale_after
        )
        for index, (start, scale) in enumerate(self.scale_schedule):
            end = self.scale_schedule[index + 1][0] if index + 1 < len(self.scale_schedule) else None
            total += count(start, end) * scale
        return round(total)

    def calibrate(self, calibration: LiveCalibration) -> None:
        with self.lock:
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE cameras SET configuration=? WHERE id=?",
                    (calibration.model_dump_json(), self.camera_id),
                )
            self.calibration = calibration
            self.counter = self._new_counter()


def live_router() -> APIRouter:
    router = APIRouter(prefix="/api/live")
    restart_lock = threading.Lock()

    def session(request: Request) -> LiveSession:
        value = getattr(request.app.state, "live", None)
        if value is None:
            raise HTTPException(503, "No live camera configured")
        return value

    @router.get("/status")
    def status(request: Request):
        value = getattr(request.app.state, "live", None)
        return value.snapshot() if value else {"enabled": False}

    @router.get("/history")
    def history(request: Request, hours: float = 6, bucket_minutes: int = 10):
        """Counted passages per time bucket, oldest first, for the station page's chart."""
        if not 0 < hours <= 48 or not 1 <= bucket_minutes <= 120:
            raise HTTPException(422, "Unsupported history range")
        value = session(request)
        step = bucket_minutes * 60
        last = int(datetime.now(UTC).timestamp() // step) * step
        first = last - (math.ceil(hours * 3600 / step) - 1) * step
        buckets = {
            start: {"in": 0, "out": 0, "linked": 0} for start in range(first, last + step, step)
        }
        with value.store.connect() as db:
            rows = db.execute(
                "SELECT observed_at,direction,identity_id FROM passages WHERE event_id=?",
                (value.event_id,),
            ).fetchall()
        for observed_at, direction, identity_id in rows:
            try:
                moment = datetime.fromisoformat(observed_at).timestamp()
            except ValueError:
                continue
            slot = buckets.get(int(moment // step) * step)
            if slot is None or direction not in ("in", "out"):
                continue
            slot[direction] += 1
            slot["linked"] += int(identity_id is not None)
            if direction == "in":
                slot["weighted_in"] = slot.get("weighted_in", 0.0) + value.entry_weight(
                    observed_at
                )
        return {
            "bucket_seconds": step,
            "buckets": [
                {
                    "start": datetime.fromtimestamp(start, UTC).isoformat(),
                    "in": round(counts.pop("weighted_in", 0.0)),
                    "out": counts["out"],
                    "linked": counts["linked"],
                }
                for start, counts in buckets.items()
            ],
        }

    @router.get("/shadow")
    def shadow(request: Request):
        """What the shadow recognizers make of this event's faces. No templates."""
        from .face_spool import FOLDER

        value = session(request)
        path = value.store._event_directory(value.event_id) / FOLDER / "report.json"
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            report = None
        return {"enabled": bool(value.face_spool), "report": report}

    @router.get("/frame.jpg")
    def frame(request: Request, camera: str = "side"):
        value = session(request)
        if camera == "facing" and hasattr(value, "facing"):
            value = value.facing
        with value.lock:
            frozen_replay = getattr(value, "is_replay", False) and (
                value.paused or value.replay_complete
            )
            if not value.jpeg or (
                time.monotonic() - value.processed_time > 3 and not frozen_replay
            ):
                raise HTTPException(503, "Waiting for a fresh camera frame")
            return Response(
                value.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
            )

    @router.get("/faces/{identity_id}/preview.jpg")
    def face_preview(request: Request, identity_id: str):
        value = session(request)
        with value.lock:
            jpeg = value.face_previews.get(identity_id)
            if not jpeg:
                raise HTTPException(404, "No saved preview for this visitor")
            return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/activity")
    def activity_windows(
        request: Request, state: str = "unlabelled", limit: int = 50, since: str | None = None
    ):
        from . import activity

        if state not in {"unlabelled", "labelled", "all"}:
            raise HTTPException(422, "state must be unlabelled, labelled or all")
        value = session(request)
        return {
            "windows": activity.list_windows(
                value.store, value.event_id, state, max(1, min(limit, 500)), since
            ),
            "scorecard": activity.scorecard(value.store, value.event_id, since),
            "capturing": value.activity is not None,
        }

    @router.get("/activity/{window_id}/{view}.jpg")
    def activity_strip(request: Request, window_id: str, view: str):
        from . import activity

        value = session(request)
        path = activity.strip_path(value.store, value.event_id, window_id, view)
        if path is None:
            raise HTTPException(404, "No stored strip for this window")
        return Response(
            path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @router.post("/activity/{window_id}/label")
    def activity_label(
        request: Request, window_id: str, data: ActivityLabel, since: str | None = None
    ):
        from . import activity

        value = session(request)
        if not activity.label_window(
            value.store, value.event_id, window_id, data.entries, data.exits, data.actor, data.note
        ):
            raise HTTPException(404, "Unknown activity window")
        return activity.scorecard(value.store, value.event_id, since)

    @router.get("/scorecard")
    def scorecard(request: Request, since: str | None = None):
        from . import activity

        value = session(request)
        return activity.scorecard(value.store, value.event_id, since)

    @router.get("/review", include_in_schema=False)
    def review_page():
        return Response(
            Path(__file__).with_name("review.html").read_text(),
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @router.put("/calibration")
    def calibration(request: Request, data: LiveCalibration):
        session(request).calibrate(data)
        return data

    @router.post("/control")
    def control(request: Request, data: LiveControl):
        value = session(request)
        with value.lock:
            value.paused = data.paused
            if hasattr(value, "facing"):
                value.facing.paused = data.paused
                if value.facing.face_worker:
                    value.facing.face_worker.reset()
            if value.face_worker:
                value.face_worker.reset()
            if hasattr(value, "association"):
                value.association.reset()
            if not getattr(value, "is_replay", False):
                value.counter = value._new_counter()
                if value.face_registry:
                    value.face_registry.clear_tracks()
        return {"paused": value.paused}

    @router.post("/replay/start")
    def restart_replay(request: Request):
        with restart_lock:
            value = session(request)
            if not getattr(value, "is_replay", False):
                raise HTTPException(409, "This is a live camera, not a recording replay")
            if not value.replay_complete and not value.error:
                raise HTTPException(409, "Wait for the current replay to finish")
            value.stop()
            fresh = type(value)(value.store, value.config_path, value.model_path)
            fresh.start()
            request.app.state.live = fresh
            return {"event_id": fresh.event_id, "started": True}

    return router

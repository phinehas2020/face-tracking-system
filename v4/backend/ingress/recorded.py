"""Feed bounded recording excerpts through the same live recognition worker."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from .live import CrossingCounter, LiveCalibration, LiveSession


class ReplayClip(BaseModel):
    path: Path
    label: str
    camera: str = Field(pattern=r"^[a-z0-9-]+$")
    station: str
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0, le=600)
    calibration: LiveCalibration = Field(
        default_factory=lambda: LiveCalibration(
            axis="horizontal",
            position=0.75,
            entry_positive=True,
        )
    )


class RecordedSession(LiveSession):
    is_replay = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clips = [ReplayClip.model_validate(clip) for clip in self.config.replay_clips]
        if not self.clips or not 0.1 <= self.config.playback_speed <= 4:
            raise ValueError("Replay needs clips and a playback speed between 0.1 and 4")
        for clip in self.clips:
            if not clip.path.is_file():
                raise ValueError(f"Missing recording: {clip.path.name}")
        # Every launch is a separate measurement run, never count an old replay twice.
        self.event_id = f"fair-2025-replay-{uuid.uuid4().hex[:10]}"
        self.camera_id = f"{self.event_id}-{self.clips[0].camera}"
        self.replay_complete = False
        self.clip_index = 0
        self.source_info = {}
        self.clip_results = []
        self.replay_epoch = datetime.now(UTC)
        self.analysis_time = 0.0
        self.frame_observed_at = self.replay_epoch.isoformat(timespec="milliseconds")

    def _capture(self):
        import cv2

        capture = None
        try:
            # First-time model initialization must not consume the beginning of a clip.
            while self.face_status == "loading" and not self.stop_event.wait(0.1):
                pass
            if self.face_status != "ready":
                return
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE events SET title='2025 Fair crowd rehearsal',"
                    "source_label='Recorded excerpts; encoded playback time' WHERE id=?",
                    (self.event_id,),
                )
            for index, clip in enumerate(self.clips):
                if self.stop_event.is_set():
                    break
                camera_id = f"{self.event_id}-{clip.camera}"
                with self.store.transaction() as db:
                    db.execute(
                        """INSERT OR IGNORE INTO cameras
                        (id,event_id,name,role,lane,source,status,configuration)
                        VALUES (?,?,?,'recording',?,?,'replay',?)""",
                        (
                            camera_id,
                            self.event_id,
                            clip.camera,
                            clip.station,
                            str(clip.path),
                            clip.calibration.model_dump_json(),
                        ),
                    )
                    db.execute(
                        "UPDATE cameras SET name=?,role='recording',lane=?,source=? WHERE id=?",
                        (clip.camera, clip.station, str(clip.path), camera_id),
                    )
                capture = cv2.VideoCapture(str(clip.path), cv2.CAP_FFMPEG)
                fps = capture.get(cv2.CAP_PROP_FPS)
                total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if not capture.isOpened() or not fps or clip.start_seconds >= total / fps:
                    raise RuntimeError(f"Cannot decode selected excerpt: {clip.path.name}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(clip.start_seconds * fps))
                frame_count = min(
                    round(clip.duration_seconds * fps), int(total) - round(clip.start_seconds * fps)
                )
                with self.lock:
                    self.clip_index = index
                    self.camera_id = camera_id
                    self.calibration = clip.calibration
                    self.counter = CrossingCounter(self.calibration)
                    self.generation += 1
                    self.frame = None
                    self.jpeg = None
                    self.source_info = {
                        "file": clip.path.name,
                        "label": clip.label,
                        "station": clip.station,
                        "start_seconds": clip.start_seconds,
                        "duration_seconds": frame_count / fps,
                        "position_seconds": clip.start_seconds,
                        "clip": index + 1,
                        "total_clips": len(self.clips),
                        "encoded_fps": fps,
                        "playback_speed": self.config.playback_speed,
                        "all_frames": True,
                    }
                before_frames = self.frames_processed
                before_faces = self.face_samples_processed
                for number in range(frame_count):
                    if self.face_status == "error":
                        raise RuntimeError("Recognition processing failed")
                    tick = time.monotonic()
                    while self.paused and not self.stop_event.wait(0.05):
                        tick = time.monotonic()
                    if self.stop_event.is_set():
                        break
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed within excerpt: {clip.path.name}")
                    with self.lock:
                        self.frame = frame
                        self.frame_time = time.monotonic()
                        self.frame_sequence += 1
                        self.analysis_time = (
                            sum(c.duration_seconds for c in self.clips[:index]) + number / fps
                        )
                        self.frame_observed_at = (
                            self.replay_epoch + timedelta(seconds=self.analysis_time)
                        ).isoformat(timespec="milliseconds")
                        self.size = [frame.shape[1], frame.shape[0]]
                        self.source_info["position_seconds"] = round(
                            clip.start_seconds + number / fps, 3
                        )
                    self.stop_event.wait(
                        max(0, 1 / (fps * self.config.playback_speed) - (time.monotonic() - tick))
                    )
                    # Backpressure: the same frames reach the model on every run.
                    while self.processed_sequence < self.frame_sequence:
                        if self.stop_event.wait(0.005):
                            break
                        if self.face_status == "error":
                            raise RuntimeError("Recognition processing failed")
                capture.release()
                capture = None
                # Let inference finish its last frame before changing source identity.
                deadline = time.monotonic() + 5
                while self.processed_sequence < self.frame_sequence and time.monotonic() < deadline:
                    if self.stop_event.wait(0.05):
                        break
                self.clip_results.append(
                    {
                        "label": clip.label,
                        "camera": camera_id,
                        "file": clip.path.name,
                        "station": clip.station,
                        "start_seconds": clip.start_seconds,
                        "duration_seconds": frame_count / fps,
                        "frames_processed": self.frames_processed - before_frames,
                        "face_samples": self.face_samples_processed - before_faces,
                    }
                )
            if not self.stop_event.is_set():
                report = self.snapshot()
                report["status"] = "complete"
                event_dir = self.store.data_root / "events" / self.event_id
                event_dir.mkdir(parents=True, exist_ok=True)
                (event_dir / "replay-report.json").write_text(json.dumps(report, indent=2))
                with self.lock:
                    self.replay_complete = True
                    self.status = "complete"
        except Exception as exc:
            with self.lock:
                self.error = f"Replay stopped: {exc}"
                self.status = "error"
        finally:
            if capture is not None:
                capture.release()

    def calibrate(self, calibration):
        with self.lock:
            camera = self.clips[self.clip_index].camera
            payload = json.loads(self.config_path.read_text())
            for clip in payload["replay_clips"]:
                if clip["camera"] == camera:
                    clip["calibration"] = calibration.model_dump()
            temporary = self.config_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(self.config_path)
            for clip in self.clips:
                if clip.camera == camera:
                    clip.calibration = calibration
            super().calibrate(calibration)

    def append_evidence(self, tracks, registry, namespace, observed_at, clock, width, height):
        """Keep counting evidence for calibration comparisons, without embeddings/images."""
        directory = self.store.data_root / "events" / self.event_id
        directory.mkdir(parents=True, exist_ok=True)
        points = []
        for track in tracks:
            x, y = track.footpoint
            points.append(
                [f"{namespace}:{track.track_id}", x / width, y / height, track.bbox[1] / height]
            )
        links = {
            key: {"id": state.identity_id, "after": state.resolve_after}
            for key, state in registry.tracks.items()
            if key.startswith(f"{namespace}:track-")
            and state.identity_id
            and state.state != "conflict"
        }
        row = {
            "clip": self.clip_index,
            "at": observed_at,
            "clock": clock,
            "source": dict(self.source_info),
            "points": points,
            "links": links,
        }
        with (directory / "counting-evidence.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")

    def snapshot(self):
        result = super().snapshot()
        with self.lock:
            result.update(
                is_replay=True, source=dict(self.source_info), clips=list(self.clip_results)
            )
            if self.replay_complete:
                result["status"] = "complete"
            elif self.paused:
                result["status"] = "paused"
        with self.store.connect() as db:
            result["station_results"] = [
                dict(row)
                for row in db.execute(
                    """SELECT c.name AS camera,
                c.lane AS station,COUNT(p.id) AS passages,
                SUM(CASE WHEN p.direction='in' THEN 1 ELSE 0 END) AS entries,
                COUNT(DISTINCT CASE WHEN p.direction='in'
                    THEN p.identity_id END) AS unique_entrants,
                SUM(CASE WHEN p.direction='in' AND p.identity_id IS NULL
                    THEN 1 ELSE 0 END) AS unresolved_entries
                FROM cameras c LEFT JOIN passages p ON p.camera_id=c.id
                WHERE c.event_id=? GROUP BY c.id ORDER BY c.name""",
                    (self.event_id,),
                )
            ]
        return result

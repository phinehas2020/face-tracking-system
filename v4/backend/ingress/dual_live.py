"""Two live views: side confirms passages; facing supplies shared face templates."""

import json
from dataclasses import replace
from pathlib import Path

from .live import CameraConfig, LiveSession
from .live_association import LiveAssociationCoordinator
from .store import EventStore


class DualLiveSession(LiveSession):
    def __init__(self, store: EventStore, config_path: Path, model_path: str):
        super().__init__(store, config_path, model_path)
        if not self.config.facing_camera:
            raise ValueError("Missing facing camera configuration")
        self.config = replace(self.config, camera_id="live-side", count_passages=True)
        self.camera_id = self.config.camera_id
        facing = CameraConfig(
            **{
                **self.config.facing_camera,
                "event_id": self.event_id,
                "camera_id": "live-facing",
                "count_passages": False,
                "activity_capture": self.config.activity_capture,
            }
        )
        self.facing = LiveSession(store, config_path, model_path, facing)
        # Registry observations are serialized, but each view retains its own tracks.
        self.facing.lock = self.lock
        self.association = self.facing.association = LiveAssociationCoordinator(
            similarity_threshold=self.config.association_threshold,
            margin_threshold=self.config.association_margin,
            min_samples=self.config.association_min_samples,
            max_track_gap_seconds=self.config.association_overlap_seconds,
            sole_candidate_threshold=self.config.association_sole_candidate_threshold,
            zone_mode=self.config.association_zone,
            zone_min_samples=self.config.association_zone_min_samples,
        )
        self.association_view = "side"
        self.facing.association_view = "facing"

    def start(self):
        super().start(start_workers=False)
        self.facing.start(start_workers=False)
        if self.activity and self.facing.frame_ring:
            self.activity.rings["facing"] = self.facing.frame_ring
        with self.lock:
            self.facing.face_registry.templates = self.face_registry.templates
            self.facing.face_registry.labels = self.face_registry.labels
            self.facing.face_registry.confirmed_variants = self.face_registry.confirmed_variants
            self.facing.face_previews = self.face_previews
            self.facing.face_preview_quality = self.face_preview_quality
            if self.association.zone_mode != "off":
                self._load_zone_samples()
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE cameras SET role='face_identity', "
                    "name='Facing · recognition' WHERE id=?",
                    (self.facing.camera_id,),
                )
                db.execute(
                    "UPDATE cameras SET name='Side · crossings' WHERE id=?", (self.camera_id,)
                )

        self._start_workers()
        self.facing._start_workers()

    def _load_zone_samples(self):
        """Refit the crossing zone from the most recent stored samples."""
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT json_extract(evidence,'$.cross_view_zone') AS zone FROM passages "
                "WHERE event_id=? AND json_extract(evidence,'$.cross_view_zone.sample')=1 "
                "ORDER BY observed_at DESC LIMIT 600",
                (self.event_id,),
            ).fetchall()
        for row in reversed(rows):
            record = json.loads(row["zone"])
            self.association.zone.add(record.get("direction", "any"), record.get("state"))

    def stop(self):
        self.facing.stop_event.set()
        super().stop()
        self.facing.stop()

    def snapshot(self):
        side = super().snapshot()
        facing = self.facing.snapshot()
        side["is_dual_live"] = True
        side["facing"] = facing
        side["face_samples_processed"] += facing["face_samples_processed"]
        side["face_status"] = (
            "error"
            if "error" in {side["face_status"], facing["face_status"]}
            else "ready"
            if side["face_status"] == facing["face_status"] == "ready"
            else "loading"
        )
        side["station_status"] = (
            "live" if side["status"] == facing["status"] == "live" else "degraded"
        )
        with self.lock:
            side["cross_camera_association"] = self.association.summary()
        return side

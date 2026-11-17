"""Two independently tracked views on one reconstructed recording timeline."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .face_registry import FaceRegistry
from .live import CrossingCounter, LiveCalibration, LiveSession
from .paired_appearance import AppearancePairMatcher
from .paired_geometry import PairMatcher, ViewObservation, validate_clock_index
from .paired_handoff import IdentityHandoff
from .paired_identity import PairedIdentityFusion
from .store import EventStore


class PairedView(BaseModel):
    role: str = Field(pattern="^(front|side)$")
    path: Path
    clock_index: list[dict]

    @model_validator(mode="after")
    def validate_index(self):
        validate_clock_index(self.clock_index)
        if not self.path.is_file():
            raise ValueError(f"Missing paired recording: {self.path.name}")
        return self


class PairedPlan(BaseModel):
    views: list[PairedView]
    start_clock: float
    duration_seconds: float = Field(gt=0, le=600)
    front_to_side: list[list[float]] | None = None
    synchronization_note: str
    calibration: LiveCalibration = Field(
        default_factory=lambda: LiveCalibration(
            axis="horizontal", position=0.88, require_body_motion=True
        )
    )
    max_skew_seconds: float = Field(default=0.8, gt=0, le=2)
    max_ground_distance: float = Field(default=0.075, gt=0, le=0.15)
    min_appearance: float = Field(default=0.65, ge=0, le=1)
    confirmations: int = Field(default=2, ge=2)
    appearance_matching: bool = False
    appearance_weights: Path | None = None
    appearance_cache: Path | None = None
    appearance_min_similarity: float = Field(default=0.55, ge=0, le=1)
    appearance_margin: float = Field(default=0.08, ge=0, le=1)
    observation_cache: Path | None = None
    playback_speed: float = Field(default=1, gt=0, le=32)
    fusion_enabled: bool = True
    minimum_foot_y: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_views(self):
        if sorted(v.role for v in self.views) != ["front", "side"]:
            raise ValueError("Exactly one front and one side view are required")
        if self.front_to_side is not None:
            PairMatcher(self.front_to_side)
        for v in self.views:
            if (
                v.clock_index[0]["clock"] > self.start_clock
                or v.clock_index[-1]["clock"] < self.start_clock + self.duration_seconds
            ):
                raise ValueError("Both clock indexes must cover the full paired excerpt")
        return self


class PairedSession(LiveSession):
    is_replay = True
    is_paired = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = PairedPlan.model_validate(self.config.paired_replay)
        self.event_id = f"paired-replay-{uuid.uuid4().hex[:10]}"
        self.camera_id = f"{self.event_id}-side"
        self.replay_complete = False
        self.source_info = {
            "label": "Station 2 · Front + side",
            "station": "station-02",
            "file": "Two synchronized camera views",
            "start_seconds": 0,
            "duration_seconds": self.plan.duration_seconds,
            "position_seconds": 0,
            "clip": 1,
            "total_clips": 1,
            "playback_speed": self.plan.playback_speed,
        }
        self.views = {}
        self.pair_links = []
        self.association_audit = []
        self.audited_pairs = set()
        self.link_evidence = {}
        self.identity_sources = {}
        self.handoff_evidence = {}
        self.identity_handoffs = []
        self.baseline_passages = []
        self.baseline_registry = None
        self.paired_frames = 0
        self.sync_skew = None
        self.calibration = self.plan.calibration
        self.counter = CrossingCounter(self.calibration)
        self.report_path = self.store.data_root / "events" / self.event_id / "paired-report.json"

    def _capture(self):
        # Both decoders are advanced by the shared-clock processing worker.
        return

    def _initialize_identity_views(self):
        # Controls live below the parent event directory and are purged with it.
        control = EventStore(self.report_path.parent / "camera-controls")
        control.initialize()
        self.local_registries = {}
        for role in ("front", "side"):
            with control.transaction() as db:
                now = datetime.now(UTC).isoformat()
                db.execute(
                    "INSERT INTO events (id,title,event_date,status,source_label,created_at) "
                    "VALUES (?,?,?,'live',?,?)",
                    (
                        role,
                        role + " independent control",
                        "2025-11-28",
                        "Identical cached or live observations",
                        now,
                    ),
                )
            self.local_registries[role] = FaceRegistry(control, role)
        self.baseline_registry = self.local_registries["side"]
        self.fusion = PairedIdentityFusion(self.face_registry)
        self.identity_sources = self.fusion.sources

    def _observe(
        self,
        role,
        frame,
        tracks,
        detections,
        continuity,
        clock,
        observed_at,
        registries=None,
        descriptors=None,
    ):
        import cv2

        from .vision.adapters import associate_faces

        registry = self.local_registries[role]
        associated = associate_faces(detections, tracks)
        descriptor_hints = {}
        if role == "front" and descriptors:
            for i, face in enumerate(detections):
                body_id = next((k for k, v in associated.items() if v is face), None)
                if body_id in descriptors:
                    descriptor_hints[i] = descriptors[body_id]
        face_ids = continuity.update(detections, clock, descriptor_hints)
        body_for = {id(face): f"{role}:{body}" for body, face in associated.items()}
        body_keys = {f"{role}:{t.track_id}" for t in tracks}
        face_keys = {f"{role}:{key}" for key in face_ids.values()}
        for index, detected in enumerate(detections):
            key = f"{role}:{face_ids[index]}"
            state = registry.observe(
                key, detected.embedding, detected.quality, face_keys, clock, observed_at
            )
            body = body_for.get(id(detected))
            if body:
                registry.observe(
                    body, detected.embedding, detected.quality, body_keys, clock, observed_at
                )
                registry.bind_face_to_body(body, state, body_keys, clock, observed_at)
            adopted = self.fusion.adopt(role, registry, state)
            if adopted and state.state in {"new", "matched"}:
                x1, y1, x2, y2 = map(int, detected.bbox)
                crop = frame[max(0, y1) : y2, max(0, x1) : x2]
                if crop.size:
                    ok, jpeg = cv2.imencode(".jpg", cv2.resize(crop, (100, 100)))
                    if ok:
                        self._save_face_preview(
                            adopted.identity_id, jpeg.tobytes(), detected.quality
                        )
            self.face_samples_processed += 1
            self.body_face_samples += int(body is not None)
        self.fusion.sync_local(role, registry, body_keys, clock, observed_at)
        return body_keys

    def _annotate_cross_view(self, key, evidence, observed_at):
        with self.store.transaction() as db:
            cutoff = (datetime.fromisoformat(observed_at) - timedelta(seconds=10)).isoformat(
                timespec="milliseconds"
            )
            db.execute(
                """UPDATE passages SET evidence=json_set(evidence,
                '$.identity_basis','paired_view','$.cross_view',json(?))
                WHERE event_id=? AND track_id=? AND identity_id=? AND observed_at>=?""",
                (json.dumps(evidence), self.event_id, key, evidence["identity_id"], cutoff),
            )

    def _annotate_handoff(self, key, evidence, observed_at):
        cutoff = (datetime.fromisoformat(observed_at) - timedelta(seconds=10)).isoformat(
            timespec="milliseconds"
        )
        with self.store.transaction() as db:
            db.execute(
                "UPDATE passages SET evidence=json_set(evidence,'$.identity_basis','body_handoff',"
                "'$.handoff',json(?)) WHERE event_id=? AND track_id=? "
                "AND identity_id=? AND observed_at>=?",
                (json.dumps(evidence), self.event_id, key, evidence["identity_id"], cutoff),
            )

    def _record(self, track_id, direction, observed_at=None):
        super()._record(track_id, direction, observed_at)
        if track_id in self.handoff_evidence:
            self._annotate_handoff(track_id, self.handoff_evidence[track_id], observed_at)
        proof = self.link_evidence.get(track_id)
        if proof:
            self._annotate_cross_view(track_id, proof, observed_at)

    def _process(self):
        import cv2

        from .vision.adapters import DetectedFace, LiveFaceEncoder, UltralyticsPersonTracker
        from .vision.continuity import FaceContinuity, TrackContinuity
        from .vision.contracts import FramePacket, Track

        captures = {}
        try:
            matcher = (
                PairMatcher(
                    self.plan.front_to_side,
                    max_distance=self.plan.max_ground_distance,
                    min_appearance=self.plan.min_appearance,
                    confirmations=self.plan.confirmations,
                    max_skew=self.plan.max_skew_seconds,
                )
                if self.plan.front_to_side is not None
                else None
            )
            if self.plan.appearance_matching:
                matcher = AppearancePairMatcher(
                    self.plan.appearance_min_similarity,
                    self.plan.appearance_margin,
                    self.plan.confirmations,
                    self.plan.max_skew_seconds,
                )
            appearance_vectors = {}
            appearance_encoder = None
            if self.plan.appearance_matching:
                if self.plan.appearance_cache:
                    with np.load(self.plan.appearance_cache, allow_pickle=False) as archive:
                        appearance_vectors = dict(
                            zip(archive["keys"], archive["features"], strict=True)
                        )
                elif self.plan.appearance_weights:
                    from .vision.reid import AppearanceEncoder

                    appearance_encoder = AppearanceEncoder(self.plan.appearance_weights)
                else:
                    raise ValueError("Appearance matching requires model weights or cached vectors")
            handoff = IdentityHandoff()
            cache = {}
            if self.plan.observation_cache:
                manifest_path = self.plan.observation_cache.with_name("manifest.json")
                manifest = json.loads(manifest_path.read_text())
                expected = [
                    {"role": v.role, "path": str(v.path), "bytes": v.path.stat().st_size}
                    for v in self.plan.views
                ]
                if manifest.get("views") != expected:
                    raise ValueError("Observation cache belongs to different recordings")
                if (
                    manifest.get("model_sha256")
                    != hashlib.sha256(Path(self.model_path).read_bytes()).hexdigest()
                ):
                    raise ValueError("Observation cache belongs to a different detector model")
                with gzip.open(self.plan.observation_cache, "rt") as source:
                    for line in source:
                        row = json.loads(line)
                        key = (row["role"], row["frame"])
                        if key in cache:
                            raise ValueError("Duplicate cached camera frame")
                        cache[key] = row
            trackers = (
                {r: UltralyticsPersonTracker(self.model_path) for r in ["front", "side"]}
                if not cache
                else {}
            )
            bodies = {
                r: TrackContinuity(max_gap=3 if r == "front" else 1.2) for r in ["front", "side"]
            }
            faces = {
                r: FaceContinuity(
                    max_gap=3 if r == "front" else 1.5, max_motion_widths=3 if r == "front" else 0
                )
                for r in bodies
            }
            encoder = LiveFaceEncoder() if not cache else None
            self._initialize_identity_views()
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE events SET title='Station 2 paired rehearsal',"
                    "source_label='Clock-aligned front and side recordings' WHERE id=?",
                    (self.event_id,),
                )
            rows = {
                v.role: [
                    r
                    for r in v.clock_index
                    if self.plan.start_clock
                    <= r["clock"]
                    <= self.plan.start_clock + self.plan.duration_seconds
                ]
                for v in self.plan.views
            }
            events = sorted(
                (row["clock"], role, row["frame"]) for role, items in rows.items() for row in items
            )
            current = {}
            last_scan = {"side": -1e9, "front": -1e9}
            with self.lock:
                self.face_status = "ready"
                self.calibration = self.plan.calibration
                self.counter = CrossingCounter(self.calibration)
                self.frames_expected = len(events)
            for v in self.plan.views:
                captures[v.role] = cv2.VideoCapture(str(v.path))
                if not captures[v.role].isOpened():
                    raise RuntimeError(f"Cannot open {v.role} recording")
            start_wall = time.monotonic()
            for clock, role, idx in events:
                while self.paused and not self.stop_event.wait(0.05):
                    start_wall = (
                        time.monotonic()
                        - (clock - self.plan.start_clock) / self.plan.playback_speed
                    )
                if self.stop_event.is_set():
                    break
                cap = captures[role]
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {role} frame {idx}")
                observed_at = (
                    datetime(2025, 11, 28, tzinfo=UTC) + timedelta(seconds=clock)
                ).isoformat(timespec="milliseconds")
                cached = cache.get((role, idx)) if cache else None
                if cache:
                    if cached is None or abs(cached["clock"] - clock) > 1e-6:
                        raise ValueError("Observation cache does not match this frame clock")
                    tracks = [
                        Track(t["id"], tuple(t["bbox"]), t["confidence"], t["age"])
                        for t in cached["tracks"]
                    ]
                else:
                    tracks = trackers[role].track(
                        FramePacket(role, idx, time.monotonic_ns(), observed_at, frame)
                    )
                    tracks = bodies[role].update(tracks, frame, clock)
                descriptors = {}
                if self.plan.appearance_matching:
                    if appearance_vectors:
                        for t in tracks:
                            cache_key = f"{role}:{idx}:{t.track_id}"
                            if cache_key not in appearance_vectors:
                                raise ValueError("Missing appearance observation")
                            descriptors[t.track_id] = appearance_vectors[cache_key]
                    else:
                        descriptors = appearance_encoder.encode(frame, tracks)
                observations = [
                    ViewObservation(
                        f"{role}:{t.track_id}",
                        (t.footpoint[0] / frame.shape[1], t.footpoint[1] / frame.shape[0]),
                        descriptors[t.track_id]
                        if self.plan.appearance_matching
                        else bodies[role].appearance(frame, t.bbox),
                        idx,
                        clock,
                        t.bbox[3] < frame.shape[0] * 0.985,
                    )
                    for t in tracks
                ]
                scan = (
                    cached["scan"]
                    if cached
                    else (role == "front" or clock - last_scan[role] >= 0.2 - 1e-8)
                )
                detections = (
                    [
                        DetectedFace(
                            tuple(f["bbox"]),
                            np.asarray(f["embedding"], dtype=np.float32),
                            f["quality"],
                        )
                        for f in cached["faces"]
                    ]
                    if cached
                    else encoder.detect(frame)
                    if scan
                    else []
                )
                with self.lock:
                    active = {f"{role}:{t.track_id}" for t in tracks}
                    if scan:
                        registries = None
                        active = self._observe(
                            role,
                            frame,
                            tracks,
                            detections,
                            faces[role],
                            clock,
                            observed_at,
                            registries,
                            descriptors,
                        )
                        last_scan[role] = clock
                        self.face_scans += 1
                        self.face_detections += len(detections)
                    current[role] = {
                        "frame": frame,
                        "tracks": tracks,
                        "observations": observations,
                        "clock": clock,
                        "index": idx,
                        "active": active,
                    }
                    all_active = set().union(*(v["active"] for v in current.values()))
                    self.face_registry.expire(all_active, clock)
                    local = self.local_registries[role]
                    local.expire(active, clock)
                    self.fusion.sync_local(role, local, active, clock, observed_at)
                    if role == "side" and self.plan.appearance_matching:
                        for item in handoff.update(
                            observations, self.face_registry, clock, observed_at
                        ):
                            self.identity_handoffs.append(item)
                            self.handoff_evidence[item["to"]] = item
                            self._annotate_handoff(item["to"], item, observed_at)
                    matches = (
                        matcher.update(
                            current.get("front", {}).get("observations", []),
                            current.get("side", {}).get("observations", []),
                        )
                        if matcher
                        else []
                    )
                    self.pair_links = []
                    for proposal in matches:
                        audit_key = (proposal["front_frame"], proposal["front"], proposal["side"])
                        if audit_key not in self.audited_pairs:
                            self.audited_pairs.add(audit_key)
                            front_debug = self.face_registry.tracks.get(proposal["front"])
                            self.association_audit.append(
                                {
                                    **proposal,
                                    "clock": clock,
                                    "face_state": front_debug.state if front_debug else None,
                                    "face_identity": front_debug.label
                                    if front_debug and front_debug.identity_id
                                    else None,
                                    "face_age": clock - front_debug.last_face
                                    if front_debug
                                    else None,
                                }
                            )
                    if self.plan.fusion_enabled and "side" in current:
                        for match in matches:
                            front_state = self.face_registry.tracks.get(match["front"])
                            side_state = self.face_registry.tracks.get(match["side"])
                            if (
                                not front_state
                                or not front_state.identity_id
                                or front_state.state not in {"new", "matched"}
                            ):
                                continue
                            if (
                                clock - front_state.last_face > 1.5
                                or abs(front_state.last_face - current["front"]["clock"]) > 0.1
                            ):
                                continue
                            if side_state and side_state.identity_id:
                                self.fusion.merge_confirmed_pair(front_state, side_state, match)
                            # Unresolved opposing evidence stays a disagreement.
                            disagreement = bool(
                                side_state
                                and side_state.identity_id
                                and side_state.identity_id != front_state.identity_id
                            )
                            self.pair_links.append(
                                {**match, "label": front_state.label, "disagreement": disagreement}
                            )
                            if disagreement:
                                continue
                            before = side_state.identity_id if side_state else None
                            self.face_registry.bind_face_to_body(
                                match["side"],
                                front_state,
                                current["side"]["active"],
                                clock,
                                observed_at,
                            )
                            after = self.face_registry.tracks.get(match["side"])
                            if (
                                after
                                and after.identity_id == front_state.identity_id
                                and (not before or match["side"] in self.link_evidence)
                            ):
                                proof = {
                                    **match,
                                    "identity_id": after.identity_id,
                                    "label": after.label,
                                    "clock": clock,
                                }
                                self.link_evidence[match["side"]] = proof
                                self._annotate_cross_view(match["side"], proof, observed_at)
                    self.source_info["position_seconds"] = round(clock - self.plan.start_clock, 3)
                    if role == "side":
                        for t in tracks:
                            if t.footpoint[1] / frame.shape[0] < self.plan.minimum_foot_y:
                                continue
                            key = f"side:{t.track_id}"
                            direction = self.counter.update(
                                key,
                                (t.footpoint[0] / frame.shape[1], t.footpoint[1] / frame.shape[0]),
                                clock,
                                body_anchor=(
                                    (t.bbox[0] + t.bbox[2]) / 2 / frame.shape[1],
                                    t.bbox[1] / frame.shape[0],
                                ),
                            )
                            if direction:
                                self._record(key, direction, observed_at)
                                state = self.baseline_registry.tracks.get(key)
                                self.baseline_passages.append(
                                    {
                                        "key": key,
                                        "direction": direction,
                                        "clock": clock,
                                        "at": observed_at,
                                        "identity": state.identity_id
                                        if state and state.state != "conflict"
                                        else None,
                                    }
                                )
                        self.counter.expire(clock)
                    for p in self.baseline_passages:
                        state = self.baseline_registry.tracks.get(p["key"])
                        if (
                            not p["identity"]
                            and clock - p["clock"] <= 10
                            and state
                            and state.identity_id
                            and state.state != "conflict"
                            and (not state.resolve_after or p["at"] >= state.resolve_after)
                        ):
                            p["identity"] = state.identity_id
                    self.views = {
                        r: {
                            "frame": v["index"],
                            "clock": v["clock"],
                            "age_seconds": round(clock - v["clock"], 3),
                            "people": len(v["tracks"]),
                        }
                        for r, v in current.items()
                    }
                    self.sync_skew = (
                        round(abs(current["front"]["clock"] - current["side"]["clock"]), 3)
                        if len(current) == 2
                        else None
                    )
                    tiles = []
                    for r in ["front", "side"]:
                        if r not in current:
                            tiles.append(np.zeros((580, 960, 3), dtype=np.uint8))
                            continue
                        v = current[r]
                        im = v["frame"].copy()
                        h, w = im.shape[:2]
                        for t in v["tracks"]:
                            key = f"{r}:{t.track_id}"
                            state = self.face_registry.tracks.get(key)
                            color = (20, 160, 220) if key in self.link_evidence else (160, 170, 150)
                            x1, y1, x2, y2 = map(int, t.bbox)
                            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                            label = (
                                state.label
                                if state and state.identity_id and state.state != "conflict"
                                else key
                            )
                            cv2.putText(
                                im,
                                label,
                                (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                color,
                                2,
                            )
                        if r == "side":
                            pos = int(
                                self.calibration.position
                                * (w if self.calibration.axis == "vertical" else h)
                            )
                            cv2.line(
                                im,
                                (pos, 0) if self.calibration.axis == "vertical" else (0, pos),
                                (pos, h) if self.calibration.axis == "vertical" else (w, pos),
                                (40, 140, 240),
                                3,
                            )
                        tile = cv2.copyMakeBorder(
                            cv2.resize(im, (960, 540)),
                            40,
                            0,
                            0,
                            0,
                            cv2.BORDER_CONSTANT,
                            value=(32, 40, 43),
                        )
                        label = (
                            f"{r.upper()} | frame {v['index']} | "
                            f"common clock {v['clock']:.1f}s | "
                            f"age {clock - v['clock']:.1f}s"
                        )
                        cv2.putText(
                            tile, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 230), 1
                        )
                        tiles.append(tile)
                    composite = np.hstack(tiles)
                    ok, jpeg = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        self.jpeg = jpeg.tobytes()
                    self.processed_time = time.monotonic()
                    self.frames_processed += 1
                    self.frame_sequence += 1
                    self.size = [1920, 580]
                    self.people_visible = len(current.get("side", {}).get("tracks", []))
                    self.peak_people_visible = max(self.peak_people_visible, self.people_visible)
                    self.fps = round(
                        self.frames_processed / max(time.monotonic() - start_wall, 0.1), 1
                    )
                    self.status = "live"
                    self.paired_frames += int(len(current) == 2)
                self.stop_event.wait(
                    max(
                        0,
                        (clock - self.plan.start_clock) / self.plan.playback_speed
                        - (time.monotonic() - start_wall),
                    )
                )
            if not self.stop_event.is_set():
                with self.lock:
                    self.replay_complete = True
                    self.status = "complete"
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                self.report_path.write_text(json.dumps(self.snapshot(), indent=2))
                self.report_path.with_name("association-audit.json").write_text(
                    json.dumps(self.association_audit, indent=2)
                )
        except Exception as exc:
            with self.lock:
                self.error = f"Paired replay stopped: {type(exc).__name__}: {exc}"
                self.status = "error"
                self.face_status = "error"
        finally:
            for cap in captures.values():
                cap.release()

    def snapshot(self):
        d = super().snapshot()
        entries = [p for p in self.baseline_passages if p["direction"] == "in"]
        with self.store.connect() as db:
            linked = db.execute(
                "SELECT track_id,observed_at,evidence FROM passages "
                "WHERE event_id=? AND direction='in' AND identity_id IS NOT NULL",
                (self.event_id,),
            ).fetchall()
        baseline_links = {(p["key"], p["at"]) for p in entries if p["identity"]}
        assisted = sum(
            (row["track_id"], row["observed_at"]) not in baseline_links for row in linked
        )
        d.update(
            is_replay=True,
            is_paired=True,
            source=dict(self.source_info),
            views=self.views,
            pair_links=self.pair_links,
            sync_skew_seconds=self.sync_skew,
            synchronization_note=self.plan.synchronization_note,
            spatial_fusion_enabled=self.plan.front_to_side is not None,
            appearance_fusion_enabled=self.plan.appearance_matching,
            association_method="Repeated OSNet appearance + timestamp agreement"
            if self.plan.appearance_matching
            else "Direct face sharing",
            cached_observations=bool(self.plan.observation_cache),
            baseline_passages=list(self.baseline_passages),
            identity_handoffs=list(self.identity_handoffs),
            cross_view_evidence=list(self.link_evidence.values()),
            cross_view_identity_merges=list(self.fusion.merges) if hasattr(self, "fusion") else [],
            identity_sources={
                self.face_registry.labels.get(k, k): sorted(v)
                for k, v in self.identity_sources.items()
            },
            frames_expected=getattr(self, "frames_expected", 0),
            comparison={
                "side_only_linked_entries": sum(bool(p["identity"]) for p in entries),
                "side_only_entries": len(entries),
                "additional_linked_entries": assisted,
                "cross_view_assisted_entries": sum(
                    (r["track_id"], r["observed_at"]) not in baseline_links
                    and json.loads(r["evidence"]).get("identity_basis") == "paired_view"
                    for r in linked
                ),
                "handoff_assisted_entries": sum(
                    (r["track_id"], r["observed_at"]) not in baseline_links
                    and json.loads(r["evidence"]).get("identity_basis") == "body_handoff"
                    for r in linked
                ),
                "paired_linked_entries": len(linked),
                "net_link_gain": len(linked) - len(baseline_links),
            },
        )
        if self.replay_complete:
            d["status"] = "complete"
        elif self.paused:
            d["status"] = "paused"
        return d

    def calibrate(self, calibration):
        if not self.replay_complete:
            raise ValueError("Change paired calibration between runs")
        payload = json.loads(self.config_path.read_text())
        payload["paired_replay"]["calibration"] = calibration.model_dump()
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(self.config_path)
        super().calibrate(calibration)

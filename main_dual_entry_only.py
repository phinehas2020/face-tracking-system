#!/usr/bin/env python3
"""
Dual Entry-Only Face Recognition People Counter
================================================

This variant uses two independent ENTRY cameras (for two separate doors).
Every recognized face is counted as an entry event; exits are intentionally
ignored so you can total everyone coming through either doorway without
requiring matching exit detections.

Usage:
------
1. Run with two entry cameras:
   python main_dual_entry_only.py --entry-cam 0 --exit-cam 1

2. Run with single camera simulation (split screen):
   python main_dual_entry_only.py --entry-cam 0 --exit-cam 0 --split-screen

3. View statistics:
   curl http://localhost:8000/stats

Author: Face Tracking System
Date: 2025
"""

import os
import sys
import time
import sqlite3
import pickle
import logging
import math
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
from threading import Thread, Lock
import signal

import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_tracking.db"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

FACE_MATCH_THRESHOLD = 0.3  # Primary cosine similarity threshold
FALLBACK_MATCH_THRESHOLD = 0.22  # Allow looser match for recently seen people
MIN_RECENT_MATCH = 0.17  # Aggressive reuse for people seen seconds ago
RECENT_PERSON_WINDOW = 120  # Seconds to consider someone "recent"
PERSON_MERGE_THRESHOLD = 0.45  # Merge temp people if similarity above this

MIN_FACE_SIZE = 90  # Minimum face size in pixels
FACE_COOLDOWN_TIME = 10  # Seconds before same person can be counted again
EMBEDDING_EXPIRE_TIME = 300  # Seconds before temporary embeddings expire
EMBEDDING_REFRESH_INTERVAL = 45  # Seconds before we store another embedding sample
MAX_EMBEDDINGS_PER_PERSON = 5  # Keep up to N embeddings per person in memory
THUMBNAIL_SIZE = (200, 200)
ENTRY_POSITION_SUPPRESSION_RADIUS = 70  # Pixels within which repeat detections are ignored
ENTRY_POSITION_SUPPRESSION_WINDOW = 2.0  # Seconds to suppress repeat hits at same spot
ENTRY_RECENT_SIM_THRESHOLD = 0.22  # Cosine similarity to reuse fresh temp IDs
ENTRY_RECENT_WINDOW = 3.0  # Seconds to retain very recent entry candidates
MAX_FACE_YAW_DEG = 25  # Reject faces turned too far sideways
MAX_FACE_PITCH_DEG = 20  # Reject faces tilted up/down too far

# Global variables
app = FastAPI(title="Dual Entry People Counter API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
stats_lock = Lock()
global_stats = {
    "unique_visitors": 0,
    "total_in": 0,
    "total_out": 0,
    "avg_dwell_minutes": 0.0,
    "current_occupancy": 0
}

COUNTER_INSTANCE = None

class StatsResponse(BaseModel):
    unique_visitors: int
    avg_dwell_minutes: float
    total_in: int
    total_out: int
    current_occupancy: int
    timestamp: str


CameraSource = Union[int, str]


def normalize_camera_source(value: CameraSource) -> CameraSource:
    """Normalize CLI/API camera inputs into either int indexes or string URLs."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Camera indexes must be >= 0")
        return value

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Camera source cannot be empty")
        # Try to coerce numeric strings into ints for backward compatibility
        try:
            num = int(cleaned, 10)
            if num < 0:
                raise ValueError("Camera indexes must be >= 0")
            return num
        except ValueError:
            return cleaned

    raise TypeError(f"Unsupported camera source type: {type(value)}")


class CameraConfigRequest(BaseModel):
    entry_cam: CameraSource
    exit_cam: CameraSource
    split_screen: bool = False


class AsyncVideoCapture:
    """Background reader that always exposes the freshest frame."""

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ret = False
        self.last_frame_time = 0.0
        self.stopped = False
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                # brief sleep prevents tight loop when camera hiccups
                time.sleep(0.01)
                continue
            timestamp = time.time()
            with self.lock:
                self.latest_ret = True
                self.latest_frame = frame
                self.last_frame_time = timestamp

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        with self.lock:
            if not self.latest_ret or self.latest_frame is None:
                return False, None, None
            frame = self.latest_frame.copy()
            timestamp = self.last_frame_time
        age = time.time() - timestamp if timestamp else None
        return True, frame, age

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

class DualEntryOnlyCounter:
    def __init__(self, entry_cam_id: CameraSource = 0, exit_cam_id: CameraSource = 1, split_screen=False):
        """Initialize the dual camera people counter system"""
        self.entry_cam_id = normalize_camera_source(entry_cam_id)
        self.exit_cam_id = normalize_camera_source(exit_cam_id)
        self.split_screen = split_screen  # For single camera simulation

        THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

        self.yolo_model = None
        self.face_app = None
        self.db_conn = None

        # Face tracking
        self.known_embeddings = defaultdict(lambda: deque(maxlen=MAX_EMBEDDINGS_PER_PERSON))
        self.last_embedding_update = defaultdict(lambda: 0.0)
        self.temp_embeddings = {}  # temp_id -> (embedding, first_seen_time) (temporary)
        self.person_thumbnails = {}
        self.person_last_seen = {}  # person_id -> {entry: time}
        self.camera_lock = Lock()
        self.camera_reload_requested = False
        self._last_entry_latency_log = 0.0
        self._last_exit_latency_log = 0.0
        self.recent_entry_positions = deque()  # (x, y, timestamp)
        self.recent_entry_candidates = deque(maxlen=50)  # (person_id, embedding, timestamp)

        # Frame counters
        self.entry_frame_count = 0
        self.exit_frame_count = 0
        
        # Initialize components
        self._init_yolo()
        self._init_face_analyzer()
        self._init_database()
        self._load_known_faces()
    
    def _init_yolo(self):
        """Initialize YOLO model for detection"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
    
    def _init_face_analyzer(self):
        """Initialize InsightFace analyzer"""
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Face analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face analyzer: {e}")
            raise
    
    def _init_database(self):
        """Initialize database connection"""
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()
        logger.info(f"Connected to database: {DB_PATH}")

    def _ensure_schema(self):
        """Ensure new columns exist for older databases"""
        cursor = self.db_conn.cursor()
        cursor.execute("PRAGMA table_info(persons)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'thumbnail_path' not in columns:
            cursor.execute("ALTER TABLE persons ADD COLUMN thumbnail_path TEXT")
            self.db_conn.commit()
    
    def _load_known_faces(self):
        """Load known face embeddings from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT p.person_id, p.name, p.thumbnail_path, f.embedding
            FROM persons p
            JOIN faces f ON p.person_id = f.person_id
            ORDER BY f.created_ts ASC
        """)
        rows = cursor.fetchall()

        for person_id, name, thumbnail_path, embedding_blob in rows:
            embedding = pickle.loads(embedding_blob)
            self.known_embeddings[person_id].append(embedding)
            if thumbnail_path and person_id not in self.person_thumbnails:
                self.person_thumbnails[person_id] = thumbnail_path
            logger.debug(f"Loaded embedding for {name} (ID: {person_id})")
        
        logger.info(f"Loaded embeddings for {len(self.known_embeddings)} known people")

    def list_available_cameras(self, max_devices: int = 6) -> List[Dict[str, int]]:
        """Probe a handful of device indices to see which cameras respond"""
        cameras = []
        for device_id in range(max_devices):
            cap = cv2.VideoCapture(device_id)
            available = bool(cap is not None and cap.isOpened())
            if cap is not None and cap.isOpened():
                cap.release()
            cameras.append({
                "id": device_id,
                "available": available
            })
        return cameras

    def save_thumbnail(self, person_id: str, face_image: Optional[np.ndarray]) -> Optional[str]:
        """Save a thumbnail for the person and return the file path"""
        if face_image is None or face_image.size == 0:
            return None

        try:
            thumb = cv2.resize(face_image, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
        except Exception:
            thumb = face_image

        thumb_path = THUMBNAIL_DIR / f"{person_id}.jpg"
        try:
            cv2.imwrite(str(thumb_path), thumb)
            resolved = str(thumb_path.resolve())
            self.person_thumbnails[person_id] = resolved
            return resolved
        except Exception as exc:
            logger.warning(f"Failed to save thumbnail for {person_id}: {exc}")
            return None

    def add_embedding(self, person_id: str, embedding: Optional[np.ndarray], persist: bool = False):
        """Store embedding in memory and optionally persist to DB"""
        if embedding is None:
            return

        embedding = embedding / np.linalg.norm(embedding)
        self.known_embeddings[person_id].append(embedding)

        if persist:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO faces (person_id, embedding, created_ts)
                VALUES (?, ?, ?)
            """, (person_id, pickle.dumps(embedding), datetime.now().isoformat()))
            self.db_conn.commit()

    def update_person_thumbnail(self, person_id: str, thumbnail_path: Optional[str]):
        """Persist thumbnail path for a person"""
        if not thumbnail_path:
            return

        cursor = self.db_conn.cursor()
        cursor.execute(
            "UPDATE persons SET thumbnail_path = ? WHERE person_id = ?",
            (thumbnail_path, person_id)
        )
        self.db_conn.commit()

    def maybe_refresh_embedding(self, person_id: str, embedding: Optional[np.ndarray]):
        """Occasionally capture additional embeddings for robustness"""
        if embedding is None:
            return

        now = time.time()
        last_update = self.last_embedding_update[person_id]
        if now - last_update < EMBEDDING_REFRESH_INTERVAL:
            return

        self.last_embedding_update[person_id] = now
        self.add_embedding(person_id, embedding, persist=False)

    def update_camera_config(self, entry_cam: CameraSource, exit_cam: CameraSource, split_screen: bool = False) -> Tuple[bool, str]:
        """Request a camera configuration change"""
        try:
            entry_src = normalize_camera_source(entry_cam)
            exit_src = normalize_camera_source(exit_cam)
        except (ValueError, TypeError) as exc:
            return False, str(exc)

        # If the same camera is selected, force split mode
        if entry_src == exit_src:
            split_screen = True

        with self.camera_lock:
            self.entry_cam_id = entry_src
            self.exit_cam_id = exit_src
            self.split_screen = split_screen
            self.camera_reload_requested = True
        logger.info(
            "Camera reconfiguration requested -> door A: %s door B: %s split: %s",
            entry_src, exit_src, split_screen
        )
        return True, "Camera configuration updated"

    def _create_capture(self, source: CameraSource) -> cv2.VideoCapture:
        """Create a VideoCapture for either a device index or a URL."""
        backend = cv2.CAP_FFMPEG if isinstance(source, str) else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if (cap is None or not cap.isOpened()) and backend != cv2.CAP_ANY:
            # Fall back to default backend if FFMPEG is unavailable
            cap = cv2.VideoCapture(source)
        if not cap or not cap.isOpened():
            raise RuntimeError(f"Failed to open source {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _open_cameras(self) -> Tuple[Optional[cv2.VideoCapture], Optional[cv2.VideoCapture]]:
        """Open camera streams based on the current configuration"""
        cap_entry = AsyncVideoCapture(self._create_capture(self.entry_cam_id))

        if self.split_screen:
            cap_exit = None
        else:
            try:
                exit_capture = self._create_capture(self.exit_cam_id)
                cap_exit = AsyncVideoCapture(exit_capture)
            except Exception:
                cap_entry.stop()
                raise

        logger.info(
            "Entry-only cameras active -> door A: %s door B: %s split: %s",
            self.entry_cam_id,
            self.exit_cam_id if not self.split_screen else self.entry_cam_id,
            self.split_screen
        )
        return cap_entry, cap_exit

    def _release_cameras(self, cap_entry, cap_exit):
        if cap_entry is not None:
            cap_entry.stop()
        if cap_exit is not None and cap_exit is not cap_entry:
            cap_exit.stop()

    def _combine_frames(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """Safely stack two frames horizontally, normalizing their heights if needed."""
        if frame_a is None:
            return frame_b
        if frame_b is None:
            return frame_a

        h1, w1 = frame_a.shape[:2]
        h2, w2 = frame_b.shape[:2]

        if h1 != h2:
            target = min(h1, h2)
            new_w1 = max(1, int(round(w1 * target / h1)))
            new_w2 = max(1, int(round(w2 * target / h2)))
            frame_a = cv2.resize(frame_a, (new_w1, target), interpolation=cv2.INTER_AREA)
            frame_b = cv2.resize(frame_b, (new_w2, target), interpolation=cv2.INTER_AREA)

        return np.hstack([frame_a, frame_b])

    def _entry_should_suppress(self, center: Tuple[float, float]) -> bool:
        """Prevent rapid-fire duplicates from the same spot near the door."""
        now = time.time()
        while self.recent_entry_positions and now - self.recent_entry_positions[0][2] > ENTRY_POSITION_SUPPRESSION_WINDOW:
            self.recent_entry_positions.popleft()

        for px, py, ts in self.recent_entry_positions:
            if math.hypot(center[0] - px, center[1] - py) <= ENTRY_POSITION_SUPPRESSION_RADIUS:
                return True
        return False

    def _remember_entry_position(self, center: Tuple[float, float]):
        self.recent_entry_positions.append((center[0], center[1], time.time()))

    def _match_recent_entry_candidate(self, embedding: np.ndarray) -> Optional[str]:
        """If a brand-new embedding almost matches a person seen seconds ago, reuse that ID."""
        if embedding is None:
            return None
        now = time.time()
        match_id = None
        refreshed = deque(maxlen=self.recent_entry_candidates.maxlen)
        for person_id, stored_emb, ts in list(self.recent_entry_candidates):
            if now - ts > ENTRY_RECENT_WINDOW:
                continue
            sim = float(np.dot(embedding, stored_emb))
            if sim >= ENTRY_RECENT_SIM_THRESHOLD and match_id is None:
                match_id = person_id
            refreshed.append((person_id, stored_emb, ts))
        self.recent_entry_candidates = refreshed
        return match_id

    def _remember_recent_entry_candidate(self, person_id: str, embedding: Optional[np.ndarray]):
        if embedding is None:
            return
        embedding = embedding / np.linalg.norm(embedding)
        self.recent_entry_candidates.append((person_id, embedding, time.time()))

    def _reload_cameras(self, cap_entry, cap_exit):
        try:
            new_entry, new_exit = self._open_cameras()
        except Exception as exc:
            logger.error(f"Unable to reconfigure cameras: {exc}")
            with self.camera_lock:
                self.camera_reload_requested = False
            return cap_entry, cap_exit

        self._release_cameras(cap_entry, cap_exit)
        with self.camera_lock:
            self.camera_reload_requested = False
        return new_entry, new_exit
    
    def extract_face_embedding(self, image, bbox):
        """Extract face embedding and cropped face image from a detected person"""
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Add padding around detected person to help face detector
            h, w = image.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None, None

            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(person_rgb)
            if len(faces) == 0:
                return None, None

            # Use the largest detected face
            largest_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            if getattr(largest_face, "det_score", 1.0) < 0.5:
                logger.debug(f"Rejected face due to low detection score: {largest_face.det_score:.2f}")
                return None, None

            pose = getattr(largest_face, "pose", None)
            if pose is not None:
                yaw, pitch, roll = map(abs, pose)
                if yaw > MAX_FACE_YAW_DEG or pitch > MAX_FACE_PITCH_DEG:
                    logger.debug(
                        "Rejected face due to pose (yaw %.1f°, pitch %.1f°)", yaw, pitch
                    )
                    return None, None

            face_width = largest_face.bbox[2] - largest_face.bbox[0]
            face_height = largest_face.bbox[3] - largest_face.bbox[1]
            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                return None, None

            embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)

            fx1, fy1, fx2, fy2 = map(int, largest_face.bbox)
            fx1 = max(0, min(fx1, person_crop.shape[1]))
            fx2 = max(0, min(fx2, person_crop.shape[1]))
            fy1 = max(0, min(fy1, person_crop.shape[0]))
            fy2 = max(0, min(fy2, person_crop.shape[0]))

            face_crop = person_crop[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                face_crop = person_crop.copy()
            else:
                face_crop = face_crop.copy()

            return embedding, face_crop

        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
            return None, None
    
    def match_face(self, embedding):
        """Match face embedding against all known faces (permanent and temporary)"""
        best_match = None
        best_similarity = -1.0
        match_is_temp = False

        # Check permanent embeddings (multiple samples per person)
        for person_id, embeddings in self.known_embeddings.items():
            for known_embedding in embeddings:
                similarity = float(np.dot(embedding, known_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id
                    match_is_temp = False

        # Check temporary embeddings
        current_time = time.time()
        expired_temps = []
        for temp_id, (temp_embedding, first_seen) in self.temp_embeddings.items():
            if current_time - first_seen > EMBEDDING_EXPIRE_TIME:
                expired_temps.append(temp_id)
                continue

            similarity = float(np.dot(embedding, temp_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = temp_id
                match_is_temp = True

        # Clean up expired temporary embeddings
        for temp_id in expired_temps:
            self.temp_embeddings.pop(temp_id, None)
            self.person_last_seen.pop(temp_id, None)

        if best_match and best_similarity >= FACE_MATCH_THRESHOLD:
            logger.debug(
                f"Face matched to {best_match} (temp={match_is_temp}) with similarity {best_similarity:.3f}"
            )
            return best_match

        # Fallback for recently seen persons to reduce duplicate IDs
        if best_match and best_similarity >= FALLBACK_MATCH_THRESHOLD:
            last_seen = self.person_last_seen.get(best_match, {})
            last_activity = max(last_seen.values(), default=0)
            if last_activity and current_time - last_activity <= RECENT_PERSON_WINDOW:
                logger.debug(
                    f"Fallback match to {best_match} with similarity {best_similarity:.3f}"
                )
                return best_match

        # Aggressive fallback: if we saw this face within cooldown window, reuse ID
        if best_match and best_similarity >= MIN_RECENT_MATCH:
            last_seen = self.person_last_seen.get(best_match, {})
            last_activity = max(last_seen.values(), default=0)
            if last_activity and current_time - last_activity <= FACE_COOLDOWN_TIME:
                logger.debug(
                    f"Recent-match override for {best_match} with similarity {best_similarity:.3f}"
                )
                return best_match

        return None
    
    def create_temporary_person(self, embedding):
        """Create a temporary person ID for unrecognized faces"""
        temp_id = f"temp_{int(time.time()*1000)}"
        self.temp_embeddings[temp_id] = (embedding, time.time())
        logger.info(f"Created temporary person: {temp_id}")
        return temp_id
    
    def process_entry_detection(self, embedding, face_image=None):
        """Process a face detection from the entry camera"""
        if embedding is None:
            return None

        # Try to match the face
        person_id = self.match_face(embedding)
        if person_id:
            logger.info(f"ENTRY match -> reusing existing ID {person_id}")
        
        # If no match, create temporary person
        if person_id is None:
            recent_id = self._match_recent_entry_candidate(embedding)
            if recent_id:
                logger.info(f"ENTRY recent cache -> reusing temp ID {recent_id}")
                person_id = recent_id
            else:
                person_id = self.create_temporary_person(embedding)
                logger.info(f"ENTRY created new temp ID {person_id}")
        
        # Check cooldown
        current_time = time.time()
        if person_id in self.person_last_seen:
            last_entry = self.person_last_seen[person_id].get('entry', 0)
            if current_time - last_entry < FACE_COOLDOWN_TIME:
                return None  # Too soon, skip

        # Record entry
        final_id = person_id
        history = self.person_last_seen.setdefault(person_id, {})

        if person_id.startswith('temp_'):
            merged_id = self.create_permanent_person(person_id, embedding, face_image)
            if merged_id != person_id:
                final_id = merged_id
                merged_history = self.person_last_seen.setdefault(final_id, {})
                merged_history.update(history)
                history = merged_history
                if person_id in self.person_last_seen:
                    del self.person_last_seen[person_id]
        
        history['entry'] = current_time

        if not final_id.startswith('temp_'):
            if final_id not in self.person_thumbnails and face_image is not None:
                thumbnail_path = self.save_thumbnail(final_id, face_image)
                self.update_person_thumbnail(final_id, thumbnail_path)
            self.log_crossing(final_id, 'in', current_time)
            self.maybe_refresh_embedding(final_id, embedding)
        else:
            self.log_crossing(final_id, 'in', current_time)

        # Update stats
        self.update_stats()

        self._remember_recent_entry_candidate(final_id, embedding)
        logger.info(
            "ENTRY (door camera): %s entered at %s",
            final_id,
            datetime.now().strftime('%H:%M:%S')
        )
        return final_id
    
    def create_permanent_person(self, temp_id, embedding, face_image=None):
        """Convert temporary person to permanent in database"""
        cursor = self.db_conn.cursor()

        name = f"Person {temp_id[-6:]}"
        consent_ts = datetime.now().isoformat()
        thumbnail_path = self.save_thumbnail(temp_id, face_image)

        cursor.execute("""
            INSERT OR IGNORE INTO persons (person_id, name, consent_ts, thumbnail_path)
            VALUES (?, ?, ?, ?)
        """, (temp_id, name, consent_ts, thumbnail_path))

        if thumbnail_path:
            self.person_thumbnails[temp_id] = thumbnail_path

        self.db_conn.commit()

        # Store embedding in DB & memory
        self.add_embedding(temp_id, embedding, persist=True)
        self.last_embedding_update[temp_id] = time.time()

        # Remove temp record if present
        self.temp_embeddings.pop(temp_id, None)
        
        return self.merge_if_duplicate(temp_id)

    def merge_if_duplicate(self, source_id: str) -> str:
        """Merge source person into an existing one if embeddings are similar"""
        source_embeddings = list(self.known_embeddings.get(source_id, []))
        if not source_embeddings:
            return source_id

        best_target = None
        best_similarity = -1.0

        for target_id, embeddings in self.known_embeddings.items():
            if target_id == source_id:
                continue
            for emb in embeddings:
                for source_emb in source_embeddings:
                    similarity = float(np.dot(source_emb, emb))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = target_id

        if best_target and best_similarity >= PERSON_MERGE_THRESHOLD:
            logger.info(
                f"Merging {source_id} into {best_target} (similarity {best_similarity:.2f})"
            )
            self._merge_person_records(source_id, best_target)
            return best_target

        return source_id

    def _merge_person_records(self, source_id: str, target_id: str):
        """Move DB and memory records from source to target"""
        if source_id == target_id:
            return

        cursor = self.db_conn.cursor()
        cursor.execute("UPDATE faces SET person_id=? WHERE person_id=?", (target_id, source_id))
        cursor.execute("UPDATE crossings SET person_id=? WHERE person_id=?", (target_id, source_id))
        cursor.execute("DELETE FROM persons WHERE person_id=?", (source_id,))
        self.db_conn.commit()

        # Merge embeddings
        if source_id in self.known_embeddings:
            for emb in self.known_embeddings[source_id]:
                self.known_embeddings[target_id].append(emb)
            del self.known_embeddings[source_id]

        # Move thumbnail if target missing
        if target_id not in self.person_thumbnails and source_id in self.person_thumbnails:
            self.person_thumbnails[target_id] = self.person_thumbnails[source_id]
        self.person_thumbnails.pop(source_id, None)

        # Merge last seen
        if source_id in self.person_last_seen:
            target_history = self.person_last_seen.setdefault(target_id, {})
            for key, value in self.person_last_seen[source_id].items():
                target_history[key] = value
            del self.person_last_seen[source_id]

        # Remove temp embedding cache
        self.temp_embeddings.pop(source_id, None)
    
    def log_crossing(self, person_id, direction, timestamp):
        """Log crossing event to database"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO crossings (person_id, direction, t_cross)
            VALUES (?, ?, ?)
        """, (person_id, direction, timestamp))
        self.db_conn.commit()
    
    def update_stats(self):
        """Update global statistics"""
        cursor = self.db_conn.cursor()
        
        with stats_lock:
            # Unique visitors
            cursor.execute("SELECT COUNT(DISTINCT person_id) FROM crossings")
            global_stats["unique_visitors"] = cursor.fetchone()[0] or 0
            
            # Total in/out
            cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'in'")
            global_stats["total_in"] = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'out'")
            global_stats["total_out"] = cursor.fetchone()[0] or 0
            
            # Current occupancy is simply total entries (no exits tracked)
            global_stats["current_occupancy"] = global_stats["total_in"]
            
            # Average dwell time
            cursor.execute("""
                SELECT person_id, 
                       MIN(CASE WHEN direction = 'in' THEN t_cross END) as first_in,
                       MAX(CASE WHEN direction = 'out' THEN t_cross END) as last_out
                FROM crossings
                GROUP BY person_id
                HAVING first_in IS NOT NULL AND last_out IS NOT NULL
            """)
            
            dwell_times = []
            for _, first_in, last_out in cursor.fetchall():
                if first_in and last_out and last_out > first_in:
                    dwell_times.append(last_out - first_in)
            
            if dwell_times:
                avg_dwell_seconds = np.mean(dwell_times)
                global_stats["avg_dwell_minutes"] = round(avg_dwell_seconds / 60, 2)
            else:
                global_stats["avg_dwell_minutes"] = 0.0
    
    def process_frame(self, frame, camera_type='door_a', frame_age: Optional[float] = None):
        start_time = time.time()
        """Process a single frame from either camera"""
        height, width = frame.shape[:2]
        
        # Run YOLO detection
        results = self.yolo_model(frame, conf=0.5, verbose=False, classes=[0])  # Only detect confident persons
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                
                # Extract face embedding + cropped face image
                embedding, face_image = self.extract_face_embedding(frame, box)

                if embedding is not None:
                    # Process based on camera type
                    person_id = None
                    if not self._entry_should_suppress(center):
                        person_id = self.process_entry_detection(embedding, face_image)
                        if person_id:
                            self._remember_entry_position(center)
                    else:
                        logger.info("%s suppression: ignoring repeated detection near doorway", camera_type.upper())
                    color = (0, 255, 0) if person_id else (128, 128, 0)  # Green if counted, amber otherwise
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if person_id:
                        label = f"{person_id[:10]}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # No face detected, just draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        
        # Add camera label
        label = f"{camera_type.replace('_', ' ').upper()} ENTRY"
        cv2.putText(frame, label, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display statistics on the first camera overlay to keep layout clean
        if camera_type == 'door_a':
            with stats_lock:
                stats_text = [
                    f"Unique Visitors: {global_stats['unique_visitors']}",
                    f"Total Entries: {global_stats['total_in']}",
                    f"Total Inside: {global_stats['current_occupancy']}",
                    f"Avg Dwell: {global_stats['avg_dwell_minutes']:.1f} min"
                ]
            
            y_offset = 50
            for text in stats_text:
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 18

            if frame_age is not None:
                cv2.putText(frame, f"Lag: {frame_age:.2f}s", (10, y_offset + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        proc_ms = (time.time() - start_time) * 1000
        cv2.putText(frame, f"Proc: {proc_ms:.0f} ms", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return frame
    
    def run_dual_cameras(self):
        """Run the dual camera system"""
        try:
            cap_entry, cap_exit = self._open_cameras()
        except Exception as exc:
            logger.error(exc)
            return

        logger.info("Started dual entry-only system")
        logger.info("Press 'q' to quit, 's' to save snapshot")
        
        try:
            while True:
                if self.camera_reload_requested:
                    cap_entry, cap_exit = self._reload_cameras(cap_entry, cap_exit)
                    continue

                # Read from entry camera
                ret_entry, frame_entry, age_entry = cap_entry.read()
                if not ret_entry or frame_entry is None:
                    time.sleep(0.01)
                    continue
                if age_entry and age_entry > 1.5:
                    now = time.time()
                    if now - self._last_entry_latency_log > 1.0:
                        logger.warning("Door A frame lag %.2fs", age_entry)
                        self._last_entry_latency_log = now
                
                if self.split_screen:
                    # Split the single camera frame
                    height, width = frame_entry.shape[:2]
                    frame_exit = frame_entry[:, width//2:]
                    frame_entry = frame_entry[:, :width//2]
                    age_exit = age_entry
                else:
                    # Read from second entry camera
                    ret_exit, frame_exit, age_exit = cap_exit.read()
                    if not ret_exit or frame_exit is None:
                        time.sleep(0.01)
                        continue
                    if age_exit and age_exit > 1.5:
                        now = time.time()
                        if now - self._last_exit_latency_log > 1.0:
                            logger.warning("Door B frame lag %.2fs", age_exit)
                            self._last_exit_latency_log = now
                
                loop_start = time.time()
                processed_entry = self.process_frame(frame_entry, 'door_a', age_entry)
                processed_exit = self.process_frame(frame_exit, 'door_b', age_exit)
                loop_time = time.time() - loop_start
                if loop_time > 0.5:
                    logger.warning("Processing pipeline took %.2fs this cycle", loop_time)
                
                # Combine frames side by side for display
                combined = self._combine_frames(processed_entry, processed_exit)
                
                # Display
                cv2.imshow("Dual Entry People Counter", combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, combined)
                    logger.info(f"Saved snapshot: {filename}")
        
        finally:
            self._release_cameras(cap_entry, cap_exit)
            cv2.destroyAllWindows()
            if self.db_conn:
                self.db_conn.close()

# FastAPI endpoints
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current statistics"""
    with stats_lock:
        return StatsResponse(
            unique_visitors=global_stats["unique_visitors"],
            avg_dwell_minutes=global_stats["avg_dwell_minutes"],
            total_in=global_stats["total_in"],
            total_out=global_stats["total_out"],
            current_occupancy=global_stats["current_occupancy"],
            timestamp=datetime.now().isoformat()
        )


@app.get("/cameras")
async def get_cameras():
    """Return list of detected cameras and current configuration"""
    if COUNTER_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Camera system not initialized")

    return {
        "available": COUNTER_INSTANCE.list_available_cameras(),
        "active": {
            "entry_cam": str(COUNTER_INSTANCE.entry_cam_id),
            "exit_cam": str(COUNTER_INSTANCE.exit_cam_id),
            "split_screen": COUNTER_INSTANCE.split_screen
        }
    }


@app.post("/cameras/config")
async def set_cameras(config: CameraConfigRequest):
    """Update camera assignment on the fly"""
    if COUNTER_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Camera system not initialized")

    success, message = COUNTER_INSTANCE.update_camera_config(
        config.entry_cam,
        config.exit_cam,
        config.split_screen
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "ok", "message": message}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dual Entry Face Recognition People Counter API",
        "version": "2.0.0",
        "endpoints": {
            "/stats": "Get current statistics",
            "/": "This help message"
        },
        "info": "System uses two entry cameras to count everyone walking through either door via face recognition"
    }

def run_api():
    """Run FastAPI server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def signal_handler(sig, frame):
    """Handle shutdown signal"""
    logger.info("Shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Dual Entry Face Recognition People Counter')
    parser.add_argument('--entry-cam', type=str, default="0",
                       help='Door A camera source (index like 0 or RTSP/USB URL)')
    parser.add_argument('--exit-cam', type=str, default="1",
                       help='Door B camera source (index like 1 or RTSP/USB URL)')
    parser.add_argument('--split-screen', action='store_true', 
                       help='Use single camera split-screen simulation')
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if database exists
    if not Path(DB_PATH).exists():
        logger.error(f"Database not found at {DB_PATH}")
        logger.error("Please run 'python database_init.py' first to initialize the database")
        sys.exit(1)
    
    # Start API server in background thread
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("FastAPI server started at http://localhost:8000")
    
    # Start dual camera counter
    global COUNTER_INSTANCE
    counter = DualEntryOnlyCounter(
        entry_cam_id=args.entry_cam,
        exit_cam_id=args.exit_cam,
        split_screen=args.split_screen
    )
    COUNTER_INSTANCE = counter
    
    # Update initial stats
    counter.update_stats()
    
    # Run the dual camera system
    counter.run_dual_cameras()

if __name__ == "__main__":
    main()

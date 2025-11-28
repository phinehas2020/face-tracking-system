#!/usr/bin/env python3
"""
Dual Camera Face Recognition People Counter (Entry-Only)
=========================================================

This system now uses two cameras as complementary entry viewpoints:
- Camera 1 (ENTRY A): Detects faces entering
- Camera 2 (ENTRY B): Detects faces entering from a second angle

Every detection on either camera is treated as an entry; exits are not tracked.

Usage:
------
1. Run with two cameras:
   python main_dual_camera.py --entry-cam 0 --exit-cam 1

2. Run with single camera simulation (split screen):
   python main_dual_camera.py --entry-cam 0 --exit-cam 0 --split-screen

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
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_tracking.db"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
RECORDING_DIR = BASE_DIR / "recordings"

from config import (
    FACE_MATCH_THRESHOLD, FALLBACK_MATCH_THRESHOLD, MIN_RECENT_MATCH,
    RECENT_PERSON_WINDOW, PERSON_MERGE_THRESHOLD, MIN_FACE_SIZE,
    FACE_COOLDOWN_TIME, EMBEDDING_EXPIRE_TIME, EMBEDDING_REFRESH_INTERVAL,
    MAX_EMBEDDINGS_PER_PERSON, THUMBNAIL_SIZE, ENTRY_POSITION_SUPPRESSION_RADIUS,
    ENTRY_POSITION_SUPPRESSION_WINDOW, ENTRY_RECENT_SIM_THRESHOLD,
    ENTRY_RECENT_WINDOW, MAX_FACE_YAW_DEG, MAX_FACE_PITCH_DEG,
    ENABLE_RECORDING, APPEARANCE_SIM_THRESHOLD, APPEARANCE_MAX_AGE,
    APPEARANCE_MIN_AREA
)
from video_recorder import ThreadedVideoRecorder
from watchlist import WatchlistManager

PEER_URL = os.getenv("PEER_URL")  # URL of the peer instance
WATCHLIST_DIR = BASE_DIR / "watchlist"

# Global variables
app = FastAPI(title="Dual Camera People Counter API", version="2.0.0")
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
    "known_faces": 0,
    "total_in": 0,
    "total_out": 0,
    "avg_dwell_minutes": 0.0,
    "current_occupancy": 0,
    "body_in": 0,
    "body_out": 0,
    "body_net": 0,
    "count_drift": 0,
    "last_body_event": None,
    "peer_status": "disabled",
    "peer_url": PEER_URL,
    "peer_data": None
}

COUNTER_INSTANCE = None

class StatsResponse(BaseModel):
    unique_visitors: int
    known_faces: int  # Total known faces synced across all stations
    avg_dwell_minutes: float
    total_in: int
    total_out: int
    current_occupancy: int
    body_in: int
    body_out: int
    body_net: int
    count_drift: int
    last_body_event: Optional[str] = None
    peer_status: str
    peer_data: Optional[Dict] = None
    timestamp: str

class SyncFaceRequest(BaseModel):
    person_id: str
    name: str
    embedding: List[float]


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

class DualCameraCounter:
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
        self.person_last_seen = {}  # person_id -> {entry: time, exit: time}
        self.active_persons = set()  # Currently inside the premises
        self.active_persons = set()  # Currently inside the premises
        self.appearance_signatures = {}  # person_id -> (signature_vec, ts)
        self.person_names = {}  # person_id -> name
        self.next_visitor_idx = 1
        self.camera_lock = Lock()
        self.camera_reload_requested = False
        self._last_entry_latency_log = 0.0
        self._last_exit_latency_log = 0.0
        self._last_appearance_cleanup = 0.0
        self.recent_entry_positions = deque()  # (x, y, timestamp)
        self.recent_entry_candidates = deque(maxlen=50)  # (person_id, embedding, timestamp)

        # Video Recorders
        self.entry_recorder = None
        self.exit_recorder = None

        # Frame counters
        self.entry_frame_count = 0
        self.exit_frame_count = 0
        
        # Initialize components
        self._init_yolo()
        self._init_face_analyzer()
        self._init_database()
        self._load_known_faces()
        self._init_watchlist()
        self.sync_from_peer()  # Pull existing faces from peer station

    def _init_yolo(self):
        """Initialize YOLO model for detection"""
        try:
            self.yolo_model = YOLO('yolov8s.pt')
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
            self.face_app.prepare(ctx_id=0, det_size=(480, 480))
            logger.info("Face analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face analyzer: {e}")
            raise

    def _init_watchlist(self):
        """Initialize watchlist manager for flagged persons"""
        try:
            self.watchlist = WatchlistManager(WATCHLIST_DIR, DB_PATH, self.face_app)
            logger.info(f"Watchlist initialized with {len(self.watchlist.get_watchlist_names())} people")
        except Exception as e:
            logger.error(f"Failed to initialize watchlist: {e}")
            self.watchlist = None

    def _init_database(self):
        """Initialize database connection"""
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()
        self._migrate_temp_ids()
        logger.info(f"Connected to database: {DB_PATH}")

    def _migrate_temp_ids(self):
        """Migrate legacy temp_ IDs to visitor_N format"""
        cursor = self.db_conn.cursor()
        
        # 1. Determine next available visitor index from DB
        cursor.execute("SELECT name FROM persons WHERE name LIKE 'Visitor %'")
        max_idx = 0
        for (name,) in cursor.fetchall():
            try:
                idx = int(name.split(" ")[1])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
        
        current_idx = max_idx + 1
        
        # 2. Find temp IDs
        cursor.execute("SELECT person_id, thumbnail_path FROM persons WHERE person_id LIKE 'temp_%'")
        temp_rows = cursor.fetchall()
        
        if not temp_rows:
            return

        logger.info(f"Migrating {len(temp_rows)} legacy temp IDs...")
        
        for temp_id, old_thumb_path in temp_rows:
            new_id = f"visitor_{current_idx}"
            new_name = f"Visitor {current_idx}"
            current_idx += 1
            
            # Rename folder
            old_folder = BASE_DIR / "consent_gallery" / "auto_learned" / temp_id
            new_folder = BASE_DIR / "consent_gallery" / "auto_learned" / new_id
            
            new_thumb_path = old_thumb_path
            if old_folder.exists():
                try:
                    if new_folder.exists():
                        import shutil
                        shutil.rmtree(new_folder)
                    old_folder.rename(new_folder)
                    
                    # Update thumbnail path string if it contains the old ID
                    if old_thumb_path and temp_id in old_thumb_path:
                        new_thumb_path = old_thumb_path.replace(temp_id, new_id)
                except Exception as e:
                    logger.error(f"Failed to move folder {old_folder} to {new_folder}: {e}")
            
            # Update DB tables
            try:
                # Create new person record
                cursor.execute("""
                    INSERT INTO persons (person_id, name, consent_ts, thumbnail_path)
                    SELECT ?, ?, consent_ts, ? FROM persons WHERE person_id = ?
                """, (new_id, new_name, new_thumb_path, temp_id))
                
                # Move children
                cursor.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (new_id, temp_id))
                cursor.execute("UPDATE crossings SET person_id = ? WHERE person_id = ?", (new_id, temp_id))
                
                # Delete old person record
                cursor.execute("DELETE FROM persons WHERE person_id = ?", (temp_id,))
                
                logger.info(f"Migrated {temp_id} -> {new_id}")
                
            except Exception as e:
                logger.error(f"Database migration failed for {temp_id}: {e}")
                
        self.db_conn.commit()
        self.next_visitor_idx = current_idx

    def _ensure_schema(self):
        """Ensure new columns exist for older databases"""
        cursor = self.db_conn.cursor()
        cursor.execute("PRAGMA table_info(persons)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'thumbnail_path' not in columns:
            cursor.execute("ALTER TABLE persons ADD COLUMN thumbnail_path TEXT")
            self.db_conn.commit()
        # Body counter table for side-view verification
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS body_crossings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
                t_cross REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_body_crossings_time ON body_crossings(t_cross)")
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
            
            # Cache name
            self.person_names[person_id] = name
            
            # Update sequential counter if applicable
            if name and name.startswith("Visitor "):
                try:
                    idx = int(name.split(" ")[1])
                    if idx >= self.next_visitor_idx:
                        self.next_visitor_idx = idx + 1
                except ValueError:
                    pass
                    
            logger.debug(f"Loaded embedding for {name} (ID: {person_id})")

        logger.info(f"Loaded embeddings for {len(self.known_embeddings)} known people. Next Visitor ID: {self.next_visitor_idx}")

    def _find_matching_person_by_embedding(self, embedding: np.ndarray) -> Optional[str]:
        """Check if embedding matches any known person using PERSON_MERGE_THRESHOLD"""
        best_match = None
        best_similarity = -1.0

        for person_id, embeddings in self.known_embeddings.items():
            for known_embedding in embeddings:
                similarity = float(np.dot(embedding, known_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id

        if best_match and best_similarity >= PERSON_MERGE_THRESHOLD:
            return best_match
        return None

    def sync_from_peer(self):
        """Pull all existing faces from peer station on startup"""
        if not PEER_URL:
            return

        logger.info(f"PEER SYNC: Attempting to pull faces from {PEER_URL}")

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{PEER_URL}/sync/faces/all")
                if response.status_code != 200:
                    logger.warning(f"PEER SYNC: Failed to get faces from peer: {response.status_code}")
                    return

                data = response.json()
                faces = data.get("faces", [])
                imported = 0
                skipped_id = 0
                skipped_embedding = 0

                for face in faces:
                    person_id = face.get("person_id")
                    name = face.get("name")
                    embedding_list = face.get("embedding")

                    if not person_id or not embedding_list:
                        continue

                    # Skip if we already have this exact person_id
                    if person_id in self.known_embeddings:
                        skipped_id += 1
                        continue

                    # Normalize embedding
                    embedding = np.array(embedding_list, dtype=np.float32)
                    embedding = embedding / np.linalg.norm(embedding)

                    # Check if this face matches an existing person by embedding similarity
                    existing_match = self._find_matching_person_by_embedding(embedding)
                    if existing_match:
                        logger.info(f"PEER SYNC: Skipping {person_id} - matches existing {existing_match} by embedding")
                        skipped_embedding += 1
                        continue

                    # Add to database and memory
                    cursor = self.db_conn.cursor()

                    # Check if person exists in DB
                    cursor.execute("SELECT 1 FROM persons WHERE person_id = ?", (person_id,))
                    if not cursor.fetchone():
                        cursor.execute("""
                            INSERT INTO persons (person_id, name, consent_ts)
                            VALUES (?, ?, ?)
                        """, (person_id, name, datetime.now().isoformat()))

                        cursor.execute("""
                            INSERT INTO faces (person_id, embedding, created_ts)
                            VALUES (?, ?, ?)
                        """, (person_id, pickle.dumps(embedding), datetime.now().isoformat()))

                        self.db_conn.commit()

                    # Add to memory
                    self.known_embeddings[person_id].append(embedding)
                    self.person_names[person_id] = name

                    # Update visitor index if needed
                    if name and name.startswith("Visitor "):
                        try:
                            idx = int(name.split(" ")[1])
                            if idx >= self.next_visitor_idx:
                                self.next_visitor_idx = idx + 1
                        except ValueError:
                            pass

                    imported += 1

                logger.info(f"PEER SYNC: Imported {imported} faces, skipped {skipped_id} by ID, {skipped_embedding} by embedding match")

                # Update stats to reflect new known_faces count
                if imported > 0:
                    self.update_stats()

        except Exception as e:
            logger.warning(f"PEER SYNC: Could not sync from peer: {e}")

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
        """
        Saves a reference face image for a new person to the consent gallery for auto-learning,
        and also saves a smaller thumbnail for immediate UI display.
        """
        if face_image is None or face_image.size == 0:
            logger.debug(f"Save thumbnail called for {person_id} with empty image.")
            return None

        try:
            # 1. Save the canonical, high-quality reference image to the consent gallery
            person_gallery_path = BASE_DIR / "consent_gallery" / "auto_learned" / person_id
            person_gallery_path.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            reference_image_path = person_gallery_path / f"face_{timestamp}.jpg"
            cv2.imwrite(str(reference_image_path), face_image)
            resolved_reference_path = str(reference_image_path.resolve())
            logger.info(f"Saved new reference face for {person_id} to: {resolved_reference_path}")

            # 2. Save a resized thumbnail for any immediate UI that uses the /thumbnails dir
            display_thumb_path = THUMBNAIL_DIR / f"{person_id}.jpg"
            try:
                thumb = cv2.resize(face_image, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(display_thumb_path), thumb)
                # Update in-memory dict for live display
                self.person_thumbnails[person_id] = str(display_thumb_path.resolve())
            except Exception as e:
                logger.warning(f"Could not create display thumbnail for {person_id}: {e}")

            # Return the path to the canonical reference image to be stored in the DB
            return resolved_reference_path

        except Exception as exc:
            logger.error(f"Failed to save thumbnail for {person_id}: {exc}")
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
        logger.info(f"Refreshed embedding for {person_id}")
        
        # Sync update to peer
        if PEER_URL:
             # Use current name (which is just ID now)
             name = person_id
             Thread(target=self.sync_to_peer, args=(person_id, name, embedding), daemon=True).start()

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
            "Camera reconfiguration requested -> entry: %s exit: %s split: %s",
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
            "Cameras active -> entry: %s exit: %s split: %s",
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
        # Stop existing recorders before switching
        if self.entry_recorder:
            self.entry_recorder.stop()
            self.entry_recorder = None
        if self.exit_recorder:
            self.exit_recorder.stop()
            self.exit_recorder = None

        try:
            new_entry, new_exit = self._open_cameras()
            
            # Initialize new recorders if enabled
            if ENABLE_RECORDING:
                self._init_recorders(new_entry, new_exit)
                
        except Exception as exc:
            logger.error(f"Unable to reconfigure cameras: {exc}")
            with self.camera_lock:
                self.camera_reload_requested = False
            return cap_entry, cap_exit

        self._release_cameras(cap_entry, cap_exit)
        with self.camera_lock:
            self.camera_reload_requested = False
        return new_entry, new_exit
    
    def _init_recorders(self, cap_entry, cap_exit):
        """Initialize video recorders for active cameras"""
        if cap_entry:
            ret, frame, _ = cap_entry.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if self.split_screen:
                    w = w // 2  # Split screen records half width per virtual camera
                
                self.entry_recorder = ThreadedVideoRecorder(
                    RECORDING_DIR, "entry_cam", w, h
                )
                
                if self.split_screen:
                     self.exit_recorder = ThreadedVideoRecorder(
                        RECORDING_DIR, "exit_cam", w, h
                    )
        
        if cap_exit and not self.split_screen:
            ret, frame, _ = cap_exit.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                self.exit_recorder = ThreadedVideoRecorder(
                    RECORDING_DIR, "exit_cam", w, h
                )

    def _crop_person_region(self, image: np.ndarray, bbox, pad: int = 20) -> Optional[np.ndarray]:
        """Return a padded crop around a detected person box."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = image[y1:y2, x1:x2]
            return crop if crop.size > 0 else None
        except Exception:
            return None

    def compute_appearance_signature(self, person_crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Compute a simple color signature for appearance-based matching (for side/back views).
        Uses HSV histograms to stay stable under lighting shifts.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h * w < APPEARANCE_MIN_AREA:
            return None

        try:
            resized = cv2.resize(person_crop, (96, 192), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            signature = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
            norm = np.linalg.norm(signature)
            if norm == 0:
                return None
            return signature / norm
        except Exception:
            return None

    def _remember_appearance(self, person_id: str, signature: Optional[np.ndarray], ts: Optional[float] = None):
        """Cache an appearance signature for fallback matching."""
        if signature is None:
            return
        timestamp = ts or time.time()
        self.appearance_signatures[person_id] = (signature, timestamp)

    def _prune_appearance_signatures(self, now: Optional[float] = None):
        """Drop very old appearance signatures to avoid stale matches."""
        now = now or time.time()
        if now - self._last_appearance_cleanup < 5.0:
            return
        self._last_appearance_cleanup = now
        expired = [
            pid for pid, (_, ts) in self.appearance_signatures.items()
            if now - ts > APPEARANCE_MAX_AGE
        ]
        for pid in expired:
            self.appearance_signatures.pop(pid, None)

    def extract_face_embedding(self, image, bbox, person_crop: Optional[np.ndarray] = None):
        """Extract face embedding and cropped face image from a detected person"""
        try:
            if person_crop is None:
                person_crop = self._crop_person_region(image, bbox)
            if person_crop is None or person_crop.size == 0:
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

            # Stricter detection score (was 0.5)
            if getattr(largest_face, "det_score", 1.0) < 0.65:
                logger.debug(f"Rejected face due to low detection score: {largest_face.det_score:.2f}")
                return None, None

            pose = getattr(largest_face, "pose", None)
            if pose is not None:
                yaw, pitch, roll = map(abs, pose)
                # Stricter pose limits (was 40)
                if yaw > 30 or pitch > 30:
                    logger.debug(
                        "Rejected face due to pose (yaw %.1f°, pitch %.1f°)", yaw, pitch
                    )
                    return None, None

            face_width = largest_face.bbox[2] - largest_face.bbox[0]
            face_height = largest_face.bbox[3] - largest_face.bbox[1]
            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                return None, None
                
            # Check face-to-crop ratio to avoid tiny faces in large crops (background people)
            face_area = face_width * face_height
            crop_area = person_crop.shape[0] * person_crop.shape[1]
            if crop_area > 0 and (face_area / crop_area) < 0.03:
                logger.debug("Rejected face: too small relative to person crop")
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

        # Check primary match threshold
        if best_match and best_similarity >= FACE_MATCH_THRESHOLD:
            logger.info(
                f"MATCH FOUND: Face matched to {best_match} (temp={match_is_temp}) with similarity {best_similarity:.3f}"
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
    
    def process_entry_detection(self, embedding, face_image=None, appearance_signature=None):
        """Process a face detection from the entry camera"""
        if embedding is None:
            return None

        # Check watchlist for flagged persons
        if self.watchlist:
            watchlist_match = self.watchlist.check_face(embedding)
            if watchlist_match:
                name, similarity = watchlist_match
                self.watchlist.record_alert(name, similarity)

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
        last_entry = self.person_last_seen.get(person_id, {}).get('entry', 0)
        if current_time - last_entry < FACE_COOLDOWN_TIME:
            return None  # Too soon, skip

        # Record entry
        final_id = person_id
        
        # Check if we need to convert temp ID to permanent
        # Do this even if they are already active, in case they were active as temp
        if person_id.startswith('temp_'):
            merged_id = self.create_permanent_person(person_id, embedding, face_image)
            if merged_id != person_id:
                final_id = merged_id
                # Migrate history if needed
                if person_id in self.person_last_seen:
                    history = self.person_last_seen.pop(person_id)
                    self.person_last_seen.setdefault(final_id, {}).update(history)

        history = self.person_last_seen.setdefault(final_id, {})
        history['entry'] = current_time
        
        if appearance_signature is not None:
            self._remember_appearance(final_id, appearance_signature, current_time)

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
        logger.info(f"ENTRY: {final_id} entered at {datetime.now().strftime('%H:%M:%S')}")
        return final_id

    def _record_exit(self, person_id: str, embedding=None, face_image=None,
                     appearance_signature=None, timestamp: Optional[float] = None,
                     reason: str = "face"):
        """Shared exit logger supported by both face- and appearance-based matches."""
        if person_id is None:
            return None

        current_time = timestamp or time.time()
        history = self.person_last_seen.setdefault(person_id, {})
        last_exit = history.get('exit', 0)
        if current_time - last_exit < FACE_COOLDOWN_TIME:
            return None  # Too soon, skip

        should_log = False

        if person_id in self.active_persons:
            self.active_persons.remove(person_id)
            should_log = True
        else:
            last_entry = history.get('entry', 0)
            if last_entry and (current_time - last_entry) <= RECENT_PERSON_WINDOW:
                should_log = True
            else:
                logger.warning(
                    f"Exit detected for {person_id} but no recent entry; skipping"
                )
                return None

        history['exit'] = current_time
        if appearance_signature is not None:
            self._remember_appearance(person_id, appearance_signature, current_time)

        if person_id not in self.person_thumbnails and face_image is not None:
            thumbnail_path = self.save_thumbnail(person_id, face_image)
            self.update_person_thumbnail(person_id, thumbnail_path)

        if should_log:
            self.log_crossing(person_id, 'out', current_time)
            if embedding is not None:
                self.maybe_refresh_embedding(person_id, embedding)
            self.update_stats()
            logger.info(
                f"EXIT ({reason}): {person_id} at {datetime.now().strftime('%H:%M:%S')}"
            )
            return person_id

        return None

    def process_exit_detection(self, embedding, face_image=None, appearance_signature=None):
        """Entry-only mode: treat the second camera as another entry viewpoint."""
        return self.process_entry_detection(embedding, face_image, appearance_signature)

    def process_exit_by_appearance(self, appearance_signature: Optional[np.ndarray], timestamp: Optional[float] = None):
        """Fallback for side/back views: match clothing colors against active visitors."""
        if appearance_signature is None or not self.active_persons:
            return None

        now = timestamp or time.time()
        self._prune_appearance_signatures(now)

        best_id = None
        best_sim = -1.0

        for person_id in list(self.active_persons):
            cached = self.appearance_signatures.get(person_id)
            if not cached:
                continue
            cached_sig, sig_ts = cached
            if now - sig_ts > APPEARANCE_MAX_AGE:
                continue

            similarity = float(np.dot(appearance_signature, cached_sig))
            if similarity > best_sim:
                best_sim = similarity
                best_id = person_id

        if best_id and best_sim >= APPEARANCE_SIM_THRESHOLD:
            logger.info(
                "EXIT appearance assist -> %s (sim=%.2f)",
                best_id,
                best_sim
            )
            return self._record_exit(
                best_id,
                embedding=None,
                face_image=None,
                appearance_signature=appearance_signature,
                timestamp=now,
                reason="appearance"
            )

        return None
    
    def create_permanent_person(self, temp_id, embedding, face_image=None):
        """Convert temporary person to permanent in database"""
        # Check if already converted
        if temp_id in self.person_names and self.person_names[temp_id].startswith("Visitor "):
            return temp_id

        cursor = self.db_conn.cursor()

        # Generate new ID and Name
        new_id = f"visitor_{self.next_visitor_idx}"
        name = f"Visitor {self.next_visitor_idx}"
        self.next_visitor_idx += 1
        
        consent_ts = datetime.now().isoformat()
        
        # Use new_id for thumbnail so folder is named correctly
        thumbnail_path = self.save_thumbnail(new_id, face_image)

        cursor.execute("""
            INSERT OR IGNORE INTO persons (person_id, name, consent_ts, thumbnail_path)
            VALUES (?, ?, ?, ?)
        """, (new_id, name, consent_ts, thumbnail_path))

        if thumbnail_path:
            self.person_thumbnails[new_id] = thumbnail_path

        self.db_conn.commit()
        logger.info(f"Created permanent person: {new_id} ({name})")
        
        # Update name cache
        self.person_names[new_id] = name

        # Store embedding in DB & memory using NEW ID
        self.add_embedding(new_id, embedding, persist=True)
        self.last_embedding_update[new_id] = time.time()

        # Remove temp record if present
        self.temp_embeddings.pop(temp_id, None)
        self.person_last_seen.pop(temp_id, None)
        self.active_persons.discard(temp_id)
        
        # Merge with an existing person if similarity is high
        merged_id = self.merge_if_duplicate(new_id)
        if merged_id != new_id:
            logger.info(f"Created permanent person: {merged_id} (merged from {temp_id} -> {new_id})")
        else:
            logger.info(f"Created permanent person: {new_id} (from {temp_id})")
        self.last_embedding_update[merged_id] = time.time()

        # Sync to peer if configured (after merge resolution)
        if PEER_URL:
            Thread(
                target=self.sync_to_peer,
                args=(merged_id, self.person_names.get(merged_id, name), embedding),
                daemon=True
            ).start()

        return merged_id

    def sync_to_peer(self, person_id, name, embedding):
        """Send new person to peer instance"""
        try:
            if embedding is None:
                return
                
            payload = {
                "person_id": person_id,
                "name": name,
                "embedding": embedding.tolist()
            }
            
            with httpx.Client() as client:
                response = client.post(f"{PEER_URL}/sync/face", json=payload, timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"Successfully synced {person_id} to peer")
                else:
                    logger.warning(f"Failed to sync to peer: {response.status_code} {response.text}")
                    
        except Exception as e:
            logger.error(f"Error syncing to peer: {e}")
    
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

        # Carry over appearance signature if newer
        if source_id in self.appearance_signatures:
            sig, ts = self.appearance_signatures.pop(source_id)
            existing = self.appearance_signatures.get(target_id)
            if not existing or ts > existing[1]:
                self.appearance_signatures[target_id] = (sig, ts)

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

        # Update active persons
        if source_id in self.active_persons:
            self.active_persons.remove(source_id)
            self.active_persons.add(target_id)

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
            # Total known faces (synced across all stations)
            global_stats["known_faces"] = len(self.known_embeddings)

            # Unique visitors who entered at THIS station
            cursor.execute("SELECT COUNT(DISTINCT person_id) FROM crossings")
            global_stats["unique_visitors"] = cursor.fetchone()[0] or 0

            # Total in/out at this station
            cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'in'")
            global_stats["total_in"] = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'out'")
            global_stats["total_out"] = cursor.fetchone()[0] or 0

            # Body counter (side-view) stats for cross-checking
            body_in = 0
            body_out = 0
            last_body_ts = None
            try:
                cursor.execute("SELECT COUNT(*) FROM body_crossings WHERE direction = 'in'")
                body_in = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM body_crossings WHERE direction = 'out'")
                body_out = cursor.fetchone()[0] or 0
                cursor.execute("SELECT t_cross FROM body_crossings ORDER BY t_cross DESC LIMIT 1")
                ts_row = cursor.fetchone()
                last_body_ts = ts_row[0] if ts_row and ts_row[0] is not None else None
            except sqlite3.Error as exc:
                logger.debug("Body counter stats unavailable: %s", exc)
            global_stats["body_in"] = body_in
            global_stats["body_out"] = body_out
            global_stats["body_net"] = body_in - body_out
            net_face = global_stats["total_in"] - global_stats["total_out"]
            global_stats["count_drift"] = global_stats["body_net"] - net_face
            global_stats["last_body_event"] = (
                datetime.fromtimestamp(last_body_ts).isoformat() if last_body_ts else None
            )
            
            # Entry-only mode: occupancy equals total entries (no exits tracked)
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
    
    def process_frame(self, frame, camera_type='entry', frame_age: Optional[float] = None):
        start_time = time.time()
        """Process a single frame from either camera"""
        height, width = frame.shape[:2]
        self._prune_appearance_signatures()
        
        # Run YOLO detection
        results = self.yolo_model(frame, conf=0.5, verbose=False, classes=[0])  # Only detect confident persons
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                
                person_crop = self._crop_person_region(frame, box)
                appearance_sig = self.compute_appearance_signature(person_crop)
                embedding, face_image = self.extract_face_embedding(frame, box, person_crop)

                person_id = None
                match_hint = None

                if embedding is not None:
                    # Both cameras are entry viewpoints in this mode
                    if not self._entry_should_suppress(center):
                        person_id = self.process_entry_detection(embedding, face_image, appearance_sig)
                        if person_id:
                            self._remember_entry_position(center)
                    else:
                        logger.info("ENTRY suppression: ignoring repeated detection near doorway")
                    color = (0, 255, 0) if person_id else (128, 128, 0)  # Green if entered, amber otherwise
                    match_hint = "face" if person_id else None
                else:
                    # Default when no face was extracted
                    color = (128, 128, 128)

                # Appearance-based fallback for side/back exits
                # if camera_type == 'exit' and person_id is None:
                #     fallback_id = self.process_exit_by_appearance(appearance_sig)
                #     if fallback_id:
                #         person_id = fallback_id
                #         color = (0, 165, 255)  # Orange for assisted match
                #         match_hint = "assist"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if person_id:
                    # Use name if available, otherwise ID
                    display_name = self.person_names.get(person_id, person_id)
                    if display_name.startswith("temp_"):
                        display_name = "Analyzing..."
                        
                    label = f"{display_name}"
                    if match_hint == "assist":
                        label += " ·A"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add camera label
        if camera_type in ("entry", "entry_a"):
            label = "ENTRY A CAMERA"
        else:
            label = "ENTRY B CAMERA"
        cv2.putText(frame, label, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display statistics on primary entry camera only
        if camera_type in ('entry', 'entry_a'):
            with stats_lock:
                stats_text = [
                    f"Unique Visitors: {global_stats['unique_visitors']}",
                    f"Total Entries: {global_stats['total_in']}",
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
            if ENABLE_RECORDING:
                self._init_recorders(cap_entry, cap_exit)
        except Exception as exc:
            logger.error(exc)
            return

        logger.info("Started dual camera system")
        logger.info("Press 'q' to quit, 's' to save snapshot")
        
        try:
            while True:
                if self.camera_reload_requested:
                    cap_entry, cap_exit = self._reload_cameras(cap_entry, cap_exit)
                    continue

                # Read from entry camera A
                ret_entry, frame_entry, age_entry = cap_entry.read()
                if not ret_entry or frame_entry is None:
                    time.sleep(0.01)
                    continue
                
                # Record entry frame (raw)
                if self.entry_recorder and not self.split_screen:
                     self.entry_recorder.record_frame(frame_entry)

                if age_entry and age_entry > 1.5:
                    now = time.time()
                    if now - self._last_entry_latency_log > 1.0:
                        logger.warning("Entry frame lag %.2fs", age_entry)
                        self._last_entry_latency_log = now
                
                if self.split_screen:
                    # Split the single camera frame into A/B
                    height, width = frame_entry.shape[:2]
                    frame_exit = frame_entry[:, width//2:]
                    frame_entry = frame_entry[:, :width//2]
                    age_exit = age_entry
                    
                    # Record split frames
                    if self.entry_recorder:
                        self.entry_recorder.record_frame(frame_entry)
                    if self.exit_recorder:
                        self.exit_recorder.record_frame(frame_exit)
                else:
                    # Read from separate entry camera B
                    ret_exit, frame_exit, age_exit = cap_exit.read()
                    if not ret_exit or frame_exit is None:
                        time.sleep(0.01)
                        continue
                    
                    # Record entry B frame (raw)
                    if self.exit_recorder:
                        self.exit_recorder.record_frame(frame_exit)

                    if age_exit and age_exit > 1.5:
                        now = time.time()
                        if now - self._last_exit_latency_log > 1.0:
                            logger.warning("Entry B frame lag %.2fs", age_exit)
                            self._last_exit_latency_log = now
                
                loop_start = time.time()
                processed_entry = self.process_frame(frame_entry, 'entry_a', age_entry)
                processed_exit = self.process_frame(frame_exit, 'entry_b', age_exit)
                loop_time = time.time() - loop_start
                if loop_time > 0.5:
                    logger.warning("Processing pipeline took %.2fs this cycle", loop_time)
                
                # Combine frames side by side for display
                combined = self._combine_frames(processed_entry, processed_exit)
                
                # Display
                cv2.imshow("Dual Camera People Counter", combined)
                
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
            if self.entry_recorder:
                self.entry_recorder.stop()
            if self.exit_recorder:
                self.exit_recorder.stop()
            cv2.destroyAllWindows()
            if self.db_conn:
                self.db_conn.close()

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current statistics"""
    with stats_lock:
        return StatsResponse(
            unique_visitors=global_stats["unique_visitors"],
            known_faces=global_stats.get("known_faces", 0),
            avg_dwell_minutes=global_stats["avg_dwell_minutes"],
            total_in=global_stats["total_in"],
            total_out=global_stats["total_out"],
            current_occupancy=global_stats["current_occupancy"],
            body_in=global_stats["body_in"],
            body_out=global_stats["body_out"],
            body_net=global_stats["body_net"],
            count_drift=global_stats["count_drift"],
            last_body_event=global_stats["last_body_event"],
            peer_status=global_stats["peer_status"],
            peer_data=global_stats["peer_data"],
            timestamp=datetime.now().isoformat()
        )

@app.post("/sync/face")
async def sync_face(request: SyncFaceRequest):
    """Receive a new face from peer"""
    try:
        # Validate and normalize embedding
        embedding_arr = np.asarray(request.embedding, dtype=np.float32)
        if embedding_arr.ndim != 1:
            raise HTTPException(status_code=400, detail="Embedding must be 1D")
        if len(embedding_arr) < 128 or len(embedding_arr) > 2048:
            raise HTTPException(status_code=400, detail="Embedding length out of expected range")
        if not np.all(np.isfinite(embedding_arr)):
            raise HTTPException(status_code=400, detail="Embedding contains non-finite values")
        norm = float(np.linalg.norm(embedding_arr))
        if not np.isfinite(norm) or norm < 1e-6:
            raise HTTPException(status_code=400, detail="Invalid embedding norm")
        embedding = embedding_arr / norm
        embedding_blob = pickle.dumps(embedding)
        
        # Add to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if person already exists
        cursor.execute("SELECT 1 FROM persons WHERE person_id = ?", (request.person_id,))
        exists = cursor.fetchone()
        
        if not exists:
            # Insert person
            cursor.execute("""
                INSERT INTO persons (person_id, name, consent_ts)
                VALUES (?, ?, ?)
            """, (request.person_id, request.name, datetime.now().isoformat()))
        
        # Always insert face (even if person exists, this is a new embedding)
        cursor.execute("""
            INSERT INTO faces (person_id, embedding, created_ts)
            VALUES (?, ?, ?)
        """, (request.person_id, embedding_blob, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Update in-memory known embeddings
        if hasattr(app.state, 'counter') and app.state.counter:
             app.state.counter.add_embedding(request.person_id, embedding, persist=False)
             app.state.counter.person_names[request.person_id] = request.name
             
             # Update sequence if needed
             if request.name.startswith("Visitor "):
                 try:
                     idx = int(request.name.split(" ")[1])
                     if idx >= app.state.counter.next_visitor_idx:
                         app.state.counter.next_visitor_idx = idx + 1
                 except ValueError:
                     pass
                     
             logger.info(f"PEER SYNC RECEIVED: Loaded person {request.name} (ID: {request.person_id}) from peer station")
             
             # Check for duplicates and merge if necessary
             merged_id = app.state.counter.merge_if_duplicate(request.person_id)
             if merged_id != request.person_id:
                 logger.info(f"PEER SYNC MERGE: {request.person_id} merged into {merged_id}")
        
        return {"status": "synced"}

    except Exception as e:
        logger.error(f"Error processing sync request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sync/faces/all")
async def get_all_faces():
    """Return all known faces for initial sync with peer"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.person_id, p.name, f.embedding
            FROM persons p
            JOIN faces f ON p.person_id = f.person_id
            ORDER BY f.created_ts DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        faces = []
        seen_ids = set()
        for person_id, name, embedding_blob in rows:
            # Only send one embedding per person (most recent)
            if person_id in seen_ids:
                continue
            seen_ids.add(person_id)

            embedding = pickle.loads(embedding_blob)
            faces.append({
                "person_id": person_id,
                "name": name,
                "embedding": embedding.tolist()
            })

        logger.info(f"PEER SYNC: Sending {len(faces)} faces to peer")
        return {"faces": faces, "count": len(faces)}

    except Exception as e:
        logger.error(f"Error getting all faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cameras")
async def get_cameras():
    """Return list of detected cameras and current configuration"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None:
        raise HTTPException(status_code=503, detail="Camera system not initialized")

    return {
        "available": counter.list_available_cameras(),
        "active": {
            "entry_cam": str(counter.entry_cam_id),
            "exit_cam": str(counter.exit_cam_id),
            "split_screen": counter.split_screen
        }
    }


@app.post("/cameras/config")
async def set_cameras(config: CameraConfigRequest):
    """Update camera assignment on the fly"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None:
        raise HTTPException(status_code=503, detail="Camera system not initialized")

    success, message = counter.update_camera_config(
        config.entry_cam,
        config.exit_cam,
        config.split_screen
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "ok", "message": message}


@app.get("/alerts")
async def get_alerts(since: int = 3600, unacknowledged: bool = True):
    """Get recent watchlist alerts"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None or counter.watchlist is None:
        return {"alerts": [], "count": 0, "watchlist_enabled": False}

    alerts = counter.watchlist.get_recent_alerts(since, unacknowledged)
    return {
        "alerts": alerts,
        "count": len(alerts),
        "watchlist_enabled": True,
        "watchlist_names": counter.watchlist.get_watchlist_names()
    }


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Acknowledge a specific alert"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None or counter.watchlist is None:
        raise HTTPException(status_code=503, detail="Watchlist not initialized")

    success = counter.watchlist.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/alerts/acknowledge-all")
async def acknowledge_all_alerts():
    """Acknowledge all pending alerts"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None or counter.watchlist is None:
        raise HTTPException(status_code=503, detail="Watchlist not initialized")

    count = counter.watchlist.acknowledge_all()
    return {"status": "acknowledged", "count": count}


@app.post("/watchlist/reload")
async def reload_watchlist():
    """Reload watchlist from folder (after adding new photos)"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if counter is None or counter.watchlist is None:
        raise HTTPException(status_code=503, detail="Watchlist not initialized")

    counter.watchlist.reload_watchlist()
    names = counter.watchlist.get_watchlist_names()
    return {"status": "reloaded", "count": len(names), "names": names}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dual Camera Face Recognition People Counter API",
        "version": "2.0.0",
        "endpoints": {
            "/stats": "Get current statistics",
            "/": "This help message"
        },
        "info": "System uses two cameras - entry and exit - to track people by face recognition"
    }

async def sync_faces_from_peer(client: httpx.AsyncClient):
    """Pull any missing faces from the peer"""
    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
    if not counter:
        return 0

    try:
        response = await client.get(f"{PEER_URL}/sync/faces/all", timeout=5.0)
        if response.status_code != 200:
            return 0

        data = response.json()
        faces = data.get("faces", [])
        imported = 0
        skipped_embedding = 0

        for face in faces:
            person_id = face.get("person_id")
            name = face.get("name")
            embedding_list = face.get("embedding")

            if not person_id or not embedding_list:
                continue

            # Skip if we already have this exact person_id
            if person_id in counter.known_embeddings:
                continue

            # Normalize embedding
            embedding = np.array(embedding_list, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Check if this face matches an existing person by embedding similarity
            existing_match = counter._find_matching_person_by_embedding(embedding)
            if existing_match:
                logger.debug(f"PEER SYNC: Skipping {person_id} - matches existing {existing_match} by embedding")
                skipped_embedding += 1
                continue

            cursor = counter.db_conn.cursor()

            # Check if person exists in DB
            cursor.execute("SELECT 1 FROM persons WHERE person_id = ?", (person_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO persons (person_id, name, consent_ts)
                    VALUES (?, ?, ?)
                """, (person_id, name, datetime.now().isoformat()))

                cursor.execute("""
                    INSERT INTO faces (person_id, embedding, created_ts)
                    VALUES (?, ?, ?)
                """, (person_id, pickle.dumps(embedding), datetime.now().isoformat()))

                counter.db_conn.commit()

            # Add to memory
            counter.known_embeddings[person_id].append(embedding)
            counter.person_names[person_id] = name

            # Update visitor index if needed
            if name and name.startswith("Visitor "):
                try:
                    idx = int(name.split(" ")[1])
                    if idx >= counter.next_visitor_idx:
                        counter.next_visitor_idx = idx + 1
                except ValueError:
                    pass

            imported += 1

        if imported > 0 or skipped_embedding > 0:
            logger.info(f"PEER SYNC: Imported {imported} new faces, skipped {skipped_embedding} duplicates by embedding")
            if imported > 0:
                counter.update_stats()

        return imported

    except Exception as e:
        logger.debug(f"PEER SYNC: Error during continuous sync: {e}")
        return 0


async def monitor_peer_connection():
    """Background task to check peer connection and sync faces"""
    while True:
        if not PEER_URL:
            with stats_lock:
                global_stats["peer_status"] = "disabled"
            await asyncio.sleep(60)
            continue

        status = "error"
        data = None

        try:
            async with httpx.AsyncClient() as client:
                # Get peer stats
                response = await client.get(f"{PEER_URL}/stats", timeout=2.0)
                if response.status_code == 200:
                    status = "connected"
                    data = response.json()

                    # Sync faces if peer has more visitors than we know about
                    peer_visitors = data.get("unique_visitors", 0)
                    counter = COUNTER_INSTANCE or getattr(app.state, "counter", None)
                    local_known = len(counter.known_embeddings) if counter else 0

                    # Always try to sync - peer might have faces we don't
                    await sync_faces_from_peer(client)

        except Exception as e:
            logger.debug(f"Peer connection error: {e}")

        with stats_lock:
            global_stats["peer_status"] = status
            global_stats["peer_data"] = data

        await asyncio.sleep(5)  # Sync every 5 seconds

def run_api(port=8000):
    """Run FastAPI server in a separate thread"""
    # Add startup event for background task
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(monitor_peer_connection())

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

def signal_handler(sig, frame):
    """Handle shutdown signal"""
    logger.info("Shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    global COUNTER_INSTANCE
    parser = argparse.ArgumentParser(description='Dual Camera Face Recognition People Counter')
    parser.add_argument('--entry-cam', type=str, default="0",
                       help='Entry camera source (index like 0 or RTSP URL)')
    parser.add_argument('--exit-cam', type=str, default="1",
                       help='Exit camera source (index like 1 or RTSP URL)')
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
    port = int(os.getenv("PORT", 8000))
    api_thread = Thread(target=run_api, args=(port,), daemon=True)
    api_thread.start()
    logger.info(f"FastAPI server started at http://localhost:{port}")
    
    # Start dual camera counter
    # Start people counter
    counter = DualCameraCounter(
        entry_cam_id=args.entry_cam,
        exit_cam_id=args.exit_cam,
        split_screen=args.split_screen
    )
    app.state.counter = counter # Expose to API
    COUNTER_INSTANCE = counter
    
    # Update initial stats
    counter.update_stats()
    
    # Run the dual camera system
    counter.run_dual_cameras()

if __name__ == "__main__":
    main()

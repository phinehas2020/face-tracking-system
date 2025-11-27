"""
Watchlist Manager for Face Tracking System
==========================================

Drop photos into the watchlist/ folder with filenames like:
- John_Smith.jpg
- Jane_Doe.png

The system will alert when these people are detected.
"""

import os
import logging
import pickle
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from threading import Lock

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Watchlist matching threshold (slightly lower than normal to catch more matches)
WATCHLIST_MATCH_THRESHOLD = 0.35


class WatchlistManager:
    """
    Manages a folder of photos for people to watch for.
    Extracts face embeddings and checks incoming faces against the watchlist.
    """

    def __init__(self, watchlist_dir: Path, db_path: Path, face_app):
        self.watchlist_dir = Path(watchlist_dir)
        self.db_path = db_path
        self.face_app = face_app
        self.lock = Lock()

        # name -> (embedding, photo_path)
        self.watchlist: Dict[str, Tuple[np.ndarray, str]] = {}

        # Track recent alerts to avoid spamming (name -> last_alert_time)
        self.recent_alerts: Dict[str, float] = {}
        self.alert_cooldown = 60  # Seconds before same person can trigger another alert

        # Ensure directories exist
        self.watchlist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database table
        self._init_database()

        # Load watchlist
        self.reload_watchlist()

    def _init_database(self):
        """Create watchlist_alerts table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                similarity REAL NOT NULL,
                photo_path TEXT,
                detected_at REAL NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_time ON watchlist_alerts(detected_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_ack ON watchlist_alerts(acknowledged)")
        conn.commit()
        conn.close()
        logger.info("Watchlist alerts table initialized")

    def reload_watchlist(self):
        """Scan watchlist folder and extract face embeddings from photos."""
        with self.lock:
            self.watchlist.clear()

            if not self.watchlist_dir.exists():
                logger.info("Watchlist directory does not exist, creating it")
                self.watchlist_dir.mkdir(parents=True, exist_ok=True)
                return

            # Supported image extensions
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

            for file_path in self.watchlist_dir.iterdir():
                if file_path.suffix.lower() not in extensions:
                    continue

                # Extract name from filename (replace underscores with spaces)
                name = file_path.stem.replace('_', ' ')

                try:
                    # Load image
                    image = cv2.imread(str(file_path))
                    if image is None:
                        logger.warning(f"Could not load image: {file_path}")
                        continue

                    # Convert to RGB for face detection
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect faces
                    faces = self.face_app.get(image_rgb)

                    if len(faces) == 0:
                        logger.warning(f"No face found in watchlist photo: {file_path}")
                        continue

                    if len(faces) > 1:
                        logger.warning(f"Multiple faces in {file_path}, using largest")

                    # Use largest face
                    largest_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )

                    # Normalize embedding
                    embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)

                    self.watchlist[name] = (embedding, str(file_path))
                    logger.info(f"Loaded watchlist person: {name}")

                except Exception as e:
                    logger.error(f"Error processing watchlist photo {file_path}: {e}")

            logger.info(f"Watchlist loaded: {len(self.watchlist)} people")

    def check_face(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Check if a face embedding matches anyone on the watchlist.
        Returns (name, similarity) if match found, None otherwise.
        """
        if embedding is None or len(self.watchlist) == 0:
            return None

        with self.lock:
            best_match = None
            best_similarity = -1.0

            for name, (watchlist_embedding, _) in self.watchlist.items():
                similarity = float(np.dot(embedding, watchlist_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name

            if best_match and best_similarity >= WATCHLIST_MATCH_THRESHOLD:
                return (best_match, best_similarity)

            return None

    def record_alert(self, name: str, similarity: float) -> bool:
        """
        Record a watchlist alert if not in cooldown period.
        Returns True if alert was recorded, False if in cooldown.
        """
        now = time.time()

        # Check cooldown
        last_alert = self.recent_alerts.get(name, 0)
        if now - last_alert < self.alert_cooldown:
            return False

        # Update cooldown tracker
        self.recent_alerts[name] = now

        # Get photo path
        photo_path = None
        with self.lock:
            if name in self.watchlist:
                _, photo_path = self.watchlist[name]

        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO watchlist_alerts (name, similarity, photo_path, detected_at)
                VALUES (?, ?, ?, ?)
            """, (name, similarity, photo_path, now))
            conn.commit()
            conn.close()

            logger.warning(f"WATCHLIST ALERT: {name} detected (similarity: {similarity:.2f})")
            return True

        except Exception as e:
            logger.error(f"Error recording watchlist alert: {e}")
            return False

    def get_recent_alerts(self, since_seconds: int = 3600, unacknowledged_only: bool = True) -> List[dict]:
        """Get recent watchlist alerts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = time.time() - since_seconds

            if unacknowledged_only:
                cursor.execute("""
                    SELECT id, name, similarity, photo_path, detected_at, acknowledged
                    FROM watchlist_alerts
                    WHERE detected_at > ? AND acknowledged = 0
                    ORDER BY detected_at DESC
                    LIMIT 50
                """, (cutoff,))
            else:
                cursor.execute("""
                    SELECT id, name, similarity, photo_path, detected_at, acknowledged
                    FROM watchlist_alerts
                    WHERE detected_at > ?
                    ORDER BY detected_at DESC
                    LIMIT 50
                """, (cutoff,))

            rows = cursor.fetchall()
            conn.close()

            alerts = []
            for row in rows:
                alerts.append({
                    "id": row[0],
                    "name": row[1],
                    "similarity": row[2],
                    "photo_path": row[3],
                    "detected_at": row[4],
                    "detected_at_iso": datetime.fromtimestamp(row[4]).isoformat(),
                    "acknowledged": bool(row[5])
                })

            return alerts

        except Exception as e:
            logger.error(f"Error getting watchlist alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE watchlist_alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return affected > 0
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False

    def acknowledge_all(self) -> int:
        """Mark all alerts as acknowledged. Returns count of affected rows."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE watchlist_alerts SET acknowledged = 1 WHERE acknowledged = 0")
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return affected
        except Exception as e:
            logger.error(f"Error acknowledging all alerts: {e}")
            return 0

    def get_watchlist_names(self) -> List[str]:
        """Get list of all people on the watchlist."""
        with self.lock:
            return list(self.watchlist.keys())

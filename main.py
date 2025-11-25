#!/usr/bin/env python3
"""
Face Recognition People Counter with Dwell Time Tracking
=========================================================

A real-time people counting system using YOLO v8/v11 tracking with ByteTrack,
InsightFace for face recognition, and FastAPI for analytics dashboard.

Features:
- Real-time people detection and tracking using YOLO
- Face embedding extraction using InsightFace (ArcFace)
- Direction detection (in/out) when crossing virtual line
- Unique visitor counting via face recognition
- Dwell time calculation
- SQLite database for persistent storage
- FastAPI dashboard for real-time statistics

Usage:
------
1. First, set up the environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Initialize the database and load consent gallery:
   ```bash
   python database_init.py
   ```

3. Run the main application:
   ```bash
   python main.py
   ```

4. View statistics dashboard:
   Open browser to http://localhost:8000/stats

5. Test with curl:
   ```bash
   curl http://localhost:8000/stats
   ```

Configuration:
--------------
- Virtual line position: Adjust DOOR_Y_POSITION (default: middle of frame)
- Camera source: Change source parameter in main() (default: 0 for webcam)
- Face matching threshold: Adjust FACE_MATCH_THRESHOLD (default: 0.4)

Requirements:
-------------
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- Webcam or video file for input

Author: Face Tracking System
Date: 2025
"""

import os
import sys
import time
import sqlite3
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
from collections import defaultdict, deque
from threading import Thread, Lock
import signal

import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "people_tracking.db"
FACE_MATCH_THRESHOLD = 0.4  # Cosine similarity threshold for face matching
DOOR_Y_POSITION = 0.5  # Virtual line position (0.5 = middle of frame)
TRACK_HISTORY_SIZE = 30  # Number of frames to keep in track history
MIN_FACE_SIZE = 50  # Minimum face size in pixels
FACE_CAPTURE_INTERVAL = 10  # Capture face every N frames for each track
PEER_URL = os.getenv("PEER_URL")  # URL of the peer instance (e.g., "http://other-ip:8000")

# Global variables
app = FastAPI(title="People Counter API", version="1.0.0")
stats_lock = Lock()
global_stats = {
    "unique_visitors": 0,
    "total_in": 0,
    "total_out": 0,
    "avg_dwell_minutes": 0.0,
    "current_occupancy": 0,
    "peer_status": "disabled",  # disabled, connected, error
    "peer_url": PEER_URL
}

class StatsResponse(BaseModel):
    unique_visitors: int
    avg_dwell_minutes: float
    total_in: int
    total_out: int
    current_occupancy: int
    peer_status: str
    timestamp: str

class SyncFaceRequest(BaseModel):
    person_id: str
    name: str
    embedding: List[float]

class PeopleCounter:
    def __init__(self):
        """Initialize the people counter system"""
        self.yolo_model = None
        self.face_app = None
        self.db_conn = None
        self.known_embeddings = {}  # person_id -> embedding
        self.track_histories = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_SIZE))
        self.track_faces = {}  # track_id -> best face embedding
        self.track_persons = {}  # track_id -> person_id
        self.track_crossed = {}  # track_id -> bool (has crossed line)
        self.track_last_capture = {}  # track_id -> frame_number
        self.frame_count = 0
        self.door_y = 0  # Will be set based on frame height
        
        # Initialize components
        self._init_yolo()
        self._init_face_analyzer()
        self._init_database()
        self._load_known_faces()
    
    def _init_yolo(self):
        """Initialize YOLO model for tracking"""
        try:
            # Use YOLOv8n for speed, upgrade to YOLOv8m or YOLOv8l for better accuracy
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
        logger.info(f"Connected to database: {DB_PATH}")
    
    def _load_known_faces(self):
        """Load known face embeddings from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT p.person_id, p.name, f.embedding
            FROM persons p
            JOIN faces f ON p.person_id = f.person_id
        """)
        
        for person_id, name, embedding_blob in cursor.fetchall():
            embedding = pickle.loads(embedding_blob)
            self.known_embeddings[person_id] = embedding
            logger.info(f"Loaded face embedding for {name} (ID: {person_id})")
        
        logger.info(f"Loaded {len(self.known_embeddings)} known face embeddings")
    
    def extract_face_embedding(self, image, bbox):
        """Extract face embedding from a detected person"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding around the detected person for better face detection
            h, w = image.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            # Crop person region
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            # Convert BGR to RGB
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the person crop
            faces = self.face_app.get(person_rgb)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face (usually the most prominent)
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Check minimum face size
            face_width = largest_face.bbox[2] - largest_face.bbox[0]
            face_height = largest_face.bbox[3] - largest_face.bbox[1]
            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                return None
            
            # Normalize embedding
            embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)
            
            return embedding
        
        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
            return None
    
    def match_face(self, embedding):
        """Match face embedding against known faces"""
        if not self.known_embeddings:
            return None
        
        best_match = None
        best_similarity = -1
        
        for person_id, known_embedding in self.known_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, known_embedding)
            
            if similarity > best_similarity and similarity > FACE_MATCH_THRESHOLD:
                best_similarity = similarity
                best_match = person_id
        
        if best_match:
            logger.debug(f"Face matched to {best_match} with similarity {best_similarity:.3f}")
        
        return best_match
    
    def detect_crossing(self, track_id, current_y):
        """Detect if a person crossed the virtual line and determine direction"""
        history = self.track_histories[track_id]
        
        if len(history) < 2:
            return None
        
        # Get previous and current positions relative to door
        prev_y = history[-2]
        was_above = prev_y < self.door_y
        is_above = current_y < self.door_y
        
        # Check if crossed
        if was_above and not is_above:
            return 'in'  # Crossed from top to bottom
        elif not was_above and is_above:
            return 'out'  # Crossed from bottom to top
        
        return None
    
    def log_crossing(self, person_id, direction, timestamp):
        """Log crossing event to database"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO crossings (person_id, direction, t_cross)
            VALUES (?, ?, ?)
        """, (person_id, direction, timestamp))
        self.db_conn.commit()
        
        logger.info(f"Crossing logged: {person_id} went {direction} at {timestamp}")
    
    def create_new_person(self, track_id, embedding):
        """Create a new person entry when no match is found"""
        # Generate unique person ID
        person_id = f"unknown_{int(time.time())}_{track_id}"
        name = f"Person {track_id}"
        
        cursor = self.db_conn.cursor()
        
        # Insert person
        cursor.execute("""
            INSERT INTO persons (person_id, name, consent_ts)
            VALUES (?, ?, ?)
        """, (person_id, name, datetime.now().isoformat()))
        
        # Store embedding
        if embedding is not None:
            embedding_blob = pickle.dumps(embedding)
            cursor.execute("""
                INSERT INTO faces (person_id, embedding, created_ts)
                VALUES (?, ?, ?)
            """, (person_id, embedding_blob, datetime.now().isoformat()))
            
            # Add to known embeddings for future matching
            self.known_embeddings[person_id] = embedding
        
        self.db_conn.commit()
        logger.info(f"Created new person: {person_id}")
        
        # Sync to peer if configured
        if PEER_URL:
            asyncio.create_task(self.sync_to_peer(person_id, name, embedding))
        
        return person_id

    async def sync_to_peer(self, person_id, name, embedding):
        """Send new person to peer instance"""
        try:
            if embedding is None:
                return
                
            payload = {
                "person_id": person_id,
                "name": name,
                "embedding": embedding.tolist()
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{PEER_URL}/sync/face", json=payload, timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"Successfully synced {person_id} to peer")
                else:
                    logger.warning(f"Failed to sync to peer: {response.status_code} {response.text}")
                    
        except Exception as e:
            logger.error(f"Error syncing to peer: {e}")
    
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
            
            # Current occupancy
            global_stats["current_occupancy"] = global_stats["total_in"] - global_stats["total_out"]
            
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
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Set door position if not set
        if self.door_y == 0:
            self.door_y = int(height * DOOR_Y_POSITION)
        
        # Run YOLO tracking
        results = self.yolo_model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            classes=[0],  # Only detect persons (class 0 in COCO)
            verbose=False
        )
        
        # Draw virtual line
        cv2.line(frame, (0, self.door_y), (width, self.door_y), (0, 255, 0), 2)
        cv2.putText(frame, "ENTRY/EXIT LINE", (10, self.door_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            for i, (box, track_id) in enumerate(zip(boxes.xyxy, boxes.id)):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) // 2
                
                # Update track history
                self.track_histories[track_id].append(center_y)
                
                # Draw bounding box
                color = (255, 0, 0)  # Blue by default
                label = f"ID: {track_id}"
                
                # Check if we have identified this person
                if track_id in self.track_persons:
                    person_id = self.track_persons[track_id]
                    label = f"{person_id[:10]}"
                    color = (0, 255, 0)  # Green for identified
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Try to extract face periodically
                should_capture = (
                    track_id not in self.track_faces or
                    track_id not in self.track_last_capture or
                    self.frame_count - self.track_last_capture.get(track_id, 0) > FACE_CAPTURE_INTERVAL
                )
                
                if should_capture:
                    embedding = self.extract_face_embedding(frame, box)
                    if embedding is not None:
                        # Store or update the best face for this track
                        self.track_faces[track_id] = embedding
                        self.track_last_capture[track_id] = self.frame_count
                        
                        # Try to match face if we haven't identified this track yet
                        if track_id not in self.track_persons:
                            person_id = self.match_face(embedding)
                            if person_id:
                                self.track_persons[track_id] = person_id
                                logger.info(f"Track {track_id} identified as {person_id}")
                
                # Check for line crossing
                if track_id not in self.track_crossed:
                    direction = self.detect_crossing(track_id, center_y)
                    
                    if direction:
                        self.track_crossed[track_id] = True
                        
                        # Get or create person ID
                        if track_id in self.track_persons:
                            person_id = self.track_persons[track_id]
                        else:
                            # Create new person with face embedding if available
                            embedding = self.track_faces.get(track_id)
                            person_id = self.create_new_person(track_id, embedding)
                            self.track_persons[track_id] = person_id
                        
                        # Log crossing
                        self.log_crossing(person_id, direction, time.time())
                        
                        # Update stats
                        self.update_stats()
                        
                        # Visual feedback
                        cv2.putText(frame, f"{direction.upper()}!", 
                                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
        
        # Display statistics
        with stats_lock:
            stats_text = [
                f"Unique Visitors: {global_stats['unique_visitors']}",
                f"Total In: {global_stats['total_in']}",
                f"Total Out: {global_stats['total_out']}",
                f"Occupancy: {global_stats['current_occupancy']}",
                f"Avg Dwell: {global_stats['avg_dwell_minutes']:.1f} min"
            ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def run(self, source=0):
        """Run the people counter"""
        # Open video source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return
        
        logger.info(f"Started processing video from source: {source}")
        logger.info("Press 'q' to quit, 's' to save snapshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("People Counter - Face Recognition", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Saved snapshot: {filename}")
        
        finally:
            cap.release()
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
            peer_status=global_stats["peer_status"],
            timestamp=datetime.now().isoformat()
        )

@app.post("/sync/face")
async def sync_face(request: SyncFaceRequest):
    """Receive a new face from peer"""
    try:
        # Validate and normalize embedding
        embedding_arr = np.asarray(request.embedding, dtype=np.float32)
        if embedding_arr.ndim != 1:
            raise HTTPException(status_code=400, detail="Embedding must be a 1D array")
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
        if cursor.fetchone():
            conn.close()
            return {"status": "exists"}
            
        # Insert person
        cursor.execute("""
            INSERT INTO persons (person_id, name, consent_ts)
            VALUES (?, ?, ?)
        """, (request.person_id, request.name, datetime.now().isoformat()))
        
        # Insert face
        cursor.execute("""
            INSERT INTO faces (person_id, embedding, created_ts)
            VALUES (?, ?, ?)
        """, (request.person_id, embedding_blob, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Update in-memory known embeddings (need to access the global counter instance somehow)
        # Since FastAPI and the loop are in different threads/contexts, this is tricky.
        # But we can use a global reference or just rely on the DB for next reload.
        # Ideally, we update the live counter.
        # Let's assume 'counter' is available globally or we can attach it to app.state.
        if hasattr(app.state, 'counter') and app.state.counter:
             app.state.counter.known_embeddings[request.person_id] = embedding
             logger.info(f"Received and loaded synced person: {request.name}")
        
        return {"status": "synced"}
        
    except Exception as e:
        logger.error(f"Error processing sync request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition People Counter API",
        "version": "1.0.0",
        "endpoints": {
            "/stats": "Get current statistics",
            "/": "This help message"
        }
    }

async def monitor_peer_connection():
    """Background task to check peer connection status"""
    while True:
        if not PEER_URL:
            with stats_lock:
                global_stats["peer_status"] = "disabled"
            await asyncio.sleep(60)
            continue

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{PEER_URL}/", timeout=2.0)
                status = "connected" if response.status_code == 200 else "error"
        except Exception:
            status = "error"
        
        with stats_lock:
            global_stats["peer_status"] = status
            
        await asyncio.sleep(30)

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
    
    # Start people counter
    counter = PeopleCounter()
    app.state.counter = counter # Expose to API
    
    # Update initial stats
    counter.update_stats()
    
    # Run with webcam (source=0)
    # For video file, use: counter.run("path/to/video.mp4")
    counter.run(source=0)

if __name__ == "__main__":
    main()

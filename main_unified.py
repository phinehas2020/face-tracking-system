#!/usr/bin/env python3
"""
Unified Inference Face Tracking System
======================================

Architecture: "The Funnel"
--------------------------
1. Capture Threads (x2): Read frames -> Push to Queue(maxsize=1).
   - If Queue full, DROP frame. This guarantees 0 lag.
2. Main Thread:
   - Pop frames from both queues.
   - Batch Inference: Run YOLO on both frames at once (if possible) or sequentially.
   - Sensor Fusion: Link Face (Cam A) with Appearance (Cam B).
   - Render UI.

"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
import sys
import os
import pickle
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

import argparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from threading import Thread, Lock
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
QUEUE_SIZE = 1  # Critical: Keep at 1 to force frame skipping

# Global Stats
stats_lock = Lock()
global_stats = {
    "unique_visitors": 0,
    "known_faces": 0,
    "total_in": 0,
    "total_out": 0,
    "avg_dwell_minutes": 0.0,
    "current_occupancy": 0,
    "peer_status": "disabled",
}

app = FastAPI(title="Unified Face Tracking API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

class StatsResponse(BaseModel):
    unique_visitors: int
    known_faces: int
    avg_dwell_minutes: float
    total_in: int
    total_out: int
    current_occupancy: int
    peer_status: str
    timestamp: str

# Configuration
MIN_FACE_SIZE = 20
APPEARANCE_MIN_AREA = 5000
FACE_MATCH_THRESHOLD = 0.6
EMBEDDING_EXPIRE_TIME = 3600.0
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_tracking.db"

class CaptureThread(threading.Thread):
    def __init__(self, camera_id, name):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.name = name
        self.queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.last_frame_time = 0
        
    def run(self):
        logger.info(f"Starting capture thread: {self.name}")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Resize if needed
            if frame.shape[1] != FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
            # Smart Queue: If full, drop the old frame and put new one? 
            # Actually, standard queue.put(block=False) raises Full.
            # We want to DROP the current frame if queue is full (consumer is busy).
            # OR drop the OLD frame in the queue?
            # Dropping the current frame is simplest and ensures we don't block.
            
            try:
                self.queue.put_nowait((frame, time.time()))
            except queue.Full:
                # Queue full means consumer is busy. Drop this frame.
                pass
                
        cap.release()
        logger.info(f"Stopped capture thread: {self.name}")

    def stop(self):
        self.stop_event.set()

class StateTracker:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.known_embeddings = defaultdict(list)
        self.person_names = {}
        self.temp_embeddings = {}
        self.person_last_seen = {}
        self.active_persons = set()
        self.appearance_signatures = {} # pid -> (sig, ts)
        self.next_visitor_idx = 1
        
        self._load_known_faces()
        
    def _load_known_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM persons WHERE name LIKE 'Visitor %'")
        max_idx = 0
        for (name,) in cursor.fetchall():
            try:
                idx = int(name.split(" ")[1])
                if idx > max_idx: max_idx = idx
            except: pass
        self.next_visitor_idx = max_idx + 1

        cursor.execute("""
            SELECT p.person_id, p.name, f.embedding
            FROM persons p
            JOIN faces f ON p.person_id = f.person_id
        """)
        for pid, name, blob in cursor.fetchall():
            emb = pickle.loads(blob)
            self.known_embeddings[pid].append(emb)
            self.person_names[pid] = name
            
    def match_face(self, embedding):
        best_match = None
        best_sim = -1.0
        
        # Check known
        for pid, embs in self.known_embeddings.items():
            for e in embs:
                sim = float(np.dot(embedding, e))
                if sim > best_sim:
                    best_sim = sim
                    best_match = pid
                    
        # Check temp
        now = time.time()
        to_del = []
        for tid, (temb, ts) in self.temp_embeddings.items():
            if now - ts > EMBEDDING_EXPIRE_TIME:
                to_del.append(tid)
                continue
            sim = float(np.dot(embedding, temb))
            if sim > best_sim:
                best_sim = sim
                best_match = tid
        
        for t in to_del: del self.temp_embeddings[t]
            
        if best_match and best_sim >= FACE_MATCH_THRESHOLD:
            return best_match
        return None

    def create_temporary_person(self, embedding):
        temp_id = f"temp_{int(time.time()*1000)}"
        self.temp_embeddings[temp_id] = (embedding, time.time())
        return temp_id

    def process_detection(self, embedding, appearance=None):
        if embedding is None: return None
        
        pid = self.match_face(embedding)
        if not pid:
            pid = self.create_temporary_person(embedding)
            
        # Update last seen & appearance
        self.person_last_seen[pid] = time.time()
        if appearance is not None:
            self.appearance_signatures[pid] = (appearance, time.time())
            
        self.active_persons.add(pid)
        return pid

    def process_side_view_detection(self, appearance):
        if appearance is None or not self.active_persons:
            return None
            
        best_pid = None
        best_sim = -1.0
        
        now = time.time()
        
        for pid in list(self.active_persons):
            if pid not in self.appearance_signatures: continue
            
            sig, ts = self.appearance_signatures[pid]
            if now - ts > 60: # Expire old signatures
                continue
                
            sim = float(np.dot(appearance, sig))
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
                
        if best_pid and best_sim > 0.7: # Threshold for appearance match
            self.person_last_seen[best_pid] = now
            return best_pid
            
        return None

    def update_stats(self):
        """Update global statistics"""
        cursor = self.conn.cursor()
        
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
            global_stats["known_faces"] = len(self.known_embeddings)
            
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

# --- API Endpoints ---
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    with stats_lock:
        return StatsResponse(
            unique_visitors=global_stats["unique_visitors"],
            known_faces=global_stats["known_faces"],
            avg_dwell_minutes=global_stats["avg_dwell_minutes"],
            total_in=global_stats["total_in"],
            total_out=global_stats["total_out"],
            current_occupancy=global_stats["current_occupancy"],
            peer_status=global_stats["peer_status"],
            timestamp=datetime.now().isoformat()
        )

def run_api(port=8000):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

# --- Helper Functions (Ported) ---
def crop_person_region(image, bbox, pad=20):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
    except: return None

def compute_appearance_signature(person_crop):
    if person_crop is None or person_crop.size == 0: return None
    h, w = person_crop.shape[:2]
    if h * w < APPEARANCE_MIN_AREA: return None
    try:
        resized = cv2.resize(person_crop, (96, 192), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        signature = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = np.linalg.norm(signature)
        return signature / norm if norm > 0 else None
    except: return None

def extract_face_embedding(face_app, image, person_crop):
    try:
        if person_crop is None or person_crop.size == 0: return None, None
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        faces = face_app.get(person_rgb)
        if not faces: return None, None
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        if largest_face.det_score < 0.50: return None, None
        embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)
        return embedding, largest_face.bbox
    except: return None, None

def main():
    import torch
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    parser = argparse.ArgumentParser(description="Unified Face Tracking System")
    parser.add_argument("--entry-cam", default=0, help="Camera ID for Entry (Face) View")
    parser.add_argument("--exit-cam", default=1, help="Camera ID for Side (Body) View")
    args, _ = parser.parse_known_args()

    # Handle numeric or string camera IDs
    def parse_cam(val):
        try: return int(val)
        except: return val

    cam_a_id = parse_cam(args.entry_cam)
    cam_b_id = parse_cam(args.exit_cam)

    # Initialize Models
    logger.info("Initializing Models...")
    yolo_model = YOLO('yolov8s.pt')
    if torch.backends.mps.is_available():
        yolo_model.to('mps')
        logger.info("YOLO using MPS")
    
    face_app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize Tracker
    tracker = StateTracker()
    tracker.update_stats()
    
    # Start API Server
    port = int(os.getenv("PORT", 8000))
    api_thread = Thread(target=run_api, args=(port,), daemon=True)
    api_thread.start()
    logger.info(f"API Server started on port {port}")
    
    # Start Capture Threads
    cam_a = CaptureThread(cam_a_id, "Camera A")
    cam_b = CaptureThread(cam_b_id, "Camera B")
    cam_a.start()
    cam_b.start()
    
    logger.info("System Started. Press 'q' to quit.")
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Fetch Frames (Non-blocking)
            frame_a, frame_b = None, None
            try: frame_a, _ = cam_a.queue.get_nowait()
            except queue.Empty: pass
            
            try: frame_b, _ = cam_b.queue.get_nowait()
            except queue.Empty: pass
            
            if frame_a is None and frame_b is None:
                time.sleep(0.001)
                continue
                
            # 2. Batch Inference (The "Funnel")
            # We can stack frames to run YOLO once if we have both
            frames_to_process = []
            if frame_a is not None: frames_to_process.append(frame_a)
            if frame_b is not None: frames_to_process.append(frame_b)
            
            detections_a = []
            detections_b = []
            
            if frames_to_process:
                # Run YOLO
                results = yolo_model(frames_to_process, verbose=False, classes=[0])
                
                # Process Results
                idx = 0
                if frame_a is not None:
                    res = results[idx]
                    idx += 1
                    if res.boxes:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = crop_person_region(frame_a, (x1, y1, x2, y2))
                            app = compute_appearance_signature(crop)
                            emb, fbox = extract_face_embedding(face_app, frame_a, crop)
                            
                            pid = tracker.process_detection(emb, app)
                            detections_a.append((x1, y1, x2, y2, pid))
                            tracker.update_stats()
                            
                if frame_b is not None:
                    res = results[idx]
                    if res.boxes:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = crop_person_region(frame_b, (x1, y1, x2, y2))
                            app = compute_appearance_signature(crop)
                            
                            pid = tracker.process_side_view_detection(app)
                            detections_b.append((x1, y1, x2, y2, pid))
                            tracker.update_stats()

            # 3. Render
            if frame_a is not None:
                for x1, y1, x2, y2, pid in detections_a:
                    color = (0, 255, 0) if pid else (0, 0, 255)
                    cv2.rectangle(frame_a, (x1, y1), (x2, y2), color, 2)
                    if pid:
                        name = tracker.person_names.get(pid, pid)
                        cv2.putText(frame_a, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Camera A (Face)", frame_a)
                
            if frame_b is not None:
                for x1, y1, x2, y2, pid in detections_b:
                    color = (255, 165, 0) if pid else (128, 128, 128)
                    cv2.rectangle(frame_b, (x1, y1), (x2, y2), color, 2)
                    if pid:
                        name = tracker.person_names.get(pid, pid)
                        cv2.putText(frame_b, f"{name} (Side)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Camera B (Side)", frame_b)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # FPS Calculation
            proc_time = time.time() - start_time
            # logger.info(f"Loop time: {proc_time*1000:.1f}ms")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        cam_a.stop()
        cam_b.stop()
        cam_a.join()
        cam_b.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

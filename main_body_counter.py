#!/usr/bin/env python3
"""
Side-View People Counter (Body Tracking)
========================================

This system uses a single camera positioned from the side to count people 
crossing a virtual line. It acts as a failsafe/verification for the 
main face recognition system.

It uses:
- YOLOv8 for person detection (body only).
- ByteTrack (via Supervision or custom logic) for tracking.
- A simple line-crossing algorithm.

Usage:
------
python main_body_counter.py --cam <camera_id> --line-pos 0.5

Author: Face Tracking System
Date: 2025
"""

import cv2
import time
import sqlite3
import logging
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [BODY] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_tracking.db"
RECORDING_DIR = BASE_DIR / "recordings"

# Import config for recording settings
from config import ENABLE_RECORDING
# Import video recorder if enabled
try:
    from video_recorder import ThreadedVideoRecorder
except ImportError:
    ThreadedVideoRecorder = None

class BodyCounter:
    def __init__(self, camera_id=0, line_position=0.5, sensitivity=50):
        """
        Initialize the body counter.
        
        Args:
            camera_id: Camera index or URL.
            line_position: Vertical line position (0.0 to 1.0) across the image width.
                           0.5 means the line is in the middle of the screen.
            sensitivity: Minimum distance (pixels) a track must move to be counted.
        """
        self.camera_id = self._normalize_source(camera_id)
        self.line_rel_pos = line_position
        self.sensitivity = sensitivity
        
        self.model = None
        self.db_conn = None
        self.recorder = None
        
        # Tracking state
        # track_id -> {'path': deque([(x,y,t)...]), 'counted': bool, 'direction': None}
        self.tracks = {}
        self.max_track_age = 2.0  # Seconds to keep a lost track
        self.track_history_len = 30
        
        self.stats = {
            'in': 0,
            'out': 0
        }
        
        self._init_model()
        self._init_database()

    def _normalize_source(self, value):
        try:
            return int(value)
        except ValueError:
            return value

    def _init_model(self):
        logger.info("Loading YOLOv8n model for body detection...")
        # Using nano model for speed as we just need person blobs
        self.model = YOLO('yolov8n.pt') 
        logger.info("Model loaded.")

    def _init_database(self):
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Create table if it doesn't exist (redundant if init script ran, but safe)
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS body_crossings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
                t_cross REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.db_conn.commit()

    def _init_recorder(self, width, height):
        if ENABLE_RECORDING and ThreadedVideoRecorder:
            self.recorder = ThreadedVideoRecorder(
                RECORDING_DIR, "body_cam", width, height
            )

    def log_crossing(self, direction, timestamp):
        """Log crossing event to database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO body_crossings (direction, t_cross)
                VALUES (?, ?)
            """, (direction, timestamp))
            self.db_conn.commit()
            self.stats[direction] += 1
            logger.info(f"Body Counted {direction.upper()} at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Database error: {e}")

    def process_tracks(self, detections, frame_width):
        """
        Update tracks and check for line crossings.
        detections: list of (x1, y1, x2, y2, track_id)
        """
        current_time = time.time()
        line_x = int(frame_width * self.line_rel_pos)
        
        active_ids = set()
        
        for x1, y1, x2, y2, track_id in detections:
            active_ids.add(track_id)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'path': deque(maxlen=self.track_history_len),
                    'counted': False,
                    'last_seen': current_time
                }
            
            track = self.tracks[track_id]
            track['last_seen'] = current_time
            track['path'].append((center_x, center_y, current_time))
            
            # Check crossing if not yet counted and we have history
            if not track['counted'] and len(track['path']) >= 2:
                # Get start and end of current trajectory
                start_x = track['path'][0][0]
                curr_x = center_x
                
                # Check if crossed the line
                # We assume Left -> Right is IN, Right -> Left is OUT
                # Adjust based on your physical setup
                if start_x < line_x and curr_x > line_x:
                    if curr_x - start_x > self.sensitivity: # Ensure significant movement
                        self.log_crossing('in', current_time)
                        track['counted'] = True
                elif start_x > line_x and curr_x < line_x:
                    if start_x - curr_x > self.sensitivity:
                        self.log_crossing('out', current_time)
                        track['counted'] = True

        # Clean up old tracks
        expired = [
            tid for tid, t in self.tracks.items() 
            if current_time - t['last_seen'] > self.max_track_age
        ]
        for tid in expired:
            del self.tracks[tid]

    def run(self):
        logger.info(f"Starting Body Counter on camera {self.camera_id}")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {self.camera_id}")
            return

        # Initialize recorder once we know the frame size
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._init_recorder(width, height)

        logger.info(f"Line position: x={int(width * self.line_rel_pos)} ({(self.line_rel_pos*100):.0f}%)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                if self.recorder:
                    self.recorder.record_frame(frame)
                
                # Run YOLO tracking
                # persist=True enables the built-in BoT-SORT/ByteTrack in YOLOv8
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=[0], # Person class only
                    verbose=False,
                    conf=0.3,
                    tracker="bytetrack.yaml"
                )
                
                detections = []
                if results and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    
                    for box, track_id in zip(boxes, track_ids):
                        detections.append((*box, track_id))
                
                self.process_tracks(detections, width)
                
                # Draw visualization
                line_x = int(width * self.line_rel_pos)
                cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
                cv2.putText(frame, "INSIDE", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "OUTSIDE", (line_x - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw stats
                cv2.putText(frame, f"IN: {self.stats['in']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"OUT: {self.stats['out']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw detections
                for x1, y1, x2, y2, tid in detections:
                    xc, yc = int((x1+x2)/2), int((y1+y2)/2)
                    color = (0, 255, 0) if self.tracks.get(tid, {}).get('counted') else (200, 200, 200)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, str(tid), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw trail
                    if tid in self.tracks:
                        path = list(self.tracks[tid]['path'])
                        for i in range(1, len(path)):
                            pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                            pt2 = (int(path[i][0]), int(path[i][1]))
                            cv2.line(frame, pt1, pt2, color, 1)

                cv2.imshow("Body Counter (Side View)", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.recorder:
                self.recorder.stop()
            cap.release()
            cv2.destroyAllWindows()
            if self.db_conn:
                self.db_conn.close()

def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', default="2", help="Camera index or URL for side view")
    parser.add_argument('--line-pos', type=float, default=0.5, help="Line position (0.0-1.0)")
    args = parser.parse_args()
    
    counter = BodyCounter(camera_id=args.cam, line_position=args.line_pos)
    counter.run()

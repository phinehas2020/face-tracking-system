import cv2
import threading
import queue
import time
import os
import logging
from datetime import datetime
from pathlib import Path

from config import RECORDING_SEGMENT_DURATION, RECORDING_FRAME_RATE

logger = logging.getLogger(__name__)

class ThreadedVideoRecorder:
    """
    A high-performance, non-blocking video recorder.
    It receives frames via a queue and writes them to disk in a background thread.
    It automatically handles file rotation (creating new files every N minutes).
    """
    def __init__(self, base_dir, camera_name, width, height, fps=RECORDING_FRAME_RATE):
        self.base_dir = Path(base_dir)
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        
        self.queue = queue.Queue(maxsize=300) # Buffer ~10-20 seconds of video if disk is slow
        self.stop_event = threading.Event()
        self.writer = None
        self.current_file_path = None
        self.file_start_time = 0
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started video recorder for '{camera_name}' ({width}x{height} @ {fps}fps)")

    def record_frame(self, frame):
        """
        Add a frame to the recording queue.
        This method returns immediately to avoid blocking the main application loop.
        """
        if self.stop_event.is_set():
            return
        
        try:
            # If queue is full, drop the oldest frame (FIFO) to keep latency low
            # This is better than blocking the main thread.
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.queue.put(frame, block=False)
        except queue.Full:
            pass # Should be handled above, but safety first

    def stop(self):
        """Signal the recorder to stop and wait for it to finish writing the queue."""
        self.stop_event.set()
        self.thread.join()
        self._close_writer()
        logger.info(f"Stopped video recorder for '{self.camera_name}'")

    def _record_loop(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Wait for a frame with a timeout so we can check stop_event periodically
                frame = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._check_rotation()
            
            if self.writer:
                try:
                    # Resize if the incoming frame doesn't match the writer config
                    # (Rare, but protects against crashes if camera config changes mid-stream)
                    h, w = frame.shape[:2]
                    if w != self.width or h != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                        
                    self.writer.write(frame)
                except Exception as e:
                    logger.error(f"Error writing frame to video: {e}")

    def _check_rotation(self):
        """Rotates the video file if the segment duration has passed."""
        now = time.time()
        if self.writer is None or (now - self.file_start_time > RECORDING_SEGMENT_DURATION):
            self._start_new_segment()

    def _start_new_segment(self):
        self._close_writer()
        
        # Create daily subfolder: recordings/2023-10-27/
        today_str = datetime.now().strftime('%Y-%m-%d')
        daily_dir = self.base_dir / today_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: camera_name_HH-MM-SS.mp4
        timestamp = datetime.now().strftime('%H-%M-%S')
        filename = f"{self.camera_name}_{timestamp}.mp4"
        self.current_file_path = str(daily_dir / filename)
        
        # Initialize VideoWriter
        # 'mp4v' is widely supported. 'avc1' (H.264) is better but depends on system codecs.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.writer = cv2.VideoWriter(
            self.current_file_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        self.file_start_time = time.time()
        logger.info(f"Started new video segment: {self.current_file_path}")

    def _close_writer(self):
        if self.writer:
            self.writer.release()
            self.writer = None

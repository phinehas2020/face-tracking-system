import cv2
import threading
import queue
import time
import os
import subprocess
import logging
import atexit
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    RECORDING_SEGMENT_DURATION,
    RECORDING_FRAME_RATE,
    RECORDING_COMPRESS_CRF,
    RECORDING_COMPRESS_PRESET
)

logger = logging.getLogger(__name__)

# Global registry of active recorders for cleanup
_active_recorders = []
_cleanup_lock = threading.Lock()


def _cleanup_all_recorders():
    """Cleanup function called on exit to ensure all files are properly closed."""
    with _cleanup_lock:
        for recorder in _active_recorders:
            try:
                recorder.stop()
            except Exception as e:
                logger.error(f"Error stopping recorder: {e}")


# Register cleanup on normal exit
atexit.register(_cleanup_all_recorders)


def repair_video_file(file_path: str) -> bool:
    """
    Attempt to repair a corrupted video file using ffmpeg.
    Returns True if repair was successful.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    # Check if file is too small (likely corrupt/empty)
    if file_path.stat().st_size < 10000:  # Less than 10KB
        logger.warning(f"File too small, likely empty: {file_path}")
        return False

    output_path = file_path.with_suffix('.repaired.mp4')

    try:
        cmd = [
            'ffmpeg',
            '-err_detect', 'ignore_err',  # Ignore errors
            '-i', str(file_path),
            '-c', 'copy',  # Just remux, don't re-encode
            '-movflags', '+faststart',
            '-y',
            '-loglevel', 'error',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            # Replace original with repaired
            file_path.unlink()
            output_path.rename(file_path)
            logger.info(f"Repaired video file: {file_path}")
            return True
        else:
            if output_path.exists():
                output_path.unlink()
            return False

    except FileNotFoundError:
        logger.warning("ffmpeg not found - cannot repair video")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout repairing {file_path}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Error repairing {file_path}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


class VideoCompressor:
    """
    Background video compressor that watches for completed segments
    and compresses them using ffmpeg (H.264).
    """
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.stop_event = threading.Event()
        self.pending_files = queue.Queue()

        self.thread = threading.Thread(target=self._compress_loop, daemon=True)
        self.thread.start()
        logger.info("Started background video compressor")

    def queue_for_compression(self, file_path):
        """Add a completed video file to the compression queue."""
        self.pending_files.put(file_path)
        logger.info(f"Queued for compression: {file_path}")

    def stop(self):
        """Stop the compressor and finish pending work."""
        self.stop_event.set()
        self.thread.join(timeout=60)  # Wait up to 60s for current compression
        logger.info("Stopped video compressor")

    def _compress_loop(self):
        while not self.stop_event.is_set():
            try:
                # Wait for a file to compress
                file_path = self.pending_files.get(timeout=5.0)
            except queue.Empty:
                continue

            self._compress_file(file_path)

    def _compress_file(self, input_path):
        """Compress a video file using ffmpeg H.264."""
        input_path = Path(input_path)

        if not input_path.exists():
            logger.warning(f"File not found for compression: {input_path}")
            return

        # Create compressed filename
        output_path = input_path.with_suffix('.compressed.mp4')

        try:
            # Get original file size
            original_size = input_path.stat().st_size / (1024 * 1024)  # MB

            # Run ffmpeg compression
            cmd = [
                'ffmpeg',
                '-err_detect', 'ignore_err',  # Handle potentially corrupt input
                '-i', str(input_path),
                '-c:v', 'libx264',
                '-preset', RECORDING_COMPRESS_PRESET,
                '-crf', str(RECORDING_COMPRESS_CRF),
                '-movflags', '+faststart',  # Enable streaming
                '-y',  # Overwrite output
                '-loglevel', 'error',
                str(output_path)
            ]

            logger.info(f"Compressing: {input_path.name} ({original_size:.1f} MB)")
            start_time = time.time()

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                # Get compressed file size
                compressed_size = output_path.stat().st_size / (1024 * 1024)  # MB
                reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
                duration = time.time() - start_time

                # Replace original with compressed
                input_path.unlink()
                output_path.rename(input_path)

                logger.info(
                    f"Compressed {input_path.name}: {original_size:.1f}MB -> {compressed_size:.1f}MB "
                    f"({reduction:.0f}% reduction) in {duration:.1f}s"
                )
            else:
                logger.error(f"Compression failed for {input_path}: {result.stderr}")
                # Clean up failed output
                if output_path.exists():
                    output_path.unlink()

        except FileNotFoundError:
            logger.warning("ffmpeg not found - skipping compression. Install with: brew install ffmpeg")
        except subprocess.TimeoutExpired:
            logger.error(f"Compression timeout for {input_path}")
            if output_path.exists():
                output_path.unlink()
        except Exception as e:
            logger.error(f"Error compressing {input_path}: {e}")
            # Clean up on error
            if output_path.exists():
                output_path.unlink()


class ThreadedVideoRecorder:
    """
    A high-performance, non-blocking video recorder.
    It receives frames via a queue and writes them to disk in a background thread.
    It automatically handles file rotation (creating new files every N minutes).
    Completed segments are automatically queued for compression.
    """

    # Shared compressor instance for all recorders
    _compressor: Optional[VideoCompressor] = None
    _compressor_lock = threading.Lock()

    @classmethod
    def get_compressor(cls, base_dir):
        """Get or create the shared compressor instance."""
        with cls._compressor_lock:
            if cls._compressor is None:
                cls._compressor = VideoCompressor(base_dir)
            return cls._compressor

    @classmethod
    def stop_compressor(cls):
        """Stop the shared compressor."""
        with cls._compressor_lock:
            if cls._compressor is not None:
                cls._compressor.stop()
                cls._compressor = None

    def __init__(self, base_dir, camera_name, width, height, fps=RECORDING_FRAME_RATE):
        self.base_dir = Path(base_dir)
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps

        self.queue = queue.Queue(maxsize=300)  # Buffer ~10-20 seconds of video if disk is slow
        self.stop_event = threading.Event()
        self.writer = None
        self.current_file_path = None
        self.file_start_time = 0
        self._stopped = False

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Get shared compressor
        self.compressor = self.get_compressor(base_dir)

        # Register for cleanup
        with _cleanup_lock:
            _active_recorders.append(self)

        # NOT a daemon thread - we want it to finish properly on exit
        self.thread = threading.Thread(target=self._record_loop, daemon=False)
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
            pass  # Should be handled above, but safety first

    def stop(self):
        """Signal the recorder to stop and wait for it to finish writing the queue."""
        if self._stopped:
            return

        self._stopped = True
        self.stop_event.set()

        # Wait for thread to finish (with timeout)
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

        # Close writer and queue final file for compression
        old_file = self.current_file_path
        self._close_writer()

        # Queue the final segment for compression
        if old_file and Path(old_file).exists():
            self.compressor.queue_for_compression(old_file)

        # Unregister from cleanup
        with _cleanup_lock:
            if self in _active_recorders:
                _active_recorders.remove(self)

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
        # Close current writer and queue for compression
        old_file = self.current_file_path
        self._close_writer()

        # Queue the completed segment for compression
        if old_file and Path(old_file).exists():
            self.compressor.queue_for_compression(old_file)

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
            try:
                self.writer.release()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
            self.writer = None

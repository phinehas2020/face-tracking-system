"""Isolated RTSP decoder. Credentials arrive on stdin, never in process arguments."""

import json
import os
import struct
import sys
import time


def fit_width(frame, max_width, cv2):
    """Shrink a frame to max_width (keeping its shape) before it is re-encoded."""
    height, width = frame.shape[:2]
    if not max_width or width <= max_width:
        return frame
    size = (max_width, round(height * max_width / width))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def main():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    import cv2

    from .live import CameraConfig

    config = CameraConfig(**json.loads(sys.stdin.readline()))
    capture = cv2.VideoCapture(
        config.url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            5000,
        ],
    )
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            captured_at = time.monotonic()
            frame = fit_width(frame, config.capture_max_width, cv2)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                data = jpeg.tobytes()
                sys.stdout.buffer.write(struct.pack("!Id", len(data), captured_at) + data)
                sys.stdout.buffer.flush()
    finally:
        capture.release()


if __name__ == "__main__":
    main()

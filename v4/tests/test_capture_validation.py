import importlib.util
import json
import stat
import struct
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "capture_validation", Path(__file__).parents[1] / "tools" / "capture_validation.py"
)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def test_packet_parser_handles_split_headers_payloads_and_multiple_packets():
    payload = b"testjpeg"
    packet = struct.pack("!Id", len(payload), 123.4) + payload
    buffer = bytearray(packet[:5])
    assert list(capture._packets(buffer)) == []
    buffer.extend(packet[5:14])
    assert list(capture._packets(buffer)) == []
    buffer.extend(packet[14:] + packet)
    assert list(capture._packets(buffer)) == [(123.4, payload), (123.4, payload)]
    assert buffer == bytearray()


@pytest.mark.parametrize(
    "header",
    [
        struct.pack("!Id", 0, 123),
        struct.pack("!Id", capture.MAX_BUFFER, 123),
        struct.pack("!Id", 1, float("nan")),
    ],
)
def test_bad_packets_rejected_before_unbounded_buffering(header):
    with pytest.raises(capture.CaptureError):
        list(capture._packets(bytearray(header)))


def test_invalid_duration_cannot_connect_to_camera(tmp_path):
    with pytest.raises(capture.CaptureError, match="seconds_must"):
        capture.record(tmp_path / "missing", tmp_path / "output", 121)


def test_two_fake_workers_private_native_video_and_capture_clock(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    config = tmp_path / "private.json"
    config.write_text(
        json.dumps(
            {
                "host": "fixture-side",
                "username": "secret-user",
                "password": "secret-password",
                "facing_camera": {"host": "fixture-front", "password": "secret-password"},
            }
        )
    )
    real_popen = capture.subprocess.Popen
    launches = []
    script = """
import sys,json,time,struct
import cv2,numpy as np
config=json.loads(sys.stdin.readline())
frame=np.full((48,64,3),80 if config['host']=='fixture-side' else 160,dtype=np.uint8)
ok,jpeg=cv2.imencode('.jpg',frame);payload=jpeg.tobytes()
while True:
 sys.stdout.buffer.write(struct.pack('!Id',len(payload),time.monotonic())+payload)
 sys.stdout.buffer.flush()
 time.sleep(.06)
"""

    def fake_worker(argv, **kwargs):
        launches.append(argv)
        assert argv == [sys.executable, "-m", "ingress.capture_worker"]
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(capture.subprocess, "Popen", fake_worker)
    output = tmp_path / "recording"
    summary = capture.record(config, output, seconds=2, fps=15)
    assert len(launches) == 2
    assert summary["error"] is None
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert "secret-" not in (output / "capture.json").read_text()
    for role in ("side", "facing"):
        view = summary["views"][role]
        assert view["frame_count"] > 5
        assert view["native_size"] == (64, 48)
        records = [json.loads(line) for line in (output / f"{role}.jsonl").read_text().splitlines()]
        assert len(records) == view["frame_count"]
        assert [row["frame"] for row in records] == list(range(view["frame_count"]))
        assert all(
            row["captured_at"] - summary["started_monotonic"] == row["elapsed_seconds"]
            for row in records
        )
        video = cv2.VideoCapture(str(output / f"{role}.mp4"))
        try:
            assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == view["frame_count"]
            ok, image = video.read()
            assert ok and image.shape == (48, 64, 3)
        finally:
            video.release()
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    with pytest.raises(FileExistsError):
        capture.record(config, output, seconds=1)

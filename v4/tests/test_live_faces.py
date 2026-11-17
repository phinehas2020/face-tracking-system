import threading
import time

from ingress.live_faces import AsyncFaceWorker, FaceRequest, FaceResult


def request(sequence, generation=1, age=0):
    return FaceRequest(
        None, [], {}, sequence, generation, time.monotonic() - age, "2026-10-07T00:00:00+00:00"
    )


def test_latest_frame_queue_and_old_generation_are_discarded():
    worker = AsyncFaceWorker(lambda: None)
    worker.submit(request(1))
    worker.submit(request(2))
    assert worker.pending[0].sequence == 2
    worker.results.append(FaceResult(request(2), [], {}, {}, 10))
    assert worker.take(2, time.monotonic()) is None
    worker.results.append(FaceResult(request(3, age=3), [], {}, {}, 10))
    assert worker.take(1, time.monotonic()) is None
    worker.results.append(FaceResult(request(4), [], {}, {}, 10))
    assert worker.take(1, time.monotonic()).request.sequence == 4
    assert worker.take(1, time.monotonic()) is None


def test_slow_face_analysis_does_not_block_submissions():
    entered = threading.Event()
    release = threading.Event()

    class Encoder:
        def detect(self, image, person_boxes=None):
            entered.set()
            release.wait(2)
            return []

    worker = AsyncFaceWorker(Encoder)
    worker.start()
    try:
        worker.submit(request(1))
        assert entered.wait(2)
        start = time.monotonic()
        for sequence in range(2, 20):
            worker.submit(request(sequence))
        assert time.monotonic() - start < 0.2
        assert worker.pending[0].sequence == 19
    finally:
        release.set()
        worker.stop()
    assert not worker.thread.is_alive()


def test_reset_invalidates_an_inflight_face_result():
    entered = threading.Event()
    release = threading.Event()

    class Encoder:
        def detect(self, image, person_boxes=None):
            entered.set()
            release.wait(2)
            return []

    worker = AsyncFaceWorker(Encoder)
    worker.start()
    try:
        worker.submit(request(1))
        assert entered.wait(2)
        worker.reset()
        release.set()
        worker.thread.join(0.15)
        assert worker.take(1, time.monotonic()) is None
    finally:
        release.set()
        worker.stop()


def test_completed_face_evidence_is_not_replaced_by_a_later_empty_frame():
    worker = AsyncFaceWorker(lambda: None)
    worker.results.append(FaceResult(request(1), ["face"], {}, {}, 10))
    worker.results.append(FaceResult(request(2), [], {}, {}, 10))
    assert worker.take(1, time.monotonic()).detections == ["face"]
    assert worker.take(1, time.monotonic()).detections == []
    assert worker.take(1, time.monotonic()) is None

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from ingress.vision.accelerator import MPS_LOCK, mps_guard, serialized_mps
from ingress.vision.adapters import UltralyticsPersonDetector
from ingress.vision.contracts import FramePacket


def test_all_mps_instances_share_one_reentrant_lock():
    inside = threading.Event()
    release = threading.Event()
    other_entered = threading.Event()

    class Model:
        device = "mps"

        @serialized_mps("device")
        def run(self, first):
            with mps_guard(self.device):  # Nested calls must not deadlock.
                if first:
                    inside.set()
                    assert release.wait(2)
                else:
                    other_entered.set()

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(Model().run, True)
        assert inside.wait(1)
        second = pool.submit(Model().run, False)
        try:
            assert not other_entered.wait(0.05)
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert other_entered.is_set()


def test_cpu_and_coreml_are_not_blocked_by_mps_lock():
    class Model:
        def __init__(self, device):
            self.device = device

        @serialized_mps("device")
        def run(self):
            return self.device

    with ThreadPoolExecutor(2) as pool, MPS_LOCK:
        cpu = pool.submit(Model("cpu").run)
        coreml = pool.submit(Model("CoreMLExecutionProvider").run)
        assert cpu.result(timeout=1) == "cpu"
        assert coreml.result(timeout=1) == "CoreMLExecutionProvider"


def test_failure_releases_mps_lock_for_another_thread():
    class Model:
        device = "mps"

        @serialized_mps("device")
        def run(self):
            raise ValueError("fixture")

    with pytest.raises(ValueError):
        Model().run()

    def attempt():
        with mps_guard("mps:0"):
            return True

    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(attempt).result(timeout=1)


def test_detector_lock_covers_result_cpu_transfer_not_only_forward():
    transferring = threading.Event()
    release = threading.Event()
    second_forward = threading.Event()

    def tensor(values, block=False):
        def cpu():
            if block:
                transferring.set()
                assert release.wait(2)
            return SimpleNamespace(numpy=lambda: np.asarray(values))

        return SimpleNamespace(cpu=cpu)

    def detector(first):
        model = UltralyticsPersonDetector.__new__(UltralyticsPersonDetector)
        model.selected_device = "mps"
        model.imgsz = 640
        model.confidence = 0.25
        model.roi = None

        def predict(image, **kwargs):
            if not first:
                second_forward.set()
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=tensor([[10, 10, 40, 80]], first),
                        conf=tensor([0.9]),
                        cls=tensor([0]),
                    )
                )
            ]

        model.model = SimpleNamespace(predict=predict)
        return model

    packet = FramePacket("side", 0, 0, "", np.zeros((100, 100, 3), dtype=np.uint8))
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(detector(True).detect, packet)
        assert transferring.wait(1)
        second = pool.submit(detector(False).detect, packet)
        try:
            assert not second_forward.wait(0.05)
        finally:
            release.set()
        assert len(first.result(timeout=2)) == len(second.result(timeout=2)) == 1
    assert second_forward.is_set()

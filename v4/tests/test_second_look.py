import numpy as np
import pytest
from ingress.live_counting import PassageCalibration
from ingress.vision.contracts import Detection
from ingress.vision.second_look import SecondLook


class Model:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


frame = np.zeros((100, 100, 3), np.uint8)
front = Detection((10, 20, 40, 90), .8, 0)
neighbor = Detection((40, 20, 65, 95), .6, 0)


def test_second_look_keeps_primary_and_rescues_only_current_frame():
    model = Model([front, neighbor])
    look = SecondLook(model, "rf.onnx")
    assert look.review(frame, [front], 1., None, None) == [front, neighbor]
    model.detections = []
    assert look.review(frame, [front], 1.2, None, None) == [front]
    assert look.metrics["completed"] == 2
    assert look.metrics["additional_boxes"] == 1


def test_corroborates_weak_observation_without_double_counting():
    look = SecondLook(Model([front, neighbor]), "rf.onnx")
    weak = Detection(front.bbox, .15, 0)
    assert look.review(frame, [weak], 0., None, (0., 0., .45, 1.)) == [front]
    assert look.metrics["strengthened_boxes"] == 1


def test_motion_can_trigger_an_independent_look_when_yolo_sees_no_people():
    look = SecondLook(Model([]), "rf.onnx")
    gate = PassageCalibration()
    look.review(frame, [], 0., gate, None)
    look.review(frame, [], .7, gate, None)
    assert look.metrics["completed"] == 1
    look.review(255-frame, [], .8, gate, None)
    assert look.metrics["last_reason"] == "gate motion"
    assert look.metrics["completed"] == 2


def test_failed_second_look_preserves_primary_and_retries(monkeypatch):
    model = Model([])
    look = SecondLook(model, "rf.onnx")

    def fail(image):
        raise RuntimeError("test")

    with monkeypatch.context() as m:
        m.setattr(model, "detect", fail)
        assert look.review(frame, [front], 0., None, None) == [front]
    assert look.metrics["errors"] == 1
    assert look.review(frame, [front], 2.1, None, None) == [front]
    assert "last_error" not in look.metrics


def test_independent_cameras_keep_ids_across_peer_reset():
    pytest.importorskip("ultralytics")
    from ingress.vision.crowd import CrowdConsensus
    from ultralytics.engine.results import Boxes

    first = CrowdConsensus(False)
    box = Boxes(np.array([[10, 10, 30, 90, .95, 0]], dtype=np.float32), (100, 100))
    for i in range(4):
        first.update(box, i / 10, [])
    prior = first._last_id
    peer = CrowdConsensus(False)
    peer.update(box, 0., [])
    assert first._last_id == prior
    both = Boxes(np.array([[10, 10, 30, 90, .95, 0], [65, 10, 85, 90, .95, 0]],
                          dtype=np.float32), (100, 100))
    for i in range(4, 8):
        tracks = first.update(both, i / 10, [])
    assert len(set(tracks[:, 4])) == 2


def test_weak_rf_proposal_requires_a_fresh_repeat_and_preserves_its_score():
    weak = Detection(neighbor.bbox, .32, 0)
    look = SecondLook(Model([weak]), "rf.onnx")
    assert look.review(frame, [front], 0., None, None) == [front]
    assert look.review(frame, [front], .2, None, None) == [front, weak]
    assert look.metrics["confirmed_low_score"] == 1
    look.reset()
    assert look.review(frame, [front], 3., None, None) == [front]

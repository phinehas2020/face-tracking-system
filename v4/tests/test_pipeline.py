from dataclasses import dataclass

import numpy as np
from ingress.portal import PortalStateMachine
from ingress.vision.contracts import FramePacket, Track
from ingress.vision.pipeline import VisionPipeline


@dataclass
class FakeTracker:
    bottoms: list[float]

    def track(self, _frame):
        bottom = self.bottoms.pop(0)
        return [Track("cam-a:7", (30, 10, 70, bottom), 0.95, 4)]


class FakeFaceEncoder:
    def encode(self, _image, _bbox):
        return np.asarray([0.6, 0.8], dtype=np.float32), 0.9


def test_pipeline_builds_track_template_and_emits_once():
    pipeline = VisionPipeline(
        tracker=FakeTracker([35, 55, 65, 75]),
        portal=PortalStateMachine((0.0, 0.5), (1.0, 0.5), stable_frames=2),
        face_encoder=FakeFaceEncoder(),
    )
    emitted = []
    for sequence in range(4):
        emitted.extend(
            pipeline.process(
                FramePacket(
                    camera_id="cam-a",
                    sequence=sequence,
                    captured_monotonic_ns=sequence,
                    captured_utc=f"2026-08-28T10:00:0{sequence}Z",
                    image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                )
            )
        )
    assert len(emitted) == 1
    assert emitted[0].supporting_frames == 3
    assert np.allclose(emitted[0].face_embedding, [0.6, 0.8])

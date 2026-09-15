import csv

import pytest
from ingress.domain import PairEvidence
from ingress.identity import FusionModel
from ingress.replay import ReplayManifest
from ingress.training import LabeledPair, train_fusion_model


def test_replay_manifest_hashes_sources_deterministically(tmp_path):
    lane_a = tmp_path / "lane-a.mp4"
    wide = tmp_path / "wide.mp4"
    lane_a.write_bytes(b"lane-a-recording")
    wide.write_bytes(b"wide-recording")

    manifest = ReplayManifest.build("event-1", {"wide": wide, "lane-a": lane_a})
    output = tmp_path / "manifest.json"
    manifest.save(output)

    assert [item.camera_id for item in manifest.recordings] == ["lane-a", "wide"]
    assert len(manifest.recordings[0].sha256) == 64
    assert '"event_id": "event-1"' in output.read_text()


def test_fusion_training_and_csv_loader(tmp_path):
    pairs = []
    for index in range(30):
        pairs.append(
            LabeledPair(
                PairEvidence(
                    face_similarity=0.92 + index / 1000,
                    body_similarity=0.84,
                    face_quality=0.9,
                    body_quality=0.8,
                    supporting_frames=14,
                    time_delta_seconds=5,
                    topology_score=0.95,
                    candidate_margin=0.2,
                ),
                True,
            )
        )
        pairs.append(
            LabeledPair(
                PairEvidence(
                    face_similarity=0.2 + index / 1000,
                    body_similarity=0.35,
                    face_quality=0.8,
                    body_quality=0.75,
                    supporting_frames=12,
                    time_delta_seconds=4,
                    topology_score=0.2,
                    impossible_overlap=index % 4 == 0,
                    candidate_margin=0.3,
                ),
                False,
            )
        )

    result = train_fusion_model(pairs, epochs=400)
    assert result.metrics["validation_accuracy"] >= 0.9
    model_path = tmp_path / "fusion.json"
    result.model.save(model_path)
    assert FusionModel.load(model_path).weights == result.model.weights

    csv_path = tmp_path / "pairs.csv"
    fields = [
        "face_similarity", "body_similarity", "face_quality", "body_quality",
        "supporting_frames", "time_delta_seconds", "topology_score",
        "candidate_margin", "impossible_overlap", "same_person",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs[:20]:
            row = pair.evidence.as_features()
            row.update(
                supporting_frames=pair.evidence.supporting_frames,
                time_delta_seconds=pair.evidence.time_delta_seconds,
                topology_score=pair.evidence.topology_score,
                candidate_margin=pair.evidence.candidate_margin,
                impossible_overlap=pair.evidence.impossible_overlap,
                same_person=pair.same_person,
            )
            writer.writerow({field: row[field] for field in fields})
    assert len(LabeledPair.load_csv(csv_path)) == 20


def test_training_rejects_tiny_review_sets():
    with pytest.raises(ValueError, match="at least 20"):
        train_fusion_model([])

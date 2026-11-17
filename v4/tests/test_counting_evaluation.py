import json

from ingress.counting_evaluation import evaluate_counting


def test_counting_comparison_preserves_late_face_link(tmp_path):
    rows = []
    for i in range(8):
        rows.append(
            {
                "clip": 0,
                "at": f"2025-11-28T10:00:0{i}+00:00",
                "clock": i * 0.1,
                "source": {"label": "fixture"},
                "points": [["body", 0.5, 0.2 if i < 3 else 0.8, 0.1 if i < 3 else 0.2]],
                "links": {"body": {"id": "known", "after": None}} if i >= 6 else {},
            }
        )
    path = tmp_path / "evidence.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    old = evaluate_counting(path, 0.75, True)
    assert old["entries"] == 1 and old["linked_entries"] == 1
    assert evaluate_counting(path, 0.9, True)["entries"] == 0

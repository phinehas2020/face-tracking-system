from fastapi.testclient import TestClient
from ingress.app import create_app
from ingress.config import Settings
from ingress.seed import DEMO_EVENT_ID


def test_dashboard_api_contract(tmp_path):
    settings = Settings(
        data_root=tmp_path / "data",
        host="127.0.0.1",
        port=8000,
        demo_mode=True,
        demo_tick_seconds=0.01,
        allowed_origins=("http://terminal.local:4173",),
        web_dist=tmp_path / "missing-web-dist",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        summary = client.get(f"/api/events/{DEMO_EVENT_ID}/summary")
        assert summary.status_code == 200
        assert summary.json()["raw_passages"] == 4417

        timeline = client.get(f"/api/events/{DEMO_EVENT_ID}/timeline").json()["samples"]
        assert len(timeline) == 96

        response = client.put(
            f"/api/events/{DEMO_EVENT_ID}/cameras/cam-lane-a/calibration",
            json={
                "camera_id": "cam-lane-a",
                "approach_zone": [[0.1, 0.1], [0.9, 0.1], [0.8, 0.7]],
                "commit_line": [[0.2, 0.7], [0.8, 0.7]],
                "direction": "in",
            },
        )
        assert response.status_code == 200
        assert response.json()["camera"]["configuration"]["direction"] == "in"

        decision = client.post(
            f"/api/events/{DEMO_EVENT_ID}/candidates/candidate-002/decision",
            json={"decision": "different", "actor": "pytest"},
        )
        assert decision.status_code == 200
        assert decision.json()["summary"]["unresolved"] == 2

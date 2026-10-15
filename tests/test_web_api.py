"""Unit tests for Starlette Web API server."""

import pytest
from starlette.testclient import TestClient

from web_api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "fiducia" in data["service"]


def test_api_scenarios(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    scenario_ids = [s["id"] for s in data["scenarios"]]
    assert "real_binance" in scenario_ids
    assert "bait_wall" in scenario_ids
    assert "real_wall" in scenario_ids


def test_api_replay(client):
    response = client.get("/api/replay?scenario=bait_wall&offset=0&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["scenario"] == "bait_wall"
    assert len(data["frames"]) == 10

    first_frame = data["frames"][0]
    required_keys = [
        "index",
        "timestamp_ms",
        "mid_price",
        "best_bid",
        "best_ask",
        "candle",
        "depth_levels",
        "active_walls",
        "probabilities",
        "expected_value_bps",
        "action",
        "action_label",
        "diagnostic_summary",
        "shap_features",
    ]
    for key in required_keys:
        assert key in first_frame, f"Missing key {key} in frame"

    # Candle properties
    candle = first_frame["candle"]
    assert "open" in candle
    assert "high" in candle
    assert "low" in candle
    assert "close" in candle
    assert candle["high"] >= candle["low"]

    # Probabilities sum to 1.0
    probs = first_frame["probabilities"]
    prob_sum = probs["down"] + probs["neutral"] + probs["up"]
    assert pytest.approx(prob_sum, abs=1e-3) == 1.0

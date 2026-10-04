from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_player_intelligence_capability_registry():
    from basketball_ai.api.routes.player_intelligence_v2 import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/v2/player-intelligence/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "1.0"
    assert body["summary"]["total"] >= 10
    assert body["summary"]["available"] >= 1
    assert body["summary"]["partial"] >= 1
    assert body["summary"]["missing"] >= 1

    by_key = {item["key"]: item for item in body["capabilities"]}
    assert "why_performing" in by_key
    assert "metric_explanation" in by_key["why_performing"]["missing_analyses"]
    assert by_key["current_role"]["status"] == "available"
    assert by_key["stability"]["status"] == "available"
    assert by_key["regression_risk"]["status"] == "available"
    assert by_key["causal_team_effect"]["status"] == "missing"
    assert "causal_team_effect" in by_key["causal_team_effect"]["missing_analyses"]


def test_registry_is_metadata_only():
    from basketball_ai.api.question_analysis_registry import get_capability

    capability = get_capability("team_counterfactual")
    assert capability is not None
    assert capability.status == "partial"
    assert "causal_team_effect" in capability.missing_analyses

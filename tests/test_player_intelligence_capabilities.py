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
    assert body["version"] == "1.1"
    assert body["summary"]["total"] >= 10
    assert body["summary"]["available"] >= 1
    assert body["summary"]["partial"] >= 1
    assert body["summary"]["missing"] == 0

    by_key = {item["key"]: item for item in body["capabilities"]}
    assert "why_performing" in by_key
    assert "metric_explanation" in by_key["why_performing"]["analyses"]
    assert "metric_explanation" not in by_key["why_performing"]["missing_analyses"]
    assert by_key["current_role"]["status"] == "available"
    assert by_key["stability"]["status"] == "available"
    assert by_key["shot_profile"]["analyses"] == [
        "performance_decomposition", "shooting_decomposition", "metric_explanation", "model_evidence"
    ]
    assert by_key["defensive_reason"]["analyses"] == [
        "performance_decomposition", "role_analysis", "defensive_decomposition", "metric_explanation", "model_evidence"
    ]
    assert by_key["regression_risk"]["status"] == "available"
    assert by_key["causal_team_effect"]["status"] == "partial"
    assert by_key["team_usage"]["analyses"] == [
        "role_analysis", "performance_decomposition", "team_usage_analysis",
        "metric_explanation", "model_evidence",
    ]
    assert "causal_identification" in by_key["causal_team_effect"]["missing_analyses"]
    assert "causal_teammate_effect_attribution" in by_key["causal_team_effect"]["missing_analyses"]
    assert "canonical_bb_rating_1_100" in by_key["causal_team_effect"]["evidence"]
    assert "minutes" in by_key["causal_team_effect"]["evidence"]
    assert "observed_role" in by_key["causal_team_effect"]["evidence"]
    assert "lineup_context_association" in by_key["causal_team_effect"]["analyses"]
    assert "within_team_season_lineup_net_rating" in by_key["causal_team_effect"]["evidence"]
    assert "lineup_context_association" in by_key["performance_attribution"]["analyses"]
    assert "observed_teammate_lineup_associations" in by_key["performance_attribution"]["analyses"]
    assert "teammate_lineup_presence_within_player_minutes" in by_key["performance_attribution"]["evidence"]
    assert "ScenarioLineupStats sourced from SimulationLineupStints" in by_key["performance_attribution"]["optional_data"]
    assert any("100 possessions per group" in item for item in by_key["performance_attribution"]["minimum_evidence"])
    assert "canonical BB-Rating engine on the public 1-100 scale" in by_key["performance_attribution"]["required_data"]


def test_registry_is_metadata_only():
    from basketball_ai.api.question_analysis_registry import get_capability

    capability = get_capability("team_counterfactual")
    assert capability is not None
    assert capability.status == "available"
    assert capability.analyses == ("player_team", "model_evidence")
    assert capability.limitation and "non un effetto causale" in capability.limitation
    assert "squadra target esplicita" in capability.limitation

"""Tests for the production-only API v2 surface."""
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_main_mounts_v2_only():
    from basketball_ai.api.main import create_app

    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/api/v2/predictions/player-team" in paths
    assert not any(path.startswith("/api/v1") for path in paths)
    assert "/predictions/batch" not in paths
    assert "/predictions/{player_id}/explain" not in paths


def test_production_requires_service_auth(monkeypatch):
    import basketball_ai.api.main as main_mod

    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://wordpress.example")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(main_mod, "SLOWAPI_AVAILABLE", True)

    try:
        main_mod._validate_production_security()
    except RuntimeError as exc:
        assert "JWT_SECRET or API_KEY" in str(exc)
    else:
        raise AssertionError("production API accepted an unauthenticated configuration")


def test_v2_player_team_contract_uses_real_global_ids():
    from basketball_ai.api.routes.predictions_v2 import router

    class FakeEngine:
        def predict_in_team(self, player_id: int, team_id: int, season: int):
            assert player_id == 101
            assert team_id == 202
            assert season == 2025
            return SimpleNamespace(
                predicted_rating=7.25,
                confidence_low=6.8,
                confidence_high=7.7,
            )

    app = FastAPI()
    app.include_router(router)
    app.state.engine = FakeEngine()
    app.state.data = {
        "player_dict": {
            101: {"id": 101, "global_id": "PLAYER-IDGLOBAL"},
        },
        "team_dict": {
            202: {"id": 202, "global_id": "TEAM-IDGLOBAL"},
        },
    }
    app.state.model_metadata = {
        "model_run_id": "run-1",
        "model_version": "2.0.0",
        "feature_version": "forecast-t-plus-1-v1",
        "data_cutoff": "2025-12-31",
    }

    response = TestClient(app).post(
        "/api/v2/predictions/player-team",
        json={
            "player_global_id": "PLAYER-IDGLOBAL",
            "team_global_id": "TEAM-IDGLOBAL",
            "league": "ITA1",
            "season": 2025,
            "competition": "RS",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["player_global_id"] == "PLAYER-IDGLOBAL"
    assert body["team_global_id"] == "TEAM-IDGLOBAL"
    assert body["target_season"] == 2026
    assert body["predicted_rating"] == 7.25


def test_health_ready_is_503_without_loaded_model():
    from basketball_ai.api.main import create_app

    app = create_app()
    app.state.data = {"player_dict": {1: {"id": 1}}}
    app.state.engine = None
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["model"] is False

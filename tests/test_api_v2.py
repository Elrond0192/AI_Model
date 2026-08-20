"""Tests for the production-only API v2 surface."""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _fake_state(app) -> None:
    class FakeEngine:
        def predict_in_team(self, player_id: int, team_id: int, season: int):
            assert player_id == 101
            assert team_id == 202
            return SimpleNamespace(
                predicted_rating=7.25,
                confidence_low=6.8,
                confidence_high=7.7,
            )

    app.state.engine = FakeEngine()
    app.state.data = {
        "player_dict": {101: {"id": 101, "global_id": "PLAYER-IDGLOBAL"}},
        "team_dict": {202: {"id": 202, "global_id": "TEAM-IDGLOBAL"}},
    }
    app.state.model_metadata = {
        "model_run_id": "run-1",
        "model_version": "2.0.0",
        "feature_version": "forecast-t-plus-1-v1",
        "data_cutoff": "2025-12-31",
    }


def _payload() -> dict:
    return {
        "player_global_id": "PLAYER-IDGLOBAL",
        "team_global_id": "TEAM-IDGLOBAL",
        "league": "ITA1",
        "season": 2025,
        "competition": "RS",
    }


def test_main_mounts_v2_only(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    from basketball_ai.api.main import create_app

    app = create_app()
    paths = {path for route in app.routes if (path := getattr(route, "path", None))}
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


def test_production_security_accepts_api_key(monkeypatch):
    import basketball_ai.api.main as main_mod

    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://wordpress.example")
    monkeypatch.setenv("API_KEY", "k" * 32)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(main_mod, "SLOWAPI_AVAILABLE", True)
    main_mod._validate_production_security()


def test_empty_data_contract():
    from basketball_ai.api.main import _empty_data

    data = _empty_data()
    assert data["player_dict"] == {}
    assert data["team_dict"] == {}
    assert data["players"].empty
    assert data["player_stats"].empty


def test_model_metadata_defaults_and_file(tmp_path):
    from basketball_ai.api.main import _model_metadata

    assert _model_metadata(str(tmp_path))["model_run_id"] == "unversioned"
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "model_run_id": "run-42",
                "model_version": "2.1.0",
                "feature_version": "features-v2",
                "data_cutoff": "2026-06-30",
            }
        ),
        encoding="utf-8",
    )
    metadata = _model_metadata(str(tmp_path))
    assert metadata == {
        "model_run_id": "run-42",
        "model_version": "2.1.0",
        "feature_version": "features-v2",
        "data_cutoff": "2026-06-30",
    }


def test_model_metadata_invalid_json_falls_back(tmp_path):
    from basketball_ai.api.main import _model_metadata

    (tmp_path / "metadata.json").write_text("not-json", encoding="utf-8")
    assert _model_metadata(str(tmp_path))["model_version"] == "v2"


def test_v2_player_team_contract_uses_real_global_ids():
    from basketball_ai.api.routes.predictions_v2 import router

    app = FastAPI()
    app.include_router(router)
    _fake_state(app)

    response = TestClient(app).post("/api/v2/predictions/player-team", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["player_global_id"] == "PLAYER-IDGLOBAL"
    assert body["team_global_id"] == "TEAM-IDGLOBAL"
    assert body["target_season"] == 2026
    assert body["predicted_rating"] == 7.25


def test_v2_returns_404_for_unknown_global_id():
    from basketball_ai.api.routes.predictions_v2 import router

    app = FastAPI()
    app.include_router(router)
    _fake_state(app)
    payload = _payload()
    payload["player_global_id"] = "UNKNOWN"
    response = TestClient(app).post("/api/v2/predictions/player-team", json=payload)
    assert response.status_code == 404


def test_v2_returns_503_without_engine():
    from basketball_ai.api.routes.predictions_v2 import router

    app = FastAPI()
    app.include_router(router)
    app.state.engine = None
    app.state.data = {"player_dict": {}, "team_dict": {}}
    response = TestClient(app).post("/api/v2/predictions/player-team", json=_payload())
    assert response.status_code == 503


def test_health_endpoints_and_readiness(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    from basketball_ai.api.main import create_app

    app = create_app()
    app.state.data = {"player_dict": {1: {"id": 1}}}
    app.state.engine = None
    client = TestClient(app)

    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert live.headers.get("x-content-type-options") == "nosniff"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["data_loaded"] is True

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["model"] is False

    app.state.engine = object()
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_api_key_protects_v2_and_allows_service_call(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.setenv("API_KEY", "service-key-12345678901234567890")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    from basketball_ai.api.main import create_app

    app = create_app()
    _fake_state(app)
    client = TestClient(app)

    denied = client.post("/api/v2/predictions/player-team", json=_payload())
    assert denied.status_code == 401

    allowed = client.post(
        "/api/v2/predictions/player-team",
        json=_payload(),
        headers={"X-API-Key": "service-key-12345678901234567890"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["predicted_rating"] == 7.25


def test_request_body_limit_short_circuits(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "1")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    from basketball_ai.api.main import create_app

    response = TestClient(create_app()).get(
        "/health/live",
        headers={"content-length": "1000001"},
    )
    assert response.status_code == 413


def test_parse_allowed_origins_modes(monkeypatch):
    import basketball_ai.api.main as main_mod

    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    assert main_mod._parse_allowed_origins() == []

    monkeypatch.setenv("API_ENV", "development")
    assert main_mod._parse_allowed_origins() == ["*"]

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://one.example, https://two.example")
    assert main_mod._parse_allowed_origins() == ["https://one.example", "https://two.example"]

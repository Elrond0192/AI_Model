"""Tests for the production-only API v2 surface."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

def test_main_mounts_v2_only(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    from basketball_ai.api.main import create_app

    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/api/v2/predictions/player-team" not in paths
    assert "/api/v2/bb-rating-prediction/player" in paths
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
    assert data["team_season_stats"].empty


def test_model_metadata_defaults_and_file(tmp_path):
    from basketball_ai.api.main import _model_metadata

    assert _model_metadata(str(tmp_path))["model_run_id"] == "unversioned"
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "model_run_id": "run-42",
                "model_version": "2.2.0",
                "feature_version": "forecast-t-plus-1-competition-v1",
                "data_cutoff": "2026-06-30",
            }
        ),
        encoding="utf-8",
    )
    metadata = _model_metadata(str(tmp_path))
    assert metadata == {
        "model_run_id": "run-42",
        "model_version": "2.2.0",
        "feature_version": "forecast-t-plus-1-competition-v1",
        "data_cutoff": "2026-06-30",
    }


def test_model_metadata_invalid_json_falls_back(tmp_path):
    from basketball_ai.api.main import _model_metadata

    (tmp_path / "metadata.json").write_text("not-json", encoding="utf-8")
    assert _model_metadata(str(tmp_path))["model_version"] == "v2"


def test_legacy_prediction_route_is_not_mounted(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    from basketball_ai.api.main import create_app

    paths = set(create_app().openapi()["paths"])
    assert "/api/v2/predictions/player-team" not in paths
    assert "/api/v2/bb-rating-prediction/player" in paths



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
    app.state.bb_rating_engine = object()
    ready = client.get("/health/ready")
    assert ready.status_code == 200


def test_api_key_protects_v2_and_allows_service_call(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.setenv("API_KEY", "service-key-12345678901234567890")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    from basketball_ai.api.main import create_app

    app = create_app()
    client = TestClient(app)
    payload = {
        "player_global_id": "p1",
        "source_season": 2025,
        "source_league": "ITA1",
        "source_competition": "RS",
        "target_league": "ITA1",
        "target_competition": "RS",
    }
    denied = client.post("/api/v2/bb-rating-prediction/player", json=payload)
    assert denied.status_code == 401
    allowed = client.post(
        "/api/v2/bb-rating-prediction/player",
        json=payload,
        headers={"X-API-Key": "service-key-12345678901234567890"},
    )
    assert allowed.status_code in {200, 503}


def test_request_body_limit_short_circuits(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "1")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    from basketball_ai.api.main import create_app

    response = TestClient(create_app()).get(
        "/health/live", headers={"content-length": "1000001"}
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
    assert main_mod._parse_allowed_origins() == [
        "https://one.example", "https://two.example"
    ]

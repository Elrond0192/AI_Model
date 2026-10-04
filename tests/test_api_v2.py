"""Tests for the production-only API v2 surface."""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from basketball_ai.api.contracts_v2 import (
    CompatibilityComparisonV2,
    CompatibilityPlayerProfileV2,
    CompatibilityTeamProfileV2,
)


def _fake_state(app) -> None:
    ensemble = SimpleNamespace(
        _conformal_by_competition={"PO": 0.45},
        _conformal_samples_by_competition={"RS": 30, "PO": 18},
    )
    app.state.engine = SimpleNamespace(ensemble=ensemble)
    app.state.data = {
        "player_dict": {101: {"id": 101, "global_id": "PLAYER-IDGLOBAL"}},
        "team_dict": {202: {"id": 202, "global_id": "TEAM-IDGLOBAL"}},
    }
    app.state.bb_rating_engine = SimpleNamespace(
        rate_player=lambda player_global_id, *, league, season, phase: SimpleNamespace(
            score=79,
            quality="good",
            metric_coverage=0.88,
        )
    )
    app.state.model_metadata = {
        "model_run_id": "run-1",
        "model_version": "2.2.0",
        "feature_version": "forecast-t-plus-1-competition-v1",
        "data_cutoff": "2025-12-31",
    }


def _patch_inference(monkeypatch, expected_competition: str = "RS") -> dict:
    import basketball_ai.api.routes.predictions_v2 as route

    seen: dict = {}

    def fake_snapshot(data, source_season):
        seen["source_season"] = source_season
        return data

    def fake_resolve_league(data, league):
        assert league == "ITA1"
        seen["league_id"] = 303
        return 303

    def fake_scope(data, player_id, team_id, league_id, competition, source_season):
        assert (player_id, team_id, league_id, source_season) == (101, 202, 303, 2025)
        assert competition == expected_competition
        seen["scope_competition"] = competition
        return {
            "player_dict": {101: {"id": 101, "global_id": "PLAYER-IDGLOBAL"}},
            "team_dict": {202: {"id": 202, "global_id": "TEAM-IDGLOBAL"}},
            "_prediction_league_id": 303,
            "_prediction_competition": competition,
            "_competition_support": {
                "mode": "isolated",
                "competition": competition,
                "source_seasons": 2 if competition == "PO" else 4,
                "source_games": 20 if competition == "PO" else 80,
                "exact_source_games": 8 if competition == "PO" else 30,
            },
        }

    class FakeStrictWhatIfEngine:
        def __init__(self, ensemble, data):
            seen["ensemble"] = ensemble
            seen["snapshot"] = data

        def predict_in_team(
            self,
            player_id: int,
            team_id: int,
            season: int,
            competition: str = "RS",
        ):
            assert player_id == 101
            assert team_id == 202
            assert season == 2025
            assert competition == expected_competition
            seen["competition"] = competition
            return SimpleNamespace(
                predicted_rating=7.25,
                confidence_low=6.8,
                confidence_high=7.7,
            )

    monkeypatch.setattr(route, "build_historical_snapshot", fake_snapshot)
    monkeypatch.setattr(route, "resolve_league_id", fake_resolve_league)
    monkeypatch.setattr(route, "scope_prediction_context", fake_scope)
    monkeypatch.setattr(route, "StrictWhatIfEngine", FakeStrictWhatIfEngine)
    monkeypatch.setattr(
        route,
        "_compatibility_comparison",
        lambda *args, **kwargs: CompatibilityComparisonV2(
            selected_team_score=0.74,
            real_team_score=0.68,
            score_delta_vs_real_team=0.06,
            score_delta_vs_neutral=0.24,
            selected_team=CompatibilityTeamProfileV2(
                team_global_id="TEAM-IDGLOBAL",
                team_name="Team",
                season=2025,
                competition=expected_competition,
                pace=75.0,
                three_point_attempt_rate=0.35,
                assists_per_game=20.0,
                star_player_usage=0.25,
                offensive_rating=110.0,
                defensive_rating=108.0,
            ),
            real_team=CompatibilityTeamProfileV2(
                team_global_id="REAL-TEAM",
                team_name="Real Team",
                season=2025,
                competition=expected_competition,
                pace=74.0,
                three_point_attempt_rate=0.34,
                assists_per_game=19.0,
                star_player_usage=0.24,
                offensive_rating=109.0,
                defensive_rating=107.0,
            ),
            player_profile=CompatibilityPlayerProfileV2(
                position="SG",
                usg_pct=0.21,
                ts_pct=0.56,
                points=14.0,
                three_par=0.40,
                dbpm=0.4,
            ),
        ),
    )

    return seen


def _payload(competition: str = "RS") -> dict:
    return {
        "player_global_id": "PLAYER-IDGLOBAL",
        "team_global_id": "TEAM-IDGLOBAL",
        "league": "ITA1",
        "season": 2025,
        "competition": competition,
    }


def test_main_mounts_v2_only(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    from basketball_ai.api.main import create_app

    app = create_app()
    paths = set(app.openapi()["paths"])
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


def test_v2_regular_season_uses_isolated_context(monkeypatch):
    from basketball_ai.api.routes.predictions_v2 import router

    seen = _patch_inference(monkeypatch, "RS")
    app = FastAPI()
    app.include_router(router)
    _fake_state(app)

    response = TestClient(app).post("/api/v2/predictions/player-team", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert seen["source_season"] == 2025
    assert seen["league_id"] == 303
    assert seen["scope_competition"] == "RS"
    assert body["target_season"] == 2026
    assert body["predicted_rating"] == 7.25
    assert body["competition"] == "RS"
    assert body["competition_support"]["mode"] == "isolated"
    assert body["competition_support"]["calibration_scope"] == "global"
    assert body["compatibility"]["selected_team_score"] == 0.74
    assert body["compatibility"]["real_team_score"] == 0.68
    assert body["compatibility"]["score_delta_vs_real_team"] == 0.06
    assert body["performance_vs_expectation"]["available"] is False
    assert body["performance_vs_expectation"]["target_season"] == 2026
    assert body["explanation"]["context"] == "isolated_league_competition_as_of_source_season"


def test_v2_performance_vs_expectation_uses_calibrated_range(monkeypatch):
    from basketball_ai.api.routes.predictions_v2 import router

    _patch_inference(monkeypatch, "RS")
    app = FastAPI()
    app.include_router(router)
    _fake_state(app)
    app.state.prediction_calibration = {
        "calibration_version": "1.0",
        "fit": {
            "x_thresholds": [0.0, 10.0],
            "y_thresholds_native": [0.0, 10.0],
        },
    }

    response = TestClient(app).post(
        "/api/v2/predictions/player-team",
        json=_payload(),
    )
    assert response.status_code == 200
    body = response.json()
    pve = body["performance_vs_expectation"]
    assert pve["available"] is True
    assert pve["target_season"] == 2026
    assert pve["expected_rating_100"] == 72.775
    assert pve["actual_rating_100"] == 79.0
    assert pve["delta_rating_points"] == 6.225
    assert pve["assessment"] == "above_expectations"
    assert pve["actual_quality"] == "good"
    assert pve["actual_metric_coverage"] == 0.88

def test_v2_playoffs_are_normalised_and_isolated(monkeypatch):
    from basketball_ai.api.routes.predictions_v2 import router

    seen = _patch_inference(monkeypatch, "PO")
    app = FastAPI()
    app.include_router(router)
    _fake_state(app)
    response = TestClient(app).post(
        "/api/v2/predictions/player-team", json=_payload("playoffs")
    )
    assert response.status_code == 200
    body = response.json()
    assert seen["competition"] == "PO"
    assert body["competition"] == "PO"
    assert body["competition_support"]["source_games"] == 20
    assert body["competition_support"]["exact_source_games"] == 8
    assert body["competition_support"]["calibration_scope"] == "competition"
    assert body["competition_support"]["calibration_samples"] == 18


def test_v2_returns_422_when_isolated_context_is_unavailable(monkeypatch):
    import basketball_ai.api.routes.predictions_v2 as route
    from basketball_ai.api.routes.predictions_v2 import router

    monkeypatch.setattr(route, "build_historical_snapshot", lambda data, season: data)
    monkeypatch.setattr(route, "resolve_league_id", lambda data, league: 303)

    def fail_scope(*args, **kwargs):
        raise ValueError("Player has no isolated PO data")

    monkeypatch.setattr(route, "scope_prediction_context", fail_scope)
    app = FastAPI()
    app.include_router(router)
    _fake_state(app)
    response = TestClient(app).post(
        "/api/v2/predictions/player-team", json=_payload("PO")
    )
    assert response.status_code == 422
    assert "isolated PO data" in response.json()["detail"]


def test_v2_returns_404_for_unknown_global_id(monkeypatch):
    from basketball_ai.api.routes.predictions_v2 import router

    _patch_inference(monkeypatch)
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
    app.state.engine = object()
    ready = client.get("/health/ready")
    assert ready.status_code == 200


def test_api_key_protects_v2_and_allows_service_call(monkeypatch):
    monkeypatch.setenv("API_ENV", "development")
    monkeypatch.setenv("API_KEY", "service-key-12345678901234567890")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    from basketball_ai.api.main import create_app

    _patch_inference(monkeypatch)
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

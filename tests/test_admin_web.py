from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import basketball_ai.admin_web.app as admin


def _data(seasons=range(2021, 2026)):
    rows = []
    for season in seasons:
        rows.append(
            {
                "player_id": 1,
                "league_id": 10,
                "season": season,
                "competition": "RS",
                "games_played": 10,
                "rating": 7.0,
            }
        )
    return {
        "players": pd.DataFrame([{"id": 1, "global_id": "p1", "name": "Player", "position": "PG"}]),
        "teams": pd.DataFrame([{"id": 2, "global_id": "t1", "name": "Team", "league_id": 10}]),
        "player_stats": pd.DataFrame(rows),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 2,
                    "league_id": 10,
                    "season": season,
                    "competition": "RS",
                    "pace": 72.0,
                    "offensive_rating": 110.0,
                    "defensive_rating": 105.0,
                }
                for season in seasons
            ]
        ),
        "source_contract": "competition-v1",
    }


def test_training_seasons_filters_in_progress_and_blocks_unknown(monkeypatch, tmp_path):
    _reset_state()
    data = _data(seasons=range(2021, 2027))
    lifecycle_path = tmp_path / "season-lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "2021": {"status": "complete"},
                "2022": {"status": "complete"},
                "2023": {"status": "complete"},
                "2024": {"status": "complete"},
                "2025": {"status": "complete"},
                "2026": {"status": "in_progress"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(admin, "SEASON_LIFECYCLE_FILE", lifecycle_path)

    training, excluded, blocked = admin._training_seasons(admin._seasons(data))
    assert training == [2021, 2022, 2023, 2024, 2025]
    assert excluded == [2026]
    assert blocked == []

    filtered = admin._filter_training_data(data)
    assert sorted(filtered["player_stats"]["season"].tolist()) == [2021, 2022, 2023, 2024, 2025]

    lifecycle_path.write_text(
        json.dumps(
            {
                "2021": {"status": "complete"},
                "2022": {"status": "complete"},
                "2023": {"status": "complete"},
                "2024": {"status": "complete"},
                "2025": {"status": "complete"},
                "2026": {"status": "unclassified"},
            }
        ),
        encoding="utf-8",
    )
    training, excluded, blocked = admin._training_seasons(admin._seasons(data))
    assert training == [2021, 2022, 2023, 2024, 2025]
    assert excluded == []
    assert blocked == [2026]


def _request(cookies: dict[str, str] | None = None) -> Request:
    cookie = "; ".join(f"{k}={v}" for k, v in (cookies or {}).items())
    headers = [(b"cookie", cookie.encode())] if cookie else []
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def _reset_state():
    with admin.STATE.lock:
        admin.STATE.data = None
        admin.STATE.active_profile = None
        admin.STATE.summary = {}
        admin.STATE.snapshot = None
        admin.STATE.bb_rating = {
            "status": "idle",
            "stage": "Ready",
            "progress": 0,
            "message": "No BB-Rating calibration run in progress.",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "calibration_version": None,
            "bb_rating_version": None,
            "output": None,
        }
        admin.STATE.training = {
            "status": "idle",
            "stage": "Ready",
            "progress": 0,
            "message": "No training run in progress.",
            "run_id": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }


def test_helpers_and_static_assets(monkeypatch, tmp_path):
    _reset_state()
    data = _data()
    assert admin._seasons(data) == [2021, 2022, 2023, 2024, 2025]
    assert admin._competitions(data) == ["RS"]
    summary = admin._summary(data)
    assert summary["player_rows"] == 5
    assert summary["season_min"] == 2021
    assert summary["season_max"] == 2025
    assert admin._summary(None) == {}

    monkeypatch.setattr(admin, "load_profiles", lambda: {"production": {"password": "secret"}})
    assert "secret" not in admin._safe_error(RuntimeError("password secret leaked"))

    monkeypatch.setattr(admin, "get_promotion_status", lambda _: {"production": {"run_id": "run-1"}, "candidate": None, "previous": None, "history": []})
    assert admin._registry()["production"]["run_id"] == "run-1"
    monkeypatch.setattr(admin, "get_promotion_status", lambda _: (_ for _ in ()).throw(RuntimeError("bad registry")))
    assert admin._registry()["production"] is None

    quality = admin._candidate_quality(
        {
            "backtest": {
                "overall": {"rmse": 1.1, "mae": 0.9, "r2": 0.5, "interval_coverage": 0.9, "n": 42},
                "folds": [{"valid": True, "target_season": 2025, "rmse": 1.2}],
                "by_competition": {"RS": {"n": 42}},
            }
        }
    )
    assert quality["competitions"] == 1
    assert quality["folds"][0]["target_season"] == 2025

    audit_path = tmp_path / "audit.log"
    audit_path.write_text('{"ts":"now","action":"login","actor":"admin"}\ninvalid\n', encoding="utf-8")
    monkeypatch.setattr(admin, "AUDIT_LOG_FILE", audit_path)
    assert admin._audit_rows(20)[0]["action"] == "login"

    html = (admin.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    css = (admin.STATIC_ROOT / "app.css").read_text(encoding="utf-8")
    js = (admin.STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "Operational readiness" in html
    assert "Come funziona" in html
    assert "Prepara storico modello" in html
    assert "Dati e snapshot" in html
    assert "Model Registry" in html
    assert "BB-Rating" in html
    assert 'id="diag-team-search"' in html
    assert 'id="diag-team"' in html
    assert "bb-rating-calibrate" in html
    assert "--sidebar" in css
    assert "renderOverview" in js
    assert "renderBBRating" in js
    assert "/admin-api/bb-rating" in js
    assert "AbortController" in js
    assert "$('.diag-run').forEach" in js
    assert "\n    running++; $('.diag-run').forEach" not in js
    assert "App.controllers.forEach" in js
    future_js = (admin.STATIC_ROOT / "future_performance_ui.js").read_text(encoding="utf-8")
    assert "renderGeneration" in future_js
    assert "page.isConnected" in future_js
    assert "const setHtml" in future_js
    assert "streamlit" not in (html + css + js).lower()
    assert admin.healthz() == {"status": "ok"}
    assert Path(admin.index().path).name == "index.html"



def test_bb_rating_status_reads_calibration_artifacts(monkeypatch, tmp_path):
    _reset_state()
    root = tmp_path / "models_saved"
    calibration = root / "bb_rating_calibration"
    calibration.mkdir(parents=True)
    (calibration / "bb_rating_calibration.json").write_text(
        json.dumps(
            {
                "calibration_version": "1.16",
                "bb_rating_version": "1.8",
                "dataset": {
                    "rows": 53625,
                    "players": 4337,
                    "season_min": 2018,
                    "season_max": 2026,
                    "leagues": ["ITA1"],
                    "competitions": ["RS", "PO", "TOT"],
                    "source_contract": "competition-v2:canonical",
                },
                "configuration": {"uncertainty_min_samples": 50},
                "validation_signals": {
                    "score_available_rate": 0.99,
                    "primary_context_share": 0.95,
                    "limited_context_share": 0.01,
                    "explainable_metric_count": 25,
                },
                "uncertainty_calibration": {
                    "status": "ready",
                    "training_target_seasons": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
                    "excluded_target_seasons": [2026],
                    "training_rows": 49000,
                    "support": {
                        "available_exact_cells": 120,
                        "primary_exact_share": 0.91,
                    },
                },
                "uncertainty_validation": {
                    "selected_structure": {
                        "mean_interval_width_p90": 8.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (calibration / "bb_rating_uncertainty.json").write_text(
        json.dumps({"status": "fitted", "calibration_version": "1.16", "bb_rating_version": "1.8"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(admin, "MODEL_ROOT", root)
    result = admin._bb_rating_status()
    assert result["ready"] is True
    assert result["calibration_version"] == "1.16"
    assert result["bb_rating_version"] == "1.8"
    assert result["dataset"]["rows"] == 53625
    assert result["uncertainty"]["support"]["available_exact_cells"] == 120
    assert result["files"]["uncertainty"].endswith("bb_rating_uncertainty.json")


def test_bb_rating_calibration_start(monkeypatch, tmp_path):
    _reset_state()
    monkeypatch.setattr(admin, "MODEL_ROOT", tmp_path / "models_saved")
    with admin.STATE.lock:
        admin.STATE.active_profile = "production"

    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, *args):
            self.submitted.append((fn, args))

    executor = Executor()
    monkeypatch.setattr(admin.STATE, "bb_rating_executor", executor)
    monkeypatch.setattr(admin, "_utcnow", lambda: "now")
    monkeypatch.setattr(admin, "_audit", lambda *args, **kwargs: None)
    result = admin.start_bb_rating_calibration(
        {"username": "admin", "role": "admin", "token": "x"}
    )
    assert result["ok"] is True
    assert result["profile"] == "production"
    assert result["job"]["status"] == "queued"
    assert executor.submitted



def test_diagnostic_request_builds_whitelisted_payloads():
    prediction = admin._diagnostic_request(
        admin.DiagnosticPayload(
            model="prediction",
            player_global_id="p1",
            team_global_id="t1",
            league="ita1",
            season=2025,
            competition="po",
        )
    )
    assert prediction == (
        "/api/v2/predictions/player-team",
        {
            "player_global_id": "p1",
            "team_global_id": "t1",
            "league": "ITA1",
            "season": 2025,
            "competition": "PO",
        },
    )
    bb = admin._diagnostic_request(
        admin.DiagnosticPayload(model="bb_rating", player_global_id="p1")
    )
    assert bb[0] == "/api/v2/bb-rating/player"
    assert bb[1]["include_history"] is True
    future = admin._diagnostic_request(
        admin.DiagnosticPayload(model="future_performance", player_global_id="p1")
    )
    assert future[0] == "/api/v2/future-performance/player"
    metric = admin._diagnostic_request(
        admin.DiagnosticPayload(model="metric_rating", player_global_id="p1")
    )
    assert metric[0] == "/api/v2/metric-rating/player-snapshot"
    assert metric[1]["metrics"] == ["RAPTOR", "LEBRON", "VORP"]


def test_diagnostic_assertions_validate_model_response():
    bb = admin._diagnostic_assertions(
        "bb_rating",
        {"ok": True, "body": {"bb_rating": 78, "history": [], "uncertainty": {}}},
    )
    assert all(check["ok"] for check in bb)
    future = admin._diagnostic_assertions(
        "future_performance",
        {"ok": True, "body": {"targets": {"PTS/36": {}}, "future_performance_version": "1.0"}},
    )
    assert all(check["ok"] for check in future)
    prediction = admin._diagnostic_assertions(
        "prediction",
        {"ok": True, "body": {"predicted_rating": 7.2, "confidence_low": 6.1, "confidence_high": 8.0}},
    )
    assert all(check["ok"] for check in prediction)


def test_diagnostic_endpoint_uses_internal_api(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        admin,
        "_api_request",
        lambda path, payload: captured.update(path=path, payload=payload) or {
            "ok": True,
            "status": 200,
            "latency_ms": 42.5,
            "body": {"bb_rating": 82, "history": [], "uncertainty": {}},
        },
    )
    monkeypatch.setattr(admin, "_audit", lambda *args, **kwargs: None)
    result = admin.diagnostics_test(
        admin.DiagnosticPayload(model="bb_rating", player_global_id="p1", league="ITA1", season=2025),
        {"username": "admin", "role": "admin", "token": "x"},
    )
    assert result["overall_ok"] is True
    assert captured["path"] == "/api/v2/bb-rating/player"
    assert captured["payload"]["player_global_id"] == "p1"


def test_authentication_helpers_and_csrf(monkeypatch):
    monkeypatch.setattr(admin, "validate_session_token", lambda token: ("admin", "admin") if token == "good" else None)
    user = admin._user_from_request(_request({admin.SESSION_COOKIE: "good"}))
    assert user["username"] == "admin"
    with pytest.raises(HTTPException) as exc:
        admin._user_from_request(_request())
    assert exc.value.status_code == 401

    assert admin._operator({"username": "analyst", "role": "analyst", "token": "x"})["role"] == "analyst"
    with pytest.raises(HTTPException):
        admin._operator({"username": "viewer", "role": "viewer", "token": "x"})
    with pytest.raises(HTTPException):
        admin._admin({"username": "analyst", "role": "analyst", "token": "x"})

    req = _request({admin.CSRF_COOKIE: "csrf"})
    assert admin._csrf(req, "csrf", {"username": "admin", "role": "admin", "token": "x"})["username"] == "admin"
    with pytest.raises(HTTPException) as exc:
        admin._csrf(req, "wrong", {"username": "admin", "role": "admin", "token": "x"})
    assert exc.value.status_code == 403


def test_scenario_entity_search_uses_loaded_dataset():
    _reset_state()
    data = _data()
    data["players"].loc[0, "name"] = "Leandro Nicolas Bolmaro"
    data["players"].loc[0, "global_id"] = "174019"
    data["players"].loc[0, "current_league_key"] = "EL"
    data["teams"].loc[0, "name"] = "Virtus Bologna"
    data["teams"].loc[0, "global_id"] = "933060010445620088"
    data["teams"].loc[0, "short_name"] = "Virtus"

    with admin.STATE.lock:
        admin.STATE.active_profile = "production"
        admin.STATE.data = data

    players = admin.scenario_entities(
        entity="player",
        q="Bolmaro",
        limit=12,
        user={"username": "admin", "role": "admin", "token": "x"},
    )
    assert players["items"][0]["name"] == "Leandro Nicolas Bolmaro"
    assert players["items"][0]["global_id"] == "174019"
    assert players["items"][0]["identity_status"] == "canonical"

    teams = admin.scenario_entities(
        entity="team",
        q="Virtus",
        limit=12,
        user={"username": "admin", "role": "admin", "token": "x"},
    )
    assert teams["items"][0]["name"] == "Virtus Bologna"
    assert teams["items"][0]["global_id"] == "933060010445620088"
    assert teams["items"][0]["subtitle"] == "Virtus"


def test_scenario_proxy_passes_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "internal-key")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'{"scenario":"player_team","result":{"rating":7.1}}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        captured["payload"] = json.loads(request.data)
        assert timeout == 45.0
        return Response()

    monkeypatch.setattr(admin, "urlopen", fake_urlopen)
    response = admin._evaluate_scenario({"scenario": "player_team"})
    assert response["scenario"] == "player_team"
    assert captured["url"].endswith("/api/v2/scenarios/evaluate")
    assert captured["api_key"] == "internal-key"
    assert captured["payload"] == {"scenario": "player_team"}


def test_login_session_logout(monkeypatch):
    monkeypatch.setattr(admin, "check_credentials", lambda u, p: (True, {"role": "admin"}))
    monkeypatch.setattr(admin, "create_session_token", lambda u, r, ttl_days: "raw-token")
    response = admin.login(admin.LoginPayload(username="Admin", password="password"))
    assert response.status_code == 200
    assert admin.SESSION_COOKIE in response.headers.get("set-cookie", "")

    monkeypatch.setattr(admin, "check_credentials", lambda u, p: (False, {}))
    with pytest.raises(HTTPException) as exc:
        admin.login(admin.LoginPayload(username="Admin", password="bad"))
    assert exc.value.status_code == 401

    assert admin.session({"username": "admin", "role": "admin", "token": "x"})["authenticated"] is True
    revoked = []
    monkeypatch.setattr(admin, "revoke_session_token", lambda token: revoked.append(token))
    out = admin.logout(_request(), {"username": "admin", "role": "admin", "token": "x"})
    assert out.status_code == 200
    assert revoked == ["x"]


@pytest.mark.asyncio
async def test_security_headers():
    async def call_next(_):
        return Response("ok")

    response = await admin.security_headers(_request(), call_next)
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(admin, "ensure_default_admin", lambda: calls.append(True))
    admin.startup()
    assert calls == [True]


@pytest.mark.asyncio
async def test_overview(monkeypatch):
    _reset_state()
    data = _data()
    with admin.STATE.lock:
        admin.STATE.data = data
        admin.STATE.active_profile = "production"
        admin.STATE.summary = admin._summary(data)
    monkeypatch.setattr(
        admin,
        "_registry",
        lambda: {
            "production": {"run_id": "prod-1"},
            "candidate": {"run_id": "cand-1", "backtest": {"overall": {"rmse": 1.0}, "folds": [], "by_competition": {}}},
            "previous": None,
            "history": [],
        },
    )
    monkeypatch.setattr(admin, "_probe", lambda path: {"ok": True, "status": 200, "body": {"status": "ok"}})
    result = await admin.overview({"username": "admin", "role": "admin", "token": "x"})
    assert result["database"] == "production"
    assert result["api"]["online"] is True
    assert all(row["status"] == "ok" for row in result["readiness"])


def test_profiles_and_save(monkeypatch):
    _reset_state()
    with admin.STATE.lock:
        admin.STATE.active_profile = "production"
    profiles_map = {
        "production": {
            "host": "host.docker.internal",
            "port": 5432,
            "database": "hoopmetrics",
            "user": "ai_model",
            "password": "secret",
            "source_schema": "ai_source",
            "ai_schema": "ai",
        }
    }
    monkeypatch.setattr(admin, "load_profiles", lambda: profiles_map)
    result = admin.profiles({"username": "admin", "role": "admin", "token": "x"})
    assert result["profiles"]["production"]["password"] == "••••••••"

    saved = {}
    monkeypatch.setattr(admin, "save_profile", lambda name, profile: saved.update(name=name, profile=profile))
    payload = admin.ProfilePayload(
        name="production",
        host="host.docker.internal",
        database="hoopmetrics",
        user="ai_model",
        password="",
    )
    out = admin.save_database_profile(payload, {"username": "admin", "role": "admin", "token": "x"})
    assert out["ok"] is True
    assert saved["profile"]["password"] == "secret"

    monkeypatch.setattr(admin, "load_profiles", lambda: {})
    with pytest.raises(HTTPException):
        admin.save_database_profile(payload, {"username": "admin", "role": "admin", "token": "x"})


@pytest.mark.asyncio
async def test_profile_test_and_load(monkeypatch):
    _reset_state()
    profile = {"source_schema": "ai_source"}
    monkeypatch.setattr(admin, "load_profiles", lambda: {"production": profile})
    monkeypatch.setattr(admin, "profile_url", lambda name: "postgresql://redacted")

    class Conn:
        def exec_driver_sql(self, sql):
            assert sql == "SELECT 1"

    class Context:
        def __enter__(self):
            return Conn()
        def __exit__(self, *args):
            return False

    class Engine:
        def connect(self):
            return Context()

    monkeypatch.setattr(admin, "get_engine", lambda url: Engine())
    tested = await admin.test_database_profile("production", {"username": "admin", "role": "admin", "token": "x"})
    assert tested["ok"] is True

    data = _data()
    received = {}

    def fake_load_all_data(url, schema, *, include_optional=None, purpose="analysis"):
        assert purpose == "training"
        received.update(url=url, schema=schema, include_optional=include_optional)
        return data

    monkeypatch.setattr(admin, "load_all_data", fake_load_all_data)
    monkeypatch.setattr(
        admin,
        "create_training_snapshot",
        lambda data, *, profile: {"snapshot_id": "snapshot-1", "profile": profile},
    )
    loaded = await admin.load_database_profile("production", {"username": "admin", "role": "admin", "token": "x"})
    assert loaded["summary"]["player_rows"] == 5
    assert admin.STATE.active_profile == "production"
    assert received["include_optional"] is False
    assert loaded["snapshot"]["snapshot_id"] == "snapshot-1"

    with pytest.raises(HTTPException):
        await admin.test_database_profile("missing", {"username": "admin", "role": "admin", "token": "x"})


@pytest.mark.asyncio
async def test_snapshot_list_and_activation(monkeypatch):
    _reset_state()
    data = _data()
    manifest = {"snapshot_id": "snapshot-1", "profile": "production", "sha256": "abc"}
    monkeypatch.setattr(admin, "list_training_snapshots", lambda: [manifest])
    monkeypatch.setattr(admin, "load_training_snapshot", lambda snapshot_id: data)

    listed = admin.snapshots({"username": "admin", "role": "admin", "token": "x"})
    assert listed["snapshots"] == [manifest]
    activated = await admin.activate_snapshot(
        "snapshot-1", {"username": "admin", "role": "admin", "token": "x"}
    )
    assert activated["snapshot"] == manifest
    assert admin.STATE.snapshot == manifest
    assert admin.STATE.summary["player_rows"] == 5


def test_training_view_and_start(monkeypatch):
    _reset_state()
    data = _data()
    with admin.STATE.lock:
        admin.STATE.data = data
        admin.STATE.active_profile = "production"
        admin.STATE.summary = admin._summary(data)
    lifecycle = {str(season): {"status": "complete"} for season in admin._seasons(data)}
    lifecycle_path = Path("/tmp/test-season-lifecycle.json")
    lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")
    monkeypatch.setattr(admin, "SEASON_LIFECYCLE_FILE", lifecycle_path)
    view = admin.training({"username": "admin", "role": "admin", "token": "x"})
    assert view["can_start"] is True
    assert len(view["contract"]) == 4

    submitted = []

    class Executor:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    monkeypatch.setattr(admin.STATE, "executor", Executor())
    result = admin.start_training({"username": "admin", "role": "admin", "token": "x"})
    assert result["ok"] is True
    assert submitted
    with pytest.raises(HTTPException) as exc:
        admin.start_training({"username": "admin", "role": "admin", "token": "x"})
    assert exc.value.status_code == 409

    _reset_state()
    with pytest.raises(HTTPException):
        admin.start_training({"username": "admin", "role": "admin", "token": "x"})


def test_training_worker_success_and_failure(monkeypatch, tmp_path):
    data = _data(seasons=range(2021, 2027))
    lifecycle_path = tmp_path / "season-lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "2021": {"status": "complete"},
                "2022": {"status": "complete"},
                "2023": {"status": "complete"},
                "2024": {"status": "complete"},
                "2025": {"status": "complete"},
                "2026": {"status": "in_progress"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(admin, "SEASON_LIFECYCLE_FILE", lifecycle_path)
    monkeypatch.setattr(admin, "MODEL_ROOT", tmp_path)
    captured = {}

    class Model:
        def train(self, data):
            captured["seasons"] = sorted(data["player_stats"]["season"].tolist())
            return {"metric": 1}
        def save(self, path, metadata):
            Path(path).mkdir(parents=True, exist_ok=True)

    import basketball_ai.models.strict_production as strict
    import basketball_ai.models.backtest as backtest
    import basketball_ai.models.promote as promote

    monkeypatch.setattr(strict, "StrictProductionEnsembleModel", Model)
    monkeypatch.setattr(backtest, "run_backtest", lambda *a, **k: {"valid": True, "folds": [{"valid": True, "target_season": 2025, "rmse": 1.0}], "overall": {"rmse": 1.0}})
    registered = []
    monkeypatch.setattr(promote, "register_candidate", lambda *a, **k: registered.append(a))
    admin._training_worker(data, "production")
    assert admin.STATE.training["status"] == "complete"
    assert admin.STATE.training["run_id"]
    assert captured["seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert registered

    class BrokenModel:
        def train(self, data):
            raise RuntimeError("boom")

    monkeypatch.setattr(strict, "StrictProductionEnsembleModel", BrokenModel)
    admin._training_worker(data, "production")
    assert admin.STATE.training["status"] == "failed"
    assert "boom" in admin.STATE.training["error"]


def test_backtests_registry_promote_rollback(monkeypatch):
    registry_data = {
        "production": {"run_id": "prod"},
        "candidate": {"run_id": "cand", "backtest": {"overall": {"rmse": 1.2}, "folds": [], "by_competition": {}}},
        "previous": {"run_id": "prev"},
        "history": [],
    }
    monkeypatch.setattr(admin, "_registry", lambda: registry_data)
    assert admin.backtests({"username": "admin", "role": "admin", "token": "x"})["candidate"]["run_id"] == "cand"
    assert admin.registry({"username": "admin", "role": "admin", "token": "x"})["production"]["run_id"] == "prod"

    monkeypatch.setattr(admin, "promote_if_better", lambda _: {"promoted": True, "reason": "ok"})
    assert admin.promote(admin.ConfirmPayload(confirm=True), {"username": "admin", "role": "admin", "token": "x"})["promoted"]
    with pytest.raises(HTTPException):
        admin.promote(admin.ConfirmPayload(confirm=False), {"username": "admin", "role": "admin", "token": "x"})
    monkeypatch.setattr(admin, "promote_if_better", lambda _: {"promoted": False, "reason": "gates"})
    with pytest.raises(HTTPException) as exc:
        admin.promote(admin.ConfirmPayload(confirm=True), {"username": "admin", "role": "admin", "token": "x"})
    assert exc.value.status_code == 409

    monkeypatch.setattr(admin, "rollback_to_previous", lambda _: {"rolled_back": True, "reason": "ok"})
    assert admin.rollback(admin.ConfirmPayload(confirm=True), {"username": "admin", "role": "admin", "token": "x"})["rolled_back"]
    with pytest.raises(HTTPException):
        admin.rollback(admin.ConfirmPayload(confirm=False), {"username": "admin", "role": "admin", "token": "x"})


def test_scenario_entity_search_and_server_side_resolution():
    _reset_state()
    data = _data()
    with admin.STATE.lock:
        admin.STATE.data = data
    user = {"username": "admin", "role": "admin", "token": "x"}

    players = admin.scenario_entities("player", "play", 12, user)
    assert players["items"] == [
        {"selection_id": "1", "name": "Player", "subtitle": "", "identity_status": "canonical"}
    ]

    teams = admin.scenario_entities("team", "team", 12, user)
    assert teams["items"][0]["selection_id"] == "2"
    assert teams["items"][0]["name"] == "Team"

    resolved = admin._resolve_scenario_selections(
        {
            "scenario": "player_team",
            "player_selection_ids": ["1"],
            "team_selection_ids": ["2"],
        }
    )
    assert resolved["player_global_ids"] == ["p1"]
    assert resolved["team_global_ids"] == ["t1"]
    assert "player_selection_ids" not in resolved
    assert "team_selection_ids" not in resolved


@pytest.mark.asyncio
async def test_api_health(monkeypatch):
    monkeypatch.setattr(admin, "_probe", lambda path: {"ok": path.endswith("live"), "status": 200 if path.endswith("live") else 503})
    result = await admin.api_health({"username": "admin", "role": "admin", "token": "x"})
    assert result["live"]["ok"] is True
    assert result["ready"]["ok"] is False
    assert "/api/v2/scenarios/evaluate" in result["endpoints"][-1]


def test_audit_settings_and_password(monkeypatch, tmp_path):
    audit_path = tmp_path / "audit.log"
    audit_path.write_text(json.dumps({"action": "login", "actor": "admin"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(admin, "AUDIT_LOG_FILE", audit_path)
    assert admin.audit(10, {"username": "admin", "role": "admin", "token": "x"})["entries"]

    _reset_state()
    with admin.STATE.lock:
        admin.STATE.active_profile = "production"
    settings = admin.settings({"username": "admin", "role": "admin", "token": "x"})
    assert settings["runtime"]["database_profile"] == "production"
    assert "min_improvement_pct" in settings["promotion_gates"]

    monkeypatch.setattr(admin, "change_password", lambda u, old, new: True)
    revoked = []
    monkeypatch.setattr(admin, "revoke_user_sessions", lambda u: revoked.append(u))
    response = admin.update_password(
        admin.PasswordPayload(old_password="OldPassword1!", new_password="NewPassword2!"),
        _request(),
        {"username": "admin", "role": "admin", "token": "x"},
    )
    assert response.status_code == 200
    assert revoked == ["admin"]

    monkeypatch.setattr(admin, "change_password", lambda u, old, new: False)
    with pytest.raises(HTTPException):
        admin.update_password(
            admin.PasswordPayload(old_password="bad", new_password="NewPassword2!"),
            _request(),
            {"username": "admin", "role": "admin", "token": "x"},
        )

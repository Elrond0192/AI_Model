"""Tests for new P1/P2 features: drift detection, age curve, JWT auth, caching."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Drift detection (PSI)
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_capture_reference_returns_dict(self, tiny_data):
        from basketball_ai.monitoring.drift import capture_reference
        ref = capture_reference(tiny_data)
        assert isinstance(ref, dict)
        # At least some features should be captured
        assert len(ref) > 0

    def test_psi_no_drift_same_data(self, tiny_data):
        from basketball_ai.monitoring.drift import capture_reference, compute_psi_report
        ref = capture_reference(tiny_data)
        report = compute_psi_report(tiny_data, ref)
        # No drift expected when same data is used
        assert isinstance(report, dict)
        for col, psi in report.items():
            assert psi >= 0.0

    def test_psi_empty_reference(self, tiny_data):
        from basketball_ai.monitoring.drift import compute_psi_report
        report = compute_psi_report(tiny_data, None)
        assert report == {}

    def test_psi_empty_reference_dict(self, tiny_data):
        from basketball_ai.monitoring.drift import compute_psi_report
        report = compute_psi_report(tiny_data, {})
        assert report == {}


# ---------------------------------------------------------------------------
# Age curve empirical fit
# ---------------------------------------------------------------------------

class TestAgeCurveEmpiricalFit:
    def setup_method(self):
        from basketball_ai.models.age_curve import reset_fitted_params
        reset_fitted_params()

    def teardown_method(self):
        from basketball_ai.models.age_curve import reset_fitted_params
        reset_fitted_params()

    def test_default_factor_within_range(self):
        from basketball_ai.models.age_curve import age_performance_factor
        for pos in ["PG", "SG", "SF", "PF", "C", "PF/C"]:
            for age in [18, 22, 26, 30, 35, 40]:
                f = age_performance_factor(age, pos)
                assert 0.40 <= f <= 1.00, f"Out of range for {pos} age {age}: {f}"

    def test_fit_from_data_replaces_defaults(self):
        from basketball_ai.models.age_curve import fit_from_data, age_performance_factor, _fitted_peak_ages
        # Generate simple data: PG peaks at age 24
        data = {"PG": [(age, 1.0 - abs(age - 24) * 0.05) for age in range(18, 38)] * 3}
        fit_from_data(data, min_samples=3)
        # After fit, _fitted_peak_ages should have PG
        assert "PG" in _fitted_peak_ages
        # Factor at age 24 should be max
        f_24 = age_performance_factor(24, "PG")
        f_30 = age_performance_factor(30, "PG")
        assert f_24 >= f_30

    def test_reset_clears_fitted(self):
        from basketball_ai.models.age_curve import fit_from_data, reset_fitted_params, _fitted_peak_ages
        data = {"PG": [(age, 1.0) for age in range(18, 38)] * 3}
        fit_from_data(data, min_samples=3)
        reset_fitted_params()
        assert len(_fitted_peak_ages) == 0


# ---------------------------------------------------------------------------
# JWT auth route
# ---------------------------------------------------------------------------

class TestJWTAuth:
    def test_token_endpoint_returns_501_when_not_configured(self):
        """When JWT_SECRET is not set the endpoint returns 501."""
        import os
        os.environ.pop("JWT_SECRET", None)
        from fastapi.testclient import TestClient
        from basketball_ai.api.main import create_app
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.post("/api/v1/auth/token", json={"username": "x", "password": "y"})
        assert resp.status_code == 501

    def test_token_endpoint_returns_401_for_bad_credentials(self, monkeypatch):
        """When JWT is configured, wrong credentials → 401."""
        import json
        monkeypatch.setenv("JWT_SECRET", "test-secret-key")
        monkeypatch.setenv("JWT_USERS", json.dumps({"admin": "pass123"}))
        # Reimport to pick up env vars
        import importlib
        import basketball_ai.api.routes.auth as auth_mod
        importlib.reload(auth_mod)
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(auth_mod.router, prefix="/api/v1")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/auth/token", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_full_token_flow(self, monkeypatch):
        """Successful login → access + refresh tokens both present."""
        import json
        import jwt
        monkeypatch.setenv("JWT_SECRET", "test-secret-key")
        monkeypatch.setenv("JWT_USERS", json.dumps({"admin": "pass123"}))
        import importlib
        import basketball_ai.api.routes.auth as auth_mod
        importlib.reload(auth_mod)
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(auth_mod.router, prefix="/api/v1")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/auth/token", json={"username": "admin", "password": "pass123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        # Tokens should decode correctly
        payload = jwt.decode(body["access_token"], "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "admin"
        assert payload["kind"] == "access"


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

class TestPredictionCache:
    def test_cache_hit_returns_same_result(self, tiny_data, ensemble):
        """Calling predict twice with the same args should hit the cache."""
        players = list(tiny_data["player_dict"].keys())
        teams   = list(tiny_data["team_dict"].keys())
        if not players or not teams:
            pytest.skip("No players/teams in tiny_data")
        pid, tid = players[0], teams[0]
        r1 = ensemble.predict(pid, tid, tiny_data)
        r2 = ensemble.predict(pid, tid, tiny_data)
        assert r1.predicted_rating == r2.predicted_rating

    def test_clear_cache(self, ensemble):
        ensemble.clear_cache()
        assert len(ensemble._prediction_cache) == 0


# ---------------------------------------------------------------------------
# Rate limiter module
# ---------------------------------------------------------------------------

class TestLimiterModule:
    def test_limiter_is_importable(self):
        from basketball_ai.api.limiter import limiter, RATE_LIMIT_PREDICTIONS, RATE_LIMIT_CHAT
        assert limiter is not None
        assert isinstance(RATE_LIMIT_PREDICTIONS, str)
        assert isinstance(RATE_LIMIT_CHAT, str)


# ---------------------------------------------------------------------------
# CORS origins helper
# ---------------------------------------------------------------------------

class TestCORSOrigins:
    def test_parse_empty_returns_empty_in_production(self, monkeypatch):
        monkeypatch.setenv("API_ENV", "production")
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        import importlib
        import basketball_ai.api.main as main_mod
        importlib.reload(main_mod)
        origins = main_mod._parse_allowed_origins()
        assert origins == []

    def test_parse_comma_list(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com,https://app.example.com")
        import importlib
        import basketball_ai.api.main as main_mod
        importlib.reload(main_mod)
        origins = main_mod._parse_allowed_origins()
        assert "https://example.com" in origins
        assert "https://app.example.com" in origins
        assert len(origins) == 2

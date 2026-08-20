"""Tests for drift detection, age curve, prediction cache and API helpers."""
from __future__ import annotations

import pytest


class TestDriftDetection:
    def test_capture_reference_returns_dict(self, tiny_data):
        from basketball_ai.monitoring.drift import capture_reference

        ref = capture_reference(tiny_data)
        assert isinstance(ref, dict)
        assert len(ref) > 0

    def test_psi_no_drift_same_data(self, tiny_data):
        from basketball_ai.monitoring.drift import capture_reference, compute_psi_report

        ref = capture_reference(tiny_data)
        report = compute_psi_report(tiny_data, ref)
        assert isinstance(report, dict)
        for psi in report.values():
            assert psi >= 0.0

    def test_psi_empty_reference(self, tiny_data):
        from basketball_ai.monitoring.drift import compute_psi_report

        assert compute_psi_report(tiny_data, None) == {}
        assert compute_psi_report(tiny_data, {}) == {}


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
                factor = age_performance_factor(age, pos)
                assert 0.40 <= factor <= 1.00

    def test_fit_from_data_replaces_defaults(self):
        from basketball_ai.models.age_curve import _fitted_peak_ages, age_performance_factor, fit_from_data

        data = {"PG": [(age, 1.0 - abs(age - 24) * 0.05) for age in range(18, 38)] * 3}
        fit_from_data(data, min_samples=3)
        assert "PG" in _fitted_peak_ages
        assert age_performance_factor(24, "PG") >= age_performance_factor(30, "PG")

    def test_reset_clears_fitted(self):
        from basketball_ai.models.age_curve import _fitted_peak_ages, fit_from_data, reset_fitted_params

        data = {"PG": [(age, 1.0) for age in range(18, 38)] * 3}
        fit_from_data(data, min_samples=3)
        reset_fitted_params()
        assert len(_fitted_peak_ages) == 0

    def test_get_peak_age_prefers_fitted(self):
        from basketball_ai.constants import get_peak_age
        from basketball_ai.models.age_curve import _fitted_peak_ages, reset_fitted_params

        reset_fitted_params()
        try:
            _fitted_peak_ages["PG"] = 24.6
            assert get_peak_age("PG") == 25
            assert get_peak_age("PG/SG") == 25
        finally:
            reset_fitted_params()

    def test_get_peak_age_falls_back_to_static_prior(self):
        from basketball_ai.constants import POSITIONAL_PEAK_AGES, get_peak_age

        assert get_peak_age("PF/C") == POSITIONAL_PEAK_AGES["PF/C"]

    def test_maybe_fit_from_db_tiny_data(self, tiny_data):
        from basketball_ai.models.age_curve import _fitted_peak_ages, maybe_fit_from_db

        assert maybe_fit_from_db(tiny_data, min_samples=5) is True
        assert len(_fitted_peak_ages) > 0


class TestPredictionCache:
    def test_cache_hit_returns_same_result(self, tiny_data, ensemble):
        players = list(tiny_data["player_dict"].keys())
        teams = list(tiny_data["team_dict"].keys())
        if not players or not teams:
            pytest.skip("No players/teams in tiny_data")
        player_id, team_id = players[0], teams[0]
        first = ensemble.predict(player_id, team_id, tiny_data)
        second = ensemble.predict(player_id, team_id, tiny_data)
        assert first.predicted_rating == second.predicted_rating

    def test_clear_cache(self, ensemble):
        ensemble.clear_cache()
        assert len(ensemble._prediction_cache) == 0


class TestLimiterModule:
    def test_limiter_is_importable(self):
        from basketball_ai.api.limiter import RATE_LIMIT_PREDICTIONS, limiter

        assert limiter is not None
        assert isinstance(RATE_LIMIT_PREDICTIONS, str)


class TestCORSOrigins:
    def test_parse_empty_returns_empty_in_production(self, monkeypatch):
        monkeypatch.setenv("API_ENV", "production")
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        import basketball_ai.api.main as main_mod

        assert main_mod._parse_allowed_origins() == []

    def test_parse_comma_list(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com,https://app.example.com")
        import basketball_ai.api.main as main_mod

        origins = main_mod._parse_allowed_origins()
        assert set(origins) == {"https://example.com", "https://app.example.com"}
        assert len(origins) == 2

"""Tests for the exact production model contract."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd


def test_position_as_of_uses_roster_position_and_ignores_future_state():
    from basketball_ai.models.strict_production import position_as_of

    relations = pd.DataFrame(
        [
            {"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"},
            {"player_id": 7, "team_id": 10, "season": 2024, "role": "SF"},
        ]
    )
    assert position_as_of(7, relations, 2022, "PG") == "SG"
    assert position_as_of(7, relations, 2023, "PG") == "SG"
    assert position_as_of(7, relations, 2024, "PG") == "SF"


def test_position_as_of_rejects_non_position_legacy_role():
    from basketball_ai.models.strict_production import position_as_of

    relations = pd.DataFrame(
        [{"player_id": 7, "team_id": 10, "season": 2022, "role": "starter"}]
    )
    assert position_as_of(7, relations, 2022, "PF") == "PF"


def test_runtime_snapshot_patches_player_position_without_mutating_input():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    data = {
        "player_dict": {7: {"id": 7, "position": "SF"}},
        "team_player_relations": pd.DataFrame(
            [
                {"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"},
                {"player_id": 7, "team_id": 10, "season": 2024, "role": "SF"},
            ]
        ),
    }
    model = StrictProductionEnsembleModel()
    patched = model._data_with_asof_position(7, data, 2022)
    assert patched["player_dict"][7]["position"] == "SG"
    assert data["player_dict"][7]["position"] == "SF"


def test_historical_snapshot_removes_future_team_history(monkeypatch):
    import basketball_ai.models.strict_production as strict

    parent_snapshot = {
        "team_season_stats": pd.DataFrame(
            [
                {"team_id": 10, "season": 2022},
                {"team_id": 10, "season": 2025},
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [{"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"}]
        ),
        "players": pd.DataFrame([{"id": 7, "position": "SF"}]),
        "player_dict": {7: {"id": 7, "position": "SF"}},
    }
    monkeypatch.setattr(
        strict,
        "_build_historical_snapshot",
        lambda data, season: parent_snapshot.copy(),
    )
    snapshot = strict.build_historical_snapshot({}, 2022)
    assert snapshot["team_season_stats"]["season"].tolist() == [2022]
    assert snapshot["player_dict"][7]["position"] == "SG"
    assert snapshot["_as_of_season"] == 2022


def test_finite_sample_conformal_uses_single_direct_quantile(monkeypatch):
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    monkeypatch.setenv("MODEL_TARGET_INTERVAL_COVERAGE", "0.90")
    residuals = [float(value) / 10.0 for value in range(1, 21)]
    model = StrictProductionEnsembleModel()
    model._calibrate_conformal(residuals)

    rank = min(len(residuals), math.ceil((len(residuals) + 1) * 0.90))
    expected = sorted(residuals)[rank - 1]
    assert model._conformal_q_lo == expected
    assert model._conformal_q_hi == expected
    assert model._conformal_nominal_coverage == 0.90


def test_finite_sample_conformal_refuses_too_small_holdout():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel()
    model._calibrate_conformal([0.1] * 9)
    assert model._conformal_q_lo is None
    assert model._conformal_q_hi is None


def test_none_season_means_latest_observed_source_season(monkeypatch):
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel()
    seen = {}

    def fake_predict_uncached(player_id, team_id, data, season, target_age, competition):
        seen["season"] = season
        return SimpleNamespace(predicted_rating=7.0)

    monkeypatch.setattr(model, "_predict_uncached", fake_predict_uncached)
    data = {"player_stats": pd.DataFrame([{"season": 2024}, {"season": 2025}])}
    model.predict(1, 2, data, season=None)
    assert seen["season"] == 2025


def test_team_features_use_as_of_roster_not_future_roster():
    from basketball_ai.features.team_features import compute_team_features

    team = {
        "id": 10,
        "name": "Team",
        "league_id": 1,
        "playing_style": "motion_offense",
        "formation": "",
        "pace": 75.0,
        "offensive_rating": 110.0,
        "defensive_rating": 108.0,
        "three_point_attempt_rate": 0.35,
        "assists_per_game": 20.0,
        "star_player_usage": 0.25,
        "league_tier": 1,
        "short_name": "T",
        "net_rtg": 2.0,
    }
    data = {
        "_as_of_season": 2022,
        "team_dict": {10: team},
        "league_dict": {1: {"id": 1, "competitiveness_score": 1.0}},
        "leagues": pd.DataFrame([{"id": 1, "competitiveness_score": 1.0}]),
        "team_player_relations": pd.DataFrame(
            [
                {"team_id": 10, "player_id": 1, "season": 2022},
                {"team_id": 10, "player_id": 2, "season": 2025},
            ]
        ),
        "player_stats": pd.DataFrame(
            [
                {"player_id": 1, "season": 2022, "rating": 6.0},
                {"player_id": 2, "season": 2025, "rating": 10.0},
            ]
        ),
    }
    features = compute_team_features(10, data)
    assert features["avg_teammate_rating"] == 6.0
    assert features["league_tier_factor"] == 1.0

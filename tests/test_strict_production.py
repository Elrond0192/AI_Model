"""Tests for the exact production model contract."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest


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
                {"team_id": 10, "season": 2022, "competition": "PO"},
                {"team_id": 10, "season": 2025, "competition": "PO"},
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [{"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"}]
        ),
        "players": pd.DataFrame([{"id": 7, "position": "SF"}]),
        "player_dict": {7: {"id": 7, "position": "SF"}},
    }
    monkeypatch.setattr(strict, "_build_historical_snapshot", lambda data, season: parent_snapshot.copy())
    snapshot = strict.build_historical_snapshot({}, 2022)
    assert snapshot["team_season_stats"]["season"].tolist() == [2022]
    assert snapshot["player_dict"][7]["position"] == "SG"
    assert snapshot["_as_of_season"] == 2022


def _competition_training_data() -> dict:
    rows = []
    for year in (2022, 2023, 2024):
        rows.append(
            {
                "player_id": 1, "team_id": 10, "league_id": 1,
                "season": year, "competition": "RS", "games_played": 25,
                "minutes_per_game": 25.0, "points": 12.0, "rating": 6.0 + (year - 2022) * 0.1,
            }
        )
        rows.append(
            {
                "player_id": 1, "team_id": 10, "league_id": 1,
                "season": year, "competition": "PO", "games_played": 8,
                "minutes_per_game": 27.0, "points": 14.0, "rating": 6.5 + (year - 2022) * 0.2,
            }
        )
    # Same player in another league must never create a cross-league pair.
    rows.append(
        {
            "player_id": 1, "team_id": 20, "league_id": 2,
            "season": 2024, "competition": "PO", "games_played": 6,
            "minutes_per_game": 20.0, "points": 9.0, "rating": 5.5,
        }
    )
    return {
        "players": pd.DataFrame(
            [{"id": 1, "position": "PG", "birth_date": "2000-01-01", "age": 24}]
        ),
        "player_stats": pd.DataFrame(rows),
        "team_player_relations": pd.DataFrame(),
        "leagues": pd.DataFrame(
            [
                {"id": 1, "name": "ITA1", "max_games": 30},
                {"id": 2, "name": "EL", "max_games": 34},
            ]
        ),
    }



def test_asof_prepare_features_accepts_rating_distributions():
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    model = AsOfPositionPerformanceModel()
    data = _competition_training_data()
    # The parent train() always forwards rating_distributions, even when None.
    # The strict override must preserve that public signature.
    X, y = model.prepare_features(data, rating_distributions=None)
    assert len(X) == len(y)


def test_training_pairs_are_same_league_and_same_competition():
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    model = AsOfPositionPerformanceModel()
    X, y = model.prepare_features(_competition_training_data())
    assert len(X) == 4
    assert len(y) == 4
    assert model._last_competitions.count("RS") == 2
    assert model._last_competitions.count("PO") == 2
    assert set(model._last_league_ids) == {1}
    assert model.competition_encoding["RS"] != model.competition_encoding["PO"]


def test_descriptive_splits_stay_observed_but_never_train(monkeypatch):
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    data = _competition_training_data()
    descriptive = []
    for year in (2022, 2023, 2024):
        for competition in ("Home", "Away"):
            descriptive.append(
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": year,
                    "competition": competition,
                    "games_played": 12,
                    "minutes_per_game": 25.0,
                    "points": 12.0,
                    "rating": 7.0,
                }
            )
    data["player_stats"] = pd.concat(
        [data["player_stats"], pd.DataFrame(descriptive)], ignore_index=True
    )
    observed_rows = len(data["player_stats"])

    model = AsOfPositionPerformanceModel()
    X, _ = model.prepare_features(data)

    assert len(data["player_stats"]) == observed_rows
    assert len(X) == 4
    assert set(model._last_competitions) == {"RS", "PO"}
    assert model._excluded_descriptive_rows == 6
    assert "HOME" not in model.competition_encoding
    assert "AWAY" not in model.competition_encoding


def test_persistence_features_and_delta_target_are_source_only():
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    model = AsOfPositionPerformanceModel()
    X, y = model.prepare_features(_competition_training_data())

    expected_features = {
        "last_rating",
        "rating_delta_1",
        "rating_delta_2",
        "recent_rating_mean",
        "recent_rating_std",
    }
    assert expected_features.issubset(X.columns)
    assert model.target_mode == "delta_vs_prior"

    # Two RS transitions (+0.1) followed by two PO transitions (+0.2).
    assert list(y) == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert X.iloc[0]["last_rating"] == pytest.approx(6.0)
    assert X.iloc[1]["last_rating"] == pytest.approx(6.1)
    assert X.iloc[1]["rating_delta_1"] == pytest.approx(0.1)
    assert X.iloc[1]["rating_delta_2"] == pytest.approx(0.0)
    assert X.iloc[1]["recent_rating_mean"] == pytest.approx(6.05)
    assert X.iloc[1]["recent_rating_std"] == pytest.approx(0.05)


def test_delta_target_is_reconstructed_to_absolute_rating(monkeypatch):
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    model = AsOfPositionPerformanceModel()
    model.is_trained = True
    monkeypatch.setattr(
        model,
        "predict_from_features",
        lambda feature_dict: 0.35,
    )

    prediction = model.predict_target_rating({"last_rating": 6.10})
    assert prediction == pytest.approx(6.45)


def test_delta_native_prediction_is_not_absolute_clipped():
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    class _Model:
        def predict(self, _features):
            return [-0.75]

    model = AsOfPositionPerformanceModel()
    model.is_trained = True
    model.target_mode = "delta_vs_prior"
    model.feature_names = ["last_rating"]
    model.model = _Model()

    assert model.predict_from_features({"last_rating": 6.10}) == pytest.approx(-0.75)


def test_forecast_pairs_use_minimum_games_and_reliability_weights(monkeypatch):
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    monkeypatch.setenv("MODEL_MIN_TRAIN_GAMES", "7")
    data = _competition_training_data()
    data["player_stats"].loc[
        (data["player_stats"]["competition"] == "PO")
        & (data["player_stats"]["season"] == 2023),
        "games_played",
    ] = 2

    model = AsOfPositionPerformanceModel()
    X, _ = model.prepare_features(data)

    assert len(X) == 2
    assert model._last_competitions == ["RS", "RS"]
    assert model._skipped_low_sample_pairs == 2
    assert all(0.25 <= weight <= 1.0 for weight in model._last_sample_weights)


def test_competition_aliases_are_open_and_normalized():
    from basketball_ai.models.competition_training import normalize_competition

    assert normalize_competition("playoffs") == "PO"
    assert normalize_competition("regular-season") == "RS"
    assert normalize_competition("Super Cup") == "SUPERCUP"
    assert normalize_competition("Final Four") == "FINAL_FOUR"


def test_scope_prediction_context_keeps_only_requested_competition():
    from basketball_ai.models.competition_training import scope_prediction_context

    stats = pd.DataFrame(
        [
            {"player_id": 1, "team_id": 10, "league_id": 1, "season": 2024, "competition": "RS", "games_played": 30, "rating": 7.0},
            {"player_id": 1, "team_id": 10, "league_id": 1, "season": 2024, "competition": "PO", "games_played": 8, "rating": 7.6},
        ]
    )
    team_history = pd.DataFrame(
        [
            {"team_id": 10, "global_id": "T10", "name": "Team", "league_id": 1, "league_key": "ITA1", "season": 2024, "competition": "RS", "pace": 75.0, "offensive_rating": 110.0, "defensive_rating": 108.0, "three_point_attempt_rate": 0.3, "assists_per_game": 20.0, "star_player_usage": 0.2, "net_rtg": 2.0},
            {"team_id": 10, "global_id": "T10", "name": "Team", "league_id": 1, "league_key": "ITA1", "season": 2024, "competition": "PO", "pace": 70.0, "offensive_rating": 106.0, "defensive_rating": 104.0, "three_point_attempt_rate": 0.25, "assists_per_game": 18.0, "star_player_usage": 0.3, "net_rtg": 2.0},
        ]
    )
    data = {
        "player_stats": stats,
        "team_season_stats": team_history,
        "teams": pd.DataFrame([{"id": 10, "global_id": "T10", "name": "Team", "league_id": 1}]),
        "players": pd.DataFrame([{"id": 1, "global_id": "P1", "position": "PG", "current_team_id": 10, "current_league_id": 1}]),
    }
    scoped = scope_prediction_context(data, 1, 10, 1, "PO", 2024)
    assert set(scoped["player_stats"]["competition"]) == {"PO"}
    assert set(scoped["team_season_stats"]["competition"]) == {"PO"}
    assert scoped["team_dict"][10]["pace"] == 70.0
    assert scoped["_competition_support"]["exact_source_games"] == 8


def test_scope_prediction_context_rejects_missing_exact_po_data():
    from basketball_ai.models.competition_training import scope_prediction_context

    data = {
        "player_stats": pd.DataFrame(
            [{"player_id": 1, "team_id": 10, "league_id": 1, "season": 2024, "competition": "RS", "games_played": 30, "rating": 7.0}]
        ),
        "team_season_stats": pd.DataFrame(
            [{"team_id": 10, "global_id": "T10", "name": "Team", "league_id": 1, "league_key": "ITA1", "season": 2024, "competition": "PO", "pace": 70.0, "offensive_rating": 106.0, "defensive_rating": 104.0, "three_point_attempt_rate": 0.25, "assists_per_game": 18.0, "star_player_usage": 0.3, "net_rtg": 2.0}]
        ),
        "teams": pd.DataFrame([{"id": 10, "name": "Team", "league_id": 1}]),
        "players": pd.DataFrame([{"id": 1, "position": "PG"}]),
    }
    with pytest.raises(ValueError, match="no isolated PO data"):
        scope_prediction_context(data, 1, 10, 1, "PO", 2024)


def test_future_only_role_is_not_in_training_vocabulary():
    from basketball_ai.models.strict_production import AsOfPositionPerformanceModel

    stats = pd.DataFrame(
        [
            {
                "player_id": 1, "team_id": 10, "league_id": 1, "season": year,
                "games_played": 20, "minutes_per_game": 25.0, "points": 10.0,
                "rating": 6.0 + index * 0.1, "competition": "RS",
                "ruolo_combinato": role, "ruolo_offensivo": role, "ruolo_difensivo": role,
            }
            for index, (year, role) in enumerate(
                [(2022, "ROLE_A"), (2023, "ROLE_B"), (2024, "FUTURE_ONLY")]
            )
        ]
    )
    data = {
        "players": pd.DataFrame([{"id": 1, "position": "PG", "birth_date": "2000-01-01", "age": 24}]),
        "player_stats": stats,
        "team_player_relations": pd.DataFrame(),
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1", "max_games": 30}]),
    }
    model = AsOfPositionPerformanceModel()
    model.prepare_features(data)
    assert "FUTURE_ONLY" not in model.role_encoding
    assert model._role_vocabulary_cutoff == 2023


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


def test_competition_specific_conformal_falls_back_only_when_sparse(monkeypatch):
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    monkeypatch.setenv("MODEL_MIN_COMPETITION_CALIBRATION_SAMPLES", "3")
    model = StrictProductionEnsembleModel()
    model._conformal_nominal_coverage = 0.90
    records = [
        {"competition": "PO", "prediction": 7.0, "actual": 7.1},
        {"competition": "PO", "prediction": 7.0, "actual": 7.3},
        {"competition": "PO", "prediction": 7.0, "actual": 7.2},
        {"competition": "CUP", "prediction": 7.0, "actual": 7.1},
    ]
    model._calibrate_competitions(records)
    assert "PO" in model._conformal_by_competition
    assert "CUP" not in model._conformal_by_competition
    assert model._conformal_samples_by_competition == {"PO": 3, "CUP": 1}


def test_none_season_means_latest_observed_source_season(monkeypatch):
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel()
    model._competition_encoding_state = {"RS": 0}
    seen = {}

    def fake_predict_uncached(player_id, team_id, data, season, target_age, competition):
        seen["season"] = season
        return SimpleNamespace(predicted_rating=7.0)

    monkeypatch.setattr(model, "_predict_uncached", fake_predict_uncached)
    data = {"player_stats": pd.DataFrame([{"season": 2024}, {"season": 2025}])}
    model.predict(1, 2, data, season=None, competition="RS")
    assert seen["season"] == 2025


def test_strict_model_rejects_competition_not_seen_at_training():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel()
    model._competition_encoding_state = {"RS": 0}
    with pytest.raises(ValueError, match="was not present when this model was trained"):
        model.predict(1, 2, {"player_stats": pd.DataFrame([{"season": 2025}])}, season=2025, competition="PO")


def test_team_features_use_as_of_roster_not_future_roster():
    from basketball_ai.features.team_features import compute_team_features

    team = {
        "id": 10, "name": "Team", "league_id": 1, "playing_style": "motion_offense",
        "formation": "", "pace": 75.0, "offensive_rating": 110.0,
        "defensive_rating": 108.0, "three_point_attempt_rate": 0.35,
        "assists_per_game": 20.0, "star_player_usage": 0.25, "league_tier": 1,
        "short_name": "T", "net_rtg": 2.0,
    }
    data = {
        "_as_of_season": 2022,
        "team_dict": {10: team},
        "league_dict": {1: {"id": 1, "competitiveness_score": 1.0}},
        "leagues": pd.DataFrame([{"id": 1, "competitiveness_score": 1.0}]),
        "team_player_relations": pd.DataFrame(
            [{"team_id": 10, "player_id": 1, "season": 2022}, {"team_id": 10, "player_id": 2, "season": 2025}]
        ),
        "player_stats": pd.DataFrame(
            [{"player_id": 1, "season": 2022, "rating": 6.0}, {"player_id": 2, "season": 2025, "rating": 10.0}]
        ),
    }
    features = compute_team_features(10, data)
    assert features["avg_teammate_rating"] == 6.0
    assert features["league_tier_factor"] == 1.0
def test_persistence_shrinkage_alpha_uses_prior_calibration_only():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    records = [
        {"actual": 7.2, "persistence_prediction": 7.0, "prediction": 8.0},
        {"actual": 6.8, "persistence_prediction": 7.0, "prediction": 6.0},
        {"actual": 7.1, "persistence_prediction": 7.0, "prediction": 7.5},
    ]
    result = StrictProductionEnsembleModel._fit_persistence_shrinkage(records)
    assert result["valid"] is True
    assert 0.0 <= result["alpha"] <= 1.0
    assert result["alpha"] == pytest.approx(0.2888888889)
    assert result["blended_rmse"] < result["model_rmse"]


def test_persistence_shrinkage_can_be_disabled_for_raw_diagnostics():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel(enable_persistence_shrinkage=False)
    assert model._enable_persistence_shrinkage is False
    assert model._persistence_shrinkage_alpha == 0.0


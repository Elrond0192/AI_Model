"""Regression tests for leakage-safe production ML behavior."""
from __future__ import annotations

from types import MethodType

import numpy as np
import pandas as pd


def _season_data(seasons: list[int]) -> dict:
    return {
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1", "max_games": 30}]),
        "players": pd.DataFrame(
            [
                {
                    "id": 1,
                    "global_id": "P1",
                    "name": "Player",
                    "position": "PG",
                    "birth_date": "2000-01-01",
                    "age": 26,
                }
            ]
        ),
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": season,
                    "games_played": 20,
                    "minutes_per_game": 25.0,
                    "points": 12.0 + (season - min(seasons)),
                    "rating": 6.0 + (season - min(seasons)) * 0.1,
                    "competition": "RS",
                }
                for season in seasons
            ]
        ),
    }


def test_prepare_features_uses_only_exact_consecutive_seasons():
    from basketball_ai.models.production_training import SeasonAheadPerformanceModel

    model = SeasonAheadPerformanceModel()
    X, y = model.prepare_features(_season_data([2021, 2022, 2024, 2025, 2026]))
    assert len(X) == 3
    assert len(y) == 3
    assert model._last_source_years == [2021, 2024, 2025]
    assert model._last_season_years == [2022, 2025, 2026]
    assert model._skipped_gap_pairs >= 1


def test_train_reorders_target_seasons_with_feature_rows(monkeypatch):
    import basketball_ai.models.production_training as production

    model = production.SeasonAheadPerformanceModel()

    def fake_prepare(self, data, extra_metrics=None, split_season=None,
                     rating_distributions=None):
        self._last_season_years = [2024, 2022, 2023]
        self._last_source_years = [2023, 2021, 2022]
        self._skipped_gap_pairs = 0
        return (
            pd.DataFrame({"marker": [2024.0, 2022.0, 2023.0]}),
            np.asarray([24.0, 22.0, 23.0]),
        )

    model.prepare_features = MethodType(fake_prepare, model)
    fitted = []

    class FakeXGB:
        best_iteration = 0

        def fit(self, X, y, **kwargs):
            fitted.append((X.copy(), np.asarray(y).copy(), kwargs))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(production, "_xgb", lambda *args, **kwargs: FakeXGB())
    monkeypatch.setattr(
        production,
        "compute_baselines",
        lambda *args, **kwargs: {
            "baseline_mean_val_rmse": 99.0,
            "baseline_linear_val_rmse": 99.0,
            "baseline_rf_val_rmse": 99.0,
        },
    )
    model.train({})

    selector_X, selector_y, _ = fitted[0]
    assert selector_X["marker"].tolist() == [2022.0]
    assert selector_y.tolist() == [22.0]
    eval_X, eval_y = fitted[0][2]["eval_set"][0]
    assert eval_X["marker"].tolist() == [2023.0]
    assert np.asarray(eval_y).tolist() == [23.0]
    assert model._split_metadata["calibration_season"] == 2024


def _snapshot_data() -> dict:
    return {
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1"}]),
        "teams": pd.DataFrame(
            [
                {
                    "id": 10,
                    "global_id": "T10",
                    "name": "Team",
                    "league_id": 1,
                    "league_key": "ITA1",
                    "pace": 99.0,
                    "offensive_rating": 120.0,
                    "defensive_rating": 100.0,
                }
            ]
        ),
        "players": pd.DataFrame(
            [
                {
                    "id": 1,
                    "global_id": "P1",
                    "name": "Player",
                    "position": "PG",
                    "birth_date": "2000-01-01",
                    "age": 25,
                    "current_team_id": 10,
                    "current_league_id": 1,
                }
            ]
        ),
        "player_stats": pd.DataFrame(
            [
                {"player_id": 1, "team_id": 10, "league_id": 1, "season": 2022, "rating": 6.0},
                {"player_id": 1, "team_id": 10, "league_id": 1, "season": 2023, "rating": 6.5},
                {"player_id": 1, "team_id": 10, "league_id": 1, "season": 2025, "rating": 7.0},
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [
                {"player_id": 1, "team_id": 10, "season": 2022},
                {"player_id": 1, "team_id": 10, "season": 2023},
                {"player_id": 1, "team_id": 10, "season": 2025},
            ]
        ),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 10,
                    "global_id": "T10",
                    "name": "Team",
                    "league_id": 1,
                    "league_key": "ITA1",
                    "season": 2022,
                    "pace": 70.0,
                    "offensive_rating": 105.0,
                    "defensive_rating": 108.0,
                    "three_point_attempt_rate": 0.30,
                    "assists_per_game": 18.0,
                    "star_player_usage": 0.20,
                    "net_rtg": -3.0,
                },
                {
                    "team_id": 10,
                    "global_id": "T10",
                    "name": "Team",
                    "league_id": 1,
                    "league_key": "ITA1",
                    "season": 2025,
                    "pace": 99.0,
                    "offensive_rating": 120.0,
                    "defensive_rating": 100.0,
                    "three_point_attempt_rate": 0.45,
                    "assists_per_game": 26.0,
                    "star_player_usage": 0.30,
                    "net_rtg": 20.0,
                },
            ]
        ),
        "league_dict": {1: {"id": 1, "name": "ITA1"}},
        "team_dict": {10: {"id": 10, "global_id": "T10", "league_id": 1}},
        "player_dict": {1: {"id": 1, "global_id": "P1", "position": "PG"}},
        "league_teams": {1: [10]},
    }


def test_historical_snapshot_excludes_future_player_and_team_state():
    from basketball_ai.models.production_training import build_historical_snapshot

    snapshot = build_historical_snapshot(_snapshot_data(), 2022)
    assert set(snapshot["player_stats"]["season"]) == {"2022"}
    assert snapshot["team_dict"][10]["pace"] == 70.0
    assert snapshot["team_dict"][10]["offensive_rating"] == 105.0
    assert snapshot["player_dict"][1]["age"] == 22
    assert snapshot["_as_of_season"] == 2022


def test_temporal_compatibility_selects_team_state_before_target():
    from basketball_ai.models.production_training import TemporalCompatibilityModel

    row = TemporalCompatibilityModel._team_row_as_of(
        10, _snapshot_data(), 2025, strict_before=True
    )
    assert row is not None
    assert int(row["season"]) == 2022
    assert row["pace"] == 70.0


def test_metric_summary_reports_error_bias_and_coverage():
    from basketball_ai.models.production_training import metric_summary

    summary = metric_summary(
        [
            {
                "actual": 7.0,
                "prediction": 7.2,
                "confidence_low": 6.8,
                "confidence_high": 7.5,
            },
            {
                "actual": 6.0,
                "prediction": 5.8,
                "confidence_low": 5.5,
                "confidence_high": 6.1,
            },
        ]
    )
    assert summary["n"] == 2
    assert np.isclose(summary["rmse"], 0.2)
    assert np.isclose(summary["mae"], 0.2)
    assert np.isclose(summary["bias"], 0.0)
    assert summary["interval_coverage"] == 1.0


def test_compatibility_score_is_centred_and_can_penalise():
    from basketball_ai.models.competition_training import CompetitionTemporalCompatibilityModel

    model = CompetitionTemporalCompatibilityModel()
    model.is_trained = True

    class Scaler:
        def transform(self, value):
            return value

    class KNN:
        def predict(self, value):
            return np.asarray([0.40])

    model.scaler = Scaler()
    model.knn = KNN()
    data = _snapshot_data()
    data["player_stats"] = pd.DataFrame(
        [
            {
                "player_id": 1, "team_id": 10, "league_id": 1,
                "season": 2024, "competition": "PO",
                "games_played": 8, "rating": 7.0,
            }
        ]
    )
    data["team_season_stats"]["competition"] = "PO"
    data["_prediction_league_id"] = 1
    data["_prediction_competition"] = "PO"
    data["_as_of_season"] = 2024

    assert model.score(1, 10, data) == 0.40


def test_two_way_score_preserves_defensive_sign():
    from basketball_ai.models.performance_model import PerformanceModel

    row = pd.Series(
        {
            "player_id": 1,
            "season": 2024,
            "rating": 7.0,
            "minutes_per_game": 25.0,
            "points": 15.0,
            "raptor_off": 2.0,
            "raptor_def": -3.0,
            "competition": "RS",
        }
    )
    model = PerformanceModel()
    features = model._build_row(
        row,
        age=24,
        position="PG",
        player_stats_history=pd.DataFrame([row]),
        league_max_games={1: 30},
    )
    assert np.isclose(features["two_way_score"], -1.0)


def test_empirical_age_curve_state_round_trips():
    from basketball_ai.models.age_curve import (
        get_fitted_params,
        reset_fitted_params,
        set_fitted_params,
    )

    state = {
        "peak_ages": {"PG": 27.0},
        "sigma_before": {"PG": 4.2},
        "sigma_after": {"PG": 3.8},
    }
    set_fitted_params(state)
    assert get_fitted_params() == state
    reset_fitted_params()


def test_competition_model_uses_one_hot_nominal_features():
    from basketball_ai.models.competition_training import CompetitionSeasonAheadPerformanceModel

    data = _season_data([2021, 2022, 2023, 2024])
    stats = data["player_stats"].copy()
    stats["ruolo_combinato"] = ["ROLE_A", "ROLE_B", "ROLE_A", "ROLE_B"]
    stats["ruolo_offensivo"] = ["OFF_A", "OFF_A", "OFF_B", "OFF_B"]
    stats["ruolo_difensivo"] = ["DEF_A", "DEF_B", "DEF_A", "DEF_B"]
    data["player_stats"] = stats

    model = CompetitionSeasonAheadPerformanceModel()
    X, _ = model.prepare_features(data)
    assert "competition_enc" not in X.columns
    assert "role_enc" not in X.columns
    assert "role_off_enc" not in X.columns
    assert "role_def_enc" not in X.columns
    assert "competition_1" in X.columns
    assert "role_combo_1" in X.columns
    assert "role_off_1" in X.columns
    assert "role_def_1" in X.columns

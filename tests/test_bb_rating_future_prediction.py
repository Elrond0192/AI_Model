import pandas as pd
import pytest

from basketball_ai.bb_rating.future_prediction import (
    BBRatingFuturePredictionModel,
    _team_context_features,
    _team_context_index,
    _target_league_tier,
)
from basketball_ai.bb_rating.prediction_scenarios import classify_transition


def _synthetic_data():
    players = []
    stats = []
    team_stats = []
    positions = ["PG", "SG", "SF", "PF", "C"]
    for player_id in range(1, 61):
        players.append({
            "id": player_id,
            "global_id": f"P{player_id}",
            "name": f"Player {player_id}",
            "birth_date": f"{1988 + player_id % 12}-01-01",
            "position": positions[player_id % len(positions)],
        })
        for season in range(2020, 2025):
            v = float(player_id % 60)
            stats.append({
                "player_id": player_id,
                "league_id": 1,
                "league_key": "ITA1",
                "season": season,
                "competition": "RS",
                "team_id": player_id % 2 + 1,
                "games_played": 20 + player_id % 10,
                "raptor_total": v / 10 + (season - 2020) * 0.1,
                "vorp": (player_id % 13) / 5,
                "per": 10 + player_id % 20,
                "ts_pct": 0.45 + (player_id % 15) * 0.01,
                "ast_pct": 0.05 + (player_id % 20) * 0.01,
                "tov_pct": 0.05 + (player_id % 10) * 0.01,
                "raptor_def": -2 + (player_id % 12) * 0.3,
                "net_rtg_diff": -5 + (player_id % 25) * 0.4,
                "on_net_rtg": 100 + player_id % 10,
                "off_net_rtg": 99,
                "stl_pct": 0.01 + (player_id % 10) * 0.001,
                "blk_pct": 0.01 + (player_id % 10) * 0.001,
                "reb_pct": 0.05 + (player_id % 20) * 0.01,
                "hustle_index": 10 + player_id % 30,
                "foul_drawing_rate": 0.1 + (player_id % 10) * 0.01,
            })
    for season in range(2020, 2025):
        for team_id in (1, 2):
            team_stats.append({
                "team_id": team_id, "league_id": 1, "season": season,
                "competition": "RS", "global_id": f"TEAM{team_id}",
                "pace": 70 + season - 2020, "offensive_rating": 105 + team_id,
                "defensive_rating": 108 - team_id, "three_point_attempt_rate": 0.35,
            })
    return {
        "players": pd.DataFrame(players),
        "player_stats": pd.DataFrame(stats),
        "team_season_stats": pd.DataFrame(team_stats),
        "leagues": pd.DataFrame([{
            "id": 1, "name": "ITA1", "league_key": "ITA1", "tier": 2,
        }]),
    }


def test_target_league_tier_uses_local_registry_over_stale_db_values():
    data = {
        "leagues": pd.DataFrame([
            {"id": 1, "name": "LIT1", "league_key": "LIT1", "tier": 1},
            {"id": 2, "name": "ITA2", "league_key": "ITA2", "tier": 1},
        ])
    }

    assert _target_league_tier(data, "EL") == 1
    assert _target_league_tier(data, "LIT1") == 2
    assert _target_league_tier(data, "ITA2") == 3


def test_competition_phase_transitions_are_distinguished():
    assert classify_transition("A", "A", "ITA1", "ITA1", "RS", "PO") == (
        "competition_change_same_team_same_league"
    )
    assert classify_transition("A", "B", "ITA1", "ITA1", "RS", "CUP") == (
        "team_and_competition_change_same_league"
    )
    assert classify_transition("A", "B", "ITA1", "EL", "RS", "PO") == (
        "team_league_and_competition_change"
    )


def test_direct_bb_rating_model_trains_oof_predicts_and_roundtrips(tmp_path):
    data = _synthetic_data()
    model = BBRatingFuturePredictionModel(min_oof_samples=20, random_state=7)
    metadata = model.fit(data)

    assert metadata["target"] == "official_bb_rating_t_plus_1"
    assert metadata["target_scale"] == [1, 100]
    assert metadata["oof_samples"] >= 20
    assert "target_team__pace" in model.feature_columns
    assert metadata["walk_forward_target_seasons"] == sorted(metadata["walk_forward_target_seasons"])
    assert metadata["walk_forward_fold_count"] == len(metadata["walk_forward_target_seasons"])
    assert metadata["minimum_walk_forward_target_seasons"] == 3
    assert "ITA1" in metadata["source_league_metrics"]
    assert any(key.startswith("ITA1=>ITA1::") for key in metadata["source_destination_scenario_metrics"])
    # A small fixture must not be allowed to pass promotion merely on row count.
    assert metadata["promotion_gate_passed"] is False
    assert metadata["oof_mae_after_calibration"] >= 0

    prediction = model.predict_from_data(
        data, player_global_id="P1", source_season=2022,
        target_league="ITA1", target_competition="RS",
        target_team_global_id="TEAM2",
    )
    assert 1 <= prediction["source_bb_rating"] <= 100
    assert 1 <= prediction["predicted_bb_rating"] <= 100
    assert 1 <= prediction["confidence_low_100"] <= prediction["confidence_high_100"] <= 100
    assert prediction["target_scale"] == "BB-Rating 1-100"

    model.save(tmp_path)
    loaded = BBRatingFuturePredictionModel.load(tmp_path)
    restored = loaded.predict_from_data(
        data, player_global_id="P1", source_season=2022,
        target_league="ITA1", target_competition="RS",
        target_team_global_id="TEAM2",
    )
    assert restored["predicted_bb_rating"] == pytest.approx(
        prediction["predicted_bb_rating"], abs=0.1
    )



def test_destination_team_context_uses_only_information_as_of_source_season():
    data = _synthetic_data()
    context = _team_context_features(
        _team_context_index(data), "TEAM2", "ITA1", source_year=2022
    )
    assert context["target_team__pace"] == pytest.approx(72.0)
    assert context["target_team_context_seasons_ago"] == pytest.approx(0.0)


def test_bad_transfer_segment_uses_context_baseline_fallback():
    import numpy as np

    class _ConstantModel:
        def predict(self, _encoded):
            return np.asarray([80.0])

    class _ConstantCalibrator:
        def predict(self, _score, *, scenario, target_league_id):
            return {
                "predicted_bb_rating": 75.0,
                "calibration_level": "destination_league+scenario",
                "calibration_samples": 500,
                "scenario": scenario,
                "target_league_id": target_league_id,
            }

    model = BBRatingFuturePredictionModel()
    model.is_trained = True
    model.model = _ConstantModel()
    model.calibrator = _ConstantCalibrator()
    model.feature_columns = []
    model.residual_intervals = {
        "global": {"n": 100, "low": -10.0, "high": 10.0},
        "LIT1::team_league_and_competition_change_same_tier": {
            "n": 100, "low": -12.0, "high": 12.0,
        },
    }
    model.baseline_means = {
        "by_league_competition": {"LIT1::RS": 51.0},
        "by_league": {"LIT1": 52.0},
        "global": 50.0,
    }
    model.scenario_fallback_policy = {
        "minimum_samples": 40,
        "group_dimensions": {
            "source_destination_scenario": {
                "EC=>LIT1::team_league_and_competition_change_same_tier": {
                    "n": 418, "mae": 23.1, "baseline_mae": 14.9,
                    "allowed_mae": 16.6, "use_model": False,
                },
            },
            "destination_league_scenario": {},
            "scenario": {},
            "source_league": {},
            "destination_league": {},
        },
        "unknown_scenario_always_fallback": True,
        "unvalidated_cross_context_fallback": True,
    }

    prediction = model._predict_features({
        "source_league": "EC",
        "source_competition": "RS",
        "source_position": "G",
        "source_role": "Guard",
        "source_tier": "1",
        "target_league": "LIT1",
        "target_competition": "RS",
        "target_tier": "2",
        "tier_transition": "same_tier",
        "target_team_context_known": "no",
        "target_team_context_league": "",
        "scenario": "team_league_and_competition_change_same_tier",
        "source_bb_rating": 74.0,
    })

    assert prediction["predicted_bb_rating"] == pytest.approx(51.0)
    assert prediction["baseline_bb_rating"] == pytest.approx(51.0)
    assert prediction["prediction_method"] == "context_baseline_fallback"
    assert prediction["fallback_reason"] == "validation_gate_failed:source_destination_scenario"

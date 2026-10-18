import pandas as pd
import pytest

from basketball_ai.bb_rating.future_prediction import BBRatingFuturePredictionModel
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
            })
    return {
        "players": pd.DataFrame(players),
        "player_stats": pd.DataFrame(stats),
        "team_season_stats": pd.DataFrame(team_stats),
        "leagues": pd.DataFrame([{
            "id": 1, "name": "ITA1", "league_key": "ITA1", "tier": 2,
        }]),
    }


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

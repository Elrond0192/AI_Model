"""Tests for competition-aware model features (RS / PO / CUP / SUPERCUP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from basketball_ai.data.loader import load_all_data
from basketball_ai.data.models import PlayerStats
from basketball_ai.models.performance_model import (
    COMPETITION_ENCODING,
    FEATURE_COLS,
    PerformanceModel,
    compute_po_features,
)
from basketball_ai.models.ensemble import EnsembleModel, PredictionResult
from basketball_ai.scenarios.engine import WhatIfEngine


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_stat_row(**overrides) -> dict:
    base = dict(
        player_id=1, season="2023-24", team_id=1, league_id=1,
        games_played=70, minutes_per_game=32.0,
        points=18.0, rebounds=5.0, offensive_rebounds=1.5,
        defensive_rebounds=3.5, assists=6.0, steals=1.2,
        blocks=0.4, turnovers=2.5, personal_fouls=2.0,
        fg_pct=0.47, three_point_pct=0.36, ft_pct=0.82,
        plus_minus=3.0, per=18.0, ts_pct=0.58,
        usg_pct=22.0, bpm=1.5, vorp=2.0,
        win_shares=5.0, ast_ratio=18.0, reb_pct=8.0,
        rating=7.5, competition="RS",
    )
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def tiny_data_with_competition():
    """Small dataset that includes both RS and PO rows for each player."""
    np.random.seed(99)

    leagues = pd.DataFrame([{
        "id": 1, "name": "NBA", "country": "USA", "tier": 1,
        "competitiveness_score": 1.0, "avg_pace": 100.0, "avg_offensive_rating": 113.0,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i+1}", "league_id": 1,
        "playing_style": ["pace_and_space", "pick_and_roll", "isolation",
                          "defensive", "motion_offense", "post_up"][i % 6],
        "formation": "small_ball",
        "pace": 95.0 + i * 2,
        "offensive_rating": 108.0 + i,
        "defensive_rating": 107.0 + i,
        "three_point_attempt_rate": 0.35 + i * 0.01,
        "assists_per_game": 24.0 + i,
        "star_player_usage": 0.28,
        "league_tier": 1,
    } for i in range(6)])

    players = pd.DataFrame([{
        "id": i + 1,
        "name": f"Player {i+1}",
        "age": 22 + (i % 12),
        "position": ["PG", "SG", "SF", "PF", "C", "PG/SG"][i % 6],
        "nationality": "American",
        "height_cm": 190 + i % 20,
        "weight_kg": 90 + i % 30,
        "dominant_hand": "right",
        "current_team_id": (i % 6) + 1,
        "current_league_id": 1,
        "draft_year": None, "draft_pick": None,
    } for i in range(30)])

    _ROLES_COMBO = ["playmaker", "scorer", "forward", "big", "wing"]
    _ROLES_OFF   = ["scorer", "facilitator", "spot_up", "post", "cutter"]
    _ROLES_DEF   = ["lockdown", "stopper", "help_side", "rim_protector", "versatile"]
    stat_rows = []
    for _, p in players.iterrows():
        pid = int(p["id"])
        # assign a per-player PO tendency (positive = improves in PO)
        rng = np.random.default_rng(pid)
        po_tendency = float(rng.normal(0.0, 0.15))
        for season in ["2021-22", "2022-23", "2023-24"]:
            mpg = float(rng.uniform(18, 36))
            rs_rating = float(rng.uniform(5.0, 9.0))
            base = dict(
                player_id=pid, season=season,
                team_id=int(p["current_team_id"]), league_id=1,
                games_played=int(rng.integers(40, 80)),
                minutes_per_game=round(mpg, 1),
                points=round(float(rng.uniform(8, 26)), 1),
                rebounds=round(float(rng.uniform(2, 12)), 1),
                offensive_rebounds=round(float(rng.uniform(0.5, 3)), 1),
                defensive_rebounds=round(float(rng.uniform(2, 9)), 1),
                assists=round(float(rng.uniform(1, 8)), 1),
                steals=round(float(rng.uniform(0.3, 2.0)), 2),
                blocks=round(float(rng.uniform(0.1, 2.5)), 2),
                turnovers=round(float(rng.uniform(0.5, 4.0)), 1),
                personal_fouls=round(float(rng.uniform(1, 4)), 1),
                fg_pct=round(float(rng.uniform(0.38, 0.58)), 3),
                three_point_pct=round(float(rng.uniform(0.28, 0.45)), 3),
                ft_pct=round(float(rng.uniform(0.65, 0.90)), 3),
                plus_minus=round(float(rng.uniform(-8, 8)), 1),
                per=round(float(rng.uniform(10, 25)), 2),
                ts_pct=round(float(rng.uniform(0.50, 0.65)), 3),
                usg_pct=round(float(rng.uniform(14, 30)), 2),
                bpm=round(float(rng.uniform(-3, 6)), 2),
                obpm=round(float(rng.uniform(-2, 4)), 2),
                dbpm=round(float(rng.uniform(-2, 3)), 2),
                vorp=round(float(rng.uniform(-0.5, 4)), 2),
                win_shares=round(float(rng.uniform(0, 12)), 2),
                ast_ratio=round(float(rng.uniform(5, 30)), 2),
                reb_pct=round(float(rng.uniform(3, 20)), 2),
                tov_pct=round(float(rng.uniform(8, 20)), 1),
                ast_pct=round(float(rng.uniform(5, 30)), 1),
                orb_pct=round(float(rng.uniform(1, 8)), 1),
                drb_pct=round(float(rng.uniform(5, 25)), 1),
                ruolo_combinato=_ROLES_COMBO[pid % len(_ROLES_COMBO)],
                ruolo_offensivo=_ROLES_OFF[pid % len(_ROLES_OFF)],
                ruolo_difensivo=_ROLES_DEF[pid % len(_ROLES_DEF)],
            )
            # RS row
            stat_rows.append({**base, "rating": round(rs_rating, 3), "competition": "RS"})
            # PO row with modified rating based on playoff tendency
            po_rating = float(np.clip(rs_rating + po_tendency + rng.normal(0, 0.05), 3.5, 10.0))
            stat_rows.append({
                **base,
                "games_played": int(rng.integers(5, 20)),
                "rating": round(po_rating, 3),
                "competition": "PO",
            })

    stats = pd.DataFrame(stat_rows)
    rels = pd.DataFrame([{
        "team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
        "season": "2023-24",
        "role": "starter" if i % 3 == 0 else "rotation",
        "jersey_number": (i % 99) + 1,
    } for i, (_, p) in enumerate(players.iterrows())])

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict   = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}
    league_teams: dict = {}
    for _, t in teams.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    return {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": league_teams,
    }


# ---------------------------------------------------------------------------
# COMPETITION_ENCODING
# ---------------------------------------------------------------------------

class TestCompetitionEncoding:
    def test_rs_encodes_to_zero(self):
        assert COMPETITION_ENCODING["RS"] == 0

    def test_po_encodes_to_one(self):
        assert COMPETITION_ENCODING["PO"] == 1

    def test_cup_and_supercup_encoded(self):
        assert COMPETITION_ENCODING["CUP"] == 2
        assert COMPETITION_ENCODING["SUPERCUP"] == 3

    def test_feature_cols_contains_competition_features(self):
        for feat in ("competition_enc", "po_vs_rs_delta", "po_games_played"):
            assert feat in FEATURE_COLS, f"Missing: {feat}"

    def test_compute_po_features_returns_delta(self):
        stats = pd.DataFrame([
            {"competition": "RS", "rating": 7.0, "games_played": 70},
            {"competition": "RS", "rating": 7.5, "games_played": 72},
            {"competition": "PO", "rating": 8.0, "games_played": 12},
        ])
        result = compute_po_features(stats)
        assert abs(result["po_vs_rs_delta"] - (8.0 - 7.25)) < 0.01
        assert result["po_games_played"] == 12.0

    def test_compute_po_features_no_po_data_returns_zero_delta(self):
        stats = pd.DataFrame([
            {"competition": "RS", "rating": 7.0, "games_played": 70},
        ])
        result = compute_po_features(stats)
        assert result["po_vs_rs_delta"] == 0.0

    def test_compute_po_features_missing_competition_column_returns_zeros(self):
        stats = pd.DataFrame([{"rating": 7.0, "games_played": 70}])
        result = compute_po_features(stats)
        assert result["po_vs_rs_delta"] == 0.0
        assert result["po_games_played"] == 0.0


# ---------------------------------------------------------------------------
# PlayerStats model
# ---------------------------------------------------------------------------

class TestPlayerStatsCompetitionField:
    def test_default_is_rs(self):
        s = PlayerStats(
            player_id=1, season="2023-24", team_id=1, league_id=1,
            games_played=70, minutes_per_game=32.0, points=18.0,
            rebounds=5.0, offensive_rebounds=1.5, defensive_rebounds=3.5,
            assists=6.0, steals=1.2, blocks=0.4, turnovers=2.5,
            personal_fouls=2.0, fg_pct=0.47, three_point_pct=0.36,
            ft_pct=0.82, plus_minus=3.0, per=18.0, ts_pct=0.58,
            usg_pct=22.0, bpm=1.5, vorp=2.0, win_shares=5.0,
            ast_ratio=18.0, reb_pct=8.0, rating=7.5,
        )
        assert s.competition == "RS"

    def test_po_can_be_set(self):
        s = PlayerStats(
            player_id=1, season="2023-24", team_id=1, league_id=1,
            games_played=15, minutes_per_game=34.0, points=20.0,
            rebounds=6.0, offensive_rebounds=1.5, defensive_rebounds=4.5,
            assists=7.0, steals=1.5, blocks=0.5, turnovers=2.0,
            personal_fouls=1.8, fg_pct=0.49, three_point_pct=0.38,
            ft_pct=0.85, plus_minus=5.0, per=21.0, ts_pct=0.61,
            usg_pct=24.0, bpm=2.5, vorp=0.8, win_shares=2.0,
            ast_ratio=20.0, reb_pct=9.0, rating=8.0,
            competition="PO",
        )
        assert s.competition == "PO"


# ---------------------------------------------------------------------------
# PerformanceModel – competition features in training
# ---------------------------------------------------------------------------

class TestPerformanceModelCompetitionFeatures:
    def test_build_row_includes_competition_enc(self, tiny_data_with_competition):
        model = PerformanceModel()
        stats_df = tiny_data_with_competition["player_stats"]
        # Grab a PO row for player 1
        po_row = stats_df[
            (stats_df["player_id"] == 1) & (stats_df["competition"] == "PO")
        ].iloc[0]
        history = stats_df[stats_df["player_id"] == 1]
        row = model._build_row(po_row, 25, "PG", history)
        assert "competition_enc" in row
        assert row["competition_enc"] == float(COMPETITION_ENCODING["PO"])

    def test_build_row_computes_po_vs_rs_delta(self, tiny_data_with_competition):
        model = PerformanceModel()
        stats_df = tiny_data_with_competition["player_stats"]
        rs_row = stats_df[
            (stats_df["player_id"] == 1) & (stats_df["competition"] == "RS")
        ].iloc[0]
        history = stats_df[stats_df["player_id"] == 1]
        row = model._build_row(rs_row, 25, "PG", history)
        assert "po_vs_rs_delta" in row
        assert isinstance(row["po_vs_rs_delta"], float)

    def test_build_row_po_games_played_non_negative(self, tiny_data_with_competition):
        model = PerformanceModel()
        stats_df = tiny_data_with_competition["player_stats"]
        rs_row = stats_df[
            (stats_df["player_id"] == 1) & (stats_df["competition"] == "RS")
        ].iloc[0]
        history = stats_df[stats_df["player_id"] == 1]
        row = model._build_row(rs_row, 25, "PG", history)
        assert row["po_games_played"] >= 0.0

    def test_train_succeeds_with_competition_column(self, tiny_data_with_competition):
        model = PerformanceModel()
        metrics = model.train(tiny_data_with_competition)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] < 5.0


# ---------------------------------------------------------------------------
# EnsembleModel – competition parameter
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_ensemble(tiny_data_with_competition):
    e = EnsembleModel()
    e.train(tiny_data_with_competition)
    return e


class TestEnsembleCompetition:
    def test_prediction_result_has_competition_field(
        self, trained_ensemble, tiny_data_with_competition
    ):
        result = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition="RS")
        assert hasattr(result, "competition")
        assert result.competition == "RS"

    def test_po_prediction_result_competition_is_po(
        self, trained_ensemble, tiny_data_with_competition
    ):
        result = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition="PO")
        assert result.competition == "PO"

    def test_predictions_differ_across_competitions(
        self, trained_ensemble, tiny_data_with_competition
    ):
        rs = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition="RS")
        po = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition="PO")
        # RS and PO use different competition_enc values → ratings should differ
        assert rs.predicted_rating != po.predicted_rating

    def test_explanation_contains_competition_label(
        self, trained_ensemble, tiny_data_with_competition
    ):
        for comp in ("RS", "PO", "CUP"):
            r = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition=comp)
            # The explanation encodes competition in the narrative: "competizione: RS" etc.
            assert f"competizione: {comp}" in r.explanation

    def test_predicted_rating_in_range(
        self, trained_ensemble, tiny_data_with_competition
    ):
        for comp in ("RS", "PO"):
            r = trained_ensemble.predict(1, 1, tiny_data_with_competition, competition=comp)
            assert 3.5 <= r.predicted_rating <= 10.0


# ---------------------------------------------------------------------------
# WhatIfEngine – competition propagation
# ---------------------------------------------------------------------------

class TestWhatIfEngineCompetition:
    def test_predict_in_team_with_competition(
        self, trained_ensemble, tiny_data_with_competition
    ):
        engine = WhatIfEngine(trained_ensemble, tiny_data_with_competition)
        rs_r = engine.predict_in_team(1, 1, competition="RS")
        po_r = engine.predict_in_team(1, 1, competition="PO")
        assert isinstance(rs_r, PredictionResult)
        assert isinstance(po_r, PredictionResult)
        assert rs_r.competition == "RS"
        assert po_r.competition == "PO"

    def test_compare_scenarios_competition_in_results(
        self, trained_ensemble, tiny_data_with_competition
    ):
        engine = WhatIfEngine(trained_ensemble, tiny_data_with_competition)
        result = engine.compare_scenarios(1, [1, 2, 3], competition="PO")
        for s in result.scenarios:
            assert s["competition"] == "PO"

    def test_simulate_transfer_competition_param(
        self, trained_ensemble, tiny_data_with_competition
    ):
        engine = WhatIfEngine(trained_ensemble, tiny_data_with_competition)
        result = engine.simulate_transfer(1, 1, 2, competition="PO")
        assert isinstance(result.rating_delta, float)


# ---------------------------------------------------------------------------
# load_all_data – competition column defaults
# ---------------------------------------------------------------------------

class TestLoaderCompetitionColumn:
    def test_load_all_data_adds_competition_default_when_missing(self, tmp_path):
        pd.DataFrame([{"id": 1, "name": "NBA", "country": "USA", "tier": 1,
                        "competitiveness_score": 1.0, "avg_pace": 100.0,
                        "avg_offensive_rating": 113.0}]).to_csv(
            tmp_path / "leagues.csv", index=False
        )
        pd.DataFrame([{"id": 1, "name": "T1", "league_id": 1,
                        "playing_style": "pace_and_space", "formation": "small_ball",
                        "pace": 98.0, "offensive_rating": 112.0, "defensive_rating": 108.0,
                        "three_point_attempt_rate": 0.42, "assists_per_game": 26.0,
                        "star_player_usage": 0.30, "league_tier": 1}]).to_csv(
            tmp_path / "teams.csv", index=False
        )
        pd.DataFrame([{"id": 1, "name": "Player", "age": 25, "position": "PG",
                        "nationality": "US", "height_cm": 192, "weight_kg": 90,
                        "dominant_hand": "right",
                        "current_team_id": 1, "current_league_id": 1,
                        "draft_year": 2020, "draft_pick": 5}]).to_csv(
            tmp_path / "players.csv", index=False
        )
        # player_stats WITHOUT a competition column (legacy format)
        pd.DataFrame([{"player_id": 1, "season": "2023-24", "team_id": 1, "league_id": 1,
                        "games_played": 70, "minutes_per_game": 32.0,
                        "points": 18.0, "rebounds": 5.0, "rating": 7.5}]).to_csv(
            tmp_path / "player_stats.csv", index=False
        )
        pd.DataFrame([{"team_id": 1, "player_id": 1, "season": "2023-24",
                        "role": "starter", "jersey_number": 7}]).to_csv(
            tmp_path / "team_player_relations.csv", index=False
        )

        data = load_all_data(str(tmp_path))
        assert "competition" in data["player_stats"].columns
        assert (data["player_stats"]["competition"] == "RS").all()

    def test_load_all_data_preserves_existing_competition_column(self, tmp_path):
        pd.DataFrame([{"id": 1, "name": "NBA", "country": "USA", "tier": 1,
                        "competitiveness_score": 1.0, "avg_pace": 100.0,
                        "avg_offensive_rating": 113.0}]).to_csv(
            tmp_path / "leagues.csv", index=False
        )
        pd.DataFrame([{"id": 1, "name": "T1", "league_id": 1,
                        "playing_style": "pace_and_space", "formation": "small_ball",
                        "pace": 98.0, "offensive_rating": 112.0, "defensive_rating": 108.0,
                        "three_point_attempt_rate": 0.42, "assists_per_game": 26.0,
                        "star_player_usage": 0.30, "league_tier": 1}]).to_csv(
            tmp_path / "teams.csv", index=False
        )
        pd.DataFrame([{"id": 1, "name": "Player", "age": 25, "position": "PG",
                        "nationality": "US", "height_cm": 192, "weight_kg": 90,
                        "dominant_hand": "right",
                        "current_team_id": 1, "current_league_id": 1,
                        "draft_year": 2020, "draft_pick": 5}]).to_csv(
            tmp_path / "players.csv", index=False
        )
        # player_stats WITH an explicit competition column containing PO
        pd.DataFrame([{"player_id": 1, "season": "2023-24", "team_id": 1, "league_id": 1,
                        "games_played": 12, "minutes_per_game": 34.0,
                        "points": 22.0, "rebounds": 6.0, "rating": 8.2,
                        "competition": "PO"}]).to_csv(
            tmp_path / "player_stats.csv", index=False
        )
        pd.DataFrame([{"team_id": 1, "player_id": 1, "season": "2023-24",
                        "role": "starter", "jersey_number": 7}]).to_csv(
            tmp_path / "team_player_relations.csv", index=False
        )

        data = load_all_data(str(tmp_path))
        assert data["player_stats"]["competition"].iloc[0] == "PO"

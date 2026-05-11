"""Tests for basketball feature engineering."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data.models import Player, PlayerStats
from src.features.player_features import (
    compute_player_features_from_objects,
    compute_form_score,
    compute_consistency_score,
    compute_career_trajectory,
    POSITIONAL_PEAK_AGES,
    _peak_age,
)
from src.features.team_features import (
    compute_team_style_vector,
    get_style_position_compat,
    style_vector_similarity,
)
from src.features.context_features import (
    compute_league_adaptation,
    compute_context_features,
    _default_context_features,
)
from src.models.age_curve import age_performance_factor, PEAK_AGES, peak_age_window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player(pos: str = "PG", age: int = 25) -> Player:
    return Player(
        id=1, name="Test Player", age=age, position=pos,
        nationality="American", height_cm=193, weight_kg=90,
        dominant_hand="right",
        current_team_id=1, current_league_id=1,
        draft_year=2020, draft_pick=5,
    )


def _make_stat(season: str, rating: float, **kwargs) -> PlayerStats:
    defaults = dict(
        player_id=1, team_id=1, league_id=1,
        games_played=70, minutes_per_game=32.0,
        points=18.0, rebounds=5.0, offensive_rebounds=1.5,
        defensive_rebounds=3.5, assists=6.0, steals=1.2,
        blocks=0.4, turnovers=2.5, personal_fouls=2.0,
        fg_pct=0.47, three_point_pct=0.36, ft_pct=0.82,
        plus_minus=3.0, per=18.0, ts_pct=0.58,
        usg_pct=22.0, bpm=1.5, vorp=2.0,
        win_shares=5.0, ast_ratio=18.0, reb_pct=8.0,
    )
    defaults.update(kwargs)
    return PlayerStats(season=season, rating=rating, **defaults)


# ---------------------------------------------------------------------------
# Player features
# ---------------------------------------------------------------------------

class TestPlayerFeatures:
    def test_basic_output_keys(self):
        player = _make_player("PG", 25)
        stats  = [_make_stat("2023-24", 7.5)]
        feats  = compute_player_features_from_objects(player, stats)
        required = [
            "form_score", "consistency_score", "pts_per_36", "ast_per_36",
            "reb_per_36", "avg_per", "avg_ts_pct", "avg_usg_pct", "avg_bpm",
            "peak_rating", "career_trajectory", "age_vs_peak_age",
            "positional_peak_age", "scoring_profile", "playmaking_score",
            "defensive_score", "versatility_score",
        ]
        for k in required:
            assert k in feats, f"Missing key: {k}"

    def test_form_score_recent_bias(self):
        player = _make_player()
        s1 = _make_stat("2021-22", 6.0)
        s2 = _make_stat("2022-23", 7.0)
        s3 = _make_stat("2023-24", 8.5)
        feats = compute_player_features_from_objects(player, [s1, s2, s3])
        # With weights [0.2, 0.3, 0.5] → 6*0.2 + 7*0.3 + 8.5*0.5 = 7.55
        assert feats["form_score"] == pytest.approx(7.55, abs=0.01)

    def test_consistency_score_range(self):
        player = _make_player()
        stats  = [_make_stat(f"202{i}-2{i+1}", 7.0 + i * 0.1) for i in range(4)]
        feats  = compute_player_features_from_objects(player, stats)
        assert 0.0 <= feats["consistency_score"] <= 1.0

    def test_career_trajectory_positive(self):
        player = _make_player()
        stats  = [_make_stat(f"202{i}-2{i+1}", 6.0 + i) for i in range(4)]
        feats  = compute_player_features_from_objects(player, stats)
        assert feats["career_trajectory"] > 0

    def test_empty_stats_returns_defaults(self):
        player = _make_player()
        feats  = compute_player_features_from_objects(player, [])
        assert feats["form_score"] == 5.0
        assert feats["consistency_score"] == 0.5

    def test_hybrid_position_peak_age(self):
        player = _make_player("PF/C", 27)
        stats  = [_make_stat("2023-24", 7.0)]
        feats  = compute_player_features_from_objects(player, stats)
        assert feats["positional_peak_age"] == POSITIONAL_PEAK_AGES["PF/C"]
        assert feats["age_vs_peak_age"] == 27 - POSITIONAL_PEAK_AGES["PF/C"]

    def test_all_positions_have_peak_age(self):
        for pos in ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]:
            assert _peak_age(pos) >= 24


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

class TestFeatureHelpers:
    def _make_df(self, ratings):
        return pd.DataFrame({
            "season": [f"202{i}-2{i+1}" for i in range(len(ratings))],
            "rating": ratings,
        })

    def test_form_score_single(self):
        df = self._make_df([7.0])
        assert compute_form_score(df) == pytest.approx(7.0)

    def test_form_score_three(self):
        df = self._make_df([6.0, 7.0, 8.0])
        expected = 6 * 0.2 + 7 * 0.3 + 8 * 0.5
        assert compute_form_score(df) == pytest.approx(expected, abs=0.01)

    def test_consistency_perfect(self):
        df = self._make_df([7.0, 7.0, 7.0, 7.0])
        assert compute_consistency_score(df) == pytest.approx(1.0)

    def test_trajectory_increasing(self):
        df = self._make_df([5.0, 6.0, 7.0, 8.0])
        assert compute_career_trajectory(df) > 0

    def test_trajectory_decreasing(self):
        df = self._make_df([8.0, 7.0, 6.0, 5.0])
        assert compute_career_trajectory(df) < 0


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

class TestTeamFeatures:
    def test_style_compat_all_positions(self):
        for pos in ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]:
            s = get_style_position_compat("pace_and_space", pos)
            assert 0.0 <= s <= 1.0

    def test_style_compat_center_in_post_up(self):
        s = get_style_position_compat("post_up", "C")
        assert s >= 0.95

    def test_style_compat_pg_in_pace_and_space(self):
        s = get_style_position_compat("pace_and_space", "PG")
        assert s >= 0.90

    def test_style_vector_6d(self):
        from src.data.models import Team
        t = Team(
            id=1, name="NBA Team", league_id=1,
            playing_style="pace_and_space", formation="small_ball",
            pace=103.0, offensive_rating=115.0, defensive_rating=110.0,
            three_point_attempt_rate=0.42, assists_per_game=27.0,
            star_player_usage=0.30, league_tier=1,
        )
        v = compute_team_style_vector(t)
        assert v.shape == (6,)
        assert all(0.0 <= x <= 1.0 for x in v)

    def test_style_similarity_identical(self):
        v = np.array([0.8, 0.6, 0.4, 0.3, 0.7, 0.5])
        assert style_vector_similarity(v, v) == pytest.approx(1.0)

    def test_style_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        assert style_vector_similarity(v1, v2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

class TestContextFeatures:
    def test_league_adaptation_same_tier(self):
        f = compute_league_adaptation(1, 1)
        assert f == pytest.approx(1.0)

    def test_league_adaptation_step_up(self):
        f = compute_league_adaptation(3, 1)  # from tier 3 to tier 1 (tougher)
        assert f < 1.0

    def test_league_adaptation_step_down(self):
        f = compute_league_adaptation(1, 3)  # from tier 1 to tier 3 (easier)
        assert f >= 1.0

    def test_league_adaptation_same_country_bonus(self):
        f_no  = compute_league_adaptation(2, 1, same_country=False)
        f_yes = compute_league_adaptation(2, 1, same_country=True)
        assert f_yes > f_no

    def test_default_context_keys(self):
        d = _default_context_features()
        for k in ["position_team_fit", "style_compatibility", "role_opportunity",
                  "league_adaptation_factor", "spacing_fit"]:
            assert k in d

    def test_context_features_with_dict(self):
        leagues = pd.DataFrame([{"id": 1, "name": "NBA", "country": "USA", "tier": 1, "competitiveness_score": 1.0, "avg_pace": 100, "avg_offensive_rating": 113}])
        teams   = pd.DataFrame([{"id": 1, "name": "Team A", "league_id": 1, "playing_style": "pace_and_space", "formation": "small_ball", "pace": 103.0, "offensive_rating": 115.0, "defensive_rating": 110.0, "three_point_attempt_rate": 0.42, "assists_per_game": 27.0, "star_player_usage": 0.30, "league_tier": 1}])
        players = pd.DataFrame([{"id": 1, "name": "P1", "age": 24, "position": "PG", "nationality": "American", "height_cm": 190, "weight_kg": 85, "dominant_hand": "right", "current_team_id": 1, "current_league_id": 1, "draft_year": 2021, "draft_pick": 3}])
        stats   = pd.DataFrame([{"player_id": 1, "season": "2023-24", "team_id": 1, "league_id": 1, "games_played": 70, "minutes_per_game": 30, "points": 18, "rebounds": 5, "offensive_rebounds": 1, "defensive_rebounds": 4, "assists": 6, "steals": 1.2, "blocks": 0.3, "turnovers": 2, "personal_fouls": 2, "fg_pct": 0.47, "three_point_pct": 0.38, "ft_pct": 0.82, "plus_minus": 3, "per": 18, "ts_pct": 0.58, "usg_pct": 22, "bpm": 1.5, "vorp": 2, "win_shares": 5, "ast_ratio": 18, "reb_pct": 8, "rating": 7.5}])
        rels    = pd.DataFrame([{"team_id": 1, "player_id": 1, "season": "2023-24", "role": "starter", "jersey_number": 7}])

        data = {
            "players": players, "teams": teams, "leagues": leagues,
            "player_stats": stats, "team_player_relations": rels,
            "player_dict": {1: players.iloc[0].to_dict()},
            "team_dict":   {1: teams.iloc[0].to_dict()},
            "league_dict": {1: leagues.iloc[0].to_dict()},
            "league_teams": {1: [1]},
        }
        ctx = compute_context_features(1, 1, data)
        assert 0.0 <= ctx["position_team_fit"] <= 1.0
        assert 0.0 <= ctx["style_compatibility"] <= 1.0
        assert 0.0 <= ctx["role_opportunity"] <= 1.0


# ---------------------------------------------------------------------------
# Age curve
# ---------------------------------------------------------------------------

class TestAgeCurve:
    def test_peak_factor_is_one(self):
        for pos in PEAK_AGES:
            peak = PEAK_AGES[pos]
            assert age_performance_factor(peak, pos) == pytest.approx(1.0, abs=0.01)

    def test_young_player_below_peak(self):
        assert age_performance_factor(19, "PG") < 1.0

    def test_old_player_below_peak(self):
        assert age_performance_factor(38, "C") < 1.0

    def test_factor_clipped_at_040(self):
        assert age_performance_factor(50, "PG") >= 0.40

    def test_all_hybrid_positions(self):
        for pos in ["PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]:
            f = age_performance_factor(PEAK_AGES[pos], pos)
            assert f == pytest.approx(1.0, abs=0.05)

    def test_peak_window_returns_tuple(self):
        start, end = peak_age_window("PG", threshold=0.95)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= PEAK_AGES["PG"] <= end

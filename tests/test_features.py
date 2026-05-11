"""Tests for feature engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.player_features import (
    compute_form_score,
    compute_consistency_score,
    compute_per90,
    compute_technical_score,
    compute_defensive_score,
    compute_career_trajectory,
    compute_age_vs_peak,
    compute_player_features,
    _default_player_features,
    PEAK_AGES,
)
from src.features.team_features import compute_team_features, style_similarity
from src.features.context_features import (
    compute_position_fit,
    compute_league_adaptation,
    compute_context_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_history():
    """Three seasons of player stats."""
    return pd.DataFrame({
        "player_id": [1, 1, 1],
        "season": [2022, 2023, 2024],
        "rating": [7.1, 7.4, 7.6],
        "goals": [10, 14, 18],
        "assists": [5, 8, 10],
        "minutes": [2700.0, 2900.0, 3000.0],
        "pass_accuracy": [82.0, 84.0, 85.0],
        "dribbles": [2.5, 2.8, 3.0],
        "tackles": [1.5, 1.6, 1.7],
        "interceptions": [0.8, 0.9, 1.0],
        "aerial_duels_won": [1.2, 1.3, 1.4],
        "xG": [8.0, 12.0, 16.0],
        "xA": [4.5, 7.0, 9.5],
        "progressive_passes": [4.0, 5.0, 6.0],
        "key_passes": [1.5, 2.0, 2.5],
        "team_id": [1, 1, 2],
        "league_id": [1, 1, 1],
        "matches_played": [30, 32, 34],
    })


@pytest.fixture
def minimal_data(sample_history):
    """Minimal data dict for player feature tests."""
    players = pd.DataFrame([{
        "id": 1, "name": "Test Player", "age": 24, "position": "W",
        "nationality": "English", "foot": "right", "height": 178.0,
        "weight": 75.0, "current_team_id": 2, "current_league_id": 1,
    }])
    teams = pd.DataFrame([{
        "id": 2, "name": "Test FC", "league_id": 1, "playing_style": "possession",
        "formation": "4-3-3", "avg_possession": 58.0, "pressing_intensity": 7.5,
        "defensive_line": 7.0, "passing_tempo": 8.0, "league_tier": 1,
    }])
    leagues = pd.DataFrame([{
        "id": 1, "name": "Premier League", "country": "England",
        "tier": 1, "competitiveness_score": 0.98,
    }])
    rel = pd.DataFrame(columns=["team_id", "player_id", "season", "role", "jersey_number"])
    return {
        "players": players,
        "teams": teams,
        "leagues": leagues,
        "player_stats": sample_history,
        "team_player_relations": rel,
        "player_dict": {1: players.iloc[0].to_dict()},
        "team_dict": {2: teams.iloc[0].to_dict()},
        "league_dict": {1: leagues.iloc[0].to_dict()},
        "league_teams": {1: [2]},
    }


# ---------------------------------------------------------------------------
# Player feature tests
# ---------------------------------------------------------------------------

class TestFormScore:
    def test_recent_seasons_weighted_more(self, sample_history):
        form = compute_form_score(sample_history)
        # Highest weight on most-recent season (7.6), so form > simple mean
        simple_mean = sample_history["rating"].mean()
        assert form > simple_mean
        assert 7.0 < form < 8.0

    def test_single_season(self):
        df = pd.DataFrame({"rating": [7.2], "season": [2024]})
        assert compute_form_score(df) == pytest.approx(7.2, abs=1e-6)

    def test_empty_returns_default(self):
        assert compute_form_score(pd.DataFrame({"rating": []})) == 6.0


class TestConsistencyScore:
    def test_consistent_player_near_one(self):
        df = pd.DataFrame({"rating": [7.5, 7.5, 7.5, 7.5]})
        score = compute_consistency_score(df)
        assert score > 0.95

    def test_erratic_player_lower(self):
        consistent = pd.DataFrame({"rating": [7.4, 7.5, 7.6]})
        erratic = pd.DataFrame({"rating": [5.0, 9.0, 6.0, 8.5]})
        assert compute_consistency_score(consistent) > compute_consistency_score(erratic)


class TestPer90:
    def test_goals_per_90(self, sample_history):
        result = compute_per90(sample_history)
        total_min = sample_history["minutes"].sum()
        total_goals = sample_history["goals"].sum()
        expected = total_goals / (total_min / 90)
        assert result["goals_per_90"] == pytest.approx(expected, rel=1e-5)

    def test_zero_minutes(self):
        df = pd.DataFrame({"goals": [5], "assists": [3], "xG": [4], "xA": [2], "minutes": [0]})
        result = compute_per90(df)
        assert all(v == 0.0 for v in result.values())


class TestTechnicalScore:
    def test_high_stats_higher_score(self):
        high = pd.Series({"pass_accuracy": 92, "dribbles": 4.0, "key_passes": 3.0})
        low = pd.Series({"pass_accuracy": 65, "dribbles": 0.5, "key_passes": 0.2})
        assert compute_technical_score(high) > compute_technical_score(low)

    def test_bounded_0_1(self):
        row = pd.Series({"pass_accuracy": 99, "dribbles": 10.0, "key_passes": 10.0})
        assert 0.0 <= compute_technical_score(row) <= 1.0


class TestDefensiveScore:
    def test_high_stats_higher_score(self):
        high = pd.Series({"tackles": 5.0, "interceptions": 3.0, "aerial_duels_won": 6.0})
        low = pd.Series({"tackles": 0.2, "interceptions": 0.1, "aerial_duels_won": 0.1})
        assert compute_defensive_score(high) > compute_defensive_score(low)


class TestCareerTrajectory:
    def test_improving_player_positive_slope(self, sample_history):
        slope = compute_career_trajectory(sample_history)
        assert slope > 0

    def test_declining_player_negative_slope(self):
        df = pd.DataFrame({"rating": [8.0, 7.5, 7.0, 6.5], "season": [2021, 2022, 2023, 2024]})
        assert compute_career_trajectory(df) < 0


class TestAgeVsPeak:
    def test_at_peak_zero(self):
        assert compute_age_vs_peak(PEAK_AGES["ST"], "ST") == 0.0

    def test_younger_negative(self):
        assert compute_age_vs_peak(20, "CM") < 0

    def test_older_positive(self):
        assert compute_age_vs_peak(35, "GK") > 0


class TestComputePlayerFeatures:
    def test_returns_all_keys(self, minimal_data):
        feats = compute_player_features(1, minimal_data)
        expected_keys = {
            "form_score", "consistency_score", "goals_per_90", "assists_per_90",
            "xG_per_90", "xA_per_90", "technical_score", "defensive_score",
            "peak_rating", "career_trajectory", "age_vs_peak", "pass_accuracy",
            "dribbles", "tackles", "interceptions", "aerial_duels_won",
            "progressive_passes", "key_passes", "minutes", "matches_played",
            "age", "position_enc",
        }
        assert expected_keys.issubset(set(feats.keys()))

    def test_unknown_player_returns_defaults(self, minimal_data):
        feats = compute_player_features(9999, minimal_data)
        assert feats == _default_player_features()


# ---------------------------------------------------------------------------
# Team feature tests
# ---------------------------------------------------------------------------

class TestTeamFeatures:
    def test_tier1_team_factor_one(self, minimal_data):
        feats = compute_team_features(2, minimal_data)
        assert feats["league_tier_factor"] == 1.0
        assert feats["league_tier"] == 1.0

    def test_style_vector_normalised(self, minimal_data):
        feats = compute_team_features(2, minimal_data)
        for key in ["style_possession", "style_pressing", "style_def_line", "style_passing"]:
            assert 0.0 <= feats[key] <= 1.0

    def test_unknown_team_defaults(self, minimal_data):
        feats = compute_team_features(9999, minimal_data)
        assert feats["league_tier_factor"] == 1.0

    def test_style_similarity_same_team(self, minimal_data):
        t = compute_team_features(2, minimal_data)
        sim = style_similarity(t, t)
        assert sim == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Context feature tests
# ---------------------------------------------------------------------------

class TestContextFeatures:
    def test_position_fit_gk_always_high(self):
        fit = compute_position_fit("GK", "4-3-3", "possession")
        assert fit >= 0.8

    def test_unknown_formation_uses_default(self):
        fit = compute_position_fit("ST", "4-6-0", "direct")
        assert 0.0 <= fit <= 1.0

    def test_league_adaptation_same_tier(self):
        factor = compute_league_adaptation(1, 1)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_league_adaptation_step_up(self):
        factor_up = compute_league_adaptation(3, 1)  # tier3 → tier1
        factor_same = compute_league_adaptation(1, 1)
        assert factor_up < factor_same

    def test_league_adaptation_step_down(self):
        factor_down = compute_league_adaptation(1, 3)
        assert factor_down >= 1.0

    def test_compute_context_features_keys(self, minimal_data):
        ctx = compute_context_features(1, 2, minimal_data)
        assert "position_team_fit" in ctx
        assert "style_compatibility" in ctx
        assert "role_opportunity" in ctx
        assert "league_adaptation_factor" in ctx

    def test_compute_context_unknown_player(self, minimal_data):
        ctx = compute_context_features(9999, 2, minimal_data)
        assert all(isinstance(v, float) for v in ctx.values())

"""Tests for basketball feature engineering."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from basketball_ai.data.models import Player, PlayerStats
from basketball_ai.features.player_features import (
    compute_player_features,
    compute_player_features_from_objects,
    compute_form_score,
    compute_consistency_score,
    compute_career_trajectory,
    POSITIONAL_PEAK_AGES,
    _peak_age,
)
from basketball_ai.features.team_features import (
    compute_team_style_vector,
    get_style_position_compat,
    style_vector_similarity,
)
from basketball_ai.features.context_features import (
    compute_league_adaptation,
    compute_context_features,
    _default_context_features,
)
from basketball_ai.models.age_curve import (
    age_performance_factor,
    PEAK_AGES,
    peak_age_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_player(pos: str = "PG", age: int = 25) -> Player:
    return Player(
        id=1,
        name="Test Player",
        age=age,
        position=pos,
        nationality="American",
        height_cm=193,
        weight_kg=90,
        dominant_hand="right",
        current_team_id=1,
        current_league_id=1,
        draft_year=2020,
        draft_pick=5,
    )


def _make_stat(season: str, rating: float, **kwargs) -> PlayerStats:
    defaults = dict(
        player_id=1,
        team_id=1,
        league_id=1,
        games_played=70,
        minutes_per_game=32.0,
        points=18.0,
        rebounds=5.0,
        offensive_rebounds=1.5,
        defensive_rebounds=3.5,
        assists=6.0,
        steals=1.2,
        blocks=0.4,
        turnovers=2.5,
        personal_fouls=2.0,
        fg_pct=0.47,
        three_point_pct=0.36,
        ft_pct=0.82,
        plus_minus=3.0,
        per=18.0,
        ts_pct=0.58,
        usg_pct=22.0,
        bpm=1.5,
        vorp=2.0,
        win_shares=5.0,
        ast_ratio=18.0,
        reb_pct=8.0,
    )
    defaults.update(kwargs)
    return PlayerStats(season=season, rating=rating, **defaults)


# ---------------------------------------------------------------------------
# Player features
# ---------------------------------------------------------------------------


class TestPlayerFeatures:
    def test_basic_output_keys(self):
        player = _make_player("PG", 25)
        stats = [_make_stat("2023-24", 7.5)]
        feats = compute_player_features_from_objects(player, stats)
        required = [
            "form_score",
            "consistency_score",
            "pts_per_36",
            "ast_per_36",
            "reb_per_36",
            "avg_per",
            "avg_ts_pct",
            "avg_usg_pct",
            "avg_bpm",
            "peak_rating",
            "career_trajectory",
            "age_vs_peak_age",
            "positional_peak_age",
            "scoring_profile",
            "playmaking_score",
            "defensive_score",
            "versatility_score",
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
        stats = [_make_stat(f"202{i}-2{i + 1}", 7.0 + i * 0.1) for i in range(4)]
        feats = compute_player_features_from_objects(player, stats)
        assert 0.0 <= feats["consistency_score"] <= 1.0

    def test_career_trajectory_positive(self):
        player = _make_player()
        stats = [_make_stat(f"202{i}-2{i + 1}", 6.0 + i) for i in range(4)]
        feats = compute_player_features_from_objects(player, stats)
        assert feats["career_trajectory"] > 0

    def test_empty_stats_returns_defaults(self):
        player = _make_player()
        feats = compute_player_features_from_objects(player, [])
        assert feats["form_score"] == 5.0
        assert feats["consistency_score"] == 0.5

    def test_dict_api_ignores_extra_source_columns(self):
        """Source-only PostgreSQL columns must not discard valid PlayerStats rows."""
        player = _make_player("PG", 25)
        stat = _make_stat("2023", 7.5, per=18.0, usg_pct=22.0)
        row = dict(stat.__dict__)
        row.update({
            "global_id": "player-global-1",
            "player_global_id": "player-global-1",
            "league_key": "ITA1",
            "unmapped_source_column": "ignored",
        })
        players = pd.DataFrame([player.__dict__])
        data = {
            "player_dict": {1: player.__dict__},
            "player_stats": pd.DataFrame([row]),
            "leagues": pd.DataFrame([{
                "id": 1,
                "name": "Test League",
                "max_games": 30,
            }]),
        }
        feats = compute_player_features(1, data, season=2024)
        assert feats["form_score"] == pytest.approx(7.5)
        assert feats["avg_per"] == pytest.approx(18.0)
        assert feats["avg_usg_pct"] == pytest.approx(22.0)

    def test_hybrid_position_peak_age(self):
        from basketball_ai.models.age_curve import reset_fitted_params

        reset_fitted_params()
        try:
            player = _make_player("PF/C", 27)
            stats = [_make_stat("2023-24", 7.0)]
            feats = compute_player_features_from_objects(player, stats)
            assert feats["positional_peak_age"] == POSITIONAL_PEAK_AGES["PF/C"]
            assert feats["age_vs_peak_age"] == 27 - POSITIONAL_PEAK_AGES["PF/C"]
        finally:
            reset_fitted_params()

    def test_peak_age_uses_fitted_when_available(self):
        from basketball_ai.models.age_curve import (
            _fitted_peak_ages,
            reset_fitted_params,
        )

        reset_fitted_params()
        try:
            _fitted_peak_ages["PF/C"] = 29.2
            player = _make_player("PF/C", 27)
            stats = [_make_stat("2023-24", 7.0)]
            feats = compute_player_features_from_objects(player, stats)
            assert feats["positional_peak_age"] == int(round(29.2))
            assert feats["age_vs_peak_age"] == -2
        finally:
            reset_fitted_params()

    def test_all_positions_have_peak_age(self):
        for pos in [
            "PG",
            "SG",
            "SF",
            "PF",
            "C",
            "PG/SG",
            "SG/SF",
            "SF/PF",
            "PF/C",
            "SG/PF",
        ]:
            assert _peak_age(pos) >= 24

    def test_durability_league_aware_european(self):
        """European player (30-game league) should have higher durability than NBA calc."""
        player = _make_player("C", 27)
        # 25 games played in a 30-game league → 83% durability
        stat = _make_stat("2023-24", 7.0, games_played=25, league_id=99)
        feats_euro = compute_player_features_from_objects(
            player, [stat], league_max_games={99: 30}
        )
        feats_nba = compute_player_features_from_objects(
            player, [stat]
        )  # defaults to 82
        assert feats_euro["durability_score"] > feats_nba["durability_score"]
        assert feats_euro["durability_score"] == pytest.approx(25 / 30, abs=0.01)

    def test_durability_missing_league_defaults_to_82(self):
        """Unknown league_id should fall back to 82 (NBA default)."""
        from basketball_ai.constants import LEAGUE_MAX_GAMES_DEFAULT

        player = _make_player()
        stat = _make_stat("2023-24", 7.0, games_played=50, league_id=9999)
        feats = compute_player_features_from_objects(player, [stat])
        assert feats["durability_score"] == pytest.approx(
            50 / LEAGUE_MAX_GAMES_DEFAULT, abs=0.01
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


class TestFeatureHelpers:
    def _make_df(self, ratings):
        return pd.DataFrame(
            {
                "season": [f"202{i}-2{i + 1}" for i in range(len(ratings))],
                "rating": ratings,
            }
        )

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
        for pos in [
            "PG",
            "SG",
            "SF",
            "PF",
            "C",
            "PG/SG",
            "SG/SF",
            "SF/PF",
            "PF/C",
            "SG/PF",
        ]:
            s = get_style_position_compat("pace_and_space", pos)
            assert 0.0 <= s <= 1.0

    def test_style_compat_center_in_post_up(self):
        s = get_style_position_compat("post_up", "C")
        assert s >= 0.95

    def test_style_compat_pg_in_pace_and_space(self):
        s = get_style_position_compat("pace_and_space", "PG")
        assert s >= 0.90

    def test_style_vector_6d(self):
        from basketball_ai.data.models import Team

        t = Team(
            id=1,
            name="NBA Team",
            league_id=1,
            playing_style="pace_and_space",
            formation="small_ball",
            pace=103.0,
            offensive_rating=115.0,
            defensive_rating=110.0,
            three_point_attempt_rate=0.42,
            assists_per_game=27.0,
            star_player_usage=0.30,
            league_tier=1,
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
        f_no = compute_league_adaptation(2, 1, same_country=False)
        f_yes = compute_league_adaptation(2, 1, same_country=True)
        assert f_yes > f_no

    def test_default_context_keys(self):
        d = _default_context_features()
        for k in [
            "position_team_fit",
            "style_compatibility",
            "role_opportunity",
            "league_adaptation_factor",
            "spacing_fit",
        ]:
            assert k in d

    def test_context_features_with_dict(self):
        leagues = pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "NBA",
                    "country": "USA",
                    "tier": 1,
                    "competitiveness_score": 1.0,
                    "avg_pace": 100,
                    "avg_offensive_rating": 113,
                }
            ]
        )
        teams = pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "Team A",
                    "league_id": 1,
                    "playing_style": "pace_and_space",
                    "formation": "small_ball",
                    "pace": 103.0,
                    "offensive_rating": 115.0,
                    "defensive_rating": 110.0,
                    "three_point_attempt_rate": 0.42,
                    "assists_per_game": 27.0,
                    "star_player_usage": 0.30,
                    "league_tier": 1,
                }
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "P1",
                    "age": 24,
                    "position": "PG",
                    "nationality": "American",
                    "height_cm": 190,
                    "weight_kg": 85,
                    "dominant_hand": "right",
                    "current_team_id": 1,
                    "current_league_id": 1,
                    "draft_year": 2021,
                    "draft_pick": 3,
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": "2023-24",
                    "team_id": 1,
                    "league_id": 1,
                    "games_played": 70,
                    "minutes_per_game": 30,
                    "points": 18,
                    "rebounds": 5,
                    "offensive_rebounds": 1,
                    "defensive_rebounds": 4,
                    "assists": 6,
                    "steals": 1.2,
                    "blocks": 0.3,
                    "turnovers": 2,
                    "personal_fouls": 2,
                    "fg_pct": 0.47,
                    "three_point_pct": 0.38,
                    "ft_pct": 0.82,
                    "plus_minus": 3,
                    "per": 18,
                    "ts_pct": 0.58,
                    "usg_pct": 22,
                    "bpm": 1.5,
                    "vorp": 2,
                    "win_shares": 5,
                    "ast_ratio": 18,
                    "reb_pct": 8,
                    "rating": 7.5,
                }
            ]
        )
        rels = pd.DataFrame(
            [
                {
                    "team_id": 1,
                    "player_id": 1,
                    "season": "2023-24",
                    "role": "starter",
                    "jersey_number": 7,
                }
            ]
        )

        data = {
            "players": players,
            "teams": teams,
            "leagues": leagues,
            "player_stats": stats,
            "team_player_relations": rels,
            "player_dict": {1: players.iloc[0].to_dict()},
            "team_dict": {1: teams.iloc[0].to_dict()},
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


# ---------------------------------------------------------------------------
# New DB-schema advanced feature keys
# ---------------------------------------------------------------------------


class TestAdvancedDBSchemaFeatures:
    """Verify that compute_player_features_from_objects returns all new
    DB-schema derived features (SPM, RAPTOR, LEBRON, clutch, on/off, per-40,
    hustle, scoring efficiency)."""

    def _make_player(self, pos="PG", age=25):
        return Player(
            id=1,
            name="Advanced Player",
            age=age,
            position=pos,
            nationality="American",
            height_cm=195,
            weight_kg=92,
            dominant_hand="right",
            current_team_id=1,
            current_league_id=1,
            draft_year=2018,
            draft_pick=4,
        )

    def _make_stat(self, season: str, rating: float = 7.5, **kw) -> PlayerStats:
        defaults = dict(
            player_id=1,
            team_id=1,
            league_id=1,
            games_played=70,
            minutes_per_game=32.0,
            points=18.0,
            rebounds=5.0,
            offensive_rebounds=1.5,
            defensive_rebounds=3.5,
            assists=6.0,
            steals=1.2,
            blocks=0.4,
            turnovers=2.5,
            personal_fouls=2.0,
            fg_pct=0.47,
            three_point_pct=0.36,
            ft_pct=0.82,
            plus_minus=3.0,
            per=18.0,
            ts_pct=0.58,
            usg_pct=22.0,
            bpm=1.5,
            vorp=2.0,
            win_shares=5.0,
            ast_ratio=18.0,
            reb_pct=8.0,
        )
        defaults.update(kw)
        return PlayerStats(season=season, rating=rating, **defaults)

    def test_new_feature_keys_present(self):
        """All new DB-schema feature keys must appear in the output dict."""
        player = self._make_player()
        stat = self._make_stat(
            "2023-24",
            7.5,
            spm=1.2,
            obpm=0.8,
            dbpm=0.4,
            gm_sc=12.5,
            fic=18.0,
            ows=3.0,
            dws=2.0,
            raptor_off=2.1,
            raptor_def=0.5,
            raptor_total=2.6,
            lebron_off=1.8,
            lebron_def=0.3,
            lebron_total=2.1,
            scoring_efficiency=1.15,
            hustle_index=55.0,
            foul_drawing_rate=0.28,
            net_rtg_diff=4.2,
            ortg_diff=3.1,
            clutch_games=10,
            clutch_pts=14.0,
            clutch_ts_pct=0.60,
            clutch_ast_to_tov=2.0,
            clutch_net_rtg=3.5,
            clutch_efg_pct=0.55,
            pts_per_40=22.5,
            ast_per_40=7.5,
        )
        feats = compute_player_features_from_objects(player, [stat])
        new_keys = [
            "avg_spm",
            "avg_raptor_total",
            "avg_raptor_off",
            "avg_raptor_def",
            "avg_lebron_total",
            "avg_obpm",
            "avg_dbpm",
            "avg_gm_sc",
            "avg_fic",
            "avg_scoring_efficiency",
            "avg_hustle_index",
            "avg_foul_drawing_rate",
            "avg_net_rtg_diff",
            "avg_ortg_diff",
            "clutch_pts_per_36",
            "avg_clutch_ts_pct",
            "avg_clutch_net_rtg",
            "avg_clutch_efg_pct",
            "clutch_games_career",
            "avg_pts_per_40",
            "avg_ast_per_40",
        ]
        for k in new_keys:
            assert k in feats, f"Missing new feature key: {k}"

    def test_spm_raptor_lebron_values(self):
        """SPM, RAPTOR, LEBRON values should match the input stat."""
        player = self._make_player()
        stat = self._make_stat(
            "2023-24", 7.5, spm=3.0, raptor_total=4.0, lebron_total=2.5
        )
        feats = compute_player_features_from_objects(player, [stat])
        assert feats["avg_spm"] == pytest.approx(3.0, abs=0.01)
        assert feats["avg_raptor_total"] == pytest.approx(4.0, abs=0.01)
        assert feats["avg_lebron_total"] == pytest.approx(2.5, abs=0.01)

    def test_clutch_metrics_computed(self):
        """Clutch metrics aggregate correctly over multiple seasons."""
        player = self._make_player()
        s1 = self._make_stat(
            "2022-23", 7.0, clutch_games=8, clutch_ts_pct=0.58, clutch_net_rtg=2.0
        )
        s2 = self._make_stat(
            "2023-24", 7.5, clutch_games=12, clutch_ts_pct=0.62, clutch_net_rtg=4.0
        )
        feats = compute_player_features_from_objects(player, [s1, s2])
        assert feats["clutch_games_career"] == 20
        assert feats["avg_clutch_ts_pct"] == pytest.approx(0.60, abs=0.01)
        assert feats["avg_clutch_net_rtg"] == pytest.approx(3.0, abs=0.01)

    def test_on_off_differential(self):
        """On/off net rating differential should be averaged."""
        player = self._make_player()
        s1 = self._make_stat("2022-23", 7.0, net_rtg_diff=3.0, ortg_diff=2.5)
        s2 = self._make_stat("2023-24", 7.5, net_rtg_diff=5.0, ortg_diff=4.5)
        feats = compute_player_features_from_objects(player, [s1, s2])
        assert feats["avg_net_rtg_diff"] == pytest.approx(4.0, abs=0.01)
        assert feats["avg_ortg_diff"] == pytest.approx(3.5, abs=0.01)

    def test_empty_stats_new_keys_present(self):
        """New keys must be present even with no stats history."""
        player = self._make_player()
        feats = compute_player_features_from_objects(player, [])
        for k in [
            "avg_spm",
            "avg_raptor_total",
            "avg_clutch_ts_pct",
            "clutch_games_career",
            "avg_pts_per_40",
        ]:
            assert k in feats, f"Empty-stats missing: {k}"


class TestMetricCatalog:
    """Verify the METRIC_CATALOG includes all new DB-schema metric groups."""

    def test_clutch_metrics_in_catalog(self):
        from basketball_ai.models.performance_model import METRIC_CATALOG

        for m in [
            "clutch_pts",
            "clutch_ts_pct",
            "clutch_net_rtg",
            "clutch_efg_pct",
            "clutch_ast_to_tov",
        ]:
            assert m in METRIC_CATALOG, f"Missing from METRIC_CATALOG: {m}"

    def test_raptor_lebron_spm_in_catalog(self):
        from basketball_ai.models.performance_model import METRIC_CATALOG

        for m in [
            "spm",
            "raptor_total",
            "raptor_off",
            "raptor_def",
            "lebron_total",
            "lebron_off",
            "lebron_def",
            "obpm",
            "dbpm",
            "gm_sc",
            "fic",
        ]:
            assert m in METRIC_CATALOG, f"Missing from METRIC_CATALOG: {m}"

    def test_per40_in_catalog(self):
        from basketball_ai.models.performance_model import METRIC_CATALOG

        for m in ["pts_per_40", "ast_per_40", "tr_per_40", "stl_per_40", "blk_per_40"]:
            assert m in METRIC_CATALOG, f"Missing from METRIC_CATALOG: {m}"

    def test_hustle_efficiency_in_catalog(self):
        from basketball_ai.models.performance_model import METRIC_CATALOG

        for m in [
            "scoring_efficiency",
            "hustle_index",
            "foul_drawing_rate",
            "ppsa",
            "true_usg_pct",
        ]:
            assert m in METRIC_CATALOG, f"Missing from METRIC_CATALOG: {m}"

    def test_new_feature_cols_length(self):
        from basketball_ai.models.performance_model import FEATURE_COLS

        # Should have at least 30 features now (was 18 before)
        assert len(FEATURE_COLS) >= 30

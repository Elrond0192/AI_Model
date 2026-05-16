"""Property-based tests using Hypothesis.

Tests invariants that must hold for any valid input, not just specific examples.
"""
from __future__ import annotations
import pytest
import numpy as np

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed"
)


if _HYPOTHESIS_AVAILABLE:
    class TestPlayerFeaturesProperties:
        """Invariants for player feature engineering."""

        @given(
            age=st.integers(min_value=15, max_value=45),
            rating=st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            games_played=st.integers(min_value=1, max_value=100),
        )
        @settings(max_examples=50, deadline=5000)
        def test_feature_values_in_valid_range(self, age, rating, games_played):
            """Any valid player stats should produce features in expected ranges."""
            from basketball_ai.data.models import Player, PlayerStats
            from basketball_ai.features.player_features import compute_player_features_from_objects

            player = Player(
                id=1, name="Test", age=age, position="PG",
                nationality="American", height_cm=193, weight_kg=90,
                dominant_hand="right", current_team_id=1, current_league_id=1,
                draft_year=2000, draft_pick=10,
            )
            stat = PlayerStats(
                player_id=1, team_id=1, league_id=1,
                season="2023-24", rating=rating, games_played=games_played,
                minutes_per_game=25.0, points=15.0, rebounds=5.0,
                offensive_rebounds=1.5, defensive_rebounds=3.5,
                assists=4.0, steals=1.0, blocks=0.5,
                turnovers=2.0, personal_fouls=2.0,
                fg_pct=0.45, three_point_pct=0.35, ft_pct=0.80,
                plus_minus=2.0, per=16.0, ts_pct=0.55,
                usg_pct=20.0, bpm=0.5, vorp=1.5, win_shares=4.0,
                ast_ratio=15.0, reb_pct=8.0,
            )
            feats = compute_player_features_from_objects(player, [stat])

            # Invariants
            assert 0.0 <= feats["consistency_score"] <= 1.0, "consistency out of [0,1]"
            assert 0.0 <= feats["durability_score"] <= 1.0, "durability out of [0,1]"
            assert 0.0 <= feats["scoring_profile"] <= 1.0, "scoring_profile out of [0,1]"
            assert feats["positional_peak_age"] >= 20, "peak age too low"
            assert feats["pts_per_36"] >= 0, "negative per-36 points"

    class TestAgeCurveProperties:
        """Invariants for the age-performance curve."""

        @given(age=st.integers(min_value=18, max_value=45))
        @settings(max_examples=40, deadline=2000)
        def test_age_factor_in_range(self, age):
            """Age performance factor must be in (0, 1.2]."""
            from basketball_ai.models.age_curve import age_performance_factor
            for pos in ["PG", "SG", "SF", "PF", "C"]:
                f = age_performance_factor(age, pos)
                assert 0 < f <= 1.2, f"age_factor({age}, {pos}) = {f} out of range"

        @given(age=st.integers(min_value=18, max_value=30))
        @settings(max_examples=30, deadline=2000)
        def test_young_player_grows(self, age):
            """A 18-25 year old should have lower factor than their peak."""
            from basketball_ai.models.age_curve import age_performance_factor, PEAK_AGES
            assume(age < min(PEAK_AGES.values()))
            pos = "PG"
            peak = PEAK_AGES.get(pos, 27)
            f_young = age_performance_factor(age, pos)
            f_peak = age_performance_factor(peak, pos)
            assert f_young <= f_peak, f"young player ({age}) factor > peak ({peak}) factor"

    class TestDataSignatureProperties:
        """Invariants for data signature (MD5 hash)."""

        @given(n_rows=st.integers(min_value=1, max_value=100))
        @settings(max_examples=20, deadline=5000)
        def test_same_data_same_signature(self, n_rows):
            """Same data must always produce the same signature."""
            import pandas as pd
            import numpy as np
            from basketball_ai.models.performance_model import _compute_data_signature
            rng = np.random.default_rng(42)
            X = pd.DataFrame(rng.random((n_rows, 5)), columns=list("ABCDE"))
            y = rng.random(n_rows)
            sig1 = _compute_data_signature(X, y)
            sig2 = _compute_data_signature(X, y)
            assert sig1 == sig2

        @given(n_rows=st.integers(min_value=2, max_value=50))
        @settings(max_examples=20, deadline=5000)
        def test_different_data_different_signature(self, n_rows):
            """Distinct data should (almost certainly) produce different signatures."""
            import pandas as pd
            import numpy as np
            from basketball_ai.models.performance_model import _compute_data_signature
            rng = np.random.default_rng(99)
            X1 = pd.DataFrame(rng.random((n_rows, 5)), columns=list("ABCDE"))
            X2 = pd.DataFrame(rng.random((n_rows, 5)), columns=list("ABCDE"))
            y = rng.random(n_rows)
            assert _compute_data_signature(X1, y) != _compute_data_signature(X2, y)

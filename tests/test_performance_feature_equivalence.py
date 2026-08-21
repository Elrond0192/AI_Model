"""Equivalence tests for cumulative player-history feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from basketball_ai.models.performance_model import PerformanceModel


def test_cumulative_history_matches_legacy_row_by_row():
    """The O(k) cache must reproduce the former history slice for every row."""
    grp = pd.DataFrame(
        {
            "player_id": [7] * 6,
            "season": ["2019-20", "2020-21", "2020-21", "2021-22", "2022-23", "2023-24"],
            "competition": ["RS", "RS", "PO", "RS", "PO", "RS"],
            "league_id": [1, 1, 1, 2, 2, 2],
            "minutes_per_game": [18.0, 22.0, 11.0, 26.0, 14.0, 30.0],
            "rating": [5.8, 6.1, 6.5, 7.0, 6.8, 7.4],
            "games_played": [30.0, 42.0, 8.0, np.nan, 10.0, 58.0],
            "points": [7.0, 9.0, 8.0, 12.0, 10.0, 15.0],
            "assists": [2.0, 2.5, 2.0, 3.5, 3.0, 4.0],
            "rebounds": [3.0, 3.5, 4.0, 5.0, 4.5, 6.0],
            "steals": [0.5, 0.7, 0.6, 0.8, 0.7, 1.0],
            "blocks": [0.2, 0.3, 0.4, 0.5, 0.4, 0.6],
            "per": [11.0, 12.5, np.nan, 14.0, 13.5, 16.0],
            "ts_pct": [0.50, np.nan, 0.54, 0.57, 0.55, 0.60],
            "usg_pct": [16.0, 18.0, 21.0, 22.0, 23.0, 25.0],
            "obpm": [-1.0, -0.5, 0.2, 1.0, 0.8, 1.8],
            "dbpm": [0.3, 0.5, 0.7, 1.0, 0.9, 1.2],
            "reb_pct": [5.0, 5.5, 7.0, 8.0, 8.5, 9.0],
            "raptor_off": [-0.8, -0.2, 0.1, 0.9, 0.7, 1.6],
            "raptor_def": [0.1, 0.3, 0.5, 0.8, 0.7, 1.0],
            "vorp": [np.nan, 0.4, np.nan, 1.2, 1.0, 2.0],
        }
    ).sort_values("season")
    model = PerformanceModel()
    league_max_games = {1: 50, 2: 60}
    extra_metrics = ["vorp"]
    cumulative = model._precompute_history_features(
        grp,
        extra_metrics=extra_metrics,
        league_max_games=league_max_games,
    )

    positions = pd.Series(np.arange(len(grp), dtype=int))
    season_keys = pd.Series(grp["season"].to_numpy())
    history_ends = positions.groupby(season_keys, dropna=False).transform("max").to_numpy()

    for source_position in range(len(grp)):
        source = grp.iloc[source_position]
        legacy_history = grp[grp["season"] <= source["season"]]
        legacy = model._build_row(
            source,
            age=24 + source_position,
            position="PG/SG",
            player_stats_history=legacy_history,
            extra_metrics=extra_metrics,
            league_max_games=league_max_games,
        )
        optimized = model._build_row(
            source,
            age=24 + source_position,
            position="PG/SG",
            player_stats_history=grp.iloc[:0],
            extra_metrics=extra_metrics,
            league_max_games=league_max_games,
            precomputed_history=cumulative[int(history_ends[source_position])],
        )

        assert legacy.keys() == optimized.keys()
        np.testing.assert_allclose(
            list(optimized.values()),
            list(legacy.values()),
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )

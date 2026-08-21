"""Regression tests for low-risk DataFrame vectorization paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from basketball_ai.models.compatibility_model import (
    CompatibilityModel,
    _style_vector_from_prior,
    _with_prior_player_means,
)
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine


def test_prior_player_means_match_strict_legacy_filters():
    stats = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 11, 20, 20],
            "season": [2019, 2020, 2020, 2022, 2020, 2021],
            "rating": [5.5, 6.0, 6.4, 7.0, 6.2, 6.6],
            "usg_pct": [16.0, 18.0, np.nan, 24.0, 20.0, 22.0],
            "ts_pct": [0.50, 0.54, 0.56, 0.60, 0.52, 0.55],
            "points": [8.0, 10.0, 11.0, 15.0, 12.0, 14.0],
            "three_par": [0.25, 0.30, 0.35, 0.42, 0.32, 0.36],
            "dbpm": [-0.5, 0.0, 0.4, 1.0, 0.3, 0.7],
        }
    )
    player_dict = {
        1: {"position": "PG/SG"},
        2: {"position": "PF"},
    }
    data = {"player_stats": stats, "player_dict": player_dict}
    cached = _with_prior_player_means(stats, "season")
    legacy_model = CompatibilityModel()

    for row in cached.to_dict("records"):
        pid = int(row["player_id"])
        season = int(row["season"])
        legacy_prior = stats[
            (stats["player_id"] == pid) & (stats["season"] < season)
        ]
        assert int(row["_prior_rows"]) == len(legacy_prior)
        if legacy_prior.empty:
            continue
        np.testing.assert_allclose(
            _style_vector_from_prior(row, player_dict),
            legacy_model._player_style_vector(
                pid, data, before_season=season
            ),
            rtol=0.0,
            atol=1e-15,
        )
        assert np.isclose(
            float(row["_prior_mean_rating"]),
            float(legacy_prior["rating"].mean()),
            rtol=0.0,
            atol=1e-15,
        )


def test_transfer_self_merge_matches_legacy_nested_loop():
    stats = pd.DataFrame(
        {
            "player_id": [2, 1, 1, 1, 1, 2, 2, 2],
            "league_id": [10, 10, 20, 20, 10, 10, 20, 20],
            "season": [2020, 2019, 2020, 2020, 2021, 2021, 2021, 2022],
            "competition": ["RS"] * 8,
            "rating": [6.0, 5.5, 6.0, 6.2, np.nan, 6.4, 6.8, 7.0],
        }
    )
    engine = object.__new__(ChatScenarioEngine)
    engine.data = {"player_stats": stats}

    expected: list[float] = []
    scoped = stats[stats["season"] <= 2022].copy()
    for _, group in scoped.groupby("player_id"):
        sources = group[group["league_id"] == 10]
        targets = group[group["league_id"] == 20]
        target_by_year = {
            int(row["season"]): row
            for row in targets.to_dict("records")
        }
        for source in sources.to_dict("records"):
            target = target_by_year.get(int(source["season"]) + 1)
            if target is None or pd.isna(source["rating"]) or pd.isna(target["rating"]):
                continue
            expected.append(float(target["rating"]) - float(source["rating"]))

    assert engine._transfer_samples(10, 20, "RS", "RS", 2021) == expected

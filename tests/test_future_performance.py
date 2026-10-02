from types import SimpleNamespace

import pandas as pd
import pytest

from basketball_ai.future_performance.model import (
    TARGET_SPECS,
    PlayerFuturePerformanceModel,
)


def _sample_data(years=(2018, 2019, 2020, 2021), n_players=4):
    players = pd.DataFrame(
        [
            {"id": index, "global_id": f"P-{index}", "name": f"Player {index}",
             "position": ("PG", "SG", "SF", "C")[index % 4], "birth_date": "1995-01-01"}
            for index in range(1, n_players + 1)
        ]
    )
    leagues = pd.DataFrame([{"id": 1, "league_key": "ITA1", "name": "ITA1"}])
    rows = []
    for pid in range(1, n_players + 1):
        for year in years:
            base = float(10 + pid + (year - years[0]) * 0.25)
            rows.append(
                {
                    "player_id": pid, "league_id": 1, "league_key": "ITA1",
                    "season": f"{year}-{str(year + 1)[-2:]}",
                    "competition": "RS",
                    "minutes_per_game": 24.0 + pid,
                    "games_played": 30 + pid,
                    "starter_pct": 0.5,
                    "rating": 6.0 + pid * 0.1 + (year - years[0]) * 0.05,
                    "points": base,
                    "assists": 3.0 + pid * 0.1,
                    "rebounds": 5.0 + pid * 0.1,
                    "steals": 1.0,
                    "blocks": 0.5,
                    "ts_pct": 0.55 + pid * 0.005,
                    "usg_pct": 0.20 + pid * 0.01,
                    "tov_pct": 0.10 + pid * 0.002,
                    "ast_pct": 0.12 + pid * 0.003,
                    "orb_pct": 0.05,
                    "drb_pct": 0.15,
                    "three_point_pct": 0.35,
                    "ft_pct": 0.78,
                    "three_point_attempts": 4.0,
                    "free_throw_attempts": 3.0,
                }
            )
    return {"player_stats": pd.DataFrame(rows), "players": players, "leagues": leagues}


def test_build_pairs_only_uses_exact_consecutive_contexts():
    data = _sample_data(years=(2018, 2019, 2021))
    model = PlayerFuturePerformanceModel()
    pairs = model._build_pairs(data)

    assert set(pairs["__meta__target_year"].astype(int)) == {2019}
    assert set(pairs["__meta__source_year"].astype(int)) == {2018}


def test_missing_targets_are_not_converted_to_zero():
    data = _sample_data()
    data["player_stats"].loc[0, "ft_pct"] = float("nan")
    model = PlayerFuturePerformanceModel()
    pairs = model._build_pairs(data)

    assert pd.isna(pairs.loc[0, "__target__ft_pct"])


class _FakePredictor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, values):
        return [self.value] * len(values)


def test_predict_player_returns_targets_and_derived_per_game():
    data = _sample_data(years=(2021, 2022))
    model = PlayerFuturePerformanceModel()
    model.is_trained = True
    model.feature_names = model._feature_columns(["ITA1"])
    model.metadata = {"leagues": ["ITA1"]}
    model.uncertainty = {
        spec.key: {"p50": 0.1, "p75": 0.2, "p90": 0.3}
        for spec in TARGET_SPECS
    }
    model.models = {
        spec.key: _FakePredictor(
            25.0 if spec.kind == "per36"
            else 0.6 if spec.kind == "rate"
            else 30.0
        )
        for spec in TARGET_SPECS
    }

    result = model.predict_player(
        data,
        player_id=1,
        league_key="ITA1",
        season=2021,
        competition="RS",
    )

    assert result["target_season"] == 2022
    assert result["targets"]["minutes_per_game"]["value"] == pytest.approx(30.0)
    assert result["targets"]["pts_per_36"]["derived_per_game"] == pytest.approx(20.833333, rel=1e-5)
    assert result["targets"]["ts_pct"]["range_p90"]["lower"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_future_performance_api_resolves_global_player_and_canonical_league():
    from basketball_ai.api.routes.future_performance_v2 import future_player

    data = _sample_data(years=(2021, 2022))
    data["player_dict"] = {
        1: {"id": 1, "global_id": "P-1"},
    }

    fake_model = SimpleNamespace()

    def predict_player(**kwargs):
        assert kwargs["league_key"] == "ITA1"
        return {
            "future_performance_version": "1.0",
            "feature_version": "player-future-performance-v1",
            "player_id": 1,
            "player_global_id": "P-1",
            "player_name": "Player 1",
            "league": "ITA1",
            "source_season": 2021,
            "target_season": 2022,
            "competition": "RS",
            "targets": {},
            "model_status": "production",
        }

    fake_model.predict_player = predict_player

    app = SimpleNamespace(state=SimpleNamespace(
        future_performance_model=fake_model,
        data=data,
    ))
    request = SimpleNamespace(app=app)

    body = SimpleNamespace(player_global_id="P-1", league="ITA1", season=2021, competition="RS")
    result = await future_player(body, request)

    assert result.league == "ITA1"
    assert result.target_season == 2022

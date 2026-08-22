from __future__ import annotations

import numpy as np
import pandas as pd


def test_numeric_seasons_vectorized_mixed_values():
    from basketball_ai.models.production_training import _numeric_seasons

    frame = pd.DataFrame({"season": [2022, "2023-24", "2024", None]})
    values = _numeric_seasons(frame)
    assert values.iloc[0] == 2022
    assert values.iloc[1] == 2023
    assert values.iloc[2] == 2024
    assert pd.isna(values.iloc[3])


def test_compatibility_prefix_fallback_handles_missing_optional_metric():
    from basketball_ai.models.competition_training import (
        CompetitionTemporalCompatibilityModel,
    )

    prefix = CompetitionTemporalCompatibilityModel._prefix_mean(
        pd.Series(dtype=float),
        0.30,
    )
    assert CompetitionTemporalCompatibilityModel._mean_before(prefix, 3) == 0.30


def test_position_cache_is_snapshot_specific_and_never_leaks_future_role():
    from basketball_ai.models.strict_production import position_as_of

    full = pd.DataFrame(
        [
            {"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"},
            {"player_id": 7, "team_id": 10, "season": 2024, "role": "SF"},
        ]
    )
    # Warm the full-data cache first.
    assert position_as_of(7, full, 2024, "PG") == "SF"

    historical = full[full["season"] <= 2022].copy()
    # A different DataFrame object must build its own bounded cache.
    assert position_as_of(7, historical, 2024, "PG") == "SG"


def _scope_data() -> dict:
    return {
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2023,
                    "competition": "PO",
                    "games_played": 5,
                    "rating": 6.8,
                },
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2024,
                    "competition": "PO",
                    "games_played": 8,
                    "rating": 7.2,
                },
                {
                    "player_id": 2,
                    "team_id": 11,
                    "league_id": 1,
                    "season": 2024,
                    "competition": "PO",
                    "games_played": 7,
                    "rating": 6.9,
                },
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2024,
                    "competition": "RS",
                    "games_played": 30,
                    "rating": 7.0,
                },
            ]
        ),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 10,
                    "global_id": "T10",
                    "name": "Team 10",
                    "league_id": 1,
                    "league_key": "ITA1",
                    "season": 2024,
                    "competition": "PO",
                    "pace": 70.0,
                    "offensive_rating": 108.0,
                    "defensive_rating": 104.0,
                    "three_point_attempt_rate": 0.30,
                    "assists_per_game": 19.0,
                    "star_player_usage": 0.27,
                    "net_rtg": 4.0,
                },
                {
                    "team_id": 11,
                    "global_id": "T11",
                    "name": "Team 11",
                    "league_id": 1,
                    "league_key": "ITA1",
                    "season": 2024,
                    "competition": "PO",
                    "pace": 72.0,
                    "offensive_rating": 106.0,
                    "defensive_rating": 105.0,
                    "three_point_attempt_rate": 0.32,
                    "assists_per_game": 18.0,
                    "star_player_usage": 0.25,
                    "net_rtg": 1.0,
                },
            ]
        ),
        "teams": pd.DataFrame(
            [
                {"id": 10, "global_id": "T10", "name": "Team 10", "league_id": 1},
                {"id": 11, "global_id": "T11", "name": "Team 11", "league_id": 1},
            ]
        ),
        "players": pd.DataFrame(
            [
                {"id": 1, "position": "PG", "current_team_id": 10, "current_league_id": 1},
                {"id": 2, "position": "SG", "current_team_id": 11, "current_league_id": 1},
            ]
        ),
        "player_dict": {
            1: {"id": 1, "position": "PG", "current_team_id": 10, "current_league_id": 1},
            2: {"id": 2, "position": "SG", "current_team_id": 11, "current_league_id": 1},
        },
        "league_dict": {1: {"id": 1, "name": "ITA1", "tier": 1}},
        "team_player_relations": pd.DataFrame(
            [
                {"player_id": 1, "team_id": 10, "season": 2024, "role": "PG"},
                {"player_id": 2, "team_id": 11, "season": 2024, "role": "SG"},
            ]
        ),
    }


def test_scope_cache_reuses_context_and_passes_only_target_histories():
    from basketball_ai.models.competition_training import scope_prediction_context

    data = _scope_data()
    one = scope_prediction_context(data, 1, 10, 1, "PO", 2024)
    two = scope_prediction_context(data, 2, 11, 1, "PO", 2024)

    assert set(one["player_stats"]["player_id"]) == {1}
    assert set(two["player_stats"]["player_id"]) == {2}
    assert set(one["team_season_stats"]["team_id"]) == {10}
    assert set(two["team_season_stats"]["team_id"]) == {11}
    assert one["_competition_support"]["exact_source_games"] == 8
    assert two["_competition_support"]["exact_source_games"] == 7
    assert len(data["_competition_scope_cache"]) == 1


def test_training_loader_skips_optional_simulation_views(monkeypatch):
    import basketball_ai.data.postgres_loader as loader

    class Engine:
        disposed = False

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(loader, "get_engine", lambda url=None: engine)

    core = {
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1"}]),
        "teams": pd.DataFrame([{"id": 10, "global_id": "T10", "name": "Team", "league_id": 1}]),
        "players": pd.DataFrame([{"id": 1, "global_id": "P1", "name": "Player", "position": "PG"}]),
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2024,
                    "rating": 7.0,
                    "competition": "RS",
                }
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [{"player_id": 1, "team_id": 10, "season": 2024}]
        ),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2024,
                    "competition": "RS",
                }
            ]
        ),
    }
    calls = []

    def fake_load_views(engine_arg, schema, views, *, optional=False):
        calls.append(optional)
        if optional:
            raise AssertionError("optional simulation views must not be queried")
        return {key: value.copy() for key, value in core.items()}

    monkeypatch.setattr(loader, "_load_views", fake_load_views)
    monkeypatch.setattr(loader, "_validate_contract", lambda data, schema: None)
    monkeypatch.setattr(loader, "_normalise_ids", lambda data: None)
    monkeypatch.setattr(loader, "_derive_playing_style", lambda teams: None)
    monkeypatch.setattr(loader, "_compute_star_player_usage", lambda teams, stats: None)
    monkeypatch.setattr(loader, "_fill_current_team_league", lambda players, stats: None)
    monkeypatch.setattr(loader, "_build_lookups", lambda data: None)

    result = loader.load_all_data(
        "postgresql+psycopg://example",
        "ai_source",
        include_optional=False,
    )
    assert calls == [False]
    assert engine.disposed is True
    assert result["source_contract"] == "competition-v2:canonical"
    assert all(
        result[key].empty
        for key in (
            "pbp_events",
            "lineup_stints",
            "play_type_stats",
            "shot_profiles",
            "causal_panel",
        )
    )


def test_model_threads_respect_configured_cpu_limit(monkeypatch):
    import basketball_ai.models.production_training as training

    monkeypatch.setattr(training.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("MODEL_CPU_THREADS", "6")
    assert training._model_threads() == 6

    monkeypatch.setenv("MODEL_CPU_THREADS", "64")
    assert training._model_threads() == 8

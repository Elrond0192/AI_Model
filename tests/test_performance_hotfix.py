from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_numeric_seasons_vectorized_mixed_values():
    from basketball_ai.models.production_training import _numeric_seasons

    frame = pd.DataFrame({"season": [2022, "2023-24", "2024", None]})
    values = _numeric_seasons(frame)
    assert values.iloc[0] == 2022
    assert values.iloc[1] == 2023
    assert values.iloc[2] == 2024
    assert pd.isna(values.iloc[3])


def test_missing_usg_uses_fractional_default_and_interaction_contract():
    from basketball_ai.models.performance_model import PerformanceModel

    model = PerformanceModel()
    history = pd.DataFrame(
        [
            {
                "player_id": 1,
                "league_id": 1,
                "season": 2024,
                "competition": "RS",
                "rating": 7.0,
                "minutes_per_game": 30.0,
                "usg_pct": np.nan,
                "obpm": 1.0,
                "games_played": 30,
            }
        ]
    )
    row = model._build_row(
        history.iloc[0],
        27,
        "PG",
        history,
        league_max_games={1: 34},
    )

    assert row["avg_usg_pct"] == pytest.approx(0.18)
    assert row["obpm_x_usg"] == pytest.approx(0.18)



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

def test_production_inference_uses_dataframe_builder_when_dataclass_fields_are_missing(monkeypatch):
    import basketball_ai.models.ensemble as ensemble_module
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.models.performance_model import PerformanceModel

    # This mimics the PostgreSQL source contract: core prediction columns are
    # present, while several legacy PlayerStats dataclass-required columns are
    # intentionally absent. The old inference path silently converted this
    # into _empty_features() defaults.
    data = {
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "team_id": 10,
                    "league_id": 1,
                    "season": 2024,
                    "competition": "RS",
                    "games_played": 30,
                    "minutes_per_game": 28.0,
                    "points": 18.0,
                    "rating": 7.1,
                }
            ]
        ),
        "player_dict": {
            1: {
                "id": 1,
                "name": "Player",
                "position": "PG",
                "age": 27,
                "current_team_id": 10,
                "current_league_id": 1,
            }
        },
        "team_dict": {
            10: {"id": 10, "league_id": 1},
        },
        "leagues": pd.DataFrame(),
    }

    perf = PerformanceModel()
    perf.role_encoding = {}
    perf.role_off_encoding = {}
    perf.role_def_encoding = {}
    perf.role_feature_encodings = {}
    perf.competition_feature_encodings = {}
    perf.feature_names = []
    perf.target_mode = "delta_vs_prior"
    perf.predict_from_features = lambda features: 0.0
    perf.predict_target_rating = lambda features: 7.0
    perf.get_shap_values = lambda features: {}

    class Compat:
        def score(self, player_id, team_id, scoped_data):
            return 0.5

    model = EnsembleModel(performance_model=perf, compatibility_model=Compat())
    model._capture_diagnostic_features = True
    model._league_factors = {"1": 1.0}
    model._mpg_baseline = 30.0

    monkeypatch.setattr(
        ensemble_module,
        "compute_context_features",
        lambda *args, **kwargs: {
            "position_team_fit": 0.8,
            "style_compatibility": 0.8,
            "role_opportunity": 0.8,
            "league_adaptation_factor": 1.0,
            "spacing_fit": 0.8,
        },
    )
    monkeypatch.setattr(ensemble_module, "age_performance_factor", lambda *args: 1.0)

    result = model._predict_uncached(
        1,
        10,
        data,
        season=2024,
        competition="RS",
    )

    assert result.predicted_rating >= 3.5
    # A real source row must not be mistaken for an empty history.
    # In particular, the old path returned form_score=5.0 and omitted age.
    assert model._last_prediction_features["form_score"] == 7.1
    assert model._last_prediction_features["age"] == 27.0
    assert model._last_prediction_features["pts_per_36"] == 23.14



def test_source_rate_normalization_handles_mixed_units_and_per40_outliers():
    from basketball_ai.models.performance_model import _canonical_source_value

    assert _canonical_source_value("usg_pct", 0.18) == 0.18
    assert _canonical_source_value("usg_pct", 18.0) == 0.18
    assert _canonical_source_value("ts_pct", 1.5) == 0.015
    assert _canonical_source_value("orb_pct", 6.0) == 0.06
    assert _canonical_source_value("pts_per_40", 24.0) == 24.0
    assert _canonical_source_value("pts_per_40", 240.0) == 60.0


def test_invalid_snapshot_age_uses_neutral_peak_age_instead_of_floor():
    from basketball_ai.constants import _peak_age
    from basketball_ai.models.production_training import _adjust_players_for_snapshot

    players = pd.DataFrame(
        [{"id": 1, "position": "PG", "age": 0}]
    )
    full_stats = pd.DataFrame(
        [
            {"player_id": 1, "season": 2024},
            {"player_id": 1, "season": 2025},
        ]
    )
    adjusted = _adjust_players_for_snapshot(
        players,
        full_stats,
        pd.DataFrame(),
        {},
        2024,
    )
    assert adjusted.loc[0, "age"] == int(round(_peak_age("PG")))
    assert adjusted.loc[0, "age"] != 14


def test_delta_model_uses_raw_xgb_target_as_final_prediction(monkeypatch):
    import basketball_ai.models.ensemble as ensemble_module
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.models.performance_model import PerformanceModel

    data = {
        "player_stats": pd.DataFrame(
            [{
                "player_id": 1, "team_id": 10, "league_id": 1, "season": 2024,
                "competition": "RS", "games_played": 30,
                "minutes_per_game": 28.0, "points": 18.0, "rating": 7.1,
            }]
        ),
        "player_dict": {
            1: {"id": 1, "position": "C", "age": 27, "current_team_id": 10, "current_league_id": 1}
        },
        "team_dict": {10: {"id": 10, "league_id": 1}},
        "leagues": pd.DataFrame(),
    }

    perf = PerformanceModel()
    perf.role_encoding = {}
    perf.role_off_encoding = {}
    perf.role_def_encoding = {}
    perf.role_feature_encodings = {}
    perf.competition_feature_encodings = {}
    perf.feature_names = []
    perf.target_mode = "delta_vs_prior"
    perf.predict_from_features = lambda features: -4.2
    perf.get_shap_values = lambda features: {}

    class Compat:
        def score(self, player_id, team_id, scoped_data):
            return 0.5

    model = EnsembleModel(performance_model=perf, compatibility_model=Compat())
    model._persistence_shrinkage_alpha = 0.0
    model._league_factors = {"1": 0.20}
    model._mpg_baseline = 30.0

    monkeypatch.setattr(
        ensemble_module,
        "age_performance_factor",
        lambda age, position: 1.0 if age == 27 else 0.40,
    )
    monkeypatch.setattr(
        ensemble_module,
        "compute_context_features",
        lambda *args, **kwargs: {
            "position_team_fit": 0.1,
            "style_compatibility": 0.1,
            "role_opportunity": 0.1,
            "league_adaptation_factor": 0.1,
            "spacing_fit": 0.1,
        },
    )

    result = model._predict_uncached(
        1,
        10,
        data,
        season=2024,
        target_age=22,
        competition="RS",
    )

    # 7.1 prior + (-4.2) XGB delta = 2.9. No 3.5 floor, age multiplier,
    # league/context multiplier, or persistence shrinkage may alter it.
    assert result.predicted_rating == pytest.approx(2.9)
    assert result.base_rating == pytest.approx(2.9)
    assert result.league_factor == 1.0
    assert result.context_adjustment == 1.0
    assert result._diagnostic_stages["final_prediction"] == pytest.approx(2.9)
    assert result._diagnostic_stages["persistence_shrinkage_alpha"] == pytest.approx(1.0)



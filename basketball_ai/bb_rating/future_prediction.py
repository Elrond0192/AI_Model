"""Direct season-ahead prediction of the official contextual BB-Rating.

This is deliberately separate from Prediction Model 2.6.0. Training labels are
recomputed by BBRatingEngine in each target-season/league/competition context;
features are restricted to the source season. Validation is rolling-origin OOF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from basketball_ai.bb_rating.engine import BBRatingEngine, BB_RATING_VERSION
from basketball_ai.bb_rating.competition_tiers import competition_tier
from basketball_ai.bb_rating.prediction_scenarios import (
    BBRatingScenarioCalibrator,
    SCENARIO_UNKNOWN,
    classify_transition,
)
from basketball_ai.models.competition_training import normalize_competition
from basketball_ai.models.production_training import _numeric_seasons

MODEL_VERSION = "1.2.0"
FEATURE_VERSION = "bb-rating-direct-target-v3-context-compatible"
_TEAM_NUMERIC_COLUMNS = (
    "pace", "offensive_rating", "defensive_rating", "net_rating",
    "three_point_attempt_rate", "assists_per_game", "star_player_usage",
    "turnover_rate", "rebounds_per_game", "offensive_rebounds_per_game",
    "three_point_pct", "free_throw_pct", "points_per_game", "win_pct",
    "bench_minutes", "roster_depth",
)
_ID_COLUMNS = {
    "id", "player_id", "league_id", "team_id", "opponent_team_id",
    "season", "global_id", "player_global_id", "league_key", "competition",
    "rating", "bb_rating", "target_rating", "prediction", "score",
}
_EXCLUDED_NUMERIC = _ID_COLUMNS | {
    "age", "age_band", "position_family", "competition_tier_label",
    "competition_tier", "birth_date", "date_of_birth",
}


def _year(value: Any) -> int | None:
    try:
        parsed = int(str(value).split("-")[0])
        return parsed if 1900 <= parsed <= 2200 else None
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _team_global_map(data: dict[str, Any]) -> dict[tuple[str, str, int], str]:
    frame = data.get("team_season_stats", pd.DataFrame())
    if frame is None or frame.empty:
        return {}
    observed: dict[tuple[str, str, int], set[str]] = {}
    for row in frame.to_dict("records"):
        year = _year(row.get("season"))
        team = _clean(row.get("team_id"))
        league = _clean(row.get("league_id"))
        global_id = _clean(row.get("global_id"))
        if year is not None and team and league and global_id:
            observed.setdefault((team, league, year), set()).add(global_id)
    # A local team/league/season key that maps to multiple global IDs is
    # ambiguous and must not be used to infer continuity or a transfer.
    return {
        key: next(iter(global_ids))
        for key, global_ids in observed.items()
        if len(global_ids) == 1
    }


def _target_league_tier(data: dict[str, Any], league_key: str) -> Any:
    wanted = str(league_key).strip().upper()
    # The shipped map is authoritative for known league codes. In particular,
    # never let a stale DB value such as tier=1 for LIT1 or ITA2 distort a scenario.
    configured_tier = competition_tier(wanted)
    if configured_tier is not None:
        return configured_tier

    leagues = data.get("leagues", pd.DataFrame())
    if leagues is not None and not leagues.empty and "tier" in leagues.columns:
        for row in leagues.to_dict("records"):
            candidates = {
                _clean(row.get("name")).upper(),
                _clean(row.get("league_key")).upper(),
            }
            if wanted not in candidates:
                continue
            try:
                tier = int(float(row.get("tier")))
            except (TypeError, ValueError):
                continue
            if tier > 0:
                return tier
    return None


def _team_identity(row: pd.Series, team_map: dict[tuple[str, str, int], str]) -> str | None:
    """Return only a resolved global team identity; local IDs are not cross-season proof."""
    team = _clean(row.get("team_id"))
    league = _clean(row.get("league_id"))
    year = _year(row.get("season"))
    if not team or not league or year is None:
        return None
    return team_map.get((team, league, year))


def _team_context_index(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    history = data.get("team_season_stats", pd.DataFrame())
    if history is None or history.empty or "global_id" not in history.columns:
        return {}
    rows = history.copy()
    rows["_year"] = _numeric_seasons(rows)
    leagues = data.get("leagues", pd.DataFrame())
    league_map: dict[str, str] = {}
    if leagues is not None and not leagues.empty:
        for item in leagues.to_dict("records"):
            league_id = _clean(item.get("id"))
            league_key = _clean(item.get("league_key")) or _clean(item.get("name"))
            if league_id and league_key:
                league_map[league_id] = league_key.upper()
    rows["_league_key"] = rows.get("league_key", pd.Series("", index=rows.index)).fillna("").astype(str).str.upper()
    if "league_id" in rows.columns:
        fallback = rows["league_id"].map(lambda value: league_map.get(_clean(value), ""))
        rows["_league_key"] = rows["_league_key"].where(rows["_league_key"].ne(""), fallback)
    rows = rows.dropna(subset=["_year", "global_id"])
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows.sort_values("_year", kind="stable").to_dict("records"):
        key = _clean(row.get("global_id"))
        if key:
            index.setdefault(key, []).append(row)
    return index


def _team_context_features(
    index: dict[str, list[dict[str, Any]]],
    team_global_id: str | None,
    target_league: str,
    source_year: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"target_team__{column}": np.nan for column in _TEAM_NUMERIC_COLUMNS
    }
    result["target_team_context_known"] = "no"
    result["target_team_context_league"] = ""
    result["target_team_context_seasons_ago"] = np.nan
    if not team_global_id:
        return result
    candidates = [
        row for row in index.get(str(team_global_id), [])
        if _year(row.get("season")) is not None and _year(row.get("season")) <= int(source_year)
    ]
    if not candidates:
        return result
    same_league = [
        row for row in candidates
        if _clean(row.get("_league_key")).upper() == str(target_league).upper()
    ]
    chosen = max(same_league or candidates, key=lambda row: _year(row.get("season")) or 0)
    context_year = _year(chosen.get("season"))
    for column in _TEAM_NUMERIC_COLUMNS:
        value = pd.to_numeric(pd.Series([chosen.get(column)]), errors="coerce").iloc[0]
        result[f"target_team__{column}"] = float(value) if pd.notna(value) and np.isfinite(value) else np.nan
    result["target_team_context_known"] = "yes"
    result["target_team_context_league"] = _clean(chosen.get("_league_key")).upper()
    result["target_team_context_seasons_ago"] = (
        float(int(source_year) - context_year) if context_year is not None else np.nan
    )
    return result


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in _EXCLUDED_NUMERIC or column.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(columns)


def _row_features(
    row: pd.Series,
    *,
    source_year: int,
    target_league: str,
    target_competition: str,
    source_team: str | None,
    target_team: str | None,
    numeric_columns: list[str],
    target_tier: Any = None,
    source_bb_rating: float | None = None,
    team_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in numeric_columns:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        result[f"stat__{column}"] = float(value) if pd.notna(value) and np.isfinite(value) else np.nan
    birth_date = row.get("birth_date", row.get("date_of_birth"))
    birth_year = _year(birth_date)
    age = source_year - birth_year if birth_year else pd.to_numeric(
        pd.Series([row.get("age")]), errors="coerce"
    ).iloc[0]
    result["age_as_of_source"] = float(age) if pd.notna(age) else np.nan
    result["source_bb_rating"] = float(source_bb_rating) if source_bb_rating is not None and np.isfinite(source_bb_rating) else np.nan
    result["source_league"] = _clean(row.get("league_key")) or _clean(row.get("league_id"))
    result["source_competition"] = normalize_competition(row.get("competition"))
    result["source_position"] = _clean(row.get("position"))
    result["source_role"] = _clean(row.get("ruolo_combinato"))
    result["source_tier"] = _clean(row.get("competition_tier"))
    resolved_target_tier = target_tier
    if resolved_target_tier is None or pd.isna(resolved_target_tier):
        resolved_target_tier = competition_tier(str(target_league).strip().upper())
    result["target_tier"] = _clean(resolved_target_tier)
    try:
        source_tier_num, target_tier_num = int(float(row.get("competition_tier"))), int(float(resolved_target_tier))
        result["tier_transition"] = (
            "up_tier" if target_tier_num < source_tier_num
            else "down_tier" if target_tier_num > source_tier_num
            else "same_tier"
        )
    except (TypeError, ValueError):
        result["tier_transition"] = "unknown_tier"
    result["target_league"] = str(target_league).strip().upper()
    result["target_competition"] = normalize_competition(target_competition)
    result.update(team_context or _team_context_features({}, None, target_league, source_year))
    result["scenario"] = classify_transition(
        source_team, target_team,
        _clean(row.get("league_key")) or _clean(row.get("league_id")),
        str(target_league).strip().upper(),
        normalize_competition(row.get("competition")),
        normalize_competition(target_competition),
        row.get("competition_tier"), resolved_target_tier,
    )
    return result


def _encode(frame: pd.DataFrame, categorical: list[str], columns: list[str] | None = None):
    encoded = pd.get_dummies(frame, columns=categorical, dummy_na=True, dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded.fillna(0.0).astype(float)


_POLICY_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "source_destination_scenario": ("source_league", "target_league", "scenario"),
    "destination_league_scenario": ("target_league", "scenario"),
    "scenario": ("scenario",),
    "source_league": ("source_league",),
    "destination_league": ("target_league",),
}
_POLICY_DIMENSION_ORDER = (
    "source_destination_scenario",
    "destination_league_scenario",
    "scenario",
    "source_league",
    "destination_league",
)


def _row_policy_keys(row: Any) -> dict[str, str]:
    source = _clean(row.get("source_league")).upper()
    destination = _clean(row.get("target_league")).upper()
    scenario = _clean(row.get("scenario")) or SCENARIO_UNKNOWN
    return {
        "source_destination_scenario": f"{source}=>{destination}::{scenario}",
        "destination_league_scenario": f"{destination}::{scenario}",
        "scenario": scenario,
        "source_league": source,
        "destination_league": destination,
    }


def _support_counts(frame: pd.DataFrame) -> dict[str, int | float | None]:
    """Expose row volume separately from independent player/transition/context support."""
    rows = int(len(frame))
    players = (
        int(frame["player_global_id"].nunique(dropna=True))
        if "player_global_id" in frame.columns else None
    )
    transition_columns = [column for column in ("player_global_id", "target_season")
                          if column in frame.columns]
    target_columns = [column for column in (
        "player_global_id", "target_season", "target_league", "target_competition"
    ) if column in frame.columns]
    transitions = (
        int(frame[transition_columns].dropna(subset=transition_columns).drop_duplicates().shape[0])
        if len(transition_columns) == 2 else None
    )
    target_contexts = (
        int(frame[target_columns].dropna(subset=target_columns).drop_duplicates().shape[0])
        if len(target_columns) == 4 else None
    )
    return {
        "rows_evaluated": rows,
        "unique_players": players,
        "unique_player_season_transitions": transitions,
        "unique_target_contexts": target_contexts,
        "rows_per_target_context": (
            round(rows / target_contexts, 3) if target_contexts else None
        ),
    }


def _player_season_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each player-season transition equal total weight across its target contexts."""
    if not {"player_global_id", "target_season"}.issubset(frame.columns) or frame.empty:
        return np.ones(len(frame), dtype=float)
    counts = frame.groupby(
        ["player_global_id", "target_season"], dropna=False
    )["target_season"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(counts, 1.0)
    mean_weight = float(np.mean(weights))
    return weights / mean_weight if mean_weight > 0 else np.ones(len(frame), dtype=float)


def _build_scenario_policy_maps(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    min_samples: int = 40,
) -> dict[str, dict[str, dict[str, float | int | bool]]]:
    """Measure whether the calibrated model is safe to use within OOF cohorts."""
    maps: dict[str, dict[str, dict[str, float | int | bool]]] = {
        dimension: {} for dimension in _POLICY_DIMENSION_ORDER
    }
    if frame.empty:
        return maps
    work = frame.copy()
    work["_policy_prediction"] = np.asarray(predictions, dtype=float)
    work["_policy_actual"] = pd.to_numeric(work["observed_bb_rating"], errors="coerce")
    work["_policy_baseline"] = pd.to_numeric(work["baseline_bb_rating"], errors="coerce")
    key_columns: dict[str, tuple[str, ...]] = {
        "source_destination_scenario": ("source_league", "target_league", "scenario"),
        "destination_league_scenario": ("target_league", "scenario"),
        "scenario": ("scenario",),
        "source_league": ("source_league",),
        "destination_league": ("target_league",),
    }
    for dimension in _POLICY_DIMENSION_ORDER:
        columns = key_columns[dimension]
        for values, group in work.groupby(list(columns), dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            n = int(len(group))
            support = _support_counts(group)
            # Once player identity is available, row count is not accepted as a
            # substitute for independent support. Missing IDs retain the legacy
            # row-count behaviour only for standalone callers without identity data.
            effective_support = support["unique_players"] if support["unique_players"] is not None else n
            if int(effective_support) < min_samples:
                continue
            if dimension == "source_destination_scenario":
                key = f"{_clean(values[0]).upper()}=>{_clean(values[1]).upper()}::{_clean(values[2]) or SCENARIO_UNKNOWN}"
            elif dimension == "destination_league_scenario":
                key = f"{_clean(values[0]).upper()}::{_clean(values[1]) or SCENARIO_UNKNOWN}"
            elif dimension == "source_league":
                key = _clean(values[0]).upper()
            elif dimension == "destination_league":
                key = _clean(values[0]).upper()
            else:
                key = _clean(values[0]) or SCENARIO_UNKNOWN

            actual = group["_policy_actual"].to_numpy(dtype=float)
            model_prediction = group["_policy_prediction"].to_numpy(dtype=float)
            baseline_prediction = group["_policy_baseline"].to_numpy(dtype=float)
            sample_weight = _player_season_weights(group)
            model_mae = float(mean_absolute_error(actual, model_prediction, sample_weight=sample_weight))
            baseline_mae = float(mean_absolute_error(actual, baseline_prediction, sample_weight=sample_weight))
            allowed_mae = baseline_mae * 1.10 + 0.25
            maps[dimension][key] = {
                "n": n,
                **support,
                "support_unit": "unique_players" if support["unique_players"] is not None else "rows",
                "mae": model_mae,
                "baseline_mae": baseline_mae,
                "allowed_mae": float(allowed_mae),
                "use_model": bool(model_mae <= allowed_mae),
            }
    return maps


def _choose_scenario_prediction(
    row: Any,
    model_prediction: float,
    baseline_prediction: float,
    policy_maps: dict[str, dict[str, dict[str, float | int | bool]]],
) -> tuple[float, str]:
    """Apply only cohort decisions that could have been learned before this row."""
    scenario = _clean(row.get("scenario")) or SCENARIO_UNKNOWN
    if scenario == SCENARIO_UNKNOWN:
        return float(baseline_prediction), "unknown_scenario"

    keys = _row_policy_keys(row)
    matching_policies = [
        (dimension, policy_maps.get(dimension, {}).get(keys[dimension]))
        for dimension in _POLICY_DIMENSION_ORDER
    ]
    matching_policies = [
        (dimension, policy) for dimension, policy in matching_policies
        if policy is not None
    ]
    failed = next(
        ((dimension, policy) for dimension, policy in matching_policies
         if not bool(policy.get("use_model"))),
        None,
    )
    if failed is not None:
        return float(baseline_prediction), f"validation_gate_failed:{failed[0]}"

    if matching_policies:
        return float(model_prediction), "validated_model"

    source_league = _clean(row.get("source_league")).upper()
    target_league = _clean(row.get("target_league")).upper()
    source_competition = normalize_competition(row.get("source_competition"))
    target_competition = normalize_competition(row.get("target_competition"))
    same_context = (
        source_league == target_league
        and source_competition == target_competition
    )
    if not same_context:
        return float(baseline_prediction), "cross_context_not_yet_validated"
    return float(model_prediction), "same_context_no_failing_gate"


def _build_baseline_means(records: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(records["observed_bb_rating"], errors="coerce")
    valid = records.loc[target.notna()].copy()
    valid["_observed_target"] = target[target.notna()].astype(float)
    by_league_competition = {
        f"{_clean(league).upper()}::{normalize_competition(competition)}": float(group["_observed_target"].mean())
        for (league, competition), group in valid.groupby(["target_league", "target_competition"], dropna=False)
    }
    by_league = {
        _clean(league).upper(): float(group["_observed_target"].mean())
        for league, group in valid.groupby("target_league", dropna=False)
    }
    return {
        "by_league_competition": by_league_competition,
        "by_league": by_league,
        "global": float(valid["_observed_target"].mean()),
    }


def _context_baseline(features: dict[str, Any], means: dict[str, Any]) -> float | None:
    source_league = _clean(features.get("source_league")).upper()
    target_league = _clean(features.get("target_league")).upper()
    source_competition = normalize_competition(features.get("source_competition"))
    target_competition = normalize_competition(features.get("target_competition"))
    source_rating = pd.to_numeric(pd.Series([features.get("source_bb_rating")]), errors="coerce").iloc[0]
    if (
        source_league == target_league
        and source_competition == target_competition
        and pd.notna(source_rating)
        and np.isfinite(source_rating)
    ):
        return float(np.clip(source_rating, 1, 100))
    key = f"{target_league}::{target_competition}"
    value = means.get("by_league_competition", {}).get(key)
    if value is None:
        value = means.get("by_league", {}).get(target_league)
    if value is None:
        value = means.get("global")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return float(np.clip(value, 1, 100)) if np.isfinite(value) else None


class BBRatingFuturePredictionModel:
    """Predict official BB-Rating(t+1) directly on its 1-100 scale."""

    def __init__(self, *, min_oof_samples: int = 500, random_state: int = 42) -> None:
        self.min_oof_samples = max(20, int(min_oof_samples))
        self.random_state = int(random_state)
        self.model: HistGradientBoostingRegressor | None = None
        self.calibrator: BBRatingScenarioCalibrator | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns = [
            "source_league", "source_competition", "source_position", "source_role",
            "target_league", "target_competition", "scenario", "source_tier",
            "target_tier", "tier_transition", "target_team_context_known",
            "target_team_context_league",
        ]
        self.residual_intervals: dict[str, dict[str, float | int]] = {}
        self.baseline_means: dict[str, Any] = {}
        self.scenario_fallback_policy: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.is_trained = False

    def _build_records(
        self, data: dict[str, Any], *, include_latest_target_season: bool = False
    ) -> pd.DataFrame:
        stats = data.get("player_stats", pd.DataFrame())
        players = data.get("players", pd.DataFrame())
        if stats is None or stats.empty:
            raise ValueError("BB Prediction Model requires canonical player season statistics")
        engine = BBRatingEngine(stats, players, data.get("leagues"))
        frame = engine.frame.copy()
        # Join date of birth only to derive source-season age. Never use the
        # mutable current-age field as a temporal feature when birth date exists.
        if players is not None and not players.empty and "id" in players.columns:
            birth_columns = ["id"] + [
                column for column in ("birth_date", "date_of_birth")
                if column in players.columns and column not in frame.columns
            ]
            if len(birth_columns) > 1:
                birth = players[birth_columns].drop_duplicates("id").rename(columns={"id": "player_id"})
                frame = frame.merge(birth, on="player_id", how="left")
        frame["_year"] = _numeric_seasons(frame)
        frame = frame.dropna(subset=["_year", "player_global_id", "league_key", "competition"])
        frame["_year"] = frame["_year"].astype(int)
        frame["competition"] = frame["competition"].map(normalize_competition)
        frame["league_key"] = frame["league_key"].astype(str).str.strip().str.upper()
        latest_observed_year = int(frame["_year"].max())
        # Match BBRatingEngine.rate_player identity semantics and avoid overweighting
        # provider-specific duplicate player IDs in the same canonical context.
        frame = frame.drop_duplicates(
            subset=["player_global_id", "league_key", "_year", "competition"], keep="first"
        ).copy()
        # Official labels are computed from the full peer context for each target
        # season. The model never receives target-season statistics as features.
        labels: dict[tuple[str, int, str, str], int] = {}
        for (league, season, competition), group in frame.groupby(
            ["league_key", "_year", "competition"], sort=False
        ):
            for player_global_id in group["player_global_id"].astype(str).unique():
                try:
                    result = engine.rate_player(
                        player_global_id, league=str(league), season=int(season), phase=str(competition)
                    )
                    labels[(str(player_global_id), int(season), str(league), str(competition))] = int(result.score)
                except (ValueError, KeyError):
                    continue

        team_map = _team_global_map(data)
        team_context_index = _team_context_index(data)
        self.numeric_columns = _numeric_feature_columns(frame)
        records: list[dict[str, Any]] = []
        # Canonical global identity supports provider-specific player IDs changing.
        grouped = {key: group for key, group in frame.groupby("player_global_id", sort=False)}
        for player_global_id, group in grouped.items():
            source_rows = group.to_dict("records")
            by_year: dict[int, list[dict[str, Any]]] = {}
            for row in source_rows:
                year = int(row["_year"])
                by_year.setdefault(year, []).append(row)
            for source_year, origins in by_year.items():
                target_year = source_year + 1
                if not include_latest_target_season and target_year >= latest_observed_year:
                    continue
                destinations = by_year.get(target_year, [])
                if not destinations:
                    continue
                # Build one row per target player/season/league/statistical context.
                # Competition codes also contain views/phases (HOME, AWAY, RS, TOT,
                # etc.); they are not independent competition identities. Pair only
                # like-for-like contexts and never cross-join all origins to all targets.
                for target_raw in destinations:
                    target = pd.Series(target_raw)
                    target_league = _clean(target.get("league_key")).upper()
                    target_comp = normalize_competition(target.get("competition"))
                    label_key = (str(player_global_id), source_year + 1, target_league, target_comp)
                    target_score = labels.get(label_key)
                    if target_score is None:
                        continue

                    compatible_origins = [
                        origin for origin in origins
                        if normalize_competition(origin.get("competition")) == target_comp
                    ]
                    if not compatible_origins:
                        continue

                    # Prefer the source row from the destination league when it
                    # exists. Otherwise select one deterministic primary source
                    # context by usage; this prevents target reuse across origins.
                    same_league_origins = [
                        origin for origin in compatible_origins
                        if _clean(origin.get("league_key")).upper() == target_league
                    ]
                    candidates = same_league_origins or compatible_origins
                    candidate_frame = pd.DataFrame(candidates).copy()
                    usage_columns = [
                        column for column in ("games_played", "minutes", "minutes_per_game")
                        if column in candidate_frame.columns
                    ]
                    for column in usage_columns:
                        candidate_frame[column] = pd.to_numeric(candidate_frame[column], errors="coerce")
                    sort_columns = list(usage_columns)
                    ascending = [False] * len(usage_columns)
                    if "league_key" in candidate_frame.columns:
                        sort_columns.append("league_key")
                        ascending.append(True)
                    source_raw = (
                        candidate_frame.sort_values(
                            sort_columns, ascending=ascending, na_position="last", kind="stable"
                        ).iloc[0].to_dict()
                        if sort_columns else candidates[0]
                    )
                    source = pd.Series(source_raw)
                    source_team = _team_identity(source, team_map)
                    source_league = _clean(source.get("league_key")).upper()
                    source_comp = normalize_competition(source.get("competition"))
                    source_score = labels.get(
                        (str(player_global_id), source_year, source_league, source_comp)
                    )
                    if source_score is None:
                        continue

                    target_team = _team_identity(target, team_map)
                    features = _row_features(
                        source, source_year=source_year,
                        target_league=target_league, target_competition=target_comp,
                        source_team=source_team, target_team=target_team,
                        numeric_columns=self.numeric_columns,
                        target_tier=target.get("competition_tier"),
                        source_bb_rating=float(source_score),
                        team_context=_team_context_features(
                            team_context_index, target_team, target_league, source_year
                        ),
                    )
                    records.append({
                        **features,
                        "observed_bb_rating": float(target_score),
                        "target_season": source_year + 1,
                        "player_global_id": str(player_global_id),
                        "source_team": source_team or "",
                        "target_team": target_team or "",
                        "source_league": source_league,
                        "source_competition": source_comp,
                    })
        result = pd.DataFrame(records)
        if result.empty:
            raise ValueError("No consecutive-season BB-Rating target pairs could be constructed")
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.dropna(subset=["observed_bb_rating", "target_season"])
        result["observed_bb_rating"] = pd.to_numeric(result["observed_bb_rating"], errors="coerce")
        result = result[result["observed_bb_rating"].between(1, 100)]
        if result.empty:
            raise ValueError("No valid official BB-Rating labels in the 1-100 range")
        target_key = ["player_global_id", "target_season", "target_league", "target_competition"]
        if result.duplicated(subset=target_key).any():
            raise ValueError(
                "BB Prediction Model pairing invariant failed: duplicate player-season target contexts"
            )
        return result

    def fit(
        self, data: dict[str, Any], *, include_latest_target_season: bool = False
    ) -> dict[str, Any]:
        records = self._build_records(
            data, include_latest_target_season=include_latest_target_season
        )
        feature_frame = records.drop(columns=[
            "observed_bb_rating", "target_season", "player_global_id", "source_team", "target_team"
        ])
        encoded = _encode(feature_frame, self.categorical_columns)
        self.feature_columns = list(encoded.columns)
        y = records["observed_bb_rating"].to_numpy(dtype=float)
        sample_weights = _player_season_weights(records)
        years = records["target_season"].to_numpy(dtype=int)
        oof_predictions = np.full(len(records), np.nan, dtype=float)
        baseline_predictions = np.full(len(records), np.nan, dtype=float)
        # Expanding walk-forward: each fold trains strictly before its target season.
        for target_year in sorted(set(years)):
            test_idx = np.flatnonzero(years == target_year)
            train_idx = np.flatnonzero(years < target_year)
            if len(train_idx) < 100 or len(test_idx) == 0:
                continue
            fold = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=2.0, random_state=self.random_state,
            )
            fold.fit(
                encoded.iloc[train_idx], y[train_idx],
                sample_weight=sample_weights[train_idx],
            )
            oof_predictions[test_idx] = np.clip(fold.predict(encoded.iloc[test_idx]), 1, 100)

            train_records = records.iloc[train_idx]
            global_mean = float(np.mean(y[train_idx]))
            for idx in test_idx:
                sample = records.iloc[idx]
                same_context = (
                    str(sample["source_league"]).upper() == str(sample["target_league"]).upper()
                    and str(sample["source_competition"]).upper() == str(sample["target_competition"]).upper()
                )
                if same_context and pd.notna(sample.get("source_bb_rating")):
                    baseline_predictions[idx] = float(sample["source_bb_rating"])
                    continue
                destination = train_records[
                    (train_records["target_league"].astype(str) == str(sample["target_league"]))
                    & (train_records["target_competition"].astype(str) == str(sample["target_competition"]))
                ]
                if destination.empty:
                    destination = train_records[
                        train_records["target_league"].astype(str) == str(sample["target_league"])
                    ]
                baseline_predictions[idx] = float(
                    destination["observed_bb_rating"].mean()
                    if not destination.empty else global_mean
                )

        valid_oof = np.isfinite(oof_predictions)
        oof = records.loc[valid_oof].copy()
        oof["predicted_bb_rating"] = oof_predictions[valid_oof]
        oof["baseline_bb_rating"] = baseline_predictions[valid_oof]
        if len(oof) < self.min_oof_samples:
            raise ValueError(
                f"Only {len(oof)} walk-forward OOF BB-Rating predictions; "
                f"at least {self.min_oof_samples} are required. No artifact was created."
            )

        # Evaluate calibration itself without fitting it on the row being scored:
        # each target-season fold uses only OOF calibration rows from earlier years.
        nested_calibrated = oof["predicted_bb_rating"].to_numpy(float).copy()
        target_seasons = oof["target_season"].to_numpy(dtype=int)
        for target_year in sorted(oof["target_season"].unique()):
            test_positions = np.flatnonzero(target_seasons == target_year)
            prior_positions = np.flatnonzero(target_seasons < target_year)
            prior = oof.iloc[prior_positions].copy()
            if len(prior) < 40:
                continue
            prior["target_league_id"] = prior["target_league"].astype(str)
            fold_calibrator = BBRatingScenarioCalibrator(min_samples=40)
            try:
                fold_calibrator.fit(prior)
                for position in test_positions:
                    row = oof.iloc[position]
                    nested_calibrated[position] = fold_calibrator.predict(
                        float(row["predicted_bb_rating"]),
                        scenario=str(row["scenario"]),
                        target_league_id=str(row["target_league"]),
                    )["predicted_bb_rating"]
            except (ValueError, RuntimeError):
                # Without prior calibration support, keep the raw OOF prediction.
                continue

        # Validate the *serving policy* as a temporal forecast. A previously
        # failing or unvalidated context falls back to the exact same baseline
        # used by the walk-forward gate, rather than shipping a known regression.
        nested_policy_predictions = nested_calibrated.copy()
        fallback_reasons: dict[str, int] = {}
        for target_year in sorted(oof["target_season"].unique()):
            test_positions = np.flatnonzero(target_seasons == target_year)
            prior_positions = np.flatnonzero(target_seasons < target_year)
            prior = oof.iloc[prior_positions].copy()
            prior_policies = _build_scenario_policy_maps(
                prior, nested_calibrated[prior_positions], min_samples=40
            )
            for position in test_positions:
                row = oof.iloc[position]
                selected, reason = _choose_scenario_prediction(
                    row,
                    float(nested_calibrated[position]),
                    float(row["baseline_bb_rating"]),
                    prior_policies,
                )
                nested_policy_predictions[position] = selected
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        # The final serving policy is learned from historical OOF evidence only.
        # The reported gate metrics above are still based on temporally nested
        # decisions, not on this full-history policy.
        self.scenario_fallback_policy = {
            "minimum_samples": 40,
            "group_dimensions": _build_scenario_policy_maps(
                oof, nested_calibrated, min_samples=40
            ),
            "unknown_scenario_always_fallback": True,
            "unvalidated_cross_context_fallback": True,
        }
        self.baseline_means = _build_baseline_means(records)

        oof["model_calibrated_bb_rating"] = np.clip(nested_calibrated, 1, 100)
        oof["calibrated_bb_rating"] = np.clip(nested_policy_predictions, 1, 100)
        self.calibrator = BBRatingScenarioCalibrator(min_samples=40)
        oof["target_league_id"] = oof["target_league"].astype(str)
        calibration_support_report = self.calibrator.fit(oof)
        residual = oof["observed_bb_rating"].to_numpy(float) - nested_policy_predictions
        global_interval_support = _support_counts(oof)
        self.residual_intervals = {"global": {
            "n": int(len(residual)),
            "unique_players": int(global_interval_support["unique_players"] or 0),
            "unique_player_season_transitions": int(
                global_interval_support["unique_player_season_transitions"] or 0
            ),
            "low": float(np.quantile(residual, 0.10)),
            "high": float(np.quantile(residual, 0.90)),
        }}
        for (league, scenario), group in oof.assign(_residual=residual).groupby(
            ["target_league", "scenario"], sort=False
        ):
            interval_support = _support_counts(group)
            supported = (
                int(interval_support["unique_players"]) >= 40
                if interval_support["unique_players"] is not None else len(group) >= 40
            )
            if supported:
                self.residual_intervals[f"{league}::{scenario}"] = {
                    "n": int(len(group)),
                    "unique_players": int(interval_support["unique_players"] or 0),
                    "unique_player_season_transitions": int(
                        interval_support["unique_player_season_transitions"] or 0
                    ),
                    "low": float(group["_residual"].quantile(0.10)),
                    "high": float(group["_residual"].quantile(0.90)),
                }

        actual = oof["observed_bb_rating"].to_numpy(float)
        baseline = oof["baseline_bb_rating"].to_numpy(float)
        calibrated = oof["calibrated_bb_rating"].to_numpy(float)

        def metric_pack(mask: np.ndarray) -> dict[str, Any]:
            if not np.any(mask):
                return {
                    "n": 0, "rows_evaluated": 0, "unique_players": 0,
                    "unique_player_season_transitions": 0, "unique_target_contexts": 0,
                    "rows_per_target_context": None, "mae": None, "rmse": None,
                    "bias": None, "calibration_error": None, "baseline_mae": None,
                    "baseline_rmse": None, "beats_baseline": None,
                }
            group = oof.loc[mask].copy()
            support = _support_counts(group)
            truth = group["observed_bb_rating"].to_numpy(dtype=float)
            prediction = group["calibrated_bb_rating"].to_numpy(dtype=float)
            baseline_prediction = group["baseline_bb_rating"].to_numpy(dtype=float)
            sample_weight = _player_season_weights(group)
            mae = float(mean_absolute_error(truth, prediction, sample_weight=sample_weight))
            baseline_mae_group = float(mean_absolute_error(
                truth, baseline_prediction, sample_weight=sample_weight
            ))
            prediction_rmse = float(np.sqrt(np.average(
                np.square(prediction - truth), weights=sample_weight
            )))
            baseline_rmse = float(np.sqrt(np.average(
                np.square(baseline_prediction - truth), weights=sample_weight
            )))
            weighted_prediction_mean = float(np.average(prediction, weights=sample_weight))
            weighted_actual_mean = float(np.average(truth, weights=sample_weight))
            return {
                "n": int(mask.sum()),
                **support,
                "mae": mae,
                "rmse": prediction_rmse,
                "bias": float(np.average(prediction - truth, weights=sample_weight)),
                "calibration_error": float(abs(weighted_prediction_mean - weighted_actual_mean)),
                "baseline_mae": baseline_mae_group,
                "baseline_rmse": baseline_rmse,
                "beats_baseline": bool(mae < baseline_mae_group),
            }

        by_scenario = {}
        for scenario in sorted(oof["scenario"].astype(str).unique()):
            by_scenario[scenario] = metric_pack(oof["scenario"].astype(str).to_numpy() == scenario)
        by_source = {}
        for league in sorted(oof["source_league"].astype(str).unique()):
            by_source[league] = metric_pack(oof["source_league"].astype(str).to_numpy() == league)
        by_destination = {}
        for league in sorted(oof["target_league"].astype(str).unique()):
            by_destination[league] = metric_pack(oof["target_league"].astype(str).to_numpy() == league)
        by_source_destination_scenario = {}
        for (source_league, target_league, scenario), group in oof.groupby(
            ["source_league", "target_league", "scenario"], sort=True
        ):
            key = f"{source_league}=>{target_league}::{scenario}"
            by_source_destination_scenario[key] = metric_pack(
                (oof["source_league"].astype(str).to_numpy() == str(source_league))
                & (oof["target_league"].astype(str).to_numpy() == str(target_league))
                & (oof["scenario"].astype(str).to_numpy() == str(scenario))
            )
        by_destination_scenario = {}
        for (league, scenario), group in oof.groupby(["target_league", "scenario"], sort=True):
            key = f"{league}::{scenario}"
            by_destination_scenario[key] = metric_pack(
                (oof["target_league"].astype(str).to_numpy() == str(league))
                & (oof["scenario"].astype(str).to_numpy() == str(scenario))
            )
        overall_metrics = metric_pack(np.ones(len(oof), dtype=bool))
        mae_model = float(overall_metrics["mae"])
        mae_baseline = float(overall_metrics["baseline_mae"])

        segment_regressions = []
        under_supported_segments = []
        for dimension, metrics_by_group in (
            ("scenario", by_scenario),
            ("source_league", by_source),
            ("destination_league", by_destination),
            ("destination_league_scenario", by_destination_scenario),
            ("source_destination_scenario", by_source_destination_scenario),
        ):
            for group_name, metrics in metrics_by_group.items():
                effective_support = metrics.get("unique_players")
                support_count = int(
                    effective_support if effective_support is not None
                    else (metrics.get("n") or 0)
                )
                if support_count < 40:
                    under_supported_segments.append({
                        "dimension": dimension,
                        "group": str(group_name),
                        "n": int(metrics.get("n") or 0),
                        "rows_evaluated": int(metrics.get("rows_evaluated") or metrics.get("n") or 0),
                        "unique_players": metrics.get("unique_players"),
                        "unique_player_season_transitions": metrics.get("unique_player_season_transitions"),
                        "unique_target_contexts": metrics.get("unique_target_contexts"),
                        "support_unit": "unique_players" if effective_support is not None else "rows",
                        "status": "insufficient_independent_player_support_for_segment_gate",
                    })
                    continue
                model_mae = metrics.get("mae")
                baseline_mae_group = metrics.get("baseline_mae")
                if model_mae is None or baseline_mae_group is None:
                    continue
                allowed_mae = float(baseline_mae_group) * 1.10 + 0.25
                if float(model_mae) > allowed_mae:
                    segment_regressions.append({
                        "dimension": dimension, "group": str(group_name),
                        "n": int(metrics["n"]),
                        "rows_evaluated": int(metrics.get("rows_evaluated") or metrics["n"]),
                        "unique_players": metrics.get("unique_players"),
                        "unique_player_season_transitions": metrics.get("unique_player_season_transitions"),
                        "unique_target_contexts": metrics.get("unique_target_contexts"),
                        "mae": float(model_mae),
                        "baseline_mae": float(baseline_mae_group),
                    })
        walk_forward_target_seasons = sorted(int(value) for value in oof["target_season"].unique())
        # OOF predictions from a single held-out season are not sufficient to
        # establish temporal generalisation, even when their row count is large.
        overall_unique_players = int(overall_metrics.get("unique_players") or 0)
        promotion_gate_passed = bool(
            len(walk_forward_target_seasons) >= 3
            and overall_unique_players >= 40
            and mae_model < mae_baseline
            and not segment_regressions
        )

        self.model = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=self.random_state,
        )
        self.model.fit(encoded, y, sample_weight=sample_weights)
        calibrated_oof = np.clip(calibrated, 1, 100)
        weighted_oof = _player_season_weights(oof)
        raw_calibrated_model = np.clip(nested_calibrated, 1, 100)
        oof_after_calibration_mae = float(mean_absolute_error(
            actual, raw_calibrated_model, sample_weight=weighted_oof
        ))
        oof_after_fallback_mae = float(mean_absolute_error(
            actual, calibrated_oof, sample_weight=weighted_oof
        ))
        self.metadata = {
            "model": "BBRatingFuturePredictionModel",
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "target": "official_bb_rating_t_plus_1",
            "target_scale": [1, 100],
            "bb_rating_version": BB_RATING_VERSION,
            "calibration_method": "walk_forward_oof_isotonic_with_baseline_safe_scenario_fallback",
            "training_samples": int(len(records)),
            "oof_samples": int(len(oof)),
            "training_support": _support_counts(records),
            "oof_support": _support_counts(oof),
            "calibration_support": calibration_support_report,
            "duplicate_target_context_pairs": int(records.duplicated(
                subset=["player_global_id", "target_season", "target_league", "target_competition"]
            ).sum()),
            "unresolved_team_identity_rows": int((
                records["source_team"].astype(str).eq("")
                | records["target_team"].astype(str).eq("")
            ).sum()),
            "target_seasons": sorted(int(v) for v in set(years)),
            "oof_mae_before_calibration": float(mean_absolute_error(
                oof["observed_bb_rating"], oof["predicted_bb_rating"], sample_weight=weighted_oof
            )),
            "oof_mae_calibrated_model": float(mean_absolute_error(
                oof["observed_bb_rating"], nested_calibrated, sample_weight=weighted_oof
            )),
            "oof_mae_after_calibration": oof_after_calibration_mae,
            "oof_mae_after_scenario_fallback": oof_after_fallback_mae,
            "oof_rmse_calibrated_model": float(np.sqrt(np.average(
                np.square(nested_calibrated - actual), weights=weighted_oof
            ))),
            "oof_rmse_after_calibration": float(np.sqrt(np.average(
                np.square(raw_calibrated_model - actual), weights=weighted_oof
            ))),
            "oof_bias_after_calibration": float(np.average(
                raw_calibrated_model - actual, weights=weighted_oof
            )),
            "oof_calibration_error": float(abs(
                np.average(raw_calibrated_model, weights=weighted_oof)
                - np.average(actual, weights=weighted_oof)
            )),
            "oof_rmse_after_scenario_fallback": float(np.sqrt(np.average(
                np.square(calibrated_oof - actual), weights=weighted_oof
            ))),
            "oof_baseline_mae": mae_baseline,
            "oof_baseline_rmse": float(np.sqrt(np.average(
                np.square(actual - baseline), weights=weighted_oof
            ))),
            "promotion_gate_passed": promotion_gate_passed,
            "segment_regressions": segment_regressions,
            "under_supported_segments": under_supported_segments,
            "scenario_fallback_policy": self.scenario_fallback_policy,
            "baseline_context_means": self.baseline_means,
            "oof_prediction_policy_counts": fallback_reasons,
            "walk_forward_target_seasons": walk_forward_target_seasons,
            "walk_forward_fold_count": int(len(walk_forward_target_seasons)),
            "minimum_walk_forward_target_seasons": 3,
            "minimum_oof_unique_players": 40,
            "minimum_supported_segment_samples": 40,
            "minimum_supported_segment_players": 40,
            "segment_support_unit": "unique player_global_id values; rows are reported separately",
            "pairing_policy": "one primary source context per player-target-season-league-competition; source and target statistical context codes must match",
            "team_identity_policy": "only resolved global team IDs establish team continuity; local fallback is not treated as verified",
            "promotion_gate": "the player-season-balanced temporally nested serving policy must beat baseline overall, have at least 3 OOF target seasons and at least 40 distinct OOF players, and keep supported source-league, destination-league, transition-scenario, destination-league-scenario and source-destination-scenario cohorts within 10 percent plus 0.25 points of baseline; cohort support requires 40 distinct players; unvalidated or regressing contexts use the contextual baseline",
            "scenario_metrics": by_scenario,
            "source_league_metrics": by_source,
            "destination_league_metrics": by_destination,
            "source_destination_scenario_metrics": by_source_destination_scenario,
            "destination_league_scenario_metrics": by_destination_scenario,
            "scenario_counts": {
                str(k): int(v) for k, v in records["scenario"].value_counts().items()
            },
            "residual_interval_method": "empirical_oof_10_90_percentiles",
            "data_cutoff_season": int(records["target_season"].max()),
            "latest_target_season_included": bool(include_latest_target_season),
        }
        self.is_trained = True
        return dict(self.metadata)

    def _predict_features(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.is_trained or self.model is None or self.calibrator is None:
            raise RuntimeError("BB Prediction Model artifact is not trained")
        encoded = _encode(pd.DataFrame([features]), self.categorical_columns, self.feature_columns)
        raw = float(np.clip(self.model.predict(encoded)[0], 1, 100))
        scenario = str(features.get("scenario") or SCENARIO_UNKNOWN)
        league = str(features.get("target_league") or "")
        calibrated = self.calibrator.predict(
            raw, scenario=scenario, target_league_id=league
        )
        interval = self.residual_intervals.get(
            f"{league}::{scenario}", self.residual_intervals["global"]
        )
        model_score = float(calibrated["predicted_bb_rating"])
        score = model_score
        prediction_method = "scenario_calibrated_model"
        fallback_reason = None
        policy = getattr(self, "scenario_fallback_policy", {}) or {}
        means = getattr(self, "baseline_means", {}) or {}
        if policy.get("group_dimensions") and means:
            baseline = _context_baseline(features, means)
            if baseline is not None:
                row = {
                    "source_league": features.get("source_league"),
                    "target_league": features.get("target_league"),
                    "scenario": scenario,
                    "source_competition": features.get("source_competition"),
                    "target_competition": features.get("target_competition"),
                }
                # A supported historical regression at any applicable level
                # is enough to trigger fallback; unsupported cross-context
                # combinations also fall back rather than extrapolate blindly.
                keys = _row_policy_keys(row)
                maps = policy.get("group_dimensions", {})
                matching = [
                    (dimension, maps.get(dimension, {}).get(keys[dimension]))
                    for dimension in _POLICY_DIMENSION_ORDER
                ]
                matching = [
                    (dimension, item) for dimension, item in matching if item is not None
                ]
                failed = next(
                    ((dimension, item) for dimension, item in matching
                     if not bool(item.get("use_model"))),
                    None,
                )
                if scenario == SCENARIO_UNKNOWN:
                    fallback_reason = "unknown_scenario"
                elif failed is not None:
                    fallback_reason = f"validation_gate_failed:{failed[0]}"
                elif not matching and (
                    _clean(features.get("source_league")).upper() != league
                    or normalize_competition(features.get("source_competition"))
                    != normalize_competition(features.get("target_competition"))
                ):
                    fallback_reason = "cross_context_not_yet_validated"
                if fallback_reason:
                    score = baseline
                    prediction_method = "context_baseline_fallback"
        if prediction_method == "context_baseline_fallback":
            calibrated_level = "context_baseline_fallback"
            calibration_samples = 0
        else:
            calibrated_level = calibrated["calibration_level"]
            calibration_samples = int(calibrated["calibration_samples"])
        baseline_score = _context_baseline(features, means) if means else None
        return {
            "predicted_bb_rating": round(float(np.clip(score, 1, 100)), 1),
            "baseline_bb_rating": (
                round(float(np.clip(baseline_score, 1, 100)), 1)
                if baseline_score is not None else None
            ),
            "prediction_method": prediction_method,
            "fallback_reason": fallback_reason,
            "confidence_low_100": round(float(np.clip(score + interval["low"], 1, 100)), 1),
            "confidence_high_100": round(float(np.clip(score + interval["high"], 1, 100)), 1),
            "calibration_level": calibrated_level,
            "calibration_samples": calibration_samples,
            "interval_samples": int(interval["n"]),
            "scenario": scenario,
            "source_bb_rating": round(float(features["source_bb_rating"]), 1) if pd.notna(features.get("source_bb_rating")) else None,
            "bb_rating_version": BB_RATING_VERSION,
            "source_league": features.get("source_league"),
            "source_competition": features.get("source_competition"),
            "target_league": league,
            "target_competition": features.get("target_competition"),
            "target_season": features.get("target_season"),
            "model_version": MODEL_VERSION,
            "target_scale": "BB-Rating 1-100",
        }

    def predict_from_data(
        self, data: dict[str, Any], *, player_global_id: str, source_season: int,
        target_league: str, target_competition: str = "RS",
        target_team_global_id: str | None = None,
        source_league: str | None = None, source_competition: str = "RS",
    ) -> dict[str, Any]:
        stats = data.get("player_stats", pd.DataFrame())
        players = data.get("players", pd.DataFrame())
        engine = BBRatingEngine(stats, players, data.get("leagues"))
        frame = engine.frame.copy()
        if players is not None and not players.empty and "id" in players.columns:
            birth_columns = ["id"] + [
                column for column in ("birth_date", "date_of_birth")
                if column in players.columns and column not in frame.columns
            ]
            if len(birth_columns) > 1:
                birth = players[birth_columns].drop_duplicates("id").rename(columns={"id": "player_id"})
                frame = frame.merge(birth, on="player_id", how="left")
        frame["_year"] = _numeric_seasons(frame)
        frame["league_key"] = frame["league_key"].astype(str).str.strip().str.upper()
        frame["competition"] = frame["competition"].map(normalize_competition)
        frame = frame.drop_duplicates(
            subset=["player_global_id", "league_key", "_year", "competition"], keep="first"
        ).copy()
        source = frame[
            (frame["player_global_id"].astype(str) == str(player_global_id))
            & (frame["_year"] == int(source_season))
        ].copy()
        if source.empty:
            raise ValueError("Player has no source-season statistics")
        target_league = str(target_league).strip().upper()
        target_competition = normalize_competition(target_competition)
        source_competition = normalize_competition(source_competition)
        if source_competition:
            matching_phase = source[
                source["competition"].map(normalize_competition) == source_competition
            ]
            if not matching_phase.empty:
                source = matching_phase
            else:
                raise ValueError(
                    f"Player has no source-season rows for competition {source_competition}"
                )
        if source_league:
            source = source[
                source["league_key"].astype(str).str.upper() == str(source_league).strip().upper()
            ]
            if source.empty:
                raise ValueError(f"Player has no source-season rows for league {source_league}")
        # If the player has multiple simultaneous league contexts and the caller
        # did not specify one, select the most-used source context deterministically.
        for usage_column in ("games_played", "minutes_per_game", "minutes"):
            if usage_column in source.columns:
                source[usage_column] = pd.to_numeric(source[usage_column], errors="coerce")
        sort_columns = [column for column in ("games_played", "minutes_per_game", "minutes", "league_key") if column in source.columns]
        row = source.sort_values(sort_columns, ascending=[False] * (len(sort_columns) - 1) + [True] if sort_columns else [], na_position="last").iloc[0]
        team_map = _team_global_map(data)
        team_context_index = _team_context_index(data)
        source_team = _team_identity(row, team_map)
        source_league = _clean(row.get("league_key")).upper()
        source_competition = normalize_competition(row.get("competition"))
        try:
            source_result = engine.rate_player(
                str(player_global_id), league=source_league,
                season=int(source_season), phase=source_competition,
            )
            source_bb_rating = float(source_result.score)
        except ValueError:
            source_bb_rating = None
        features = _row_features(
            row, source_year=int(source_season), target_league=target_league,
            target_competition=target_competition, source_team=source_team,
            target_team=target_team_global_id, numeric_columns=self.numeric_columns,
            target_tier=_target_league_tier(data, target_league),
            source_bb_rating=source_bb_rating,
            team_context=_team_context_features(
                team_context_index, target_team_global_id, target_league, int(source_season)
            ),
        )
        features["target_season"] = int(source_season) + 1
        return self._predict_features(features)

    def save(self, directory: str | Path) -> dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Refusing to save an untrained BB Prediction Model")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "bb_rating_future_model.joblib")
        metadata = dict(self.metadata)
        metadata["artifact_file"] = "bb_rating_future_model.joblib"
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.calibrator is not None:
            (path / "calibration.json").write_text(
                json.dumps(self.calibrator.to_dict(), indent=2), encoding="utf-8"
            )
        return metadata

    @classmethod
    def load(cls, directory: str | Path) -> "BBRatingFuturePredictionModel":
        path = Path(directory)
        metadata_path = path / "metadata.json"
        artifact_path = path / "bb_rating_future_model.joblib"
        if not metadata_path.is_file() or not artifact_path.is_file():
            raise FileNotFoundError(f"Incomplete BB Prediction Model release: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(artifact_path)
        if not isinstance(model, cls) or not model.is_trained:
            raise ValueError("Invalid BB Prediction Model artifact")
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported BB Prediction Model feature version")
        if metadata.get("target") != "official_bb_rating_t_plus_1" or metadata.get("target_scale") != [1, 100]:
            raise ValueError("BB Prediction Model target contract is not official 1-100 BB-Rating")
        if metadata.get("model_version") != model.metadata.get("model_version"):
            raise ValueError("BB Prediction Model metadata/artifact version mismatch")
        if model.calibrator is None or not model.residual_intervals.get("global"):
            raise ValueError("BB Prediction Model calibration or uncertainty artifact is incomplete")
        return model

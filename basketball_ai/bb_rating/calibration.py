"""BB-Rating calibration and out-of-sample uncertainty validation.

This module does not change production weights, peer thresholds, model artifacts,
PostgreSQL source data, or the Prediction Model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from basketball_ai.bb_rating.engine import (
    AGE_BANDS,
    BAND_LABELS,
    BB_RATING_METRICS,
    BB_RATING_VERSION,
    BBRatingEngine,
    MIN_CONTEXT_SAMPLES,
    MIN_PEER_SAMPLES,
    RATE_COLUMNS,
    RatingMetricSpec,
)
from basketball_ai.bb_rating.semantics import METRIC_SEMANTICS

CALIBRATION_VERSION = "1.15"


@dataclass(frozen=True)
class BBRatingCalibrationConfig:
    min_peer_samples: int = MIN_PEER_SAMPLES
    min_context_samples: int = MIN_CONTEXT_SAMPLES
    uncertainty_min_samples: int = 50


def _normalise_series(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Canonical AI_Source stores unavailable NET_RTG_DIFF as 0 to preserve
    # the Prediction Model's numeric contract. BB-Rating treats that value as
    # missing because the audited authoritative On/Off source has no genuine
    # zero differential in 2018-2025. OnNetRtg/OffNetRtg themselves may
    # legitimately be exactly zero and must not be masked.
    if column == "net_rtg_diff":
        values = values.mask(values.abs() < 1e-12)
    if column in RATE_COLUMNS:
        values = values.mask(values.abs() > 1.0, values / 100.0)
    return values.astype(float)


def _score_from_percentile(percentile: pd.Series) -> pd.Series:
    return (1.0 + 99.0 * percentile).round().clip(1, 100).astype("Int64")


def _summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0, "mean": None, "stddev": None, "min": None,
            "p10": None, "p25": None, "p50": None, "p75": None,
            "p90": None, "max": None,
        }
    q = clean.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "stddev": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p10": float(q.loc[0.10]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "max": float(clean.max()),
    }


def _assign_peer_groups(
    frame: pd.DataFrame,
    *,
    min_peer_samples: int,
    min_context_samples: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["_peer_source"] = "limited_context"
    result["_peer_key"] = pd.NA
    result["_peer_sample_size"] = 0

    context_cols = ["league_key", "season", "competition"]
    grouped = result.groupby(context_cols, dropna=False, sort=False).groups

    for context_key, labels in grouped.items():
        ctx = result.loc[list(labels)]
        if ctx.empty:
            continue

        league, season, competition = context_key
        context_count = len(ctx)
        if context_count >= min_context_samples:
            source = "league+season+phase"
            size = context_count
            peer_key = (
                f"{league}|{int(season)}|{competition}|league+season+phase"
            )
        else:
            source = "limited_context"
            size = context_count
            peer_key = (
                f"{league}|{int(season)}|{competition}|limited_context"
            )

        result.loc[ctx.index, "_peer_source"] = source
        result.loc[ctx.index, "_peer_sample_size"] = int(size)
        result.loc[ctx.index, "_peer_key"] = peer_key

    return result


def _metric_percentiles(
    frame: pd.DataFrame,
    spec: RatingMetricSpec,
) -> tuple[pd.Series, pd.Series]:
    values = _normalise_series(frame[spec.source_column], spec.source_column)
    if spec.key == "NET_RTG_DIFF":
        on = pd.to_numeric(frame.get("on_net_rtg"), errors="coerce")
        off = pd.to_numeric(frame.get("off_net_rtg"), errors="coerce")
        complete_onoff = on.notna() & off.notna()
        values = values.where(complete_onoff)
    valid = values.notna()
    valid_n = (
        values.where(valid)
        .groupby(frame["_peer_key"], dropna=False)
        .transform("count")
    )
    rank_average = values.groupby(
        frame["_peer_key"], dropna=False
    ).rank(method="average")

    # Exactly match BBRatingEngine._percentile:
    # (count below + count at-or-below) / (2*n).
    percentile = (rank_average - 0.5) / valid_n
    percentile = percentile.clip(0.0, 1.0).where(valid)
    if spec.direction == "lower_better":
        percentile = (1.0 - percentile).where(valid)
    return values, percentile


def _registry_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BB_RATING_METRICS:
        semantic = METRIC_SEMANTICS.get(spec.key)
        rows.append(
            {
                "metric": spec.key,
                "source_column": spec.source_column,
                "column_exists": spec.source_column in frame.columns,
                "dimension": spec.dimension,
                "weight": float(spec.weight),
                "direction": spec.direction,
                "semantic_exists": semantic is not None,
                "semantic_source_matches": bool(
                    semantic is not None
                    and semantic.source_column == spec.source_column
                ),
                "rating_enabled": bool(spec.weight > 0.0),
            }
        )
    return rows


def _metric_summary(
    frame: pd.DataFrame,
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in registry:
        key = row["metric"]
        result = dict(row)
        if not row["column_exists"]:
            result.update(
                {
                    "n_rows": int(len(frame)),
                    "n_available": 0,
                    "missing_pct": 1.0,
                    "zero_pct_all_rows": None,
                    "zero_pct_available": None,
                    "percentile_summary": _summary(pd.Series(dtype=float)),
                    "effective_weight_mean": None,
                }
            )
            output.append(result)
            continue

        values = frame[f"_value_{key}"]
        percentiles = frame[f"_pct_{key}"]
        available = values.notna()
        n_available = int(available.sum())
        effective = None
        if row["rating_enabled"]:
            effective_values = frame[f"_effective_weight_{key}"]
            if effective_values.notna().any():
                effective = float(effective_values.dropna().mean())

        result.update(
            {
                "n_rows": int(len(frame)),
                "n_available": n_available,
                "missing_pct": (
                    float(1.0 - n_available / len(frame)) if len(frame) else 1.0
                ),
                "zero_pct_all_rows": (
                    float(values.eq(0).mean()) if len(frame) else None
                ),
                "zero_pct_available": (
                    float(values.loc[available].eq(0).mean())
                    if n_available else None
                ),
                "percentile_summary": _summary(percentiles),
                "effective_weight_mean": effective,
            }
        )
        output.append(result)
    return output



def _exposure_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    def numeric_summary(column: str) -> dict[str, Any]:
        if column not in frame.columns:
            return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
        return {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "p10": float(values.quantile(0.10)),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
        }

    return {
        "games_played": numeric_summary("games_played"),
        "minutes_per_game": numeric_summary("minutes_per_game"),
    }



def _score_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result = _summary(frame["_bb_rating"])
    if not result["n"]:
        result.update(
            {
                "share_1_20": None,
                "share_21_40": None,
                "share_41_60": None,
                "share_61_80": None,
                "share_81_100": None,
                "share_floor_1": None,
                "share_ceiling_100": None,
            }
        )
        return result

    values = pd.to_numeric(frame["_bb_rating"], errors="coerce").dropna()
    result.update(
        {
            "share_1_20": float((values <= 20).mean()),
            "share_21_40": float(((values >= 21) & (values <= 40)).mean()),
            "share_41_60": float(((values >= 41) & (values <= 60)).mean()),
            "share_61_80": float(((values >= 61) & (values <= 80)).mean()),
            "share_81_100": float((values >= 81).mean()),
            "share_floor_1": float((values == 1).mean()),
            "share_ceiling_100": float((values == 100).mean()),
        }
    )
    return result


def _group_score_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = {
            column: (int(value) if column == "season" and pd.notna(value) else value)
            for column, value in zip(group_cols, keys)
        }
        row.update(_score_summary(group))
        row["n"] = int(len(group))
        rows.append(row)
    return rows


def _variation_summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean_abs_change": None,
            "p25_abs_change": None,
            "p50_abs_change": None,
            "p75_abs_change": None,
            "p90_abs_change": None,
            "share_abs_change_le_5": None,
            "share_abs_change_le_10": None,
        }
    return {
        "n": int(len(clean)),
        "mean_abs_change": float(clean.mean()),
        "p25_abs_change": float(clean.quantile(0.25)),
        "p50_abs_change": float(clean.quantile(0.50)),
        "p75_abs_change": float(clean.quantile(0.75)),
        "p90_abs_change": float(clean.quantile(0.90)),
        "share_abs_change_le_5": float((clean <= 5).mean()),
        "share_abs_change_le_10": float((clean <= 10).mean()),
    }




def _consecutive_rating_change_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one row per player/league/competition t -> t+1 transition."""
    ordered = frame.sort_values(
        ["player_global_id", "league_key", "competition", "season"]
    ).copy()
    group_cols = ["player_global_id", "league_key", "competition"]

    ordered["_next_season"] = ordered.groupby(group_cols)["season"].shift(-1)
    ordered["_next_score"] = ordered.groupby(group_cols)["_bb_rating"].shift(-1)

    if "minutes_total" in ordered.columns:
        exposure = pd.to_numeric(ordered["minutes_total"], errors="coerce")
        if exposure.notna().sum() < 3:
            exposure = pd.Series(np.nan, index=ordered.index, dtype=float)
    else:
        exposure = pd.Series(np.nan, index=ordered.index, dtype=float)

    if exposure.isna().all():
        games = (
            pd.to_numeric(ordered["games_played"], errors="coerce")
            if "games_played" in ordered.columns
            else pd.Series(np.nan, index=ordered.index, dtype=float)
        )
        mpg = (
            pd.to_numeric(ordered["minutes_per_game"], errors="coerce")
            if "minutes_per_game" in ordered.columns
            else pd.Series(np.nan, index=ordered.index, dtype=float)
        )
        exposure = games * mpg

    ordered["_exposure_minutes"] = exposure

    pairs = ordered.loc[
        ordered["_next_season"].eq(ordered["season"] + 1)
        & ordered["_bb_rating"].notna()
        & ordered["_next_score"].notna()
        & ordered["_exposure_minutes"].notna()
    ].copy()
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "player_global_id", "league_key", "competition", "season",
                "target_season", "current_rating", "next_rating",
                "abs_rating_change", "exposure_minutes",
            ]
        )

    pairs["target_season"] = pd.to_numeric(
        pairs["_next_season"], errors="coerce"
    ).astype("Int64")
    pairs["current_rating"] = pd.to_numeric(
        pairs["_bb_rating"], errors="coerce"
    )
    pairs["next_rating"] = pd.to_numeric(
        pairs["_next_score"], errors="coerce"
    )
    pairs["abs_rating_change"] = (
        pairs["next_rating"] - pairs["current_rating"]
    ).abs()
    pairs["exposure_minutes"] = pd.to_numeric(
        pairs["_exposure_minutes"], errors="coerce"
    )
    return pairs[
        [
            "player_global_id", "league_key", "competition", "season",
            "target_season", "current_rating", "next_rating",
            "abs_rating_change", "exposure_minutes",
        ]
    ].dropna(
        subset=[
            "target_season", "current_rating", "next_rating",
            "abs_rating_change", "exposure_minutes", "league_key",
        ]
    )


def _quantile_edges(values: pd.Series) -> np.ndarray | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 8 or clean.nunique() < 4:
        return None
    return np.asarray(
        np.quantile(clean.to_numpy(dtype=float), [0.25, 0.50, 0.75]),
        dtype=float,
    )


def _assign_exposure_band(
    values: pd.Series,
    edges: np.ndarray | None,
) -> pd.Series:
    if edges is None:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna()
    if valid.any():
        codes = np.searchsorted(
            edges,
            numeric.loc[valid].to_numpy(dtype=float),
            side="right",
        ) + 1
        result.loc[valid] = pd.Series(
            np.clip(codes, 1, 4),
            index=numeric.loc[valid].index,
            dtype="Int64",
        )
    return result


def _fit_exposure_edges(
    training: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    global_edges = _quantile_edges(training["exposure_minutes"])
    league_edges: dict[str, np.ndarray] = {}
    for league, group in training.groupby("league_key", sort=True):
        edges = _quantile_edges(group["exposure_minutes"])
        if edges is not None:
            league_edges[str(league)] = edges
    return global_edges, league_edges


def _quantile_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    min_samples: int,
) -> dict[tuple[str, ...], dict[str, Any]]:
    tables: dict[tuple[str, ...], dict[str, Any]] = {}
    if frame.empty:
        return tables
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        clean = pd.to_numeric(group["abs_rating_change"], errors="coerce").dropna()
        if len(clean) < min_samples:
            continue
        tables[tuple(str(v) for v in keys)] = {
            "n": int(len(clean)),
            "p50": float(clean.quantile(0.50)),
            "p75": float(clean.quantile(0.75)),
            "p90": float(clean.quantile(0.90)),
        }
    return tables


def _global_quantile_table(
    frame: pd.DataFrame,
    *,
    min_samples: int,
) -> dict[str, Any] | None:
    clean = pd.to_numeric(frame["abs_rating_change"], errors="coerce").dropna()
    if len(clean) < min_samples:
        return None
    return {
        "n": int(len(clean)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
    }


def _predict_league_exposure(
    row: pd.Series,
    *,
    global_table: dict[str, Any] | None,
    league_tables: dict[tuple[str, ...], dict[str, Any]],
    exposure_tables: dict[tuple[str, ...], dict[str, Any]],
    league_exposure_tables: dict[tuple[str, ...], dict[str, Any]],
    global_band: Any,
    league_band: Any,
) -> tuple[dict[str, Any] | None, str]:
    league = str(row["league_key"])
    lb = int(league_band) if pd.notna(league_band) else None
    gb = int(global_band) if pd.notna(global_band) else None

    if lb is not None:
        table = league_exposure_tables.get((league, str(lb)))
        if table is not None:
            return table, "league+exposure"

    table = league_tables.get((league,))
    if table is not None:
        return table, "league"

    if gb is not None:
        table = exposure_tables.get((str(gb),))
        if table is not None:
            return table, "exposure"

    if global_table is not None:
        return global_table, "global"

    return None, "unavailable"


def _aggregate_uncertainty_results(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "model": model_name,
            "n_oos": 0,
            "coverage": {"p50": None, "p75": None, "p90": None},
            "coverage_error": {"p50": None, "p75": None, "p90": None},
            "mean_interval_width_p90": None,
            "median_interval_width_p90": None,
            "folds": [],
        }

    observed = pd.DataFrame(rows)
    actual = pd.to_numeric(observed["abs_rating_change"], errors="coerce")
    coverage: dict[str, float] = {}
    errors: dict[str, float] = {}
    targets = {"p50": 0.50, "p75": 0.75, "p90": 0.90}

    for quantile, target in targets.items():
        threshold = pd.to_numeric(observed[quantile], errors="coerce")
        observed_coverage = float(actual.le(threshold).mean())
        coverage[quantile] = observed_coverage
        errors[quantile] = observed_coverage - target

    widths = 2.0 * pd.to_numeric(observed["p90"], errors="coerce")
    by_fold = []
    for fold, group in observed.groupby("target_season", sort=True):
        actual_fold = pd.to_numeric(group["abs_rating_change"], errors="coerce")
        fold_row = {
            "target_season": int(fold),
            "n_oos": int(len(group)),
            "coverage": {},
        }
        for quantile, target in targets.items():
            threshold = pd.to_numeric(group[quantile], errors="coerce")
            observed_coverage = float(actual_fold.le(threshold).mean())
            fold_row["coverage"][quantile] = observed_coverage
            fold_row[f"{quantile}_error"] = observed_coverage - target
        by_fold.append(fold_row)

    return {
        "model": model_name,
        "n_oos": int(len(observed)),
        "coverage": coverage,
        "coverage_error": errors,
        "mean_interval_width_p90": float(widths.mean()),
        "median_interval_width_p90": float(widths.median()),
        "folds": by_fold,
    }


def _oos_uncertainty_validation(
    frame: pd.DataFrame,
    *,
    min_samples: int,
) -> dict[str, Any]:
    """Validate empirical uncertainty using only earlier target seasons."""
    pairs = _consecutive_rating_change_pairs(frame)
    if pairs.empty:
        return {
            "status": "insufficient_data",
            "method": "expanding_walk_forward_empirical_quantiles",
            "target": "absolute next-season BB-Rating change",
            "quantiles": [0.50, 0.75, 0.90],
            "min_samples": int(min_samples),
            "first_possible_test_season": None,
            "test_seasons": [],
            "folds": [],
            "models": [],
            "selected_structure": {
                "features": ["league", "exposure_band"],
                "fallback_order": [
                    "league+exposure", "league", "exposure", "global"
                ],
                "primary_share": 0.0,
                "fallback_shares": {},
                "coverage": {"p50": None, "p75": None, "p90": None},
                "coverage_error": {"p50": None, "p75": None, "p90": None},
                "mean_interval_width_p90": None,
                "median_interval_width_p90": None,
            },
        }

    target_seasons = sorted(
        int(value) for value in pd.to_numeric(
            pairs["target_season"], errors="coerce"
        ).dropna().unique()
    )
    first_possible = target_seasons[1] if len(target_seasons) >= 2 else None

    model_rows = {name: [] for name in (
        "global", "league", "exposure", "league+exposure"
    )}
    fold_metadata: list[dict[str, Any]] = []

    for target_season in target_seasons:
        training = pairs.loc[pairs["target_season"] < target_season].copy()
        test = pairs.loc[pairs["target_season"] == target_season].copy()
        if training.empty or test.empty:
            continue

        training_seasons = sorted(
            int(value) for value in pd.to_numeric(
                training["target_season"], errors="coerce"
            ).dropna().unique()
        )
        global_edges, league_edges = _fit_exposure_edges(training)

        training["global_band"] = _assign_exposure_band(
            training["exposure_minutes"], global_edges
        )
        training["league_band"] = pd.Series(
            pd.NA, index=training.index, dtype="Int64"
        )
        for league, indices in training.groupby("league_key", sort=False).groups.items():
            training.loc[indices, "league_band"] = _assign_exposure_band(
                training.loc[indices, "exposure_minutes"],
                league_edges.get(str(league)),
            )

        test["global_band"] = _assign_exposure_band(
            test["exposure_minutes"], global_edges
        )
        test["league_band"] = pd.Series(
            pd.NA, index=test.index, dtype="Int64"
        )
        for league, indices in test.groupby("league_key", sort=False).groups.items():
            test.loc[indices, "league_band"] = _assign_exposure_band(
                test.loc[indices, "exposure_minutes"],
                league_edges.get(str(league)),
            )

        global_table = _global_quantile_table(training, min_samples=min_samples)
        league_tables = _quantile_table(
            training, ["league_key"], min_samples=min_samples
        )
        exposure_tables = _quantile_table(
            training.dropna(subset=["global_band"]),
            ["global_band"], min_samples=min_samples
        )
        league_exposure_tables = _quantile_table(
            training.dropna(subset=["league_band"]),
            ["league_key", "league_band"], min_samples=min_samples
        )

        fold_usage = {name: {} for name in model_rows}
        for _, row in test.iterrows():
            predictions = {
                "global": (
                    global_table,
                    "global",
                ),
                "league": (
                    league_tables.get((str(row["league_key"]),))
                    or global_table,
                    "league" if (str(row["league_key"]),) in league_tables else "global",
                ),
                "exposure": (
                    exposure_tables.get(
                        (str(int(row["global_band"])),)
                    ) if pd.notna(row["global_band"]) else global_table,
                    "exposure" if pd.notna(row["global_band"]) and
                    (str(int(row["global_band"])),) in exposure_tables else "global",
                ),
                "league+exposure": _predict_league_exposure(
                    row,
                    global_table=global_table,
                    league_tables=league_tables,
                    exposure_tables=exposure_tables,
                    league_exposure_tables=league_exposure_tables,
                    global_band=row["global_band"],
                    league_band=row["league_band"],
                ),
            }

            for model_name, (table, source) in predictions.items():
                if table is None:
                    continue
                model_rows[model_name].append(
                    {
                        "target_season": int(target_season),
                        "league_key": str(row["league_key"]),
                        "abs_rating_change": float(row["abs_rating_change"]),
                        "p50": float(table["p50"]),
                        "p75": float(table["p75"]),
                        "p90": float(table["p90"]),
                        "source": source,
                    }
                )
                fold_usage[model_name][source] = (
                    fold_usage[model_name].get(source, 0) + 1
                )

        fold_metadata.append(
            {
                "target_season": int(target_season),
                "training_target_seasons": training_seasons,
                "training_rows": int(len(training)),
                "test_rows": int(len(test)),
                "usable_global_table": global_table is not None,
                "usage_by_model": fold_usage,
            }
        )

    models = [
        _aggregate_uncertainty_results(model_rows[name], model_name=name)
        for name in ("global", "league", "exposure", "league+exposure")
    ]
    selected = next(
        model for model in models if model["model"] == "league+exposure"
    )

    source_counts: dict[str, int] = {}
    for row in model_rows["league+exposure"]:
        source = str(row["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    total_selected = len(model_rows["league+exposure"])

    return {
        "status": (
            "validated_oos"
            if any(model["n_oos"] > 0 for model in models)
            else "insufficient_data"
        ),
        "method": "expanding_walk_forward_empirical_quantiles",
        "target": (
            "absolute next-season BB-Rating change; quantiles for each OOS "
            "season are fitted only from earlier target seasons"
        ),
        "quantiles": [0.50, 0.75, 0.90],
        "min_samples": int(min_samples),
        "first_possible_test_season": first_possible,
        "test_seasons": [int(meta["target_season"]) for meta in fold_metadata],
        "folds": fold_metadata,
        "models": models,
        "selected_structure": {
            "features": ["league", "exposure_band"],
            "fallback_order": [
                "league+exposure", "league", "exposure", "global"
            ],
            "primary_share": (
                float(source_counts.get("league+exposure", 0) / total_selected)
                if total_selected else 0.0
            ),
            "fallback_shares": {
                source: float(count / total_selected)
                for source, count in sorted(source_counts.items())
                if source != "league+exposure" and total_selected
            },
            "coverage": selected["coverage"],
            "coverage_error": selected["coverage_error"],
            "mean_interval_width_p90": selected["mean_interval_width_p90"],
            "median_interval_width_p90": selected["median_interval_width_p90"],
        },
        "interpretation": (
            "Coverage is evaluated out-of-sample by applying empirical P50/P75/P90 "
            "thresholds fitted only on earlier target seasons. The public "
            "BB-Rating value is unchanged."
        ),
    }


def _diagnostic_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    contexts = report["contexts"]
    if contexts["share_below_min_peer_samples"] > 0.50:
        warnings.append(
            "Oltre il 50% dei contesti ha meno del numero minimo di peer; "
            "la qualità del confronto può essere frequentemente inferiore a medium."
        )

    for metric in report["metrics"]:
        if metric["rating_enabled"] and metric["missing_pct"] > 0.30:
            warnings.append(
                f"{metric['metric']}: oltre il 30% delle righe non ha un valore "
                "utilizzabile; la copertura va verificata prima di fissare il peso."
            )
        if (
            metric["rating_enabled"]
            and metric["n_available"]
            and metric["zero_pct_available"] is not None
            and metric["zero_pct_available"] >= 0.99
        ):
            warnings.append(
                f"{metric['metric']}: almeno il 99% dei valori disponibili è esattamente "
                "zero; verificare se il campo è un placeholder di copertura anziché "
                "un segnale discriminante."
            )
        percentile_summary = metric["percentile_summary"]
        if (
            metric["rating_enabled"]
            and percentile_summary["n"] >= 50
            and percentile_summary["stddev"] is not None
            and percentile_summary["stddev"] < 0.02
        ):
            warnings.append(
                f"{metric['metric']}: la distribuzione dei percentili è quasi costante; "
                "il contributo al BB-Rating è attualmente poco discriminante."
            )

    net_diff = next((m for m in report["metrics"] if m["metric"] == "NET_RTG_DIFF"), None)
    if net_diff and net_diff["rating_enabled"] and net_diff["zero_pct_available"] is not None:
        if net_diff["zero_pct_available"] >= 0.30:
            warnings.append(
                "NET_RTG_DIFF: oltre il 30% dei valori disponibili è esattamente zero; "
                "verificare la provenienza On/Off prima di consolidare il peso."
            )

    score = report["score_distribution"]
    if score["n"]:
        if score["stddev"] is not None and score["stddev"] < 8.0:
            warnings.append(
                "La deviazione standard del BB-Rating complessivo è inferiore "
                "a 8 punti: verificare una possibile compressione."
            )
        if score["share_ceiling_100"] > 0.05:
            warnings.append(
                "Più del 5% dei rating è esattamente 100: verificare una "
                "possibile saturazione sul soffitto."
            )
        if score["share_floor_1"] > 0.05:
            warnings.append(
                "Più del 5% dei rating è esattamente 1: verificare una "
                "possibile saturazione sul pavimento."
            )
    return warnings


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BB-Rating Calibration Report",
        "",
        f"- Calibration version: {report['calibration_version']}",
        f"- BB-Rating version tested: {report['bb_rating_version']}",
        f"- Rows: **{report['dataset']['rows']}**",
        f"- Players: **{report['dataset']['players']}**",
        f"- Seasons: **{report['dataset']['season_min']} -> {report['dataset']['season_max']}**",
        "",
        "## Scope",
        "",
        "Calibration and temporal uncertainty validation only. No production "
        "weights, peer thresholds, model artifacts, PostgreSQL source data, "
        "or Prediction Model logic is changed.",
        "",
        "## Registry audit",
        "",
        "| Metric | Column | Exists | Semantic | Weight | Direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["registry_audit"]:
        semantic_ok = row["semantic_exists"] and row["semantic_source_matches"]
        lines.append(
            f"| {row['metric']} | {row['source_column']} | "
            f"{'yes' if row['column_exists'] else 'NO'} | "
            f"{'yes' if semantic_ok else 'NO'} | {row['weight']:.4f} | "
            f"{row['direction']} |"
        )

    lines += [
        "",
        "## Context and peer populations",
        "",
        f"- Contexts: **{report['contexts']['n_contexts']}**",
        f"- Median context size: **{report['contexts']['median_context_size']:.1f}**",
        f"- Contexts below minimum peer size: **{report['contexts']['n_below_min_peer_samples']}** "
        f"({report['contexts']['share_below_min_peer_samples']:.1%})",
        f"- Primary league+season+phase share: **{report['validation_signals']['primary_context_share']:.1%}**",
        f"- Peer fallback share: **{report['validation_signals']['peer_fallback_share']:.1%}**",
        "",
        "| Peer source | Rows | Contexts | Median peer size |",
        "|---|---:|---:|---:|",
    ]
    for row in report["peer_sources"]:
        lines.append(
            f"| {row['peer_source']} | {row['rows']} ({row['share_rows']:.1%}) | "
            f"{row['contexts']} ({row['share_contexts']:.1%}) | "
            f"{row['median_peer_sample_size']:.1f} |"
        )

    lines += [
        "",
        "## Exposure distribution",
        "",
        f"- Games played P10/P50/P90: **{report['exposure']['games_played']['p10']} / "
        f"{report['exposure']['games_played']['p50']} / "
        f"{report['exposure']['games_played']['p90']}**",
        f"- Minutes per game P10/P50/P90: **{report['exposure']['minutes_per_game']['p10']} / "
        f"{report['exposure']['minutes_per_game']['p50']} / "
        f"{report['exposure']['minutes_per_game']['p90']}**",
        "",
        "## Metric coverage and distributions",
        "",
        "| Metric | Weight | Available | Missing | Zero among available | P50 percentile | Effective weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        summary = row["percentile_summary"]
        p50 = summary["p50"]
        effective = row["effective_weight_mean"]
        lines.append(
            f"| {row['metric']} | {row['weight']:.4f} | "
            f"{row['n_available']}/{row['n_rows']} | {row['missing_pct']:.1%} | "
            f"{'—' if row['zero_pct_available'] is None else f'{row['zero_pct_available']:.1%}'} | "
            f"{'—' if p50 is None else f'{p50:.3f}'} | "
            f"{'—' if effective is None else f'{effective:.4f}'} |"
        )

    score = report["score_distribution"]
    uncertainty = report["uncertainty_validation"]
    lines += [
        "",
        "## BB-Rating distribution",
        "",
        f"- N: **{score['n']}**",
        f"- Mean: **{score['mean']:.2f}**",
        f"- Stddev: **{score['stddev']:.2f}**",
        f"- P10/P50/P90: **{score['p10']:.1f} / {score['p50']:.1f} / {score['p90']:.1f}**",
        f"- Share 81–100: **{score['share_81_100']:.1%}**",
        f"- Share 1–20: **{score['share_1_20']:.1%}**",
        "",
        "## Temporal uncertainty validation",
        "",
        f"- Method: **{uncertainty['method']}**",
        f"- First possible OOS target season: **{uncertainty['first_possible_test_season'] or '—'}**",
        f"- OOS target seasons: **{', '.join(str(v) for v in uncertainty['test_seasons']) if uncertainty['test_seasons'] else '—'}**",
        f"- Minimum samples per empirical cell: **{uncertainty['min_samples']}**",
        "",
        "| Model | OOS N | P50 coverage | P75 coverage | P90 coverage | P90 mean interval width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in uncertainty["models"]:
        lines.append(
            f"| {model['model']} | {model['n_oos']} | "
            f"{'—' if model['coverage']['p50'] is None else f'{model['coverage']['p50']:.1%}'} | "
            f"{'—' if model['coverage']['p75'] is None else f'{model['coverage']['p75']:.1%}'} | "
            f"{'—' if model['coverage']['p90'] is None else f'{model['coverage']['p90']:.1%}'} | "
            f"{'—' if model['mean_interval_width_p90'] is None else f'{model['mean_interval_width_p90']:.2f}'} |"
        )

    selected = uncertainty["selected_structure"]
    lines += [
        "",
        "### League + exposure OOS structure",
        "",
        f"- Primary exact league+exposure share: **{selected['primary_share']:.1%}**",
        f"- P50/P75/P90 coverage: **{selected['coverage']['p50']:.1%} / "
        f"{selected['coverage']['p75']:.1%} / {selected['coverage']['p90']:.1%}**"
        if selected["coverage"]["p50"] is not None else
        "- P50/P75/P90 coverage: **—**",
        f"- P50/P75/P90 coverage error: **{selected['coverage_error']['p50']:+.1%} / "
        f"{selected['coverage_error']['p75']:+.1%} / {selected['coverage_error']['p90']:+.1%}**"
        if selected["coverage_error"]["p50"] is not None else
        "- P50/P75/P90 coverage error: **—**",
        f"- Mean P90 interval width: **{selected['mean_interval_width_p90']:.2f}**"
        if selected["mean_interval_width_p90"] is not None else
        "- Mean P90 interval width: **—**",
        "",
        "Fallback order: league+exposure → league → exposure → global.",
        "",
        "No uncertainty correction is applied to the public BB-Rating.",
        "",
        "## Validation signals",
        "",
        f"- Registry columns OK: **{report['validation_signals']['registry_columns_ok']}**",
        f"- Scoring registry OK: **{report['validation_signals']['scoring_registry_ok']}**",
        f"- Explanation catalog OK: **{report['validation_signals']['explanation_catalog_ok']}**",
        f"- Score available rate: **{report['validation_signals']['score_available_rate']:.1%}**",
        f"- Primary context share: **{report['validation_signals']['primary_context_share']:.1%}**",
        f"- Limited-context share: **{report['validation_signals']['limited_context_share']:.1%}**",
        "",
        "## Diagnostic warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No heuristic diagnostic warnings triggered.")
    return "\n".join(lines) + "\n"


def build_calibration_report(
    data: dict[str, Any],
    *,
    config: BBRatingCalibrationConfig | None = None,
) -> dict[str, Any]:
    config = config or BBRatingCalibrationConfig()
    min_peer_samples = max(5, int(config.min_peer_samples))
    min_context_samples = max(1, int(config.min_context_samples))

    engine = BBRatingEngine(
        data.get("player_stats", pd.DataFrame()),
        data.get("players", pd.DataFrame()),
        min_peer_samples=min_peer_samples,
    )
    frame = engine.frame.copy()
    if frame.empty:
        raise ValueError("BB-Rating calibration requires a non-empty player_stats frame")

    frame = _assign_peer_groups(
        frame,
        min_peer_samples=min_peer_samples,
        min_context_samples=min_context_samples,
    )

    registry_audit = _registry_audit(frame)
    scoring_specs = [spec for spec in BB_RATING_METRICS if spec.weight > 0]

    metric_columns: dict[str, pd.Series] = {}
    for spec in BB_RATING_METRICS:
        if spec.source_column not in frame.columns:
            metric_columns[f"_value_{spec.key}"] = pd.Series(
                np.nan, index=frame.index, dtype=float
            )
            metric_columns[f"_pct_{spec.key}"] = pd.Series(
                np.nan, index=frame.index, dtype=float
            )
            continue

        values, percentiles = _metric_percentiles(frame, spec)
        metric_columns[f"_value_{spec.key}"] = values
        metric_columns[f"_pct_{spec.key}"] = percentiles

    if metric_columns:
        frame = pd.concat(
            [frame, pd.DataFrame(metric_columns, index=frame.index)],
            axis=1,
        )

    used_weight = pd.Series(0.0, index=frame.index, dtype=float)
    numerator = pd.Series(0.0, index=frame.index, dtype=float)

    for spec in scoring_specs:
        percentiles = frame[f"_pct_{spec.key}"]
        available = percentiles.notna()
        weight = float(spec.weight)
        used_weight = used_weight + available.astype(float) * weight
        numerator = numerator + percentiles.fillna(0.0) * weight

    frame["_composite_pct"] = (
        numerator / used_weight.replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    frame["_bb_rating"] = _score_from_percentile(frame["_composite_pct"])
    total_scoring_weight = float(sum(spec.weight for spec in scoring_specs))
    frame["_metric_coverage"] = (
        (used_weight / total_scoring_weight).clip(0.0, 1.0)
        if total_scoring_weight > 0.0
        else 0.0
    )

    effective_columns: dict[str, pd.Series] = {}
    for spec in scoring_specs:
        available = frame[f"_pct_{spec.key}"].notna() & used_weight.gt(0)
        effective = pd.Series(np.nan, index=frame.index, dtype=float)
        effective.loc[available] = float(spec.weight) / used_weight.loc[available]
        effective_columns[f"_effective_weight_{spec.key}"] = effective

    if effective_columns:
        frame = pd.concat(
            [frame, pd.DataFrame(effective_columns, index=frame.index)],
            axis=1,
        )

    metrics = _metric_summary(frame, registry_audit)

    context_sizes = (
        frame.groupby(
            ["league_key", "season", "competition"], dropna=False
        ).size().rename("context_sample_size").reset_index()
    )
    below = int(
        (context_sizes["context_sample_size"] < min_peer_samples).sum()
    ) if not context_sizes.empty else 0

    peer_sources: list[dict[str, Any]] = []
    total_rows = len(frame)
    total_contexts = len(context_sizes)
    for source, group in frame.groupby("_peer_source", dropna=False, sort=True):
        source_contexts = int(
            group[["league_key", "season", "competition"]]
            .drop_duplicates().shape[0]
        )
        peer_sources.append(
            {
                "peer_source": str(source),
                "rows": int(len(group)),
                "share_rows": float(len(group) / total_rows) if total_rows else 0.0,
                "contexts": source_contexts,
                "share_contexts": (
                    float(source_contexts / total_contexts)
                    if total_contexts else 0.0
                ),
                "median_peer_sample_size": float(group["_peer_sample_size"].median()),
                "mean_peer_sample_size": float(group["_peer_sample_size"].mean()),
                "min_peer_sample_size": int(group["_peer_sample_size"].min()),
                "max_peer_sample_size": int(group["_peer_sample_size"].max()),
            }
        )

    seasons = pd.to_numeric(frame["season"], errors="coerce").dropna()
    valid_ids = frame["player_global_id"].replace({"nan": np.nan}).dropna()

    report: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "bb_rating_version": BB_RATING_VERSION,
        "dataset": {
            "rows": int(len(frame)),
            "players": int(valid_ids.nunique()),
            "leagues": sorted(str(v) for v in frame["league_key"].dropna().unique()),
            "seasons": sorted(int(v) for v in seasons.unique()),
            "season_min": int(seasons.min()),
            "season_max": int(seasons.max()),
            "competitions": sorted(str(v) for v in frame["competition"].dropna().unique()),
            "source_contract": data.get("source_contract"),
        },
        "configuration": {
            "min_peer_samples": min_peer_samples,
            "min_context_samples": min_context_samples,
            "uncertainty_min_samples": max(25, int(config.uncertainty_min_samples)),
            "age_bands": [list(v) for v in AGE_BANDS],
        },
        "registry_audit": registry_audit,
        "contexts": {
            "n_contexts": int(len(context_sizes)),
            "median_context_size": (
                float(context_sizes["context_sample_size"].median())
                if not context_sizes.empty else 0.0
            ),
            "mean_context_size": (
                float(context_sizes["context_sample_size"].mean())
                if not context_sizes.empty else 0.0
            ),
            "n_below_min_peer_samples": below,
            "share_below_min_peer_samples": (
                float(below / len(context_sizes)) if len(context_sizes) else 0.0
            ),
        },
        "peer_sources": peer_sources,
        "exposure": _exposure_diagnostics(frame),
        "metrics": metrics,
        "score_distribution": _score_summary(frame),
        "by_league": _group_score_summary(frame, ["league_key"]),
        "by_season": _group_score_summary(frame, ["season"]),
        "by_competition": _group_score_summary(frame, ["competition"]),
        "by_league_season": _group_score_summary(frame, ["league_key", "season"]),
        "uncertainty_validation": _oos_uncertainty_validation(
            frame,
            min_samples=max(25, int(config.uncertainty_min_samples)),
        ),
    }

    report["warnings"] = _diagnostic_warnings(report)
    scoring_registry_ok = all(
        row["column_exists"]
        and row["semantic_exists"]
        and row["semantic_source_matches"]
        for row in registry_audit
        if row["rating_enabled"]
    )
    explanation_catalog_ok = all(
        row["semantic_exists"] and row["semantic_source_matches"]
        for row in registry_audit
    )
    source_by_name = {
        row["peer_source"]: row["share_rows"] for row in peer_sources
    }
    constant_scoring_metrics = [
        row["metric"]
        for row in metrics
        if row["rating_enabled"]
        and row["n_available"]
        and row["zero_pct_available"] is not None
        and row["zero_pct_available"] >= 0.99
    ]
    primary_context_share = (
        source_by_name.get("league+season+phase", 0.0)
        if len(frame) else 0.0
    )
    role_peer_share = 0.0
    report["validation_signals"] = {
        "registry_columns_ok": scoring_registry_ok,
        "scoring_registry_ok": scoring_registry_ok,
        "explanation_catalog_ok": explanation_catalog_ok,
        "scoring_metrics_present": all(
            row["column_exists"]
            for row in registry_audit
            if row["rating_enabled"]
        ),
        "score_available_rate": (
            float(frame["_bb_rating"].notna().mean()) if len(frame) else 0.0
        ),
        "primary_context_share": primary_context_share,
        "peer_fallback_share": (
            float(1.0 - primary_context_share)
            if len(frame) else 1.0
        ),
        "peer_source_shares": source_by_name,
        "role_peer_share": role_peer_share,
        "limited_context_share": source_by_name.get("limited_context", 0.0),
        "constant_scoring_metrics": constant_scoring_metrics,
        "explainable_metric_count": int(
            sum(row["semantic_exists"] for row in registry_audit)
        ),
    }
    return report


def write_calibration_report(
    report: dict[str, Any],
    out_dir: str,
) -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bb_rating_calibration.json"
    md_path = root / "bb_rating_calibration.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

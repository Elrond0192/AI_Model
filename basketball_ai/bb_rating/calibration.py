"""BB-Rating calibration and real-data validation.

Diagnostic only: this module never changes production weights, peer thresholds,
model artifacts, PostgreSQL source data, or the Prediction Model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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

CALIBRATION_VERSION = "1.10"


@dataclass(frozen=True)
class BBRatingCalibrationConfig:
    min_peer_samples: int = MIN_PEER_SAMPLES
    min_context_samples: int = MIN_CONTEXT_SAMPLES


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



def _role_peer_population_diagnostics(
    frame: pd.DataFrame,
    *,
    min_peer_samples: int,
) -> dict[str, Any]:
    valid = frame.loc[
        frame["age_band"].astype(str).str.strip().ne("")
        & frame["ruolo_combinato"].astype(str).str.strip().ne("")
    ].copy()
    group_cols = [
        "league_key",
        "season",
        "competition",
        "position_family",
        "age_band",
        "ruolo_combinato",
    ]
    if valid.empty:
        return {
            "rows_with_role_and_age": 0,
            "role_present_share": 0.0,
            "distinct_role_groups": 0,
            "group_size": {
                "mean": None,
                "median": None,
                "p90": None,
                "max": None,
            },
            "thresholds": {},
            "role_only_groups": 0,
            "role_only_group_size": {
                "mean": None,
                "median": None,
                "p90": None,
                "max": None,
            },
            "role_only_thresholds": {},
            "configured_min_peer_samples": int(min_peer_samples),
        }

    counts = valid.groupby(group_cols, dropna=False).size().rename("n")
    role_only = valid.loc[
        valid["ruolo_combinato"].astype(str).str.strip().ne("")
    ].groupby(
        ["league_key", "season", "competition", "position_family", "ruolo_combinato"],
        dropna=False,
    ).size().rename("n")
    thresholds = {}
    role_only_thresholds = {}
    for threshold in (5, 10, 15, 20, 25):
        eligible = counts.ge(threshold)
        thresholds[str(threshold)] = {
            "groups": int(eligible.sum()),
            "share_groups": float(eligible.mean()),
            "rows_eligible": int(counts.loc[eligible].sum()) if eligible.any() else 0,
            "share_rows": (
                float(counts.loc[eligible].sum() / len(valid))
                if len(valid) else 0.0
            ),
        }
        role_eligible = role_only.ge(threshold)
        role_only_thresholds[str(threshold)] = {
            "groups": int(role_eligible.sum()),
            "share_groups": (
                float(role_eligible.mean()) if len(role_only) else 0.0
            ),
            "rows_eligible": (
                int(role_only.loc[role_eligible].sum())
                if role_eligible.any() else 0
            ),
            "share_rows": (
                float(role_only.loc[role_eligible].sum() / len(valid))
                if len(valid) and role_eligible.any() else 0.0
            ),
        }

    return {
        "rows_with_role_and_age": int(len(valid)),
        "role_present_share": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "distinct_role_groups": int(len(counts)),
        "group_size": {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "p90": float(counts.quantile(0.90)),
            "max": int(counts.max()),
        },
        "thresholds": thresholds,
        "role_only_groups": int(len(role_only)),
        "role_only_group_size": {
            "mean": float(role_only.mean()) if len(role_only) else None,
            "median": float(role_only.median()) if len(role_only) else None,
            "p90": float(role_only.quantile(0.90)) if len(role_only) else None,
            "max": int(role_only.max()) if len(role_only) else None,
        },
        "role_only_thresholds": role_only_thresholds,
        "configured_min_peer_samples": int(min_peer_samples),
    }


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


def _stability(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["player_global_id", "league_key", "competition", "season"]
    ).copy()
    group_cols = ["player_global_id", "league_key", "competition"]
    ordered["_next_season"] = ordered.groupby(group_cols)["season"].shift(-1)
    ordered["_next_score"] = ordered.groupby(group_cols)["_bb_rating"].shift(-1)

    pairs = ordered.loc[
        ordered["_next_season"].eq(ordered["season"] + 1)
        & ordered["_bb_rating"].notna()
        & ordered["_next_score"].notna()
    ]
    if pairs.empty:
        return {
            "n_pairs": 0,
            "mean_abs_change": None,
            "median_abs_change": None,
            "share_abs_change_le_5": None,
            "share_abs_change_le_10": None,
            "score_correlation": None,
        }

    changes = (pairs["_next_score"] - pairs["_bb_rating"]).abs()
    correlation = None
    if (
        len(pairs) >= 3
        and pairs["_bb_rating"].nunique() >= 2
        and pairs["_next_score"].nunique() >= 2
    ):
        correlation = float(
            pairs[["_bb_rating", "_next_score"]].corr().iloc[0, 1]
        )

    return {
        "n_pairs": int(len(pairs)),
        "mean_abs_change": float(changes.mean()),
        "median_abs_change": float(changes.median()),
        "share_abs_change_le_5": float((changes <= 5).mean()),
        "share_abs_change_le_10": float((changes <= 10).mean()),
        "score_correlation": correlation,
    }




def _pair_correlation(
    current: pd.Series,
    following: pd.Series,
) -> dict[str, Any]:
    pair = pd.DataFrame({"current": current, "following": following}).dropna()
    if len(pair) < 3:
        return {
            "n_pairs": int(len(pair)),
            "pearson": None,
            "spearman": None,
        }

    pearson = None
    spearman = None
    if pair["current"].nunique() >= 2 and pair["following"].nunique() >= 2:
        pearson = float(pair["current"].corr(pair["following"]))
        spearman = float(
            pair["current"].rank(method="average").corr(
                pair["following"].rank(method="average")
            )
        )
    return {
        "n_pairs": int(len(pair)),
        "pearson": pearson,
        "spearman": spearman,
    }


def _stability_group_summary(
    pairs: pd.DataFrame,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pairs.empty:
        return rows

    for key, group in pairs.groupby(group_columns, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        clean = group[["current_score", "next_score"]].dropna()
        changes = (
            (clean["next_score"] - clean["current_score"]).abs()
            if not clean.empty
            else pd.Series(dtype=float)
        )
        corr = _pair_correlation(
            clean["current_score"] if not clean.empty else pd.Series(dtype=float),
            clean["next_score"] if not clean.empty else pd.Series(dtype=float),
        )
        row: dict[str, Any] = {
            column: (
                int(value)
                if column == "season" and pd.notna(value)
                else value
            )
            for column, value in zip(group_columns, keys)
        }
        row.update(
            {
                "n_pairs": int(len(clean)),
                "mean_abs_change": float(changes.mean()) if not changes.empty else None,
                "median_abs_change": float(changes.median()) if not changes.empty else None,
                "share_abs_change_le_5": (
                    float((changes <= 5).mean()) if not changes.empty else None
                ),
                "share_abs_change_le_10": (
                    float((changes <= 10).mean()) if not changes.empty else None
                ),
                "pearson": corr["pearson"],
                "spearman": corr["spearman"],
            }
        )
        rows.append(row)
    return rows


def _stability_diagnostics(
    frame: pd.DataFrame,
    scoring_specs: list[RatingMetricSpec],
) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["player_global_id", "league_key", "competition", "season"]
    ).copy()
    group_cols = ["player_global_id", "league_key", "competition"]

    ordered["_next_season"] = ordered.groupby(group_cols)["season"].shift(-1)
    ordered["_next_score"] = ordered.groupby(group_cols)["_bb_rating"].shift(-1)
    ordered["_next_composite_pct"] = ordered.groupby(group_cols)["_composite_pct"].shift(-1)

    exposure_source = None
    if "minutes_total" in ordered.columns:
        exposure = pd.to_numeric(ordered["minutes_total"], errors="coerce")
        if exposure.notna().sum() >= 3:
            exposure_source = "minutes_total"
    if exposure_source is None:
        games = (
            pd.to_numeric(ordered["games_played"], errors="coerce")
            if "games_played" in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        mpg = (
            pd.to_numeric(ordered["minutes_per_game"], errors="coerce")
            if "minutes_per_game" in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        exposure = games * mpg
        exposure_source = "games_played_x_minutes_per_game"
    ordered["_exposure_minutes"] = exposure
    ordered["_next_exposure_minutes"] = ordered.groupby(group_cols)["_exposure_minutes"].shift(-1)

    valid_pairs = ordered.loc[
        ordered["_next_season"].eq(ordered["season"] + 1)
        & ordered["_bb_rating"].notna()
        & ordered["_next_score"].notna()
        & ordered["_composite_pct"].notna()
        & ordered["_next_composite_pct"].notna()
    ].copy()

    base = _pair_correlation(
        valid_pairs["_bb_rating"],
        valid_pairs["_next_score"],
    )
    composite = _pair_correlation(
        valid_pairs["_composite_pct"],
        valid_pairs["_next_composite_pct"],
    )

    score_changes = (
        (valid_pairs["_next_score"] - valid_pairs["_bb_rating"]).abs()
        if not valid_pairs.empty
        else pd.Series(dtype=float)
    )
    composite_changes = (
        (valid_pairs["_next_composite_pct"] - valid_pairs["_composite_pct"]).abs()
        if not valid_pairs.empty
        else pd.Series(dtype=float)
    )

    exposure_source_values = valid_pairs["_exposure_minutes"].dropna()
    exposure_summary = {
        "source": exposure_source,
        "n": int(len(exposure_source_values)),
        "p10": float(exposure_source_values.quantile(0.10)) if not exposure_source_values.empty else None,
        "p25": float(exposure_source_values.quantile(0.25)) if not exposure_source_values.empty else None,
        "p50": float(exposure_source_values.quantile(0.50)) if not exposure_source_values.empty else None,
        "p75": float(exposure_source_values.quantile(0.75)) if not exposure_source_values.empty else None,
        "p90": float(exposure_source_values.quantile(0.90)) if not exposure_source_values.empty else None,
        "max": float(exposure_source_values.max()) if not exposure_source_values.empty else None,
    }

    exposure_pairs = valid_pairs.loc[valid_pairs["_exposure_minutes"].notna()].copy()
    exposure_groups: list[dict[str, Any]] = []
    if len(exposure_pairs) >= 12:
        ranks = exposure_pairs["_exposure_minutes"].rank(method="first")
        exposure_pairs["_exposure_quartile"] = pd.qcut(
            ranks,
            4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        )
        exposure_groups = _stability_group_summary(
            exposure_pairs.rename(
                columns={
                    "_bb_rating": "current_score",
                    "_next_score": "next_score",
                    "_exposure_quartile": "exposure_group",
                }
            ),
            ["exposure_group"],
        )
        exposure_medians = exposure_pairs.groupby(
            "_exposure_quartile", observed=True
        )["_exposure_minutes"].median()
        for row in exposure_groups:
            row["median_exposure"] = float(
                exposure_medians.loc[row["exposure_group"]]
            )
    else:
        exposure_groups = []

    by_league = _stability_group_summary(
        valid_pairs.rename(
            columns={"_bb_rating": "current_score", "_next_score": "next_score"}
        ),
        ["league_key"],
    )

    by_position = _stability_group_summary(
        valid_pairs.rename(
            columns={"_bb_rating": "current_score", "_next_score": "next_score"}
        ),
        ["position_family"],
    )

    by_age_band = _stability_group_summary(
        valid_pairs.rename(
            columns={"_bb_rating": "current_score", "_next_score": "next_score"}
        ),
        ["age_band"],
    )

    metric_stability: list[dict[str, Any]] = []
    for spec in scoring_specs:
        current_col = f"_pct_{spec.key}"
        if current_col not in ordered.columns:
            continue

        next_percentiles = ordered.groupby(group_cols)[current_col].shift(-1)
        pair_mask = (
            ordered["_next_season"].eq(ordered["season"] + 1)
            & ordered[current_col].notna()
            & next_percentiles.notna()
        )
        current_values = ordered.loc[pair_mask, current_col]
        next_values = next_percentiles.loc[pair_mask]
        corr = _pair_correlation(current_values, next_values)
        changes = (
            (next_values - current_values).abs()
            if not current_values.empty
            else pd.Series(dtype=float)
        )
        coverage_n = int(ordered[current_col].notna().sum())
        metric_stability.append(
            {
                "metric": spec.key,
                "n_pairs": int(len(current_values)),
                "mean_abs_percentile_change": (
                    float(changes.mean()) if not changes.empty else None
                ),
                "median_abs_percentile_change": (
                    float(changes.median()) if not changes.empty else None
                ),
                "pearson": corr["pearson"],
                "spearman": corr["spearman"],
                "coverage": (
                    float(coverage_n / len(ordered)) if len(ordered) else 0.0
                ),
                "weight": float(spec.weight),
            }
        )

    return {
        "n_pairs": int(len(valid_pairs)),
        "score": {
            "pearson": base["pearson"],
            "spearman": base["spearman"],
        },
        "composite_percentile": {
            "pearson": composite["pearson"],
            "spearman": composite["spearman"],
            "mean_abs_change": (
                float(composite_changes.mean()) if not composite_changes.empty else None
            ),
            "median_abs_change": (
                float(composite_changes.median()) if not composite_changes.empty else None
            ),
        },
        "exposure": {
            "summary": exposure_summary,
            "quartiles": exposure_groups,
        },
        "by_league": by_league,
        "by_position": by_position,
        "by_age_band": by_age_band,
        "by_metric": metric_stability,
        "reliability": _empirical_reliability_diagnostics(
            {
                "score": {
                    "spearman": base["spearman"],
                },
                "exposure": {
                    "quartiles": exposure_groups,
                },
            },
            metric_stability,
        ),
    }




def _empirical_reliability_diagnostics(
    stability: dict[str, Any],
    metric_stability: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute empirical persistence proxies without changing production scores.

    Exposure reliability is fitted from observed consecutive-season Spearman
    persistence by exposure quartile. Metric reliability uses each active metric's
    consecutive-season Spearman persistence. The output is diagnostic only and
    is intentionally not applied as score shrinkage or weight adjustment.
    """
    active_metrics = [
        row for row in metric_stability
        if row["n_pairs"] >= 3 and row["spearman"] is not None
    ]
    weighted_parts = []
    for row in active_metrics:
        reliability = max(0.0, min(1.0, float(row["spearman"])))
        coverage = max(0.0, min(1.0, float(row["coverage"])))
        weight = max(0.0, float(row["weight"]))
        weighted_parts.append((weight, coverage, reliability))

    denominator = sum(weight * coverage for weight, coverage, _ in weighted_parts)
    weighted_metric_reliability = (
        sum(weight * coverage * reliability for weight, coverage, reliability in weighted_parts)
        / denominator
        if denominator > 0
        else None
    )

    exposure_points = []
    for row in stability.get("exposure", {}).get("quartiles", []):
        spearman = row.get("spearman")
        median_exposure = row.get("median_exposure")
        if spearman is None or median_exposure is None:
            continue
        if not np.isfinite(float(spearman)) or not np.isfinite(float(median_exposure)):
            continue
        exposure_points.append(
            {
                "exposure_group": row["exposure_group"],
                "median_exposure": float(median_exposure),
                "observed_spearman": max(0.0, min(1.0, float(spearman))),
                "n_pairs": int(row["n_pairs"]),
            }
        )

    curve = {
        "method": "saturating_exposure_curve",
        "formula": "reliability_proxy = asymptote * exposure / (exposure + half_exposure)",
        "fitted": False,
        "asymptote": None,
        "half_exposure": None,
        "rmse": None,
        "r2": None,
    }
    fitted_by_group: dict[str, float] = {}

    if len(exposure_points) >= 3:
        x = np.array([point["median_exposure"] for point in exposure_points], dtype=float)
        y = np.array([point["observed_spearman"] for point in exposure_points], dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            asymptote, half_exposure = params
            return asymptote * x / (x + half_exposure) - y

        max_y = float(np.max(y))
        median_x = float(np.median(x))
        initial_asymptote = min(0.99, max(0.5, max_y + 0.05))
        initial_half_exposure = max(
            1.0,
            median_x * max(initial_asymptote / max(max_y, 1e-6) - 1.0, 0.25),
        )
        fitted = least_squares(
            residuals,
            x0=np.array([initial_asymptote, initial_half_exposure], dtype=float),
            bounds=(
                np.array([0.0, 1e-6], dtype=float),
                np.array([1.0, np.inf], dtype=float),
            ),
        )
        asymptote, half_exposure = (float(value) for value in fitted.x)
        predicted = asymptote * x / (x + half_exposure)
        rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
        ss_res = float(np.sum((predicted - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

        curve.update(
            {
                "fitted": bool(fitted.success),
                "asymptote": asymptote,
                "half_exposure": half_exposure,
                "rmse": rmse,
                "r2": r2,
            }
        )
        for point, prediction in zip(exposure_points, predicted):
            fitted_by_group[point["exposure_group"]] = float(
                max(0.0, min(1.0, prediction))
            )

    max_observed = max(
        (point["observed_spearman"] for point in exposure_points),
        default=None,
    )
    for point in exposure_points:
        point["fitted_reliability_proxy"] = fitted_by_group.get(point["exposure_group"])
        point["relative_to_high_exposure"] = (
            float(point["observed_spearman"] / max_observed)
            if max_observed and max_observed > 0
            else None
        )

    return {
        "status": "diagnostic_only",
        "meaning": (
            "Test-retest persistence proxies only. No reliability correction is "
            "applied to BB-Rating v1.8."
        ),
        "observed_composite_spearman": stability.get("score", {}).get("spearman"),
        "weighted_active_metric_spearman": weighted_metric_reliability,
        "active_metric_count": int(len(active_metrics)),
        "exposure_curve": {
            **curve,
            "points": exposure_points,
        },
        "interpretation": {
            "low_exposure_has_lower_persistence": bool(
                len(exposure_points) >= 2
                and exposure_points[0]["observed_spearman"]
                < exposure_points[-1]["observed_spearman"]
            ),
            "automatic_application": False,
        },
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
        "Diagnostic only. No production weights, peer thresholds, model artifacts, "
        "PostgreSQL source data, or Prediction Model logic is changed.",
        "",
        "## BB-Rating data source",
        "",
        "BB-Rating reads the canonical observed contract. On/Off fields are "
        "resolved upstream by ai_source_full.sql from the Analisi On/Off tables "
        "when the primary observation is NULL/zero and an authoritative source "
        "value is available.",
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

    role_diag = report["role_peer_population"]
    lines += [
        "",
        "### Role peer population diagnostics",
        "",
        f"- Rows with role + age: **{role_diag['rows_with_role_and_age']}** "
        f"({role_diag['role_present_share']:.1%})",
        f"- Distinct position+age+role groups: **{role_diag['distinct_role_groups']}**",
        f"- Median role-group size: **{role_diag['group_size']['median']:.1f}**",
        f"- P90 role-group size: **{role_diag['group_size']['p90']:.1f}**",
        f"- Position+role groups (age ignored): **{role_diag['role_only_groups']}**",
        f"- Median position+role group size: **{role_diag['role_only_group_size']['median']:.1f}**"
        if role_diag["role_only_group_size"]["median"] is not None else
        "- Median position+role group size: **—**",
        "",
        "| Minimum age+role group size | Groups | Rows eligible |",
        "|---:|---:|---:|",
    ]
    for threshold, row in role_diag["thresholds"].items():
        lines.append(
            f"| {threshold} | {row['groups']} ({row['share_groups']:.1%}) | "
            f"{row['rows_eligible']} ({row['share_rows']:.1%}) |"
        )
    lines += [
        "",
        "| Minimum position+role group size | Groups | Rows eligible |",
        "|---:|---:|---:|",
    ]
    for threshold, row in role_diag["role_only_thresholds"].items():
        lines.append(
            f"| {threshold} | {row['groups']} ({row['share_groups']:.1%}) | "
            f"{row['rows_eligible']} ({row['share_rows']:.1%}) |"
        )

    lines += [
        "",
        "### Exposure diagnostics",
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
        "## Stability",
        "",
        f"- Consecutive player-season pairs: **{report['stability']['n_pairs']}**",
        f"- Pearson correlation (1–100): **{report['stability_diagnostics']['score']['pearson'] if report['stability_diagnostics']['score']['pearson'] is not None else '—'}**",
        f"- Spearman correlation (1–100): **{report['stability_diagnostics']['score']['spearman'] if report['stability_diagnostics']['score']['spearman'] is not None else '—'}**",
        f"- Pearson correlation (continuous composite): **{report['stability_diagnostics']['composite_percentile']['pearson'] if report['stability_diagnostics']['composite_percentile']['pearson'] is not None else '—'}**",
        f"- Spearman correlation (continuous composite): **{report['stability_diagnostics']['composite_percentile']['spearman'] if report['stability_diagnostics']['composite_percentile']['spearman'] is not None else '—'}**",
        "",
        "### Exposure stability",
        "",
        f"- Exposure source: **{report['stability_diagnostics']['exposure']['summary']['source']}**",
        "",
        "| Exposure group | N | Mean abs change | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['exposure_group']} | {row['n_pairs']} | "
            f"{'—' if row['mean_abs_change'] is None else f'{row['mean_abs_change']:.3f}'} | "
            f"{'—' if row['pearson'] is None else f'{row['pearson']:.3f}'} | "
            f"{'—' if row['spearman'] is None else f'{row['spearman']:.3f}'} |"
            for row in report['stability_diagnostics']['exposure']['quartiles']
        ],
        "",
        "### Stability by league",
        "",
        "| League | N | Mean abs change | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['league_key']} | {row['n_pairs']} | "
            f"{'—' if row['mean_abs_change'] is None else f'{row['mean_abs_change']:.3f}'} | "
            f"{'—' if row['pearson'] is None else f'{row['pearson']:.3f}'} | "
            f"{'—' if row['spearman'] is None else f'{row['spearman']:.3f}'} |"
            for row in report['stability_diagnostics']['by_league']
        ],
        "",
        "### Stability by metric",
        "",
        "| Metric | N | Mean abs percentile change | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['metric']} | {row['n_pairs']} | "
            f"{'—' if row['mean_abs_percentile_change'] is None else f'{row['mean_abs_percentile_change']:.4f}'} | "
            f"{'—' if row['pearson'] is None else f'{row['pearson']:.3f}'} | "
            f"{'—' if row['spearman'] is None else f'{row['spearman']:.3f}'} |"
            for row in report['stability_diagnostics']['by_metric']
        ],
        "",
        f"- Consecutive player-season pairs: **{report['stability']['n_pairs']}**",
        f"- Mean absolute change: **{report['stability']['mean_abs_change'] if report['stability']['mean_abs_change'] is not None else '—'}**",
        f"- Median absolute change: **{report['stability']['median_abs_change'] if report['stability']['median_abs_change'] is not None else '—'}**",
        f"- Correlation: **{report['stability']['score_correlation'] if report['stability']['score_correlation'] is not None else '—'}**",
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
        "role_peer_population": _role_peer_population_diagnostics(
            frame,
            min_peer_samples=min_peer_samples,
        ),
        "exposure": _exposure_diagnostics(frame),
        "metrics": metrics,
        "score_distribution": _score_summary(frame),
        "by_league": _group_score_summary(frame, ["league_key"]),
        "by_season": _group_score_summary(frame, ["season"]),
        "by_competition": _group_score_summary(frame, ["competition"]),
        "by_league_season": _group_score_summary(frame, ["league_key", "season"]),
        "stability": _stability(frame),
        "stability_diagnostics": _stability_diagnostics(
            frame,
            scoring_specs,
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
    role_peer_share = (
        source_by_name.get("position+age+role", 0.0)
        + source_by_name.get("position+role", 0.0)
    ) if len(frame) else 0.0
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

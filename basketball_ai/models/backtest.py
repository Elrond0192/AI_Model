"""Expanding walk-forward backtest for the exact production ensemble."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from basketball_ai.models.strict_production import (
    AsOfPositionPerformanceModel,
    StrictProductionEnsembleModel,
    build_historical_snapshot,
    evaluate_target_season,
    metric_summary,
    season_year,
)

_logger = logging.getLogger(__name__)


class _AbsoluteTargetPerformanceModel(AsOfPositionPerformanceModel):
    """Diagnostic-only variant that learns absolute t+1 ratings."""

    target_mode = "rating"


class _AbsoluteTargetEnsemble(StrictProductionEnsembleModel):
    """Diagnostic-only production path with the pre-#60 absolute target."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(performance_model=_AbsoluteTargetPerformanceModel(), **kwargs)




def _rmse(records: Iterable[Dict[str, Any]], key: str) -> float:
    rows = list(records)
    if not rows:
        return float("nan")
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    prediction = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def _diagnostic_stage_rmse(records: List[Dict[str, Any]]) -> Dict[str, float]:
    stages = {
        "raw_xgb_rmse": "raw_xgb_prediction",
        "base_before_age_rmse": "base_before_age_prediction",
        "base_after_age_rmse": "base_after_age_prediction",
        "after_compatibility_rmse": "after_compatibility_prediction",
        "after_league_rmse": "after_league_prediction",
        "after_context_rmse": "after_context_prediction",
        "final_ensemble_rmse": "prediction",
    }
    return {name: _rmse(records, key) for name, key in stages.items()}



def _persistence_reconciliation_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify that production shrinkage can be reconstructed exactly from diagnostics."""
    rows = [
        row for row in records
        if row.get("persistence_prediction") is not None
        and row.get("pre_shrinkage_prediction") is not None
        and row.get("persistence_shrinkage_alpha") is not None
        and row.get("prediction") is not None
    ]
    if not rows:
        return {"n": 0, "valid": False, "reason": "missing shrinkage diagnostics"}

    persistence = np.asarray([float(row["persistence_prediction"]) for row in rows], dtype=float)
    raw = np.asarray([float(row["pre_shrinkage_prediction"]) for row in rows], dtype=float)
    alpha = np.asarray([float(row["persistence_shrinkage_alpha"]) for row in rows], dtype=float)
    actual = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    reconstructed = np.clip(persistence + alpha * (raw - persistence), 3.5, 10.0)
    absolute_diff = np.abs(actual - reconstructed)

    return {
        "n": int(len(rows)),
        "valid": bool(np.all(np.isfinite(absolute_diff))),
        "max_abs_diff": float(np.max(absolute_diff)),
        "mean_abs_diff": float(np.mean(absolute_diff)),
        "rmse_diff": float(np.sqrt(np.mean(absolute_diff ** 2))),
        "exact_matches": int(np.sum(absolute_diff <= 1e-12)),
        "mismatches": int(np.sum(absolute_diff > 1e-12)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
    }

def _segment_metrics(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        groups.setdefault(str(value), []).append(row)
    return {name: metric_summary(rows) for name, rows in groups.items()}


def _compatibility_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe the learned compatibility correction without changing inference."""
    rows = [
        row for row in records
        if row.get("after_compatibility_prediction") is not None
        and row.get("base_after_age_prediction") is not None
        and row.get("actual") is not None
        and row.get("compatibility_factor") is not None
    ]
    if not rows:
        return {"n": 0}

    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    base = np.asarray(
        [float(row["base_after_age_prediction"]) for row in rows], dtype=float
    )
    after = np.asarray(
        [float(row["after_compatibility_prediction"]) for row in rows], dtype=float
    )
    score = np.asarray(
        [float(row["compatibility_factor"]) for row in rows], dtype=float
    )
    # Compatibility is currently a contextual signal only; it is not added to the prediction.
    adjustment = after - base
    residual_before = actual - base

    def _corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
        if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    def _stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    base_metrics = metric_summary(
        [{"actual": float(a), "prediction": float(p)} for a, p in zip(actual, base)]
    )
    after_metrics = metric_summary(
        [{"actual": float(a), "prediction": float(p)} for a, p in zip(actual, after)]
    )

    return {
        "n": int(len(rows)),
        "compatibility_score": _stats(score),
        "compatibility_adjustment": _stats(adjustment),
        "base_before_compatibility": base_metrics,
        "after_compatibility": after_metrics,
        "rmse_delta_after_minus_before": (
            after_metrics["rmse"] - base_metrics["rmse"]
        ),
        "correlation": {
            "score_vs_actual_minus_base": _corr(score, residual_before),
            "adjustment_vs_actual_minus_base": _corr(adjustment, residual_before),
            "score_vs_base_minus_actual": _corr(score, -residual_before),
        },
    }


def _component_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Isolate the rating-point contribution of each post-base component."""
    required = (
        "actual", "base_after_age_prediction", "after_league_prediction",
        "after_context_prediction", "prediction", "league_factor",
        "context_multiplier", "mpg_factor",
    )
    rows = [row for row in records if all(row.get(key) is not None for key in required)]
    if not rows:
        return {"n": 0}

    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    base = np.asarray([float(row["base_after_age_prediction"]) for row in rows], dtype=float)
    after_league = np.asarray([float(row["after_league_prediction"]) for row in rows], dtype=float)
    after_context = np.asarray([float(row["after_context_prediction"]) for row in rows], dtype=float)
    final = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    league_factor = np.asarray([float(row["league_factor"]) for row in rows], dtype=float)
    context_multiplier = np.asarray([float(row["context_multiplier"]) for row in rows], dtype=float)
    mpg_factor = np.asarray([float(row["mpg_factor"]) for row in rows], dtype=float)

    def corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
        if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    def metrics(prediction: np.ndarray) -> Dict[str, float]:
        error = prediction - actual
        return {
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }

    league_only = np.clip(base * league_factor, 3.5, 10.0)
    context_only = np.clip(base * context_multiplier, 3.5, 10.0)
    mpg_only = np.clip(base * mpg_factor, 3.5, 10.0)

    components = {
        "league": {
            "factor": {
                "mean": float(np.mean(league_factor)),
                "std": float(np.std(league_factor)),
                "min": float(np.min(league_factor)),
                "max": float(np.max(league_factor)),
            },
            "standalone_metrics": metrics(league_only),
            "sequential_metrics": metrics(after_league),
            "correction_mean": float(np.mean(after_league - base)),
            "correction_std": float(np.std(after_league - base)),
            "correction_residual_correlation": corr(after_league - base, actual - base),
        },
        "context": {
            "multiplier": {
                "mean": float(np.mean(context_multiplier)),
                "std": float(np.std(context_multiplier)),
                "min": float(np.min(context_multiplier)),
                "max": float(np.max(context_multiplier)),
            },
            "standalone_metrics": metrics(context_only),
            "sequential_metrics": metrics(after_context),
            "correction_mean": float(np.mean(after_context - after_league)),
            "correction_std": float(np.std(after_context - after_league)),
            "correction_residual_correlation": corr(
                after_context - after_league, actual - after_league
            ),
        },
        "mpg": {
            "factor": {
                "mean": float(np.mean(mpg_factor)),
                "std": float(np.std(mpg_factor)),
                "min": float(np.min(mpg_factor)),
                "max": float(np.max(mpg_factor)),
            },
            "standalone_metrics": metrics(mpg_only),
            "sequential_metrics": metrics(final),
            "correction_mean": float(np.mean(final - after_context)),
            "correction_std": float(np.std(final - after_context)),
            "correction_residual_correlation": corr(
                final - after_context, actual - after_context
            ),
        },
    }

    return {
        "n": int(len(rows)),
        "base_metrics": metrics(base),
        "after_context_metrics": metrics(after_context),
        "final_metrics": metrics(final),
        "components": components,
        "final_gain_vs_after_context_rmse": (
            float(np.sqrt(np.mean((after_context - actual) ** 2)))
            - float(np.sqrt(np.mean((final - actual) ** 2)))
        ),
    }


def _persistence_blend_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """OOF-only diagnostic for blending the learned delta with persistence.

    This does not change production inference. It answers whether the current
    model contains useful information beyond persistence, and how much of it
    survives when the prediction is shrunk back toward the persistence prior.
    """
    rows = [
        row for row in records
        if row.get("actual") is not None
        and row.get("persistence_prediction") is not None
        and row.get("prediction") is not None
    ]
    if len(rows) < 2:
        return {"n": len(rows)}

    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    persistence = np.asarray([float(row["persistence_prediction"]) for row in rows], dtype=float)
    prediction = np.asarray([float(row.get("pre_shrinkage_prediction", row["prediction"])) for row in rows], dtype=float)
    delta = prediction - persistence
    residual = actual - persistence

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - actual) ** 2)))

    denominator = float(np.sum(delta * delta))
    alpha = float(np.sum(delta * residual) / denominator) if denominator > 0 else 0.0
    alpha_clipped = float(np.clip(alpha, 0.0, 1.0))
    blended = persistence + alpha_clipped * delta

    return {
        "n": int(len(rows)),
        "persistence_rmse": rmse(persistence),
        "model_rmse": rmse(prediction),
        "raw_model_delta_rmse_gain_vs_persistence": rmse(persistence) - rmse(prediction),
        "optimal_oof_alpha_unbounded": alpha,
        "optimal_oof_alpha_0_1": alpha_clipped,
        "blended_rmse_0_1": rmse(blended),
        "blended_gain_vs_persistence": rmse(persistence) - rmse(blended),
        "delta_std": float(np.std(delta)),
        "delta_residual_correlation": (
            float(np.corrcoef(delta, residual)[0, 1])
            if np.std(delta) > 0.0 and np.std(residual) > 0.0
            else None
        ),
    }


def _nested_temporal_shrinkage_alpha(
    data: Dict[str, Any],
    outer_source_season: int,
) -> Dict[str, Any]:
    """Estimate shrinkage using only seasons before the outer target."""
    inner_target = int(outer_source_season)
    inner_train_through = inner_target - 1
    try:
        inner_train_data = build_historical_snapshot(data, inner_train_through)
        inner_ensemble = StrictProductionEnsembleModel(enable_persistence_shrinkage=False)
        inner_ensemble.train(inner_train_data)
        inner_records = evaluate_target_season(inner_ensemble, data, inner_target)
        diagnostics = _persistence_blend_diagnostics(inner_records)
        alpha = float(np.clip(
            diagnostics.get("optimal_oof_alpha_0_1", 0.0), 0.0, 1.0
        ))
        return {
            "valid": bool(inner_records) and math.isfinite(alpha),
            "alpha": alpha,
            "inner_target_season": inner_target,
            "inner_train_through_season": inner_train_through,
            "inner_n": len(inner_records),
            "inner_persistence_rmse": diagnostics.get("persistence_rmse"),
            "inner_model_rmse": diagnostics.get("model_rmse"),
            "inner_blended_rmse": diagnostics.get("blended_rmse_0_1"),
            "inner_blended_gain_vs_persistence": diagnostics.get("blended_gain_vs_persistence"),
        }
    except Exception as exc:
        _logger.warning(
            "[Backtest] Nested shrinkage calibration failed for inner target=%s: %s",
            inner_target, exc,
        )
        return {
            "valid": False, "alpha": 0.0,
            "inner_target_season": inner_target,
            "inner_train_through_season": inner_train_through,
            "inner_n": 0, "error": str(exc),
        }


def _nested_role_shrinkage_alpha(
    data: Dict[str, Any],
    outer_source_season: int,
) -> Dict[str, Any]:
    """Estimate role-specific shrinkage using only seasons before the outer target.

    Diagnostic-only: this never changes the production prediction path. A role
    alpha is accepted only when the inner OOF sample has at least 30 rows;
    otherwise the inner global alpha is used as the role fallback.
    """
    inner_target = int(outer_source_season)
    inner_train_through = inner_target - 1
    try:
        inner_train_data = build_historical_snapshot(data, inner_train_through)
        inner_ensemble = StrictProductionEnsembleModel(enable_persistence_shrinkage=False)
        inner_ensemble.train(inner_train_data)
        inner_records = evaluate_target_season(inner_ensemble, data, inner_target)
        global_diag = _persistence_blend_diagnostics(inner_records)
        global_alpha = float(np.clip(
            global_diag.get("optimal_oof_alpha_0_1", 0.0), 0.0, 1.0
        ))
        by_role: Dict[str, Any] = {}
        for role in sorted({
            str(row.get("position"))
            for row in inner_records
            if row.get("position") is not None
        }):
            rows = [
                row for row in inner_records
                if str(row.get("position")) == role
                and row.get("actual") is not None
                and row.get("persistence_prediction") is not None
                and row.get("prediction") is not None
            ]
            if len(rows) < 30:
                by_role[role] = {
                    "n": len(rows),
                    "valid": False,
                    "alpha": global_alpha,
                    "fallback": "global",
                }
                continue
            actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
            persistence = np.asarray(
                [float(row["persistence_prediction"]) for row in rows], dtype=float
            )
            prediction = np.asarray(
                [float(row.get("pre_shrinkage_prediction", row["prediction"])) for row in rows], dtype=float
            )
            delta = prediction - persistence
            residual = actual - persistence
            denominator = float(np.sum(delta * delta))
            alpha = (
                float(np.sum(delta * residual) / denominator)
                if denominator > 0.0 else global_alpha
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            blended = persistence + alpha * delta
            persistence_rmse = float(np.sqrt(np.mean((persistence - actual) ** 2)))
            blended_rmse = float(np.sqrt(np.mean((blended - actual) ** 2)))
            by_role[role] = {
                "n": len(rows),
                "valid": True,
                "alpha": alpha,
                "fallback": None,
                "persistence_rmse": persistence_rmse,
                "model_rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
                "blended_rmse": blended_rmse,
                "gain_vs_persistence": persistence_rmse - blended_rmse,
            }
        return {
            "valid": bool(inner_records) and math.isfinite(global_alpha),
            "inner_target_season": inner_target,
            "inner_train_through_season": inner_train_through,
            "inner_n": len(inner_records),
            "global_alpha": global_alpha,
            "roles": by_role,
        }
    except Exception as exc:
        _logger.warning(
            "[Backtest] Role-conditioned nested shrinkage failed for inner target=%s: %s",
            inner_target, exc,
        )
        return {
            "valid": False,
            "inner_target_season": inner_target,
            "inner_train_through_season": inner_train_through,
            "inner_n": 0,
            "global_alpha": 0.0,
            "roles": {},
            "error": str(exc),
        }


def _role_residual_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure whether model-vs-persistence residuals are role-dependent."""
    rows = [
        row for row in records
        if row.get("position") is not None
        and row.get("actual") is not None
        and row.get("persistence_prediction") is not None
        and row.get("prediction") is not None
    ]
    result: Dict[str, Any] = {}
    for position in sorted({str(row["position"]) for row in rows}):
        group = [row for row in rows if str(row["position"]) == position]
        if len(group) < 20:
            continue
        actual = np.asarray([float(row["actual"]) for row in group], dtype=float)
        persistence = np.asarray([float(row["persistence_prediction"]) for row in group], dtype=float)
        prediction = np.asarray([float(row.get("pre_shrinkage_prediction", row["prediction"])) for row in group], dtype=float)
        result[position] = {
            "n": len(group),
            "persistence_rmse": float(np.sqrt(np.mean((persistence - actual) ** 2))),
            "model_rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
            "model_bias": float(np.mean(prediction - actual)),
            "persistence_bias": float(np.mean(persistence - actual)),
        }
    return result

def _age_band(age: int) -> str:
    if age <= 21:
        return "<=21"
    if age <= 24:
        return "22-24"
    if age <= 28:
        return "25-28"
    if age <= 32:
        return "29-32"
    return "33+"


def run_backtest(
    data: Dict[str, Any],
    n_folds: Optional[int] = None,
    output_path: Optional[str] = "models_saved/backtest_report.json",
    min_samples_per_fold: Optional[int] = None,
    include_stage_metrics: bool = False,
    compare_target_modes: bool = False,
) -> Dict[str, Any]:
    """Evaluate the complete production path on all eligible future seasons.

    By default every eligible target season is evaluated. Each fold
    trains/calibrates using only seasons up to ``target-1`` and then
    predicts exact same-league/same-competition outcomes in ``target``.  The
    final ensemble is compared with base XGBoost and persistence and is also
    reported independently for every observed competition.
    """
    stats = data.get("player_stats", pd.DataFrame())
    if stats.empty or "season" not in stats.columns:
        return {
            "error": "player_stats empty or missing season column",
            "valid": False,
            "folds": [],
        }
    if data.get("team_season_stats") is None or data["team_season_stats"].empty:
        return {
            "error": "team competition history is required",
            "valid": False,
            "folds": [],
        }

    seasons = sorted({season_year(value) for value in stats["season"].dropna()})
    if len(seasons) < 5:
        return {
            "error": "at least five seasons are required",
            "valid": False,
            "folds": [],
        }

    eligible_targets = seasons[4:]
    # By default evaluate every eligible historical target season. ``n_folds``
    # is an explicit cap for faster diagnostics, not the production default.
    # With 2018-2025 source history this evaluates 2022-2025, while each fold
    # trains only on seasons strictly before its target season.
    if n_folds is None:
        targets = eligible_targets
    else:
        requested_folds = max(1, int(n_folds))
        targets = eligible_targets[-min(requested_folds, len(eligible_targets)):]
    minimum = (
        int(os.getenv("BACKTEST_MIN_SAMPLES_PER_FOLD", "1"))
        if min_samples_per_fold is None
        else int(min_samples_per_fold)
    )

    folds: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []

    for index, target_season in enumerate(targets, 1):
        source_season = target_season - 1
        try:
            train_data = build_historical_snapshot(data, source_season)
            ensemble = StrictProductionEnsembleModel()
            train_metrics = ensemble.train(train_data)
            records = evaluate_target_season(ensemble, data, target_season)
            if len(records) < minimum:
                raise RuntimeError(
                    f"only {len(records)} evaluable samples; minimum is {minimum}"
                )

            nested_shrinkage = None
            nested_summary = None
            nested_role_shrinkage = None
            nested_role_summary = None
            if include_stage_metrics:
                nested_shrinkage = _nested_temporal_shrinkage_alpha(data, source_season)
                nested_alpha = float(nested_shrinkage.get("alpha", 0.0))
                for row in records:
                    persistence = float(row["persistence_prediction"])
                    model_prediction = float(row.get("pre_shrinkage_prediction", row["prediction"]))
                    row["nested_shrinkage_prediction"] = float(np.clip(
                        persistence + nested_alpha * (model_prediction - persistence),
                        3.5, 10.0,
                    ))
                nested_summary = metric_summary([
                    {"actual": row["actual"], "prediction": row["nested_shrinkage_prediction"]}
                    for row in records
                ])

                nested_role_shrinkage = _nested_role_shrinkage_alpha(data, source_season)
                global_role_alpha = float(
                    nested_role_shrinkage.get("global_alpha", nested_alpha)
                )
                role_alphas = {
                    str(role): float(info.get("alpha", global_role_alpha))
                    for role, info in nested_role_shrinkage.get("roles", {}).items()
                }
                for row in records:
                    persistence = float(row["persistence_prediction"])
                    model_prediction = float(row.get("pre_shrinkage_prediction", row["prediction"]))
                    role = str(row.get("position"))
                    alpha = role_alphas.get(role, global_role_alpha)
                    row["nested_role_shrinkage_prediction"] = float(np.clip(
                        persistence + alpha * (model_prediction - persistence),
                        3.5, 10.0,
                    ))
                nested_role_summary = metric_summary([
                    {"actual": row["actual"], "prediction": row["nested_role_shrinkage_prediction"]}
                    for row in records
                ])

            summary = metric_summary(records)
            shrinkage_reconciliation = _persistence_reconciliation_diagnostics(records)
            target_mode_comparison = None
            if compare_target_modes:
                absolute_ensemble = _AbsoluteTargetEnsemble(enable_persistence_shrinkage=False)
                absolute_ensemble.train(train_data)
                absolute_records = evaluate_target_season(
                    absolute_ensemble,
                    data,
                    target_season,
                )
                target_mode_comparison = {
                    "delta_target": {
                        "raw_xgb_rmse": _rmse(records, "raw_xgb_prediction"),
                        "base_rmse": _rmse(records, "base_prediction"),
                        "ensemble_rmse": _rmse(records, "prediction"),
                        "n": len(records),
                    },
                    "absolute_target": {
                        "raw_xgb_rmse": _rmse(absolute_records, "raw_xgb_prediction"),
                        "base_rmse": _rmse(absolute_records, "base_prediction"),
                        "ensemble_rmse": _rmse(absolute_records, "prediction"),
                        "n": len(absolute_records),
                    },
                }
            base_rmse = _rmse(records, "base_prediction")
            persistence_rmse = _rmse(records, "persistence_prediction")
            for row in records:
                row["age_band"] = _age_band(int(row.get("age", 0) or 0))

            fold = {
                "fold": index,
                "train_through_season": source_season,
                "target_season": target_season,
                "train_history_start": min(seasons) if seasons else None,
                "train_history_end": source_season,
                "train_history_seasons": [
                    year for year in seasons if year <= source_season
                ],
                "valid": True,
                **summary,
                "base_rmse": base_rmse,
                "persistence_rmse": persistence_rmse,
                "target_mode_comparison": target_mode_comparison,
                "ensemble_vs_base_delta": base_rmse - summary["rmse"],
                "ensemble_vs_persistence_delta": persistence_rmse - summary["rmse"],
                "by_league": _segment_metrics(records, "league_id"),
                "by_competition": _segment_metrics(records, "competition"),
                "by_position": _segment_metrics(records, "position"),
                "by_age_band": _segment_metrics(records, "age_band"),
                "training": {
                    "validation_rmse": train_metrics.get("val_rmse"),
                    "calibration": train_metrics.get("final_calibration"),
                    "consecutive_pairs": train_metrics.get("consecutive_pairs"),
                    "competition_vocabulary": train_metrics.get("competition_vocabulary"),
                    "compatibility_samples_by_competition": train_metrics.get(
                        "compatibility_samples_by_competition"
                    ),
                },
            }
            folds.append(fold)
            if include_stage_metrics:
                fold["stage_metrics"] = _diagnostic_stage_rmse(records)
                fold["compatibility_diagnostics"] = _compatibility_diagnostics(records)
                fold["component_diagnostics"] = _component_diagnostics(records)
                fold["persistence_blend_diagnostics"] = _persistence_blend_diagnostics(records)
                fold["shrinkage_reconciliation"] = shrinkage_reconciliation
                fold["role_residual_diagnostics"] = _role_residual_diagnostics(records)
                fold["nested_shrinkage"] = {
                    **(nested_shrinkage or {}),
                    "outer_rmse": nested_summary["rmse"],
                    "outer_mae": nested_summary["mae"],
                    "outer_bias": nested_summary["bias"],
                    "outer_r2": nested_summary["r2"],
                    "outer_gain_vs_persistence": persistence_rmse - nested_summary["rmse"],
                    "outer_gain_vs_raw_model": summary["rmse"] - nested_summary["rmse"],
                }
                fold["nested_role_shrinkage"] = {
                    **(nested_role_shrinkage or {}),
                    "outer_rmse": nested_role_summary["rmse"],
                    "outer_mae": nested_role_summary["mae"],
                    "outer_bias": nested_role_summary["bias"],
                    "outer_r2": nested_role_summary["r2"],
                    "outer_gain_vs_persistence": persistence_rmse - nested_role_summary["rmse"],
                    "outer_gain_vs_raw_model": summary["rmse"] - nested_role_summary["rmse"],
                }
            all_records.extend(records)
            _logger.info(
                "[Backtest] target=%s n=%d ensemble_rmse=%.4f base=%.4f persistence=%.4f",
                target_season,
                len(records),
                summary["rmse"],
                base_rmse,
                persistence_rmse,
            )
        except Exception as exc:
            _logger.exception("[Backtest] fold target=%s failed", target_season)
            folds.append(
                {
                    "fold": index,
                    "train_through_season": source_season,
                    "target_season": target_season,
                    "valid": False,
                    "error": str(exc),
                }
            )

    overall = metric_summary(all_records)
    base_rmse = _rmse(all_records, "base_prediction")
    persistence_rmse = _rmse(all_records, "persistence_prediction")
    shrinkage_reconciliation = _persistence_reconciliation_diagnostics(all_records)
    nested_overall = None
    nested_shrinkage_rmse = float("nan")
    nested_role_overall = None
    nested_role_shrinkage_rmse = float("nan")
    if include_stage_metrics and all_records:
        nested_overall = metric_summary([
            {"actual": row["actual"], "prediction": row["nested_shrinkage_prediction"]}
            for row in all_records
        ])
        nested_shrinkage_rmse = nested_overall.get("rmse", float("nan"))
        nested_role_overall = metric_summary([
            {"actual": row["actual"], "prediction": row["nested_role_shrinkage_prediction"]}
            for row in all_records
        ])
        nested_role_shrinkage_rmse = nested_role_overall.get("rmse", float("nan"))
    if all_records:
        for row in all_records:
            row["age_band"] = _age_band(int(row.get("age", 0) or 0))

    valid = (
        bool(folds)
        and len(folds) == len(targets)
        and all(fold.get("valid") for fold in folds)
        and bool(overall)
        and math.isfinite(float(overall.get("rmse", float("nan"))))
    )
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "expanding_walk_forward_full_ensemble_competition_aware",
        "n_folds": len(folds),
        "target_seasons": targets,
        "folds": folds,
        "overall": overall,
        "overall_rmse": overall.get("rmse", float("nan")),
        "base_rmse": base_rmse,
        "persistence_rmse": persistence_rmse,
        "ensemble_vs_base_delta": (
            base_rmse - overall["rmse"] if overall and math.isfinite(base_rmse) else float("nan")
        ),
        "ensemble_vs_persistence_delta": (
            persistence_rmse - overall["rmse"]
            if overall and math.isfinite(persistence_rmse)
            else float("nan")
        ),
        "by_league": _segment_metrics(all_records, "league_id") if all_records else {},
        "by_competition": _segment_metrics(all_records, "competition") if all_records else {},
        "by_position": _segment_metrics(all_records, "position") if all_records else {},
        "by_age_band": _segment_metrics(all_records, "age_band") if all_records else {},
        "target_mode_comparison": None,
        "valid": valid,
    }

    if compare_target_modes:
        report["target_mode_comparison"] = [
            {
                "fold": fold["fold"],
                "target_season": fold["target_season"],
                **fold["target_mode_comparison"],
            }
            for fold in folds
            if fold.get("valid") and fold.get("target_mode_comparison")
        ]

    if include_stage_metrics and all_records:
        report["stage_metrics"] = _diagnostic_stage_rmse(all_records)
        report["compatibility_diagnostics"] = _compatibility_diagnostics(all_records)
        report["component_diagnostics"] = _component_diagnostics(all_records)
        report["persistence_blend_diagnostics"] = _persistence_blend_diagnostics(all_records)
        report["shrinkage_reconciliation"] = shrinkage_reconciliation
        report["role_residual_diagnostics"] = _role_residual_diagnostics(all_records)
        report["nested_shrinkage"] = {
            **(nested_overall or {}),
            "gain_vs_persistence": persistence_rmse - nested_shrinkage_rmse,
            "gain_vs_raw_model": overall["rmse"] - nested_shrinkage_rmse,
            "alphas": [
                {
                    "target_season": fold["target_season"],
                    "alpha": fold["nested_shrinkage"].get("alpha"),
                    "valid": fold["nested_shrinkage"].get("valid"),
                }
                for fold in folds
                if fold.get("valid") and fold.get("nested_shrinkage")
            ],
        }
        report["nested_role_shrinkage"] = {
            **(nested_role_overall or {}),
            "gain_vs_persistence": persistence_rmse - nested_role_shrinkage_rmse,
            "gain_vs_raw_model": overall["rmse"] - nested_role_shrinkage_rmse,
            "alphas": [
                {
                    "target_season": fold["target_season"],
                    "global_alpha": fold.get("nested_role_shrinkage", {}).get("global_alpha"),
                    "roles": fold.get("nested_role_shrinkage", {}).get("roles", {}),
                    "valid": fold.get("nested_role_shrinkage", {}).get("valid"),
                }
                for fold in folds
                if fold.get("valid") and fold.get("nested_role_shrinkage")
            ],
        }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _logger.info("[Backtest] report saved to %s", path)
    return report

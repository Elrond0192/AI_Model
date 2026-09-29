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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from basketball_ai.models.strict_production import (
    AsOfPositionPerformanceModel,
    StrictProductionEnsembleModel,
    build_historical_snapshot,
    evaluate_target_season,
    metric_summary,
    season_year,
)

_logger = logging.getLogger(__name__)


def _rmse(records: Iterable[Dict[str, Any]], key: str) -> float:
    rows = list(records)
    if not rows:
        return float("nan")
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    prediction = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def _segment_metrics(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    """Compare the model correction with persistence by segment."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        groups.setdefault(str(value), []).append(row)

    result: Dict[str, Dict[str, float]] = {}
    for name, rows in groups.items():
        actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
        persistence = np.asarray(
            [float(row["persistence_prediction"]) for row in rows], dtype=float
        )
        model = np.asarray(
            [float(row.get("pre_shrinkage_prediction", row["prediction"])) for row in rows],
            dtype=float,
        )
        persistence_error = persistence - actual
        model_error = model - actual
        result[name] = {
            "n": int(len(rows)),
            "persistence_rmse": float(np.sqrt(np.mean(persistence_error ** 2))),
            "model_rmse": float(np.sqrt(np.mean(model_error ** 2))),
            "gain_vs_persistence": float(
                np.sqrt(np.mean(persistence_error ** 2))
                - np.sqrt(np.mean(model_error ** 2))
            ),
            "persistence_bias": float(np.mean(persistence_error)),
            "model_bias": float(np.mean(model_error)),
            "correction_mean": float(np.mean(model - persistence)),
            "correction_std": float(np.std(model - persistence)),
        }
    return result


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


def _raw_xgb_stage_diagnostics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Diagnose the raw residual prediction and the age-stage transformation."""
    rows = [
        row for row in records
        if row.get("actual") is not None
        and row.get("persistence_prediction") is not None
        and row.get("raw_xgb_prediction") is not None
        and row.get("base_before_age") is not None
        and row.get("base_prediction") is not None
    ]
    if len(rows) < 3:
        return {"n": len(rows), "valid": False}

    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    persistence = np.asarray(
        [float(row["persistence_prediction"]) for row in rows], dtype=float
    )
    raw_xgb = np.asarray(
        [float(row["raw_xgb_prediction"]) for row in rows], dtype=float
    )
    base_before_age = np.asarray(
        [float(row["base_before_age"]) for row in rows], dtype=float
    )
    base = np.asarray([float(row["base_prediction"]) for row in rows], dtype=float)

    actual_delta = actual - persistence
    base_delta = base - persistence
    age_adjustment = base - base_before_age

    def _corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
        if np.std(left) <= 0.0 or np.std(right) <= 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "n": int(len(rows)),
        "valid": True,
        "actual_delta_mean": float(np.mean(actual_delta)),
        "actual_delta_std": float(np.std(actual_delta)),
        "raw_xgb_mean": float(np.mean(raw_xgb)),
        "raw_xgb_std": float(np.std(raw_xgb)),
        "raw_xgb_bias_as_delta": float(np.mean(raw_xgb - actual_delta)),
        "raw_xgb_rmse_as_delta": float(np.sqrt(np.mean((raw_xgb - actual_delta) ** 2))),
        "raw_xgb_correlation": _corr(raw_xgb, actual_delta),
        "base_delta_mean": float(np.mean(base_delta)),
        "base_delta_std": float(np.std(base_delta)),
        "age_adjustment_mean": float(np.mean(age_adjustment)),
        "age_adjustment_std": float(np.std(age_adjustment)),
        "age_adjustment_rmse": float(np.sqrt(np.mean(age_adjustment ** 2))),
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


def _feature_predictive_stability(folds: List[Dict[str, Any]], top_n: int = 30) -> Dict[str, Any]:
    """Aggregate train/validation target-correlation stability across folds.

    Diagnostic-only: this summarizes the feature/target relationship drift already
    measured per fold. It does not alter training, feature selection, or inference.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for fold in folds:
        if not fold.get("valid"):
            continue
        diagnostics = fold.get("feature_shift_diagnostics") or {}
        for item in diagnostics.get("features", []):
            feature = item.get("feature")
            if not feature:
                continue
            grouped.setdefault(str(feature), []).append({
                "target_season": fold.get("target_season"),
                "train_corr": item.get("train_target_correlation"),
                "validation_corr": item.get("validation_target_correlation"),
                "corr_shift": (
                    item.get("validation_target_correlation")
                    - item.get("train_target_correlation")
                    if item.get("train_target_correlation") is not None
                    and item.get("validation_target_correlation") is not None
                    else None
                ),
            })

    rows: List[Dict[str, Any]] = []
    for feature, observations in grouped.items():
        train_corr = np.asarray(
            [float(x["train_corr"]) for x in observations if x["train_corr"] is not None],
            dtype=float,
        )
        validation_corr = np.asarray(
            [float(x["validation_corr"]) for x in observations if x["validation_corr"] is not None],
            dtype=float,
        )
        shifts = np.asarray(
            [float(x["corr_shift"]) for x in observations if x["corr_shift"] is not None],
            dtype=float,
        )
        if validation_corr.size == 0:
            continue
        sign_flips = sum(
            1 for x in observations
            if x["train_corr"] is not None
            and x["validation_corr"] is not None
            and float(x["train_corr"]) * float(x["validation_corr"]) < 0.0
        )
        rows.append({
            "feature": feature,
            "folds_present": int(len(observations)),
            "train_corr_mean": float(np.mean(train_corr)) if train_corr.size else None,
            "validation_corr_mean": float(np.mean(validation_corr)),
            "mean_abs_validation_corr": float(np.mean(np.abs(validation_corr))),
            "validation_corr_std": float(np.std(validation_corr)),
            "mean_abs_corr_shift": float(np.mean(np.abs(shifts))) if shifts.size else None,
            "max_abs_corr_shift": float(np.max(np.abs(shifts))) if shifts.size else None,
            "sign_flip_count": int(sign_flips),
            "observations": observations,
        })

    rows.sort(
        key=lambda row: (
            row["mean_abs_corr_shift"] if row["mean_abs_corr_shift"] is not None else -1.0,
            row["mean_abs_validation_corr"] if row["mean_abs_validation_corr"] is not None else -1.0,
        ),
        reverse=True,
    )
    return {
        "n_features": len(rows),
        "top_n": min(int(top_n), len(rows)),
        "features": rows[:top_n],
    }



def _feature_distribution_diagnostics(
    ensemble: Any,
    oos_features: pd.DataFrame,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Compare the exact final-fit feature distribution with untouched OOS features.

    Diagnostic-only. It reports robust quantiles, scale/range shifts, missingness
    and the share of OOS values outside the training 1st/99th percentile.
    """
    model = getattr(ensemble, "perf_model", None)
    fit_raw = getattr(model, "_diagnostic_fit_X_raw", None)
    if not isinstance(fit_raw, pd.DataFrame) or fit_raw.empty:
        return {"valid": False, "reason": "production_fit_frame_unavailable"}
    if not isinstance(oos_features, pd.DataFrame) or oos_features.empty:
        return {"valid": False, "reason": "oos_feature_frame_unavailable"}

    metrics = getattr(model, "_last_metrics", {}) or {}
    stability = metrics.get("feature_shift_diagnostics") or {}
    ranked = [row.get("feature") for row in stability.get("features", []) if row.get("feature")]
    if not ranked:
        ranked = list(fit_raw.columns[:max(1, int(top_n))])
    ranked = ranked[:max(1, int(top_n))]

    fit = fit_raw.reindex(columns=ranked)
    oos = oos_features.reindex(columns=ranked)
    rows: List[Dict[str, Any]] = []

    def numeric(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    for feature in ranked:
        tr = numeric(fit[feature])
        oo = numeric(oos[feature])
        if tr.size == 0 or oo.size == 0:
            rows.append({
                "feature": feature,
                "gain": next(
                    (float(item.get("gain", 0.0)) for item in stability.get("features", [])
                     if item.get("feature") == feature),
                    0.0,
                ),
                "valid": False,
                "train_n": int(tr.size),
                "oos_n": int(oo.size),
            })
            continue

        p01, p25, p50, p75, p99 = np.percentile(tr, [1, 25, 50, 75, 99])
        o01, o25, o50, o75, o99 = np.percentile(oo, [1, 25, 50, 75, 99])
        train_iqr = float(max(p75 - p25, 0.0))
        oos_iqr = float(max(o75 - o25, 0.0))
        train_std = float(np.std(tr))
        oos_std = float(np.std(oo))
        train_range = float(np.max(tr) - np.min(tr))
        oos_range = float(np.max(oo) - np.min(oo))
        train_missing = float(pd.to_numeric(fit[feature], errors="coerce").isna().mean() * 100.0)
        oos_missing = float(pd.to_numeric(oos[feature], errors="coerce").isna().mean() * 100.0)

        outside = float(np.mean((oo < p01) | (oo > p99)) * 100.0)
        robust_scale = train_iqr if train_iqr > 1e-9 else max(train_std, 1e-9)
        median_shift = float((o50 - p50) / robust_scale)
        std_ratio = float(oos_std / train_std) if train_std > 1e-9 else None
        iqr_ratio = float(oos_iqr / train_iqr) if train_iqr > 1e-9 else None
        range_ratio = float(oos_range / train_range) if train_range > 1e-9 else None
        missing_delta_pp = float(oos_missing - train_missing)

        flags: List[str] = []
        if outside > 2.0:
            flags.append("oos_outside_train_p01_p99")
        if abs(median_shift) > 1.0:
            flags.append("median_shift_gt_1_iqr")
        if std_ratio is not None and (std_ratio > 2.0 or std_ratio < 0.5):
            flags.append("std_ratio_outside_0_5_2")
        if iqr_ratio is not None and (iqr_ratio > 2.0 or iqr_ratio < 0.5):
            flags.append("iqr_ratio_outside_0_5_2")
        if range_ratio is not None and (range_ratio > 3.0 or range_ratio < 1.0 / 3.0):
            flags.append("range_ratio_outside_1_3_3")
        if abs(missing_delta_pp) > 5.0:
            flags.append("missingness_delta_gt_5pp")

        gain = next(
            (
                float(item.get("gain", 0.0))
                for item in stability.get("features", [])
                if item.get("feature") == feature
            ),
            0.0,
        )
        rows.append({
            "feature": feature,
            "valid": True,
            "gain": gain,
            "train_n": int(tr.size),
            "oos_n": int(oo.size),
            "train_p01": float(p01),
            "train_p25": float(p25),
            "train_p50": float(p50),
            "train_p75": float(p75),
            "train_p99": float(p99),
            "train_min": float(np.min(tr)),
            "train_max": float(np.max(tr)),
            "train_std": train_std,
            "oos_p01": float(o01),
            "oos_p25": float(o25),
            "oos_p50": float(o50),
            "oos_p75": float(o75),
            "oos_p99": float(o99),
            "oos_min": float(np.min(oo)),
            "oos_max": float(np.max(oo)),
            "oos_std": oos_std,
            "train_missing_pct": train_missing,
            "oos_missing_pct": oos_missing,
            "missing_delta_pp": missing_delta_pp,
            "oos_outside_train_p01_p99_pct": outside,
            "robust_median_shift_iqr": median_shift,
            "std_ratio": std_ratio,
            "iqr_ratio": iqr_ratio,
            "range_ratio": range_ratio,
            "anomaly_flags": flags,
        })

    flagged = [row["feature"] for row in rows if row.get("anomaly_flags")]
    return {
        "valid": True,
        "comparison": "final_fit_vs_oos_target",
        "top_n": len(rows),
        "features_flagged": int(len(flagged)),
        "flagged_features": flagged,
        "features": rows,
    }


def _ridge_oos_diagnostics(
    ensemble: Any,
    oos_features: pd.DataFrame,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit a standardized Ridge on the exact XGB fit sample and evaluate on OOS.

    Diagnostic-only. The Ridge model is never used by production prediction.
    The target is the same native delta_vs_prior used by the production model.
    """
    model = getattr(ensemble, "perf_model", None)
    fit_X = getattr(model, "_diagnostic_fit_X", None)
    fit_y = getattr(model, "_diagnostic_fit_y", None)
    feature_names = list(getattr(model, "feature_names", []) or [])
    if not isinstance(fit_X, pd.DataFrame) or fit_X.empty:
        return {"valid": False, "reason": "production_fit_matrix_unavailable"}
    if fit_y is None or len(fit_y) != len(fit_X):
        return {"valid": False, "reason": "production_fit_target_unavailable"}
    if not isinstance(oos_features, pd.DataFrame) or oos_features.empty:
        return {"valid": False, "reason": "oos_feature_frame_unavailable"}
    if len(records) != len(oos_features):
        return {
            "valid": False,
            "reason": "oos_feature_record_length_mismatch",
            "feature_rows": len(oos_features),
            "records": len(records),
        }
    if not feature_names:
        return {"valid": False, "reason": "feature_names_unavailable"}

    X_fit = fit_X.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_oos = oos_features.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_fit = np.asarray(fit_y, dtype=float)
    persistence = np.asarray(
        [float(row["persistence_prediction"]) for row in records],
        dtype=float,
    )
    actual = np.asarray([float(row["actual"]) for row in records], dtype=float)
    actual_delta = actual - persistence

    if len(X_oos) < 2 or np.std(actual_delta) <= 1e-12:
        return {"valid": False, "reason": "insufficient_oos_variance", "n": int(len(X_oos))}

    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    ridge.fit(X_fit, y_fit)
    pred_delta = np.asarray(ridge.predict(X_oos), dtype=float)

    error = pred_delta - actual_delta
    persistence_error = persistence - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_error ** 2)))
    sst = float(np.sum((actual_delta - np.mean(actual_delta)) ** 2))
    corr = (
        float(np.corrcoef(pred_delta, actual_delta)[0, 1])
        if np.std(pred_delta) > 1e-12
        else None
    )
    return {
        "valid": True,
        "model": "Ridge",
        "alpha": 1.0,
        "standardized": True,
        "target_mode": str(getattr(model, "target_mode", "unknown")),
        "fit_n": int(len(X_fit)),
        "oos_n": int(len(X_oos)),
        "fit_target_mean": float(np.mean(y_fit)),
        "fit_target_std": float(np.std(y_fit)),
        "oos_actual_delta_mean": float(np.mean(actual_delta)),
        "oos_actual_delta_std": float(np.std(actual_delta)),
        "oos_prediction_mean": float(np.mean(pred_delta)),
        "oos_prediction_std": float(np.std(pred_delta)),
        "oos_bias_as_delta": float(np.mean(error)),
        "oos_rmse_as_delta": rmse,
        "oos_mae_as_delta": float(np.mean(np.abs(error))),
        "oos_r2_as_delta": float(1.0 - np.sum(error ** 2) / sst) if sst > 0.0 else 0.0,
        "oos_correlation": corr,
        "persistence_rmse": persistence_rmse,
        "gain_vs_persistence": float(persistence_rmse - rmse),
        "prediction_minus_persistence_std": float(np.std(pred_delta)),
        "oos_sse_as_delta": float(np.sum(error ** 2)),
    }


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
    compare_target_modes: bool = False,  # retained for CLI compatibility; no longer emitted
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
            ensemble.train(train_data)
            if include_stage_metrics:
                ensemble._capture_diagnostic_features = True
            records = evaluate_target_season(ensemble, data, target_season)
            if len(records) < minimum:
                raise RuntimeError(
                    f"only {len(records)} evaluable samples; minimum is {minimum}"
                )

            nested_shrinkage = None
            nested_summary = None
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


            summary = metric_summary(records)
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
                "ensemble_vs_base_delta": base_rmse - summary["rmse"],
                "ensemble_vs_persistence_delta": persistence_rmse - summary["rmse"],
                "by_league": _segment_metrics(records, "league_id"),
                "by_competition": _segment_metrics(records, "competition"),
                "by_position": _segment_metrics(records, "position"),
                "by_age_band": _segment_metrics(records, "age_band"),
            }
            folds.append(fold)
            if include_stage_metrics:
                last_metrics = getattr(ensemble.perf_model, "_last_metrics", {}) or {}
                fold["target_diagnostics"] = dict(
                    last_metrics.get("target_diagnostics", {})
                )
                fold["feature_shift_diagnostics"] = dict(
                    last_metrics.get("feature_shift_diagnostics", {})
                )
                fold["feature_distribution_diagnostics"] = _feature_distribution_diagnostics(
                    ensemble,
                    getattr(ensemble, "_last_oos_feature_frame", pd.DataFrame()),
                    top_n=20,
                )
                fold["ridge_oos_diagnostics"] = _ridge_oos_diagnostics(
                    ensemble,
                    getattr(ensemble, "_last_oos_feature_frame", pd.DataFrame()),
                    records,
                )
                fold["persistence_blend_diagnostics"] = _persistence_blend_diagnostics(records)
                fold["raw_xgb_stage_diagnostics"] = _raw_xgb_stage_diagnostics(records)
                fold["nested_shrinkage"] = {
                    **(nested_shrinkage or {}),
                    "outer_rmse": nested_summary["rmse"],
                    "outer_mae": nested_summary["mae"],
                    "outer_bias": nested_summary["bias"],
                    "outer_r2": nested_summary["r2"],
                    "outer_gain_vs_persistence": persistence_rmse - nested_summary["rmse"],
                    "outer_gain_vs_raw_model": summary["rmse"] - nested_summary["rmse"],
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
    nested_overall = None
    nested_shrinkage_rmse = float("nan")
    if include_stage_metrics and all_records:
        nested_overall = metric_summary([
            {"actual": row["actual"], "prediction": row["nested_shrinkage_prediction"]}
            for row in all_records
        ])
        nested_shrinkage_rmse = nested_overall.get("rmse", float("nan"))
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
        "valid": valid,
    }

    if include_stage_metrics:
        report["feature_predictive_stability"] = _feature_predictive_stability(folds)

    if include_stage_metrics and all_records:
        report["persistence_blend_diagnostics"] = _persistence_blend_diagnostics(all_records)
        report["raw_xgb_stage_diagnostics"] = _raw_xgb_stage_diagnostics(all_records)
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
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _logger.info("[Backtest] report saved to %s", path)
    return report

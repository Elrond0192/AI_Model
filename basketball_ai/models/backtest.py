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
            "model_bias": float(np.mean(model_error)),
            "correction_mean": float(np.mean(model - persistence)),
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


def _production_ablation_diagnostics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """OOS decomposition of raw target, floor, shrinkage and final clipping.

    A = raw XGB absolute target without a production floor.
    B = A with the 3.5-10.0 production floor.
    P = actual pre-shrinkage production input.
    C = actual production shrinkage without the final floor.
    D = final production prediction.

    This is diagnostic-only; it never changes inference.
    """
    rows = [
        row for row in records
        if row.get("actual") is not None
        and row.get("persistence_prediction") is not None
        and row.get("raw_target_prediction") is not None
        and row.get("pre_shrinkage_prediction") is not None
        and row.get("persistence_shrinkage_alpha") is not None
    ]
    if len(rows) < 2:
        return {"n": len(rows), "valid": False}

    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    persistence = np.asarray([float(row["persistence_prediction"]) for row in rows], dtype=float)
    raw_target = np.asarray([float(row["raw_target_prediction"]) for row in rows], dtype=float)
    pre_shrinkage = np.asarray([float(row["pre_shrinkage_prediction"]) for row in rows], dtype=float)
    alpha = np.asarray([float(row["persistence_shrinkage_alpha"]) for row in rows], dtype=float)

    raw_xgb_floor = np.clip(raw_target, 3.5, 10.0)
    post_shrinkage_unclipped = persistence + alpha * (pre_shrinkage - persistence)
    final_prediction = np.clip(post_shrinkage_unclipped, 3.5, 10.0)

    def _rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - actual) ** 2)))

    def _mae(values: np.ndarray) -> float:
        return float(np.mean(np.abs(values - actual)))

    def _bias(values: np.ndarray) -> float:
        return float(np.mean(values - actual))

    stage_values = {
        "A_raw_xgb": raw_target,
        "B_raw_xgb_floor": raw_xgb_floor,
        "P_pre_shrinkage": pre_shrinkage,
        "C_shrinkage_unclipped": post_shrinkage_unclipped,
        "D_final_production": final_prediction,
    }
    stages = {
        name: {"rmse": _rmse(values), "mae": _mae(values), "bias": _bias(values)}
        for name, values in stage_values.items()
    }

    return {
        "n": int(len(rows)),
        "valid": True,
        "stages": stages,
        "effects": {
            "raw_floor_effect_rmse": stages["B_raw_xgb_floor"]["rmse"] - stages["A_raw_xgb"]["rmse"],
            "pre_shrinkage_vs_raw_rmse": stages["P_pre_shrinkage"]["rmse"] - stages["A_raw_xgb"]["rmse"],
            "shrinkage_effect_rmse": stages["C_shrinkage_unclipped"]["rmse"] - stages["P_pre_shrinkage"]["rmse"],
            "final_floor_effect_rmse": stages["D_final_production"]["rmse"] - stages["C_shrinkage_unclipped"]["rmse"],
            "total_post_processing_rmse": stages["D_final_production"]["rmse"] - stages["A_raw_xgb"]["rmse"],
        },
        "raw_floor": {
            "low_count": int(np.sum(raw_target < 3.5)),
            "high_count": int(np.sum(raw_target > 10.0)),
            "low_pct": float(np.mean(raw_target < 3.5) * 100.0),
            "high_pct": float(np.mean(raw_target > 10.0) * 100.0),
        },
        "pre_shrinkage_floor": {
            "low_count": int(np.sum(pre_shrinkage < 3.5)),
            "high_count": int(np.sum(pre_shrinkage > 10.0)),
            "low_pct": float(np.mean(pre_shrinkage < 3.5) * 100.0),
            "high_pct": float(np.mean(pre_shrinkage > 10.0) * 100.0),
        },
        "final_floor": {
            "low_count": int(np.sum(post_shrinkage_unclipped < 3.5)),
            "high_count": int(np.sum(post_shrinkage_unclipped > 10.0)),
            "low_pct": float(np.mean(post_shrinkage_unclipped < 3.5) * 100.0),
            "high_pct": float(np.mean(post_shrinkage_unclipped > 10.0) * 100.0),
        },
        "alpha": {
            "mean": float(np.mean(alpha)),
            "min": float(np.min(alpha)),
            "max": float(np.max(alpha)),
            "changed_pct": float(np.mean(alpha < 0.999999) * 100.0),
        },
        "shrinkage_movement": {
            "mean": float(np.mean(post_shrinkage_unclipped - pre_shrinkage)),
            "mean_abs": float(np.mean(np.abs(post_shrinkage_unclipped - pre_shrinkage))),
        },
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


def _feature_sanity_diagnostics(
    ensemble: Any,
    oos_features: pd.DataFrame,
) -> Dict[str, Any]:
    """Compact sanity checks for the few source features currently under review."""
    if not isinstance(oos_features, pd.DataFrame) or oos_features.empty:
        return {"valid": False, "reason": "oos_feature_frame_unavailable"}

    checks = {
        "age": ("age",),
        "avg_usg_pct": ("avg_usg_pct",),
        "avg_pts_per_40": ("avg_pts_per_40",),
        "avg_orb_pct": ("avg_orb_pct",),
    }
    result: Dict[str, Any] = {"valid": True}
    for feature in checks:
        if feature not in oos_features.columns:
            result[feature] = {"valid": False}
            continue
        values = pd.to_numeric(oos_features[feature], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if values.empty:
            result[feature] = {"valid": False, "n": 0}
            continue
        item: Dict[str, Any] = {
            "valid": True,
            "n": int(len(values)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if feature == "age":
            item["pct_at_floor_14"] = float((values <= 14).mean() * 100.0)
        elif feature == "avg_usg_pct":
            item["pct_gt_1"] = float((values > 1.0).mean() * 100.0)
        elif feature == "avg_pts_per_40":
            item["pct_gt_60"] = float((values > 60.0).mean() * 100.0)
        elif feature == "avg_orb_pct":
            item["pct_gt_1"] = float((values > 1.0).mean() * 100.0)
        result[feature] = item
    return result


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
                "by_competition": _segment_metrics(records, "competition"),
                "by_position": _segment_metrics(records, "position"),
            }
            folds.append(fold)
            if include_stage_metrics:
                fold["feature_sanity"] = _feature_sanity_diagnostics(
                    ensemble,
                    getattr(ensemble, "_last_oos_feature_frame", pd.DataFrame()),
                )
                fold["persistence_blend"] = _persistence_blend_diagnostics(records)
                fold["raw_xgb"] = _raw_xgb_stage_diagnostics(records)
                fold["production_ablation"] = _production_ablation_diagnostics(records)
                fold["nested_shrinkage"] = {
                    "valid": bool((nested_shrinkage or {}).get("valid", False)),
                    "alpha": (nested_shrinkage or {}).get("alpha"),
                    "inner_n": (nested_shrinkage or {}).get("inner_n"),
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
        "by_competition": _segment_metrics(all_records, "competition") if all_records else {},
        "by_position": _segment_metrics(all_records, "position") if all_records else {},
        "valid": valid,
    }

    if include_stage_metrics and all_records:
        report["feature_sanity"] = (
            folds[-1].get("feature_sanity", {})
            if folds else {}
        )
        report["persistence_blend"] = _persistence_blend_diagnostics(all_records)
        report["raw_xgb"] = _raw_xgb_stage_diagnostics(all_records)
        report["production_ablation"] = _production_ablation_diagnostics(all_records)
        report["nested_shrinkage"] = {
            "valid": bool(nested_overall),
            "rmse": nested_shrinkage_rmse,
            "mae": (nested_overall or {}).get("mae"),
            "bias": (nested_overall or {}).get("bias"),
            "r2": (nested_overall or {}).get("r2"),
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

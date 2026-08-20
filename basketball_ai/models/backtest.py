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
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        groups.setdefault(str(value), []).append(row)
    return {name: metric_summary(rows) for name, rows in groups.items()}


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
    n_folds: int = 3,
    output_path: Optional[str] = "models_saved/backtest_report.json",
    min_samples_per_fold: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the complete production path on untouched future seasons.

    Each fold trains/calibrates using only seasons up to ``target-1`` and then
    predicts the actual player/team outcome in ``target``.  The report compares
    the final ensemble against both its base XGBoost score and a persistence
    baseline (previous-season rating), and measures interval coverage.
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
            "error": "team_season_stats is required",
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

            summary = metric_summary(records)
            base_rmse = _rmse(records, "base_prediction")
            persistence_rmse = _rmse(records, "persistence_prediction")
            for row in records:
                row["age_band"] = _age_band(int(row.get("age", 0) or 0))

            fold = {
                "fold": index,
                "train_through_season": source_season,
                "target_season": target_season,
                "valid": True,
                **summary,
                "base_rmse": base_rmse,
                "persistence_rmse": persistence_rmse,
                "ensemble_vs_base_delta": base_rmse - summary["rmse"],
                "ensemble_vs_persistence_delta": persistence_rmse - summary["rmse"],
                "by_league": _segment_metrics(records, "league_id"),
                "by_position": _segment_metrics(records, "position"),
                "by_age_band": _segment_metrics(records, "age_band"),
                "training": {
                    "validation_rmse": train_metrics.get("val_rmse"),
                    "calibration": train_metrics.get("final_calibration"),
                    "consecutive_pairs": train_metrics.get("consecutive_pairs"),
                },
            }
            folds.append(fold)
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
        "method": "expanding_walk_forward_full_ensemble",
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
        "by_position": _segment_metrics(all_records, "position") if all_records else {},
        "by_age_band": _segment_metrics(all_records, "age_band") if all_records else {},
        "valid": valid,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _logger.info("[Backtest] report saved to %s", path)
    return report

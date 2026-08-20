"""Walk-forward backtesting for the PerformanceModel.

Splits the player_stats data chronologically into training/validation windows
and reports per-fold and per-league RMSE metrics.

Usage::

    from basketball_ai.models.backtest import run_backtest
    report = run_backtest(data, n_folds=3)
    # report: {"folds": [...], "overall_rmse": ..., "by_league": {...}}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def run_backtest(
    data: Dict[str, Any],
    n_folds: int = 3,
    output_path: Optional[str] = "models_saved/backtest_report.json",
) -> Dict[str, Any]:
    """Run walk-forward cross-validation and return a backtest report.

    Args:
        data:        Full data dict from the PostgreSQL loader.
        n_folds:     Number of temporal folds.
        output_path: If set, write JSON report to this path.

    Returns:
        Dict with keys ``folds``, ``overall_rmse``, ``by_league``.
    """
    from basketball_ai.models.performance_model import PerformanceModel

    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    players_df:   pd.DataFrame = data.get("players",      pd.DataFrame())

    if player_stats.empty or "season" not in player_stats.columns:
        return {"error": "player_stats empty or missing season column"}

    latest_data_year = max((int(str(s).split("-")[0]) for s in player_stats["season"].dropna()), default=datetime.now(timezone.utc).year)
    birth_year_map: Dict[int, int] = {}
    if not players_df.empty and "id" in players_df.columns and "age" in players_df.columns:
        for _, _pr in players_df.iterrows():
            try:
                _pid = int(_pr["id"])
                _age = _pr.get("age")
                if _age is not None and not (isinstance(_age, float) and np.isnan(_age)):
                    birth_date = _pr.get("birth_date", _pr.get("date_of_birth"))
                    birth_year_map[_pid] = int(str(birth_date)[:4]) if birth_date else latest_data_year - int(_age)
            except (TypeError, ValueError):
                pass

    position_map: Dict[int, str] = {}
    if not players_df.empty and "id" in players_df.columns and "position" in players_df.columns:
        for _, _pr in players_df.iterrows():
            try:
                position_map[int(_pr["id"])] = str(_pr["position"])
            except (TypeError, ValueError):
                pass

    seasons = sorted(player_stats["season"].unique())
    # PerformanceModel requires three target-season blocks internally, which
    # means at least four source seasons before the first out-of-time fold.
    if len(seasons) < 5:
        return {"error": "at least five seasons are required", "valid": False, "folds": []}
    n_folds = min(n_folds, len(seasons) - 4)
    folds_results: List[Dict[str, Any]] = []
    all_preds: List[float] = []
    all_truths: List[float] = []

    for i in range(n_folds):
        train_end = 4 + i
        train_seasons = seasons[:train_end]
        val_seasons = seasons[train_end:train_end + 1]
        if not val_seasons:
            break

        train_stats = player_stats[player_stats["season"].isin(train_seasons)]
        val_stats   = player_stats[player_stats["season"].isin(val_seasons)]

        if train_stats.empty or val_stats.empty:
            folds_results.append({"fold": i + 1, "train_seasons": [str(s) for s in train_seasons], "val_seasons": [str(s) for s in val_seasons], "valid": False, "error": "empty train or validation partition"})
            continue

        fold_data = {**data, "player_stats": train_stats}
        model = PerformanceModel()
        try:
            model.train(fold_data)
        except Exception as exc:
            _logger.warning("[Backtest] Fold %d train failed: %s", i, exc)
            folds_results.append({"fold": i + 1, "train_seasons": [str(s) for s in train_seasons], "val_seasons": [str(s) for s in val_seasons], "valid": False, "error": str(exc)})
            continue

        # Predict on validation fold
        fold_preds: List[float] = []
        fold_truths: List[float] = []
        fold_leagues: List[Any] = []
        excluded_no_history = 0
        for _, row in val_stats.iterrows():
            try:
                hist = train_stats[train_stats["player_id"] == row["player_id"]]
                if hist.empty:
                    raise ValueError("player has no prior-season history")
                _pid = int(row["player_id"])
                # Compute age for the historical season rather than using
                # the player's current age (which would introduce temporal leakage)
                _season_str = str(row.get("season", latest_data_year))
                try:
                    _season_year = int(_season_str.split("-")[0])
                except (ValueError, IndexError):
                    raise ValueError("invalid season")
                _by = birth_year_map.get(_pid)
                if _by is not None:
                    age = _season_year - _by
                else:
                    age = 25  # fallback when birth_year unknown
                pos = position_map.get(_pid, "PG")
                source = hist.sort_values("season").iloc[-1]
                feat = model._build_row(source, age - 1, pos, hist)
                feat_df = pd.DataFrame([feat]).reindex(columns=model.feature_names, fill_value=0.0)
                pred = float(model.model.predict(feat_df)[0])
            except ValueError as exc:
                if "prior-season history" in str(exc):
                    excluded_no_history += 1
                    continue
                raise RuntimeError(f"Backtest fold {i + 1} prediction failed") from exc
            except Exception as exc:
                raise RuntimeError(f"Backtest fold {i + 1} prediction failed") from exc
            fold_preds.append(pred)
            fold_truths.append(float(row["rating"]))
            fold_leagues.append(row.get("league_id"))

        if not fold_preds:
            folds_results.append({"fold": i + 1, "train_seasons": [str(s) for s in train_seasons], "val_seasons": [str(s) for s in val_seasons], "valid": False, "error": "no validation players with prior history"})
            continue

        fold_preds_arr = np.array(fold_preds)
        y_true = np.array(fold_truths)
        fold_rmse = float(np.sqrt(np.mean((fold_preds_arr - y_true) ** 2)))
        all_preds.extend(fold_preds)
        all_truths.extend(y_true.tolist())

        # Per-league RMSE for this fold
        by_league: Dict[str, float] = {}
        if "league_id" in val_stats.columns:
            val_with_preds = pd.DataFrame({"league_id": fold_leagues, "rating": fold_truths, "_pred": fold_preds_arr})
            for league_id, grp in val_with_preds.groupby("league_id"):
                rmse_l = float(np.sqrt(np.mean((grp["_pred"].values - grp["rating"].values) ** 2)))
                by_league[str(league_id)] = round(rmse_l, 4)

        folds_results.append({
            "fold":          i + 1,
            "train_seasons": [str(s) for s in train_seasons],
            "val_seasons":   [str(s) for s in val_seasons],
            "val_rmse":      round(fold_rmse, 4),
            "val_n":         len(y_true),
            "by_league":     by_league,
            "valid":         True,
            "excluded_no_history": excluded_no_history,
        })
        _logger.info("[Backtest] Fold %d: val_rmse=%.4f on %d samples", i + 1, fold_rmse, len(y_true))

    overall_rmse = (
        float(np.sqrt(np.mean((np.array(all_preds) - np.array(all_truths)) ** 2)))
        if all_preds
        else float("nan")
    )

    # Aggregate by_league across folds
    agg_by_league: Dict[str, List[float]] = {}
    for fold in folds_results:
        for lig, rmse in fold.get("by_league", {}).items():
            agg_by_league.setdefault(lig, []).append(rmse)
    by_league_mean = {k: round(float(np.mean(v)), 4) for k, v in agg_by_league.items()}

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_folds":      len(folds_results),
        "overall_rmse": round(overall_rmse, 4),
        "folds":        folds_results,
        "by_league":    by_league_mean,
        "valid": bool(folds_results) and len(folds_results) == n_folds and all(f.get("valid") for f in folds_results),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _logger.info("[Backtest] Report saved to %s", p)

    return report

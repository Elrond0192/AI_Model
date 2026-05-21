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
        data:        Full data dict (from loader/sql_loader).
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

    seasons = sorted(player_stats["season"].unique())
    if len(seasons) < n_folds + 1:
        n_folds = max(1, len(seasons) - 1)

    fold_size = len(seasons) // (n_folds + 1)
    folds_results: List[Dict[str, Any]] = []
    all_preds: List[float] = []
    all_truths: List[float] = []

    for i in range(n_folds):
        train_seasons = seasons[: (i + 1) * fold_size]
        val_seasons   = seasons[(i + 1) * fold_size : (i + 2) * fold_size]
        if not val_seasons:
            break

        train_stats = player_stats[player_stats["season"].isin(train_seasons)]
        val_stats   = player_stats[player_stats["season"].isin(val_seasons)]

        if train_stats.empty or val_stats.empty:
            continue

        fold_data = {**data, "player_stats": train_stats}
        model = PerformanceModel()
        try:
            model.train(fold_data)
        except Exception as exc:
            _logger.warning("[Backtest] Fold %d train failed: %s", i, exc)
            continue

        # Predict on validation fold
        y_true = val_stats["rating"].values
        fold_preds: List[float] = []
        for _, row in val_stats.iterrows():
            try:
                hist = train_stats[train_stats["player_id"] == row["player_id"]]
                if not players_df.empty:
                    age_series = players_df.loc[players_df["id"] == row["player_id"], "age"]
                    age = int(age_series.iloc[0]) if not age_series.empty else 25
                    pos_series = players_df.loc[players_df["id"] == row["player_id"], "position"]
                    pos = str(pos_series.iloc[0]) if not pos_series.empty else "PG"
                else:
                    age, pos = 25, "PG"
                feat = model._build_row(row, age, pos, hist)
                feat_df = pd.DataFrame([feat]).reindex(columns=model.feature_names, fill_value=0.0)
                scaled = model.scaler.transform(feat_df)
                pred = float(model.model.predict(scaled)[0])
            except Exception:
                pred = float(np.mean(y_true)) if len(y_true) > 0 else 5.0
            fold_preds.append(pred)

        fold_preds_arr = np.array(fold_preds)
        fold_rmse = float(np.sqrt(np.mean((fold_preds_arr - y_true) ** 2)))
        all_preds.extend(fold_preds)
        all_truths.extend(y_true.tolist())

        # Per-league RMSE for this fold
        by_league: Dict[str, float] = {}
        if "league_id" in val_stats.columns:
            val_with_preds = val_stats.copy()
            val_with_preds["_pred"] = fold_preds_arr
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
        })
        _logger.info("[Backtest] Fold %d: val_rmse=%.4f on %d samples", i + 1, fold_rmse, len(y_true))

    overall_rmse = (
        float(np.sqrt(np.mean((np.array(all_preds) - np.array(all_truths)) ** 2)))
        if all_preds
        else 0.0
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
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _logger.info("[Backtest] Report saved to %s", p)

    return report

"""Strict production model contract shared by training, backtest and API runtime.

This module closes the remaining differences between historical evaluation and
served inference without changing the legacy model classes used by development
tests.  Production callers should use :class:`StrictProductionEnsembleModel`.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from basketball_ai.data.loader import _to_int
from basketball_ai.models.production_training import (
    ProductionEnsembleModel,
    SeasonAheadPerformanceModel,
    TemporalCompatibilityModel,
    build_historical_snapshot,
    evaluate_target_season as _evaluate_target_season,
    metric_summary,
    season_year,
)

_VALID_POSITIONS = {
    "PG", "SG", "SF", "PF", "C",
    "PG/SG", "SG/PG", "SG/SF", "SF/SG", "SF/PF", "PF/SF",
    "PF/C", "C/PF", "PG/SF", "SF/PG", "SG/PF", "PF/SG",
}


def _normalise_position(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper().replace(" ", "")
    return text if text in _VALID_POSITIONS else None


def position_as_of(
    player_id: int,
    relations: pd.DataFrame,
    season: int,
    fallback: str = "PG",
) -> str:
    """Return the roster position known at or before ``season``.

    The canonical ``team_player_relations.role`` field contains the player's
    roster position from the season-specific Anagrafiche row.  Non-position
    legacy values such as ``starter`` are ignored rather than overriding the
    fallback.
    """
    if relations is None or relations.empty or "player_id" not in relations.columns:
        return fallback
    rows = relations[relations["player_id"].map(_to_int) == _to_int(player_id)].copy()
    if rows.empty or "season" not in rows.columns:
        return fallback
    rows["_season_year"] = rows["season"].map(
        lambda value: season_year(value) if pd.notna(value) else np.nan
    )
    rows = rows[rows["_season_year"].notna() & (rows["_season_year"] <= int(season))]
    if rows.empty:
        return fallback
    for _, row in rows.sort_values("_season_year", ascending=False).iterrows():
        position = _normalise_position(row.get("role"))
        if position:
            return position
    return fallback


class AsOfPositionPerformanceModel(SeasonAheadPerformanceModel):
    """Season-ahead XGBoost model using the source-season roster position."""

    _position_relations: Optional[pd.DataFrame] = None

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        self._position_relations = data.get("team_player_relations")
        return super().prepare_features(
            data,
            extra_metrics=extra_metrics,
            split_season=split_season,
        )

    def _build_row(
        self,
        stat_row: pd.Series,
        age: int,
        position: str,
        player_stats_history: pd.DataFrame,
        extra_metrics: Optional[List[str]] = None,
        league_max_games: Optional[Dict[int, int]] = None,
    ) -> Dict[str, float]:
        try:
            source_season = season_year(stat_row.get("season"))
            player_id = _to_int(stat_row.get("player_id"))
            position = position_as_of(
                player_id,
                self._position_relations if self._position_relations is not None else pd.DataFrame(),
                source_season,
                position,
            )
        except (TypeError, ValueError):
            pass
        return super()._build_row(
            stat_row,
            age,
            position,
            player_stats_history,
            extra_metrics=extra_metrics,
            league_max_games=league_max_games,
        )


class StrictProductionEnsembleModel(ProductionEnsembleModel):
    """Exact production ensemble used by training, backtests and API serving."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("performance_model", AsOfPositionPerformanceModel())
        kwargs.setdefault("compatibility_model", TemporalCompatibilityModel())
        super().__init__(*args, **kwargs)
        self._conformal_nominal_coverage: Optional[float] = None

    def _data_with_asof_position(
        self,
        player_id: int,
        data: Dict[str, Any],
        season: int,
    ) -> Dict[str, Any]:
        player = data.get("player_dict", {}).get(_to_int(player_id))
        if not player:
            return data
        position = position_as_of(
            player_id,
            data.get("team_player_relations", pd.DataFrame()),
            season,
            str(player.get("position", "PG") or "PG"),
        )
        if position == str(player.get("position", "PG") or "PG"):
            return data
        patched = dict(data)
        patched_players = dict(data.get("player_dict", {}))
        patched_player = dict(player)
        patched_player["position"] = position
        patched_players[_to_int(player_id)] = patched_player
        patched["player_dict"] = patched_players
        return patched

    def _calibrate_conformal(self, conformal_residuals: Optional[List[float]]) -> None:
        """Finite-sample split-conformal calibration for the final ensemble.

        A single absolute-error quantile is used directly around the final
        prediction.  No reliability or player-specific rescaling is applied,
        preserving the coverage interpretation of the calibration set.
        """
        if not conformal_residuals or len(conformal_residuals) < 10:
            self._conformal_q_lo = None
            self._conformal_q_hi = None
            self._conformal_nominal_coverage = None
            return
        residuals = np.asarray(conformal_residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 10:
            self._conformal_q_lo = None
            self._conformal_q_hi = None
            self._conformal_nominal_coverage = None
            return
        coverage = float(os.getenv("MODEL_TARGET_INTERVAL_COVERAGE", "0.90"))
        coverage = float(np.clip(coverage, 0.50, 0.999))
        rank = min(len(residuals), int(math.ceil((len(residuals) + 1) * coverage)))
        quantile = float(np.sort(residuals)[rank - 1])
        self._conformal_q_lo = quantile
        self._conformal_q_hi = quantile
        self._conformal_nominal_coverage = coverage

    def _predict_uncached(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: int,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ):
        bounded_data = self._data_with_asof_position(player_id, data, int(season))
        result = super()._predict_uncached(
            player_id,
            team_id,
            bounded_data,
            season,
            target_age,
            competition,
        )
        if self._conformal_q_hi is not None:
            half_width = float(self._conformal_q_hi)
            result.confidence_low = round(
                float(np.clip(result.predicted_rating - half_width, 1.0, 10.0)),
                3,
            )
            result.confidence_high = round(
                float(np.clip(result.predicted_rating + half_width, 1.0, 10.0)),
                3,
            )
        return result


def evaluate_target_season(
    ensemble: StrictProductionEnsembleModel,
    data: Dict[str, Any],
    target_season: int,
):
    """Evaluate and label segments using the source-season roster position."""
    records = _evaluate_target_season(ensemble, data, target_season)
    source_season = int(target_season) - 1
    snapshot = build_historical_snapshot(data, source_season)
    relations = snapshot.get("team_player_relations", pd.DataFrame())
    for record in records:
        fallback = str(record.get("position", "PG") or "PG")
        record["position"] = position_as_of(
            int(record["player_id"]), relations, source_season, fallback
        )
    return records


__all__ = [
    "AsOfPositionPerformanceModel",
    "StrictProductionEnsembleModel",
    "build_historical_snapshot",
    "evaluate_target_season",
    "metric_summary",
    "position_as_of",
    "season_year",
]

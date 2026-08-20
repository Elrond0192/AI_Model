"""Strict production model contract shared by training, backtest and API runtime.

Production callers use this module so model fitting, historical evaluation,
batch publishing and online inference all execute the same leakage-safe path.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from basketball_ai.data.loader import _to_int
from basketball_ai.models.production_training import (
    ProductionEnsembleModel,
    SeasonAheadPerformanceModel,
    TemporalCompatibilityModel,
    build_historical_snapshot as _build_historical_snapshot,
    metric_summary,
    season_year,
)

_VALID_POSITIONS = {
    "PG", "SG", "SF", "PF", "C",
    "PG/SG", "SG/PG", "SG/SF", "SF/SG", "SF/PF", "PF/SF",
    "PF/C", "C/PF", "PG/SF", "SF/PG", "SG/PF", "PF/SG",
}
_STATE_FILE = "production_state.joblib"


def _normalise_position(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper().replace(" ", "")
    return text if text in _VALID_POSITIONS else None


def position_as_of(
    player_id: int,
    relations: pd.DataFrame,
    season: int,
    fallback: str = "PG",
) -> str:
    """Return the latest valid roster position known at/before ``season``."""
    if relations is None or relations.empty or "player_id" not in relations.columns:
        return fallback
    rows = relations[relations["player_id"].map(_to_int) == _to_int(player_id)].copy()
    if rows.empty or "season" not in rows.columns:
        return fallback
    rows["_season_year"] = rows["season"].map(
        lambda value: season_year(value) if pd.notna(value) else np.nan
    )
    rows = rows[
        rows["_season_year"].notna()
        & (rows["_season_year"] <= int(season))
    ]
    if rows.empty:
        return fallback
    for _, row in rows.sort_values("_season_year", ascending=False).iterrows():
        position = _normalise_position(row.get("role"))
        if position:
            return position
    return fallback


def build_historical_snapshot(data: Dict[str, Any], source_season: int) -> Dict[str, Any]:
    """Return a hard-bounded as-of snapshot with no future team/player state."""
    source_season = int(source_season)
    snapshot = _build_historical_snapshot(data, source_season)

    team_history = snapshot.get("team_season_stats")
    if team_history is not None and not team_history.empty:
        bounded = team_history.copy()
        bounded["_season_year"] = bounded["season"].map(
            lambda value: season_year(value) if pd.notna(value) else np.nan
        )
        bounded = bounded[
            bounded["_season_year"].notna()
            & (bounded["_season_year"] <= source_season)
        ].drop(columns=["_season_year"])
        snapshot["team_season_stats"] = bounded.reset_index(drop=True)

    relations = snapshot.get("team_player_relations", pd.DataFrame())
    players = snapshot.get("players")
    if players is not None and not players.empty:
        players = players.copy()
        for index, row in players.iterrows():
            pid = _to_int(row["id"])
            players.at[index, "position"] = position_as_of(
                pid,
                relations,
                source_season,
                str(row.get("position", "PG") or "PG"),
            )
        snapshot["players"] = players
        snapshot["player_dict"] = {
            _to_int(row["id"]): row.to_dict()
            for _, row in players.iterrows()
        }

    snapshot["_as_of_season"] = source_season
    return snapshot


class AsOfPositionPerformanceModel(SeasonAheadPerformanceModel):
    """Season-ahead XGBoost model using source-season roster positions."""

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
            position = position_as_of(
                _to_int(stat_row.get("player_id")),
                self._position_relations
                if self._position_relations is not None
                else pd.DataFrame(),
                season_year(stat_row.get("season")),
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
    """Exact model implementation used by every production entry point."""

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
        """Finite-sample split-conformal interval for final ensemble errors."""
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
        rank = min(
            len(residuals),
            int(math.ceil((len(residuals) + 1) * coverage)),
        )
        quantile = float(np.sort(residuals)[rank - 1])
        self._conformal_q_lo = quantile
        self._conformal_q_hi = quantile
        self._conformal_nominal_coverage = coverage

    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit every production component strictly before calibration season."""
        from basketball_ai.features.team_features import calibrate_style_bounds
        from basketball_ai.models.age_curve import reset_fitted_params

        # Age-curve parameters are module globals in the legacy implementation.
        # Production t+1 does not empirically fit them until a row-level historical
        # age contract is available; reset prevents state from another run/fold.
        reset_fitted_params()

        metrics = self.perf_model.train(data)
        calibration_season = int(
            self.perf_model._split_metadata["calibration_season"]
        )
        fit_data = build_historical_snapshot(data, calibration_season - 1)

        calibrate_style_bounds(fit_data)
        self.compat_model.train(fit_data)
        self._calibrate_mpg_baseline(fit_data)
        self._calibrate_league_factors(fit_data)

        try:
            from basketball_ai.monitoring.drift import capture_reference

            self._drift_reference = capture_reference(fit_data)
        except Exception:
            self._drift_reference = None

        calibration_records = evaluate_target_season(
            self,
            data,
            calibration_season,
        )
        if len(calibration_records) < 10:
            raise RuntimeError(
                "At least 10 final-ensemble calibration predictions are required"
            )
        residuals = [
            abs(row["prediction"] - row["actual"])
            for row in calibration_records
        ]
        self._calibrate_conformal(residuals)
        if self._conformal_q_hi is None:
            raise RuntimeError("Final-ensemble conformal calibration failed")

        calibrated_records = evaluate_target_season(
            self,
            data,
            calibration_season,
        )
        metrics["final_calibration"] = {
            "target_season": calibration_season,
            "nominal_coverage": self._conformal_nominal_coverage,
            **metric_summary(calibrated_records),
        }
        metrics["compatibility_training_samples"] = int(
            getattr(self.compat_model, "training_samples", 0)
        )
        return metrics

    def _predict_uncached(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: int,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ):
        bounded_data = self._data_with_asof_position(
            player_id,
            data,
            int(season),
        )
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
                float(np.clip(result.predicted_rating - half_width, 0.0, 10.0)),
                3,
            )
            result.confidence_high = round(
                float(np.clip(result.predicted_rating + half_width, 0.0, 10.0)),
                3,
            )
        return result

    def save(
        self,
        directory: str = "models_saved",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist estimator artifacts plus data-derived runtime calibration."""
        super().save(directory, metrics)
        from basketball_ai.features.team_features import _style_bounds

        joblib.dump(
            {
                "style_bounds": dict(_style_bounds),
                "conformal_nominal_coverage": self._conformal_nominal_coverage,
            },
            str(Path(directory) / _STATE_FILE),
        )

    def load(self, directory: str = "models_saved") -> None:
        """Restore the same data-derived state used when the run was trained."""
        from basketball_ai.features.team_features import _style_bounds
        from basketball_ai.models.age_curve import reset_fitted_params

        reset_fitted_params()
        super().load(directory)
        state_path = Path(directory) / _STATE_FILE
        if not state_path.is_file():
            raise RuntimeError(
                f"Strict production state is missing: {state_path}"
            )
        state = joblib.load(str(state_path))
        style_bounds = state.get("style_bounds")
        if not isinstance(style_bounds, dict) or not style_bounds:
            raise RuntimeError("Strict production style bounds are missing")
        _style_bounds.clear()
        _style_bounds.update(style_bounds)
        nominal = state.get("conformal_nominal_coverage")
        self._conformal_nominal_coverage = (
            float(nominal) if nominal is not None else None
        )


def evaluate_target_season(
    ensemble: StrictProductionEnsembleModel,
    data: Dict[str, Any],
    target_season: int,
) -> List[Dict[str, Any]]:
    """Evaluate exactly what production would have served before target season."""
    target_season = int(target_season)
    source_season = target_season - 1
    snapshot = build_historical_snapshot(data, source_season)
    source_stats = snapshot["player_stats"].copy()
    source_stats["_season_year"] = source_stats["season"].map(
        lambda value: season_year(value) if pd.notna(value) else np.nan
    )
    target_stats = data["player_stats"].copy()
    target_stats["_season_year"] = target_stats["season"].map(
        lambda value: season_year(value) if pd.notna(value) else np.nan
    )
    target_stats = target_stats[target_stats["_season_year"] == target_season]

    records: List[Dict[str, Any]] = []
    ensemble.clear_cache()
    for _, target in target_stats.iterrows():
        if pd.isna(target.get("player_id")) or pd.isna(target.get("team_id")):
            continue
        pid = _to_int(target["player_id"])
        tid = _to_int(target["team_id"])
        if pid not in snapshot.get("player_dict", {}) or tid not in snapshot.get("team_dict", {}):
            continue
        prior = source_stats[
            (source_stats["player_id"].map(_to_int) == pid)
            & (source_stats["_season_year"] == source_season)
        ]
        if prior.empty:
            continue
        competition = str(target.get("competition", "RS") or "RS").upper()
        try:
            result = ensemble.predict(
                pid,
                tid,
                snapshot,
                season=source_season,
                competition=competition,
            )
        except (ValueError, RuntimeError):
            continue
        records.append(
            {
                "player_id": pid,
                "team_id": tid,
                "league_id": target.get("league_id"),
                "target_season": target_season,
                "competition": competition,
                "actual": float(target["rating"]),
                "prediction": float(result.predicted_rating),
                "base_prediction": float(result.base_rating),
                "confidence_low": float(result.confidence_low),
                "confidence_high": float(result.confidence_high),
                "persistence_prediction": float(prior.iloc[-1]["rating"]),
                "position": position_as_of(
                    pid,
                    snapshot.get("team_player_relations", pd.DataFrame()),
                    source_season,
                    str(snapshot["player_dict"][pid].get("position", "PG")),
                ),
                "age": int(snapshot["player_dict"][pid].get("age", 0) or 0) + 1,
            }
        )
    ensemble.clear_cache()
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

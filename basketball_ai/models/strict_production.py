"""Strict production model contract shared by training, backtest and API runtime.

Production callers use this module so model fitting, historical evaluation,
batch publishing and online inference all execute the same leakage-safe,
competition-aware path.
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
from basketball_ai.models.competition_training import (
    CompetitionSeasonAheadPerformanceModel,
    CompetitionTemporalCompatibilityModel,
    apply_competition_encoding,
    normalize_competition,
    scope_prediction_context,
)
from basketball_ai.models.production_training import (
    ProductionEnsembleModel,
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
_ROLE_COLUMNS = ("ruolo_combinato", "ruolo_offensivo", "ruolo_difensivo")


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

    teams = snapshot.get("teams")
    if teams is not None and not teams.empty:
        teams = teams.copy()
        if "playing_style" in teams.columns:
            teams["playing_style"] = ""
        if "formation" in teams.columns:
            teams["formation"] = ""
        snapshot["teams"] = teams
        snapshot["team_dict"] = {
            _to_int(row["id"]): row.to_dict()
            for _, row in teams.iterrows()
        }

    snapshot["_as_of_season"] = source_season
    return snapshot


class AsOfPositionPerformanceModel(CompetitionSeasonAheadPerformanceModel):
    """Competition-aware XGBoost using source-season roster positions."""

    _position_relations: Optional[pd.DataFrame] = None
    _role_vocabulary_cutoff: Optional[int] = None

    @staticmethod
    def _allowed_role_vocabulary_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Hide role labels that first appear after the allowed source horizon."""
        stats = data["player_stats"].copy()
        stats["_season_year"] = stats["season"].map(
            lambda value: season_year(value) if pd.notna(value) else np.nan
        )
        target_years: set[int] = set()
        grouping = ["player_id"]
        if "league_id" in stats.columns:
            grouping.append("league_id")
        if "competition" in stats.columns:
            grouping.append("competition")
        for _, group in stats.dropna(subset=["_season_year"]).groupby(grouping):
            years = {int(value) for value in group["_season_year"].tolist()}
            target_years.update(year + 1 for year in years if year + 1 in years)
        ordered_targets = sorted(target_years)
        if len(ordered_targets) < 2:
            clean = stats.drop(columns=["_season_year"])
            return {**data, "player_stats": clean}

        cutoff = ordered_targets[-2]
        future_mask = stats["_season_year"] > cutoff
        for column in _ROLE_COLUMNS:
            if column in stats.columns:
                stats.loc[future_mask, column] = ""
        clean = stats.drop(columns=["_season_year"])
        safe_data = dict(data)
        safe_data["player_stats"] = clean
        safe_data["_role_vocabulary_cutoff"] = cutoff
        return safe_data

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        self._position_relations = data.get("team_player_relations")
        safe_data = self._allowed_role_vocabulary_data(data)
        self._role_vocabulary_cutoff = safe_data.get("_role_vocabulary_cutoff")
        return super().prepare_features(
            safe_data,
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
    """Exact competition-aware model implementation used by production."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("performance_model", AsOfPositionPerformanceModel())
        kwargs.setdefault(
            "compatibility_model", CompetitionTemporalCompatibilityModel()
        )
        super().__init__(*args, **kwargs)
        self._conformal_nominal_coverage: Optional[float] = None
        self._style_bounds_state: Optional[Dict[str, float]] = None
        self._competition_encoding_state: Dict[str, int] = {}
        self._conformal_by_competition: Dict[str, float] = {}
        self._conformal_samples_by_competition: Dict[str, int] = {}

    def _restore_style_bounds(self) -> None:
        if not self._style_bounds_state:
            return
        from basketball_ai.features.team_features import _style_bounds

        _style_bounds.clear()
        _style_bounds.update(self._style_bounds_state)

    def _restore_competition_encoding(self) -> None:
        if not self._competition_encoding_state:
            return
        apply_competition_encoding(self._competition_encoding_state)
        self.perf_model.competition_encoding = dict(self._competition_encoding_state)

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

    @staticmethod
    def _finite_conformal_quantile(
        residuals: List[float],
        coverage: float,
        minimum: int,
    ) -> Optional[float]:
        finite = np.asarray(residuals, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < int(minimum):
            return None
        rank = min(
            len(finite),
            int(math.ceil((len(finite) + 1) * coverage)),
        )
        return float(np.sort(finite)[rank - 1])

    def _calibrate_conformal(self, conformal_residuals: Optional[List[float]]) -> None:
        if not conformal_residuals:
            self._conformal_q_lo = None
            self._conformal_q_hi = None
            self._conformal_nominal_coverage = None
            return
        coverage = float(os.getenv("MODEL_TARGET_INTERVAL_COVERAGE", "0.90"))
        coverage = float(np.clip(coverage, 0.50, 0.999))
        quantile = self._finite_conformal_quantile(
            list(conformal_residuals), coverage, 10
        )
        self._conformal_q_lo = quantile
        self._conformal_q_hi = quantile
        self._conformal_nominal_coverage = coverage if quantile is not None else None

    def _calibrate_competitions(self, records: List[Dict[str, Any]]) -> None:
        self._conformal_by_competition = {}
        self._conformal_samples_by_competition = {}
        coverage = float(self._conformal_nominal_coverage or 0.90)
        minimum = int(os.getenv("MODEL_MIN_COMPETITION_CALIBRATION_SAMPLES", "10"))
        grouped: Dict[str, List[float]] = {}
        for row in records:
            competition = normalize_competition(row.get("competition"))
            grouped.setdefault(competition, []).append(
                abs(float(row["prediction"]) - float(row["actual"]))
            )
        for competition, residuals in grouped.items():
            self._conformal_samples_by_competition[competition] = len(residuals)
            quantile = self._finite_conformal_quantile(
                residuals, coverage, minimum
            )
            if quantile is not None:
                self._conformal_by_competition[competition] = quantile

    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from basketball_ai.features.team_features import calibrate_style_bounds
        from basketball_ai.models.age_curve import reset_fitted_params

        reset_fitted_params()
        metrics = self.perf_model.train(data)
        self._competition_encoding_state = dict(
            getattr(self.perf_model, "competition_encoding", {})
        )
        if not self._competition_encoding_state:
            raise RuntimeError("Training produced no competition vocabulary")
        self._restore_competition_encoding()

        calibration_season = int(
            self.perf_model._split_metadata["calibration_season"]
        )
        fit_data = build_historical_snapshot(data, calibration_season - 1)

        self._style_bounds_state = dict(calibrate_style_bounds(fit_data))
        self._restore_style_bounds()
        self.compat_model.train(fit_data)
        self._calibrate_mpg_baseline(fit_data)
        self._calibrate_league_factors(fit_data)

        try:
            from basketball_ai.monitoring.drift import capture_reference
            self._drift_reference = capture_reference(fit_data)
        except Exception:
            self._drift_reference = None

        calibration_records = evaluate_target_season(
            self, data, calibration_season
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
        self._calibrate_competitions(calibration_records)

        calibrated_records = evaluate_target_season(
            self, data, calibration_season
        )
        metrics["final_calibration"] = {
            "target_season": calibration_season,
            "nominal_coverage": self._conformal_nominal_coverage,
            **metric_summary(calibrated_records),
        }
        metrics["competition_vocabulary"] = dict(
            self._competition_encoding_state
        )
        metrics["competition_calibration_samples"] = dict(
            self._conformal_samples_by_competition
        )
        metrics["competition_specific_intervals"] = sorted(
            self._conformal_by_competition
        )
        metrics["compatibility_training_samples"] = int(
            getattr(self.compat_model, "training_samples", 0)
        )
        metrics["compatibility_samples_by_competition"] = dict(
            getattr(self.compat_model, "training_samples_by_competition", {})
        )
        metrics["role_vocabulary_cutoff"] = getattr(
            self.perf_model, "_role_vocabulary_cutoff", None
        )
        return metrics

    def predict(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: Optional[int] = None,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ):
        competition = normalize_competition(competition)
        if competition not in self._competition_encoding_state:
            raise ValueError(
                f"Competition {competition!r} was not present when this model was trained; "
                "refresh the source contract and retrain before serving it"
            )
        if season is None:
            stats = data.get("player_stats", pd.DataFrame())
            if stats.empty or "season" not in stats.columns:
                raise ValueError("season is required when data has no season")
            season = max(
                season_year(value)
                for value in stats["season"].dropna()
            )
        from basketball_ai.models.age_curve import reset_fitted_params

        reset_fitted_params()
        self._restore_style_bounds()
        self._restore_competition_encoding()
        return super().predict(
            player_id,
            team_id,
            data,
            season=season,
            target_age=target_age,
            competition=competition,
        )

    def _predict_uncached(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: int,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ):
        competition = normalize_competition(competition)
        context_competition = data.get("_prediction_competition")
        if context_competition is not None and normalize_competition(
            context_competition
        ) != competition:
            raise ValueError("Prediction data is scoped to a different competition")

        bounded_data = self._data_with_asof_position(
            player_id, data, int(season)
        )
        result = super()._predict_uncached(
            player_id,
            team_id,
            bounded_data,
            season,
            target_age,
            competition,
        )
        half_width = self._conformal_by_competition.get(
            competition, self._conformal_q_hi
        )
        if half_width is not None:
            result.confidence_low = round(
                float(np.clip(result.predicted_rating - float(half_width), 0.0, 10.0)),
                3,
            )
            result.confidence_high = round(
                float(np.clip(result.predicted_rating + float(half_width), 0.0, 10.0)),
                3,
            )
        return result

    def save(
        self,
        directory: str = "models_saved",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._style_bounds_state:
            raise RuntimeError("Production style calibration state is missing")
        if not self._competition_encoding_state:
            raise RuntimeError("Production competition vocabulary is missing")
        super().save(directory, metrics)
        joblib.dump(
            {
                "style_bounds": dict(self._style_bounds_state),
                "competition_encoding": dict(self._competition_encoding_state),
                "conformal_nominal_coverage": self._conformal_nominal_coverage,
                "conformal_by_competition": dict(self._conformal_by_competition),
                "conformal_samples_by_competition": dict(
                    self._conformal_samples_by_competition
                ),
            },
            str(Path(directory) / _STATE_FILE),
        )

    def load(self, directory: str = "models_saved") -> None:
        from basketball_ai.models.age_curve import reset_fitted_params

        reset_fitted_params()
        super().load(directory)
        state_path = Path(directory) / _STATE_FILE
        if not state_path.is_file():
            raise RuntimeError(f"Strict production state is missing: {state_path}")
        state = joblib.load(str(state_path))
        style_bounds = state.get("style_bounds")
        if not isinstance(style_bounds, dict) or not style_bounds:
            raise RuntimeError("Strict production style bounds are missing")
        competition_encoding = state.get("competition_encoding")
        if not isinstance(competition_encoding, dict) or not competition_encoding:
            raise RuntimeError("Strict production competition vocabulary is missing")
        self._style_bounds_state = {
            str(key): float(value)
            for key, value in style_bounds.items()
        }
        self._competition_encoding_state = {
            str(key): int(value)
            for key, value in competition_encoding.items()
        }
        self._conformal_by_competition = {
            str(key): float(value)
            for key, value in (state.get("conformal_by_competition") or {}).items()
        }
        self._conformal_samples_by_competition = {
            str(key): int(value)
            for key, value in (
                state.get("conformal_samples_by_competition") or {}
            ).items()
        }
        nominal = state.get("conformal_nominal_coverage")
        self._conformal_nominal_coverage = (
            float(nominal) if nominal is not None else None
        )
        self._restore_style_bounds()
        self._restore_competition_encoding()


def evaluate_target_season(
    ensemble: StrictProductionEnsembleModel,
    data: Dict[str, Any],
    target_season: int,
) -> List[Dict[str, Any]]:
    """Evaluate same-league, same-competition t -> t+1 forecasts."""
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
        if (
            pd.isna(target.get("player_id"))
            or pd.isna(target.get("team_id"))
            or pd.isna(target.get("league_id"))
        ):
            continue
        pid = _to_int(target["player_id"])
        tid = _to_int(target["team_id"])
        league_id = _to_int(target["league_id"])
        competition = normalize_competition(target.get("competition"))
        if competition not in ensemble._competition_encoding_state:
            continue
        prior = source_stats[
            (source_stats["player_id"].map(_to_int) == pid)
            & (source_stats["league_id"].map(_to_int) == league_id)
            & (source_stats["competition"].map(normalize_competition) == competition)
            & (source_stats["_season_year"] == source_season)
        ]
        if prior.empty:
            continue
        try:
            scoped = scope_prediction_context(
                snapshot,
                pid,
                tid,
                league_id,
                competition,
                source_season,
            )
            result = ensemble.predict(
                pid,
                tid,
                scoped,
                season=source_season,
                competition=competition,
            )
        except (ValueError, RuntimeError):
            continue
        records.append(
            {
                "player_id": pid,
                "team_id": tid,
                "league_id": league_id,
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
                "competition_support": scoped.get("_competition_support", {}),
            }
        )
    ensemble.clear_cache()
    return records


from basketball_ai.scenarios.engine import (  # noqa: E402
    TrajectoryPoint as _TrajectoryPoint,
    WhatIfEngine as _WhatIfEngine,
    _safe_int as _scenario_safe_int,
)


class StrictWhatIfEngine(_WhatIfEngine):
    """Scenario engine that scopes predictions to an explicit competition."""

    def __init__(
        self,
        ensemble: StrictProductionEnsembleModel,
        data: Dict[str, Any],
    ) -> None:
        self.ensemble = ensemble
        self.data = data

    def predict_in_team(
        self,
        player_id: int,
        target_team_id: int,
        season: Optional[int] = None,
        competition: str = "RS",
        league_id: Optional[int] = None,
    ):
        competition = normalize_competition(competition)
        if season is None:
            stats = self.data.get("player_stats", pd.DataFrame())
            if stats.empty:
                raise ValueError("season is required")
            season = max(season_year(value) for value in stats["season"].dropna())
        scoped = self.data
        if league_id is not None:
            scoped = scope_prediction_context(
                self.data,
                player_id,
                target_team_id,
                league_id,
                competition,
                int(season),
            )
        elif self.data.get("_prediction_league_id") is None:
            raise ValueError("league_id is required for competition-aware prediction")
        return self.ensemble.predict(
            player_id,
            target_team_id,
            scoped,
            season=season,
            competition=competition,
        )

    def predict_age_trajectory(
        self,
        player_id: int,
        season_base: Optional[int] = None,
        age_range: Optional[Tuple[int, int]] = None,
        team_id: Optional[int] = None,
    ) -> List[_TrajectoryPoint]:
        if season_base is None:
            stats = self.data.get("player_stats", pd.DataFrame())
            if stats.empty:
                raise ValueError("season_base is required")
            season_base = max(
                season_year(value) for value in stats["season"].dropna()
            )
        player = self.data["player_dict"].get(_to_int(player_id), {})
        current_age = _scenario_safe_int(player.get("age", 25), 25)
        if team_id is None:
            current_team = player.get("current_team_id")
            if current_team is None:
                raise ValueError("team_id is required when player has no current team")
            team_id = _to_int(current_team)
        low, high = age_range if age_range else (
            max(18, current_age - 4),
            min(40, current_age + 8),
        )
        points: List[_TrajectoryPoint] = []
        competition = normalize_competition(
            self.data.get("_prediction_competition", "RS")
        )
        for age in range(low, high + 1):
            source_season = int(season_base) + (age - current_age)
            prediction = self.ensemble.predict(
                player_id,
                team_id,
                self.data,
                season=source_season,
                target_age=age,
                competition=competition,
            )
            points.append(
                _TrajectoryPoint(
                    age=age,
                    season=source_season + 1,
                    predicted_rating=prediction.predicted_rating,
                    confidence_low=prediction.confidence_low,
                    confidence_high=prediction.confidence_high,
                    age_factor=prediction.age_factor,
                )
            )
        return points


__all__ = [
    "AsOfPositionPerformanceModel",
    "StrictProductionEnsembleModel",
    "StrictWhatIfEngine",
    "build_historical_snapshot",
    "evaluate_target_season",
    "metric_summary",
    "position_as_of",
    "season_year",
]

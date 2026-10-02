"""Player Future Performance Model.

A player-centric, season-ahead multivariate forecaster.  It predicts a vector
of next-season performance rates from information available in the source
season.  The model is deliberately independent from Prediction Model 2.6.0
and from the descriptive BB-Rating engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from basketball_ai.constants import _peak_age, _primary_pos
from basketball_ai.data.loader import _to_int
from basketball_ai.models.competition_training import normalize_competition
from basketball_ai.models.performance_model import _XGB_DEVICE, _canonical_source_value

logger = logging.getLogger(__name__)

FUTURE_PERFORMANCE_VERSION = "1.0"
FEATURE_VERSION = "player-future-performance-v1"
ARTIFACT_FILENAME = "metadata.json"
MODEL_DIRNAME = "models"
MIN_TRAIN_SAMPLES = 40
MIN_OOS_SAMPLES = 20
_PER36_MAX = 60.0

@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    kind: str
    source_column: Optional[str]
    lower: float
    upper: float

TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("pts_per_36", "PTS / 36", "per36", "points", 0.0, _PER36_MAX),
    TargetSpec("ast_per_36", "AST / 36", "per36", "assists", 0.0, _PER36_MAX),
    TargetSpec("reb_per_36", "REB / 36", "per36", "rebounds", 0.0, _PER36_MAX),
    TargetSpec("stl_per_36", "STL / 36", "per36", "steals", 0.0, 10.0),
    TargetSpec("blk_per_36", "BLK / 36", "per36", "blocks", 0.0, 10.0),
    TargetSpec("ts_pct", "TS%", "rate", "ts_pct", 0.0, 1.0),
    TargetSpec("usg_pct", "USG%", "rate", "usg_pct", 0.0, 1.0),
    TargetSpec("tov_pct", "TOV%", "rate", "tov_pct", 0.0, 1.0),
    TargetSpec("ast_pct", "AST%", "rate", "ast_pct", 0.0, 1.0),
    TargetSpec("orb_pct", "ORB%", "rate", "orb_pct", 0.0, 1.0),
    TargetSpec("drb_pct", "DRB%", "rate", "drb_pct", 0.0, 1.0),
    TargetSpec("three_point_pct", "3P%", "rate", "three_point_pct", 0.0, 1.0),
    TargetSpec("ft_pct", "FT%", "rate", "ft_pct", 0.0, 1.0),
    TargetSpec("three_point_attempts_per_36", "3PA / 36", "per36", "three_point_attempts", 0.0, 20.0),
    TargetSpec("free_throw_attempts_per_36", "FTA / 36", "per36", "free_throw_attempts", 0.0, 20.0),
    TargetSpec("minutes_per_game", "Minutes / game", "minutes", "minutes_per_game", 0.0, 48.0),
)

POSITION_FEATURES = ("PG", "SG", "SF", "PF", "C", "hybrid")
COMPETITION_FEATURES = ("RS", "PO", "CUP", "SUPERCUP", "TOT")
HISTORY_FEATURES = (
    ("pts_per_36", "per36", "points"),
    ("ast_per_36", "per36", "assists"),
    ("reb_per_36", "per36", "rebounds"),
    ("stl_per_36", "per36", "steals"),
    ("blk_per_36", "per36", "blocks"),
    ("ts_pct", "rate", "ts_pct"),
    ("usg_pct", "rate", "usg_pct"),
    ("tov_pct", "rate", "tov_pct"),
    ("ast_pct", "rate", "ast_pct"),
    ("orb_pct", "rate", "orb_pct"),
    ("drb_pct", "rate", "drb_pct"),
    ("three_point_pct", "rate", "three_point_pct"),
    ("ft_pct", "rate", "ft_pct"),
    ("three_point_attempts_per_36", "per36", "three_point_attempts"),
    ("free_throw_attempts_per_36", "per36", "free_throw_attempts"),
    ("minutes_per_game", "minutes", "minutes_per_game"),
)

def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")

def _source_year(value: Any) -> int:
    return int(str(value).strip().split("-")[0])

def _safe_rate(column: str, value: Any) -> float:
    return _canonical_source_value(column, value)

def _metric_from_row(row: pd.Series, kind: str, column: str) -> float:
    if kind == "per36":
        minutes = _finite(row.get("minutes_per_game"))
        raw = _finite(row.get(column))
        if not np.isfinite(minutes) or minutes <= 0 or not np.isfinite(raw):
            return float("nan")
        return float(np.clip(raw / minutes * 36.0, 0.0, _PER36_MAX))
    if kind == "rate":
        value = _safe_rate(column, row.get(column))
        return value
    value = _finite(row.get(column))
    if kind == "minutes":
        return value
    return value

def _birth_year_map(players: pd.DataFrame, latest_year: int) -> Dict[int, int]:
    result: Dict[int, int] = {}
    if players is None or players.empty:
        return result
    for row in players.to_dict("records"):
        pid = _to_int(row.get("id"))
        birth_date = row.get("birth_date", row.get("date_of_birth"))
        try:
            result[pid] = int(str(birth_date)[:4])
            continue
        except (TypeError, ValueError):
            pass
        age = _finite(row.get("age"))
        if np.isfinite(age) and 14 <= age <= 44:
            try:
                reference = int(row.get("age_reference_season"))
            except (TypeError, ValueError):
                reference = latest_year
            result[pid] = reference - int(age)
    return result

def _league_key_map(leagues: pd.DataFrame) -> Dict[int, str]:
    if leagues is None or leagues.empty:
        return {}
    result: Dict[int, str] = {}
    for row in leagues.to_dict("records"):
        lid = _to_int(row.get("id"))
        key = str(row.get("league_key", "") or row.get("name", "")).strip().upper()
        if key:
            result[lid] = key
    return result

class PlayerFuturePerformanceModel:
    """Independent season-ahead player performance forecaster."""

    def __init__(self) -> None:
        self.models: Dict[str, XGBRegressor] = {}
        self.feature_names: list[str] = []
        self.target_specs: Dict[str, TargetSpec] = {spec.key: spec for spec in TARGET_SPECS}
        self.uncertainty: Dict[str, Dict[str, float]] = {}
        self.target_metrics: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_trained = False

    @staticmethod
    def _feature_columns(league_values: Iterable[str]) -> list[str]:
        columns = [
            "age", "age_vs_peak", "minutes_per_game", "games_played",
            "starter_pct", "rating", "rating_delta_1", "rating_delta_2",
            "rating_recent_mean", "rating_recent_std", "season_experience",
        ]
        for metric, _, _ in HISTORY_FEATURES:
            columns.extend([
                f"current_{metric}",
                f"hist3_mean_{metric}",
                f"hist3_std_{metric}",
                f"delta_{metric}",
            ])
        columns.extend(f"pos_{value}" for value in POSITION_FEATURES)
        columns.extend(f"competition_{value}" for value in COMPETITION_FEATURES)
        columns.extend(
            f"league_{str(value).upper()}"
            for value in sorted({str(v).strip().upper() for v in league_values if str(v).strip()})
        )
        return columns

    def _build_feature_row(
        self,
        source: pd.Series,
        history: pd.DataFrame,
        *,
        birth_year: Optional[int],
        position: str,
        league_values: Iterable[str],
    ) -> Dict[str, float]:
        source_year = _source_year(source.get("season"))
        age = source_year - birth_year if birth_year is not None else int(round(_peak_age(position)))
        age = float(np.clip(age, 14, 45))
        row: Dict[str, float] = {
            "age": age,
            "age_vs_peak": age - float(_peak_age(position)),
            "minutes_per_game": float(np.clip(_finite(source.get("minutes_per_game")) if np.isfinite(_finite(source.get("minutes_per_game"))) else 0.0, 0.0, 48.0)),
            "games_played": float(max(0.0, _finite(source.get("games_played")) if np.isfinite(_finite(source.get("games_played"))) else 0.0)),
            "starter_pct": (lambda v: v / 100.0 if np.isfinite(v) and abs(v) > 1.0 else v)(_safe_rate("starter_pct", source.get("starter_pct"))) if "starter_pct" in source.index else 0.0,
            "rating": float(np.clip(_finite(source.get("rating")) if np.isfinite(_finite(source.get("rating"))) else 6.5, 0.0, 10.0)),
            "rating_delta_1": 0.0,
            "rating_delta_2": 0.0,
            "rating_recent_mean": 6.5,
            "rating_recent_std": 0.0,
            "season_experience": float(max(0, len(history) - 1)),
        }

        hist_ratings = pd.to_numeric(history.get("rating", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        if hist_ratings.size >= 2:
            row["rating_delta_1"] = float(hist_ratings[-1] - hist_ratings[-2])
        if hist_ratings.size >= 3:
            row["rating_delta_2"] = float(hist_ratings[-1] - hist_ratings[-3])
        if hist_ratings.size:
            recent = hist_ratings[-3:]
            row["rating_recent_mean"] = float(np.mean(recent))
            row["rating_recent_std"] = float(np.std(recent))

        for metric, kind, column in HISTORY_FEATURES:
            values = [
                _metric_from_row(item, kind, column)
                for _, item in history.iterrows()
            ]
            values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
            current = _metric_from_row(source, kind, column)
            recent = values[-3:] if values.size else np.asarray([], dtype=float)
            previous = values[-2] if values.size >= 2 else float("nan")
            row[f"current_{metric}"] = float(current) if np.isfinite(current) else 0.0
            row[f"hist3_mean_{metric}"] = float(np.mean(recent)) if recent.size else row[f"current_{metric}"]
            row[f"hist3_std_{metric}"] = float(np.std(recent)) if recent.size > 1 else 0.0
            row[f"delta_{metric}"] = (
                float(current - previous)
                if np.isfinite(current) and np.isfinite(previous)
                else 0.0
            )

        primary = _primary_pos(position)
        for value in POSITION_FEATURES:
            row[f"pos_{value}"] = 1.0 if (
                primary == value or (value == "hybrid" and primary != position)
            ) else 0.0
        competition = normalize_competition(source.get("competition"))
        for value in COMPETITION_FEATURES:
            row[f"competition_{value}"] = 1.0 if competition == value else 0.0

        current_league = str(source.get("league_key", "") or "").strip().upper()
        for value in sorted({str(v).strip().upper() for v in league_values if str(v).strip()}):
            row[f"league_{value}"] = 1.0 if current_league == value else 0.0
        return row

    def _build_pairs(
        self,
        data: Dict[str, Any],
        *,
        target_seasons: Optional[set[int]] = None,
    ) -> pd.DataFrame:
        stats = data.get("player_stats")
        players = data.get("players")
        if stats is None or stats.empty:
            raise ValueError("player_stats is empty")
        stats = stats.copy()
        stats["_season_year"] = pd.to_numeric(
            stats["season"].astype(str).str.split("-").str[0],
            errors="coerce",
        )
        stats = stats.dropna(subset=["_season_year", "player_id", "league_id", "competition"]).copy()
        stats["_season_year"] = stats["_season_year"].astype(int)
        league_map = _league_key_map(data.get("leagues", pd.DataFrame()))
        if "league_key" not in stats.columns:
            stats["league_key"] = stats["league_id"].map(league_map).fillna(stats["league_id"].astype(str))
        stats["_competition"] = stats["competition"].map(normalize_competition)
        stats = stats.sort_values(["player_id", "league_id", "_competition", "_season_year"]).reset_index(drop=True)

        birth_years = _birth_year_map(
            players,
            int(stats["_season_year"].max()),
        )
        positions = {
            _to_int(row.get("id")): str(row.get("position", "PG") or "PG")
            for row in (players.to_dict("records") if players is not None and not players.empty else [])
        }
        league_values = sorted({
            str(value).strip().upper()
            for value in stats["league_key"].dropna().tolist()
            if str(value).strip()
        })
        rows: list[Dict[str, Any]] = []
        grouped = stats.groupby(["player_id", "league_id", "_competition"], sort=False)
        for (player_id, league_id, competition), group in grouped:
            group = group.sort_values("_season_year").reset_index(drop=True)
            years = group["_season_year"].astype(int).tolist()
            year_to_index = {year: idx for idx, year in enumerate(years)}
            pid = _to_int(player_id)
            position = positions.get(pid, "PG")
            birth_year = birth_years.get(pid)
            for source_index, source_year in enumerate(years):
                target_year = source_year + 1
                if target_seasons is not None and target_year not in target_seasons:
                    continue
                target_index = year_to_index.get(target_year)
                if target_index is None:
                    continue
                source = group.iloc[source_index]
                target = group.iloc[target_index]
                history = group.iloc[:source_index + 1].drop(
                    columns=["_season_year", "_competition"],
                )
                features = self._build_feature_row(
                    source,
                    history,
                    birth_year=birth_year,
                    position=position,
                    league_values=league_values,
                )
                labels = {
                    spec.key: _metric_from_row(target, spec.kind, spec.source_column or "")
                    for spec in TARGET_SPECS
                }
                labels["source_year"] = int(source_year)
                labels["target_year"] = int(target_year)
                labels["player_id"] = pid
                labels["league_id"] = _to_int(league_id)
                labels["league_key"] = str(source.get("league_key", "")).strip().upper()
                labels["competition"] = str(competition)
                rows.append({**features, **{f"__target__{key}": value for key, value in labels.items() if key in self.target_specs}, **{f"__meta__{key}": value for key, value in labels.items() if key not in self.target_specs}})
        if not rows:
            raise ValueError("No consecutive player/league/competition t -> t+1 samples are available")
        # Keep target NaNs intact. Missing target observations must be excluded
        # per metric during training/backtest, never converted to zero.
        return pd.DataFrame(rows)

    @staticmethod
    def _model() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            device=_XGB_DEVICE,
        )

    def _fit_one(self, X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
        if len(y) < MIN_TRAIN_SAMPLES:
            raise ValueError(f"Only {len(y)} training rows for target; minimum is {MIN_TRAIN_SAMPLES}")
        model = self._model()
        model.fit(X, y.to_numpy(dtype=float), verbose=False)
        return model

    @staticmethod
    def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return {
            "n": int(len(actual)),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "mae": float(np.mean(np.abs(predicted - actual))),
            "bias": float(np.mean(predicted - actual)),
            "correlation": (
                float(np.corrcoef(actual, predicted)[0, 1])
                if np.std(actual) > 0 and np.std(predicted) > 0 else None
            ),
        }

    def walk_forward_backtest(
        self,
        data: Dict[str, Any],
        *,
        target_seasons: Optional[Iterable[int]] = None,
        min_train_target_seasons: int = 3,
    ) -> Dict[str, Any]:
        """Expanding walk-forward OOS evaluation by target season."""
        pairs = self._build_pairs(data, target_seasons=set(target_seasons) if target_seasons is not None else None)
        years = sorted(int(v) for v in pairs["__meta__target_year"].unique())
        folds: list[Dict[str, Any]] = []
        per_target: Dict[str, list[dict[str, float]]] = {spec.key: [] for spec in TARGET_SPECS}
        oos_residuals: Dict[str, list[float]] = {spec.key: [] for spec in TARGET_SPECS}
        for target_year in years:
            prior_years = [year for year in years if year < target_year]
            if len(prior_years) < min_train_target_seasons:
                continue
            train = pairs[pairs["__meta__target_year"].astype(int) < target_year]
            test = pairs[pairs["__meta__target_year"].astype(int) == target_year]
            fold_targets: Dict[str, Any] = {}
            fold_valid = False
            for spec in TARGET_SPECS:
                y_train = pd.to_numeric(train[f"__target__{spec.key}"], errors="coerce")
                y_test = pd.to_numeric(test[f"__target__{spec.key}"], errors="coerce")
                train_mask = np.isfinite(y_train.to_numpy(dtype=float))
                test_mask = np.isfinite(y_test.to_numpy(dtype=float))
                if train_mask.sum() < MIN_TRAIN_SAMPLES or test_mask.sum() < MIN_OOS_SAMPLES:
                    continue
                X_train = train.loc[train_mask, self._feature_columns(pairs["__meta__league_key"])].fillna(0.0)
                X_test = test.loc[test_mask, X_train.columns].fillna(0.0)
                model = self._fit_one(X_train, y_train.loc[train_mask])
                prediction = np.clip(model.predict(X_test.to_numpy()), spec.lower, spec.upper)
                actual_values = y_test.loc[test_mask].to_numpy(dtype=float)
                metrics = self._metrics(actual_values, prediction)
                baseline = pd.to_numeric(test.loc[test_mask, f"__target__{spec.key}"], errors="coerce").to_numpy(dtype=float)
                source_values = pd.to_numeric(test.loc[test_mask, f"current_{spec.key}"], errors="coerce").to_numpy(dtype=float)
                metrics["baseline_persistence_rmse"] = float(np.sqrt(np.mean((source_values - baseline) ** 2))) if len(baseline) else float("nan")
                metrics["model_gain_vs_persistence_rmse"] = metrics["baseline_persistence_rmse"] - metrics["rmse"] if np.isfinite(metrics["baseline_persistence_rmse"]) else float("nan")
                oos_residuals[spec.key].extend(np.abs(prediction - actual_values).tolist())
                fold_targets[spec.key] = metrics
                per_target[spec.key].append(metrics)
                fold_valid = True
            if fold_valid:
                folds.append({
                    "target_season": int(target_year),
                    "train_through_target": int(target_year - 1),
                    "n_test_rows": int(len(test)),
                    "targets": fold_targets,
                })
        summary: Dict[str, Any] = {}
        for spec in TARGET_SPECS:
            values = per_target[spec.key]
            if not values:
                continue
            summary[spec.key] = {
                "folds": len(values),
                "n_oos": int(sum(v["n"] for v in values)),
                "rmse_mean": float(np.average([v["rmse"] for v in values], weights=[max(1, v["n"]) for v in values])),
                "mae_mean": float(np.average([v["mae"] for v in values], weights=[max(1, v["n"]) for v in values])),
                "bias_mean": float(np.average([v["bias"] for v in values], weights=[max(1, v["n"]) for v in values])),
                "baseline_persistence_rmse_mean": float(np.average([v["baseline_persistence_rmse"] for v in values], weights=[max(1, v["n"]) for v in values])),
                "model_gain_vs_persistence_rmse_mean": float(np.average([v["model_gain_vs_persistence_rmse"] for v in values], weights=[max(1, v["n"]) for v in values])),
            }
        return {
            "status": "validated_oos" if folds else "insufficient_data",
            "method": "expanding_walk_forward_player_season",
            "target_seasons": years,
            "folds": folds,
            "summary": summary,
            "uncertainty_by_target": {
                key: {
                    "p50": float(np.quantile(values, 0.50)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                    "n_oos": int(len(values)),
                }
                for key, values in oos_residuals.items()
                if len(values) >= MIN_OOS_SAMPLES
            },
        }

    def fit(
        self,
        data: Dict[str, Any],
        *,
        target_seasons: Optional[Iterable[int]] = None,
        backtest: bool = True,
    ) -> Dict[str, Any]:
        target_set = set(int(value) for value in target_seasons) if target_seasons is not None else None
        pairs = self._build_pairs(data, target_seasons=target_set)
        league_values = sorted(set(str(v).strip().upper() for v in pairs["__meta__league_key"].tolist() if str(v).strip()))
        self.feature_names = self._feature_columns(league_values)

        backtest_report = (
            self.walk_forward_backtest(
                data,
                target_seasons=sorted(target_set) if target_set is not None else None,
            )
            if backtest else {"status": "disabled", "folds": [], "summary": {}}
        )

        training_targets: Dict[str, Dict[str, Any]] = {}
        all_residuals: list[float] = []
        for spec in TARGET_SPECS:
            y = pd.to_numeric(pairs[f"__target__{spec.key}"], errors="coerce")
            mask = np.isfinite(y.to_numpy(dtype=float))
            if mask.sum() < MIN_TRAIN_SAMPLES:
                continue
            X = pairs.loc[mask, self.feature_names].fillna(0.0)
            model = self._fit_one(X, y.loc[mask])
            self.models[spec.key] = model
            predictions = model.predict(X.to_numpy())
            residuals = np.abs(predictions - y.loc[mask].to_numpy(dtype=float))
            q = {
                "p50": float(np.quantile(residuals, 0.50)),
                "p75": float(np.quantile(residuals, 0.75)),
                "p90": float(np.quantile(residuals, 0.90)),
            }
            oos_q = (backtest_report.get("uncertainty_by_target") or {}).get(spec.key)
            self.uncertainty[spec.key] = dict(oos_q) if isinstance(oos_q, dict) else q
            all_residuals.extend(residuals.tolist())
            training_targets[spec.key] = {
                "n_training": int(mask.sum()),
                "available": True,
            }
            bt_summary = backtest_report.get("summary", {}).get(spec.key, {})
            self.target_metrics[spec.key] = {
                **training_targets[spec.key],
                "oos": bt_summary,
            }

        if not self.models:
            raise ValueError("No future-performance targets have enough training data")

        self.metadata = {
            "status": "fitted",
            "future_performance_version": FUTURE_PERFORMANCE_VERSION,
            "feature_version": FEATURE_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_target_seasons": sorted({
                int(v) for v in pairs["__meta__target_year"].unique()
            }),
            "excluded_target_seasons": [],
            "n_pairs": int(len(pairs)),
            "n_players": int(pairs["__meta__player_id"].nunique()),
            "leagues": league_values,
            "competitions": sorted(pairs["__meta__competition"].unique().tolist()),
            "targets": training_targets,
            "target_metrics": self.target_metrics,
            "uncertainty_by_target": self.uncertainty,
            "uncertainty_by_target": self.uncertainty,
            "uncertainty_global": {
                "p50": float(np.quantile(all_residuals, 0.50)) if all_residuals else None,
                "p75": float(np.quantile(all_residuals, 0.75)) if all_residuals else None,
                "p90": float(np.quantile(all_residuals, 0.90)) if all_residuals else None,
            },
            "backtest": backtest_report,
        }
        self.is_trained = True
        return self.metadata

    def predict_player(
        self,
        data: Dict[str, Any],
        *,
        player_id: int,
        league_key: str,
        season: int,
        competition: str = "RS",
    ) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Player Future Performance Model is not loaded")
        stats = data.get("player_stats")
        if stats is None or stats.empty:
            raise ValueError("player_stats is empty")
        competition = normalize_competition(competition)
        league_key = str(league_key).strip().upper()

        frame = stats.copy()
        if "league_key" not in frame.columns:
            league_map = _league_key_map(data.get("leagues", pd.DataFrame()))
            frame["league_key"] = frame["league_id"].map(league_map).fillna(frame["league_id"].astype(str))
        frame["_season_year"] = pd.to_numeric(frame["season"].astype(str).str.split("-").str[0], errors="coerce")
        frame["_competition"] = frame["competition"].map(normalize_competition)
        context = frame[
            (frame["player_id"].astype(str) == str(player_id))
            & (frame["league_key"].astype(str).str.upper() == league_key)
            & (frame["_competition"] == competition)
            & (frame["_season_year"] <= int(season))
        ].sort_values("_season_year")
        source = context[context["_season_year"] == int(season)]
        if source.empty:
            raise ValueError(
                f"No source-season row for player={player_id}, league={league_key}, "
                f"season={season}, competition={competition}"
            )
        source = source.iloc[-1]
        history = context.drop(columns=["_season_year", "_competition"], errors="ignore")
        players = data.get("players", pd.DataFrame())
        pid_global = None
        player_name = None
        position = "PG"
        birth_year = None
        if players is not None and not players.empty:
            candidates = players[players["id"].astype(str) == str(player_id)]
            if not candidates.empty:
                player_row = candidates.iloc[0]
                pid_global = str(player_row.get("global_id", "") or "") or None
                player_name = str(player_row.get("name", "") or "") or None
                position = str(player_row.get("position", "PG") or "PG")
                birth_year = _birth_year_map(players, int(season)).get(int(player_id))
        league_values = self.metadata.get("leagues", [league_key])
        features = self._build_feature_row(
            source,
            history,
            birth_year=birth_year,
            position=position,
            league_values=league_values,
        )
        X = pd.DataFrame([features], columns=self.feature_names).fillna(0.0)

        predictions: Dict[str, Any] = {}
        for spec in TARGET_SPECS:
            model = self.models.get(spec.key)
            if model is None:
                continue
            prediction = float(np.clip(model.predict(X.to_numpy())[0], spec.lower, spec.upper))
            q = self.uncertainty.get(spec.key) or self.metadata.get("uncertainty_global", {}) or {}
            p90 = float(max(0.0, q.get("p90", 0.0) or 0.0))
            lower = float(np.clip(prediction - p90, spec.lower, spec.upper))
            upper = float(np.clip(prediction + p90, spec.lower, spec.upper))
            predictions[spec.key] = {
                "label": spec.label,
                "value": prediction,
                "p50_abs_error": float(q.get("p50", 0.0) or 0.0),
                "p75_abs_error": float(q.get("p75", 0.0) or 0.0),
                "p90_abs_error": p90,
                "range_p90": {"lower": lower, "upper": upper},
                "oos": self.target_metrics.get(spec.key, {}).get("oos", {}),
            }

        # Derived per-game projections use predicted minutes. They are presented
        # as derived estimates, not separately trained targets.
        mpg = predictions.get("minutes_per_game", {}).get("value")
        if mpg is not None:
            for key in ("pts_per_36", "ast_per_36", "reb_per_36", "stl_per_36", "blk_per_36"):
                item = predictions.get(key)
                if not item:
                    continue
                item["derived_per_game"] = float(item["value"] * mpg / 36.0)

        return {
            "future_performance_version": FUTURE_PERFORMANCE_VERSION,
            "feature_version": FEATURE_VERSION,
            "player_id": int(player_id),
            "player_global_id": pid_global,
            "player_name": player_name,
            "league": league_key,
            "source_season": int(season),
            "target_season": int(season) + 1,
            "competition": competition,
            "targets": predictions,
            "model_status": "production",
        }

    def save(self, directory: str | Path) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        root = Path(directory)
        model_root = root / MODEL_DIRNAME
        model_root.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.metadata)
        manifest["feature_names"] = list(self.feature_names)
        manifest["target_metrics"] = self.target_metrics
        manifest["uncertainty_by_target"] = self.uncertainty
        manifest["target_metrics"] = self.target_metrics
        manifest["uncertainty_by_target"] = self.uncertainty
        manifest["model_files"] = {}
        for key, model in self.models.items():
            path = model_root / f"{key}.joblib"
            joblib.dump(model, path, compress=3)
            manifest["model_files"][key] = str(path.relative_to(root))
        (root / ARTIFACT_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> "PlayerFuturePerformanceModel":
        root = Path(directory)
        metadata_path = root / ARTIFACT_FILENAME
        if not metadata_path.exists():
            raise RuntimeError(f"Future performance metadata missing from {root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "fitted":
            raise RuntimeError("Future performance artifact is not fitted")
        instance = cls()
        instance.metadata = metadata
        instance.feature_names = list(metadata.get("feature_names") or [])
        instance.target_metrics = dict(metadata.get("target_metrics") or {})
        instance.uncertainty = dict(metadata.get("uncertainty_by_target") or {})
        for key, relative in (metadata.get("model_files") or {}).items():
            path = root / str(relative)
            if not path.exists():
                raise RuntimeError(f"Future performance model file missing: {path}")
            instance.models[key] = joblib.load(path)
        if not instance.models:
            raise RuntimeError("No future performance estimators found")
        instance.is_trained = True
        return instance

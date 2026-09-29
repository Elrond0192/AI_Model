"""Leakage-safe production training primitives for AI_Model.

This module deliberately keeps the legacy model classes load-compatible while
making the production training/evaluation path stricter:

* only real consecutive ``t -> t+1`` player-season pairs are training samples;
* train/validation/calibration are blocked by whole target seasons;
* XGBoost is early-stopped on validation, then refit on train+validation with
  the selected tree count;
* player/team compatibility uses only player and team information available
  before the target season;
* final conformal residuals are computed on the *full ensemble output*, not on
  the base XGBoost prediction;
* historical snapshots prevent retrospective API/backtests from reading future
  roster/team/player state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from basketball_ai.constants import (
    LEAGUE_MAX_GAMES_BY_NAME,
    LEAGUE_MAX_GAMES_DEFAULT,
    _peak_age,
)
from basketball_ai.data.loader import _to_int
from basketball_ai.models.compatibility_model import (
    CompatibilityModel,
    _POSITION_BUCKET,
    _style_vector_from_prior,
    _with_prior_player_means,
)
from basketball_ai.models.ensemble import EnsembleModel
from basketball_ai.models.performance_model import (
    FEATURE_COLS,
    METRIC_CATALOG,
    PerformanceModel,
    _XGB_DEVICE,
    _compute_data_signature,
    compute_baselines,
)

logger = logging.getLogger(__name__)


def season_year(value: Any) -> int:
    """Return the numeric season start year used by the production contract."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("season is null")
    return int(str(value).split("-")[0])


def _numeric_seasons(frame: pd.DataFrame) -> pd.Series:
    """Vectorised season-to-year conversion used by every hot training path."""
    raw = frame["season"]
    numeric = pd.to_numeric(raw, errors="coerce")
    missing = numeric.isna() & raw.notna()
    if missing.any():
        numeric = numeric.copy()
        numeric.loc[missing] = pd.to_numeric(
            raw.loc[missing].astype(str).str.split("-", n=1).str[0],
            errors="coerce",
        )
    return numeric


def _model_threads() -> int:
    """Bound CPU parallelism to the container/host capacity."""
    detected = max(1, os.cpu_count() or 1)
    try:
        configured = int(os.getenv("MODEL_CPU_THREADS", str(detected)))
    except ValueError:
        configured = detected
    return max(1, min(configured, detected))


def _xgb(n_estimators: int = 300, *, early_stopping: bool = False) -> XGBRegressor:
    kwargs: Dict[str, Any] = {
        "n_estimators": int(max(1, n_estimators)),
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "device": _XGB_DEVICE,
        "n_jobs": _model_threads(),
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = 50
    return XGBRegressor(**kwargs)

def _feature_shift_diagnostics(model: XGBRegressor, X_raw: pd.DataFrame, X_train: pd.DataFrame,
                               X_val: pd.DataFrame, y_train: np.ndarray, y_val: np.ndarray,
                               top_n: int = 20) -> Dict[str, Any]:
    """Diagnostic-only snapshot of XGB importance and train/validation feature shift."""
    booster = model.get_booster()
    feature_names = list(X_train.columns)
    scores = {kind: booster.get_score(importance_type=kind) for kind in ("gain", "weight", "cover")}
    ranked = sorted(feature_names, key=lambda name: float(scores["gain"].get(name, 0.0)), reverse=True)[:max(1, int(top_n))]
    raw = X_raw.reindex(columns=feature_names)
    rows: List[Dict[str, Any]] = []
    for feature in ranked:
        tr = pd.to_numeric(raw.loc[X_train.index, feature], errors="coerce")
        va = pd.to_numeric(raw.loc[X_val.index, feature], errors="coerce")
        trv, vav = tr.dropna().to_numpy(dtype=float), va.dropna().to_numpy(dtype=float)
        tr_mean, va_mean = (float(np.mean(trv)) if len(trv) else 0.0), (float(np.mean(vav)) if len(vav) else 0.0)
        tr_std, va_std = (float(np.std(trv)) if len(trv) else 0.0), (float(np.std(vav)) if len(vav) else 0.0)

        def corr(series: pd.Series, target: np.ndarray) -> Optional[float]:
            values = series.to_numpy(dtype=float)
            target = np.asarray(target, dtype=float)
            mask = np.isfinite(values) & np.isfinite(target)
            if int(mask.sum()) < 2:
                return None
            xv, yv = values[mask], target[mask]
            if np.std(xv) == 0.0 or np.std(yv) == 0.0:
                return None
            return float(np.corrcoef(xv, yv)[0, 1])

        pooled_std = float(np.sqrt((tr_std ** 2 + va_std ** 2) / 2.0))
        rows.append({
            "feature": feature,
            "gain": float(scores["gain"].get(feature, 0.0)),
            "weight": float(scores["weight"].get(feature, 0.0)),
            "cover": float(scores["cover"].get(feature, 0.0)),
            "train_mean": tr_mean, "train_std": tr_std,
            "train_min": float(np.min(trv)) if len(trv) else 0.0,
            "train_max": float(np.max(trv)) if len(trv) else 0.0,
            "validation_mean": va_mean, "validation_std": va_std,
            "validation_min": float(np.min(vav)) if len(vav) else 0.0,
            "validation_max": float(np.max(vav)) if len(vav) else 0.0,
            "train_missing_pct": float(tr.isna().mean() * 100.0),
            "validation_missing_pct": float(va.isna().mean() * 100.0),
            "train_nunique": int(tr.nunique(dropna=True)),
            "validation_nunique": int(va.nunique(dropna=True)),
            "train_target_correlation": corr(tr, y_train),
            "validation_target_correlation": corr(va, y_val),
            "mean_shift": float(va_mean - tr_mean),
            "standardized_mean_shift": float((va_mean - tr_mean) / pooled_std) if pooled_std > 1e-12 else None,
        })
    return {"top_n": len(rows), "features": rows}


class SeasonAheadPerformanceModel(PerformanceModel):
    """Strict season-ahead XGBoost model used by the production pipeline."""

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
        rating_distributions: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if extra_metrics is None:
            extra_metrics = []
        player_stats = data["player_stats"].copy()
        players = data["players"]
        # FASE H — Metric Rating Engine features (optional, additive).
        # Without rating_distributions the behaviour is exactly as before.
        rating_feature_names: List[str] = []
        if rating_distributions is not None:
            from basketball_ai.metric_rating.features import add_rating_features

            player_stats, rating_feature_names = add_rating_features(
                player_stats, rating_distributions
            )
        self.rating_feature_columns = rating_feature_names
        player_stats["_season_year"] = _numeric_seasons(player_stats)
        player_stats = player_stats.dropna(subset=["_season_year"])
        player_stats["_season_year"] = player_stats["_season_year"].astype(int)

        if split_season is not None:
            cutoff = season_year(split_season)
            player_stats = player_stats[player_stats["_season_year"] <= cutoff]

        duplicated = player_stats.duplicated(["player_id", "_season_year"], keep=False)
        if duplicated.any():
            examples = (
                player_stats.loc[duplicated, ["player_id", "season"]]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                "player_stats must contain one row per player+season; duplicates: "
                f"{examples}"
            )

        def build_encoding(column: str) -> Dict[str, int]:
            if column not in player_stats.columns:
                return {}
            values = sorted(
                {
                    str(value).strip()
                    for value in player_stats[column].dropna()
                    if str(value).strip()
                }
            )
            return {value: index + 1 for index, value in enumerate(values)}

        self.role_encoding = build_encoding("ruolo_combinato")
        self.role_off_encoding = build_encoding("ruolo_offensivo")
        self.role_def_encoding = build_encoding("ruolo_difensivo")

        extra_feature_names: List[str] = []
        for column in extra_metrics:
            info = METRIC_CATALOG.get(column)
            if info is None:
                continue
            name = f"{column}_per_36" if info.use_per36 else f"avg_{column}"
            if name not in extra_feature_names:
                extra_feature_names.append(name)
        feature_names = FEATURE_COLS + extra_feature_names + rating_feature_names

        latest_data_year = int(player_stats["_season_year"].max())
        birth_year_map: Dict[int, int] = {}
        position_map: Dict[int, str] = {}
        for player in players.to_dict("records"):
            pid = _to_int(player["id"])
            position_map[pid] = str(player.get("position", "PG") or "PG")
            birth_date = player.get("birth_date", player.get("date_of_birth"))
            try:
                birth_year_map[pid] = int(str(birth_date)[:4])
                continue
            except (TypeError, ValueError):
                pass
            age = player.get("age")
            try:
                age_value = float(age)
            except (TypeError, ValueError):
                age_value = float("nan")
            if np.isfinite(age_value) and 14 <= age_value <= 44:
                reference_season = player.get("age_reference_season")
                try:
                    reference_year = int(reference_season)
                except (TypeError, ValueError):
                    reference_year = latest_data_year
                birth_year_map[pid] = reference_year - int(age_value)

        league_max_games: Optional[Dict[int, int]] = None
        leagues = data.get("leagues")
        if leagues is not None and not leagues.empty and "id" in leagues.columns:
            if "max_games" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): int(row["max_games"])
                    for row in leagues.to_dict("records")
                    if row.get("max_games") and not pd.isna(row.get("max_games"))
                }
            elif "name" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): LEAGUE_MAX_GAMES_BY_NAME.get(
                        str(row["name"]).strip(), LEAGUE_MAX_GAMES_DEFAULT
                    )
                    for row in leagues.to_dict("records")
                }

        rows: List[Dict[str, float]] = []
        targets: List[float] = []
        target_years: List[int] = []
        source_years: List[int] = []
        skipped_gaps = 0

        for raw_pid, group in player_stats.groupby("player_id", sort=False):
            pid = _to_int(raw_pid)
            birth_year = birth_year_map.get(pid)
            if birth_year is None:
                continue
            position = position_map.get(pid, "PG")
            group = group.sort_values("_season_year").reset_index(drop=True)
            clean_group = group.drop(columns=["_season_year"])
            cumulative_history = self._precompute_history_features(
                clean_group,
                extra_metrics=extra_metrics,
                league_max_games=league_max_games,
            )
            empty_history = clean_group.iloc[:0]
            years = [int(value) for value in group["_season_year"].tolist()]
            year_to_index = {year: index for index, year in enumerate(years)}
            max_year = years[-1]
            for source_index, source_year in enumerate(years):
                target_year = source_year + 1
                target_index = year_to_index.get(target_year)
                if target_index is None:
                    if source_year < max_year:
                        skipped_gaps += 1
                    continue
                source = group.iloc[source_index]
                target = group.iloc[target_index]
                source_age = (
                    source_year - birth_year
                    if birth_year is not None
                    else int(round(_peak_age(position)))
                )
                if source_age < 14 or source_age > 45:
                    continue
                source_clean = source.drop(labels=["_season_year"])
                rows.append(
                    self._build_row(
                        source_clean,
                        source_age,
                        position,
                        empty_history,
                        extra_metrics=extra_metrics,
                        league_max_games=league_max_games,
                        precomputed_history=cumulative_history[source_index],
                    )
                )
                targets.append(float(target["rating"]))
                source_years.append(source_year)
                target_years.append(target_year)

        if not rows:
            raise ValueError("No consecutive t -> t+1 player-season samples are available")

        X_raw = pd.DataFrame(rows, columns=feature_names)
        self._last_feature_frame_raw = X_raw.copy()
        X = X_raw.fillna(0.0)
        y = np.asarray(targets, dtype=float)
        self._last_source_years = source_years
        self._last_season_years = target_years
        self._skipped_gap_pairs = skipped_gaps
        return X, y

    def train(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        cv_folds: int = 0,
        rating_distributions: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if extra_metrics is None:
            extra_metrics = []
        X, y = self.prepare_features(
            data,
            extra_metrics=extra_metrics,
            rating_distributions=rating_distributions,
        )
        self.feature_names = list(X.columns)

        target_years = np.asarray(self._last_season_years, dtype=int)
        sort_idx = np.argsort(target_years, kind="stable")
        X = X.iloc[sort_idx].reset_index(drop=True)
        X_raw = self._last_feature_frame_raw.iloc[sort_idx].reset_index(drop=True)
        self._last_feature_frame_raw = X_raw
        y = y[sort_idx]
        target_years = target_years[sort_idx]
        raw_weights = getattr(self, "_last_sample_weights", None)
        sample_weights = (
            np.asarray(raw_weights, dtype=float)[sort_idx]
            if raw_weights is not None and len(raw_weights) == len(sort_idx)
            else np.ones(len(sort_idx), dtype=float)
        )

        unique_targets = sorted(set(target_years.tolist()))
        if len(unique_targets) < 3:
            raise ValueError(
                "At least three consecutive target seasons are required for "
                "train/validation/calibration"
            )
        validation_season, calibration_season = unique_targets[-2:]
        train_mask = target_years < validation_season
        val_mask = target_years == validation_season
        cal_mask = target_years == calibration_season
        X_train, y_train = X.loc[train_mask], y[train_mask]
        X_val, y_val = X.loc[val_mask], y[val_mask]
        X_cal = X.loc[cal_mask]
        w_train = sample_weights[train_mask]
        w_val = sample_weights[val_mask]
        if X_train.empty or X_val.empty or X_cal.empty:
            raise ValueError("Season-blocked split produced an empty partition")

        # Diagnostic-only snapshot of the native season-ahead target.
        # prepare_features() constructs y directly as
        # target_rating - source_rating; no residual scaling is applied here.
        target_diagnostics = {
            "target_mode": str(getattr(self, "target_mode", "rating")),
            "target_transform": "target_rating_minus_source_rating",
            "target_scaling": False,
            "target_scale": 1.0,
            "target_all_mean": float(np.mean(y)),
            "target_all_std": float(np.std(y)),
            "target_all_min": float(np.min(y)),
            "target_all_max": float(np.max(y)),
            "target_train_mean": float(np.mean(y_train)),
            "target_train_std": float(np.std(y_train)),
            "target_train_min": float(np.min(y_train)),
            "target_train_max": float(np.max(y_train)),
            "target_validation_mean": float(np.mean(y_val)),
            "target_validation_std": float(np.std(y_val)),
            "target_validation_min": float(np.min(y_val)),
            "target_validation_max": float(np.max(y_val)),
            "target_calibration_mean": float(np.mean(y[cal_mask])),
            "target_calibration_std": float(np.std(y[cal_mask])),
            "target_calibration_min": float(np.min(y[cal_mask])),
            "target_calibration_max": float(np.max(y[cal_mask])),
        }

        selector = _xgb(300, early_stopping=True)
        selector.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        train_pred = selector.predict(X_train)
        val_pred = selector.predict(X_val)
        target_diagnostics.update({
            "train_prediction_mean": float(np.mean(train_pred)),
            "train_prediction_std": float(np.std(train_pred)),
            "validation_prediction_mean": float(np.mean(val_pred)),
            "validation_prediction_std": float(np.std(val_pred)),
        })
        feature_shift_diagnostics = _feature_shift_diagnostics(selector, X_raw, X_train, X_val, y_train, y_val, top_n=20)
        best_iteration = getattr(selector, "best_iteration", None)
        selected_trees = (
            int(best_iteration) + 1
            if best_iteration is not None and int(best_iteration) >= 0
            else 300
        )

        train_rmse = float(np.sqrt(np.average((train_pred - y_train) ** 2, weights=w_train)))
        val_rmse = float(np.sqrt(np.average((val_pred - y_val) ** 2, weights=w_val)))
        val_mae = float(np.average(np.abs(val_pred - y_val), weights=w_val))
        ss_res = float(np.sum((val_pred - y_val) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fit_mask = target_years <= validation_season
        X_fit, y_fit = X.loc[fit_mask], y[fit_mask]

        # Diagnostic-only snapshot of the exact feature frame/target used by
        # the final production XGBoost refit. This is consumed by OOS model
        # diagnostics and never affects fitting or inference.
        self._diagnostic_fit_X = X_fit.copy()
        self._diagnostic_fit_X_raw = X_raw.loc[fit_mask].copy()
        self._diagnostic_fit_y = np.asarray(y_fit, dtype=float).copy()

        self.model = _xgb(selected_trees, early_stopping=False)
        self.model.fit(X_fit, y_fit, sample_weight=sample_weights[fit_mask], verbose=False)
        self.is_trained = True

        baselines = compute_baselines(X_train, y_train, X_val, y_val)
        signature = _compute_data_signature(X, y)
        self.data_signature = signature
        self._split_metadata = {
            "train_seasons": unique_targets[:-2],
            "validation_season": validation_season,
            "calibration_season": calibration_season,
            "fit_through_season": validation_season,
        }
        metrics: Dict[str, Any] = {
            "target_diagnostics": target_diagnostics,
            "feature_shift_diagnostics": feature_shift_diagnostics,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "selected_n_estimators": selected_trees,
            "consecutive_pairs": int(len(X)),
            "skipped_non_consecutive_gaps": int(self._skipped_gap_pairs),
            "skipped_low_sample_pairs": int(
                getattr(self, "_skipped_low_sample_pairs", 0)
            ),
            "excluded_descriptive_rows": int(
                getattr(self, "_excluded_descriptive_rows", 0)
            ),
            "forecast_competitions": list(
                getattr(self, "_forecast_competitions", [])
            ),
            "sample_weight_mean": float(np.mean(sample_weights)),
            "data_signature": signature,
            **baselines,
        }
        metrics["xgboost_vs_baseline_delta"] = (
            baselines["baseline_mean_val_rmse"] - val_rmse
        )

        if cv_folds > 1:
            candidate_val_seasons = unique_targets[1:-1]
            candidate_val_seasons = candidate_val_seasons[-int(cv_folds):]
            cv_scores: List[float] = []
            for fold_season in candidate_val_seasons:
                tr = target_years < fold_season
                va = target_years == fold_season
                if not tr.any() or not va.any():
                    continue
                model = _xgb(selected_trees, early_stopping=False)
                model.fit(
                    X.loc[tr],
                    y[tr],
                    sample_weight=sample_weights[tr],
                    verbose=False,
                )
                pred = model.predict(X.loc[va])
                cv_scores.append(
                    float(
                        np.sqrt(
                            np.average(
                                (pred - y[va]) ** 2,
                                weights=sample_weights[va],
                            )
                        )
                    )
                )
            if cv_scores:
                metrics["cv_mean_rmse"] = float(np.mean(cv_scores))
                metrics["cv_std_rmse"] = float(np.std(cv_scores))
                metrics["cv_folds"] = len(cv_scores)

        metrics["lineage"] = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "data_signature": signature,
            **self._split_metadata,
        }
        self._last_metrics = metrics
        return metrics

    def predict_from_features(self, feature_dict: Dict[str, float]) -> float:
        if not self.is_trained:
            raise RuntimeError("Performance model is not trained")
        return super().predict_from_features(feature_dict)


class TemporalCompatibilityModel(CompatibilityModel):
    """KNN compatibility model with strict historical player/team context."""

    def _player_style_through(
        self,
        player_id: int,
        data: Dict[str, Any],
        through_season: Optional[int],
    ) -> np.ndarray:
        pid = _to_int(player_id)
        player = data["player_dict"].get(pid, {})
        position = str(player.get("position", "PG"))
        bucket = _POSITION_BUCKET.get(
            position, _POSITION_BUCKET.get(position.split("/")[0], 0.5)
        )
        stats = data["player_stats"]
        p_stats = stats[stats["player_id"] == pid].copy()
        if not p_stats.empty:
            p_stats["_season_year"] = _numeric_seasons(p_stats)
            if through_season is not None:
                p_stats = p_stats[p_stats["_season_year"] <= int(through_season)]

        def mean(column: str, fallback: float) -> float:
            if p_stats.empty or column not in p_stats.columns:
                return fallback
            values = pd.to_numeric(p_stats[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else fallback

        return np.asarray(
            [
                bucket,
                np.clip(mean("usg_pct", 18.0) / 40.0, 0.0, 1.0),
                np.clip(mean("ts_pct", 0.52), 0.0, 1.0),
                np.clip(mean("points", 12.0) / 40.0, 0.0, 1.0),
                np.clip(mean("three_par", 0.30), 0.0, 1.0),
                np.clip((mean("dbpm", 0.0) + 5.0) / 10.0, 0.0, 1.0),
            ],
            dtype=float,
        )

    @staticmethod
    def _team_row_as_of(
        team_id: int,
        data: Dict[str, Any],
        season: int,
        *,
        strict_before: bool,
    ) -> Optional[Dict[str, Any]]:
        history = data.get("team_season_stats")
        if history is None or history.empty:
            return None
        rows = history[history["team_id"].map(_to_int) == _to_int(team_id)].copy()
        if rows.empty:
            return None
        rows["_season_year"] = _numeric_seasons(rows)
        if strict_before:
            rows = rows[rows["_season_year"] < int(season)]
        else:
            rows = rows[rows["_season_year"] <= int(season)]
        if rows.empty:
            return None
        return rows.sort_values("_season_year").iloc[-1].to_dict()

    def train(self, data: Dict[str, Any]) -> None:
        history = data.get("team_season_stats")
        if history is None or history.empty:
            raise RuntimeError(
                "team_season_stats is required for leakage-free compatibility training"
            )
        stats = data["player_stats"].copy()
        stats["_season_year"] = _numeric_seasons(stats)
        stats = stats.dropna(subset=["_season_year"])
        stats["_season_year"] = stats["_season_year"].astype(int)
        stats = _with_prior_player_means(stats, "_season_year")

        team_history = history.copy()
        team_history["_team_year"] = _numeric_seasons(team_history)
        team_history["_tid_int"] = team_history["team_id"].map(_to_int)
        team_history = team_history.dropna(subset=["_team_year"])
        team_history["_team_year"] = team_history["_team_year"].astype(int)
        requests = (
            stats.loc[stats["team_id"].notna(), ["team_id", "_season_year"]]
            .assign(_tid_int=lambda frame: frame["team_id"].map(_to_int))
            .rename(columns={"_season_year": "_target_year"})
            [["_tid_int", "_target_year"]]
            .drop_duplicates()
            .sort_values(["_target_year", "_tid_int"], kind="stable")
        )
        available_teams = team_history.sort_values(
            ["_team_year", "_tid_int"], kind="stable"
        )
        if requests.empty or available_teams.empty:
            team_lookup: Dict[tuple[int, int], Dict[str, Any]] = {}
        else:
            matched_teams = pd.merge_asof(
                requests,
                available_teams,
                left_on="_target_year",
                right_on="_team_year",
                by="_tid_int",
                direction="backward",
                allow_exact_matches=False,
            )
            team_lookup = {
                (int(row["_tid_int"]), int(row["_target_year"])): row
                for row in matched_teams.to_dict("records")
                if not pd.isna(row.get("_team_year"))
            }

        X_rows: List[np.ndarray] = []
        y_values: List[float] = []
        for stat in stats.to_dict("records"):
            if pd.isna(stat.get("team_id")) or pd.isna(stat.get("player_id")):
                continue
            target_season = int(stat["_season_year"])
            tid = _to_int(stat["team_id"])
            if int(stat.get("_prior_rows", 0)) == 0:
                continue
            team = team_lookup.get((tid, target_season))
            if team is None:
                continue
            player_vector = _style_vector_from_prior(stat, data["player_dict"])
            vector = np.concatenate([player_vector, self._team_style_vector(team)])
            prior_mean = float(stat["_prior_mean_rating"])
            compatibility = float(
                np.clip((float(stat["rating"]) - prior_mean + 1.5) / 3.0, 0.0, 1.0)
            )
            X_rows.append(vector)
            y_values.append(compatibility)

        if not X_rows:
            raise RuntimeError("No leakage-free compatibility samples are available")
        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_values, dtype=float)
        effective_neighbors = max(1, min(self.n_neighbors, len(X)))
        self.knn = KNeighborsRegressor(
            n_neighbors=effective_neighbors,
            metric="euclidean",
            n_jobs=_model_threads(),
        )
        scaled = self.scaler.fit_transform(X)
        self.knn.fit(scaled, y)
        self.is_trained = True
        self.training_samples = len(X)
        logger.info(
            "[TemporalCompatibilityModel] trained on %d leakage-free samples",
            len(X),
        )

    def score(self, player_id: int, team_id: int, data: Dict[str, Any]) -> float:
        if not self.is_trained:
            raise RuntimeError("Compatibility model is not trained")
        as_of = data.get("_as_of_season")
        if as_of is None:
            team = data["team_dict"].get(_to_int(team_id))
            through = None
        else:
            through = int(as_of)
            team = self._team_row_as_of(
                team_id, data, through, strict_before=False
            )
        if team is None:
            raise ValueError("No historical team context is available")
        player_vector = self._player_style_through(player_id, data, through)
        vector = np.concatenate([player_vector, self._team_style_vector(team)])
        scaled = self.scaler.transform(vector.reshape(1, -1))
        return float(np.clip(self.knn.predict(scaled)[0], 0.50, 1.0))


def _adjust_players_for_snapshot(
    players: pd.DataFrame,
    full_stats: pd.DataFrame,
    relations: pd.DataFrame,
    teams_by_id: Dict[int, Dict[str, Any]],
    source_season: int,
) -> pd.DataFrame:
    players = players.copy()
    full_years = _numeric_seasons(full_stats).dropna()
    latest_data_year = int(full_years.max()) if not full_years.empty else source_season
    relation_latest: Dict[int, int] = {}
    if not relations.empty:
        rel = relations.copy()
        rel["_season_year"] = _numeric_seasons(rel)
        rel = rel[rel["_season_year"] <= source_season]
        latest_rel = (
            rel.sort_values(["player_id", "_season_year"])
            .groupby("player_id", sort=False)
            .tail(1)
        )
        relation_latest = {
            _to_int(row["player_id"]): _to_int(row["team_id"])
            for row in latest_rel.to_dict("records")
        }

    ages: List[int] = []
    current_team_ids = players.get(
        "current_team_id", pd.Series([None] * len(players), index=players.index)
    ).tolist()
    current_league_ids = players.get(
        "current_league_id", pd.Series([None] * len(players), index=players.index)
    ).tolist()
    for position, row in enumerate(players.to_dict("records")):
        pid = _to_int(row["id"])
        birth_date = row.get("birth_date", row.get("date_of_birth"))
        position_name = str(row.get("position", "PG") or "PG")
        try:
            age = source_season - int(str(birth_date)[:4])
        except (TypeError, ValueError):
            raw_age = row.get("age")
            try:
                raw_age_value = float(raw_age)
            except (TypeError, ValueError):
                raw_age_value = float("nan")
            if np.isfinite(raw_age_value) and 14 <= raw_age_value <= 44:
                reference_season = row.get("age_reference_season")
                try:
                    reference_year = int(reference_season)
                except (TypeError, ValueError):
                    reference_year = latest_data_year
                age = int(raw_age_value) - (reference_year - source_season)
            else:
                age = int(round(_peak_age(position_name)))
        if not np.isfinite(float(age)) or not (14 <= int(age) <= 45):
            age = int(round(_peak_age(position_name)))
        ages.append(int(age))
        team_id = relation_latest.get(pid)
        if team_id is not None:
            current_team_ids[position] = team_id
            team = teams_by_id.get(team_id, {})
            current_league_ids[position] = team.get("league_id")
    players["age"] = ages
    players["current_team_id"] = current_team_ids
    players["current_league_id"] = current_league_ids
    return players


def build_historical_snapshot(data: Dict[str, Any], source_season: int) -> Dict[str, Any]:
    """Return data as it could have been known at the end of ``source_season``."""
    source_season = int(source_season)
    snapshot: Dict[str, Any] = dict(data)

    stats = data["player_stats"].copy()
    stats["_season_year"] = _numeric_seasons(stats)
    stats = stats[stats["_season_year"] <= source_season].copy()
    stats["season"] = stats["_season_year"].astype(int).astype(str)
    stats = stats.drop(columns=["_season_year"])
    snapshot["player_stats"] = stats.reset_index(drop=True)

    relations = data["team_player_relations"].copy()
    relations["_season_year"] = _numeric_seasons(relations)
    relations = relations[relations["_season_year"] <= source_season].copy()
    relations["season"] = relations["_season_year"].astype(int).astype(str)
    relations = relations.drop(columns=["_season_year"])
    snapshot["team_player_relations"] = relations.reset_index(drop=True)

    team_history = data.get("team_season_stats")
    if team_history is None or team_history.empty:
        raise RuntimeError("team_season_stats is required for historical snapshots")
    hist = team_history.copy()
    hist["_season_year"] = _numeric_seasons(hist)
    hist = hist[hist["_season_year"] <= source_season]
    latest = (
        hist.sort_values(["team_id", "_season_year"])
        .groupby("team_id", as_index=False)
        .tail(1)
    )
    current_team_meta = {
        _to_int(row["id"]): row
        for row in data["teams"].to_dict("records")
        if not pd.isna(row.get("id"))
    }
    team_rows: List[Dict[str, Any]] = []
    for row in latest.to_dict("records"):
        team_id = _to_int(row["team_id"])
        base = dict(current_team_meta.get(team_id, {}))
        base.update(
            {
                "id": team_id,
                "global_id": row.get("global_id", base.get("global_id")),
                "name": row.get("name", base.get("name", f"Team {team_id}")),
                "league_id": _to_int(row.get("league_id")),
                "pace": float(row.get("pace", 75.0)),
                "offensive_rating": float(row.get("offensive_rating", 110.0)),
                "defensive_rating": float(row.get("defensive_rating", 110.0)),
                "three_point_attempt_rate": float(
                    row.get("three_point_attempt_rate", 0.35)
                ),
                "assists_per_game": float(row.get("assists_per_game", 20.0)),
                "star_player_usage": float(row.get("star_player_usage", 0.25)),
                "net_rtg": float(row.get("net_rtg", 0.0)),
                "short_name": row.get("short_name", base.get("short_name", "")),
            }
        )
        base.setdefault("playing_style", "")
        base.setdefault("formation", "")
        base.setdefault("league_tier", 1)
        team_rows.append(base)
    teams = pd.DataFrame(team_rows)
    snapshot["teams"] = teams
    snapshot["team_dict"] = {
        _to_int(row["id"]): row for row in teams.to_dict("records")
    }

    players = _adjust_players_for_snapshot(
        data["players"],
        data["player_stats"],
        relations,
        snapshot["team_dict"],
        source_season,
    )
    snapshot["players"] = players
    snapshot["player_dict"] = {
        _to_int(row["id"]): row for row in players.to_dict("records")
    }
    league_teams: Dict[int, List[int]] = {}
    for team_id, team in snapshot["team_dict"].items():
        league_id = team.get("league_id")
        if league_id is None or pd.isna(league_id):
            continue
        league_teams.setdefault(_to_int(league_id), []).append(team_id)
    snapshot["league_teams"] = league_teams
    snapshot["team_season_stats"] = team_history.copy()
    snapshot["_as_of_season"] = source_season
    return snapshot


def _target_rows(data: Dict[str, Any], target_season: int) -> pd.DataFrame:
    stats = data["player_stats"].copy()
    stats["_season_year"] = _numeric_seasons(stats)
    return stats[stats["_season_year"] == int(target_season)].copy()


def evaluate_target_season(
    ensemble: EnsembleModel,
    data: Dict[str, Any],
    target_season: int,
) -> List[Dict[str, Any]]:
    """Evaluate the exact production ensemble on one untouched target season."""
    target_season = int(target_season)
    source_season = target_season - 1
    snapshot = build_historical_snapshot(data, source_season)
    source_stats = snapshot["player_stats"].copy()
    source_stats["_season_year"] = _numeric_seasons(source_stats)
    source_rows = source_stats[source_stats["_season_year"] == source_season].copy()
    source_rows["_pid_int"] = source_rows["player_id"].map(_to_int)
    prior_rating: Dict[int, Any] = {}
    for pid, rating in zip(source_rows["_pid_int"], source_rows["rating"]):
        prior_rating[int(pid)] = rating

    records: List[Dict[str, Any]] = []
    ensemble.clear_cache()

    for target in _target_rows(data, target_season).to_dict("records"):
        if pd.isna(target.get("player_id")) or pd.isna(target.get("team_id")):
            continue
        pid = _to_int(target["player_id"])
        tid = _to_int(target["team_id"])
        if (
            pid not in snapshot["player_dict"]
            or tid not in snapshot["team_dict"]
            or pid not in prior_rating
        ):
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
                "pre_shrinkage_prediction": float(getattr(result, "_diagnostic_stages", {}).get("pre_shrinkage_prediction", result.predicted_rating)),
                "persistence_shrinkage_alpha": float(getattr(result, "_diagnostic_stages", {}).get("persistence_shrinkage_alpha", 0.0)),
                "base_prediction": float(result.base_rating),
                "confidence_low": float(result.confidence_low),
                "confidence_high": float(result.confidence_high),
                "persistence_prediction": float(prior_rating[pid]),
                "position": str(snapshot["player_dict"][pid].get("position", "")),
                "age": int(snapshot["player_dict"][pid].get("age", 0) or 0) + 1,
            }
        )
    ensemble.clear_cache()
    return records


def metric_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize predictions with optional prediction-interval diagnostics.

    Diagnostics such as compatibility compare raw base/adjusted predictions and
    therefore do not carry confidence intervals.  Core error metrics must remain
    available for those records instead of requiring production-only interval
    fields.
    """
    rows = list(records)
    if not rows:
        return {}
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = pred - actual
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))

    result: Dict[str, float] = {
        "n": int(len(rows)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "bias": float(np.mean(error)),
    }

    interval_rows = [
        row
        for row in rows
        if row.get("confidence_low") is not None
        and row.get("confidence_high") is not None
    ]
    if interval_rows:
        coverage = np.mean(
            [
                float(row["confidence_low"])
                <= float(row["actual"])
                <= float(row["confidence_high"])
                for row in interval_rows
            ]
        )
        widths = np.asarray(
            [
                float(row["confidence_high"]) - float(row["confidence_low"])
                for row in interval_rows
            ],
            dtype=float,
        )
        result["interval_coverage"] = float(coverage)
        result["interval_mean_width"] = float(np.mean(widths))

    return result


class ProductionEnsembleModel(EnsembleModel):
    """Ensemble whose training/calibration contract matches production inference."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("performance_model", SeasonAheadPerformanceModel())
        kwargs.setdefault("compatibility_model", TemporalCompatibilityModel())
        super().__init__(*args, **kwargs)

    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self.perf_model.train(data)
        calibration_season = int(self.perf_model._split_metadata["calibration_season"])
        fit_data = build_historical_snapshot(data, calibration_season - 1)

        self.compat_model.train(fit_data)
        self._calibrate_mpg_baseline(fit_data)
        self._calibrate_league_factors(fit_data)
        try:
            from basketball_ai.monitoring.drift import capture_reference

            self._drift_reference = capture_reference(fit_data)
        except Exception as exc:
            logger.warning("Could not capture drift reference: %s", exc)
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
        self._conformal_q_lo = None
        self._conformal_q_hi = None
        self._calibrate_conformal(residuals)
        calibrated_records = evaluate_target_season(
            self, data, calibration_season
        )
        summary = metric_summary(calibrated_records)
        metrics["final_calibration"] = {
            "target_season": calibration_season,
            **summary,
        }
        metrics["compatibility_training_samples"] = int(
            getattr(self.compat_model, "training_samples", 0)
        )
        return metrics

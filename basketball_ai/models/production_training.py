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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from basketball_ai.constants import (
    LEAGUE_MAX_GAMES_BY_NAME,
    LEAGUE_MAX_GAMES_DEFAULT,
)
from basketball_ai.data.loader import _to_int
from basketball_ai.models.compatibility_model import (
    CompatibilityModel,
    _POSITION_BUCKET,
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
    return frame["season"].map(lambda value: season_year(value) if pd.notna(value) else np.nan)


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
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = 50
    return XGBRegressor(**kwargs)


class SeasonAheadPerformanceModel(PerformanceModel):
    """Strict season-ahead XGBoost model used by the production pipeline."""

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if extra_metrics is None:
            extra_metrics = []
        player_stats = data["player_stats"].copy()
        players = data["players"]
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
        feature_names = FEATURE_COLS + extra_feature_names

        latest_data_year = int(player_stats["_season_year"].max())
        birth_year_map: Dict[int, int] = {}
        position_map: Dict[int, str] = {}
        for _, player in players.iterrows():
            pid = _to_int(player["id"])
            position_map[pid] = str(player.get("position", "PG") or "PG")
            birth_date = player.get("birth_date", player.get("date_of_birth"))
            try:
                birth_year_map[pid] = int(str(birth_date)[:4])
                continue
            except (TypeError, ValueError):
                pass
            age = player.get("age")
            if age is not None and not pd.isna(age):
                birth_year_map[pid] = latest_data_year - int(age)

        league_max_games: Optional[Dict[int, int]] = None
        leagues = data.get("leagues")
        if leagues is not None and not leagues.empty and "id" in leagues.columns:
            if "max_games" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): int(row["max_games"])
                    for _, row in leagues.iterrows()
                    if row.get("max_games") and not pd.isna(row.get("max_games"))
                }
            elif "name" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): LEAGUE_MAX_GAMES_BY_NAME.get(
                        str(row["name"]).strip(), LEAGUE_MAX_GAMES_DEFAULT
                    )
                    for _, row in leagues.iterrows()
                }

        rows: List[Dict[str, float]] = []
        targets: List[float] = []
        target_years: List[int] = []
        source_years: List[int] = []
        skipped_gaps = 0

        for raw_pid, group in player_stats.groupby("player_id"):
            pid = _to_int(raw_pid)
            birth_year = birth_year_map.get(pid)
            if birth_year is None:
                continue
            position = position_map.get(pid, "PG")
            group = group.sort_values("_season_year").reset_index(drop=True)
            by_year = {
                int(row["_season_year"]): row
                for _, row in group.iterrows()
            }
            years = sorted(by_year)
            for source_year in years:
                target_year = source_year + 1
                if target_year not in by_year:
                    if any(year > source_year for year in years):
                        skipped_gaps += 1
                    continue
                source = by_year[source_year]
                target = by_year[target_year]
                source_age = source_year - birth_year
                if source_age < 14 or source_age > 44:
                    continue
                history = group[group["_season_year"] <= source_year].drop(
                    columns=["_season_year"]
                )
                source_clean = source.drop(labels=["_season_year"])
                rows.append(
                    self._build_row(
                        source_clean,
                        source_age,
                        position,
                        history,
                        extra_metrics=extra_metrics,
                        league_max_games=league_max_games,
                    )
                )
                targets.append(float(target["rating"]))
                source_years.append(source_year)
                target_years.append(target_year)

        if not rows:
            raise ValueError("No consecutive t -> t+1 player-season samples are available")

        X = pd.DataFrame(rows, columns=feature_names).fillna(0.0)
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
    ) -> Dict[str, Any]:
        if extra_metrics is None:
            extra_metrics = []
        X, y = self.prepare_features(data, extra_metrics=extra_metrics)
        self.feature_names = list(X.columns)

        target_years = np.asarray(self._last_season_years, dtype=int)
        sort_idx = np.argsort(target_years, kind="stable")
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y[sort_idx]
        target_years = target_years[sort_idx]

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
        if X_train.empty or X_val.empty or X_cal.empty:
            raise ValueError("Season-blocked split produced an empty partition")

        selector = _xgb(300, early_stopping=True)
        selector.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_pred = selector.predict(X_train)
        val_pred = selector.predict(X_val)
        best_iteration = getattr(selector, "best_iteration", None)
        selected_trees = (
            int(best_iteration) + 1
            if best_iteration is not None and int(best_iteration) >= 0
            else 300
        )

        train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        ss_res = float(np.sum((val_pred - y_val) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fit_mask = target_years <= validation_season
        X_fit, y_fit = X.loc[fit_mask], y[fit_mask]
        self.model = _xgb(selected_trees, early_stopping=False)
        self.model.fit(X_fit, y_fit, verbose=False)
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
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "selected_n_estimators": selected_trees,
            "consecutive_pairs": int(len(X)),
            "skipped_non_consecutive_gaps": int(self._skipped_gap_pairs),
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
                model.fit(X.loc[tr], y[tr], verbose=False)
                pred = model.predict(X.loc[va])
                cv_scores.append(float(np.sqrt(np.mean((pred - y[va]) ** 2))))
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
        X_rows: List[np.ndarray] = []
        y_values: List[float] = []
        for _, stat in stats.iterrows():
            if pd.isna(stat.get("team_id")) or pd.isna(stat.get("player_id")):
                continue
            target_season = int(stat["_season_year"])
            pid = _to_int(stat["player_id"])
            tid = _to_int(stat["team_id"])
            prior = stats[
                (stats["player_id"].map(_to_int) == pid)
                & (stats["_season_year"] < target_season)
            ]
            if prior.empty:
                continue
            team = self._team_row_as_of(
                tid, data, target_season, strict_before=True
            )
            if team is None:
                continue
            player_vector = self._player_style_through(
                pid, data, target_season - 1
            )
            vector = np.concatenate([player_vector, self._team_style_vector(team)])
            prior_mean = float(pd.to_numeric(prior["rating"], errors="coerce").mean())
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
            n_neighbors=effective_neighbors, metric="euclidean"
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
    latest_data_year = max(
        (season_year(value) for value in full_stats["season"].dropna()),
        default=source_season,
    )
    relation_latest: Dict[int, int] = {}
    if not relations.empty:
        rel = relations.copy()
        rel["_season_year"] = _numeric_seasons(rel)
        rel = rel[rel["_season_year"] <= source_season]
        for pid, group in rel.groupby("player_id"):
            last = group.sort_values("_season_year").iloc[-1]
            relation_latest[_to_int(pid)] = _to_int(last["team_id"])

    for index, row in players.iterrows():
        pid = _to_int(row["id"])
        birth_date = row.get("birth_date", row.get("date_of_birth"))
        try:
            age = source_season - int(str(birth_date)[:4])
        except (TypeError, ValueError):
            raw_age = row.get("age")
            age = (
                int(raw_age) - (latest_data_year - source_season)
                if raw_age is not None and not pd.isna(raw_age)
                else 26
            )
        players.at[index, "age"] = max(14, min(45, int(age)))
        team_id = relation_latest.get(pid)
        if team_id is not None:
            players.at[index, "current_team_id"] = team_id
            team = teams_by_id.get(team_id, {})
            players.at[index, "current_league_id"] = team.get("league_id")
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
        _to_int(row["id"]): row.to_dict()
        for _, row in data["teams"].iterrows()
        if not pd.isna(row.get("id"))
    }
    team_rows: List[Dict[str, Any]] = []
    for _, row in latest.iterrows():
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
        _to_int(row["id"]): row.to_dict() for _, row in teams.iterrows()
    }

    players = _adjust_players_for_snapshot(
        data["players"], data["player_stats"], relations,
        snapshot["team_dict"], source_season,
    )
    snapshot["players"] = players
    snapshot["player_dict"] = {
        _to_int(row["id"]): row.to_dict() for _, row in players.iterrows()
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
    records: List[Dict[str, Any]] = []
    ensemble.clear_cache()

    for _, target in _target_rows(data, target_season).iterrows():
        if pd.isna(target.get("player_id")) or pd.isna(target.get("team_id")):
            continue
        pid = _to_int(target["player_id"])
        tid = _to_int(target["team_id"])
        if pid not in snapshot["player_dict"] or tid not in snapshot["team_dict"]:
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
                "position": str(snapshot["player_dict"][pid].get("position", "")),
                "age": int(snapshot["player_dict"][pid].get("age", 0) or 0) + 1,
            }
        )
    ensemble.clear_cache()
    return records


def metric_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = pred - actual
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    coverage = np.mean(
        [
            row["confidence_low"] <= row["actual"] <= row["confidence_high"]
            for row in rows
        ]
    )
    widths = np.asarray(
        [row["confidence_high"] - row["confidence_low"] for row in rows],
        dtype=float,
    )
    return {
        "n": int(len(rows)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "bias": float(np.mean(error)),
        "interval_coverage": float(coverage),
        "interval_mean_width": float(np.mean(widths)),
    }


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

"""Strict production model contract shared by training, backtest and API runtime.

Production callers use this module so model fitting, historical evaluation,
batch publishing and online inference all execute the same leakage-safe,
competition-aware path.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import weakref

logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd

from basketball_ai.data.loader import _to_int
from basketball_ai.models.competition_training import (
    CompetitionSeasonAheadPerformanceModel,
    CompetitionTemporalCompatibilityModel,
    apply_competition_encoding,
    forecast_competitions,
    minimum_training_games,
    normalize_competition,
    scope_prediction_context,
)
from basketball_ai.models.production_training import (
    ProductionEnsembleModel,
    _numeric_seasons,
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
_POSITION_INDEX_CACHE: Dict[
    int,
    Tuple[
        weakref.ReferenceType[pd.DataFrame],
        Dict[int, Tuple[np.ndarray, List[str]]],
    ],
] = {}


def _normalise_position(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper().replace(" ", "")
    return text if text in _VALID_POSITIONS else None


def _position_index(
    relations: pd.DataFrame,
) -> Dict[int, Tuple[np.ndarray, List[str]]]:
    """Build/cache player roster-position timelines for O(log n) lookups.

    The cache is keyed by DataFrame object identity rather than ``attrs`` so a
    filtered historical copy can never inherit an index containing future rows.
    """
    if (
        relations is None
        or relations.empty
        or "player_id" not in relations.columns
        or "season" not in relations.columns
        or "role" not in relations.columns
    ):
        return {}

    cache_key = id(relations)
    cached = _POSITION_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0]() is relations:
        return cached[1]

    rows = relations[["player_id", "season", "role"]].copy()
    rows["_pid_int"] = rows["player_id"].map(
        lambda value: _to_int(value) if pd.notna(value) else np.nan
    )
    rows["_season_year"] = _numeric_seasons(rows)
    rows["_position"] = rows["role"].map(_normalise_position)
    rows = rows.dropna(subset=["_pid_int", "_season_year", "_position"])

    index: Dict[int, Tuple[np.ndarray, List[str]]] = {}
    for pid, group in rows.groupby("_pid_int", sort=False):
        ordered = group.sort_values("_season_year", kind="stable")
        index[int(pid)] = (
            ordered["_season_year"].astype(int).to_numpy(),
            ordered["_position"].astype(str).tolist(),
        )

    def cleanup(_reference: object, key: int = cache_key) -> None:
        _POSITION_INDEX_CACHE.pop(key, None)

    _POSITION_INDEX_CACHE[cache_key] = (weakref.ref(relations, cleanup), index)
    return index


def position_as_of(
    player_id: int,
    relations: pd.DataFrame,
    season: int,
    fallback: str = "PG",
) -> str:
    """Return the latest valid roster position known at/before ``season``."""
    timeline = _position_index(relations).get(_to_int(player_id))
    if timeline is None:
        return fallback
    years, positions = timeline
    index = int(np.searchsorted(years, int(season), side="right")) - 1
    return positions[index] if index >= 0 else fallback


def build_historical_snapshot(data: Dict[str, Any], source_season: int) -> Dict[str, Any]:
    """Return a hard-bounded as-of snapshot with no future team/player state."""
    source_season = int(source_season)
    snapshot = _build_historical_snapshot(data, source_season)

    team_history = snapshot.get("team_season_stats")
    if team_history is not None and not team_history.empty:
        bounded = team_history.copy()
        bounded["_season_year"] = _numeric_seasons(bounded)
        bounded = bounded[
            bounded["_season_year"].notna()
            & (bounded["_season_year"] <= source_season)
        ].drop(columns=["_season_year"])
        snapshot["team_season_stats"] = bounded.reset_index(drop=True)

    relations = snapshot.get("team_player_relations", pd.DataFrame())
    _position_index(relations)
    players = snapshot.get("players")
    if players is not None and not players.empty:
        players = players.copy()
        positions = []
        for row in players.to_dict("records"):
            pid = _to_int(row["id"])
            positions.append(
                position_as_of(
                    pid,
                    relations,
                    source_season,
                    str(row.get("position", "PG") or "PG"),
                )
            )
        players["position"] = positions
        snapshot["players"] = players
        snapshot["player_dict"] = {
            _to_int(row["id"]): row
            for row in players.to_dict("records")
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
            _to_int(row["id"]): row
            for row in teams.to_dict("records")
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
        stats["_season_year"] = _numeric_seasons(stats)
        target_years: set[int] = set()
        grouping = ["player_id"]
        if "league_id" in stats.columns:
            grouping.append("league_id")
        if "competition" in stats.columns:
            grouping.append("competition")
        for _, group in stats.dropna(subset=["_season_year"]).groupby(
            grouping, sort=False
        ):
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
        rating_distributions: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        self._position_relations = data.get("team_player_relations")
        if self._position_relations is not None:
            _position_index(self._position_relations)
        safe_data = self._allowed_role_vocabulary_data(data)
        self._role_vocabulary_cutoff = safe_data.get("_role_vocabulary_cutoff")
        return super().prepare_features(
            safe_data,
            extra_metrics=extra_metrics,
            split_season=split_season,
            rating_distributions=rating_distributions,
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



def _league_max_games_map(data: Dict[str, Any]) -> Optional[Dict[int, int]]:
    leagues = data.get("leagues")
    if leagues is None or getattr(leagues, "empty", True) or "id" not in leagues.columns:
        return None
    result: Dict[int, int] = {}
    if "max_games" in leagues.columns:
        for row in leagues.to_dict("records"):
            value = row.get("max_games")
            try:
                if value is not None and not pd.isna(value):
                    result[_to_int(row["id"])] = max(1, int(value))
            except (TypeError, ValueError):
                continue
    return result or None


def _player_age_as_of(player: Dict[str, Any], season: int) -> int:
    for key in ("birth_date", "date_of_birth"):
        value = player.get(key)
        try:
            return int(season) - int(str(value)[:4])
        except (TypeError, ValueError):
            pass
    try:
        return int(player.get("age", 26) or 26)
    except (TypeError, ValueError):
        return 26


def build_base_oof_predictions(
    data: Dict[str, Any],
) -> Dict[Tuple[int, int, str, int], float]:
    """Return leakage-free base-XGBoost predictions for historical transitions.

    The rolling-origin OOF contract is unchanged: for each source season, the
    base model is trained only on data available through that season and then
    predicts source-season feature rows whose next-season outcome exists.

    The implementation avoids the old O(years × players × full_dataset) history
    filtering pattern. Source/target keys are indexed once, only eligible
    histories are grouped per fold, history features are precomputed once per
    player/league/competition, and predictions are batched into one XGBoost call
    per source year.
    """
    stats = data.get("player_stats", pd.DataFrame())
    if stats is None or stats.empty:
        return {}
    required = {"player_id", "league_id", "season", "competition"}
    if not required.issubset(stats.columns):
        return {}

    all_stats = stats.copy()
    all_stats["_season_year"] = _numeric_seasons(all_stats)
    all_stats["_competition"] = all_stats["competition"].map(normalize_competition)
    all_stats = all_stats[
        all_stats["_season_year"].notna()
        & all_stats["_competition"].isin(forecast_competitions())
    ].copy()
    if all_stats.empty:
        return {}

    all_stats["_pid_int"] = all_stats["player_id"].map(_to_int)
    all_stats["_lid_int"] = all_stats["league_id"].map(_to_int)
    season_values = all_stats["_season_year"].astype(int)
    available_years = set(season_values.tolist())
    source_years = sorted(
        {
            int(year)
            for year in available_years
            if int(year) + 1 in available_years
        }
    )
    if not source_years:
        return {}

    min_games = minimum_training_games()
    league_max_games = _league_max_games_map(data)
    predictions: Dict[Tuple[int, int, str, int], float] = {}

    # The key frame for each target season is tiny compared with the full stats
    # table and is reused to avoid rebuilding Python tuple sets repeatedly.
    target_keys_by_year: Dict[int, pd.DataFrame] = {}
    for target_year in source_years:
        target = all_stats.loc[
            season_values.eq(target_year + 1)
            & all_stats["player_id"].notna()
            & all_stats["league_id"].notna(),
            ["_pid_int", "_lid_int", "_competition"],
        ].drop_duplicates()
        if not target.empty:
            target_keys_by_year[target_year] = target

    for source_year in source_years:
        target_keys = target_keys_by_year.get(source_year)
        if target_keys is None or target_keys.empty:
            continue

        source = all_stats.loc[
            season_values.eq(source_year),
            ["_pid_int", "_lid_int", "_competition", "games_played"],
        ].copy()
        if source.empty:
            continue
        source_games = pd.to_numeric(source["games_played"], errors="coerce").fillna(0.0)
        source = source.loc[source_games >= min_games]
        if source.empty:
            continue

        # In the previous implementation a source row was eligible exactly when
        # its (player, league, competition) key appeared in the target season.
        # An inner merge preserves that rule while avoiding a Python tuple-set
        # construction for every target row.
        eligible = source.merge(
            target_keys,
            on=["_pid_int", "_lid_int", "_competition"],
            how="inner",
            sort=False,
            copy=False,
        )
        if eligible.empty:
            continue

        snapshot = build_historical_snapshot(data, source_year)
        try:
            base_model = CompetitionSeasonAheadPerformanceModel()
            base_model.train(snapshot)
        except (RuntimeError, ValueError, KeyError):
            continue

        snapshot_stats = snapshot["player_stats"].copy()
        snapshot_stats["_season_year"] = _numeric_seasons(snapshot_stats)
        snapshot_stats["_competition"] = snapshot_stats["competition"].map(
            normalize_competition
        )
        snapshot_stats["_pid_int"] = snapshot_stats["player_id"].map(_to_int)
        snapshot_stats["_lid_int"] = snapshot_stats["league_id"].map(_to_int)

        # Restrict the history table to only eligible player/league/competition
        # contexts before grouping. This replaces N scans of the whole snapshot
        # with one hash join + groupby.
        history = snapshot_stats.merge(
            eligible[["_pid_int", "_lid_int", "_competition"]].drop_duplicates(),
            on=["_pid_int", "_lid_int", "_competition"],
            how="inner",
            sort=False,
            copy=False,
        )
        if history.empty:
            continue

        feature_rows: List[Dict[str, float]] = []
        prediction_keys: List[Tuple[int, int, str, int]] = []

        for (pid, league_id, competition), group in history.groupby(
            ["_pid_int", "_lid_int", "_competition"], sort=False
        ):
            ordered = group.sort_values("_season_year", kind="stable").reset_index(drop=True)
            years = ordered["_season_year"].astype(int).to_numpy()

            source_index = int(np.searchsorted(years, source_year, side="left"))
            if source_index >= len(ordered) or int(years[source_index]) != source_year:
                continue

            source_row = ordered.iloc[source_index]
            player_id = int(pid)
            lid = int(league_id)
            comp = str(competition)
            player = snapshot["player_dict"].get(player_id, {})
            position = str(player.get("position", "PG") or "PG")

            history_clean = ordered.drop(
                columns=["_season_year", "_pid_int", "_lid_int", "_competition"]
            ).reset_index(drop=True)
            source_clean = source_row.drop(
                labels=["_season_year", "_pid_int", "_lid_int", "_competition"]
            ).copy()
            source_clean["competition"] = comp

            try:
                precomputed = base_model._precompute_history_features(
                    history_clean,
                    league_max_games=league_max_games,
                )
                if not precomputed:
                    continue
                feature_rows.append(
                    base_model._build_row(
                        source_clean,
                        _player_age_as_of(player, source_year),
                        position,
                        history_clean,
                        extra_metrics=[],
                        league_max_games=league_max_games,
                        precomputed_history=precomputed[source_index],
                    )
                )
                prediction_keys.append((player_id, lid, comp, source_year + 1))
            except (TypeError, ValueError, KeyError):
                continue

        if not feature_rows:
            continue

        # XGBoost prediction is vectorised. The old implementation invoked
        # model.predict() once per player/competition pair, which adds substantial
        # Python/XGBoost dispatch overhead on large historical OOF sets.
        feature_frame = pd.DataFrame(
            feature_rows,
            columns=base_model.feature_names,
        ).fillna(0.0)

        try:
            raw_predictions = np.asarray(
                base_model.model.predict(feature_frame.to_numpy(dtype=float)),
                dtype=float,
            )
            if getattr(base_model, "target_mode", "rating") == "delta_vs_prior":
                prior_ratings = feature_frame["last_rating"].to_numpy(dtype=float)
                batch_predictions = np.clip(
                    prior_ratings + raw_predictions,
                    0.0,
                    10.0,
                )
            else:
                batch_predictions = raw_predictions
        except (TypeError, ValueError):
            # Defensive fallback for an unexpected backend/feature-shape issue.
            batch_predictions = np.asarray(
                [
                    base_model.predict_from_features(row)
                    for row in feature_rows
                ],
                dtype=float,
            )

        predictions.update(
            {
                key: float(value)
                for key, value in zip(prediction_keys, batch_predictions)
            }
        )

    return predictions


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
        self._age_curve_state: Dict[str, Dict[str, float]] = {}

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

    def _records_with_calibrated_intervals(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply fitted conformal widths without rerunning model predictions."""
        calibrated: List[Dict[str, Any]] = []
        for row in records:
            updated = dict(row)
            competition = normalize_competition(updated.get("competition"))
            half_width = self._conformal_by_competition.get(
                competition, self._conformal_q_hi
            )
            if half_width is not None:
                prediction = float(updated["prediction"])
                updated["confidence_low"] = float(
                    np.clip(prediction - float(half_width), 0.0, 10.0)
                )
                updated["confidence_high"] = float(
                    np.clip(prediction + float(half_width), 0.0, 10.0)
                )
            calibrated.append(updated)
        return calibrated

    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from basketball_ai.features.team_features import calibrate_style_bounds
        from basketball_ai.models.age_curve import (
            get_fitted_params,
            maybe_fit_from_db,
            reset_fitted_params,
        )

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

        try:
            maybe_fit_from_db(fit_data)
        except Exception as exc:
            logger.warning("[StrictProduction] Empirical age-curve fit failed: %s", exc)
        self._age_curve_state = get_fitted_params()

        compatibility_oof = build_base_oof_predictions(fit_data)
        if compatibility_oof:
            self.compat_model.train(
                fit_data,
                base_predictions=compatibility_oof,
            )
        else:
            logger.warning(
                "[StrictProduction] OOF base predictions unavailable; "
                "using legacy compatibility target."
            )
            self.compat_model.train(fit_data)
        self._restore_competition_encoding()
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

        calibrated_records = self._records_with_calibrated_intervals(
            calibration_records
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
        metrics["compatibility_oof_samples"] = int(len(compatibility_oof))
        metrics["compatibility_target_mode"] = getattr(
            self.compat_model,
            "training_target_mode",
            "unknown",
        )
        metrics["compatibility_samples_by_competition"] = dict(
            getattr(self.compat_model, "training_samples_by_competition", {})
        )
        metrics["compatibility_samples_by_context_mode"] = dict(
            getattr(self.compat_model, "training_samples_by_context_mode", {})
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
        from basketball_ai.models.age_curve import set_fitted_params

        set_fitted_params(self._age_curve_state or None)
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
                "age_curve": {
                    "peak_ages": dict(self._age_curve_state.get("peak_ages", {})),
                    "sigma_before": dict(self._age_curve_state.get("sigma_before", {})),
                    "sigma_after": dict(self._age_curve_state.get("sigma_after", {})),
                },
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
        age_curve = state.get("age_curve")
        self._age_curve_state = {
            "peak_ages": {
                str(key): float(value)
                for key, value in (age_curve or {}).get("peak_ages", {}).items()
            },
            "sigma_before": {
                str(key): float(value)
                for key, value in (age_curve or {}).get("sigma_before", {}).items()
            },
            "sigma_after": {
                str(key): float(value)
                for key, value in (age_curve or {}).get("sigma_after", {}).items()
            },
        }
        from basketball_ai.models.age_curve import set_fitted_params
        set_fitted_params(self._age_curve_state or None)
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
    source_stats["_season_year"] = _numeric_seasons(source_stats)
    source_stats = source_stats[source_stats["_season_year"] == source_season].copy()
    source_stats["_pid_int"] = source_stats["player_id"].map(_to_int)
    source_stats["_lid_int"] = source_stats["league_id"].map(_to_int)
    source_stats["_competition"] = source_stats["competition"].map(
        normalize_competition
    )
    prior_map: Dict[Tuple[int, int, str], float] = {
        (
            int(row["_pid_int"]),
            int(row["_lid_int"]),
            str(row["_competition"]),
        ): float(row["rating"])
        for row in source_stats.to_dict("records")
        if pd.notna(row.get("rating"))
    }

    target_stats = data["player_stats"].copy()
    target_stats["_season_year"] = _numeric_seasons(target_stats)
    target_stats = target_stats[target_stats["_season_year"] == target_season]

    relations = snapshot.get("team_player_relations", pd.DataFrame())
    _position_index(relations)
    records: List[Dict[str, Any]] = []
    ensemble.clear_cache()
    for target in target_stats.to_dict("records"):
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
        prior_rating = prior_map.get((pid, league_id, competition))
        if prior_rating is None:
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
                "persistence_prediction": prior_rating,
                "position": position_as_of(
                    pid,
                    relations,
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


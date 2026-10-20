"""Scenario-aware calibration for future BB-Rating predictions.

Input predictions and labels must both be on the official 1-100 BB-Rating
scale. Fit only on out-of-fold predictions with labels produced by the
versioned BBRatingEngine for each target season and target league.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SCENARIO_SAME_TEAM = "same_team_same_league"
SCENARIO_TEAM_CHANGE = "team_change_same_league"
SCENARIO_LEAGUE_CHANGE = "league_change"
SCENARIO_TEAM_AND_LEAGUE_CHANGE = "team_and_league_change"
SCENARIO_UNKNOWN = "unknown"


def classify_transition(
    source_team_id: Any,
    target_team_id: Any,
    source_league_id: Any,
    target_league_id: Any,
    source_competition: Any = None,
    target_competition: Any = None,
    source_tier: Any = None,
    target_tier: Any = None,
) -> str:
    """Classify team/league transitions without treating statistic views as competitions.

    The competition arguments remain for API compatibility, but their values may
    be statistical views or phases (HOME, AWAY, RS, TOT, PO, etc.). They cannot
    establish a real competition change without an independent competition ID.
    Unknown team identity fails closed rather than assuming a same-team move.
    """
    values = (source_team_id, target_team_id, source_league_id, target_league_id)
    if any(value is None or pd.isna(value) or str(value).strip() == "" for value in values):
        return SCENARIO_UNKNOWN
    same_team = str(source_team_id) == str(target_team_id)
    same_league = str(source_league_id).strip().upper() == str(target_league_id).strip().upper()
    # A differing statistic-view code is not an independent competition identity.
    if same_team and same_league:
        scenario = SCENARIO_SAME_TEAM
    elif not same_team and same_league:
        scenario = SCENARIO_TEAM_CHANGE
    elif same_team and not same_league:
        scenario = SCENARIO_LEAGUE_CHANGE
    else:
        scenario = SCENARIO_TEAM_AND_LEAGUE_CHANGE

    # League changes can be split by tier direction where authoritative tier
    # metadata exists. Lower tier number means a stronger competition tier.
    if not same_league and source_tier is not None and target_tier is not None:
        try:
            source_rank, target_rank = int(float(source_tier)), int(float(target_tier))
            tier_label = (
                "tier_up" if target_rank < source_rank
                else "tier_down" if target_rank > source_rank
                else "same_tier"
            )
            scenario = f"{scenario}_{tier_label}"
        except (TypeError, ValueError):
            pass
    return scenario


def _calibration_support(frame: pd.DataFrame) -> dict[str, int | None]:
    rows = int(len(frame))
    unique_players = (
        int(frame["player_global_id"].nunique(dropna=True))
        if "player_global_id" in frame.columns else None
    )
    transition_columns = ["player_global_id", "target_season"]
    target_columns = [
        "player_global_id", "target_season", "target_league", "target_competition"
    ]
    transitions = (
        int(frame[transition_columns].dropna(subset=transition_columns).drop_duplicates().shape[0])
        if all(column in frame.columns for column in transition_columns) else None
    )
    target_contexts = (
        int(frame[target_columns].dropna(subset=target_columns).drop_duplicates().shape[0])
        if all(column in frame.columns for column in target_columns) else None
    )
    return {
        "rows_evaluated": rows,
        "unique_players": unique_players,
        "unique_player_season_transitions": transitions,
        "unique_target_contexts": target_contexts,
    }


def _player_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Prevent players with multiple target contexts in one season dominating calibration."""
    if not {"player_global_id", "target_season"}.issubset(frame.columns) or frame.empty:
        return np.ones(len(frame), dtype=float)
    counts = frame.groupby(
        ["player_global_id", "target_season"], dropna=False
    )["target_season"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(counts, 1.0)
    average = float(np.mean(weights))
    return weights / average if average > 0 else np.ones(len(frame), dtype=float)


@dataclass
class _Calibrator:
    model: IsotonicRegression
    n_samples: int  # raw calibration rows
    unique_players: int | None = None


class BBRatingScenarioCalibrator:
    """Fit isotonic mappings by destination league + scenario, with safe fallback.

    Fallback order: destination league + scenario, scenario, then global.
    Sparse groups are not fitted. Predictions are clipped to the 1-100 scale.
    """

    def __init__(self, min_samples: int = 40) -> None:
        self.min_samples = max(2, int(min_samples))
        self._models: dict[str, _Calibrator] = {}
        self._fitted = False
        self.fit_report: dict[str, Any] = {}

    @staticmethod
    def _key(scenario: str, league: Any = None) -> str:
        league_key = "*" if league is None or pd.isna(league) else str(league)
        return league_key + "::" + str(scenario)

    @staticmethod
    def _fit_model(
        x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> IsotonicRegression:
        model = IsotonicRegression(y_min=1.0, y_max=100.0, out_of_bounds="clip")
        model.fit(x, y, sample_weight=sample_weight)
        return model

    def fit(self, records: pd.DataFrame, prediction_col: str = "predicted_bb_rating",
            target_col: str = "observed_bb_rating", scenario_col: str = "scenario",
            destination_league_col: str = "target_league_id") -> dict[str, Any]:
        required = {prediction_col, target_col, scenario_col}
        missing = required.difference(records.columns)
        if missing:
            raise ValueError("Missing calibration columns: " + ", ".join(sorted(missing)))
        frame = records.copy()
        for column in (prediction_col, target_col):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[prediction_col, target_col, scenario_col])
        frame = frame[frame[prediction_col].between(1, 100) &
                      frame[target_col].between(1, 100)].copy()
        if frame.empty:
            raise ValueError("No valid BB-Rating calibration rows in the 1-100 range")

        frame_weights = _player_balanced_weights(frame)
        global_support = _calibration_support(frame)
        self._models = {
            "*::global": _Calibrator(
                self._fit_model(
                    frame[prediction_col].to_numpy(float),
                    frame[target_col].to_numpy(float),
                    sample_weight=frame_weights,
                ), len(frame), global_support["unique_players"])
        }
        groups_report = []
        for columns in ([scenario_col],
                        [destination_league_col, scenario_col]
                        if destination_league_col in frame.columns else []):
            if not columns:
                continue
            for values, group in frame.groupby(columns, dropna=False, sort=True):
                values = values if isinstance(values, tuple) else (values,)
                if len(columns) == 1:
                    key = self._key(str(values[0]))
                else:
                    key = self._key(str(values[1]), values[0])
                n = len(group)
                support = _calibration_support(group)
                effective_support = (
                    support["unique_players"]
                    if support["unique_players"] is not None else n
                )
                fitted = (
                    int(effective_support) >= self.min_samples
                    and group[prediction_col].nunique() >= 2
                )
                if fitted:
                    self._models[key] = _Calibrator(
                        self._fit_model(
                            group[prediction_col].to_numpy(float),
                            group[target_col].to_numpy(float),
                            sample_weight=_player_balanced_weights(group),
                        ), n, support["unique_players"])
                groups_report.append({
                    "key": key,
                    "n_samples": int(n),
                    **support,
                    "support_unit": "unique_players" if support["unique_players"] is not None else "rows",
                    "fitted": fitted,
                })
        self._fitted = True
        self.fit_report = {
            "method": "isotonic_oof_bb_rating", "scale": [1, 100],
            "n_samples": int(len(frame)),
            **_calibration_support(frame),
            "min_samples": self.min_samples,
            "min_group_support_unit": (
                "unique_players" if "player_global_id" in frame.columns else "rows"
            ),
            "fallback_order": ["destination_league+scenario", "scenario", "global"],
            "groups": groups_report,
        }
        return self.fit_report

    def predict(self, predicted_bb_rating: float, *, scenario: str,
                target_league_id: Any = None) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("BB-Rating scenario calibrator has not been fitted")
        value = float(np.clip(predicted_bb_rating, 1, 100))
        candidates = [
            (self._key(scenario, target_league_id), "destination_league+scenario"),
            (self._key(scenario), "scenario"), ("*::global", "global"),
        ]
        for key, level in candidates:
            fitted = self._models.get(key)
            if fitted is not None:
                score = float(np.clip(fitted.model.predict([value])[0], 1, 100))
                effective_samples = (
                    fitted.unique_players
                    if fitted.unique_players is not None else fitted.n_samples
                )
                return {
                    "predicted_bb_rating": score,
                    # Keep the legacy field, but its unit is now independent
                    # players whenever canonical identity is available.
                    "calibration_samples": int(effective_samples),
                    "calibration_unique_players": fitted.unique_players,
                    "calibration_rows": int(fitted.n_samples),
                    "calibration_level": level,
                    "scenario": scenario,
                    "target_league_id": target_league_id,
                }
        raise RuntimeError("Global BB-Rating calibration is unavailable")

    def to_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("BB-Rating scenario calibrator has not been fitted")
        return {"version": 1, "method": "isotonic_oof_bb_rating",
                "min_samples": self.min_samples,
                "models": {key: {"x_thresholds": item.model.X_thresholds_.tolist(),
                                 "y_thresholds": item.model.y_thresholds_.tolist(),
                                 "n_samples": item.n_samples,
                                 "unique_players": item.unique_players}
                           for key, item in self._models.items()},
                "fit_report": self.fit_report}

    @classmethod
    def from_dict(cls, artifact: dict[str, Any]) -> "BBRatingScenarioCalibrator":
        if artifact.get("method") != "isotonic_oof_bb_rating":
            raise ValueError("Unsupported BB-Rating calibration artifact")
        instance = cls(int(artifact.get("min_samples", 40)))
        for key, item in artifact.get("models", {}).items():
            x = np.asarray(item["x_thresholds"], dtype=float)
            y = np.asarray(item["y_thresholds"], dtype=float)
            if len(x) < 2 or len(x) != len(y):
                raise ValueError("Invalid isotonic thresholds for " + str(key))
            instance._models[str(key)] = _Calibrator(
                instance._fit_model(x, y),
                int(item["n_samples"]),
                int(item["unique_players"]) if item.get("unique_players") is not None else None,
            )
        if "*::global" not in instance._models:
            raise ValueError("Calibration artifact has no global fallback")
        instance._fitted = True
        instance.fit_report = dict(artifact.get("fit_report") or {})
        return instance

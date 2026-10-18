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


def classify_transition(source_team_id: Any, target_team_id: Any,
                        source_league_id: Any, target_league_id: Any) -> str:
    """Classify same-team, team-change, league-change and combined transitions."""
    values = (source_team_id, target_team_id, source_league_id, target_league_id)
    if any(value is None or pd.isna(value) for value in values):
        return SCENARIO_UNKNOWN
    same_team = str(source_team_id) == str(target_team_id)
    same_league = str(source_league_id) == str(target_league_id)
    if same_team and same_league:
        return SCENARIO_SAME_TEAM
    if same_league:
        return SCENARIO_TEAM_CHANGE
    if same_team:
        return SCENARIO_LEAGUE_CHANGE
    return SCENARIO_TEAM_AND_LEAGUE_CHANGE


@dataclass
class _Calibrator:
    model: IsotonicRegression
    n_samples: int


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
    def _fit_model(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
        model = IsotonicRegression(y_min=1.0, y_max=100.0, out_of_bounds="clip")
        model.fit(x, y)
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

        self._models = {
            "*::global": _Calibrator(
                self._fit_model(frame[prediction_col].to_numpy(float),
                                frame[target_col].to_numpy(float)), len(frame))
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
                fitted = n >= self.min_samples and group[prediction_col].nunique() >= 2
                if fitted:
                    self._models[key] = _Calibrator(
                        self._fit_model(group[prediction_col].to_numpy(float),
                                        group[target_col].to_numpy(float)), n)
                groups_report.append({"key": key, "n_samples": int(n), "fitted": fitted})
        self._fitted = True
        self.fit_report = {
            "method": "isotonic_oof_bb_rating", "scale": [1, 100],
            "n_samples": int(len(frame)), "min_samples": self.min_samples,
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
                return {"predicted_bb_rating": score, "calibration_level": level,
                        "calibration_samples": fitted.n_samples, "scenario": scenario,
                        "target_league_id": target_league_id}
        raise RuntimeError("Global BB-Rating calibration is unavailable")

    def to_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("BB-Rating scenario calibrator has not been fitted")
        return {"version": 1, "method": "isotonic_oof_bb_rating",
                "min_samples": self.min_samples,
                "models": {key: {"x_thresholds": item.model.X_thresholds_.tolist(),
                                 "y_thresholds": item.model.y_thresholds_.tolist(),
                                 "n_samples": item.n_samples}
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
            instance._models[str(key)] = _Calibrator(instance._fit_model(x, y),
                                                       int(item["n_samples"]))
        if "*::global" not in instance._models:
            raise ValueError("Calibration artifact has no global fallback")
        instance._fitted = True
        instance.fit_report = dict(artifact.get("fit_report") or {})
        return instance

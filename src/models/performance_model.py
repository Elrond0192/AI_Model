"""XGBoost performance model: predicts player rating from engineered features.

Training pipeline:
    1. Build (X, y) from all player-season rows via feature engineering.
    2. Train XGBRegressor with early stopping.
    3. Expose predict() and get_shap_values() methods.
    4. Persist / load with joblib.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# SHAP import (optional – degrade gracefully)
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

# Feature columns used during training / inference (order matters for scaling)
FEATURE_COLS: List[str] = [
    "age",
    "position_enc",
    "goals_per_90",
    "assists_per_90",
    "xG_per_90",
    "xA_per_90",
    "pass_accuracy",
    "dribbles",
    "tackles",
    "interceptions",
    "aerial_duels_won",
    "progressive_passes",
    "key_passes",
    "minutes",
    "matches_played",
]


class PerformanceModel:
    """XGBoost-based player rating predictor."""

    POSITION_ENCODING: Dict[str, int] = {
        "GK": 0, "CB": 1, "FB": 2, "CM": 3, "AM": 4, "W": 5, "ST": 6,
    }

    def __init__(self) -> None:
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = FEATURE_COLS.copy()
        self.is_trained: bool = False
        self._shap_explainer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _build_row(
        self, stat_row: pd.Series, age_at_season: int, position: str
    ) -> Dict[str, float]:
        """Build a feature row from a player-stats row."""
        total_min = float(stat_row.get("minutes", 0))
        m90 = max(total_min / 90.0, 0.01)
        return {
            "age": float(age_at_season),
            "position_enc": float(self.POSITION_ENCODING.get(position, 3)),
            "goals_per_90": float(stat_row.get("goals", 0)) / m90,
            "assists_per_90": float(stat_row.get("assists", 0)) / m90,
            "xG_per_90": float(stat_row.get("xG", 0)) / m90,
            "xA_per_90": float(stat_row.get("xA", 0)) / m90,
            "pass_accuracy": float(stat_row.get("pass_accuracy", 75)),
            "dribbles": float(stat_row.get("dribbles", 1.0)),
            "tackles": float(stat_row.get("tackles", 1.0)),
            "interceptions": float(stat_row.get("interceptions", 0.5)),
            "aerial_duels_won": float(stat_row.get("aerial_duels_won", 1.0)),
            "progressive_passes": float(stat_row.get("progressive_passes", 2.0)),
            "key_passes": float(stat_row.get("key_passes", 0.5)),
            "minutes": total_min,
            "matches_played": float(stat_row.get("matches_played", 20)),
        }

    def prepare_features(
        self, data: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build training feature matrix and target vector."""
        player_stats = data["player_stats"]
        players = data["players"]

        # birth year lookup: age_in_2024 → birth_year = 2024 - age
        birth_year_map: Dict[int, int] = {
            int(row["id"]): 2024 - int(row["age"])
            for _, row in players.iterrows()
        }
        position_map: Dict[int, str] = {
            int(row["id"]): str(row["position"])
            for _, row in players.iterrows()
        }

        rows: List[Dict[str, float]] = []
        targets: List[float] = []

        for _, stat in player_stats.iterrows():
            pid = int(stat["player_id"])
            season = int(stat["season"])
            by = birth_year_map.get(pid)
            pos = position_map.get(pid, "CM")
            if by is None:
                continue
            age_s = season - by
            if age_s < 14 or age_s > 45:
                continue
            rows.append(self._build_row(stat, age_s, pos))
            targets.append(float(stat["rating"]))

        X = pd.DataFrame(rows, columns=self.feature_names)
        y = np.array(targets, dtype=float)
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Train the model on all available player-season data.

        Returns:
            Dict with 'train_rmse' and 'val_rmse'.
        """
        print("[PerformanceModel] Building feature matrix …")
        X, y = self.prepare_features(data)
        print(f"[PerformanceModel] Training on {len(X):,} samples …")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        X_tr_sc = self.scaler.fit_transform(X_train)
        X_va_sc = self.scaler.transform(X_val)

        self.model.fit(
            X_tr_sc, y_train,
            eval_set=[(X_va_sc, y_val)],
            verbose=False,
        )

        y_pred_tr = self.model.predict(X_tr_sc)
        y_pred_va = self.model.predict(X_va_sc)
        train_rmse = float(np.sqrt(np.mean((y_pred_tr - y_train) ** 2)))
        val_rmse = float(np.sqrt(np.mean((y_pred_va - y_val) ** 2)))

        self.is_trained = True
        print(f"[PerformanceModel] Train RMSE={train_rmse:.4f}  Val RMSE={val_rmse:.4f}")
        return {"train_rmse": train_rmse, "val_rmse": val_rmse}

    def predict_from_features(self, feature_dict: Dict[str, float]) -> float:
        """Predict rating from an already-engineered feature dict."""
        if not self.is_trained:
            return 6.5
        row = pd.DataFrame([[feature_dict.get(c, 0.0) for c in self.feature_names]],
                           columns=self.feature_names)
        scaled = self.scaler.transform(row)
        return float(np.clip(self.model.predict(scaled)[0], 4.0, 10.0))

    def predict_from_stat_row(
        self, stat_row: pd.Series, age: int, position: str
    ) -> float:
        """Convenience wrapper used by the ensemble."""
        feat = self._build_row(stat_row, age, position)
        return self.predict_from_features(feat)

    def get_shap_values(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Return SHAP feature attributions (requires shap package)."""
        if not self.is_trained or not _SHAP_AVAILABLE:
            return {}
        try:
            if self._shap_explainer is None:
                self._shap_explainer = _shap.TreeExplainer(self.model)
            row = pd.DataFrame(
                [[feature_dict.get(c, 0.0) for c in self.feature_names]],
                columns=self.feature_names,
            )
            scaled = self.scaler.transform(row)
            sv = self._shap_explainer.shap_values(scaled)
            return {name: float(sv[0][i]) for i, name in enumerate(self.feature_names)}
        except Exception:  # pragma: no cover
            return {}

    def feature_importances(self) -> Dict[str, float]:
        """Return XGBoost feature importances (gain)."""
        if not self.is_trained:
            return {}
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise model to disk with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        print(f"[PerformanceModel] Saved to {path}")

    def load(self, path: str) -> None:
        """Deserialise model from disk."""
        payload = joblib.load(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.is_trained = True
        self._shap_explainer = None
        print(f"[PerformanceModel] Loaded from {path}")

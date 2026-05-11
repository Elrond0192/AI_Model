"""Basketball XGBoost performance model.

Predicts player rating (0-10) from basketball-specific features:
per-36 stats, advanced metrics (PER, BPM, TS%, USG%), age, position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Feature columns (order matters for scaling)
FEATURE_COLS: List[str] = [
    "age",
    "position_enc",
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
    "stl_per_36",
    "blk_per_36",
    "avg_per",
    "avg_ts_pct",
    "avg_usg_pct",
    "avg_bpm",
    "form_score",
    "consistency_score",
    "career_trajectory",
    "age_vs_peak_age",
]

# Encoding for all position strings (pure + hybrid)
POSITION_ENCODING: Dict[str, int] = {
    "PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4,
    "PG/SG": 5, "SG/SF": 6, "SF/PF": 7, "PF/C": 8, "SG/PF": 9,
}

POSITIONAL_PEAK_AGES: Dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}


def _primary_pos(pos: str) -> str:
    return pos.split("/")[0]


def _peak_age(pos: str) -> int:
    return POSITIONAL_PEAK_AGES.get(pos, POSITIONAL_PEAK_AGES.get(_primary_pos(pos), 26))


class PerformanceModel:
    """XGBoost player rating predictor for basketball."""

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
        self,
        stat_row: pd.Series,
        age: int,
        position: str,
        player_stats_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Build a feature row from a player-stats row."""
        mpg = float(stat_row.get("minutes_per_game", 0))
        if mpg <= 0:
            mpg = 1.0

        def per36(col: str) -> float:
            return float(stat_row.get(col, 0)) / mpg * 36

        # Career stats up to this season
        ratings = player_stats_history["rating"].tolist()
        last3 = ratings[-3:]
        weights = ([0.2, 0.3, 0.5] if len(last3) == 3
                   else ([0.4, 0.6] if len(last3) == 2 else [1.0]))
        form_score = sum(r * w for r, w in zip(last3, weights))
        mean_r = float(np.mean(ratings)) if ratings else 6.5
        std_r  = float(np.std(ratings))  if len(ratings) > 1 else 0.0
        consistency = float(np.clip(1.0 - (std_r / mean_r) if mean_r > 0 else 0.0, 0, 1))
        trajectory = (
            float(np.polyfit(np.arange(len(ratings), dtype=float), ratings, 1)[0])
            if len(ratings) >= 2 else 0.0
        )

        avg_per = float(np.mean(player_stats_history["per"].tolist())) if not player_stats_history.empty else float(stat_row.get("per", 12))
        avg_ts  = float(np.mean(player_stats_history["ts_pct"].tolist())) if not player_stats_history.empty else float(stat_row.get("ts_pct", 0.52))
        avg_usg = float(np.mean(player_stats_history["usg_pct"].tolist())) if not player_stats_history.empty else float(stat_row.get("usg_pct", 18))
        avg_bpm = float(np.mean(player_stats_history["bpm"].tolist())) if not player_stats_history.empty else float(stat_row.get("bpm", -1))

        pos_enc = float(POSITION_ENCODING.get(position, POSITION_ENCODING.get(_primary_pos(position), 0)))
        peak_age = _peak_age(position)

        return {
            "age":                float(age),
            "position_enc":       pos_enc,
            "pts_per_36":         per36("points"),
            "ast_per_36":         per36("assists"),
            "reb_per_36":         per36("rebounds"),
            "stl_per_36":         per36("steals"),
            "blk_per_36":         per36("blocks"),
            "avg_per":            avg_per,
            "avg_ts_pct":         avg_ts,
            "avg_usg_pct":        avg_usg,
            "avg_bpm":            avg_bpm,
            "form_score":         form_score,
            "consistency_score":  consistency,
            "career_trajectory":  trajectory,
            "age_vs_peak_age":    float(age - peak_age),
        }

    def prepare_features(
        self, data: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build training (X, y) from all player-season data."""
        player_stats = data["player_stats"]
        players      = data["players"]

        birth_year_map: Dict[int, int] = {
            int(row["id"]): 2024 - int(row["age"])
            for _, row in players.iterrows()
        }
        position_map: Dict[int, str] = {
            int(row["id"]): str(row["position"])
            for _, row in players.iterrows()
        }

        rows:    List[Dict[str, float]] = []
        targets: List[float]            = []

        for pid, grp in player_stats.groupby("player_id"):
            grp = grp.sort_values("season")
            pid = int(pid)
            pos = position_map.get(pid, "PG")
            by  = birth_year_map.get(pid)
            if by is None:
                continue
            for idx, stat in grp.iterrows():
                season_str = str(stat["season"])
                # approximate year from season string like "2023-24"
                try:
                    year = int(season_str.split("-")[0])
                except Exception:
                    year = 2024
                age = year - by
                if age < 14 or age > 45:
                    continue
                history_so_far = grp[grp["season"] <= stat["season"]]
                rows.append(self._build_row(stat, age, pos, history_so_far))
                targets.append(float(stat["rating"]))

        X = pd.DataFrame(rows, columns=self.feature_names)
        y = np.array(targets, dtype=float)
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Train the model and return RMSE metrics."""
        print("[PerformanceModel] Building feature matrix …")
        X, y = self.prepare_features(data)
        print(f"[PerformanceModel] Training on {len(X):,} samples …")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        X_tr_sc = self.scaler.fit_transform(X_train)
        X_va_sc = self.scaler.transform(X_val)

        self.model.fit(X_tr_sc, y_train, eval_set=[(X_va_sc, y_val)], verbose=False)

        train_rmse = float(np.sqrt(np.mean((self.model.predict(X_tr_sc) - y_train) ** 2)))
        val_rmse   = float(np.sqrt(np.mean((self.model.predict(X_va_sc) - y_val) ** 2)))
        self.is_trained = True
        print(f"[PerformanceModel] Train RMSE={train_rmse:.4f}  Val RMSE={val_rmse:.4f}")
        return {"train_rmse": train_rmse, "val_rmse": val_rmse}

    def predict_from_features(self, feature_dict: Dict[str, float]) -> float:
        """Predict rating from an already-engineered feature dict."""
        if not self.is_trained:
            return float(np.clip(feature_dict.get("form_score", 6.5), 4.0, 10.0))
        row = pd.DataFrame(
            [[feature_dict.get(c, 0.0) for c in self.feature_names]],
            columns=self.feature_names,
        )
        scaled = self.scaler.transform(row)
        return float(np.clip(self.model.predict(scaled)[0], 3.5, 10.0))

    def get_shap_values(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
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
        except Exception:
            return {}

    def feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names}, path)
        print(f"[PerformanceModel] Saved to {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.model         = payload["model"]
        self.scaler        = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.is_trained    = True
        self._shap_explainer = None
        print(f"[PerformanceModel] Loaded from {path}")

"""Basketball compatibility model.

Scores how well a player fits a team's style using KNN on
6-dimensional basketball style vectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from basketball_ai.features.team_features import compute_team_style_vector
from basketball_ai.data.models import Team


class CompatibilityModel:
    """KNN-based player–team style compatibility scorer (0–1)."""

    def __init__(self, n_neighbors: int = 10) -> None:
        self.n_neighbors = n_neighbors
        self.knn     = KNeighborsRegressor(n_neighbors=n_neighbors, metric="euclidean")
        self.scaler  = StandardScaler()
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _team_style_vector(self, team_row: Dict) -> np.ndarray:
        """Build 6-d style vector from a team row dict."""
        try:
            team = Team(**{k: team_row.get(k) for k in Team.__dataclass_fields__})
            return compute_team_style_vector(team)
        except Exception:
            pace  = float(team_row.get("pace", 97))
            tpar  = float(team_row.get("three_point_attempt_rate", 0.38))
            ast   = float(team_row.get("assists_per_game", 24))
            star  = float(team_row.get("star_player_usage", 0.28))
            ortg  = float(team_row.get("offensive_rating", 108))
            drtg  = float(team_row.get("defensive_rating", 108))
            v = np.array([
                np.clip((pace - 80) / 35, 0, 1),
                np.clip(tpar / 0.55, 0, 1),
                np.clip((ast - 14) / 21, 0, 1),
                np.clip(star / 0.45, 0, 1),
                np.clip((ortg - 90) / 35, 0, 1),
                np.clip(1.0 - (drtg - 88) / 32, 0, 1),
            ])
            return v.astype(float)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> None:
        """Train KNN on (team_style_vector, compatibility_score) pairs."""
        print("[CompatibilityModel] Building compatibility training data …")
        player_stats = data["player_stats"]
        team_dict    = data["team_dict"]

        player_mean = player_stats.groupby("player_id")["rating"].mean().to_dict()

        X_rows: List[np.ndarray] = []
        y_vals: List[float]      = []

        for _, stat in player_stats.iterrows():
            tid  = int(stat["team_id"])
            team = team_dict.get(tid)
            if team is None:
                continue
            sv   = self._team_style_vector(team)
            pid  = int(stat["player_id"])
            p_mean = player_mean.get(pid, 6.5)
            compat = float(np.clip((float(stat["rating"]) - p_mean + 1.5) / 3.0, 0.0, 1.0))
            X_rows.append(sv)
            y_vals.append(compat)

        if not X_rows:
            self.is_trained = False
            return

        X = np.array(X_rows)
        y = np.array(y_vals)
        X_sc = self.scaler.fit_transform(X)
        self.knn.fit(X_sc, y)
        self.is_trained = True
        print(f"[CompatibilityModel] Trained on {len(X):,} samples.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, team_id: int, data: Dict[str, Any]) -> float:
        """Return compatibility score (0–1) for a team's style."""
        if not self.is_trained:
            return 0.75
        team = data["team_dict"].get(int(team_id))
        if team is None:
            return 0.75
        sv = self._team_style_vector(team).reshape(1, -1)
        sv_sc = self.scaler.transform(sv)
        return float(np.clip(self.knn.predict(sv_sc)[0], 0.50, 1.0))

    def score_style_vector(self, style_vector: np.ndarray) -> float:
        """Score a raw 6-d style vector directly."""
        if not self.is_trained:
            return 0.75
        sv_sc = self.scaler.transform(style_vector.reshape(1, -1))
        return float(np.clip(self.knn.predict(sv_sc)[0], 0.50, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"knn": self.knn, "scaler": self.scaler}, path)
        print(f"[CompatibilityModel] Saved to {path}")

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.knn    = obj["knn"]
        self.scaler = obj["scaler"]
        self.is_trained = True
        print(f"[CompatibilityModel] Loaded from {path}")

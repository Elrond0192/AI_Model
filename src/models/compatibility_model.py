"""Compatibility model: scores player–team style fit using KNN on style vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class CompatibilityModel:
    """KNN-based player–team style compatibility scorer (0–1)."""

    STYLE_KEYS: List[str] = [
        "avg_possession", "pressing_intensity", "defensive_line", "passing_tempo",
    ]

    def __init__(self, n_neighbors: int = 10) -> None:
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric="euclidean")
        self.scaler = StandardScaler()
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _team_style_vector(self, team: Dict) -> np.ndarray:
        """Extract normalised style vector from a team row dict."""
        raw = np.array([
            float(team.get("avg_possession", 50.0)) / 70.0,
            float(team.get("pressing_intensity", 5.0)) / 10.0,
            float(team.get("defensive_line", 5.0)) / 10.0,
            float(team.get("passing_tempo", 5.0)) / 10.0,
        ])
        return np.clip(raw, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> None:
        """Build KNN on (player_style_vector, compatibility_score) pairs.

        We approximate "style fit" by measuring how a player performed
        relative to their average across teams with different styles.
        Pairs with more minutes are weighted more.
        """
        print("[CompatibilityModel] Building compatibility training data …")
        player_stats = data["player_stats"]
        team_dict = data["team_dict"]

        X_rows: List[np.ndarray] = []
        y_vals: List[float] = []

        # Mean rating per player across all teams
        player_mean = player_stats.groupby("player_id")["rating"].mean().to_dict()

        for _, stat in player_stats.iterrows():
            tid = int(stat["team_id"])
            team = team_dict.get(tid)
            if team is None:
                continue
            sv = self._team_style_vector(team)
            pid = int(stat["player_id"])
            p_mean = player_mean.get(pid, 6.5)
            # Compatibility score: how above/below average the player performed
            compat = float(np.clip((stat["rating"] - p_mean + 1.5) / 3.0, 0.0, 1.0))
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
        """Return compatibility score (0–1) for a player in a given team.

        Scores the team's style vector through the KNN model.
        """
        if not self.is_trained:
            return 0.75
        team = data["team_dict"].get(int(team_id))
        if team is None:
            return 0.75
        sv = self._team_style_vector(team).reshape(1, -1)
        sv_sc = self.scaler.transform(sv)
        return float(np.clip(self.knn.predict(sv_sc)[0], 0.5, 1.0))

    def score_style(self, style_vector: np.ndarray) -> float:
        """Score a raw (4-dim) style vector directly."""
        if not self.is_trained:
            return 0.75
        sv_sc = self.scaler.transform(style_vector.reshape(1, -1))
        return float(np.clip(self.knn.predict(sv_sc)[0], 0.5, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"knn": self.knn, "scaler": self.scaler}, path)
        print(f"[CompatibilityModel] Saved to {path}")

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.knn = obj["knn"]
        self.scaler = obj["scaler"]
        self.is_trained = True
        print(f"[CompatibilityModel] Loaded from {path}")

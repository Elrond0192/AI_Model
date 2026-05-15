"""Basketball compatibility model.

Scores how well a **specific player** fits a team's style using KNN on
a combined 12-dimensional vector that concatenates:
  - 6-d team style vector  (pace, 3PAR, assists, star usage, ORtg, DRtg)
  - 6-d player style vector (position bucket, USG%, scoring efficiency,
    pace preference, 3-point tendency, defensive contribution)

This makes the score player-specific: a fast PG gets a different
compatibility value than a post-up C even when evaluating the same team.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from basketball_ai.features.team_features import compute_team_style_vector
from basketball_ai.data.models import Team
from basketball_ai.data.loader import _to_int

logger = logging.getLogger(__name__)

# Position group → numeric bucket used in the player style vector
_POSITION_BUCKET: Dict[str, float] = {
    "PG": 0.0, "PG/SG": 0.1,
    "SG": 0.3, "SG/SF": 0.4, "SG/PF": 0.4,
    "SF": 0.6, "SF/PF": 0.7,
    "PF": 0.8, "PF/C": 0.9,
    "C": 1.0,
}


class CompatibilityModel:
    """KNN-based player–team style compatibility scorer (0–1).

    The KNN is trained on 12-d (player_style ∥ team_style) vectors so the
    model captures *interaction* between player attributes and team identity,
    making every player receive a distinct compatibility score for the same team.
    """

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

    def _player_style_vector(self, player_id: int, data: Dict[str, Any]) -> np.ndarray:
        """Build a 6-d style vector that characterises the *player's* attributes.

        Dimensions (all normalised to [0, 1]):
          0 – position bucket (PG=0 … C=1)
          1 – usage rate (0–40 %)
          2 – scoring efficiency (TS%)
          3 – pace preference (pts_per_36 proxy for tempo appetite)
          4 – 3-point tendency (3PA / FGA proxy)
          5 – defensive contribution (dbpm + 5, clipped)
        """
        pid = _to_int(player_id)
        player_row = data["player_dict"].get(pid, {})
        position   = str(player_row.get("position", "PG"))
        pos_bucket = _POSITION_BUCKET.get(position, _POSITION_BUCKET.get(position.split("/")[0], 0.5))

        stats_df = data["player_stats"]
        p_stats  = stats_df[stats_df["player_id"] == pid]

        def _mean(col: str, fallback: float) -> float:
            if p_stats.empty or col not in p_stats.columns:
                return fallback
            vals = p_stats[col].dropna()
            return float(vals.mean()) if not vals.empty else fallback

        usg       = np.clip(_mean("usg_pct", 18.0) / 40.0, 0.0, 1.0)
        ts        = np.clip(_mean("ts_pct",  0.52),         0.0, 1.0)
        pts36     = np.clip(_mean("points",  12.0) / 40.0,  0.0, 1.0)
        three_par = np.clip(_mean("three_par", 0.30),        0.0, 1.0)
        dbpm      = np.clip((_mean("dbpm", 0.0) + 5.0) / 10.0, 0.0, 1.0)

        return np.array([pos_bucket, usg, ts, pts36, three_par, dbpm], dtype=float)

    def _combined_vector(self, player_id: int, team_row: Dict, data: Dict[str, Any]) -> np.ndarray:
        """12-d combined (player ∥ team) feature vector."""
        pv = self._player_style_vector(player_id, data)
        tv = self._team_style_vector(team_row)
        return np.concatenate([pv, tv])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> None:
        """Train KNN on (player_style ∥ team_style, compatibility_score) pairs."""
        logger.info("[CompatibilityModel] Building player-specific compatibility training data …")
        player_stats = data["player_stats"]
        team_dict    = data["team_dict"]

        player_mean = player_stats.groupby("player_id")["rating"].mean().to_dict()

        X_rows: List[np.ndarray] = []
        y_vals: List[float]      = []

        for _, stat in player_stats.iterrows():
            if pd.isna(stat["team_id"]) or pd.isna(stat["player_id"]):
                continue
            tid  = _to_int(stat["team_id"])
            team = team_dict.get(tid)
            if team is None:
                continue
            pid    = _to_int(stat["player_id"])
            cv     = self._combined_vector(pid, team, data)
            p_mean = player_mean.get(pid, 6.5)
            compat = float(np.clip((float(stat["rating"]) - p_mean + 1.5) / 3.0, 0.0, 1.0))
            X_rows.append(cv)
            y_vals.append(compat)

        if not X_rows:
            self.is_trained = False
            return

        X = np.array(X_rows)
        y = np.array(y_vals)
        X_sc = self.scaler.fit_transform(X)
        self.knn.fit(X_sc, y)
        self.is_trained = True
        logger.info("[CompatibilityModel] Trained on %d samples.", len(X))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, player_id: int, team_id: int, data: Dict[str, Any]) -> float:
        """Return player-specific compatibility score (0–1).

        Args:
            player_id: Target player identifier.
            team_id:   Target team identifier.
            data:      Full data dict from loader.load_all_data().

        Returns:
            Float in [0.50, 1.00]; 0.75 when model is not trained.
        """
        if not self.is_trained:
            return 0.75
        team = data["team_dict"].get(_to_int(team_id))
        if team is None:
            return 0.75
        cv    = self._combined_vector(player_id, team, data).reshape(1, -1)
        cv_sc = self.scaler.transform(cv)
        return float(np.clip(self.knn.predict(cv_sc)[0], 0.50, 1.0))

    def score_style_vector(self, style_vector: np.ndarray) -> float:
        """Score a raw combined style vector directly (legacy / testing helper).

        When the vector has 6 dimensions (team-only, old interface), it is
        zero-padded to 12 dimensions so existing callers don't break.
        """
        if not self.is_trained:
            return 0.75
        if style_vector.shape[-1] == 6:
            # Legacy team-only vector: prepend neutral player vector (all 0.5)
            neutral_player = np.full(6, 0.5)
            style_vector   = np.concatenate([neutral_player, style_vector.flatten()])
        sv_sc = self.scaler.transform(style_vector.reshape(1, -1))
        return float(np.clip(self.knn.predict(sv_sc)[0], 0.50, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"knn": self.knn, "scaler": self.scaler}, path)
        logger.info("[CompatibilityModel] Saved to %s", path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.knn    = obj["knn"]
        self.scaler = obj["scaler"]
        self.is_trained = True
        logger.info("[CompatibilityModel] Loaded from %s", path)

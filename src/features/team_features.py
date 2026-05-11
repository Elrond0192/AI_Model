"""Team-level feature engineering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_team_features(
    team_id: int,
    data: Dict[str, Any],
    exclude_player_id: Optional[int] = None,
) -> Dict[str, float]:
    """Compute style vector and teammate quality features for a team.

    Args:
        team_id: Team identifier.
        data: Loaded data dictionary.
        exclude_player_id: Omit this player when computing teammate ratings.

    Returns:
        Feature dictionary.
    """
    team_dict = data["team_dict"]
    player_stats = data["player_stats"]

    team = team_dict.get(int(team_id))
    if team is None:
        return _default_team_features()

    # Normalised style vector
    poss_norm = float(team.get("avg_possession", 50.0)) / 70.0
    press_norm = float(team.get("pressing_intensity", 5.0)) / 10.0
    def_norm = float(team.get("defensive_line", 5.0)) / 10.0
    tempo_norm = float(team.get("passing_tempo", 5.0)) / 10.0

    # Average teammate rating (last 3 seasons of data)
    recent_stats = player_stats[
        (player_stats["team_id"] == team_id) &
        (player_stats["season"] >= 2021)
    ]
    if exclude_player_id is not None:
        recent_stats = recent_stats[recent_stats["player_id"] != exclude_player_id]

    if not recent_stats.empty:
        avg_teammate_rating = float(
            recent_stats.groupby("player_id")["rating"].mean().mean()
        )
        percentile = float(
            (recent_stats.groupby("player_id")["rating"].mean() <= avg_teammate_rating).mean()
        )
    else:
        avg_teammate_rating = 6.5
        percentile = 0.5

    tier = int(team.get("league_tier", 3))
    tier_map = {1: 1.00, 2: 0.90, 3: 0.80, 4: 0.70, 5: 0.60}
    tier_factor = tier_map.get(tier, 0.75)

    return {
        "style_possession": np.clip(poss_norm, 0.0, 1.0),
        "style_pressing": np.clip(press_norm, 0.0, 1.0),
        "style_def_line": np.clip(def_norm, 0.0, 1.0),
        "style_passing": np.clip(tempo_norm, 0.0, 1.0),
        "avg_teammate_rating": avg_teammate_rating,
        "teammate_quality_percentile": percentile,
        "league_tier_factor": tier_factor,
        "league_tier": float(tier),
    }


def style_similarity(team_a: Dict[str, float], team_b: Dict[str, float]) -> float:
    """Cosine similarity of the style vectors of two teams."""
    keys = ["style_possession", "style_pressing", "style_def_line", "style_passing"]
    va = np.array([team_a.get(k, 0.5) for k in keys])
    vb = np.array([team_b.get(k, 0.5) for k in keys])
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-9:
        return 0.5
    return float(np.dot(va, vb) / denom)


def _default_team_features() -> Dict[str, float]:
    return {
        "style_possession": 0.7,
        "style_pressing": 0.5,
        "style_def_line": 0.5,
        "style_passing": 0.5,
        "avg_teammate_rating": 6.5,
        "teammate_quality_percentile": 0.5,
        "league_tier_factor": 1.0,
        "league_tier": 1.0,
    }

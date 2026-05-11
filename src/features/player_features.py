"""Player-level feature engineering for the performance model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PEAK_AGES: Dict[str, int] = {
    "GK": 31, "CB": 29, "FB": 26, "CM": 28, "AM": 27, "W": 25, "ST": 27,
}

POSITION_ENCODING: Dict[str, int] = {
    "GK": 0, "CB": 1, "FB": 2, "CM": 3, "AM": 4, "W": 5, "ST": 6,
}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def compute_form_score(history: pd.DataFrame) -> float:
    """Weighted average of last 3 seasons (most recent = highest weight)."""
    recent = history.tail(3)
    if recent.empty:
        return 6.0
    n = len(recent)
    raw_weights = np.array([0.2, 0.3, 0.5][:n])
    weights = raw_weights / raw_weights.sum()
    return float(np.average(recent["rating"].values, weights=weights))


def compute_consistency_score(history: pd.DataFrame) -> float:
    """1 / (1 + std_dev) of ratings – higher = more consistent."""
    if len(history) < 2:
        return 0.5
    return float(1.0 / (1.0 + history["rating"].std()))


def compute_per90(history: pd.DataFrame) -> Dict[str, float]:
    """Per-90-minute stats averaged over available history."""
    total_min = history["minutes"].sum()
    if total_min < 1:
        return {"goals_per_90": 0.0, "assists_per_90": 0.0,
                "xG_per_90": 0.0, "xA_per_90": 0.0}
    m90 = total_min / 90.0
    return {
        "goals_per_90": float(history["goals"].sum() / m90),
        "assists_per_90": float(history["assists"].sum() / m90),
        "xG_per_90": float(history["xG"].sum() / m90),
        "xA_per_90": float(history["xA"].sum() / m90),
    }


def compute_technical_score(row: pd.Series) -> float:
    """Composite score from pass_accuracy, dribbles, key_passes."""
    pa = float(row.get("pass_accuracy", 75)) / 100.0
    dr = min(float(row.get("dribbles", 1.0)) / 5.0, 1.0)
    kp = min(float(row.get("key_passes", 0.5)) / 3.0, 1.0)
    return float(pa * 0.4 + dr * 0.3 + kp * 0.3)


def compute_defensive_score(row: pd.Series) -> float:
    """Composite score from tackles, interceptions, aerial duels won."""
    tk = min(float(row.get("tackles", 1.0)) / 5.0, 1.0)
    ic = min(float(row.get("interceptions", 0.5)) / 3.0, 1.0)
    ae = min(float(row.get("aerial_duels_won", 1.0)) / 6.0, 1.0)
    return float(tk * 0.4 + ic * 0.3 + ae * 0.3)


def compute_career_trajectory(history: pd.DataFrame) -> float:
    """Slope of rating over the last 4 seasons (positive = improving)."""
    last_4 = history.tail(4)
    if len(last_4) < 2:
        return 0.0
    x = np.arange(len(last_4), dtype=float)
    slope = float(np.polyfit(x, last_4["rating"].values.astype(float), 1)[0])
    return slope


def compute_age_vs_peak(age: int, position: str) -> float:
    """Current age relative to positional peak (negative = still growing)."""
    return float(age - PEAK_AGES.get(position, 27))


def compute_player_features(
    player_id: int,
    data: Dict[str, Any],
    season: int = 2024,
    target_age: Optional[int] = None,
) -> Dict[str, float]:
    """Compute all player-level features for model inference.

    Args:
        player_id: Player identifier.
        data: Loaded data dictionary from loader.load_all_data().
        season: Reference season for "current" context.
        target_age: Override age for trajectory projections.

    Returns:
        Feature dictionary with float values.
    """
    player_dict = data["player_dict"]
    player_stats = data["player_stats"]

    player = player_dict.get(player_id)
    if player is None:
        return _default_player_features()

    age = target_age if target_age is not None else int(player.get("age", 27))
    position = str(player.get("position", "CM"))

    history = player_stats[player_stats["player_id"] == player_id].sort_values("season")
    pre_history = history[history["season"] < season]
    if pre_history.empty:
        pre_history = history

    form_score = compute_form_score(pre_history)
    consistency = compute_consistency_score(pre_history)
    per90 = compute_per90(pre_history)

    last_row = pre_history.iloc[-1] if not pre_history.empty else history.iloc[-1]
    tech = compute_technical_score(last_row)
    dfn = compute_defensive_score(last_row)
    peak_rating = float(history["rating"].max()) if not history.empty else 6.5
    trajectory = compute_career_trajectory(pre_history)
    age_vs_peak = compute_age_vs_peak(age, position)

    return {
        "form_score": form_score,
        "consistency_score": consistency,
        "goals_per_90": per90["goals_per_90"],
        "assists_per_90": per90["assists_per_90"],
        "xG_per_90": per90["xG_per_90"],
        "xA_per_90": per90["xA_per_90"],
        "technical_score": tech,
        "defensive_score": dfn,
        "peak_rating": peak_rating,
        "career_trajectory": trajectory,
        "age_vs_peak": age_vs_peak,
        "pass_accuracy": float(last_row.get("pass_accuracy", 75.0)),
        "dribbles": float(last_row.get("dribbles", 1.0)),
        "tackles": float(last_row.get("tackles", 1.0)),
        "interceptions": float(last_row.get("interceptions", 0.5)),
        "aerial_duels_won": float(last_row.get("aerial_duels_won", 1.0)),
        "progressive_passes": float(last_row.get("progressive_passes", 2.0)),
        "key_passes": float(last_row.get("key_passes", 0.5)),
        "minutes": float(last_row.get("minutes", 1500.0)),
        "matches_played": float(last_row.get("matches_played", 20)),
        "age": float(age),
        "position_enc": float(POSITION_ENCODING.get(position, 3)),
    }


def _default_player_features() -> Dict[str, float]:
    """Return neutral features for an unknown player (cold-start)."""
    return {
        "form_score": 6.5,
        "consistency_score": 0.5,
        "goals_per_90": 0.0,
        "assists_per_90": 0.0,
        "xG_per_90": 0.0,
        "xA_per_90": 0.0,
        "technical_score": 0.5,
        "defensive_score": 0.5,
        "peak_rating": 6.5,
        "career_trajectory": 0.0,
        "age_vs_peak": 0.0,
        "pass_accuracy": 75.0,
        "dribbles": 1.0,
        "tackles": 1.0,
        "interceptions": 0.5,
        "aerial_duels_won": 1.0,
        "progressive_passes": 2.0,
        "key_passes": 0.5,
        "minutes": 1500.0,
        "matches_played": 20.0,
        "age": 25.0,
        "position_enc": 3.0,
    }

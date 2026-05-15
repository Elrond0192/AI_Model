"""Basketball context feature engineering.

Computes player–team fit based on:
  - Position-to-team-style compatibility
  - Role opportunity (is there a starter slot available?)
  - League adaptation factor (tier difference, same country bonus)
  - Spacing fit (3pt shooter in pace-and-space team)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from basketball_ai.features.team_features import get_style_position_compat
from basketball_ai.utils.helpers import normalize_id as _normalize_id

# ---------------------------------------------------------------------------
# League adaptation
# ---------------------------------------------------------------------------

def compute_league_adaptation(
    from_tier: int,
    to_tier: int,
    same_country: bool = False,
) -> float:
    """Return an adaptation multiplier when a player moves between leagues.

    Moving *up* (lower tier number = tougher) incurs a penalty.
    Moving *down* gives a small bonus. Same-country move eases adaptation.
    """
    tier_diff = to_tier - from_tier  # negative = moving to tougher league
    if tier_diff < 0:
        factor = 1.0 - abs(tier_diff) * 0.06
    else:
        factor = 1.0 + tier_diff * 0.02

    if same_country:
        factor += 0.02

    return float(np.clip(factor, 0.70, 1.05))


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

def compute_context_features(
    player_id: int,
    team_id: int,
    data: Dict[str, Any],
) -> Dict[str, float]:
    """Compute player–team context feature dictionary.

    Args:
        player_id: Target player identifier.
        team_id:   Target team identifier.
        data:      Flat data dict from loader.load_all_data().

    Returns:
        Dict with keys: position_team_fit, style_compatibility,
        role_opportunity, league_adaptation_factor, spacing_fit.
    """
    player_row = data["player_dict"].get(_normalize_id(player_id))
    team_row   = data["team_dict"].get(_normalize_id(team_id))

    if player_row is None or team_row is None:
        return _default_context_features()

    position = str(player_row.get("position", "PG"))
    style    = str(team_row.get("playing_style", "motion_offense"))

    # 1. Style × position compatibility
    style_compat = get_style_position_compat(style, position)

    # 2. Position fit (computed after role_opportunity below, since it needs
    #    same_pos_count from the starter-slot calculation).

    # 3. Role opportunity: starters at same position on target team
    rels_df = data["team_player_relations"]
    players_df = data["players"]
    starter_mask = (
        (rels_df["team_id"] == _normalize_id(team_id)) &
        (rels_df["season"] == "2023-24") &
        (rels_df["role"] == "starter")
    )
    starter_ids = rels_df[starter_mask]["player_id"].tolist()
    primary_pos = position.split("/")[0]
    if starter_ids:
        # Count starters who share at least one position component
        same_pos = players_df[
            players_df["id"].isin(starter_ids) &
            players_df["position"].str.contains(primary_pos, regex=False)
        ]
        same_pos_count = len(same_pos)
    else:
        same_pos_count = 0
    role_opportunity = float(np.clip(1.0 - same_pos_count * 0.18, 0.35, 1.0))

    # position_fit: weighted blend of style×position compat and slot availability.
    # Gives position_team_fit a genuinely different meaning from style_compatibility,
    # preventing the ctx_score from double-counting the same signal.
    position_fit = float(np.clip(style_compat * 0.60 + role_opportunity * 0.40, 0.0, 1.0))

    # 4. League adaptation
    from_league_id = player_row.get("current_league_id")
    to_league_id   = _normalize_id(team_row.get("league_id"))
    from_league    = data["league_dict"].get(_normalize_id(from_league_id), {})
    to_league      = data["league_dict"].get(to_league_id, {})
    from_tier = int(from_league.get("tier", 1))
    to_tier   = int(to_league.get("tier", 1))
    same_country = (from_league.get("country", "") == to_league.get("country", ""))
    adaptation = compute_league_adaptation(from_tier, to_tier, same_country)

    # 5. Spacing fit: 3pt specialist in pace-and-space/motion_offense gets a bonus
    spacing_fit = 0.5  # neutral
    if style in ("pace_and_space", "motion_offense"):
        # Check player's 3pt ability from latest stats
        stats_df = data["player_stats"]
        p_stats = stats_df[stats_df["player_id"] == _normalize_id(player_id)].sort_values("season")
        if not p_stats.empty:
            tpp = float(p_stats.iloc[-1].get("three_point_pct", 0.33))
            spacing_fit = float(np.clip(tpp / 0.40, 0, 1))

    return {
        "position_team_fit":      round(position_fit, 4),
        "style_compatibility":    round(style_compat, 4),
        "role_opportunity":       round(role_opportunity, 4),
        "league_adaptation_factor": round(adaptation, 4),
        "spacing_fit":            round(spacing_fit, 4),
    }


def _default_context_features() -> Dict[str, float]:
    return {
        "position_team_fit":       0.80,
        "style_compatibility":     0.80,
        "role_opportunity":        0.80,
        "league_adaptation_factor": 1.00,
        "spacing_fit":             0.50,
    }

"""Context features: player–team fit, style compatibility, league adaptation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# Formation → position fit scores
_FORMATION_FIT: Dict[str, Dict[str, float]] = {
    "GK": {"4-4-2": 1.0, "4-3-3": 1.0, "4-2-3-1": 1.0, "3-5-2": 1.0,
           "3-4-3": 1.0, "5-3-2": 1.0, "4-5-1": 1.0, "4-1-4-1": 1.0},
    "CB": {"4-4-2": 0.90, "4-3-3": 0.90, "4-2-3-1": 0.90, "3-5-2": 1.00,
           "3-4-3": 1.00, "5-3-2": 1.00, "4-5-1": 0.85, "4-1-4-1": 0.88},
    "FB": {"4-4-2": 0.90, "4-3-3": 0.95, "4-2-3-1": 0.95, "3-5-2": 0.85,
           "3-4-3": 0.80, "5-3-2": 0.85, "4-5-1": 0.90, "4-1-4-1": 0.90},
    "CM": {"4-4-2": 0.95, "4-3-3": 1.00, "4-2-3-1": 0.90, "3-5-2": 0.95,
           "3-4-3": 0.90, "5-3-2": 0.90, "4-5-1": 0.95, "4-1-4-1": 0.95},
    "AM": {"4-2-3-1": 1.00, "4-3-3": 0.90, "4-4-2": 0.85, "3-5-2": 0.85,
           "3-4-3": 0.90, "5-3-2": 0.80, "4-5-1": 0.85, "4-1-4-1": 1.00},
    "W":  {"4-3-3": 1.00, "4-2-3-1": 0.90, "3-4-3": 0.95, "4-4-2": 0.85,
           "3-5-2": 0.80, "5-3-2": 0.85, "4-5-1": 0.85, "4-1-4-1": 0.90},
    "ST": {"4-4-2": 1.00, "4-3-3": 0.95, "4-2-3-1": 0.95, "3-5-2": 0.95,
           "3-4-3": 0.90, "5-3-2": 1.00, "4-5-1": 0.90, "4-1-4-1": 0.90},
}

# (playing_style, position) → style compatibility score
_STYLE_COMPAT: Dict[str, Dict[str, float]] = {
    "possession": {"GK": 0.75, "CB": 0.90, "FB": 0.85, "CM": 0.95,
                   "AM": 0.90, "W": 0.80, "ST": 0.75},
    "high_press": {"GK": 0.70, "CB": 0.80, "FB": 0.85, "CM": 0.90,
                   "AM": 0.85, "W": 0.90, "ST": 0.90},
    "counter":    {"GK": 0.80, "CB": 0.85, "FB": 0.88, "CM": 0.80,
                   "AM": 0.78, "W": 0.92, "ST": 0.92},
    "direct":     {"GK": 0.80, "CB": 0.88, "FB": 0.80, "CM": 0.75,
                   "AM": 0.72, "W": 0.85, "ST": 0.90},
}


def compute_position_fit(position: str, formation: str, style: str) -> float:
    """Return how well a position suits a formation+style."""
    fits = _FORMATION_FIT.get(position, {})
    base = fits.get(formation, 0.80)
    style_boost = _STYLE_COMPAT.get(style, {}).get(position, 0.75) - 0.75
    return float(np.clip(base + style_boost * 0.2, 0.0, 1.0))


def compute_league_adaptation(
    from_tier: int,
    to_tier: int,
    nationality_match: bool = False,
) -> float:
    """Adaptation factor when moving between leagues.

    Moving up (higher tier) is harder; moving down is easier.
    """
    tier_diff = to_tier - from_tier  # negative = stepping up
    if tier_diff < 0:
        # Moving to a tougher league
        penalty = abs(tier_diff) * 0.06
        factor = 1.0 - penalty
    else:
        # Same tier or easier league
        factor = 1.0 + tier_diff * 0.02

    if nationality_match:
        factor += 0.02  # same country → easier adaptation

    return float(np.clip(factor, 0.70, 1.05))


def compute_context_features(
    player_id: int,
    team_id: int,
    data: Dict[str, Any],
) -> Dict[str, float]:
    """Compute player-team context features.

    Args:
        player_id: Player identifier.
        team_id: Target team identifier.
        data: Loaded data dictionary.

    Returns:
        Dictionary of context feature floats.
    """
    player_dict = data["player_dict"]
    team_dict = data["team_dict"]
    league_dict = data["league_dict"]
    team_player_relations = data["team_player_relations"]
    players_df = data["players"]

    player = player_dict.get(int(player_id))
    team = team_dict.get(int(team_id))

    if player is None or team is None:
        return _default_context_features()

    position = str(player.get("position", "CM"))
    formation = str(team.get("formation", "4-3-3"))
    style = str(team.get("playing_style", "possession"))

    position_fit = compute_position_fit(position, formation, style)
    style_compat = _STYLE_COMPAT.get(style, {}).get(position, 0.75)

    # Role opportunity: fewer starters at the same position → more opportunity
    current_starters = team_player_relations[
        (team_player_relations["team_id"] == team_id) &
        (team_player_relations["season"] == 2024) &
        (team_player_relations["role"] == "starter")
    ]
    if not current_starters.empty:
        starter_ids = current_starters["player_id"].tolist()
        same_pos_count = int(players_df[
            (players_df["id"].isin(starter_ids)) &
            (players_df["position"] == position)
        ].shape[0])
    else:
        same_pos_count = 0

    role_opportunity = float(np.clip(1.0 - same_pos_count * 0.18, 0.35, 1.0))

    # League adaptation
    from_league_id = int(player.get("current_league_id", 1))
    to_league_id = int(team.get("league_id", 1))
    from_league = league_dict.get(from_league_id, {})
    to_league = league_dict.get(to_league_id, {})
    from_tier = int(from_league.get("tier", 1))
    to_tier = int(to_league.get("tier", 1))
    nat_match = (
        from_league.get("country", "") == to_league.get("country", "")
    )
    adaptation = compute_league_adaptation(from_tier, to_tier, nat_match)

    return {
        "position_team_fit": position_fit,
        "style_compatibility": style_compat,
        "role_opportunity": role_opportunity,
        "league_adaptation_factor": adaptation,
    }


def _default_context_features() -> Dict[str, float]:
    return {
        "position_team_fit": 0.80,
        "style_compatibility": 0.80,
        "role_opportunity": 0.80,
        "league_adaptation_factor": 1.0,
    }

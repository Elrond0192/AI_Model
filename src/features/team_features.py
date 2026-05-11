"""Basketball team feature engineering."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

STYLES = ["pace_and_space","pick_and_roll","isolation","defensive","motion_offense","post_up"]

# style → ideal player attributes (higher = more compatible)
_STYLE_POS_COMPAT: Dict[str, Dict[str, float]] = {
    "pace_and_space":  {"PG":0.95,"SG":0.90,"SF":0.85,"PF":0.80,"C":0.65,
                        "PG/SG":0.92,"SG/SF":0.88,"SF/PF":0.82,"PF/C":0.70,"SG/PF":0.84},
    "pick_and_roll":   {"PG":1.00,"SG":0.80,"SF":0.75,"PF":0.85,"C":0.90,
                        "PG/SG":0.90,"SG/SF":0.78,"SF/PF":0.80,"PF/C":0.88,"SG/PF":0.80},
    "isolation":       {"PG":0.85,"SG":0.95,"SF":0.90,"PF":0.70,"C":0.60,
                        "PG/SG":0.90,"SG/SF":0.93,"SF/PF":0.80,"PF/C":0.65,"SG/PF":0.82},
    "defensive":       {"PG":0.80,"SG":0.80,"SF":0.85,"PF":0.90,"C":0.95,
                        "PG/SG":0.80,"SG/SF":0.83,"SF/PF":0.88,"PF/C":0.93,"SG/PF":0.85},
    "motion_offense":  {"PG":0.90,"SG":0.88,"SF":0.92,"PF":0.80,"C":0.70,
                        "PG/SG":0.89,"SG/SF":0.90,"SF/PF":0.86,"PF/C":0.75,"SG/PF":0.86},
    "post_up":         {"PG":0.65,"SG":0.70,"SF":0.75,"PF":0.90,"C":1.00,
                        "PG/SG":0.67,"SG/SF":0.73,"SF/PF":0.83,"PF/C":0.95,"SG/PF":0.78},
}

def get_style_position_compat(style: str, position: str) -> float:
    """Return compatibility score (0-1) between a team style and a player position."""
    return _STYLE_POS_COMPAT.get(style, {}).get(position, 0.75)

def compute_team_style_vector(team) -> np.ndarray:
    """Build a normalized 6-d style vector from a Team object.

    Dimensions: [pace_norm, three_pt_rate, ast_norm, star_usage, ortg_norm, drtg_inv_norm]
    """
    pace_norm  = float(np.clip((team.pace - 80) / 35, 0, 1))
    tpar_norm  = float(np.clip(team.three_point_attempt_rate / 0.55, 0, 1))
    ast_norm   = float(np.clip((team.assists_per_game - 14) / 21, 0, 1))
    star_norm  = float(np.clip(team.star_player_usage / 0.45, 0, 1))
    ortg_norm  = float(np.clip((team.offensive_rating - 90) / 35, 0, 1))
    drtg_norm  = float(np.clip(1 - (team.defensive_rating - 88) / 32, 0, 1))
    return np.array([pace_norm, tpar_norm, ast_norm, star_norm, ortg_norm, drtg_norm])

def compute_team_features(
    team,
    all_player_stats: List[Any],
    all_relations: List[Any],
    exclude_player_id: Optional[int] = None,
) -> Dict[str, float]:
    """Compute feature dictionary for a team.

    Args:
        team: Team dataclass instance.
        all_player_stats: List of PlayerStats objects.
        all_relations: List of TeamPlayerRelation objects.
        exclude_player_id: Skip this player when computing teammate ratings.

    Returns:
        Feature dictionary.
    """
    # Teammate quality
    roster_ids = {r.player_id for r in all_relations
                  if r.team_id == team.id and r.season == "2023-24"}
    if exclude_player_id is not None:
        roster_ids.discard(exclude_player_id)

    teammate_ratings = []
    for s in all_player_stats:
        if s.player_id in roster_ids and s.season == "2023-24":
            teammate_ratings.append(s.rating)

    if teammate_ratings:
        avg_tm = float(np.mean(teammate_ratings))
    else:
        avg_tm = 6.0

    tier_map = {1: 1.00, 2: 0.88, 3: 0.76, 4: 0.64, 5: 0.52}
    tier_factor = tier_map.get(team.league_tier, 0.70)

    pace_cat = "fast" if team.pace > 100 else ("slow" if team.pace < 93 else "medium")

    v = compute_team_style_vector(team)

    return {
        "style_pace":          float(v[0]),
        "style_3pt_rate":      float(v[1]),
        "style_ast":           float(v[2]),
        "style_star_usage":    float(v[3]),
        "style_ortg":          float(v[4]),
        "style_drtg":          float(v[5]),
        "avg_teammate_rating": round(avg_tm, 3),
        "teammate_quality_pct": float(np.clip(avg_tm / 10, 0, 1)),
        "league_tier_factor":  tier_factor,
        "league_tier":         float(team.league_tier),
        "team_pace":           team.pace,
        "team_pace_cat_fast":  float(pace_cat == "fast"),
        "team_pace_cat_slow":  float(pace_cat == "slow"),
    }

def style_vector_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two style vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.5
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0, 1))

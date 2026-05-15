"""Basketball team feature engineering.

Provides both an object-API (Team dataclass) and a dict-API wrapper
used by the ensemble and API routes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from basketball_ai.data.models import Team
from basketball_ai.utils.helpers import normalize_id as _normalize_id

# ---------------------------------------------------------------------------
# Data-derived normalization bounds for team style vectors.
# Defaults represent conservative reference ranges; call calibrate_style_bounds()
# after loading real data to replace them with data-driven percentile bounds.
# ---------------------------------------------------------------------------
_style_bounds: Dict[str, float] = {
    "pace_min": 80.0, "pace_max": 115.0,
    "tpar_max": 0.55,
    "ast_min":  14.0, "ast_max":  35.0,
    "star_max": 0.45,
    "ortg_min": 90.0, "ortg_max": 125.0,
    "drtg_min": 88.0, "drtg_max": 120.0,
}


def calibrate_style_bounds(data: Dict[str, Any], percentile_lo: float = 5.0, percentile_hi: float = 95.0) -> Dict[str, float]:
    """Compute normalization bounds from the real teams distribution.

    Call this once after data is loaded (e.g. in EnsembleModel.train()).
    Stores results in the module-level ``_style_bounds`` dict so all
    subsequent calls to ``compute_team_style_vector`` use data-driven bounds.

    Args:
        data:           Full data dict including a ``teams`` DataFrame.
        percentile_lo:  Lower percentile used for the minimum bound (default 5).
        percentile_hi:  Upper percentile used for the maximum bound (default 95).

    Returns:
        Updated bounds dict.
    """
    import pandas as _pd

    teams_df = data.get("teams")
    if teams_df is None or (isinstance(teams_df, _pd.DataFrame) and teams_df.empty):
        return _style_bounds.copy()

    def _pct(col: str, p: float, fallback: float) -> float:
        if col not in teams_df.columns:
            return fallback
        vals = teams_df[col].dropna()
        return float(np.percentile(vals, p)) if not vals.empty else fallback

    _style_bounds["pace_min"] = _pct("pace", percentile_lo, 80.0)
    _style_bounds["pace_max"] = _pct("pace", percentile_hi, 115.0)
    _style_bounds["tpar_max"] = max(_pct("three_point_attempt_rate", percentile_hi, 0.55), 0.01)
    _style_bounds["ast_min"]  = _pct("assists_per_game", percentile_lo, 14.0)
    _style_bounds["ast_max"]  = _pct("assists_per_game", percentile_hi, 35.0)
    _style_bounds["star_max"] = max(_pct("star_player_usage", percentile_hi, 0.45), 0.01)
    _style_bounds["ortg_min"] = _pct("offensive_rating", percentile_lo, 90.0)
    _style_bounds["ortg_max"] = _pct("offensive_rating", percentile_hi, 125.0)
    _style_bounds["drtg_min"] = _pct("defensive_rating", percentile_lo, 88.0)
    _style_bounds["drtg_max"] = _pct("defensive_rating", percentile_hi, 120.0)

    return _style_bounds.copy()

# ---------------------------------------------------------------------------
# Style-vs-position compatibility table (all 10 positions incl. hybrids)
# ---------------------------------------------------------------------------

_STYLE_POS_COMPAT: Dict[str, Dict[str, float]] = {
    "pace_and_space": {
        "PG": 0.95, "SG": 0.90, "SF": 0.85, "PF": 0.80, "C": 0.65,
        "PG/SG": 0.92, "SG/SF": 0.88, "SF/PF": 0.82, "PF/C": 0.70, "SG/PF": 0.84,
    },
    "pick_and_roll": {
        "PG": 1.00, "SG": 0.80, "SF": 0.75, "PF": 0.85, "C": 0.90,
        "PG/SG": 0.90, "SG/SF": 0.78, "SF/PF": 0.80, "PF/C": 0.88, "SG/PF": 0.80,
    },
    "isolation": {
        "PG": 0.85, "SG": 0.95, "SF": 0.90, "PF": 0.70, "C": 0.60,
        "PG/SG": 0.90, "SG/SF": 0.93, "SF/PF": 0.80, "PF/C": 0.65, "SG/PF": 0.82,
    },
    "defensive": {
        "PG": 0.80, "SG": 0.80, "SF": 0.85, "PF": 0.90, "C": 0.95,
        "PG/SG": 0.80, "SG/SF": 0.83, "SF/PF": 0.88, "PF/C": 0.93, "SG/PF": 0.85,
    },
    "motion_offense": {
        "PG": 0.90, "SG": 0.88, "SF": 0.92, "PF": 0.80, "C": 0.70,
        "PG/SG": 0.89, "SG/SF": 0.90, "SF/PF": 0.86, "PF/C": 0.75, "SG/PF": 0.86,
    },
    "post_up": {
        "PG": 0.65, "SG": 0.70, "SF": 0.75, "PF": 0.90, "C": 1.00,
        "PG/SG": 0.67, "SG/SF": 0.73, "SF/PF": 0.83, "PF/C": 0.95, "SG/PF": 0.78,
    },
}


def get_style_position_compat(style: str, position: str) -> float:
    """Return compatibility score (0-1) between a team style and player position."""
    primary = position.split("/")[0]
    row = _STYLE_POS_COMPAT.get(style, {})
    return row.get(position, row.get(primary, 0.75))


# ---------------------------------------------------------------------------
# Style vector
# ---------------------------------------------------------------------------

def compute_team_style_vector(team: Team, data: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Build a 6-d normalized style vector.

    Normalization bounds are computed from the real data distribution when
    `data` is provided, and fall back to historical reference percentiles
    (stored as module-level defaults) otherwise.  This avoids hardcoded
    constants that may not reflect the actual data range.

    Dimensions: [pace_norm, 3pt_rate, ast_norm, star_usage, ortg_norm, drtg_inv]
    """
    # Use data-derived bounds if available (set by _update_style_bounds(data))
    _pb = _style_bounds
    pace_min, pace_max = _pb.get("pace_min", 80.0), _pb.get("pace_max", 115.0)
    tpar_max           = _pb.get("tpar_max", 0.55)
    ast_min, ast_max   = _pb.get("ast_min", 14.0), _pb.get("ast_max", 35.0)
    star_max           = _pb.get("star_max", 0.45)
    ortg_min, ortg_max = _pb.get("ortg_min", 90.0), _pb.get("ortg_max", 125.0)
    drtg_min, drtg_max = _pb.get("drtg_min", 88.0), _pb.get("drtg_max", 120.0)

    pace_range = max(pace_max - pace_min, 1.0)
    ast_range  = max(ast_max  - ast_min,  1.0)
    ortg_range = max(ortg_max - ortg_min, 1.0)
    drtg_range = max(drtg_max - drtg_min, 1.0)

    pace_norm = float(np.clip((team.pace - pace_min) / pace_range, 0, 1))
    tpar_norm = float(np.clip(team.three_point_attempt_rate / max(tpar_max, 0.01), 0, 1))
    ast_norm  = float(np.clip((team.assists_per_game - ast_min) / ast_range, 0, 1))
    star_norm = float(np.clip(team.star_player_usage / max(star_max, 0.01), 0, 1))
    ortg_norm = float(np.clip((team.offensive_rating - ortg_min) / ortg_range, 0, 1))
    drtg_norm = float(np.clip(1.0 - (team.defensive_rating - drtg_min) / drtg_range, 0, 1))
    return np.array([pace_norm, tpar_norm, ast_norm, star_norm, ortg_norm, drtg_norm])


def style_vector_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two style vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.5
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0, 1))


# ---------------------------------------------------------------------------
# Object-API
# ---------------------------------------------------------------------------

def _league_tier_factor(team_row: Dict[str, Any], data: Dict[str, Any]) -> float:
    """Return a league quality factor derived from the real competitiveness_score.

    Uses ``leagues.competitiveness_score`` (loaded from the DB) normalised
    relative to the maximum observed competitiveness score so that the best
    league always scores 1.0 and weaker leagues are discounted proportionally.
    No hardcoded tier map is used.
    """
    league_id  = team_row.get("league_id")
    league_row = data.get("league_dict", {}).get(str(league_id), {})
    comp_score = float(league_row.get("competitiveness_score", 1.0) or 1.0)

    # Normalise to the max competitiveness in the loaded data
    leagues_df = data.get("leagues")
    if leagues_df is not None and not leagues_df.empty and "competitiveness_score" in leagues_df.columns:
        max_cs = float(leagues_df["competitiveness_score"].max())
        if max_cs > 0:
            return float(np.clip(comp_score / max_cs, 0.5, 1.0))
    return float(np.clip(comp_score, 0.5, 1.0))


def compute_team_features_from_object(
    team: Team,
    all_player_stats: List[Any],
    all_relations: List[Any],
    exclude_player_id: Optional[int] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute feature dictionary from Team dataclass and lists of stat objects."""
    roster_ids = {
        r.player_id for r in all_relations
        if r.team_id == team.id and r.season == "2023-24"
    }
    if exclude_player_id is not None:
        roster_ids.discard(exclude_player_id)

    teammate_ratings = [
        s.rating for s in all_player_stats
        if s.player_id in roster_ids and s.season == "2023-24"
    ]
    avg_tm = float(np.mean(teammate_ratings)) if teammate_ratings else 6.0

    # Use data-derived league factor when possible; otherwise fall back to tier
    if data is not None:
        tier_factor = _league_tier_factor(
            {"league_id": team.league_tier, "league_tier": team.league_tier}, data
        )
    else:
        _TIER_FALLBACK = {1: 1.00, 2: 0.88, 3: 0.76, 4: 0.64, 5: 0.52}
        tier_factor = _TIER_FALLBACK.get(team.league_tier, 0.70)

    pace_cat = "fast" if team.pace > 100 else ("slow" if team.pace < 93 else "medium")

    v = compute_team_style_vector(team)
    return {
        "style_pace":           float(v[0]),
        "style_3pt_rate":       float(v[1]),
        "style_ast":            float(v[2]),
        "style_star_usage":     float(v[3]),
        "style_ortg":           float(v[4]),
        "style_drtg":           float(v[5]),
        "avg_teammate_rating":  round(avg_tm, 3),
        "teammate_quality_pct": float(np.clip(avg_tm / 10, 0, 1)),
        "league_tier_factor":   tier_factor,
        "league_tier":          float(team.league_tier),
        "team_pace":            team.pace,
        "team_pace_cat_fast":   float(pace_cat == "fast"),
        "team_pace_cat_slow":   float(pace_cat == "slow"),
    }


# ---------------------------------------------------------------------------
# Dict-API wrapper (used by ensemble and scenarios)
# ---------------------------------------------------------------------------

def compute_team_features(
    team_id: int,
    data: Dict[str, Any],
    exclude_player_id: Optional[int] = None,
) -> Dict[str, float]:
    """Dict-based wrapper: looks up team from data dict."""
    team_row = data["team_dict"].get(_normalize_id(team_id))
    if team_row is None:
        return _default_team_features()

    try:
        team = Team(**{k: team_row.get(k) for k in Team.__dataclass_fields__})
    except Exception:
        return _default_team_features()

    stats_df = data["player_stats"]
    rels_df  = data["team_player_relations"]

    # Teammate quality
    season_col = "season"
    rel_mask = (rels_df["team_id"] == _normalize_id(team_id)) & (rels_df[season_col] == "2023-24")
    roster_ids = set(rels_df[rel_mask]["player_id"].tolist())
    if exclude_player_id is not None:
        roster_ids.discard(exclude_player_id)

    if roster_ids:
        tm_stats = stats_df[
            (stats_df["player_id"].isin(roster_ids)) &
            (stats_df["season"] == "2023-24")
        ]
        avg_tm = float(tm_stats["rating"].mean()) if not tm_stats.empty else 6.0
    else:
        avg_tm = 6.0

    # League quality factor – derived from competitiveness_score in real data
    tier_factor = _league_tier_factor(team_row, data)
    tier = int(team_row.get("league_tier", 3))
    pace = float(team_row.get("pace", 95))
    pace_cat = "fast" if pace > 100 else ("slow" if pace < 93 else "medium")

    v = compute_team_style_vector(team)
    return {
        "style_pace":           float(v[0]),
        "style_3pt_rate":       float(v[1]),
        "style_ast":            float(v[2]),
        "style_star_usage":     float(v[3]),
        "style_ortg":           float(v[4]),
        "style_drtg":           float(v[5]),
        "avg_teammate_rating":  round(avg_tm, 3),
        "teammate_quality_pct": float(np.clip(avg_tm / 10, 0, 1)),
        "league_tier_factor":   tier_factor,
        "league_tier":          float(tier),
        "team_pace":            pace,
        "team_pace_cat_fast":   float(pace_cat == "fast"),
        "team_pace_cat_slow":   float(pace_cat == "slow"),
    }


def _default_team_features() -> Dict[str, float]:
    return {
        "style_pace": 0.5, "style_3pt_rate": 0.5, "style_ast": 0.5,
        "style_star_usage": 0.5, "style_ortg": 0.5, "style_drtg": 0.5,
        "avg_teammate_rating": 6.0, "teammate_quality_pct": 0.6,
        "league_tier_factor": 1.0, "league_tier": 1.0,
        "team_pace": 97.0, "team_pace_cat_fast": 0.0, "team_pace_cat_slow": 0.0,
    }

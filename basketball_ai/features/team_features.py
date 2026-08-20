"""Basketball team feature engineering with source-season-aware roster context."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from basketball_ai.data.models import Team
from basketball_ai.utils.helpers import normalize_id as _normalize_id

# Data-derived normalization bounds. Production calibrates these on the
# pre-calibration historical snapshot and persists them with the model run.
_style_bounds: Dict[str, float] = {
    "pace_min": 80.0,
    "pace_max": 115.0,
    "tpar_max": 0.55,
    "ast_min": 14.0,
    "ast_max": 35.0,
    "star_max": 0.45,
    "ortg_min": 90.0,
    "ortg_max": 125.0,
    "drtg_min": 88.0,
    "drtg_max": 120.0,
}


def _season_year(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(str(value).split("-")[0])
    except (TypeError, ValueError):
        return None


def calibrate_style_bounds(
    data: Dict[str, Any],
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> Dict[str, float]:
    """Calibrate style-vector normalization from the supplied team snapshot."""
    teams_df = data.get("teams")
    if teams_df is None or getattr(teams_df, "empty", True):
        return _style_bounds.copy()

    def pct(column: str, percentile: float, fallback: float) -> float:
        if column not in teams_df.columns:
            return fallback
        values = pd.to_numeric(teams_df[column], errors="coerce").dropna()
        return float(np.percentile(values, percentile)) if not values.empty else fallback

    _style_bounds["pace_min"] = pct("pace", percentile_lo, 80.0)
    _style_bounds["pace_max"] = pct("pace", percentile_hi, 115.0)
    _style_bounds["tpar_max"] = max(
        pct("three_point_attempt_rate", percentile_hi, 0.55), 0.01
    )
    _style_bounds["ast_min"] = pct("assists_per_game", percentile_lo, 14.0)
    _style_bounds["ast_max"] = pct("assists_per_game", percentile_hi, 35.0)
    _style_bounds["star_max"] = max(
        pct("star_player_usage", percentile_hi, 0.45), 0.01
    )
    _style_bounds["ortg_min"] = pct("offensive_rating", percentile_lo, 90.0)
    _style_bounds["ortg_max"] = pct("offensive_rating", percentile_hi, 125.0)
    _style_bounds["drtg_min"] = pct("defensive_rating", percentile_lo, 88.0)
    _style_bounds["drtg_max"] = pct("defensive_rating", percentile_hi, 120.0)
    return _style_bounds.copy()


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
    primary = position.split("/")[0]
    row = _STYLE_POS_COMPAT.get(style, {})
    return row.get(position, row.get(primary, 0.75))


def compute_team_style_vector(
    team: Team,
    data: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build the six-dimensional style vector using the active run bounds."""
    del data  # calibration is explicit; inference must not mutate global bounds.
    bounds = _style_bounds
    pace_min, pace_max = bounds.get("pace_min", 80.0), bounds.get("pace_max", 115.0)
    ast_min, ast_max = bounds.get("ast_min", 14.0), bounds.get("ast_max", 35.0)
    ortg_min, ortg_max = bounds.get("ortg_min", 90.0), bounds.get("ortg_max", 125.0)
    drtg_min, drtg_max = bounds.get("drtg_min", 88.0), bounds.get("drtg_max", 120.0)

    pace_range = max(pace_max - pace_min, 1.0)
    ast_range = max(ast_max - ast_min, 1.0)
    ortg_range = max(ortg_max - ortg_min, 1.0)
    drtg_range = max(drtg_max - drtg_min, 1.0)
    return np.array(
        [
            np.clip((team.pace - pace_min) / pace_range, 0, 1),
            np.clip(team.three_point_attempt_rate / max(bounds.get("tpar_max", 0.55), 0.01), 0, 1),
            np.clip((team.assists_per_game - ast_min) / ast_range, 0, 1),
            np.clip(team.star_player_usage / max(bounds.get("star_max", 0.45), 0.01), 0, 1),
            np.clip((team.offensive_rating - ortg_min) / ortg_range, 0, 1),
            np.clip(1.0 - (team.defensive_rating - drtg_min) / drtg_range, 0, 1),
        ],
        dtype=float,
    )


def style_vector_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.5
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0, 1))


def _league_tier_factor(team_row: Dict[str, Any], data: Dict[str, Any]) -> float:
    """League quality relative to the best competitiveness score in this data."""
    league_id = team_row.get("league_id")
    league_dict = data.get("league_dict", {})
    normalised_id = _normalize_id(league_id)
    league_row = league_dict.get(normalised_id, league_dict.get(str(league_id), {}))
    comp_score = float(league_row.get("competitiveness_score", 1.0) or 1.0)
    leagues_df = data.get("leagues")
    if leagues_df is not None and not getattr(leagues_df, "empty", True):
        if "competitiveness_score" in leagues_df.columns:
            scores = pd.to_numeric(
                leagues_df["competitiveness_score"], errors="coerce"
            ).dropna()
            if not scores.empty and float(scores.max()) > 0:
                return float(np.clip(comp_score / float(scores.max()), 0.5, 1.0))
    return float(np.clip(comp_score, 0.5, 1.0))


def _latest_object_context_year(
    team_id: int,
    all_relations: List[Any],
    data: Optional[Dict[str, Any]],
) -> Optional[int]:
    as_of = data.get("_as_of_season") if data else None
    years = [
        _season_year(getattr(relation, "season", None))
        for relation in all_relations
        if getattr(relation, "team_id", None) == team_id
    ]
    years = [year for year in years if year is not None]
    if as_of is not None:
        years = [year for year in years if year <= int(as_of)]
    return max(years) if years else None


def compute_team_features_from_object(
    team: Team,
    all_player_stats: List[Any],
    all_relations: List[Any],
    exclude_player_id: Optional[int] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute team features from the latest roster available in context."""
    context_year = _latest_object_context_year(team.id, all_relations, data)
    roster_ids = {
        relation.player_id
        for relation in all_relations
        if relation.team_id == team.id
        and _season_year(relation.season) == context_year
    }
    if exclude_player_id is not None:
        roster_ids.discard(exclude_player_id)
    teammate_ratings = [
        stat.rating
        for stat in all_player_stats
        if stat.player_id in roster_ids
        and _season_year(stat.season) == context_year
    ]
    avg_tm = float(np.mean(teammate_ratings)) if teammate_ratings else 6.0

    if data is not None:
        tier_factor = _league_tier_factor(
            {"league_id": team.league_id, "league_tier": team.league_tier},
            data,
        )
    else:
        tier_factor = {1: 1.00, 2: 0.88, 3: 0.76, 4: 0.64, 5: 0.52}.get(
            team.league_tier, 0.70
        )

    pace_cat = "fast" if team.pace > 100 else ("slow" if team.pace < 93 else "medium")
    vector = compute_team_style_vector(team)
    return {
        "style_pace": float(vector[0]),
        "style_3pt_rate": float(vector[1]),
        "style_ast": float(vector[2]),
        "style_star_usage": float(vector[3]),
        "style_ortg": float(vector[4]),
        "style_drtg": float(vector[5]),
        "avg_teammate_rating": round(avg_tm, 3),
        "teammate_quality_pct": float(np.clip(avg_tm / 10, 0, 1)),
        "league_tier_factor": tier_factor,
        "league_tier": float(team.league_tier),
        "team_pace": team.pace,
        "team_pace_cat_fast": float(pace_cat == "fast"),
        "team_pace_cat_slow": float(pace_cat == "slow"),
    }


def _context_year_for_team(team_id: int, relations: pd.DataFrame, data: Dict[str, Any]) -> Optional[int]:
    if relations.empty or "season" not in relations.columns:
        return None
    rows = relations[relations["team_id"].map(_normalize_id) == _normalize_id(team_id)].copy()
    if rows.empty:
        return None
    rows["_season_year"] = rows["season"].map(_season_year)
    rows = rows[rows["_season_year"].notna()]
    as_of = data.get("_as_of_season")
    if as_of is not None:
        rows = rows[rows["_season_year"] <= int(as_of)]
    return int(rows["_season_year"].max()) if not rows.empty else None


def compute_team_features(
    team_id: int,
    data: Dict[str, Any],
    exclude_player_id: Optional[int] = None,
) -> Dict[str, float]:
    """Compute team features without reading a fixed or future roster season."""
    team_row = data["team_dict"].get(_normalize_id(team_id))
    if team_row is None:
        return _default_team_features()
    try:
        team = Team(**{key: team_row.get(key) for key in Team.__dataclass_fields__})
    except Exception:
        return _default_team_features()

    stats_df = data.get("player_stats", pd.DataFrame())
    relations = data.get("team_player_relations", pd.DataFrame())
    context_year = _context_year_for_team(team_id, relations, data)
    roster_ids: set[Any] = set()
    if context_year is not None:
        rel = relations.copy()
        rel["_season_year"] = rel["season"].map(_season_year)
        roster_ids = set(
            rel[
                (rel["team_id"].map(_normalize_id) == _normalize_id(team_id))
                & (rel["_season_year"] == context_year)
            ]["player_id"].tolist()
        )
    if exclude_player_id is not None:
        roster_ids.discard(exclude_player_id)

    avg_tm = 6.0
    if roster_ids and not stats_df.empty and "season" in stats_df.columns:
        stats = stats_df.copy()
        stats["_season_year"] = stats["season"].map(_season_year)
        tm_stats = stats[
            stats["player_id"].isin(roster_ids)
            & (stats["_season_year"] == context_year)
        ]
        ratings = pd.to_numeric(tm_stats.get("rating"), errors="coerce").dropna()
        if not ratings.empty:
            avg_tm = float(ratings.mean())

    tier_factor = _league_tier_factor(team_row, data)
    tier = int(team_row.get("league_tier", 3) or 3)
    pace = float(team_row.get("pace", 95) or 95)
    pace_cat = "fast" if pace > 100 else ("slow" if pace < 93 else "medium")
    vector = compute_team_style_vector(team)
    return {
        "style_pace": float(vector[0]),
        "style_3pt_rate": float(vector[1]),
        "style_ast": float(vector[2]),
        "style_star_usage": float(vector[3]),
        "style_ortg": float(vector[4]),
        "style_drtg": float(vector[5]),
        "avg_teammate_rating": round(avg_tm, 3),
        "teammate_quality_pct": float(np.clip(avg_tm / 10, 0, 1)),
        "league_tier_factor": tier_factor,
        "league_tier": float(tier),
        "team_pace": pace,
        "team_pace_cat_fast": float(pace_cat == "fast"),
        "team_pace_cat_slow": float(pace_cat == "slow"),
    }


def _default_team_features() -> Dict[str, float]:
    return {
        "style_pace": 0.5,
        "style_3pt_rate": 0.5,
        "style_ast": 0.5,
        "style_star_usage": 0.5,
        "style_ortg": 0.5,
        "style_drtg": 0.5,
        "avg_teammate_rating": 6.0,
        "teammate_quality_pct": 0.6,
        "league_tier_factor": 1.0,
        "league_tier": 1.0,
        "team_pace": 97.0,
        "team_pace_cat_fast": 0.0,
        "team_pace_cat_slow": 0.0,
    }

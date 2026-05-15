"""Basketball player feature engineering.

Provides two layers:
  1. Core functions that work with typed dataclass objects (Player, PlayerStats).
  2. Dict-API wrappers used by the ensemble and API routes, which operate on
     the flat `data` dict returned by loader.load_all_data().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from basketball_ai.data.models import Player, PlayerStats
from basketball_ai.utils.helpers import normalize_id as _normalize_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITIONAL_PEAK_AGES: Dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}


def _primary_pos(pos: str) -> str:
    """Return first component of a (potentially hybrid) position string."""
    return pos.split("/")[0]


def _peak_age(pos: str) -> int:
    return POSITIONAL_PEAK_AGES.get(pos, POSITIONAL_PEAK_AGES.get(_primary_pos(pos), 26))


def _per36(stat: float, mpg: float) -> float:
    if mpg <= 0:
        return 0.0
    return float(stat / mpg * 36)


# ---------------------------------------------------------------------------
# Core feature computation (dataclass API)
# ---------------------------------------------------------------------------

def compute_player_features_from_objects(
    player: Player,
    stats_history: List[PlayerStats],
) -> Dict[str, Any]:
    """Compute basketball feature vector from typed dataclass objects.

    Includes all legacy features plus new DB-schema derived features:
    SPM, RAPTOR, LEBRON, on/off differentials, clutch performance,
    per-40 stats, hustle metrics, and scoring efficiency.
    """
    if not stats_history:
        return _empty_features(player)

    stats_history = sorted(stats_history, key=lambda s: s.season)
    ratings = [s.rating for s in stats_history]

    # form score: weighted avg of last 3 seasons (most recent = highest weight)
    last3 = stats_history[-3:]
    weights = ([0.2, 0.3, 0.5] if len(last3) == 3
               else ([0.4, 0.6] if len(last3) == 2 else [1.0]))
    form_score = sum(s.rating * w for s, w in zip(last3, weights))

    mean_r = float(np.mean(ratings))
    std_r  = float(np.std(ratings))
    consistency_score = float(np.clip(1.0 - (std_r / mean_r) if mean_r > 0 else 0.0, 0.0, 1.0))

    latest = stats_history[-1]
    mpg = latest.minutes_per_game if latest.minutes_per_game > 0 else 1.0
    pts_per_36 = _per36(latest.points, mpg)
    ast_per_36 = _per36(latest.assists, mpg)
    reb_per_36 = _per36(latest.rebounds, mpg)
    stl_per_36 = _per36(latest.steals, mpg)
    blk_per_36 = _per36(latest.blocks, mpg)

    avg_per     = float(np.mean([s.per     for s in stats_history]))
    avg_ts_pct  = float(np.mean([s.ts_pct  for s in stats_history]))
    avg_usg_pct = float(np.mean([s.usg_pct for s in stats_history]))
    avg_bpm     = float(np.mean([s.bpm     for s in stats_history]))
    peak_rating = float(max(ratings))

    career_trajectory = (
        float(np.polyfit(np.arange(len(ratings), dtype=float), ratings, 1)[0])
        if len(ratings) >= 2 else 0.0
    )

    peak_age = _peak_age(player.position)
    age_vs_peak_age = player.age - peak_age

    # --- New DB-schema advanced metrics ------------------------------------
    avg_spm          = float(np.mean([s.spm          for s in stats_history]))
    avg_raptor_total = float(np.mean([s.raptor_total for s in stats_history]))
    avg_raptor_off   = float(np.mean([s.raptor_off   for s in stats_history]))
    avg_raptor_def   = float(np.mean([s.raptor_def   for s in stats_history]))
    avg_lebron_total = float(np.mean([s.lebron_total for s in stats_history]))
    avg_obpm         = float(np.mean([s.obpm         for s in stats_history]))
    avg_dbpm         = float(np.mean([s.dbpm         for s in stats_history]))
    avg_gm_sc        = float(np.mean([s.gm_sc        for s in stats_history]))
    avg_fic          = float(np.mean([s.fic          for s in stats_history]))

    # Efficiency / hustle
    avg_scoring_efficiency = float(np.mean([s.scoring_efficiency for s in stats_history]))
    avg_hustle_index       = float(np.mean([s.hustle_index       for s in stats_history]))
    avg_foul_drawing_rate  = float(np.mean([s.foul_drawing_rate  for s in stats_history]))

    # On/Off differentials
    avg_net_rtg_diff = float(np.mean([s.net_rtg_diff for s in stats_history]))
    avg_ortg_diff    = float(np.mean([s.ortg_diff    for s in stats_history]))

    # Clutch performance (latest season for currency; fall back to career avg)
    clutch_games_career = sum(s.clutch_games for s in stats_history)
    if clutch_games_career > 0:
        avg_clutch_ts_pct   = float(np.mean([s.clutch_ts_pct   for s in stats_history if s.clutch_games > 0]))
        avg_clutch_net_rtg  = float(np.mean([s.clutch_net_rtg  for s in stats_history if s.clutch_games > 0]))
        avg_clutch_efg_pct  = float(np.mean([s.clutch_efg_pct  for s in stats_history if s.clutch_games > 0]))
        clutch_pts_per_36   = _per36(latest.clutch_pts, mpg)
    else:
        avg_clutch_ts_pct  = avg_ts_pct
        avg_clutch_net_rtg = 0.0
        avg_clutch_efg_pct = float(latest.fg_pct)
        clutch_pts_per_36  = 0.0

    # Per-40 stats (use DB values if available, compute from per-game otherwise)
    avg_pts_per_40 = float(np.mean([s.pts_per_40 for s in stats_history])) if latest.pts_per_40 > 0 else _per36(latest.points, mpg) / 36 * 40
    avg_ast_per_40 = float(np.mean([s.ast_per_40 for s in stats_history])) if latest.ast_per_40 > 0 else _per36(latest.assists, mpg) / 36 * 40

    # --- Starter status (from Boxscore.SF) ------------------------------------
    # starter_pct is 0–1: fraction of games played as a starter.
    # It is a strong role signal: starters typically operate under more playing
    # time, higher usage, and direct comparison against opposing starters.
    avg_starter_pct = float(np.mean([s.starter_pct for s in stats_history]))

    # --- Scoring profile (now using PPSA and scoring_efficiency too) -------
    if latest.three_point_pct > 0.37 and latest.usg_pct < 22:
        scoring_profile = "3pt_specialist"
    elif latest.fg_pct > 0.55:
        scoring_profile = "paint_scorer"
    elif avg_usg_pct > 25:
        scoring_profile = "volume_scorer"
    else:
        scoring_profile = "efficient_scorer"

    playmaking_score  = ast_per_36 / avg_usg_pct if avg_usg_pct > 0 else 0.0
    defensive_score   = float(np.clip(stl_per_36 * 1.5 + blk_per_36 * 1.2, 0, 10))
    norms = [v / mx for v, mx in [(pts_per_36, 40), (ast_per_36, 15), (reb_per_36, 20), (stl_per_36, 4), (blk_per_36, 5)]]
    versatility_score = float(np.clip(1.0 - float(np.std(norms)), 0, 1))

    return {
        # Legacy features
        "form_score":              round(form_score, 4),
        "consistency_score":       round(consistency_score, 4),
        "pts_per_36":              round(pts_per_36, 2),
        "ast_per_36":              round(ast_per_36, 2),
        "reb_per_36":              round(reb_per_36, 2),
        "stl_per_36":              round(stl_per_36, 2),
        "blk_per_36":              round(blk_per_36, 2),
        "avg_per":                 round(avg_per, 2),
        "avg_ts_pct":              round(avg_ts_pct, 4),
        "avg_usg_pct":             round(avg_usg_pct, 2),
        "avg_bpm":                 round(avg_bpm, 2),
        "peak_rating":             round(peak_rating, 4),
        "career_trajectory":       round(career_trajectory, 4),
        "age_vs_peak_age":         age_vs_peak_age,
        "positional_peak_age":     peak_age,
        "scoring_profile":         scoring_profile,
        "playmaking_score":        round(playmaking_score, 4),
        "defensive_score":         round(defensive_score, 4),
        "versatility_score":       round(versatility_score, 4),
        # New advanced rating models
        "avg_spm":                 round(avg_spm, 3),
        "avg_raptor_total":        round(avg_raptor_total, 3),
        "avg_raptor_off":          round(avg_raptor_off, 3),
        "avg_raptor_def":          round(avg_raptor_def, 3),
        "avg_lebron_total":        round(avg_lebron_total, 3),
        "avg_obpm":                round(avg_obpm, 3),
        "avg_dbpm":                round(avg_dbpm, 3),
        "avg_gm_sc":               round(avg_gm_sc, 3),
        "avg_fic":                 round(avg_fic, 3),
        # Efficiency / hustle
        "avg_scoring_efficiency":  round(avg_scoring_efficiency, 3),
        "avg_hustle_index":        round(avg_hustle_index, 3),
        "avg_foul_drawing_rate":   round(avg_foul_drawing_rate, 3),
        # On/Off differentials
        "avg_net_rtg_diff":        round(avg_net_rtg_diff, 3),
        "avg_ortg_diff":           round(avg_ortg_diff, 3),
        # Clutch
        "clutch_pts_per_36":       round(clutch_pts_per_36, 2),
        "avg_clutch_ts_pct":       round(avg_clutch_ts_pct, 4),
        "avg_clutch_net_rtg":      round(avg_clutch_net_rtg, 3),
        "avg_clutch_efg_pct":      round(avg_clutch_efg_pct, 4),
        "clutch_games_career":     clutch_games_career,
        # Per-40
        "avg_pts_per_40":          round(avg_pts_per_40, 2),
        "avg_ast_per_40":          round(avg_ast_per_40, 2),
        # Starter status
        "avg_starter_pct":         round(avg_starter_pct, 4),
    }


def _empty_features(player: Player) -> Dict[str, Any]:
    peak_age = _peak_age(player.position)
    return {
        "form_score": 5.0, "consistency_score": 0.5,
        "pts_per_36": 10.0, "ast_per_36": 2.0, "reb_per_36": 4.0,
        "stl_per_36": 1.0, "blk_per_36": 0.5,
        "avg_per": 12.0, "avg_ts_pct": 0.52, "avg_usg_pct": 18.0, "avg_bpm": -1.0,
        "peak_rating": 5.0, "career_trajectory": 0.0,
        "age_vs_peak_age": player.age - peak_age, "positional_peak_age": peak_age,
        "scoring_profile": "efficient_scorer", "playmaking_score": 0.1,
        "defensive_score": 1.0, "versatility_score": 0.5,
        # New advanced metrics – neutral defaults
        "avg_spm": 0.0, "avg_raptor_total": 0.0, "avg_raptor_off": 0.0,
        "avg_raptor_def": 0.0, "avg_lebron_total": 0.0,
        "avg_obpm": 0.0, "avg_dbpm": 0.0, "avg_gm_sc": 0.0, "avg_fic": 0.0,
        "avg_scoring_efficiency": 0.0, "avg_hustle_index": 0.0,
        "avg_foul_drawing_rate": 0.0,
        "avg_net_rtg_diff": 0.0, "avg_ortg_diff": 0.0,
        "clutch_pts_per_36": 0.0, "avg_clutch_ts_pct": 0.52,
        "avg_clutch_net_rtg": 0.0, "avg_clutch_efg_pct": 0.5,
        "clutch_games_career": 0,
        "avg_pts_per_40": 0.0, "avg_ast_per_40": 0.0,
        # Starter status
        "avg_starter_pct": 0.5,
    }


# ---------------------------------------------------------------------------
# Dict-API wrappers (used by ensemble, routes)
# ---------------------------------------------------------------------------

def compute_player_features(
    player_id: int,
    data: Dict[str, Any],
    season: Optional[int] = None,
    target_age: Optional[int] = None,
) -> Dict[str, Any]:
    """Dict-based wrapper for use by the ensemble and API layer.

    Looks up player and their stats from the flat `data` dict, then
    delegates to compute_player_features_from_objects().
    """
    player_row = data["player_dict"].get(_normalize_id(player_id), {})
    stats_df   = data["player_stats"]

    # Build a minimal Player object
    pos = str(player_row.get("position", "PG"))
    age = int(player_row.get("age", 26)) if target_age is None else int(target_age)

    player = Player(
        id=_normalize_id(player_id),
        name=str(player_row.get("name", "")),
        age=age,
        position=pos,
        nationality=str(player_row.get("nationality", "")),
        height_cm=int(player_row.get("height_cm", 195) or 195),
        weight_kg=int(player_row.get("weight_kg", 95) or 95),
        dominant_hand=str(player_row.get("dominant_hand", "right")),
        current_team_id=player_row.get("current_team_id"),
        current_league_id=player_row.get("current_league_id"),
        draft_year=player_row.get("draft_year"),
        draft_pick=player_row.get("draft_pick"),
    )

    # Filter stats for this player
    mask = stats_df["player_id"] == _normalize_id(player_id)
    if season is not None:
        # include only seasons up to the target season
        mask = mask & (stats_df["season"] <= str(season))
    p_stats_df = stats_df[mask].sort_values("season")

    stat_objects: List[PlayerStats] = []
    for _, row in p_stats_df.iterrows():
        try:
            stat_objects.append(PlayerStats(**row.to_dict()))
        except Exception:
            pass

    return compute_player_features_from_objects(player, stat_objects)


def compute_form_score(stats_df: pd.DataFrame) -> float:
    """Compute weighted form score from a player's stats DataFrame."""
    if stats_df.empty:
        return 5.0
    ratings = stats_df.sort_values("season")["rating"].tolist()
    last3 = ratings[-3:]
    weights = ([0.2, 0.3, 0.5] if len(last3) == 3
               else ([0.4, 0.6] if len(last3) == 2 else [1.0]))
    return float(sum(r * w for r, w in zip(last3, weights)))


def compute_consistency_score(stats_df: pd.DataFrame) -> float:
    """Compute consistency from rating standard deviation."""
    if stats_df.empty or len(stats_df) < 2:
        return 0.5
    ratings = stats_df["rating"].values
    mean_r = float(np.mean(ratings))
    std_r  = float(np.std(ratings))
    return float(np.clip(1.0 - (std_r / mean_r) if mean_r > 0 else 0.0, 0.0, 1.0))


def compute_career_trajectory(stats_df: pd.DataFrame) -> float:
    """Compute linear slope of rating over career seasons."""
    if stats_df.empty or len(stats_df) < 2:
        return 0.0
    ratings = stats_df.sort_values("season")["rating"].values
    xs = np.arange(len(ratings), dtype=float)
    return float(np.polyfit(xs, ratings, 1)[0])

"""Basketball player feature engineering."""
from typing import List, Dict, Any
import numpy as np
from src.data.models import Player, PlayerStats

POSITIONAL_PEAK_AGES = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}

def _primary_pos(pos: str) -> str:
    return pos.split("/")[0]

def _per36(stat: float, mpg: float) -> float:
    if mpg <= 0:
        return 0.0
    return float(stat / mpg * 36)

def compute_player_features(player: Player, stats_history: List[PlayerStats]) -> Dict[str, Any]:
    """Compute basketball feature vector for a player given their stats history."""
    if not stats_history:
        return _empty_features(player)

    stats_history = sorted(stats_history, key=lambda s: s.season)
    ratings = [s.rating for s in stats_history]
    last3 = stats_history[-3:]
    weights = [0.5, 0.3, 0.2][:len(last3)][::-1]
    total_w = sum(weights)
    form_score = sum(s.rating * w for s, w in zip(last3, weights)) / total_w if total_w else ratings[-1]

    mean_r = float(np.mean(ratings))
    std_r = float(np.std(ratings))
    consistency_score = 1.0 - (std_r / mean_r) if mean_r > 0 else 0.0
    consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

    latest = stats_history[-1]
    mpg = latest.minutes_per_game if latest.minutes_per_game > 0 else 1.0
    pts_per_36 = _per36(latest.points, mpg)
    ast_per_36 = _per36(latest.assists, mpg)
    reb_per_36 = _per36(latest.rebounds, mpg)
    stl_per_36 = _per36(latest.steals, mpg)
    blk_per_36 = _per36(latest.blocks, mpg)

    avg_per = float(np.mean([s.per for s in stats_history]))
    avg_ts_pct = float(np.mean([s.ts_pct for s in stats_history]))
    avg_usg_pct = float(np.mean([s.usg_pct for s in stats_history]))
    avg_bpm = float(np.mean([s.bpm for s in stats_history]))
    peak_rating = float(max(ratings))

    if len(ratings) >= 2:
        xs = np.arange(len(ratings), dtype=float)
        career_trajectory = float(np.polyfit(xs, ratings, 1)[0])
    else:
        career_trajectory = 0.0

    peak_age = POSITIONAL_PEAK_AGES.get(player.position, POSITIONAL_PEAK_AGES.get(_primary_pos(player.position), 26))
    age_vs_peak_age = player.age - peak_age

    # scoring profile
    if latest.three_point_pct > 0.37 and latest.usg_pct < 22:
        scoring_profile = "3pt_specialist"
    elif latest.fg_pct > 0.55:
        scoring_profile = "paint_scorer"
    elif avg_usg_pct > 25:
        scoring_profile = "volume_scorer"
    else:
        scoring_profile = "efficient_scorer"

    playmaking_score = ast_per_36 / avg_usg_pct if avg_usg_pct > 0 else 0.0

    defensive_score = float(np.clip(
        stl_per_36 * 1.5 + blk_per_36 * 1.2, 0, 10
    ))

    norms = []
    for v, mx in [(pts_per_36, 40), (ast_per_36, 15), (reb_per_36, 20), (stl_per_36, 4), (blk_per_36, 5)]:
        norms.append(v / mx if mx > 0 else 0)
    versatility_score = float(1.0 - np.std(norms))
    versatility_score = float(np.clip(versatility_score, 0, 1))

    return {
        "form_score": round(form_score, 4),
        "consistency_score": round(consistency_score, 4),
        "pts_per_36": round(pts_per_36, 2),
        "ast_per_36": round(ast_per_36, 2),
        "reb_per_36": round(reb_per_36, 2),
        "stl_per_36": round(stl_per_36, 2),
        "blk_per_36": round(blk_per_36, 2),
        "avg_per": round(avg_per, 2),
        "avg_ts_pct": round(avg_ts_pct, 4),
        "avg_usg_pct": round(avg_usg_pct, 2),
        "avg_bpm": round(avg_bpm, 2),
        "peak_rating": round(peak_rating, 4),
        "career_trajectory": round(career_trajectory, 4),
        "age_vs_peak_age": age_vs_peak_age,
        "positional_peak_age": peak_age,
        "scoring_profile": scoring_profile,
        "playmaking_score": round(playmaking_score, 4),
        "defensive_score": round(defensive_score, 4),
        "versatility_score": round(versatility_score, 4),
    }

def _empty_features(player: Player) -> Dict[str, Any]:
    peak_age = POSITIONAL_PEAK_AGES.get(player.position, POSITIONAL_PEAK_AGES.get(_primary_pos(player.position), 26))
    return {
        "form_score": 5.0, "consistency_score": 0.5,
        "pts_per_36": 10.0, "ast_per_36": 2.0, "reb_per_36": 4.0,
        "stl_per_36": 1.0, "blk_per_36": 0.5,
        "avg_per": 12.0, "avg_ts_pct": 0.52, "avg_usg_pct": 18.0, "avg_bpm": -1.0,
        "peak_rating": 5.0, "career_trajectory": 0.0,
        "age_vs_peak_age": player.age - peak_age, "positional_peak_age": peak_age,
        "scoring_profile": "efficient_scorer", "playmaking_score": 0.1,
        "defensive_score": 1.0, "versatility_score": 0.5,
    }

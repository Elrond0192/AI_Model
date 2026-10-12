"""Extensible, evidence-based basketball player archetype catalog.

Scores are relative to the requested league/season/competition cohort. These are
descriptive statistical profiles, not mutually exclusive labels or probabilities.
Add archetypes here without changing the scenario engine.
"""
from __future__ import annotations

# Positive weights reward higher percentiles; negative weights reward lower
# values (e.g. turnovers). Missing metrics are omitted and remaining weights
# are renormalized. Definitions intentionally overlap: real players are multi-role.
ARCHETYPE_METRICS: dict[str, dict[str, float]] = {
    # Creation and decision-making
    "playmaker": {"assists": .22, "ast_pct": .24, "tov_pct": -.18, "bpm": .12, "ts_pct": .10, "usg_pct": .06},
    "primary_creator": {"assists": .18, "ast_pct": .22, "usg_pct": .18, "tov_pct": -.12, "bpm": .16, "points": .14},
    "combo_guard": {"assists": .16, "ast_pct": .16, "points": .18, "usg_pct": .16, "three_point_pct": .12, "ts_pct": .12, "bpm": .10},
    "secondary_creator": {"assists": .20, "ast_pct": .20, "tov_pct": -.18, "usg_pct": .08, "ts_pct": .16, "bpm": .18},
    "facilitator": {"assists": .24, "ast_pct": .26, "tov_pct": -.20, "bpm": .16, "usg_pct": -.06, "rating": .08},
    "connector": {"ast_pct": .22, "assists": .18, "tov_pct": -.18, "ts_pct": .16, "bpm": .12, "usg_pct": -.06, "rebounds": .08},
    "low_turnover_creator": {"assists": .24, "ast_pct": .24, "tov_pct": -.30, "bpm": .12, "usg_pct": .10},
    "playmaking_big": {"assists": .18, "ast_pct": .22, "rebounds": .16, "blocks": .10, "bpm": .16, "tov_pct": -.08, "ts_pct": .10},
    # Scoring and rim pressure
    "scorer": {"points": .22, "usg_pct": .18, "ts_pct": .20, "three_point_pct": .12, "bpm": .12, "tov_pct": -.06},
    "volume_scorer": {"points": .28, "usg_pct": .26, "fga": .14, "fta": .10, "bpm": .12, "ts_pct": .10},
    "efficient_scorer": {"ts_pct": .30, "efg_pct": .22, "points": .16, "two_point_pct": .12, "ft_pct": .08, "bpm": .12},
    "high_usage_engine": {"usg_pct": .30, "points": .20, "assists": .14, "ast_pct": .12, "bpm": .14, "tov_pct": -.10},
    "slasher": {"two_point_pct": .22, "fta": .18, "ft_rate": .18, "points": .16, "ts_pct": .14, "usg_pct": .12},
    "rim_finisher": {"two_point_pct": .26, "ts_pct": .22, "rebounds": .12, "blocks": .08, "points": .18, "fta": .14},
    "post_scorer": {"two_point_pct": .22, "points": .20, "usg_pct": .18, "fta": .14, "rebounds": .12, "ts_pct": .14},
    "transition_scorer": {"points": .24, "two_point_pct": .18, "ts_pct": .18, "steals": .10, "bpm": .12, "minutes_per_game": .08, "usg_pct": .10},
    # Shooting and off-ball play
    "shooter": {"three_point_pct": .30, "three_point_attempts": .20, "three_point_rate": .18, "ts_pct": .14, "ft_pct": .08, "points": .10},
    "catch_and_shoot": {"three_point_pct": .32, "three_point_attempts": .24, "three_point_rate": .20, "ts_pct": .14, "tov_pct": -.10},
    "movement_shooter": {"three_point_pct": .30, "three_point_attempts": .24, "three_point_rate": .22, "ft_pct": .08, "ts_pct": .16},
    "pull_up_shooter": {"three_point_attempts": .22, "three_point_pct": .22, "usg_pct": .18, "points": .14, "ts_pct": .14, "bpm": .10},
    "stretch_big": {"three_point_pct": .24, "three_point_attempts": .20, "three_point_rate": .18, "ts_pct": .16, "rebounds": .10, "blocks": .06, "bpm": .06},
    # Defense
    "defender": {"dbpm": .28, "steals": .14, "blocks": .14, "rebounds": .10, "bpm": .14, "tov_pct": -.04},
    "perimeter_defender": {"dbpm": .26, "steals": .22, "bpm": .16, "deflections": .14, "fouls": -.10, "minutes_per_game": .12},
    "point_of_attack_defender": {"steals": .22, "dbpm": .24, "deflections": .18, "fouls": -.12, "bpm": .12, "minutes_per_game": .12},
    "versatile_defender": {"dbpm": .24, "steals": .14, "blocks": .14, "rebounds": .14, "bpm": .20, "minutes_per_game": .14},
    "rim_protector": {"blocks": .32, "dbpm": .24, "rebounds": .18, "bpm": .16, "fouls": -.10},
    "steals_specialist": {"steals": .34, "dbpm": .20, "deflections": .18, "bpm": .14, "minutes_per_game": .14},
    "team_defender": {"dbpm": .26, "bpm": .22, "rebounds": .14, "steals": .12, "blocks": .10, "fouls": -.08, "rating": .08},
    # Rebounding, screens and interior roles
    "rebounder": {"rebounds": .30, "oreb": .18, "dreb": .22, "blocks": .10, "bpm": .12, "minutes_per_game": .08},
    "offensive_rebounder": {"oreb": .38, "rebounds": .22, "two_point_pct": .12, "fta": .10, "bpm": .10, "minutes_per_game": .08},
    "defensive_rebounder": {"dreb": .38, "rebounds": .24, "blocks": .12, "dbpm": .12, "bpm": .08, "minutes_per_game": .06},
    "roll_man": {"two_point_pct": .24, "ts_pct": .22, "rebounds": .16, "blocks": .10, "fta": .10, "bpm": .10, "turnovers": -.08},
    "post_big": {"rebounds": .20, "two_point_pct": .18, "points": .16, "blocks": .12, "dbpm": .12, "ts_pct": .14, "bpm": .08},
    "big": {"rebounds": .22, "blocks": .18, "dbpm": .18, "ts_pct": .16, "bpm": .12, "points": .08},
    "passing_center": {"assists": .20, "ast_pct": .22, "rebounds": .18, "blocks": .10, "tov_pct": -.12, "bpm": .18},
    # Role and broad utility profiles
    "three_and_d": {"three_point_pct": .22, "three_point_attempts": .14, "three_point_rate": .12, "dbpm": .22, "steals": .10, "blocks": .08, "bpm": .12},
    "two_way": {"bpm": .22, "dbpm": .20, "rating": .16, "ts_pct": .14, "steals": .08, "blocks": .08, "rebounds": .06, "ast_pct": .06},
    "high_impact_low_usage": {"bpm": .24, "dbpm": .18, "ts_pct": .18, "tov_pct": -.14, "usg_pct": -.14, "rating": .12},
    "balanced": {"rating": .24, "bpm": .18, "ts_pct": .16, "ast_pct": .12, "dbpm": .12, "points": .10, "rebounds": .08},
}

# Legacy public names are retained for API compatibility.
ARCHETYPE_ALIASES = {
    "big": "post_big",
}

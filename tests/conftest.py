"""Shared pytest fixtures available to all tests in this package."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from basketball_ai.models.ensemble import EnsembleModel


POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]


@pytest.fixture(scope="session")
def tiny_data():
    leagues = pd.DataFrame([{
        "id": 1, "name": "NBA", "country": "USA", "tier": 1,
        "competitiveness_score": 1.0, "avg_pace": 100.0, "avg_offensive_rating": 113.0,
    }, {
        "id": 2, "name": "EuroLeague", "country": "Europe", "tier": 2,
        "competitiveness_score": 0.85, "avg_pace": 93.0, "avg_offensive_rating": 108.0,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i+1}", "league_id": (i % 2) + 1,
        "playing_style": ["pace_and_space","pick_and_roll","isolation","defensive","motion_offense","post_up"][i % 6],
        "formation": "small_ball",
        "pace": 95.0 + i * 2,
        "offensive_rating": 108.0 + i,
        "defensive_rating": 107.0 + i,
        "three_point_attempt_rate": 0.35 + i * 0.01,
        "assists_per_game": 24.0 + i,
        "star_player_usage": 0.28,
        "league_tier": (i % 2) + 1,
    } for i in range(10)])

    np.random.seed(42)
    players = pd.DataFrame([{
        "id": i + 1, "name": f"Player {i+1}",
        "age": 21 + (i % 15),
        "position": POSITIONS[i % len(POSITIONS)],
        "nationality": "American",
        "height_cm": 190 + i % 20,
        "weight_kg": 90 + i % 30,
        "dominant_hand": "right",
        "current_team_id": (i % 10) + 1,
        "current_league_id": (i % 2) + 1,
        "draft_year": None, "draft_pick": None,
    } for i in range(60)])

    stat_rows = []
    _ROLES_COMBO = ["playmaker", "scorer", "forward", "big", "wing"]
    _ROLES_OFF   = ["scorer", "facilitator", "spot_up", "post", "cutter"]
    _ROLES_DEF   = ["lockdown", "stopper", "help_side", "rim_protector", "versatile"]
    for _, p in players.iterrows():
        for season in ["2021-22", "2022-23", "2023-24"]:
            mpg = np.random.uniform(15, 35)
            stat_rows.append({
                "player_id": int(p["id"]), "season": season,
                "team_id": int(p["current_team_id"]), "league_id": int(p["current_league_id"]),
                "games_played": int(np.random.randint(30, 80)),
                "minutes_per_game": round(mpg, 1),
                "points": round(np.random.uniform(6, 25), 1),
                "rebounds": round(np.random.uniform(2, 12), 1),
                "offensive_rebounds": round(np.random.uniform(0.5, 3), 1),
                "defensive_rebounds": round(np.random.uniform(2, 9), 1),
                "assists": round(np.random.uniform(1, 8), 1),
                "steals": round(np.random.uniform(0.3, 2.0), 2),
                "blocks": round(np.random.uniform(0.1, 2.5), 2),
                "turnovers": round(np.random.uniform(0.5, 4.0), 1),
                "personal_fouls": round(np.random.uniform(1, 4), 1),
                "fg_pct": round(np.random.uniform(0.38, 0.58), 3),
                "three_point_pct": round(np.random.uniform(0.28, 0.45), 3),
                "ft_pct": round(np.random.uniform(0.65, 0.90), 3),
                "plus_minus": round(np.random.uniform(-8, 8), 1),
                "per": round(np.random.uniform(10, 25), 2),
                "ts_pct": round(np.random.uniform(0.50, 0.65), 3),
                "usg_pct": round(np.random.uniform(14, 30), 2),
                "bpm": round(np.random.uniform(-3, 6), 2),
                "obpm": round(np.random.uniform(-2, 4), 2),
                "dbpm": round(np.random.uniform(-2, 3), 2),
                "vorp": round(np.random.uniform(-0.5, 4), 2),
                "win_shares": round(np.random.uniform(0, 12), 2),
                "ast_ratio": round(np.random.uniform(5, 30), 2),
                "reb_pct": round(np.random.uniform(3, 20), 2),
                "tov_pct": round(np.random.uniform(8, 20), 1),
                "ast_pct": round(np.random.uniform(5, 30), 1),
                "orb_pct": round(np.random.uniform(1, 8), 1),
                "drb_pct": round(np.random.uniform(5, 25), 1),
                "ruolo_combinato": _ROLES_COMBO[int(p["id"]) % len(_ROLES_COMBO)],
                "ruolo_offensivo": _ROLES_OFF[int(p["id"]) % len(_ROLES_OFF)],
                "ruolo_difensivo": _ROLES_DEF[int(p["id"]) % len(_ROLES_DEF)],
                "rating": round(np.random.uniform(5.0, 8.5), 3),
            })
    stats = pd.DataFrame(stat_rows)

    rels = pd.DataFrame([{
        "team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
        "season": "2023-24",
        "role": "starter" if i % 3 == 0 else "rotation",
        "jersey_number": (i % 99) + 1,
    } for i, (_, p) in enumerate(players.iterrows())])

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict   = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}
    league_teams = {}
    for _, t in teams.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    return {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": league_teams,
    }


@pytest.fixture(scope="session")
def ensemble(tiny_data):
    e = EnsembleModel()
    e.train(tiny_data)
    return e

"""Data loader: reads CSV files and builds lookup structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_all_data(data_dir: str = "data/sample") -> Dict[str, Any]:
    """Load all CSV files and return a unified data dictionary.

    Returns a dict with keys:
        leagues, teams, players, player_stats, team_player_relations
        league_dict, team_dict, player_dict  (id → row dict)
        league_teams  (league_id → list of team_ids)
    """
    base = Path(data_dir)

    leagues = pd.read_csv(base / "leagues.csv")
    teams = pd.read_csv(base / "teams.csv")
    players = pd.read_csv(base / "players.csv")
    player_stats = pd.read_csv(base / "player_stats.csv")
    team_player_relations = pd.read_csv(base / "team_player_relations.csv")

    # Build fast lookup dicts
    league_dict: Dict[int, Dict] = {
        int(row["id"]): row.to_dict() for _, row in leagues.iterrows()
    }
    team_dict: Dict[int, Dict] = {
        int(row["id"]): row.to_dict() for _, row in teams.iterrows()
    }
    player_dict: Dict[int, Dict] = {
        int(row["id"]): row.to_dict() for _, row in players.iterrows()
    }

    # league_id → [team_id, ...]
    league_teams: Dict[int, List[int]] = {}
    for _, row in teams.iterrows():
        lid = int(row["league_id"])
        league_teams.setdefault(lid, []).append(int(row["id"]))

    return {
        "leagues": leagues,
        "teams": teams,
        "players": players,
        "player_stats": player_stats,
        "team_player_relations": team_player_relations,
        "league_dict": league_dict,
        "team_dict": team_dict,
        "player_dict": player_dict,
        "league_teams": league_teams,
    }


def data_exists(data_dir: str = "data/sample") -> bool:
    """Return True if all required CSV files are present."""
    base = Path(data_dir)
    required = [
        "leagues.csv", "teams.csv", "players.csv",
        "player_stats.csv", "team_player_relations.csv",
    ]
    return all((base / f).exists() for f in required)

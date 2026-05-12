"""Basketball data loader.

Provides both a DataStore class (OOP interface) and a load_all_data()
helper that returns the flat dict expected by the ensemble and API layer.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from basketball_ai.data.models import League, Team, Player, PlayerStats, TeamPlayerRelation


def _to_int(value) -> int:
    """Convert value to int, handling decimal numbers, floats, and hex strings (e.g. '0000009B')."""
    if isinstance(value, str):
        v = value.strip()
        if v.startswith(("0x", "0X")):
            return int(v, 16)
        try:
            return int(v, 10)
        except ValueError:
            return int(v, 16)  # fallback: try hex (e.g. '0000009B')
    if isinstance(value, float):
        return int(value)
    return int(value)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample")


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_leagues(data_dir: str = DATA_DIR) -> List[League]:
    df = pd.read_csv(f"{data_dir}/leagues.csv")
    return [League(**r) for r in df.to_dict("records")]


def load_teams(data_dir: str = DATA_DIR) -> List[Team]:
    df = pd.read_csv(f"{data_dir}/teams.csv")
    return [Team(**r) for r in df.to_dict("records")]


def load_players(data_dir: str = DATA_DIR) -> List[Player]:
    df = pd.read_csv(f"{data_dir}/players.csv")
    results = []
    for r in df.to_dict("records"):
        r["current_team_id"] = None if pd.isna(r.get("current_team_id")) else _to_int(r["current_team_id"])
        r["current_league_id"] = None if pd.isna(r.get("current_league_id")) else _to_int(r["current_league_id"])
        r["draft_year"] = None if pd.isna(r.get("draft_year")) else _to_int(r["draft_year"])
        r["draft_pick"] = None if pd.isna(r.get("draft_pick")) else _to_int(r["draft_pick"])
        results.append(Player(**r))
    return results


def load_player_stats(data_dir: str = DATA_DIR) -> List[PlayerStats]:
    df = pd.read_csv(f"{data_dir}/player_stats.csv")
    return [PlayerStats(**r) for r in df.to_dict("records")]


def load_team_player_relations(data_dir: str = DATA_DIR) -> List[TeamPlayerRelation]:
    df = pd.read_csv(f"{data_dir}/team_player_relations.csv")
    return [TeamPlayerRelation(**r) for r in df.to_dict("records")]


# ---------------------------------------------------------------------------
# DataStore (OOP)
# ---------------------------------------------------------------------------

class DataStore:
    """In-memory store for all basketball data."""

    def __init__(self) -> None:
        self.leagues: List[League] = []
        self.teams: List[Team] = []
        self.players: List[Player] = []
        self.player_stats: List[PlayerStats] = []
        self.relations: List[TeamPlayerRelation] = []
        self._league_map: Dict[int, League] = {}
        self._team_map: Dict[int, Team] = {}
        self._player_map: Dict[int, Player] = {}

    def load(self, data_dir: str = DATA_DIR) -> None:
        self.leagues = load_leagues(data_dir)
        self.teams = load_teams(data_dir)
        self.players = load_players(data_dir)
        self.player_stats = load_player_stats(data_dir)
        self.relations = load_team_player_relations(data_dir)
        self._league_map = {l.id: l for l in self.leagues}
        self._team_map = {t.id: t for t in self.teams}
        self._player_map = {p.id: p for p in self.players}

    def get_league(self, lid: int) -> Optional[League]:
        return self._league_map.get(lid)

    def get_team(self, tid: int) -> Optional[Team]:
        return self._team_map.get(tid)

    def get_player(self, pid: int) -> Optional[Player]:
        return self._player_map.get(pid)

    def get_player_stats(self, pid: int) -> List[PlayerStats]:
        return [s for s in self.player_stats if s.player_id == pid]

    def get_team_players(self, tid: int, season: str = "2023-24") -> List[Player]:
        rels = [r for r in self.relations if r.team_id == tid and r.season == season]
        return [self._player_map[r.player_id] for r in rels if r.player_id in self._player_map]


# ---------------------------------------------------------------------------
# Flat dict API (used by ensemble, scenarios, API routes)
# ---------------------------------------------------------------------------

def data_exists(data_dir: str) -> bool:
    """Return True if all required CSV files exist."""
    required = ["leagues.csv", "teams.csv", "players.csv", "player_stats.csv", "team_player_relations.csv"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


def load_all_data(data_dir: str) -> Dict[str, Any]:
    """Load all CSVs and return a flat dict used across the application.

    Keys:
        leagues, teams, players, player_stats, team_player_relations  — DataFrames
        league_dict  — {id: row_dict}
        team_dict    — {id: row_dict}
        player_dict  — {id: row_dict}
        league_teams — {league_id: [team_ids]}
    """
    leagues_df = pd.read_csv(f"{data_dir}/leagues.csv")
    teams_df   = pd.read_csv(f"{data_dir}/teams.csv")
    players_df = pd.read_csv(f"{data_dir}/players.csv")
    stats_df   = pd.read_csv(f"{data_dir}/player_stats.csv")
    rels_df    = pd.read_csv(f"{data_dir}/team_player_relations.csv")

    # Normalise nullable int columns
    for col in ["current_team_id", "current_league_id", "draft_year", "draft_pick"]:
        if col in players_df.columns:
            players_df[col] = players_df[col].where(players_df[col].notna(), other=None)

    league_dict: Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()}
    team_dict:   Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in teams_df.iterrows()}
    player_dict: Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in players_df.iterrows()}

    league_teams: Dict[int, List[int]] = {}
    for _, t in teams_df.iterrows():
        league_teams.setdefault(_to_int(t["league_id"]), []).append(_to_int(t["id"]))

    return {
        "leagues": leagues_df,
        "teams": teams_df,
        "players": players_df,
        "player_stats": stats_df,
        "team_player_relations": rels_df,
        "league_dict": league_dict,
        "team_dict": team_dict,
        "player_dict": player_dict,
        "league_teams": league_teams,
    }


# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: Optional[DataStore] = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
        _store.load()
    return _store

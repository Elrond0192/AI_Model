"""Basketball data loader."""
import pandas as pd
from typing import List, Optional, Dict
from src.data.models import League, Team, Player, PlayerStats, TeamPlayerRelation

DATA_DIR = "/home/runner/work/AI_Model/AI_Model/data/sample"

def load_leagues() -> List[League]:
    df = pd.read_csv(f"{DATA_DIR}/leagues.csv")
    return [League(**r) for r in df.to_dict("records")]

def load_teams() -> List[Team]:
    df = pd.read_csv(f"{DATA_DIR}/teams.csv")
    return [Team(**r) for r in df.to_dict("records")]

def load_players() -> List[Player]:
    df = pd.read_csv(f"{DATA_DIR}/players.csv")
    results = []
    for r in df.to_dict("records"):
        r["current_team_id"] = None if pd.isna(r.get("current_team_id")) else int(r["current_team_id"])
        r["current_league_id"] = None if pd.isna(r.get("current_league_id")) else int(r["current_league_id"])
        r["draft_year"] = None if pd.isna(r.get("draft_year")) else int(r["draft_year"])
        r["draft_pick"] = None if pd.isna(r.get("draft_pick")) else int(r["draft_pick"])
        results.append(Player(**r))
    return results

def load_player_stats() -> List[PlayerStats]:
    df = pd.read_csv(f"{DATA_DIR}/player_stats.csv")
    return [PlayerStats(**r) for r in df.to_dict("records")]

def load_team_player_relations() -> List[TeamPlayerRelation]:
    df = pd.read_csv(f"{DATA_DIR}/team_player_relations.csv")
    return [TeamPlayerRelation(**r) for r in df.to_dict("records")]

class DataStore:
    def __init__(self):
        self.leagues: List[League] = []
        self.teams: List[Team] = []
        self.players: List[Player] = []
        self.player_stats: List[PlayerStats] = []
        self.relations: List[TeamPlayerRelation] = []
        self._league_map: Dict[int, League] = {}
        self._team_map: Dict[int, Team] = {}
        self._player_map: Dict[int, Player] = {}

    def load(self):
        self.leagues = load_leagues()
        self.teams = load_teams()
        self.players = load_players()
        self.player_stats = load_player_stats()
        self.relations = load_team_player_relations()
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

_store: Optional[DataStore] = None

def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
        _store.load()
    return _store

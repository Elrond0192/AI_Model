"""Basketball data loader.

Provides both a DataStore class (OOP interface) and a load_all_data()
helper that returns the flat dict expected by the ensemble and API layer.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from basketball_ai.data.models import League, Team, Player, PlayerStats, TeamPlayerRelation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1 – Schema validation at loader boundary
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "leagues": ["id", "name"],
    "teams": ["id", "name", "league_id"],
    "players": ["id", "name", "position"],
    "player_stats": ["player_id", "season", "games_played", "rating"],
    "team_player_relations": ["player_id", "team_id"],
}


def validate_dataframes(data: dict) -> list:
    """Validate that required columns are present in each DataFrame.
    Returns list of warning strings (empty = all OK)."""
    warnings_list = []
    for table, cols in REQUIRED_COLUMNS.items():
        df = data.get(table)
        if df is None:
            warnings_list.append(f"[D1] Missing table: {table}")
            continue
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warnings_list.append(f"[D1] Table '{table}' missing columns: {missing}")
    return warnings_list


def _to_int(value) -> int:
    """Convert value to int. Returns a hash of the string for non-numeric codes like 'GRC1'."""
    import math
    if isinstance(value, str):
        v = value.strip()
        if v.startswith(("0x", "0X")):
            return int(v, 16)
        try:
            return int(v, 10)
        except ValueError:
            # Handle float-formatted strings like "123.0" that pandas emits
            # when a nullable-int DB column is loaded as float64.
            try:
                f = float(v)
                if math.isfinite(f):
                    return int(f)
            except ValueError:
                pass
            if all(c in "0123456789abcdefABCDEF" for c in v):
                return int(v, 16)
            # Non-numeric code (e.g. 'GRC1'): use stable hash as int ID
            return abs(hash(v)) % (10 ** 9)
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return int(value)
    return int(value)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample")


# ---------------------------------------------------------------------------
# Post-load enrichment helpers (reused by both CSV and SQL loaders)
# ---------------------------------------------------------------------------

# Percentile boundaries used by _derive_playing_style
_STYLE_HI_PCTILE = 0.65   # "high" threshold  (top 35 %)
_STYLE_LO_PCTILE = 0.35   # "low"  threshold  (bottom 35 %)
_STYLE_MED_PCTILE = 0.50  # median threshold

# Minimum playing-time thresholds used by _compute_star_player_usage
_STAR_MIN_GAMES = 10
_STAR_MIN_MPG = 12.0

def _derive_playing_style(teams_df: pd.DataFrame) -> None:
    """Classify each team's playing style from its stats.  Modifies *teams_df* in-place.

    Classification is percentile-based so it adapts to any dataset scale:

    ┌─────────────────┬──────────────────────────────────────────────────────┐
    │ Style           │ Criteria (all relative to team population)           │
    ├─────────────────┼──────────────────────────────────────────────────────┤
    │ pace_and_space  │ pace ≥ p65  AND  3PA-rate ≥ p65                      │
    │ defensive       │ DRtg ≤ p35 (good defence)  AND  pace ≤ p35          │
    │ motion_offense  │ assists ≥ p65  AND  3PA-rate ≥ p50                   │
    │ pick_and_roll   │ assists ≥ p65  AND  3PA-rate < p50                   │
    │ post_up         │ 3PA-rate ≤ p35                                        │
    │ isolation       │ fallback (star-driven, low assists)                   │
    └─────────────────┴──────────────────────────────────────────────────────┘
    """
    if teams_df.empty:
        return
    required = ["pace", "three_point_attempt_rate", "assists_per_game", "defensive_rating"]
    if not all(c in teams_df.columns for c in required):
        return

    pace = pd.to_numeric(teams_df["pace"],                    errors="coerce").fillna(75.0)
    tpar = pd.to_numeric(teams_df["three_point_attempt_rate"], errors="coerce").fillna(0.35)
    apg  = pd.to_numeric(teams_df["assists_per_game"],         errors="coerce").fillna(22.0)
    drtg = pd.to_numeric(teams_df["defensive_rating"],         errors="coerce").fillna(110.0)

    n = len(teams_df)
    if n < 4:
        # Too few teams for meaningful percentile splits; keep existing values
        return

    pace_hi  = pace.quantile(_STYLE_HI_PCTILE)
    pace_lo  = pace.quantile(_STYLE_LO_PCTILE)
    tpar_hi  = tpar.quantile(_STYLE_HI_PCTILE)
    tpar_med = tpar.quantile(_STYLE_MED_PCTILE)
    tpar_lo  = tpar.quantile(_STYLE_LO_PCTILE)
    apg_hi   = apg.quantile(_STYLE_HI_PCTILE)
    drtg_lo  = drtg.quantile(_STYLE_LO_PCTILE)  # lower DRtg = better defence

    styles = []
    for p, t, a, d in zip(pace, tpar, apg, drtg):
        if p >= pace_hi and t >= tpar_hi:
            styles.append("pace_and_space")
        elif d <= drtg_lo and p <= pace_lo:
            styles.append("defensive")
        elif a >= apg_hi and t >= tpar_med:
            styles.append("motion_offense")
        elif a >= apg_hi:
            styles.append("pick_and_roll")
        elif t <= tpar_lo:
            styles.append("post_up")
        else:
            styles.append("isolation")

    teams_df["playing_style"] = styles


def _compute_star_player_usage(teams_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    """Set *star_player_usage* on *teams_df* from player statistics.  In-place.

    The "star player" is the player with the highest ``usg_pct`` among those
    with meaningful playing time (≥ 10 games **and** ≥ 12 min/game).
    ``star_player_usage`` is stored as a 0–1 fraction (``usg_pct / 100``),
    clipped to [0.10, 0.60].
    """
    if teams_df.empty or stats_df.empty:
        return
    if not {"team_id", "usg_pct"}.issubset(stats_df.columns):
        return

    s = stats_df.copy()
    for col in ["usg_pct", "games_played", "minutes_per_game"]:
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0.0)
        else:
            s[col] = 0.0

    qualified = s[(s["games_played"] >= _STAR_MIN_GAMES) & (s["minutes_per_game"] >= _STAR_MIN_MPG)]
    if qualified.empty:
        qualified = s  # fallback: no filtering

    # Normalise team_id to int for grouping
    qualified = qualified.copy()
    qualified["_tid"] = qualified["team_id"].apply(
        lambda v: _to_int(v) if pd.notna(v) else None
    )
    team_star = (
        qualified[qualified["_tid"].notna()]
        .groupby("_tid")["usg_pct"]
        .max()
        .div(100.0)
        .clip(0.10, 0.60)
    )

    if "id" not in teams_df.columns:
        return

    def _lookup(raw_id: Any) -> float:
        try:
            return float(team_star.get(_to_int(raw_id), 0.25))
        except Exception:
            return 0.25

    teams_df["star_player_usage"] = teams_df["id"].apply(_lookup)


def _fill_current_team_league(players_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    """Derive *current_team_id* / *current_league_id* from the most recent
    season's stats for players where those fields are null.  In-place.
    """
    if players_df.empty or stats_df.empty:
        return
    if not {"player_id", "team_id", "season"}.issubset(stats_df.columns):
        return
    if "id" not in players_df.columns:
        return

    s = stats_df.copy()
    s["_pid"] = s["player_id"].apply(lambda v: _to_int(v) if pd.notna(v) else None)
    latest = s[s["_pid"].notna()].sort_values("season").groupby("_pid").last().reset_index()
    pid_team   = dict(zip(latest["_pid"], latest["team_id"]))
    pid_league = dict(zip(latest["_pid"], latest["league_id"])) if "league_id" in latest.columns else {}

    def _resolve(raw_id: Any, mapping: dict) -> Any:
        try:
            return mapping.get(_to_int(raw_id))
        except Exception:
            return None

    for col, mapping in [("current_team_id", pid_team), ("current_league_id", pid_league)]:
        if col not in players_df.columns or not mapping:
            continue
        null_mask = players_df[col].isna()
        if null_mask.any():
            players_df.loc[null_mask, col] = players_df.loc[null_mask, "id"].apply(
                lambda v: _resolve(v, mapping)
            )


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_leagues(data_dir: str = DATA_DIR) -> List[League]:
    df = pd.read_csv(f"{data_dir}/leagues.csv")
    return [League(**r) for r in df.to_dict("records")]


def load_teams(data_dir: str = DATA_DIR) -> List[Team]:
    df = pd.read_csv(f"{data_dir}/teams.csv")
    results = []
    for r in df.to_dict("records"):
        r["id"] = _to_int(r["id"])
        r["league_id"] = _to_int(r["league_id"])
        results.append(Team(**r))
    return results


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
    if "competition" not in df.columns:
        df["competition"] = "RS"
    # Get the set of valid PlayerStats fields (those with defaults are optional)
    import dataclasses as _dc
    _ps_fields = {f.name for f in _dc.fields(PlayerStats)}
    results = []
    for r in df.to_dict("records"):
        r["player_id"] = _to_int(r["player_id"])
        r["team_id"] = _to_int(r["team_id"])
        r["league_id"] = _to_int(r["league_id"])
        # Drop any CSV columns not present in the dataclass to avoid TypeError
        filtered = {k: v for k, v in r.items() if k in _ps_fields}
        results.append(PlayerStats(**filtered))
    return results


def load_team_player_relations(data_dir: str = DATA_DIR) -> List[TeamPlayerRelation]:
    df = pd.read_csv(f"{data_dir}/team_player_relations.csv")
    results = []
    for r in df.to_dict("records"):
        r["team_id"] = _to_int(r["team_id"])
        r["player_id"] = _to_int(r["player_id"])
        results.append(TeamPlayerRelation(**r))
    return results


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
        self._league_map = {lg.id: lg for lg in self.leagues}
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

    # Convert hex ID columns in stats_df and rels_df
    for col in ["player_id", "team_id", "league_id"]:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(lambda v: None if pd.isna(v) else _to_int(v))
    for col in ["team_id", "player_id"]:
        if col in rels_df.columns:
            rels_df[col] = rels_df[col].apply(lambda v: None if pd.isna(v) else _to_int(v))

    # Ensure competition column exists (default "RS" for legacy CSV files)
    if "competition" not in stats_df.columns:
        stats_df["competition"] = "RS"

    # Enrich teams: derive playing_style and star_player_usage from actual stats
    _derive_playing_style(teams_df)
    _compute_star_player_usage(teams_df, stats_df)

    # Fill missing current_team_id / current_league_id from most recent stats
    _fill_current_team_league(players_df, stats_df)

    league_dict: Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()}
    team_dict:   Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in teams_df.iterrows()}
    player_dict: Dict[int, dict] = {_to_int(r["id"]): r.to_dict() for _, r in players_df.iterrows()}

    league_teams: Dict[int, List[int]] = {}
    for _, t in teams_df.iterrows():
        league_teams.setdefault(_to_int(t["league_id"]), []).append(_to_int(t["id"]))

    data = {
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

    for w in validate_dataframes(data):
        logger.warning(w)

    return data


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

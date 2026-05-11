"""Azure SQL Server data loader for the Basketball Performance AI.

Provides the same ``load_all_data()`` interface as the file-based
``src.data.loader`` module, but reads from an Azure SQL Server database
instead of local CSV files.

Configuration is read exclusively from environment variables (never
hardcoded).  The recommended approach is to place a ``.env`` file in the
project root (see ``.env.example``) and let ``python-dotenv`` load it.

Required tables (column names must match the dataclass fields in
``src.data.models``):

    leagues               – League rows
    teams                 – Team rows
    players               – Player rows (nullable: current_team_id,
                            current_league_id, draft_year, draft_pick)
    player_stats          – PlayerStats rows
    team_player_relations – TeamPlayerRelation rows

Environment variables
---------------------
AZURE_SQL_CONNECTION_STRING
    Full SQLAlchemy connection URL, e.g.:
    ``mssql+pyodbc://user:pass@srv.database.windows.net/db
      ?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes``
    When set, the individual variables below are ignored.

AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USERNAME,
AZURE_SQL_PASSWORD, AZURE_SQL_DRIVER
    Used to build the connection string when
    ``AZURE_SQL_CONNECTION_STRING`` is not set.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd

# Load .env if present (no-op when python-dotenv is not installed or no file)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_connection_url() -> str:
    """Build a SQLAlchemy connection URL from environment variables."""
    url = os.environ.get("AZURE_SQL_CONNECTION_STRING", "").strip()
    if url:
        return url

    server   = os.environ.get("AZURE_SQL_SERVER", "")
    database = os.environ.get("AZURE_SQL_DATABASE", "")
    username = os.environ.get("AZURE_SQL_USERNAME", "")
    password = os.environ.get("AZURE_SQL_PASSWORD", "")
    driver   = os.environ.get("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    if not all([server, database, username, password]):
        raise EnvironmentError(
            "Azure SQL credentials are incomplete. "
            "Set AZURE_SQL_CONNECTION_STRING or all of: "
            "AZURE_SQL_SERVER, AZURE_SQL_DATABASE, "
            "AZURE_SQL_USERNAME, AZURE_SQL_PASSWORD."
        )

    params = quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def _get_engine():
    """Return a SQLAlchemy engine, importing SQLAlchemy lazily."""
    try:
        from sqlalchemy import create_engine  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for SQL data loading. "
            "Install it with: pip install sqlalchemy pyodbc"
        ) from exc

    url = _build_connection_url()
    return create_engine(url, pool_pre_ping=True)


def _read_table(engine, table: str) -> pd.DataFrame:
    """Read a full table and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_table(table, conn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_connection() -> bool:
    """Return True if a connection to Azure SQL can be established.

    Raises an explanatory exception on failure.
    """
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(_text("SELECT 1"))
    return True


def load_all_data_from_sql(engine=None) -> Dict[str, Any]:
    """Load all basketball data from Azure SQL Server.

    Returns the same flat dict as ``src.data.loader.load_all_data()``:

        leagues, teams, players, player_stats, team_player_relations
            – ``pd.DataFrame``
        league_dict, team_dict, player_dict
            – ``{id: row_dict}``
        league_teams
            – ``{league_id: [team_ids]}``
    """
    if engine is None:
        engine = _get_engine()

    leagues_df = _read_table(engine, "leagues")
    teams_df   = _read_table(engine, "teams")
    players_df = _read_table(engine, "players")
    stats_df   = _read_table(engine, "player_stats")
    rels_df    = _read_table(engine, "team_player_relations")

    # Normalise nullable int columns (same logic as the file loader)
    for col in ["current_team_id", "current_league_id", "draft_year", "draft_pick"]:
        if col in players_df.columns:
            players_df[col] = players_df[col].where(players_df[col].notna(), other=None)

    league_dict: Dict[int, dict] = {
        int(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()
    }
    team_dict: Dict[int, dict] = {
        int(r["id"]): r.to_dict() for _, r in teams_df.iterrows()
    }
    player_dict: Dict[int, dict] = {
        int(r["id"]): r.to_dict() for _, r in players_df.iterrows()
    }

    league_teams: Dict[int, List[int]] = {}
    for _, t in teams_df.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    return {
        "leagues":               leagues_df,
        "teams":                 teams_df,
        "players":               players_df,
        "player_stats":          stats_df,
        "team_player_relations": rels_df,
        "league_dict":           league_dict,
        "team_dict":             team_dict,
        "player_dict":           player_dict,
        "league_teams":          league_teams,
    }


# Alias so callers can do: from src.data.sql_loader import load_all_data
load_all_data = load_all_data_from_sql


# ---------------------------------------------------------------------------
# Lazy import helper (avoids NameError when sqlalchemy is absent at import)
# ---------------------------------------------------------------------------

def _text(sql: str):
    from sqlalchemy import text  # type: ignore
    return text(sql)

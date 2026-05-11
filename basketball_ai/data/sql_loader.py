"""Azure SQL Server data loader for the Basketball Performance AI.

Provides the same ``load_all_data()`` interface as the file-based
``src.data.loader`` module, but reads from an Azure SQL Server database
instead of local CSV files.

Configuration is read exclusively from environment variables (never
hardcoded).  The recommended approach is to place a ``.env`` file in the
project root (see ``.env.example``) and let ``python-dotenv`` load it.

The loader **automatically discovers** which tables in the database
correspond to the five required datasets (leagues, teams, players,
player_stats, team_player_relations) by inspecting the schema and
scoring each candidate table according to how many of its columns match
the expected set of "signature columns".  Explicit table-name overrides
can be provided via environment variables (see below) or programmatically
via ``load_all_data_from_sql(table_mapping=...)``.

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

AZURE_SQL_TABLE_LEAGUES, AZURE_SQL_TABLE_TEAMS, AZURE_SQL_TABLE_PLAYERS,
AZURE_SQL_TABLE_PLAYER_STATS, AZURE_SQL_TABLE_TEAM_PLAYER_RELATIONS
    Optional explicit table-name overrides.  When set, auto-discovery is
    skipped for that logical table.
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
# Signature columns used for auto-discovery
# ---------------------------------------------------------------------------

# Minimum overlap fraction required to accept a candidate table.
_MIN_OVERLAP = 0.5

# For each logical dataset, the set of columns that *uniquely* identify it.
# These are drawn from the dataclass definitions in src.data.models.
_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "leagues": [
        "id", "name", "country", "tier",
        "competitiveness_score", "avg_pace", "avg_offensive_rating",
    ],
    "teams": [
        "id", "name", "league_id", "playing_style", "pace",
        "offensive_rating", "defensive_rating", "three_point_attempt_rate",
    ],
    "players": [
        "id", "name", "age", "position", "nationality",
        "height_cm", "weight_kg", "dominant_hand",
    ],
    "player_stats": [
        "player_id", "season", "team_id", "league_id",
        "games_played", "minutes_per_game", "points", "rebounds",
        "assists", "fg_pct", "rating",
    ],
    "team_player_relations": [
        "team_id", "player_id", "season", "role",
    ],
}

# Environment-variable overrides for individual table names.
_TABLE_ENV_VARS: Dict[str, str] = {
    "leagues":               "AZURE_SQL_TABLE_LEAGUES",
    "teams":                 "AZURE_SQL_TABLE_TEAMS",
    "players":               "AZURE_SQL_TABLE_PLAYERS",
    "player_stats":          "AZURE_SQL_TABLE_PLAYER_STATS",
    "team_player_relations": "AZURE_SQL_TABLE_TEAM_PLAYER_RELATIONS",
}


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


def get_engine():
    """Return a SQLAlchemy engine built from environment variables.

    Lazily imports SQLAlchemy so the module remains importable without it.
    """
    try:
        from sqlalchemy import create_engine  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for SQL data loading. "
            "Install it with: pip install sqlalchemy pyodbc"
        ) from exc

    url = _build_connection_url()
    return create_engine(url, pool_pre_ping=True)


# Internal alias kept for backwards-compat with any internal callers.
_get_engine = get_engine


def _read_table(engine, table: str) -> pd.DataFrame:
    """Read a full table and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_table(table, conn)


def _get_db_schema(engine) -> Dict[str, List[str]]:
    """Return a mapping of {table_name: [column_names]} for every table
    visible in the current database / schema."""
    from sqlalchemy import inspect as _inspect  # type: ignore
    inspector = _inspect(engine)
    result: Dict[str, List[str]] = {}
    for table_name in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns(table_name)]
        result[table_name] = cols
    return result


def _score_table(actual_cols: List[str], expected_columns: List[str]) -> float:
    """Return the fraction of *expected_columns* present in *actual_cols*.

    A score of 1.0 means every expected column was found; 0.0 means none.
    Comparison is case-insensitive.
    """
    actual_lower = {c.lower() for c in actual_cols}
    matches = sum(1 for s in expected_columns if s.lower() in actual_lower)
    return matches / len(expected_columns) if expected_columns else 0.0


def _discover_table_mapping(engine) -> Dict[str, str]:
    """Inspect the connected database and return a mapping of
    ``{logical_name: actual_table_name}`` for the five required datasets.

    For each logical dataset the function:

    1. Checks whether an explicit override exists (environment variable
       ``AZURE_SQL_TABLE_<NAME>`` or a table whose name exactly matches
       the logical key, case-insensitively).
    2. If no exact match, scores every table in the DB by computing how
       many of the dataset's expected columns are present.
    3. Selects the table with the highest score, provided it meets the
       minimum overlap threshold (:data:`_MIN_OVERLAP`).

    Raises :class:`ValueError` if no suitable table can be found for any
    of the required datasets.
    """
    schema = _get_db_schema(engine)
    table_names_lower = {t.lower(): t for t in schema}
    mapping: Dict[str, str] = {}

    for logical, expected_columns in _REQUIRED_COLUMNS.items():
        # 1. Explicit env-var override
        env_override = os.environ.get(_TABLE_ENV_VARS[logical], "").strip()
        if env_override:
            if env_override not in schema:
                raise ValueError(
                    f"Table '{env_override}' specified via "
                    f"{_TABLE_ENV_VARS[logical]} was not found in the database."
                )
            mapping[logical] = env_override
            continue

        # 2. Exact case-insensitive match on the logical name
        if logical.lower() in table_names_lower:
            mapping[logical] = table_names_lower[logical.lower()]
            continue

        # 3. Score all tables and pick the best
        best_table: Optional[str] = None
        best_score = 0.0
        for table_name, cols in schema.items():
            score = _score_table(cols, expected_columns)
            if score > best_score:
                best_score = score
                best_table = table_name

        if not schema:
            raise ValueError(
                f"Could not find a suitable table for '{logical}': "
                f"the database appears to have no visible tables. "
                f"Set {_TABLE_ENV_VARS[logical]} to specify the table name explicitly."
            )
        if best_table is None or best_score < _MIN_OVERLAP:
            raise ValueError(
                f"Could not find a suitable table for '{logical}' "
                f"(best candidate: '{best_table}', score: {best_score:.0%} "
                f"— minimum required: {_MIN_OVERLAP:.0%}). "
                f"Available tables: {sorted(schema)}. "
                f"Set the environment variable {_TABLE_ENV_VARS[logical]} "
                f"to specify the table name explicitly."
            )
        mapping[logical] = best_table

    return mapping


# ---------------------------------------------------------------------------
# Lazy import helper (avoids NameError when sqlalchemy is absent at import)
# ---------------------------------------------------------------------------

def _text(sql: str):
    from sqlalchemy import text  # type: ignore
    return text(sql)


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


def get_table_mapping(engine=None) -> Dict[str, str]:
    """Return the auto-discovered ``{logical_name: actual_table_name}`` mapping.

    Useful for inspecting which tables were selected before loading data,
    e.g. to display them in a GUI.  Accepts an optional pre-built engine;
    creates one from environment variables otherwise.
    """
    if engine is None:
        engine = _get_engine()
    return _discover_table_mapping(engine)


def load_all_data_from_sql(engine=None, table_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Load all basketball data from Azure SQL Server.

    Automatically discovers which tables in the database correspond to the
    five required datasets by inspecting the schema and scoring column
    overlap against known signatures.  Explicit overrides can be passed via
    *table_mapping* or via ``AZURE_SQL_TABLE_*`` environment variables.

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

    if table_mapping is None:
        table_mapping = _discover_table_mapping(engine)

    leagues_df = _read_table(engine, table_mapping["leagues"])
    teams_df   = _read_table(engine, table_mapping["teams"])
    players_df = _read_table(engine, table_mapping["players"])
    stats_df   = _read_table(engine, table_mapping["player_stats"])
    rels_df    = _read_table(engine, table_mapping["team_player_relations"])

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


# Alias so callers can do: from basketball_ai.data.sql_loader import load_all_data
load_all_data = load_all_data_from_sql

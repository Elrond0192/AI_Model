"""Azure SQL Server data loader for the Basketball Performance AI.

Provides the same ``load_all_data()`` interface as the file-based
``basketball_ai.data.loader`` module, but reads from an Azure SQL Server database
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

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

_logger = logging.getLogger(__name__)

import pandas as pd

# Load .env if present (no-op when python-dotenv is not installed or no file)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)
except ImportError:
    pass

from basketball_ai.data.loader import (
    _derive_playing_style,
    _compute_star_player_usage,
    _fill_current_team_league,
)


# ---------------------------------------------------------------------------
# Signature columns used for auto-discovery
# ---------------------------------------------------------------------------

# Minimum overlap fraction required to accept a candidate table.
_MIN_OVERLAP = 0.5

# For each logical dataset, the set of columns that *uniquely* identify it.
# These are drawn from the dataclass definitions in basketball_ai.data.models.
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
# Column-mapping configuration — delegated to schema_mapping.py
# ---------------------------------------------------------------------------
# The authoritative mapping lives in basketball_ai/data/schema_mapping.py.
# That file is the single place to add/edit column mappings with full docs.
# Here we only keep the env-var name registry and the thin helpers that
# merge env-var / GUI overrides on top of the schema_mapping defaults.

# Env-var names for per-logical column-rename overrides.
_COLUMN_RENAME_ENV_VARS: Dict[str, str] = {
    "leagues":               "AZURE_SQL_COLUMN_RENAMES_LEAGUES",
    "teams":                 "AZURE_SQL_COLUMN_RENAMES_TEAMS",
    "players":               "AZURE_SQL_COLUMN_RENAMES_PLAYERS",
    "player_stats":          "AZURE_SQL_COLUMN_RENAMES_PLAYER_STATS",
    "team_player_relations": "AZURE_SQL_COLUMN_RENAMES_TEAM_PLAYER_RELATIONS",
}


def _get_column_renames(logical: str) -> Dict[str, str]:
    """Return column rename map for *logical*.

    Priority (highest first):
    1. AZURE_SQL_COLUMN_RENAMES_<LOGICAL> env var (set by GUI or .env)
    2. schema_mapping.get_rename_map() defaults
    """
    try:
        from basketball_ai.data.schema_mapping import get_rename_map as _sm_renames
        renames = _sm_renames(logical)
    except ImportError:
        renames = {}

    env_key = _COLUMN_RENAME_ENV_VARS.get(logical, "")
    raw = os.environ.get(env_key, "").strip()
    if raw:
        # Env-var overrides replace the whole map (GUI saved explicit pairs)
        renames = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                src, _, dst = pair.partition(":")
                renames[src.strip()] = dst.strip()
    return renames


def _apply_column_mapping(df: "pd.DataFrame", logical: str) -> "pd.DataFrame":
    """Rename columns, apply computed derivations, fill missing defaults.

    Delegates to schema_mapping.py for all configuration; falls back to
    safe no-ops when the module is unavailable.

    Steps:
    1. Rename DB columns to logical names (case-insensitive).
    2. Apply compute functions (age from BirthDate, per-game stats, etc.).
    3. Inject default values for any expected column still missing.
    """
    # 1. Rename
    renames = _get_column_renames(logical)
    col_lower = {c.lower(): c for c in df.columns}
    rename_map = {
        col_lower[src.lower()]: dst
        for src, dst in renames.items()
        if src.lower() in col_lower and col_lower[src.lower()] != dst
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2. Computed columns (via schema_mapping)
    try:
        from basketball_ai.data.schema_mapping import apply_computes as _sm_computes
        df = _sm_computes(df, logical)
    except ImportError:
        pass

    # 3. Defaults (via schema_mapping)
    try:
        from basketball_ai.data.schema_mapping import get_defaults as _sm_defaults
        for col, default in _sm_defaults(logical).items():
            if col not in df.columns:
                df[col] = default
    except ImportError:
        pass

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_connection_string(conn_str: str) -> str:
    """Convert various SQL Server connection string formats to a SQLAlchemy URL.

    Accepts three input formats and normalizes them all to a SQLAlchemy
    ``mssql+pyodbc://`` URL:

    1. **SQLAlchemy URL** – already contains ``://``.  Returned unchanged.
    2. **ODBC connection string** – contains a ``Driver=`` key (e.g. the string
       shown by the Azure Portal "ODBC" tab).  Wrapped using the
       ``odbc_connect`` query-string parameter so that SQLAlchemy passes it
       straight through to pyodbc.
    3. **ADO.NET connection string** – the format shown by the Azure Portal
       "ADO.NET" tab, e.g.
       ``Server=tcp:host,1433;Initial Catalog=db;User ID=u;Password=p;…``.
       Parsed and converted to an ODBC string (adding a default driver when
       none is specified), then wrapped as in case 2.
    """
    s = conn_str.strip()

    # 1. Already a SQLAlchemy / generic URL  --------------------------------
    if re.match(r"^[a-zA-Z][\w+.-]*://", s):
        return s

    # Parse all key=value pairs (semicolon-separated, case-insensitive keys)
    pairs: Dict[str, str] = {}
    for part in s.split(";"):
        part = part.strip()
        if "=" in part:
            key, _, val = part.partition("=")
            pairs[key.strip().lower()] = val.strip()

    # 2. ODBC string – has an explicit Driver= key  -------------------------
    if "driver" in pairs:
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(s)}"

    # 3. ADO.NET string – map well-known key aliases to ODBC names  ---------
    server = (
        pairs.get("server")
        or pairs.get("data source")
        or pairs.get("addr")
        or pairs.get("address")
        or pairs.get("network address")
        or ""
    ).strip()
    # Strip the optional "tcp:" prefix that Azure Portal adds
    server = re.sub(r"^tcp:", "", server)

    database = (
        pairs.get("initial catalog")
        or pairs.get("database")
        or ""
    ).strip()

    uid = (
        pairs.get("user id")
        or pairs.get("uid")
        or pairs.get("user")
        or ""
    ).strip()

    pwd = (
        pairs.get("password")
        or pairs.get("pwd")
        or ""
    ).strip()

    _bool_map = {"true": "yes", "false": "no", "1": "yes", "0": "no"}
    encrypt = _bool_map.get(pairs.get("encrypt", "yes").lower(), pairs.get("encrypt", "yes"))
    trust_cert = _bool_map.get(pairs.get("trustservercertificate", "no").lower(), pairs.get("trustservercertificate", "no"))
    _default_timeout = os.environ.get("AZURE_SQL_CONNECT_TIMEOUT", "60")
    _raw_timeout = pairs.get("connection timeout")
    timeout = _raw_timeout if _raw_timeout is not None else _default_timeout
    driver = os.environ.get("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
        f"Connection Timeout={timeout};"
    )
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"


def _build_connection_url() -> str:
    """Build a SQLAlchemy connection URL from environment variables."""
    raw = os.environ.get("AZURE_SQL_CONNECTION_STRING", "").strip()
    if raw:
        return _normalize_connection_string(raw)

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

    _timeout = os.environ.get("AZURE_SQL_CONNECT_TIMEOUT", "60")
    params = quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout={_timeout};"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def _is_transient_connection_error(exc: Exception) -> bool:
    """Return True for transient network/login errors that are worth retrying.

    Covers the pyodbc / SQLAlchemy ``08001`` state (TCP timeout, login timeout)
    that occurs when Azure SQL Serverless is waking from auto-pause.
    """
    msg = str(exc)
    return "08001" in msg or "timeout" in msg.lower()

def get_engine(max_retries: int = 3, retry_delay: float = 15.0):
    """Return a SQLAlchemy engine built from environment variables.

    Lazily imports SQLAlchemy so the module remains importable without it.

    On transient connection failures (sqlstate ``08001``, e.g. Azure SQL
    Serverless waking from auto-pause) the function retries up to
    *max_retries* times, waiting *retry_delay* seconds between attempts.

    Raises ``ValueError`` if *max_retries* is less than 1.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    try:
        from sqlalchemy import create_engine, text  # type: ignore
        from sqlalchemy.exc import OperationalError as _OpError  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for SQL data loading. "
            "Install it with: pip install sqlalchemy pyodbc"
        ) from exc

    url = _build_connection_url()
    engine = create_engine(url, pool_pre_ping=True)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except _OpError as exc:
            last_exc = exc
            is_last_attempt = attempt == max_retries
            if not is_last_attempt and _is_transient_connection_error(exc):
                _logger.warning(
                    "Transient connection error (attempt %d/%d): %s — retrying in %.0f s …",
                    attempt, max_retries, exc, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                break

    # Reaching here means the loop broke due to an exception.
    # last_exc is always set because max_retries >= 1 (validated above) and
    # every iteration that reaches the except clause sets last_exc.
    raise last_exc  # type: ignore[misc]


# Internal alias kept for backwards-compat with any internal callers.
_get_engine = get_engine


def _read_table(engine, table: str) -> pd.DataFrame:
    """Read a full table and return a DataFrame.

    *table* may be a plain name or a ``schema.table`` qualified name.
    """
    with engine.connect() as conn:
        if "." in table:
            schema, tname = table.split(".", 1)
            return pd.read_sql_table(tname, conn, schema=schema)
        return pd.read_sql_table(table, conn)


def _execute_query(engine, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def _discover_league_seasons(engine) -> List[tuple]:
    """Discover (league_id, season) pairs from Anagrafiche schema table names.

    Returns e.g. [('ITA1', '2024'), ('GRC1', '2024')] based on tables like
    Anagrafiche.ITA1_2024, Anagrafiche.GRC1_2024 found in the database.
    """
    try:
        from basketball_ai.data import schema_mapping
        df = _execute_query(engine, schema_mapping.LEAGUES_DISCOVERY_QUERY)
        return [(str(r["id"]).strip(), str(r["season"]).strip()) for _, r in df.iterrows()]
    except Exception:
        return []


def _discover_normalized_name_tags(engine, league_seasons: List[tuple]) -> set:
    """Return the set of ``"<league>_<season>"`` tags whose Anagrafiche player
    table contains the ``NormalizedPlayerName`` column.

    Queries ``INFORMATION_SCHEMA.COLUMNS`` with a fully static SQL string (no
    user-controlled values interpolated) and filters the results in Python, so
    there is no risk of SQL injection.
    """
    if not league_seasons:
        return set()
    sql = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = 'Anagrafiche' "
        "AND COLUMN_NAME = 'NormalizedPlayerName'"
    )
    try:
        df = _execute_query(engine, sql)
        known_tags = {f"{l}_{s}" for l, s in league_seasons}
        return set(df["TABLE_NAME"].str.strip().tolist()) & known_tags
    except Exception:
        return set()


def _discover_existing_table_tags(
    engine,
    league_seasons: List[tuple],
    schema: str,
    prefix: str = "",
) -> set:
    """Return the set of ``"<league>_<season>"`` tags for which
    ``{schema}.{prefix}{tag}`` exists in the database.

    Queries ``INFORMATION_SCHEMA.TABLES`` using only hardcoded schema/prefix
    values (never user-supplied) and filters the results in Python, so there
    is no risk of SQL injection.

    Parameters
    ----------
    schema:
        SQL Server schema name to search (e.g. ``'Analisi'``, ``'Boxscore'``).
    prefix:
        Table-name prefix that precedes the tag (e.g. ``'AdvancedStats_Clutch_'``).
        When empty the table name itself is expected to equal the tag
        (e.g. ``Boxscore.GRC1_2024``).
    """
    if not league_seasons:
        return set()
    sql = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = '{schema}'"
    )
    try:
        df = _execute_query(engine, sql)
        known_tags = {f"{l}_{s}" for l, s in league_seasons}
        table_names: List[str] = df["TABLE_NAME"].str.strip().tolist()
        if prefix:
            found = {
                name[len(prefix):]
                for name in table_names
                if name.startswith(prefix)
            }
        else:
            # No prefix: the table name itself is the tag (e.g. Boxscore.GRC1_2024)
            found = set(table_names)
        return found & known_tags
    except Exception:
        return set()


def _get_table_queries(engine=None) -> Dict[str, str]:
    """Return SQL queries for all logical tables.

    When *engine* is supplied and the dynamic query path is enabled, leagues
    and seasons are discovered from Anagrafiche schema table names and queries
    are built dynamically (one UNION ALL branch per league/season).

    Discovers which optional source tables actually exist for each
    league/season so the query builder can emit safe empty-stub CTEs (or skip
    branches entirely) instead of referencing non-existent tables.

    Falls back to the static TABLE_QUERIES dict (empty by default).
    """
    try:
        from basketball_ai.data import schema_mapping
    except ImportError:
        return {}
    if engine is not None and _use_query_path(engine):
        league_seasons = _discover_league_seasons(engine)
        if league_seasons and hasattr(schema_mapping, "get_table_queries"):
            tags_with_normalized = _discover_normalized_name_tags(engine, league_seasons)
            tags_with_boxscore = _discover_existing_table_tags(engine, league_seasons, "Boxscore")
            tags_with_team_table = _discover_existing_table_tags(engine, league_seasons, "Anagrafiche", "Team_")
            tags_with_player_stats = _discover_existing_table_tags(engine, league_seasons, "Analisi", "AdvancedStats_Player_")
            tags_with_roles = _discover_existing_table_tags(engine, league_seasons, "Analisi", "PlayerRoles_")
            tags_with_onoff = _discover_existing_table_tags(engine, league_seasons, "Analisi", "AdvancedStatsOnOffCourt_")
            tags_with_clutch = _discover_existing_table_tags(engine, league_seasons, "Analisi", "AdvancedStats_Clutch_")
            tags_with_team_stats = _discover_existing_table_tags(engine, league_seasons, "Analisi", "AdvancedStatsTeam_")
            return schema_mapping.get_table_queries(
                league_seasons,
                tags_with_normalized_name=tags_with_normalized,
                tags_with_clutch=tags_with_clutch,
                tags_with_boxscore=tags_with_boxscore,
                tags_with_team_table=tags_with_team_table,
                tags_with_player_stats=tags_with_player_stats,
                tags_with_roles=tags_with_roles,
                tags_with_onoff=tags_with_onoff,
                tags_with_team_stats=tags_with_team_stats,
            )
    return getattr(schema_mapping, "TABLE_QUERIES", {})


def _use_query_path(engine=None) -> bool:
    enabled = os.environ.get("AZURE_SQL_USE_QUERIES", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    dialect_name = getattr(getattr(engine, "dialect", None), "name", "").lower() if engine is not None else ""
    return dialect_name != "sqlite"


def _should_use_query(logical: str, engine=None, table_queries: Optional[Dict[str, str]] = None) -> bool:
    if table_queries is not None:
        return logical in table_queries
    return _use_query_path(engine) and logical in _get_table_queries(engine)


def _normalize_identifier(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        v = value.strip()
        # Try to convert pure-numeric or hex strings to int
        try:
            return int(v, 10)
        except ValueError:
            try:
                return int(v, 16)
            except ValueError:
                return v
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _get_db_schema(engine) -> Dict[str, List[str]]:
    """Return a mapping of {schema.table: [column_names]} for every table
    visible in the current database across all non-system schemas."""
    from sqlalchemy import inspect as _inspect  # type: ignore
    inspector = _inspect(engine)
    result: Dict[str, List[str]] = {}
    # Collect all schemas; fall back to default schema only
    try:
        schemas = inspector.get_schema_names()
    except Exception:
        schemas = [None]
    _system = {"sys", "information_schema", "guest", "db_owner",
               "db_accessadmin", "db_securityadmin", "db_ddladmin",
               "db_backupoperator", "db_datareader", "db_datawriter",
               "db_denydatareader", "db_denydatawriter"}
    dialect_name = getattr(getattr(engine, "dialect", None), "name", "").lower()
    for schema in schemas:
        if schema and schema.lower() in _system:
            continue
        try:
            table_names = inspector.get_table_names(schema=schema)
        except Exception:
            continue
        for table_name in table_names:
            try:
                cols = [c["name"] for c in inspector.get_columns(table_name, schema=schema)]
            except Exception:
                cols = []
            if dialect_name == "sqlite" and schema in {None, "main"}:
                qualified = table_name
            else:
                qualified = f"{schema}.{table_name}" if schema else table_name
            result[qualified] = cols
    return result


def _score_table(actual_cols: List[str], expected_columns: List[str],
                 rename_map: Optional[Dict[str, str]] = None) -> float:
    """Return the fraction of *expected_columns* present in *actual_cols*.

    A score of 1.0 means every expected column was found; 0.0 means none.
    Comparison is case-insensitive.  When *rename_map* is provided, actual
    column names are first translated to their logical equivalents before
    matching.
    """
    actual_lower = {c.lower() for c in actual_cols}
    if rename_map:
        rename_lower = {k.lower(): v.lower() for k, v in rename_map.items()}
        actual_lower = {rename_lower.get(c, c) for c in actual_lower}
    matches = sum(1 for s in expected_columns if s.lower() in actual_lower)
    return matches / len(expected_columns) if expected_columns else 0.0


def _discover_table_mapping(engine, table_queries: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Inspect the connected database and return a mapping of logical names
    to physical tables for datasets that still use direct table reads.

    Query-backed datasets (present in *table_queries*) are skipped.
    """
    schema = _get_db_schema(engine)
    table_names_lower = {t.lower(): t for t in schema}
    mapping: Dict[str, str] = {}

    for logical, expected_columns in _REQUIRED_COLUMNS.items():
        if table_queries and logical in table_queries:
            continue

        env_override = os.environ.get(_TABLE_ENV_VARS[logical], "").strip()
        if getattr(getattr(engine, "dialect", None), "name", "").lower() == "sqlite" and "." in env_override:
            env_override = ""
        if env_override:
            if env_override not in schema:
                raise ValueError(
                    f"Table '{env_override}' specified via "
                    f"{_TABLE_ENV_VARS[logical]} was not found in the database."
                )
            mapping[logical] = env_override
            continue

        if logical.lower() in table_names_lower:
            mapping[logical] = table_names_lower[logical.lower()]
            continue

        renames = _get_column_renames(logical)
        best_table: Optional[str] = None
        best_score = 0.0
        for table_name, cols in schema.items():
            score = _score_table(cols, expected_columns, rename_map=renames)
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
    """Load all basketball data from Azure SQL Server."""
    if engine is None:
        engine = _get_engine()

    # Discover queries once — this also runs league/season discovery from table names.
    table_queries = _get_table_queries(engine)

    resolved_mapping: Dict[str, str] = dict(table_mapping or {})
    needed_tables = [
        logical
        for logical in _REQUIRED_COLUMNS
        if logical not in table_queries and logical not in resolved_mapping
    ]
    if needed_tables:
        discovered = _discover_table_mapping(engine, table_queries)
        for logical in needed_tables:
            resolved_mapping[logical] = discovered[logical]

    def _load_logical(logical: str) -> pd.DataFrame:
        if logical in table_queries:
            return _execute_query(engine, table_queries[logical])
        return _apply_column_mapping(_read_table(engine, resolved_mapping[logical]), logical)

    leagues_df = _load_logical("leagues")
    teams_df = _load_logical("teams")
    players_df = _load_logical("players")
    stats_df = _load_logical("player_stats")
    rels_df = _load_logical("team_player_relations")

    # Recompute league avg_pace and avg_offensive_rating from real team data
    # so that each league reflects its actual teams rather than a global default.
    if (
        not teams_df.empty
        and "league_id" in teams_df.columns
        and not leagues_df.empty
        and "id" in leagues_df.columns
    ):
        def _col_mean(group: "pd.DataFrame", col: str) -> "Optional[float]":
            if col not in group.columns:
                return None
            val = pd.to_numeric(group[col], errors="coerce").mean()
            return val if pd.notna(val) else None

        _league_stat_cols = [
            ("pace", "avg_pace"),
            ("offensive_rating", "avg_offensive_rating"),
        ]
        for league_id, group in teams_df.groupby("league_id"):
            mask = leagues_df["id"] == league_id
            if not mask.any():
                continue
            for src_col, dst_col in _league_stat_cols:
                val = _col_mean(group, src_col)
                if val is not None:
                    leagues_df.loc[mask, dst_col] = val

    for col in ["current_team_id", "current_league_id", "draft_year", "draft_pick"]:
        if col in players_df.columns:
            players_df[col] = players_df[col].where(players_df[col].notna(), other=None)

    # Enrich teams: derive playing_style and star_player_usage from actual stats
    _derive_playing_style(teams_df)
    _compute_star_player_usage(teams_df, stats_df)

    # Fill missing current_team_id / current_league_id from most recent stats
    _fill_current_team_league(players_df, stats_df)

    league_dict: Dict[Any, dict] = {
        _normalize_identifier(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()
    }
    team_dict: Dict[Any, dict] = {
        _normalize_identifier(r["id"]): r.to_dict() for _, r in teams_df.iterrows()
    }
    player_dict: Dict[Any, dict] = {
        _normalize_identifier(r["id"]): r.to_dict() for _, r in players_df.iterrows()
    }

    league_teams: Dict[Any, List[Any]] = {}
    for _, t in teams_df.iterrows():
        league_id = _normalize_identifier(t["league_id"])
        team_id = _normalize_identifier(t["id"])
        league_teams.setdefault(league_id, []).append(team_id)

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


# Alias so callers can do: from basketball_ai.data.sql_loader import load_all_data
load_all_data = load_all_data_from_sql

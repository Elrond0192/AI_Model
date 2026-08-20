"""Canonical PostgreSQL data access for AI_Model.

Production reads only the five read-only ``ai_source`` views created by
``basketball_ai/data/ai_source_schema.sql``.  The model never knows the
physical BBallstat table layout.

Docker reaches PostgreSQL on the host through ``host.docker.internal``; SSH is
not part of this application.
"""
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from basketball_ai.data.loader import (
    _compute_star_player_usage,
    _derive_playing_style,
    _fill_current_team_league,
    _to_int,
    validate_dataframes,
)

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "leagues": {"id", "name"},
    "teams": {"id", "global_id", "name", "league_id"},
    "players": {"id", "global_id", "name", "position"},
    "player_stats": {
        "player_id",
        "season",
        "league_id",
        "games_played",
        "minutes_per_game",
        "points",
        "rating",
        "competition",
    },
    "team_player_relations": {"player_id", "team_id", "season"},
}


def get_engine(url: str | None = None) -> Engine:
    """Create the production SQLAlchemy engine."""
    database_url = url or os.environ.get("DATABASE_URL", "")
    if not database_url:
        from basketball_ai.data.connection_profiles import active_profile_name, profile_url

        database_url = profile_url(active_profile_name())

    if not database_url.startswith("postgresql+psycopg://"):
        raise RuntimeError(
            "DATABASE_URL must use postgresql+psycopg://. "
            "SQL Server/Azure SQL URLs are not supported."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))},
    )


def _load_view(engine: Engine, schema: str, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(text(f'SELECT * FROM "{schema}"."{name}"'), engine)
    except Exception as exc:
        raise RuntimeError(
            f'Cannot read canonical view "{schema}"."{name}". '
            "Run basketball_ai/data/ai_source_schema.sql against the BBallstat "
            "PostgreSQL database and verify the AI read role has SELECT access."
        ) from exc


def _validate_contract(data: dict[str, pd.DataFrame], schema: str) -> None:
    errors: list[str] = []
    for view, required in _REQUIRED_COLUMNS.items():
        frame = data.get(view)
        if frame is None:
            errors.append(f"{schema}.{view}: missing")
            continue
        missing = sorted(required.difference(frame.columns))
        if missing:
            errors.append(f"{schema}.{view}: missing columns {', '.join(missing)}")

    errors.extend(validate_dataframes(data))

    if errors:
        raise RuntimeError("Invalid ai_source contract: " + " | ".join(errors))

    if data["players"].empty:
        raise RuntimeError(f"{schema}.players is empty")
    if data["teams"].empty:
        raise RuntimeError(f"{schema}.teams is empty")
    if data["player_stats"].empty:
        raise RuntimeError(
            f"{schema}.player_stats is empty. "
            "The adapter only publishes seasons with a non-null rating target."
        )


def _normalise_ids(data: dict[str, pd.DataFrame]) -> None:
    """Normalise canonical numeric IDs without destroying nullable relations."""
    id_columns = {
        "leagues": ("id",),
        "teams": ("id", "league_id"),
        "players": ("id", "current_team_id", "current_league_id"),
        "player_stats": ("player_id", "team_id", "league_id"),
        "team_player_relations": ("player_id", "team_id"),
    }
    for table, columns in id_columns.items():
        frame = data[table]
        for column in columns:
            if column not in frame.columns:
                continue
            frame[column] = frame[column].apply(
                lambda value: None if pd.isna(value) else _to_int(value)
            )

    if "season" in data["player_stats"].columns:
        data["player_stats"]["season"] = pd.to_numeric(
            data["player_stats"]["season"], errors="coerce"
        ).astype("Int64")
    if "season" in data["team_player_relations"].columns:
        data["team_player_relations"]["season"] = pd.to_numeric(
            data["team_player_relations"]["season"], errors="coerce"
        ).astype("Int64")


def _build_lookups(data: dict[str, pd.DataFrame]) -> None:
    def indexed(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
        if "id" not in frame.columns:
            return {}
        out: dict[int, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            if pd.isna(row["id"]):
                continue
            out[_to_int(row["id"])] = row.to_dict()
        return out

    data["league_dict"] = indexed(data["leagues"])
    data["team_dict"] = indexed(data["teams"])
    data["player_dict"] = indexed(data["players"])

    league_teams: dict[int, list[int]] = {}
    for team_id, team in data["team_dict"].items():
        league_id = team.get("league_id")
        if league_id is None or pd.isna(league_id):
            continue
        league_teams.setdefault(_to_int(league_id), []).append(team_id)
    data["league_teams"] = league_teams


def load_all_data(
    url: str | None = None,
    source_schema: str | None = None,
) -> dict[str, Any]:
    """Load and enrich the stable PostgreSQL contract used by training/API."""
    schema = source_schema or os.getenv("POSTGRES_SOURCE_SCHEMA", "ai_source")
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError("Invalid PostgreSQL source schema")

    engine = get_engine(url)
    views = (
        "leagues",
        "teams",
        "players",
        "player_stats",
        "team_player_relations",
    )
    data: dict[str, Any] = {
        name: _load_view(engine, schema, name)
        for name in views
    }

    _validate_contract(data, schema)
    _normalise_ids(data)

    _derive_playing_style(data["teams"])
    _compute_star_player_usage(data["teams"], data["player_stats"])
    _fill_current_team_league(data["players"], data["player_stats"])

    data["player_stats"] = (
        data["player_stats"]
        .dropna(subset=["player_id", "season", "rating"])
        .sort_values(["player_id", "season"])
        .reset_index(drop=True)
    )
    data["team_player_relations"] = (
        data["team_player_relations"]
        .dropna(subset=["player_id", "team_id", "season"])
        .sort_values(["season", "team_id", "player_id"])
        .reset_index(drop=True)
    )

    _build_lookups(data)
    return data

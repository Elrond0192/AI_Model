"""Canonical PostgreSQL data access for AI_Model.

Production reads the stable read-only ``ai_source`` contract.  In addition to
entity/player-season views, professional time-aware training requires
``team_season_stats`` so historical samples never see a team's future style.
The model never writes to BBallstat source schemas.
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
    "team_season_stats": {
        "team_id",
        "global_id",
        "name",
        "league_id",
        "season",
        "pace",
        "offensive_rating",
        "defensive_rating",
        "three_point_attempt_rate",
        "assists_per_game",
    },
}


def get_engine(url: str | None = None) -> Engine:
    database_url = url or os.environ.get("DATABASE_URL", "")
    if not database_url:
        from basketball_ai.data.connection_profiles import active_profile_name, profile_url

        database_url = profile_url(active_profile_name())
    if not database_url.startswith("postgresql+psycopg://"):
        raise RuntimeError(
            "DATABASE_URL must use postgresql+psycopg://. SQL Server/Azure SQL URLs are not supported."
        )
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))
        },
    )


def _load_view(engine: Engine, schema: str, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(text(f'SELECT * FROM "{schema}"."{name}"'), engine)
    except Exception as exc:
        extra = (
            " Then run basketball_ai/data/ai_source_team_season.sql."
            if name == "team_season_stats"
            else ""
        )
        raise RuntimeError(
            f'Cannot read canonical view "{schema}"."{name}". '
            "Run basketball_ai/data/ai_source_schema.sql against BBallstat PostgreSQL."
            + extra
            + " Verify the AI read role has SELECT access."
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

    core = {name: frame for name, frame in data.items() if name != "team_season_stats"}
    errors.extend(validate_dataframes(core))

    player_stats = data.get("player_stats")
    if player_stats is not None and {"player_id", "season"}.issubset(player_stats.columns):
        duplicates = player_stats.duplicated(["player_id", "season"], keep=False)
        if duplicates.any():
            errors.append(
                "player_stats must contain exactly one row per player_id+season"
            )

    team_history = data.get("team_season_stats")
    if team_history is not None and {"team_id", "season"}.issubset(team_history.columns):
        duplicates = team_history.duplicated(["team_id", "season"], keep=False)
        if duplicates.any():
            errors.append(
                "team_season_stats must contain exactly one row per team_id+season"
            )

    if errors:
        raise RuntimeError("Invalid ai_source contract: " + " | ".join(errors))
    for required_non_empty in ("players", "teams", "player_stats", "team_season_stats"):
        if data[required_non_empty].empty:
            raise RuntimeError(f"{schema}.{required_non_empty} is empty")


def _normalise_ids(data: dict[str, pd.DataFrame]) -> None:
    id_columns = {
        "leagues": ("id",),
        "teams": ("id", "league_id"),
        "players": ("id", "current_team_id", "current_league_id"),
        "player_stats": ("player_id", "team_id", "league_id"),
        "team_player_relations": ("player_id", "team_id"),
        "team_season_stats": ("team_id", "league_id"),
    }
    for table, columns in id_columns.items():
        frame = data[table]
        for column in columns:
            if column not in frame.columns:
                continue
            frame[column] = frame[column].apply(
                lambda value: None if pd.isna(value) else _to_int(value)
            )

    for table in ("player_stats", "team_player_relations", "team_season_stats"):
        if "season" in data[table].columns:
            data[table]["season"] = pd.to_numeric(
                data[table]["season"], errors="coerce"
            ).astype("Int64")


def _build_lookups(data: dict[str, pd.DataFrame]) -> None:
    def indexed(frame: pd.DataFrame, id_column: str = "id") -> dict[int, dict[str, Any]]:
        if id_column not in frame.columns:
            return {}
        out: dict[int, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            if pd.isna(row[id_column]):
                continue
            out[_to_int(row[id_column])] = row.to_dict()
        return out

    data["league_dict"] = indexed(data["leagues"])
    data["team_dict"] = indexed(data["teams"])
    data["player_dict"] = indexed(data["players"])

    team_season_dict: dict[tuple[int, int], dict[str, Any]] = {}
    for _, row in data["team_season_stats"].iterrows():
        if pd.isna(row.get("team_id")) or pd.isna(row.get("season")):
            continue
        team_season_dict[(_to_int(row["team_id"]), int(row["season"]))] = row.to_dict()
    data["team_season_dict"] = team_season_dict

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
        "team_season_stats",
    )
    data: dict[str, Any] = {name: _load_view(engine, schema, name) for name in views}

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
    data["team_season_stats"] = (
        data["team_season_stats"]
        .dropna(subset=["team_id", "season"])
        .sort_values(["team_id", "season"])
        .reset_index(drop=True)
    )

    _build_lookups(data)
    return data

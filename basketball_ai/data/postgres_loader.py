"""Canonical PostgreSQL data access for AI_Model.

Production reads entity/roster views from ``ai_source`` and the competition-
preserving statistical contract created by ``ai_source_competition.sql``.
Training therefore sees one observation per entity + league + season +
competition instead of collapsing PO/CUP/TOT into a single season row.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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

_SOURCE_VIEWS = {
    "leagues": "leagues",
    "teams": "teams",
    "players": "players",
    "player_stats": "player_competition_stats",
    "team_player_relations": "team_player_relations",
    "team_season_stats": "team_competition_stats",
}

_OPTIONAL_SOURCE_VIEWS = {
    "pbp_events": "simulation_pbp_events",
    "lineup_stints": "simulation_lineup_stints",
    "play_type_stats": "simulation_play_type_stats",
    "shot_profiles": "simulation_shot_profiles",
    "causal_panel": "simulation_causal_panel",
}

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "leagues": {"id", "name"},
    "teams": {"id", "global_id", "name", "league_id"},
    "players": {"id", "global_id", "name", "position"},
    "player_stats": {
        "player_id",
        "season",
        "league_id",
        "league_key",
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
        "league_key",
        "season",
        "competition",
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
        competition_hint = (
            " Then run basketball_ai/data/ai_source_competition.sql."
            if name in {"player_competition_stats", "team_competition_stats"}
            else ""
        )
        raise RuntimeError(
            f'Cannot read canonical view "{schema}"."{name}". '
            "Run basketball_ai/data/ai_source_schema.sql against BBallstat PostgreSQL."
            + competition_hint
            + " Verify the AI read role has SELECT access."
        ) from exc


def _load_optional_view(engine: Engine, schema: str, name: str) -> pd.DataFrame:
    """Read an additive simulation view, returning an empty frame if absent."""
    try:
        return pd.read_sql(text(f'SELECT * FROM "{schema}"."{name}"'), engine)
    except Exception:
        return pd.DataFrame()


def _load_views(
    engine: Engine,
    schema: str,
    views: dict[str, str],
    *,
    optional: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load independent canonical views concurrently with bounded workers."""
    if not views:
        return {}
    try:
        configured = int(os.getenv("POSTGRES_LOAD_WORKERS", "4"))
    except ValueError:
        configured = 4
    workers = max(1, min(configured, len(views)))
    loader = _load_optional_view if optional else _load_view
    if workers == 1:
        return {key: loader(engine, schema, view) for key, view in views.items()}

    loaded: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ai-pg-load") as pool:
        futures = {
            pool.submit(loader, engine, schema, view): key
            for key, view in views.items()
        }
        for future in as_completed(futures):
            loaded[futures[future]] = future.result()
    return {key: loaded[key] for key in views}


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
    player_key = {"player_id", "league_id", "season", "competition"}
    if player_stats is not None and player_key.issubset(player_stats.columns):
        duplicates = player_stats.duplicated(
            ["player_id", "league_id", "season", "competition"], keep=False
        )
        if duplicates.any():
            errors.append(
                "player_competition_stats must contain exactly one row per "
                "player_id+league_id+season+competition"
            )

    team_history = data.get("team_season_stats")
    team_key = {"team_id", "league_id", "season", "competition"}
    if team_history is not None and team_key.issubset(team_history.columns):
        duplicates = team_history.duplicated(
            ["team_id", "league_id", "season", "competition"], keep=False
        )
        if duplicates.any():
            errors.append(
                "team_competition_stats must contain exactly one row per "
                "team_id+league_id+season+competition"
            )

    for table in ("player_stats", "team_season_stats"):
        frame = data.get(table)
        if frame is None or "competition" not in frame.columns:
            continue
        invalid = frame["competition"].isna() | frame["competition"].astype(str).str.strip().eq("")
        if invalid.any():
            errors.append(f"{table}.competition contains null/empty values")

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

    for table in ("player_stats", "team_season_stats"):
        if "competition" in data[table].columns:
            data[table]["competition"] = (
                data[table]["competition"].fillna("RS").astype(str).str.strip().str.upper()
            )

    optional_id_columns = {
        "pbp_events": ("player_id", "offensive_player_id", "defender_id", "team_id", "opponent_team_id"),
        "lineup_stints": ("team_id", "opponent_team_id", "offense_player_id", "defense_player_id"),
        "play_type_stats": ("player_id", "team_id", "opponent_team_id"),
        "shot_profiles": ("player_id", "team_id"),
    }
    for table, columns in optional_id_columns.items():
        frame = data.get(table)
        if frame is None or frame.empty:
            continue
        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].apply(
                    lambda value: None if pd.isna(value) else _to_int(value)
                )
        if "competition" in frame.columns:
            frame["competition"] = frame["competition"].fillna("RS").astype(str).str.strip().str.upper()
        if "season" in frame.columns:
            frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")


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

    team_context_dict: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for _, row in data["team_season_stats"].iterrows():
        if (
            pd.isna(row.get("team_id"))
            or pd.isna(row.get("league_id"))
            or pd.isna(row.get("season"))
        ):
            continue
        key = (
            _to_int(row["team_id"]),
            _to_int(row["league_id"]),
            int(row["season"]),
            str(row.get("competition", "RS") or "RS").upper(),
        )
        team_context_dict[key] = row.to_dict()
    data["team_season_dict"] = team_context_dict

    league_teams: dict[int, list[int]] = {}
    for team_id, team in data["team_dict"].items():
        league_id = team.get("league_id")
        if league_id is None or pd.isna(league_id):
            continue
        league_teams.setdefault(_to_int(league_id), []).append(team_id)
    data["league_teams"] = league_teams


def _env_optional_default() -> bool:
    value = os.getenv("POSTGRES_INCLUDE_OPTIONAL", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def load_all_data(
    url: str | None = None,
    source_schema: str | None = None,
    *,
    include_optional: bool | None = None,
) -> dict[str, Any]:
    """Load canonical PostgreSQL data.

    The admin/training container can set ``POSTGRES_INCLUDE_OPTIONAL=false`` to
    skip possession-level simulation feeds. The public API keeps the default
    full loader unless explicitly configured otherwise.
    """
    schema = source_schema or os.getenv("POSTGRES_SOURCE_SCHEMA", "ai_source")
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError("Invalid PostgreSQL source schema")
    if include_optional is None:
        include_optional = _env_optional_default()

    engine = get_engine(url)
    try:
        data: dict[str, Any] = _load_views(
            engine, schema, _SOURCE_VIEWS, optional=False
        )
        if include_optional:
            data.update(
                _load_views(engine, schema, _OPTIONAL_SOURCE_VIEWS, optional=True)
            )
        else:
            data.update({key: pd.DataFrame() for key in _OPTIONAL_SOURCE_VIEWS})
    finally:
        engine.dispose()

    _validate_contract(data, schema)
    _normalise_ids(data)

    _derive_playing_style(data["teams"])
    _compute_star_player_usage(data["teams"], data["player_stats"])
    _fill_current_team_league(data["players"], data["player_stats"])

    data["player_stats"] = (
        data["player_stats"]
        .dropna(subset=["player_id", "league_id", "season", "rating", "competition"])
        .sort_values(["player_id", "league_id", "competition", "season"])
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
        .dropna(subset=["team_id", "league_id", "season", "competition"])
        .sort_values(["team_id", "league_id", "competition", "season"])
        .reset_index(drop=True)
    )

    _build_lookups(data)
    available = sorted(key for key in _OPTIONAL_SOURCE_VIEWS if not data[key].empty)
    data["simulation_feeds"] = available
    data["source_contract"] = (
        "competition-v1+simulation-v1" if include_optional else "competition-v1"
    )
    return data

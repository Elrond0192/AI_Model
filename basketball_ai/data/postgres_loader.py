"""Canonical PostgreSQL data access for AI_Model.

Production reads entity/roster tables from ``"AI_Source"`` and the competition-
preserving statistical contract created by ``ai_source_full.sql``.
Training therefore sees one observation per entity + league + season +
competition instead of collapsing PO/CUP/TOT into a single season row.
"""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "leagues": "Leagues",
    "teams": "Teams",
    "players": "Players",
    "player_stats": "PlayerCompetitionStats",
    "team_player_relations": "TeamPlayerRelations",
    "team_season_stats": "TeamCompetitionStats",
}


def _serving_tables_ready(engine: Engine, schema: str) -> bool:
    """Return true only when both materialized training stores have data."""
    try:
        query = text(
            f'SELECT EXISTS (SELECT 1 FROM "{schema}"."TrainingPlayerCompetitionStats" LIMIT 1) '
            f'AND EXISTS (SELECT 1 FROM "{schema}"."TrainingTeamCompetitionStats" LIMIT 1)'
        )
        with engine.connect() as connection:
            return bool(connection.execute(query).scalar())
    except Exception:
        return False


def _core_source_views(
    engine: Engine | None = None,
    schema: str = "AI_Source",
    *,
    prefer_serving: bool = False,
) -> dict[str, str]:
    views = dict(_SOURCE_VIEWS)
    mode = os.getenv("POSTGRES_TRAINING_SOURCE", "auto").strip().lower()
    if mode not in {"auto", "canonical", "serving"}:
        raise ValueError(
            "POSTGRES_TRAINING_SOURCE must be auto, canonical or serving"
        )
    use_serving = mode == "serving" or (
        mode == "auto"
        and prefer_serving
        and engine is not None
        and _serving_tables_ready(engine, schema)
    )
    if use_serving:
        views["player_stats"] = "TrainingPlayerCompetitionStats"
        views["team_season_stats"] = "TrainingTeamCompetitionStats"
    return views

_OPTIONAL_SOURCE_VIEWS = {
    "pbp_events": "ScenarioDefenderMatchups",
    "lineup_stints": "ScenarioLineupStats",
    "play_type_stats": "ScenarioPlayTypeStats",
    "shot_profiles": "ScenarioShotProfiles",
    "causal_panel": "SimulationCausalPanel",
}

_OPTIONAL_COLUMNS = {
    "pbp_events": ("league_key", "season", "competition", "offensive_player_id", "team_id", "opponent_team_id", "defender_id", "assignment_probability", "possessions", "points", "turnovers_forced"),
    "lineup_stints": ("league_key", "season", "competition", "player_ids", "team_id", "opponent_team_id", "offense_player_id", "defense_player_id", "assignment_probability", "possessions", "points", "points_allowed", "turnovers_forced", "ortg", "drtg", "net_rtg"),
    "play_type_stats": ("player_id", "team_id", "opponent_team_id", "league_key", "season", "competition", "play_type", "possessions", "ppp", "ppp_allowed"),
    "shot_profiles": ("player_id", "team_id", "league_key", "season", "competition", "zone", "attempts", "fg_pct"),
    "causal_panel": ("treatment", "next_outcome"),
}

_SCENARIO_FEEDS = {
    "opponent_matchup": ("play_type_stats",),
    "defensive_matchup": ("pbp_events", "lineup_stints"),
    "play_type_matchup": ("play_type_stats",),
    "shot_profile_counterfactual": ("shot_profiles",),
    "lineup_synergy": ("lineup_stints",),
    "lineup_optimizer": ("lineup_stints",),
    "roster_optimizer": ("lineup_stints",),
    "composite_scenario": ("pbp_events", "lineup_stints", "play_type_stats", "shot_profiles"),
    "causal_effect": ("causal_panel",),
}

_scenario_cache: dict[tuple[Any, ...], tuple[float, dict[str, pd.DataFrame]]] = {}
_scenario_cache_lock = threading.Lock()

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
        from basketball_ai.data.connection_profiles import (
            active_profile_name,
            profile_url,
        )

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


def _source_object_diagnostics(engine: Engine, schema: str, names: list[str]) -> dict[str, Any]:
    """Inspect canonical object existence and SELECT access for the current DB user."""
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError("Invalid PostgreSQL source schema")
    try:
        with engine.connect() as connection:
            current_user = str(connection.execute(text("SELECT current_user")).scalar() or "")
            schema_exists = bool(
                connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace "
                        "WHERE nspname = :schema)"
                    ),
                    {"schema": schema},
                ).scalar()
            )
            objects: dict[str, dict[str, bool]] = {}
            for name in names:
                if not _SCHEMA_RE.fullmatch(name):
                    continue
                qualified = f'"{schema}"."{name}"'
                exists = bool(
                    connection.execute(
                        text("SELECT to_regclass(:qualified) IS NOT NULL"),
                        {"qualified": qualified},
                    ).scalar()
                )
                select_allowed = bool(
                    connection.execute(
                        text(
                            "SELECT has_table_privilege(current_user, :qualified, 'SELECT')"
                        ),
                        {"qualified": qualified},
                    ).scalar()
                ) if exists else False
                objects[name] = {"exists": exists, "select": select_allowed}
            return {
                "current_user": current_user,
                "schema_exists": schema_exists,
                "objects": objects,
            }
    except Exception as exc:
        return {"error": str(exc)[:500]}


def _load_view(engine: Engine, schema: str, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(text(f'SELECT * FROM "{schema}"."{name}"'), engine)
    except Exception as exc:
        diagnostics = _source_object_diagnostics(engine, schema, [name])
        current_user = diagnostics.get("current_user", "unknown")
        object_info = diagnostics.get("objects", {}).get(name, {})
        if not diagnostics.get("schema_exists"):
            detail = f'Schema "{schema}" non esistente.'
        elif not object_info.get("exists"):
            detail = f'oggetto "{schema}"."{name}" non esistente.'
        elif not object_info.get("select"):
            detail = (
                f'utente PostgreSQL "{current_user}" non ha SELECT su '
                f'"{schema}"."{name}".'
            )
        else:
            detail = (
                f'lettura fallita: {str(exc).strip() or exc.__class__.__name__}'
            )
        hint = (
            ' Esegui ai_source_full.sql come owner e assicurati che il ruolo runtime '
            'abbia SELECT sulle sei tabelle canonical.'
            if name in {
                "Leagues",
                "Teams",
                "Players",
                "TeamPlayerRelations",
                "PlayerCompetitionStats",
                "TeamCompetitionStats",
            }
            else ""
        )
        raise RuntimeError(
            f'Impossibile leggere "{schema}"."{name}": {detail}.{hint}'
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
    """Load independent canonical objects concurrently with bounded workers."""
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
        raise RuntimeError('Invalid "AI_Source" contract: ' + " | ".join(errors))
    for required_non_empty in ("players", "teams", "player_stats", "team_season_stats"):
        if data[required_non_empty].empty:
            raise RuntimeError(f"{schema}.{required_non_empty} is empty")


def _nullable_int_series(values: pd.Series) -> pd.Series:
    """Normalize identifiers to pandas nullable Int64 without float coercion."""
    normalized = [
        pd.NA if pd.isna(value) else _to_int(value)
        for value in values.tolist()
    ]
    return pd.Series(
        pd.array(normalized, dtype="Int64"),
        index=values.index,
        name=values.name,
    )


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
        frame = data.get(table)
        if frame is None:
            continue
        for column in columns:
            if column not in frame.columns:
                continue
            frame[column] = _nullable_int_series(frame[column])

    for table in ("player_stats", "team_player_relations", "team_season_stats"):
        if table in data and "season" in data[table].columns:
            data[table]["season"] = pd.to_numeric(
                data[table]["season"], errors="coerce"
            ).astype("Int64")

    for table in ("player_stats", "team_season_stats"):
        if table in data and "competition" in data[table].columns:
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
                frame[column] = _nullable_int_series(frame[column])
        if "competition" in frame.columns:
            frame["competition"] = frame["competition"].fillna("RS").astype(str).str.strip().str.upper()
        if "season" in frame.columns:
            frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")


def _attach_player_global_ids(data: dict[str, pd.DataFrame]) -> None:
    """Propagate canonical player global_id onto every PlayerCompetitionStats row."""
    players = data.get("players")
    stats = data.get("player_stats")
    if not isinstance(players, pd.DataFrame) or not isinstance(stats, pd.DataFrame):
        return
    if players.empty or stats.empty or "id" not in players.columns or "global_id" not in players.columns:
        return
    if "player_id" not in stats.columns:
        return

    mapping = players[["id", "global_id"]].copy()
    mapping["id"] = mapping["id"].map(_to_int)
    mapping["global_id"] = mapping["global_id"].astype(str).str.strip()
    mapping = mapping[(mapping["id"].notna()) & (mapping["global_id"] != "")]
    if mapping.empty:
        return

    # One canonical mapping per internal Players.id. Duplicate source rows with
    # conflicting identities are ignored rather than assigning an arbitrary one.
    counts = mapping.groupby("id")["global_id"].nunique()
    stable_ids = set(counts[counts == 1].index.tolist())
    mapping = mapping[mapping["id"].isin(stable_ids)].drop_duplicates("id", keep="last")

    stats_ids = stats["player_id"].map(_to_int)
    mapped = stats_ids.map(mapping.set_index("id")["global_id"])
    if "global_id" in stats.columns:
        existing = stats["global_id"].astype(str).str.strip()
        stats["global_id"] = existing.where(existing != "", mapped)
    else:
        stats["global_id"] = mapped


def _build_lookups(data: dict[str, pd.DataFrame]) -> None:
    def indexed(frame: pd.DataFrame, id_column: str = "id") -> dict[int, dict[str, Any]]:
        if id_column not in frame.columns:
            return {}
        out: dict[int, dict[str, Any]] = {}
        for row in frame.to_dict("records"):
            if pd.isna(row[id_column]):
                continue
            out[_to_int(row[id_column])] = row
        return out

    data["league_dict"] = indexed(data["leagues"])
    data["team_dict"] = indexed(data["teams"])
    data["player_dict"] = indexed(data["players"])

    team_context_dict: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in data["team_season_stats"].to_dict("records"):
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
        team_context_dict[key] = row
    data["team_season_dict"] = team_context_dict

    league_teams: dict[int, list[int]] = {}
    for team_id, team in data["team_dict"].items():
        league_id = team.get("league_id")
        if league_id is None or pd.isna(league_id):
            continue
        league_teams.setdefault(_to_int(league_id), []).append(team_id)
    data["league_teams"] = league_teams


def _env_optional_default() -> bool:
    value = os.getenv("POSTGRES_INCLUDE_OPTIONAL", "false").strip().lower()
    return value not in {"0", "false", "no", "off"}


def load_scenario_feeds(
    scenario: str,
    *,
    player_ids: list[int],
    team_ids: list[int],
    league_keys: list[str],
    season: int,
    competition: str,
    url: str | None = None,
    source_schema: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load only the possession evidence required by one scenario request.

    Queries are bounded by season/competition/league and, where the canonical
    view supports them, by the requested players or teams. Results are cached
    briefly; the API never materialises all simulation views at startup.
    """
    schema = source_schema or os.getenv("POSTGRES_SOURCE_SCHEMA", "AI_Source")
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError("Invalid PostgreSQL source schema")
    feeds = _SCENARIO_FEEDS.get(str(scenario), ())
    empty = {key: pd.DataFrame() for key in _OPTIONAL_SOURCE_VIEWS}
    if not feeds:
        return empty

    players = tuple(sorted({_to_int(value) for value in player_ids}))
    teams = tuple(sorted({_to_int(value) for value in team_ids}))
    leagues = tuple(sorted({str(value) for value in league_keys if value}))
    cache_key = (str(scenario), players, teams, leagues, int(season), str(competition).upper())
    now = time.monotonic()
    ttl = max(0, int(os.getenv("SCENARIO_FEED_CACHE_TTL_SECONDS", "300")))
    with _scenario_cache_lock:
        cached = _scenario_cache.get(cache_key)
        if cached and now - cached[0] <= ttl:
            return {**empty, **{key: frame.copy() for key, frame in cached[1].items()}}

    def read(feed: str) -> pd.DataFrame:
        # The governed causal panel currently has no context keys. Do not ever
        # issue an unbounded read; the engine safely falls back to core history.
        if feed == "causal_panel":
            return pd.DataFrame(columns=_OPTIONAL_COLUMNS[feed])
        params: dict[str, Any] = {
            "season_min": int(season) - 3,
            "season_max": int(season),
            "competition": str(competition).upper(),
        }
        predicates = [
            "season BETWEEN :season_min AND :season_max",
            "upper(competition) = :competition",
        ]
        if leagues:
            names = []
            for index, value in enumerate(leagues):
                name = f"league_{index}"
                params[name] = value
                names.append(f":{name}")
            predicates.append(f"league_key IN ({', '.join(names)})")

        entity_terms: list[str] = []
        player_columns = {
            "pbp_events": ("offensive_player_id", "defender_id"),
            "lineup_stints": ("offense_player_id", "defense_player_id"),
            "play_type_stats": ("player_id",),
            "shot_profiles": ("player_id",),
        }.get(feed, ())
        team_columns = {
            "pbp_events": ("team_id", "opponent_team_id"),
            "lineup_stints": ("team_id", "opponent_team_id"),
            "play_type_stats": ("team_id", "opponent_team_id"),
            "shot_profiles": ("team_id",),
        }.get(feed, ())
        for index, value in enumerate(players):
            name = f"player_{index}"
            params[name] = value
            entity_terms.extend(f"{column} = :{name}" for column in player_columns)
            if feed == "lineup_stints":
                params[f"player_token_{index}"] = f"%,{value},%"
                entity_terms.append(
                    f"(',' || player_ids || ',') LIKE :player_token_{index}"
                )
        for index, value in enumerate(teams):
            name = f"team_{index}"
            params[name] = value
            entity_terms.extend(f"{column} = :{name}" for column in team_columns)
        if entity_terms:
            predicates.append("(" + " OR ".join(entity_terms) + ")")

        columns = ", ".join(f'"{column}"' for column in _OPTIONAL_COLUMNS[feed])
        view = _OPTIONAL_SOURCE_VIEWS[feed]
        statement = text(
            f'SELECT {columns} FROM "{schema}"."{view}" WHERE '
            + " AND ".join(predicates)
        )
        return pd.read_sql(statement, engine, params=params)

    engine = get_engine(url)
    try:
        loaded = {feed: read(feed) for feed in feeds}
    finally:
        engine.dispose()
    scoped = {**empty, **loaded}
    _normalise_ids(scoped)
    with _scenario_cache_lock:
        _scenario_cache[cache_key] = (now, {key: frame.copy() for key, frame in loaded.items()})
    return scoped


def load_all_data(
    url: str | None = None,
    source_schema: str | None = None,
    *,
    include_optional: bool | None = None,
    purpose: str = "analysis",
) -> dict[str, Any]:
    """Load canonical PostgreSQL data.

    Possession-level simulation feeds are excluded by default. They are loaded
    per scenario through :func:`load_scenario_feeds`, never at API bootstrap.
    """
    schema = source_schema or os.getenv("POSTGRES_SOURCE_SCHEMA", "AI_Source")
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError("Invalid PostgreSQL source schema")
    if include_optional is None:
        include_optional = _env_optional_default()
    if purpose not in {"analysis", "training"}:
        raise ValueError("purpose must be analysis or training")

    engine = get_engine(url)
    try:
        core_views = _core_source_views(
            engine,
            schema,
            prefer_serving=purpose == "training",
        )
        data: dict[str, Any] = _load_views(engine, schema, core_views, optional=False)
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
    _attach_player_global_ids(data)

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
    source_mode = (
        "serving"
        if core_views["player_stats"] == "TrainingPlayerCompetitionStats"
        else "canonical"
    )
    base_contract = f"competition-v2:{source_mode}"
    data["source_contract"] = (
        base_contract + "+simulation-v1" if include_optional else base_contract
    )
    return data

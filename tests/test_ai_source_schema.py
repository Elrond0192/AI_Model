"""Static/unit tests for the production PostgreSQL contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SQL = ROOT / "basketball_ai" / "data" / "ai_source_full.sql"
SERVING_SQL = ROOT / "basketball_ai" / "data" / "ai_scenario_serving.sql"
OUTPUT_SQL = ROOT / "basketball_ai" / "data" / "ai_schema.sql"


def test_ai_source_schema_defines_minimal_canonical_contract():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    for table in (
        '"ai_source"."leagues"',
        '"ai_source"."teams"',
        '"ai_source"."players"',
        '"ai_source"."teamplayerrelations"',
        '"ai_source"."playercompetitionstats"',
        '"ai_source"."teamcompetitionstats"',
    ):
        assert (
            f"create table {table}" in sql
            or f"create table if not exists {table}" in sql
        )

    for forbidden in (
        '"ai_source"."playerregistryinternal"',
        '"ai_source"."teamregistryinternal"',
        '"ai_source"."statsrawinternal"',
        '"ai_source"."rolesrawinternal"',
        '"ai_source"."onoffrawinternal"',
        '"ai_source"."clutchrawinternal"',
        '"ai_source"."boxscorerawinternal"',
        '"ai_source"."simulationpbpevents"',
        '"ai_source"."simulationlineupstints"',
    ):
        assert forbidden not in sql

    assert "create view " not in sql
    assert "create materialized view " not in sql
    assert "to_jsonb(" not in sql
    assert "lowerkeys" not in sql
    assert "numericvalue(" not in sql
    assert "textvalue(" not in sql
    assert "information_schema." not in sql
    assert "from pg_catalog.pg_class" in sql
    assert "create table " in sql


def test_competition_schema_preserves_player_and_team_contexts():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert 'create table "ai_source"."playercompetitionstats"' in sql
    assert 'create table "ai_source"."teamcompetitionstats"' in sql
    assert "percent_rank() over" in sql
    assert "advancedstats_player_" in sql
    assert "advancedstatsteam_" in sql
    assert "games_started" in sql


def test_competition_schema_keeps_tot_separate_from_rs():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "when 'total' then 'tot'" in sql
    assert "when 'all' then 'tot'" in sql
    assert "when 'tot' then 'tot'" in sql
    assert "when 'playoffs' then 'po'" in sql
    assert "when 'regular season' then 'rs'" in sql


def test_ai_source_does_not_own_simulation_feeds():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    for token in (
        "simulationpbpevents",
        "simulationlineupstints",
        "simulationplaytypestats",
        "simulationshotprofiles",
        "simulationcausalpanel",
        "advancedstats_lineups_quarter_",
    ):
        assert token not in sql


def test_ai_source_resolves_player_team_from_roster_team_id_when_stats_team_id_is_null():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "p_team_local_expr text" in sql
    assert "array['team', 'teamid', 'idteam']" in sql
    assert "team_source_expr := format('coalesce(%s, %s)', team_local_expr, p_team_local_expr)" in sql
    assert "pg_temp.ai_id_key(team_source_expr)" in sql
    assert "array['teamname', 'team', 'name']" in sql


def test_ai_source_deduplicates_team_player_relations_by_canonical_key():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "select distinct on (player_id, team_id, season)" in sql
    assert "relation_priority" in sql
    assert "order by" in sql


def test_ai_source_uses_native_typed_columns_for_row_transforms():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "pg_temp.ai_expr" in sql
    assert "pg_catalog.pg_attribute" in sql
    assert 'create table "ai_source"."playercompetitionstats" as' in sql
    assert "union all" in sql
    assert "hashtextextended" in sql


def test_ai_source_schemas_have_no_sql_server_ddl():
    sql = (
        SOURCE_SQL.read_text(encoding="utf-8")
        + SERVING_SQL.read_text(encoding="utf-8")
    ).lower()
    forbidden = (
        "set ansi_nulls", "set quoted_identifier", "nvarchar(", "datetime2",
        "getutcdate(", "sys.schemas", "pyodbc", "mssql+",
    )
    for token in forbidden:
        assert token not in sql


def test_scenario_serving_is_physical_indexed_and_incremental():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    for table in (
        "trainingplayercompetitionstats",
        "trainingteamcompetitionstats",
        "scenarioplaytypestats",
        "scenarioshotprofiles",
        "scenariodefendermatchups",
        "scenariolineupstats",
    ):
        assert f'create table if not exists "ai_source"."{table}"' in sql
    assert 'create or replace procedure "ai_source"."refreshscenarioserving"' in sql
    assert 'create or replace procedure "ai_source"."refreshtrainingserving"' in sql
    assert 'create or replace procedure "ai_source"."refreshcontext"' in sql
    assert 'create table if not exists "ai_source"."servingrefreshstate"' in sql
    assert "p_league_key text" in sql
    assert "p_season integer" in sql
    assert "p_competition text" in sql
    assert "create index if not exists" in sql


def test_ai_owned_schemas_and_relations_use_pascal_case():
    source = SOURCE_SQL.read_text(encoding="utf-8")
    serving = SERVING_SQL.read_text(encoding="utf-8")
    output = OUTPUT_SQL.read_text(encoding="utf-8")
    assert 'CREATE SCHEMA IF NOT EXISTS "AI_Source"' in source
    assert 'CREATE SCHEMA IF NOT EXISTS "AI"' in output
    assert '"AI_Source"."TrainingPlayerCompetitionStats"' in serving
    assert '"AI_Source"."ServingRefreshState"' in serving
    assert '"AI"."ModelRuns"' in output
    assert '"AI"."PlayerForecasts"' in output
    assert "CREATE SCHEMA IF NOT EXISTS ai_source" not in source
    assert "CREATE TABLE IF NOT EXISTS ai." not in output
    assert 'CREATE SCHEMA IF NOT EXISTS "AI_Source"' in source


def test_stable_id_keeps_null_relations_null():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "pg_temp.ai_stable_id" in sql
    assert "or btrim(%1$s::text) = ''" in sql
    assert "hashtextextended(%2$L || ':' || btrim(%1$s::text), 0)" in sql


def _valid_frames():
    return {
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1", "league_key": "ITA1"}]),
        "teams": pd.DataFrame(
            [{"id": 10, "global_id": "T10", "name": "Team", "league_id": 1}]
        ),
        "players": pd.DataFrame(
            [{"id": 20, "global_id": "P20", "name": "Player", "position": "PG"}]
        ),
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 20, "season": 2025, "league_id": 1,
                    "league_key": "ITA1", "games_played": 20,
                    "minutes_per_game": 25.0, "points": 10.0,
                    "rating": 6.5, "competition": "RS",
                }
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [{"player_id": 20, "team_id": 10, "season": 2025}]
        ),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 10, "global_id": "T10", "name": "Team",
                    "league_id": 1, "league_key": "ITA1", "season": 2025,
                    "competition": "RS", "pace": 75.0,
                    "offensive_rating": 112.0, "defensive_rating": 108.0,
                    "three_point_attempt_rate": 0.35, "assists_per_game": 20.0,
                }
            ]
        ),
    }


def test_postgres_loader_contract_accepts_competition_frames():
    from basketball_ai.data.postgres_loader import _validate_contract
    _validate_contract(_valid_frames(), "ai_source")


def test_postgres_loader_contract_requires_global_ids():
    from basketball_ai.data.postgres_loader import _validate_contract
    data = _valid_frames()
    data["players"] = data["players"].drop(columns=["global_id"])
    with pytest.raises(RuntimeError, match="global_id"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_requires_team_competition_history():
    from basketball_ai.data.postgres_loader import _validate_contract
    data = _valid_frames()
    data.pop("team_season_stats")
    with pytest.raises(RuntimeError, match="team_season_stats"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_allows_same_player_season_in_different_competitions():
    from basketball_ai.data.postgres_loader import _validate_contract
    data = _valid_frames()
    po = data["player_stats"].copy()
    po["competition"] = "PO"
    data["player_stats"] = pd.concat([data["player_stats"], po], ignore_index=True)
    team_po = data["team_season_stats"].copy()
    team_po["competition"] = "PO"
    data["team_season_stats"] = pd.concat([data["team_season_stats"], team_po], ignore_index=True)
    _validate_contract(data, "ai_source")


def test_postgres_loader_rejects_duplicate_player_competition_rows():
    from basketball_ai.data.postgres_loader import _validate_contract
    data = _valid_frames()
    data["player_stats"] = pd.concat([data["player_stats"], data["player_stats"]], ignore_index=True)
    with pytest.raises(RuntimeError, match=r"player_id\+league_id\+season\+competition"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_rejects_duplicate_team_competition_rows():
    from basketball_ai.data.postgres_loader import _validate_contract
    data = _valid_frames()
    data["team_season_stats"] = pd.concat([data["team_season_stats"], data["team_season_stats"]], ignore_index=True)
    with pytest.raises(RuntimeError, match=r"team_id\+league_id\+season\+competition"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_builds_competition_lookups():
    from basketball_ai.data.postgres_loader import _build_lookups
    data = _valid_frames()
    _build_lookups(data)
    assert data["team_dict"][10]["name"] == "Team"
    assert data["player_dict"][20]["name"] == "Player"
    assert data["league_teams"] == {1: [10]}
    assert data["team_season_dict"][(10, 1, 2025, "RS")]["pace"] == 75.0

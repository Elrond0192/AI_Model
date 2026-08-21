"""Static/unit tests for the production PostgreSQL contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SQL = ROOT / "basketball_ai" / "data" / "ai_source_schema.sql"
COMPETITION_SQL = ROOT / "basketball_ai" / "data" / "ai_source_competition.sql"
SIMULATION_SQL = ROOT / "basketball_ai" / "data" / "ai_source_simulation.sql"
SERVING_SQL = ROOT / "basketball_ai" / "data" / "ai_scenario_serving.sql"
OUTPUT_SQL = ROOT / "basketball_ai" / "data" / "ai_schema.sql"
MIGRATION_SQL = ROOT / "basketball_ai" / "data" / "ai_pascalcase_migration.sql"


def test_ai_source_schema_defines_entity_contract():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    for view in (
        '"ai_source"."leagues"',
        '"ai_source"."teams"',
        '"ai_source"."players"',
        '"ai_source"."teamplayerrelations"',
    ):
        assert f"create view {view}" in sql


def test_competition_schema_preserves_player_and_team_contexts():
    sql = COMPETITION_SQL.read_text(encoding="utf-8").lower()
    assert 'create view "ai_source"."playercompetitionstats"' in sql
    assert 'create view "ai_source"."teamcompetitionstats"' in sql
    assert "partition by pr.global_id, s.league_key, s.season, s.competition" in sql
    assert "partition by tr.global_id, ts.league_key, ts.season, ts.competition" in sql
    assert '"ai_source"."competitioncode"' in sql
    assert "advancedstats_player_" in sql
    assert "advancedstatsteam_" in sql
    assert "information_schema.columns" in sql
    assert "'gamesstarted', 'games_started'" in sql
    assert "coalesce(s.games_started, bs.games_started, 0)" in sql


def test_competition_schema_keeps_tot_separate_from_rs():
    sql = COMPETITION_SQL.read_text(encoding="utf-8").lower()
    assert "when 'tot' then 'tot'" in sql
    assert "when 'playoffs' then 'po'" in sql
    assert "when 'regular season' then 'rs'" in sql
    assert "when upper(s.competition) = 'tot' then 'rs'" not in sql


def test_competition_schema_is_independent_of_base_internal_views():
    sql = COMPETITION_SQL.read_text(encoding="utf-8").lower()
    assert "from ai_source._team_registry" not in sql
    assert "from ai_source._stats_raw" not in sql
    assert "from ai_source._team_stats_raw" not in sql


def test_simulation_schema_adapts_pbp_lineups_play_types_and_shots():
    sql = SIMULATION_SQL.read_text(encoding="utf-8").lower()
    for view in (
        '"ai_source"."simulationpbpevents"',
        '"ai_source"."simulationlineupstints"',
        '"ai_source"."simulationplaytypestats"',
        '"ai_source"."simulationshotprofiles"',
        '"ai_source"."simulationcausalpanel"',
    ):
        assert f"create view {view}" in sql
    assert "advancedstats_lineups_quarter_" in sql
    assert "inferred_not_observed" not in sql  # inference labels belong to API output
    assert "assignment_probability" in sql


def test_ai_source_schemas_have_no_sql_server_ddl():
    sql = (
        SOURCE_SQL.read_text(encoding="utf-8")
        + COMPETITION_SQL.read_text(encoding="utf-8")
        + SIMULATION_SQL.read_text(encoding="utf-8")
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
    migration = MIGRATION_SQL.read_text(encoding="utf-8")
    assert "DROP SCHEMA IF EXISTS ai_source CASCADE" in migration


def test_stable_id_keeps_null_relations_null():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "when value is null or btrim(value) = '' then null" in sql


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

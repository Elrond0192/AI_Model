"""Static/unit tests for the production PostgreSQL contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SQL = ROOT / "basketball_ai" / "data" / "ai_source_schema.sql"
TEAM_SEASON_SQL = ROOT / "basketball_ai" / "data" / "ai_source_team_season.sql"


def test_ai_source_schema_defines_core_public_contract():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    for view in (
        "ai_source.leagues",
        "ai_source.teams",
        "ai_source.players",
        "ai_source.player_stats",
        "ai_source.team_player_relations",
    ):
        assert f"create view {view}" in sql


def test_team_season_schema_defines_independent_temporal_context():
    sql = TEAM_SEASON_SQL.read_text(encoding="utf-8").lower()
    assert "create view ai_source.team_season_stats" in sql
    assert "create view ai_source._team_season_registry" in sql
    assert "create view ai_source._team_season_stats_raw" in sql
    assert "partition by tr.global_id, tr.season" in sql
    assert "advancedstatsteam_" in sql
    assert "information_schema.columns" in sql
    # Do not depend on the base adapter's internal views: otherwise re-running
    # ai_source_schema.sql would be blocked by PostgreSQL view dependencies.
    assert "from ai_source._team_registry" not in sql
    assert "from ai_source._team_stats_raw" not in sql


def test_ai_source_schema_is_postgres_unified_table_adapter():
    sql = SOURCE_SQL.read_text(encoding="utf-8")
    lower = sql.lower()
    assert "AdvancedStats_Player_" in sql
    assert "Team_" in sql
    assert "Boxscore" in sql
    assert "information_schema.columns" in lower
    assert "hashtextextended" in lower
    assert "vallegapergame" in lower


def test_ai_source_schemas_have_no_sql_server_ddl():
    sql = (
        SOURCE_SQL.read_text(encoding="utf-8")
        + TEAM_SEASON_SQL.read_text(encoding="utf-8")
    ).lower()
    forbidden = (
        "set ansi_nulls",
        "set quoted_identifier",
        "nvarchar(",
        "datetime2",
        "getutcdate(",
        "sys.schemas",
        "pyodbc",
        "mssql+",
    )
    for token in forbidden:
        assert token not in sql


def test_stable_id_keeps_null_relations_null():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "when value is null or btrim(value) = '' then null" in sql


def test_ai_source_schema_collapses_global_player_season_duplicates():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "exactly one observation per global player + season" in sql
    assert "partition by pr.global_id, s.season" in sql
    assert "where s.rn = 1" in sql


def test_future_rosters_can_come_from_anagrafiche():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "pr.team_name is not null" in sql
    assert "lower(btrim(pr.team_name)) = lower(btrim(tr.name))" in sql


def _valid_frames():
    return {
        "leagues": pd.DataFrame([{"id": 1, "name": "ITA1"}]),
        "teams": pd.DataFrame(
            [{"id": 10, "global_id": "T10", "name": "Team", "league_id": 1}]
        ),
        "players": pd.DataFrame(
            [{"id": 20, "global_id": "P20", "name": "Player", "position": "PG"}]
        ),
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 20,
                    "season": 2025,
                    "league_id": 1,
                    "games_played": 20,
                    "minutes_per_game": 25.0,
                    "points": 10.0,
                    "rating": 6.5,
                    "competition": "RS",
                }
            ]
        ),
        "team_player_relations": pd.DataFrame(
            [{"player_id": 20, "team_id": 10, "season": 2025}]
        ),
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 10,
                    "global_id": "T10",
                    "name": "Team",
                    "league_id": 1,
                    "season": 2025,
                    "pace": 75.0,
                    "offensive_rating": 112.0,
                    "defensive_rating": 108.0,
                    "three_point_attempt_rate": 0.35,
                    "assists_per_game": 20.0,
                }
            ]
        ),
    }


def test_postgres_loader_contract_accepts_temporal_frames():
    from basketball_ai.data.postgres_loader import _validate_contract

    _validate_contract(_valid_frames(), "ai_source")


def test_postgres_loader_contract_requires_global_ids():
    from basketball_ai.data.postgres_loader import _validate_contract

    data = _valid_frames()
    data["players"] = data["players"].drop(columns=["global_id"])
    with pytest.raises(RuntimeError, match="global_id"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_requires_team_season_history():
    from basketball_ai.data.postgres_loader import _validate_contract

    data = _valid_frames()
    data.pop("team_season_stats")
    with pytest.raises(RuntimeError, match="team_season_stats"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_rejects_duplicate_player_seasons():
    from basketball_ai.data.postgres_loader import _validate_contract

    data = _valid_frames()
    data["player_stats"] = pd.concat(
        [data["player_stats"], data["player_stats"]], ignore_index=True
    )
    with pytest.raises(RuntimeError, match=r"one row per player_id\+season"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_rejects_duplicate_team_seasons():
    from basketball_ai.data.postgres_loader import _validate_contract

    data = _valid_frames()
    data["team_season_stats"] = pd.concat(
        [data["team_season_stats"], data["team_season_stats"]], ignore_index=True
    )
    with pytest.raises(RuntimeError, match=r"one row per team_id\+season"):
        _validate_contract(data, "ai_source")


def test_postgres_loader_builds_temporal_lookups():
    from basketball_ai.data.postgres_loader import _build_lookups

    data = _valid_frames()
    _build_lookups(data)
    assert data["team_dict"][10]["name"] == "Team"
    assert data["player_dict"][20]["name"] == "Player"
    assert data["league_teams"] == {1: [10]}
    assert data["team_season_dict"][(10, 2025)]["pace"] == 75.0

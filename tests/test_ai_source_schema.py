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


def test_ai_source_assigns_configured_league_tiers_and_reconciles_unique_team_identity():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "when 'el' then 1" in sql
    assert "when 'ec' then 1" in sql
    assert "when 'esp1' then 2" in sql
    assert "when 'lit1' then 2" in sql
    assert "when 'ita2' then 3" in sql
    assert "1::integer as tier" not in sql
    assert "1::integer as league_tier" not in sql
    assert "having count(distinct r.team_id) = 1" in sql
    assert 'update "ai_source"."playercompetitionstats" s' in sql
    assert "coalesce(r.direct_team_global_id, r.fallback_team_global_id) is null" in sql


def test_team_stats_support_gamesplayed_and_fallback_team_identity():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    compact = "".join(sql.split())
    team_section = sql.split("-- canonical teamcompetitionstats", 1)[1].split(
        "-- player registry staging", 1
    )[0]
    team_compact = "".join(team_section.split())
    assert "array['games','gamesplayed']" in compact
    # TeamCompetitionStats must read the per-team Games value emitted by AdvancedStats.
    assert "array['games','gamesplayed']" in team_compact
    assert "coalesce(games,0)::integerasgames_played" in team_compact
    assert "quote_literal(league_key||':')" in compact
    assert "||team_local_expr||')asteam_global_id" in compact
    # The roster registry and team-season statistics must use the same fallback key.
    assert "quote_literal(league_key||':')||" in compact


def test_ai_source_projects_all_native_advance_stats_clutch_fields():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "opt_table := 'advancedstats_clutch_' || league_key" in sql
    assert "opt_table := 'advancestats_clutch_' || league_key" in sql
    assert sql.index("opt_table := 'advancedstats_clutch_' || league_key") < sql.index(
        "opt_table := 'advancestats_clutch_' || league_key"
    )
    assert "from \"analisi\".%10$i cl0" in sql
    assert "clutch_games_expr := 'coalesce('" in sql
    assert "clutch_fg2m_expr" in sql
    for field in (
        "clutch_fg2m", "clutch_fg2a", "clutch_fg3m", "clutch_fg3a",
        "clutch_ftm", "clutch_fta", "clutch_ast", "clutch_orb",
        "clutch_drb", "clutch_stl", "clutch_blk", "clutch_tov",
        "clutch_fg_pct", "clutch_fg2_pct", "clutch_fg3_pct", "clutch_ft_pct",
        "clutch_efg_pct", "clutch_ts_pct", "clutch_ftr", "clutch_three_par",
        "clutch_ortg", "clutch_drtg", "clutch_net_rtg", "clutch_team_id",
        "clutch_clutchness_rating",
    ):
        assert f"as {field}" in sql


def test_causal_panel_has_longitudinal_and_audit_contract():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    for column in (
        "global_id text", "teammate_global_id text", "league_key text",
        "competition text", "baseline_bb_rating double precision",
        "minutes_pre double precision", "usage_pre double precision",
        "age_pre double precision", "team_strength_pre double precision",
        "transition_season integer", "role_pre text", "refreshed_at timestamptz",
    ):
        assert column in sql
    assert 'alter table "ai_source"."simulationcausalpanel"' in sql
    assert "ix_simulation_causal_panel_context" in sql
    assert "ix_simulation_causal_panel_teammate" in sql
    assert "does not itself make an observational comparison causal" in sql


def test_ai_source_does_not_own_simulation_feeds():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "c.relname like 'simulation%'" not in sql
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


def test_ai_source_onoff_matches_native_player_team_id():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "select distinct on (" in sql
    assert "source_team_id" in sql
    assert "array['teamid', 'idteam']" in sql
    assert "pg_temp.ai_id_key(oo_exact.source_team_id)" in sql
    assert "pg_temp.ai_id_key(%15$s)" in sql
    assert "                comp_expr,\n                team_local_expr\n            );" in sql
    assert "onoff_team_source_expr" not in sql
    assert "array['player','id','playerid','idplayer']" in "".join(sql.split())
    assert "array['gamesstarted', 'games_started', 'starts']" in sql
    assert "array['games', 'gamesplayed']" in sql
    assert "least(1.0, greatest(0.0, coalesce(%2$s, 0) / %1$s))" in sql
    assert "LEFT JOIN LATERAL" not in sql
    assert "oo_exact" in sql
    assert "oo_fallback" in sql
    # The current set-based fallback filters null-team rows before the alias is assigned.
    assert "and %3$s is null" in sql
    assert ") q" in sql
    assert "order by" in sql
    # AdvancedStatsOnOffCourt is authoritative for On/Off-specific metrics;
    # direct player-table columns are only fallback when the On/Off row is absent.
    assert "on_net_expr := '(coalesce(oo_exact.on_net_rtg, " in sql
    assert "off_net_expr := '(coalesce(oo_exact.off_net_rtg, " in sql
    assert "net_diff_expr := '(coalesce(oo_exact.net_rtg_diff, " in sql
    assert "ortg_diff_expr := format(" in sql
    assert "ortg_on_expr := '(coalesce(oo_exact.ortg_on, " in sql
    assert "ortg_off_expr := '(coalesce(oo_exact.ortg_off, " in sql
    assert "ortg_diff_expr := '(coalesce(oo_exact.ortg_diff, " in sql
    assert "scoring_eff_expr text" in sql
    # Optional JOIN fragments must be separated so "ON true" cannot become
    # the invalid identifier "trueleft" when another optional source follows.
    assert "|| chr(10)" in sql
    assert "|| role_join" in sql
    assert "|| onoff_join" in sql
    assert "|| clutch_join" in sql
    assert "|| boxscore_join" in sql
    assert "ai_role_join_end" in sql
    assert "ai_onoff_join_end" in sql
    assert "ai_clutch_join_end" in sql
    assert "ai_boxscore_join_end" in sql



def test_ai_source_deduplicates_team_player_relations_by_canonical_key():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "select distinct on (player_id, team_id, season)" in sql
    assert "relation_priority" in sql
    assert "order by" in sql


def test_ai_source_reapplies_hm_admin_access_on_rebuild():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "'hm_admin'" in sql
    assert "grant usage on schema \"ai_source\" to %i" in sql
    assert "grant select on table" in sql


def test_ai_source_grants_full_access_to_hm_local_workers_on_rebuild():
    sql = SOURCE_SQL.read_text(encoding="utf-8").lower()
    assert "where rolname = 'hm_local_workers'" in sql
    assert 'grant usage on schema "ai_source" to hm_local_workers' in sql
    assert 'grant all privileges on all tables in schema "ai_source" to hm_local_workers' in sql
    assert 'grant all privileges on all sequences in schema "ai_source" to hm_local_workers' in sql
    assert 'alter default privileges in schema "ai_source" grant all privileges on tables to hm_local_workers' in sql
    assert 'alter default privileges in schema "ai_source" grant all privileges on sequences to hm_local_workers' in sql


def test_scheduler_runtime_write_access_is_limited_to_scheduler_metadata():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    assert "do $grant_ai_model_scheduler$" in sql
    assert "grant select, insert, update, delete on table" in sql
    assert '"ai_source"."aimodelscheduledjob"' in sql
    assert '"ai_source"."aimodeljobrun"' in sql
    assert "grant usage, select on sequence" in sql
    assert "'ai_model'" in sql
    assert "requested_role" in sql
    # The runtime role receives no blanket DML grant across AI_Source.
    assert 'grant all privileges on all tables in schema "ai_source" to ai_model' not in sql


def test_causal_panel_runtime_grants_support_guarded_persisted_refresh():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    assert '"ai_source"."simulationcausalpanel"' in sql
    assert "grant select, insert, update, delete on table" in sql
    assert "'ai_model'" in sql
    assert "'hm_admin'" in sql
    assert "'hm_local_workers'" in sql


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


def test_scenario_serving_tolerates_missing_optional_simulation_feeds():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    for relation in (
        '"ai_source"."simulationplaytypestats"',
        '"ai_source"."simulationshotprofiles"',
        '"ai_source"."simulationpbpevents"',
        '"ai_source"."simulationlineupstints"',
    ):
        assert f"to_regclass({relation!r}) is not null" in sql
    assert "skipping play-type scenario feed" in sql
    assert "skipping shot-profile scenario feed" in sql
    assert "skipping defender matchup feed" in sql
    assert "skipping lineup scenario feed" in sql
    assert "get diagnostics affected_rows = row_count" in sql
    assert "scenariolineupstats remains empty" in sql
    assert "simulationlineupstints has no matching source rows" in sql


def test_scenario_serving_owns_and_rebuilds_simulation_sources():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()

    for table in (
        "simulationpbpevents",
        "simulationlineupstints",
        "simulationplaytypestats",
        "simulationshotprofiles",
        "simulationcausalpanel",
        "simulationsourcerefreshstate",
    ):
        assert f'create table if not exists "ai_source"."{table}"' in sql

    assert 'create or replace procedure "ai_source"."refreshsimulationsources"' in sql
    assert 'from "pbp".%1$i p' in sql
    assert 'where p."season"::integer = $1' in sql
    assert 'from "analisi".%1$i q' in sql
    assert 'call "ai_source"."refreshsimulationsources"(sim_ctx.league_key, sim_ctx.season)' in sql
    assert "unknown_two_point" in sql
    assert "unknown_three_point" in sql
    assert "simulationcausalpanel is refreshed separately by the explicit guarded model releases cohort-refresh action" in sql
    assert 'create index if not exists ix_simulation_pbp_context' in sql


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
    assert "do $sync_training_serving_columns$" in sql
    assert "format_type(source.atttypid, source.atttypmod)" in sql
    training_proc = sql.split('create or replace procedure "ai_source"."refreshtrainingserving"', 1)[1].split(
        'create or replace procedure "ai_source"."refreshcontextseasonal"', 1
    )[0]
    assert "v_player_columns text" in training_proc
    assert "v_team_columns text" in training_proc
    assert 'insert into "ai_source"."trainingplayercompetitionstats" (%1$s)' in training_proc
    assert 'insert into "ai_source"."trainingteamcompetitionstats" (%1$s)' in training_proc
    assert 'insert into "ai_source"."trainingplayercompetitionstats"\n    select *' not in training_proc
    assert 'insert into "ai_source"."trainingteamcompetitionstats"\n    select *' not in training_proc
    assert "'hm_local_workers'" in sql
    assert 'create or replace procedure "ai_source"."refreshcontextseasonal"(p_season integer)' in sql
    assert '"ai_source"."refreshcontextseasonal"(integer)' in sql
    assert 'call "ai_source"."refreshtrainingserving"(' in sql
    assert 'call "ai_source"."refreshscenarioserving"(' in sql
    assert 'from "ai_source"."playercompetitionstats"' in sql
    assert 'from "ai_source"."teamcompetitionstats"' in sql
    assert 'drop procedure if exists "ai_source"."refreshcontext"(text, integer, text)' in sql
    assert 'drop procedure if exists "ai_source"."refreshcontext"(integer)' in sql
    assert 'create or replace procedure "ai_source"."refreshcontext"(' not in sql
    assert 'call "ai_source"."refreshcontext"(' not in sql
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
    assert "hashtextextended(%2$l || ':' || btrim(%1$s::text), 0)" in sql


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


def test_seasonal_refresh_runtime_has_dml_only_on_derived_serving_tables():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()
    start = sql.index("do $grant_ai_model_refresh_writes$")
    end = sql.index("$grant_ai_model_refresh_writes$;", start)
    grants = sql[start:end]

    for table in (
        "trainingplayercompetitionstats",
        "trainingteamcompetitionstats",
        "scenarioplaytypestats",
        "scenarioshotprofiles",
        "scenariodefendermatchups",
        "scenariolineupstats",
        "simulationpbpevents",
        "simulationlineupstints",
        "simulationplaytypestats",
        "simulationshotprofiles",
        "simulationsourcerefreshstate",
        "servingrefreshstate",
    ):
        assert f'"ai_source"."{table}"' in grants

    assert "grant select, insert, update, delete on table" in grants
    assert "'ai_model'" in grants
    assert "'hm_admin'" in grants
    assert "'hm_local_workers'" in grants
    assert "requested_role" in grants
    for canonical in (
        '"ai_source"."players"',
        '"ai_source"."teams"',
        '"ai_source"."playercompetitionstats"',
        '"ai_source"."teamcompetitionstats"',
    ):
        assert canonical not in grants

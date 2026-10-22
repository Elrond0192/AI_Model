-- Scalable scenario-serving layer.
-- Run after ai_source_full.sql, then refresh a complete season:
--   CALL "AI_Source"."RefreshContextSeasonal"(2026);

BEGIN;

-- Physical training feature store. This prevents every snapshot build from
-- recomputing the canonical views over the complete boxscore history.
CREATE TABLE IF NOT EXISTS "AI_Source"."TrainingPlayerCompetitionStats"
    AS SELECT * FROM "AI_Source"."PlayerCompetitionStats" WITH NO DATA;
CREATE TABLE IF NOT EXISTS "AI_Source"."TrainingTeamCompetitionStats"
    AS SELECT * FROM "AI_Source"."TeamCompetitionStats" WITH NO DATA;

-- CREATE TABLE IF NOT EXISTS does not evolve an existing physical table when
-- ai_source_full.sql adds columns to the canonical views. Add newly projected
-- columns while preserving the already refreshed historical partitions.
DO $sync_training_serving_columns$
DECLARE
    relation_map record;
    source_column record;
BEGIN
    FOR relation_map IN
        SELECT *
          FROM (VALUES
              ('TrainingPlayerCompetitionStats', 'PlayerCompetitionStats'),
              ('TrainingTeamCompetitionStats', 'TeamCompetitionStats')
          ) AS source_pairs(target_table, source_relation)
    LOOP
        FOR source_column IN
            SELECT source.attname,
                   format_type(source.atttypid, source.atttypmod) AS column_type
              FROM pg_catalog.pg_attribute source
             WHERE source.attrelid = to_regclass(
                       format('%I.%I', 'AI_Source', relation_map.source_relation)
                   )
               AND source.attnum > 0
               AND NOT source.attisdropped
               AND NOT EXISTS (
                   SELECT 1
                     FROM pg_catalog.pg_attribute target
                    WHERE target.attrelid = to_regclass(
                              format('%I.%I', 'AI_Source', relation_map.target_table)
                          )
                      AND target.attname = source.attname
                      AND target.attnum > 0
                      AND NOT target.attisdropped
               )
             ORDER BY source.attnum
        LOOP
            EXECUTE format(
                'ALTER TABLE %I.%I ADD COLUMN %I %s',
                'AI_Source', relation_map.target_table,
                source_column.attname, source_column.column_type
            );
        END LOOP;
    END LOOP;
END;
$sync_training_serving_columns$;
CREATE UNIQUE INDEX IF NOT EXISTS ux_training_player_context
    ON "AI_Source"."TrainingPlayerCompetitionStats"
       (player_id, league_id, season, competition);
CREATE UNIQUE INDEX IF NOT EXISTS ux_training_team_context
    ON "AI_Source"."TrainingTeamCompetitionStats"
       (team_id, league_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_training_player_partition
    ON "AI_Source"."TrainingPlayerCompetitionStats"
       (league_key, season, competition);
CREATE INDEX IF NOT EXISTS ix_training_team_partition
    ON "AI_Source"."TrainingTeamCompetitionStats"
       (league_key, season, competition);

CREATE TABLE IF NOT EXISTS "AI_Source"."ScenarioPlayTypeStats" (
    player_id bigint, team_id bigint, opponent_team_id bigint,
    league_key text NOT NULL, season integer NOT NULL, competition text NOT NULL,
    play_type text NOT NULL, possessions double precision NOT NULL,
    ppp double precision, ppp_allowed double precision
);

CREATE TABLE IF NOT EXISTS "AI_Source"."ScenarioShotProfiles" (
    player_id bigint NOT NULL, team_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL, zone text NOT NULL,
    attempts double precision NOT NULL, fg_pct double precision
);

CREATE TABLE IF NOT EXISTS "AI_Source"."ScenarioDefenderMatchups" (
    offensive_player_id bigint NOT NULL, defender_id bigint NOT NULL,
    team_id bigint, opponent_team_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL,
    possessions double precision NOT NULL, assignment_probability double precision,
    points double precision, turnovers_forced double precision
);

CREATE TABLE IF NOT EXISTS "AI_Source"."ScenarioLineupStats" (
    player_ids text NOT NULL, team_id bigint, opponent_team_id bigint,
    offense_player_id bigint, defense_player_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL,
    assignment_probability double precision, possessions double precision NOT NULL,
    points double precision, points_allowed double precision,
    turnovers_forced double precision, ortg double precision, drtg double precision,
    net_rtg double precision
);


-- Simulation source tables are owned by this file, not ai_source_full.sql.
-- RefreshSimulationSources rebuilds these feeds from PBP and advanced lineup stats.
CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationPbpEvents" (
    source_event_id bigint NOT NULL,
    offensive_player_id bigint NOT NULL,
    defender_id bigint NOT NULL,
    team_id bigint,
    opponent_team_id bigint,
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    game_id text NOT NULL,
    period integer,
    clock_seconds double precision,
    score_margin double precision,
    play_type text,
    points double precision NOT NULL DEFAULT 0,
    turnover boolean NOT NULL DEFAULT false,
    assignment_probability double precision NOT NULL DEFAULT 0.2
);

CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationLineupStints" (
    player_ids text NOT NULL,
    team_id bigint NOT NULL,
    opponent_team_id bigint,
    offense_player_id bigint,
    defense_player_id bigint,
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    assignment_probability double precision NOT NULL DEFAULT 1.0,
    possessions double precision NOT NULL,
    points double precision,
    points_allowed double precision,
    turnovers_forced double precision,
    ortg double precision,
    drtg double precision,
    net_rtg double precision
);

CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationPlayTypeStats" (
    player_id bigint,
    team_id bigint,
    opponent_team_id bigint,
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    play_type text NOT NULL,
    possessions double precision NOT NULL,
    ppp double precision,
    ppp_allowed double precision
);

CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationShotProfiles" (
    player_id bigint NOT NULL,
    team_id bigint,
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    zone text NOT NULL,
    attempts double precision NOT NULL,
    fg_pct double precision
);

-- Intentionally schema-only until causal treatment/control identification is designed.
CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationCausalPanel" (
    treatment text,
    next_outcome double precision,
    source_type text,
    details jsonb
);

CREATE INDEX IF NOT EXISTS ix_simulation_pbp_context
    ON "AI_Source"."SimulationPbpEvents" (league_key, season, competition);
CREATE INDEX IF NOT EXISTS ix_simulation_pbp_matchup
    ON "AI_Source"."SimulationPbpEvents"
       (offensive_player_id, defender_id, league_key, season, competition);
CREATE INDEX IF NOT EXISTS ix_simulation_lineup_context
    ON "AI_Source"."SimulationLineupStints" (league_key, season, competition, team_id);
CREATE INDEX IF NOT EXISTS ix_simulation_shot_context
    ON "AI_Source"."SimulationShotProfiles" (league_key, season, competition, player_id);
CREATE INDEX IF NOT EXISTS ix_simulation_play_context
    ON "AI_Source"."SimulationPlayTypeStats" (league_key, season, competition, player_id);

CREATE TABLE IF NOT EXISTS "AI_Source"."SimulationSourceRefreshState" (
    "LeagueKey" text NOT NULL,
    "Season" integer NOT NULL,
    "RefreshedAt" timestamptz NOT NULL DEFAULT now(),
    "Status" text NOT NULL,
    "PbpRows" bigint NOT NULL DEFAULT 0,
    "PbpEventRows" bigint NOT NULL DEFAULT 0,
    "LineupRows" bigint NOT NULL DEFAULT 0,
    "ShotProfileRows" bigint NOT NULL DEFAULT 0,
    "PlayTypeRows" bigint NOT NULL DEFAULT 0,
    "Details" text NOT NULL DEFAULT '',
    PRIMARY KEY ("LeagueKey", "Season")
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_play_type_context
    ON "AI_Source"."ScenarioPlayTypeStats" (
        league_key, season, competition,
        COALESCE(player_id, -1), COALESCE(team_id, -1),
        COALESCE(opponent_team_id, -1), play_type
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_shot_context
    ON "AI_Source"."ScenarioShotProfiles" (
        league_key, season, competition, player_id, COALESCE(team_id, -1), zone
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_matchup_context
    ON "AI_Source"."ScenarioDefenderMatchups" (
        league_key, season, competition, offensive_player_id, defender_id,
        COALESCE(team_id, -1), COALESCE(opponent_team_id, -1)
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_lineup_context
    ON "AI_Source"."ScenarioLineupStats" (
        league_key, season, competition, player_ids,
        COALESCE(team_id, -1), COALESCE(opponent_team_id, -1),
        COALESCE(offense_player_id, -1), COALESCE(defense_player_id, -1)
    );

CREATE INDEX IF NOT EXISTS ix_scenario_play_type_player
    ON "AI_Source"."ScenarioPlayTypeStats" (player_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_play_type_team
    ON "AI_Source"."ScenarioPlayTypeStats" (team_id, opponent_team_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_shot_player
    ON "AI_Source"."ScenarioShotProfiles" (player_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_matchup_pair
    ON "AI_Source"."ScenarioDefenderMatchups" (offensive_player_id, defender_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_lineup_team
    ON "AI_Source"."ScenarioLineupStats" (team_id, season, competition);

CREATE TABLE IF NOT EXISTS "AI_Source"."ServingRefreshState" (
    "LeagueKey" text NOT NULL,
    "Season" integer NOT NULL,
    "Competition" text NOT NULL,
    "RefreshedAt" timestamptz NOT NULL,
    "PlayerRows" bigint NOT NULL,
    "TeamRows" bigint NOT NULL,
    "PlayTypeRows" bigint NOT NULL,
    "ShotProfileRows" bigint NOT NULL,
    "DefenderMatchupRows" bigint NOT NULL,
    "LineupRows" bigint NOT NULL,
    PRIMARY KEY ("LeagueKey", "Season", "Competition")
);

CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshSimulationSources"(
    p_league_key text,
    p_season integer
)
LANGUAGE plpgsql
AS $refresh_simulation$
DECLARE
    v_league_key text := upper(btrim(p_league_key));
    v_pbp_table text;
    v_lineup_table text;
    v_pbp_rel regclass;
    v_lineup_rel regclass;
    v_player_table text;
    v_team_table text;
    v_player_id_col text;
    v_player_global_col text;
    v_team_id_col text;
    v_team_global_col text;
    v_attempt_and1_col text;
    v_tactical_col text;
    v_player_global_expr text;
    v_team_global_expr text;
    v_attempt_and1_expr text := 'false';
    v_tactical_expr text := 'NULL::text';
    v_missing_columns text;
    v_stage_rows bigint := 0;
    v_event_rows bigint := 0;
    v_lineup_rows bigint := 0;
    v_shot_rows bigint := 0;
    v_play_rows bigint := 0;
    v_status text := 'unavailable';
    v_details text := '';
BEGIN
    IF p_league_key IS NULL OR v_league_key !~ '^[A-Z0-9_]+$' OR p_season IS NULL THEN
        RAISE EXCEPTION 'A valid league key and season are required';
    END IF;

    -- HoopmetricsEngine stores PBP in one table per league with Season rows.
    v_pbp_table := v_league_key;
    v_lineup_table := 'AdvancedStats_Lineups_Quarter_' || v_league_key;

    DELETE FROM "AI_Source"."SimulationPbpEvents"
     WHERE league_key = v_league_key AND season = p_season;
    DELETE FROM "AI_Source"."SimulationLineupStints"
     WHERE league_key = v_league_key AND season = p_season;
    DELETE FROM "AI_Source"."SimulationPlayTypeStats"
     WHERE league_key = v_league_key AND season = p_season;
    DELETE FROM "AI_Source"."SimulationShotProfiles"
     WHERE league_key = v_league_key AND season = p_season;

    v_pbp_rel := to_regclass(format('%I.%I', 'Pbp', v_pbp_table));
    IF v_pbp_rel IS NULL THEN
        v_details := format('Pbp.%s is missing; no Simulation rows were created.', v_pbp_table);
        INSERT INTO "AI_Source"."SimulationSourceRefreshState" (
            "LeagueKey", "Season", "RefreshedAt", "Status", "PbpRows",
            "PbpEventRows", "LineupRows", "ShotProfileRows", "PlayTypeRows", "Details"
        )
        VALUES (v_league_key, p_season, now(), 'unavailable', 0, 0, 0, 0, 0, v_details)
        ON CONFLICT ("LeagueKey", "Season") DO UPDATE SET
            "RefreshedAt"=EXCLUDED."RefreshedAt", "Status"=EXCLUDED."Status",
            "PbpRows"=0, "PbpEventRows"=0, "LineupRows"=0,
            "ShotProfileRows"=0, "PlayTypeRows"=0, "Details"=EXCLUDED."Details";
        RETURN;
    END IF;

    SELECT string_agg(required.column_name, ', ' ORDER BY required.column_name)
      INTO v_missing_columns
      FROM unnest(ARRAY[
          'season', 'competition', 'gamecode', 'player', 'teamid', 'homeclub',
          'quartertime', 'score', 'quarter', 'playtype', 'homelineup', 'awaylineup'
      ]) AS required(column_name)
     WHERE NOT EXISTS (
         SELECT 1
           FROM pg_catalog.pg_attribute a
          WHERE a.attrelid = v_pbp_rel
            AND a.attnum > 0 AND NOT a.attisdropped
            AND lower(a.attname) = required.column_name
     );

    IF v_missing_columns IS NOT NULL THEN
        v_details := format('Pbp.%s is missing required columns: %s', v_pbp_table, v_missing_columns);
        INSERT INTO "AI_Source"."SimulationSourceRefreshState" (
            "LeagueKey", "Season", "RefreshedAt", "Status", "PbpRows",
            "PbpEventRows", "LineupRows", "ShotProfileRows", "PlayTypeRows", "Details"
        )
        VALUES (v_league_key, p_season, now(), 'unavailable', 0, 0, 0, 0, 0, v_details)
        ON CONFLICT ("LeagueKey", "Season") DO UPDATE SET
            "RefreshedAt"=EXCLUDED."RefreshedAt", "Status"=EXCLUDED."Status",
            "PbpRows"=0, "PbpEventRows"=0, "LineupRows"=0,
            "ShotProfileRows"=0, "PlayTypeRows"=0, "Details"=EXCLUDED."Details";
        RETURN;
    END IF;

    v_player_table := v_league_key;
    v_team_table := 'Team_' || v_league_key;
    IF to_regclass(format('%I.%I', 'Anagrafiche', v_player_table)) IS NULL
       OR to_regclass(format('%I.%I', 'Anagrafiche', v_team_table)) IS NULL THEN
        v_details := format('Anagrafiche.%s or Anagrafiche.%s is missing; canonical ID mapping is unavailable.',
                            v_player_table, v_team_table);
        INSERT INTO "AI_Source"."SimulationSourceRefreshState" (
            "LeagueKey", "Season", "RefreshedAt", "Status", "PbpRows",
            "PbpEventRows", "LineupRows", "ShotProfileRows", "PlayTypeRows", "Details"
        )
        VALUES (v_league_key, p_season, now(), 'unavailable', 0, 0, 0, 0, 0, v_details)
        ON CONFLICT ("LeagueKey", "Season") DO UPDATE SET
            "RefreshedAt"=EXCLUDED."RefreshedAt", "Status"=EXCLUDED."Status",
            "PbpRows"=0, "PbpEventRows"=0, "LineupRows"=0,
            "ShotProfileRows"=0, "PlayTypeRows"=0, "Details"=EXCLUDED."Details";
        RETURN;
    END IF;

    SELECT a.attname INTO v_player_id_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = to_regclass(format('%I.%I', 'Anagrafiche', v_player_table))
       AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) IN ('id', 'playerid', 'idplayer')
     ORDER BY array_position(ARRAY['id', 'playerid', 'idplayer'], lower(a.attname))
     LIMIT 1;

    SELECT a.attname INTO v_player_global_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = to_regclass(format('%I.%I', 'Anagrafiche', v_player_table))
       AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) IN ('idglobal', 'globalid')
     ORDER BY array_position(ARRAY['idglobal', 'globalid'], lower(a.attname))
     LIMIT 1;

    SELECT a.attname INTO v_team_id_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = to_regclass(format('%I.%I', 'Anagrafiche', v_team_table))
       AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) IN ('id', 'teamid', 'idteam')
     ORDER BY array_position(ARRAY['id', 'teamid', 'idteam'], lower(a.attname))
     LIMIT 1;

    SELECT a.attname INTO v_team_global_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = to_regclass(format('%I.%I', 'Anagrafiche', v_team_table))
       AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) IN ('idglobal', 'globalid')
     ORDER BY array_position(ARRAY['idglobal', 'globalid'], lower(a.attname))
     LIMIT 1;

    IF v_player_id_col IS NULL OR v_team_id_col IS NULL THEN
        v_details := 'Anagrafiche player/team tables have no recognized local identity columns.';
        INSERT INTO "AI_Source"."SimulationSourceRefreshState" (
            "LeagueKey", "Season", "RefreshedAt", "Status", "PbpRows",
            "PbpEventRows", "LineupRows", "ShotProfileRows", "PlayTypeRows", "Details"
        )
        VALUES (v_league_key, p_season, now(), 'unavailable', 0, 0, 0, 0, 0, v_details)
        ON CONFLICT ("LeagueKey", "Season") DO UPDATE SET
            "RefreshedAt"=EXCLUDED."RefreshedAt", "Status"=EXCLUDED."Status",
            "PbpRows"=0, "PbpEventRows"=0, "LineupRows"=0,
            "ShotProfileRows"=0, "PlayTypeRows"=0, "Details"=EXCLUDED."Details";
        RETURN;
    END IF;

    IF v_player_global_col IS NULL THEN
        v_player_global_expr := format($expr$%1$L || ':' || btrim(a.%2$I::text)$expr$,
                                       v_league_key, v_player_id_col);
    ELSE
        v_player_global_expr := format(
            $expr$coalesce(nullif(btrim(a.%1$I::text), ''), %2$L || ':' || btrim(a.%3$I::text))$expr$,
            v_player_global_col, v_league_key, v_player_id_col
        );
    END IF;

    IF v_team_global_col IS NULL THEN
        v_team_global_expr := format($expr$%1$L || ':' || btrim(a.%2$I::text)$expr$,
                                     v_league_key, v_team_id_col);
    ELSE
        v_team_global_expr := format(
            $expr$coalesce(nullif(btrim(a.%1$I::text), ''), %2$L || ':' || btrim(a.%3$I::text))$expr$,
            v_team_global_col, v_league_key, v_team_id_col
        );
    END IF;

    DROP TABLE IF EXISTS pg_temp.simulation_player_map;
    DROP TABLE IF EXISTS pg_temp.simulation_team_map;
    DROP TABLE IF EXISTS pg_temp.simulation_pbp_stage;
    DROP TABLE IF EXISTS pg_temp.simulation_advanced_lineup_stage;

    EXECUTE format($player_map$
        CREATE TEMP TABLE simulation_player_map ON COMMIT DROP AS
        SELECT DISTINCT ON (source_player_id) source_player_id, p.id::bigint AS player_id
        FROM (
            SELECT lower(regexp_replace(btrim(a.%1$I::text), '[.]0+$', '')) AS source_player_id,
                   %2$s AS global_id
              FROM "Anagrafiche".%3$I a
             WHERE a.%1$I IS NOT NULL
        ) ids
        JOIN "AI_Source"."Players" p
          ON lower(btrim(p.global_id)) = lower(btrim(ids.global_id))
        WHERE source_player_id <> ''
        ORDER BY source_player_id, p.id
    $player_map$, v_player_id_col, v_player_global_expr, v_player_table);

    EXECUTE format($team_map$
        CREATE TEMP TABLE simulation_team_map ON COMMIT DROP AS
        SELECT DISTINCT ON (source_team_id) source_team_id, t.id::bigint AS team_id
        FROM (
            SELECT lower(regexp_replace(btrim(a.%1$I::text), '[.]0+$', '')) AS source_team_id,
                   %2$s AS global_id
              FROM "Anagrafiche".%3$I a
             WHERE a.%1$I IS NOT NULL
        ) ids
        JOIN "AI_Source"."Teams" t
          ON lower(btrim(t.global_id)) = lower(btrim(ids.global_id))
        WHERE source_team_id <> ''
        ORDER BY source_team_id, t.id
    $team_map$, v_team_id_col, v_team_global_expr, v_team_table);

    SELECT a.attname INTO v_attempt_and1_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = v_pbp_rel AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) = 'attemptand1'
     LIMIT 1;
    IF v_attempt_and1_col IS NOT NULL THEN
        v_attempt_and1_expr := format(
            $and1$lower(btrim(coalesce(p.%I::text, ''))) IN ('1', 't', 'true', 'yes')$and1$,
            v_attempt_and1_col
        );
    END IF;

    SELECT a.attname INTO v_tactical_col
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = v_pbp_rel AND a.attnum > 0 AND NOT a.attisdropped
       AND lower(a.attname) IN (
           'tacticalplaytype', 'offensiveplaytype', 'playclassification',
           'playfamily', 'playcall', 'playtypecategory', 'playinfo'
       )
     ORDER BY array_position(
         ARRAY['tacticalplaytype', 'offensiveplaytype', 'playclassification',
               'playfamily', 'playcall', 'playtypecategory', 'playinfo'],
         lower(a.attname)
     )
     LIMIT 1;

    IF v_tactical_col IS NOT NULL THEN
        v_tactical_expr := format(
            $tactical$lower(regexp_replace(btrim(coalesce(p.%I::text, '')), '[^a-zA-Z0-9]+', '_', 'g'))$tactical$,
            v_tactical_col
        );
    END IF;

    EXECUTE format($stage$
        CREATE TEMP TABLE simulation_pbp_stage ON COMMIT DROP AS
        WITH raw AS (
            SELECT
                row_number() OVER ()::bigint AS source_event_id,
                nullif(btrim(p."GameCode"::text), '') AS game_id,
                CASE upper(btrim(coalesce(p."Competition"::text, 'RS')))
                    WHEN 'REGULAR SEASON' THEN 'RS'
                    WHEN 'REG SEASON' THEN 'RS'
                    WHEN 'PLAYOFFS' THEN 'PO'
                    WHEN 'PLAYOFF' THEN 'PO'
                    WHEN 'TOTAL' THEN 'TOT'
                    WHEN 'ALL' THEN 'TOT'
                    ELSE upper(btrim(coalesce(p."Competition"::text, 'RS')))
                END AS competition,
                nullif(btrim(p."Player"::text), '') AS source_player_id,
                nullif(btrim(p."TeamId"::text), '') AS source_team_id,
                nullif(btrim(p."HomeLineup"::text), '') AS home_lineup_raw,
                nullif(btrim(p."AwayLineup"::text), '') AS away_lineup_raw,
                upper(nullif(btrim(p."PlayType"::text), '')) AS play_type,
                p."Score"::text AS score_text,
                p."QuarterTime"::text AS clock_text,
                p."Quarter"::text AS period_text,
                CASE lower(btrim(coalesce(p."HomeClub"::text, '')))
                    WHEN '1' THEN true WHEN 'true' THEN true WHEN 't' THEN true
                    WHEN 'yes' THEN true WHEN 'home' THEN true
                    WHEN '0' THEN false WHEN 'false' THEN false WHEN 'f' THEN false
                    WHEN 'no' THEN false WHEN 'away' THEN false
                    ELSE NULL::boolean
                END AS home_flag,
                %2$s AS attempt_and1,
                %3$s AS tactical_play_type
            FROM "Pbp".%1$I p
            WHERE p."Season"::integer = $1
        ),
        game_teams AS (
            SELECT game_id, competition,
                   max(source_team_id) FILTER (WHERE home_flag IS TRUE) AS home_source_team_id,
                   max(source_team_id) FILTER (WHERE home_flag IS FALSE) AS away_source_team_id
              FROM raw
             WHERE game_id IS NOT NULL
             GROUP BY game_id, competition
        ),
        contextual AS (
            SELECT r.*, gt.home_source_team_id, gt.away_source_team_id,
                   coalesce(
                       r.home_flag,
                       CASE
                           WHEN lower(regexp_replace(btrim(r.source_team_id), '[.]0+$', '')) =
                                lower(regexp_replace(btrim(gt.home_source_team_id), '[.]0+$', ''))
                               THEN true
                           WHEN lower(regexp_replace(btrim(r.source_team_id), '[.]0+$', '')) =
                                lower(regexp_replace(btrim(gt.away_source_team_id), '[.]0+$', ''))
                               THEN false
                           ELSE NULL::boolean
                       END
                   ) AS effective_is_home,
                   regexp_match(
                       r.score_text,
                       '^[[:space:]]*(-?[0-9]+[.]?[0-9]*)[[:space:]]*[:-][[:space:]]*(-?[0-9]+[.]?[0-9]*)[[:space:]]*$'
                   ) AS score_parts,
                   CASE
                       WHEN r.clock_text ~ '^[0-9]+:[0-9]{1,2}$'
                           THEN split_part(r.clock_text, ':', 1)::double precision * 60
                              + split_part(r.clock_text, ':', 2)::double precision
                       WHEN r.clock_text ~ '^[0-9]+([.][0-9]+)?$'
                           THEN r.clock_text::double precision
                       ELSE NULL::double precision
                   END AS clock_seconds,
                   CASE WHEN r.period_text ~ '^[0-9]+$'
                        THEN r.period_text::integer ELSE NULL::integer END AS period
              FROM raw r
              LEFT JOIN game_teams gt USING (game_id, competition)
        ),
        mapped AS (
            SELECT c.*,
                   pm.player_id AS offensive_player_id,
                   coalesce(
                       CASE WHEN c.effective_is_home THEN th.team_id
                            WHEN c.effective_is_home IS FALSE THEN ta.team_id
                       END,
                       te.team_id
                   ) AS team_id,
                   CASE WHEN c.effective_is_home THEN ta.team_id
                        WHEN c.effective_is_home IS FALSE THEN th.team_id
                        ELSE NULL::bigint
                   END AS opponent_team_id,
                   CASE WHEN c.score_parts IS NOT NULL
                        THEN abs((c.score_parts[1])::double precision -
                                 (c.score_parts[2])::double precision)
                        ELSE NULL::double precision END AS score_margin
              FROM contextual c
              LEFT JOIN pg_temp.simulation_player_map pm
                ON pm.source_player_id = lower(regexp_replace(btrim(c.source_player_id), '[.]0+$', ''))
              LEFT JOIN pg_temp.simulation_team_map te
                ON te.source_team_id = lower(regexp_replace(btrim(c.source_team_id), '[.]0+$', ''))
              LEFT JOIN pg_temp.simulation_team_map th
                ON th.source_team_id = lower(regexp_replace(btrim(c.home_source_team_id), '[.]0+$', ''))
              LEFT JOIN pg_temp.simulation_team_map ta
                ON ta.source_team_id = lower(regexp_replace(btrim(c.away_source_team_id), '[.]0+$', ''))
        ),
        with_lineups AS (
            SELECT m.*,
                   hl.player_ids AS home_player_ids,
                   hl.player_count AS home_player_count,
                   al.player_ids AS away_player_ids,
                   al.player_count AS away_player_count
              FROM mapped m
              LEFT JOIN LATERAL (
                  SELECT string_agg(players.player_id::text, ',' ORDER BY players.player_id) AS player_ids,
                         count(*)::integer AS player_count
                    FROM (
                        SELECT DISTINCT pm.player_id
                          FROM regexp_split_to_table(coalesce(m.home_lineup_raw, ''), '[[:space:]]+-[[:space:]]+') AS token(value)
                          JOIN pg_temp.simulation_player_map pm
                            ON pm.source_player_id = lower(regexp_replace(btrim(token.value), '[.]0+$', ''))
                    ) players
              ) hl ON true
              LEFT JOIN LATERAL (
                  SELECT string_agg(players.player_id::text, ',' ORDER BY players.player_id) AS player_ids,
                         count(*)::integer AS player_count
                    FROM (
                        SELECT DISTINCT pm.player_id
                          FROM regexp_split_to_table(coalesce(m.away_lineup_raw, ''), '[[:space:]]+-[[:space:]]+') AS token(value)
                          JOIN pg_temp.simulation_player_map pm
                            ON pm.source_player_id = lower(regexp_replace(btrim(token.value), '[.]0+$', ''))
                    ) players
              ) al ON true
        )
        SELECT source_event_id, game_id, competition,
               offensive_player_id, team_id, opponent_team_id,
               period, clock_seconds, score_margin, play_type,
               CASE play_type WHEN '2FGM' THEN 2.0 WHEN '3FGM' THEN 3.0
                              WHEN 'FTM' THEN 1.0 ELSE 0.0 END AS points,
               coalesce(play_type = 'TOV', false) AS turnover,
               attempt_and1, tactical_play_type,
               home_player_ids, home_player_count, away_player_ids, away_player_count,
               CASE WHEN effective_is_home THEN home_player_ids ELSE away_player_ids END AS offensive_player_ids,
               CASE WHEN effective_is_home THEN home_player_count ELSE away_player_count END AS offensive_player_count,
               CASE WHEN effective_is_home THEN away_player_ids ELSE home_player_ids END AS defensive_player_ids,
               CASE WHEN effective_is_home THEN away_player_count ELSE home_player_count END AS defensive_player_count
          FROM with_lineups
         WHERE game_id IS NOT NULL
    $stage$, v_pbp_table, v_attempt_and1_expr, v_tactical_expr)
    USING p_season;

    SELECT count(*) INTO v_stage_rows FROM pg_temp.simulation_pbp_stage;

    INSERT INTO "AI_Source"."SimulationPbpEvents" (
        source_event_id, offensive_player_id, defender_id, team_id, opponent_team_id,
        league_key, season, competition, game_id, period, clock_seconds, score_margin,
        play_type, points, turnover, assignment_probability
    )
    SELECT s.source_event_id, s.offensive_player_id, d.defender_id, s.team_id, s.opponent_team_id,
           v_league_key, p_season, s.competition, s.game_id, s.period, s.clock_seconds,
           s.score_margin, s.play_type, s.points, s.turnover,
           1.0 / nullif(s.defensive_player_count, 0)
      FROM pg_temp.simulation_pbp_stage s
      CROSS JOIN LATERAL unnest(string_to_array(s.defensive_player_ids, ',')::bigint[]) AS d(defender_id)
     WHERE s.offensive_player_id IS NOT NULL
       AND s.defensive_player_count = 5
       AND s.team_id IS NOT NULL
       AND s.opponent_team_id IS NOT NULL
       AND s.play_type IN ('2FGM','2FGA','3FGM','3FGA','FTM','FTA','AST','TOV','ORB');
    GET DIAGNOSTICS v_event_rows = ROW_COUNT;

    -- Without shot coordinates the location is unknown; do not invent rim/corner zones.
    INSERT INTO "AI_Source"."SimulationShotProfiles" (
        player_id, team_id, league_key, season, competition, zone, attempts, fg_pct
    )
    SELECT s.offensive_player_id, s.team_id, v_league_key, p_season, s.competition,
           CASE WHEN s.play_type IN ('2FGM','2FGA') THEN 'unknown_two_point'
                ELSE 'unknown_three_point' END,
           count(*)::double precision,
           avg(CASE WHEN s.play_type IN ('2FGM','3FGM') THEN 1.0 ELSE 0.0 END)
      FROM pg_temp.simulation_pbp_stage s
     WHERE s.offensive_player_id IS NOT NULL
       AND s.team_id IS NOT NULL
       AND s.play_type IN ('2FGM','2FGA','3FGM','3FGA')
     GROUP BY s.offensive_player_id, s.team_id, s.competition,
              CASE WHEN s.play_type IN ('2FGM','2FGA') THEN 'unknown_two_point'
                   ELSE 'unknown_three_point' END;
    GET DIAGNOSTICS v_shot_rows = ROW_COUNT;

    -- Only a separate source taxonomy supplies tactical types. Raw action codes
    -- are never renamed to pick-and-roll or isolation.
    INSERT INTO "AI_Source"."SimulationPlayTypeStats" (
        player_id, team_id, opponent_team_id, league_key, season, competition,
        play_type, possessions, ppp, ppp_allowed
    )
    SELECT s.offensive_player_id, s.team_id, s.opponent_team_id,
           v_league_key, p_season, s.competition, s.tactical_play_type,
           count(*)::double precision, avg(s.points), NULL::double precision
      FROM pg_temp.simulation_pbp_stage s
     WHERE s.offensive_player_id IS NOT NULL
       AND s.team_id IS NOT NULL
       AND s.tactical_play_type IN (
           'pick_and_roll_ball_handler','pick_and_roll_roll_man','isolation',
           'post_up','spot_up','handoff','cut','off_screen','transition',
           'putback','miscellaneous'
       )
     GROUP BY s.offensive_player_id, s.team_id, s.opponent_team_id,
              s.competition, s.tactical_play_type;
    GET DIAGNOSTICS v_play_rows = ROW_COUNT;

    -- Prefer the official quarter-level lineup output if it has the expected fields.
    v_lineup_rel := to_regclass(format('%I.%I', 'Analisi', v_lineup_table));
    IF v_lineup_rel IS NOT NULL
       AND NOT EXISTS (
           SELECT 1
             FROM unnest(ARRAY[
                 'lineup','teamid','season','competition','offposs','defposs',
                 'pts','ptsallowed','opptov'
             ]) AS required(column_name)
            WHERE NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_attribute a
                 WHERE a.attrelid = v_lineup_rel
                   AND a.attnum > 0 AND NOT a.attisdropped
                   AND lower(a.attname) = required.column_name
            )
       ) THEN
        EXECUTE format($advanced_lineup$
            CREATE TEMP TABLE simulation_advanced_lineup_stage ON COMMIT DROP AS
            WITH raw AS (
                SELECT
                    nullif(btrim(q."Lineup"::text), '') AS lineup_raw,
                    nullif(btrim(q."TeamId"::text), '') AS source_team_id,
                    CASE upper(btrim(coalesce(q."Competition"::text, 'RS')))
                        WHEN 'REGULAR SEASON' THEN 'RS'
                        WHEN 'PLAYOFFS' THEN 'PO'
                        WHEN 'PLAYOFF' THEN 'PO'
                        WHEN 'TOTAL' THEN 'TOT'
                        ELSE upper(btrim(coalesce(q."Competition"::text, 'RS')))
                    END AS competition,
                    coalesce(q."OffPoss"::double precision, 0) AS off_possessions,
                    coalesce(q."DefPoss"::double precision, 0) AS def_possessions,
                    coalesce(q."Pts"::double precision, 0) AS points,
                    coalesce(q."PtsAllowed"::double precision, 0) AS points_allowed,
                    coalesce(q."OppTov"::double precision, 0) AS turnovers_forced
                FROM "Analisi".%1$I q
                WHERE q."Season"::integer = $1
                  AND upper(btrim(coalesce(q."Competition"::text, 'RS'))) NOT IN ('TOT','HOME','AWAY')
                  AND nullif(btrim(q."TeamId"::text), '') IS NOT NULL
                  AND nullif(btrim(q."Lineup"::text), '') IS NOT NULL
            ),
            mapped AS (
                SELECT r.*, tm.team_id, lu.player_ids, lu.player_count
                  FROM raw r
                  JOIN pg_temp.simulation_team_map tm
                    ON tm.source_team_id = lower(regexp_replace(btrim(r.source_team_id), '[.]0+$', ''))
                  LEFT JOIN LATERAL (
                      SELECT string_agg(players.player_id::text, ',' ORDER BY players.player_id) AS player_ids,
                             count(*)::integer AS player_count
                        FROM (
                            SELECT DISTINCT pm.player_id
                              FROM regexp_split_to_table(coalesce(r.lineup_raw, ''), '[[:space:]]+-[[:space:]]+') AS token(value)
                              JOIN pg_temp.simulation_player_map pm
                                ON pm.source_player_id = lower(regexp_replace(btrim(token.value), '[.]0+$', ''))
                        ) players
                  ) lu ON true
            )
            SELECT player_ids, team_id, competition,
                   off_possessions, def_possessions, points, points_allowed, turnovers_forced
              FROM mapped
             WHERE player_count = 5
        $advanced_lineup$, v_lineup_table)
        USING p_season;

        INSERT INTO "AI_Source"."SimulationLineupStints" (
            player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
            league_key, season, competition, assignment_probability, possessions,
            points, points_allowed, turnovers_forced, ortg, drtg, net_rtg
        )
        WITH agg AS (
            SELECT player_ids, team_id, competition,
                   sum(off_possessions) AS off_possessions,
                   sum(def_possessions) AS def_possessions,
                   sum(points) AS points,
                   sum(points_allowed) AS points_allowed,
                   sum(turnovers_forced) AS turnovers_forced
              FROM pg_temp.simulation_advanced_lineup_stage
             GROUP BY player_ids, team_id, competition
        )
        SELECT player_ids, team_id, NULL::bigint, NULL::bigint, NULL::bigint,
               v_league_key, p_season, competition, 1.0,
               greatest(off_possessions, def_possessions),
               points, points_allowed, turnovers_forced,
               100.0 * points / nullif(off_possessions, 0),
               100.0 * points_allowed / nullif(def_possessions, 0),
               100.0 * points / nullif(off_possessions, 0)
                 - 100.0 * points_allowed / nullif(def_possessions, 0)
          FROM agg
         WHERE greatest(off_possessions, def_possessions) > 0;

        GET DIAGNOSTICS v_lineup_rows = ROW_COUNT;
        DROP TABLE IF EXISTS pg_temp.simulation_advanced_lineup_stage;
    END IF;

    -- Use a PBP-derived fallback where no advanced lineup aggregates were usable.
    IF v_lineup_rows = 0 THEN
        WITH off_rows AS (
            SELECT offensive_player_ids AS player_ids, team_id, opponent_team_id,
                   competition,
                   greatest(sum(
                       CASE WHEN play_type IN ('2FGM','2FGA','3FGM','3FGA') THEN 1.0
                            WHEN play_type IN ('FTM','FTA') AND NOT attempt_and1 THEN 0.44
                            WHEN play_type = 'TOV' THEN 1.0
                            WHEN play_type = 'ORB' THEN -1.0
                            ELSE 0.0 END
                   ), 0) AS off_possessions,
                   sum(points) AS points
              FROM pg_temp.simulation_pbp_stage
             WHERE offensive_player_count = 5 AND team_id IS NOT NULL AND opponent_team_id IS NOT NULL
             GROUP BY offensive_player_ids, team_id, opponent_team_id, competition
        ),
        def_rows AS (
            SELECT defensive_player_ids AS player_ids, opponent_team_id AS team_id,
                   team_id AS opponent_team_id, competition,
                   greatest(sum(
                       CASE WHEN play_type IN ('2FGM','2FGA','3FGM','3FGA') THEN 1.0
                            WHEN play_type IN ('FTM','FTA') AND NOT attempt_and1 THEN 0.44
                            WHEN play_type = 'TOV' THEN 1.0
                            WHEN play_type = 'ORB' THEN -1.0
                            ELSE 0.0 END
                   ), 0) AS def_possessions,
                   sum(points) AS points_allowed,
                   sum(CASE WHEN turnover THEN 1.0 ELSE 0.0 END) AS turnovers_forced
              FROM pg_temp.simulation_pbp_stage
             WHERE defensive_player_count = 5 AND team_id IS NOT NULL AND opponent_team_id IS NOT NULL
             GROUP BY defensive_player_ids, opponent_team_id, team_id, competition
        ),
        combined AS (
            SELECT coalesce(o.player_ids, d.player_ids) AS player_ids,
                   coalesce(o.team_id, d.team_id) AS team_id,
                   coalesce(o.opponent_team_id, d.opponent_team_id) AS opponent_team_id,
                   coalesce(o.competition, d.competition) AS competition,
                   coalesce(o.off_possessions, 0) AS off_possessions,
                   coalesce(d.def_possessions, 0) AS def_possessions,
                   coalesce(o.points, 0) AS points,
                   coalesce(d.points_allowed, 0) AS points_allowed,
                   coalesce(d.turnovers_forced, 0) AS turnovers_forced
              FROM off_rows o
              FULL OUTER JOIN def_rows d
                ON d.player_ids=o.player_ids AND d.team_id=o.team_id
               AND d.opponent_team_id=o.opponent_team_id AND d.competition=o.competition
        )
        INSERT INTO "AI_Source"."SimulationLineupStints" (
            player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
            league_key, season, competition, assignment_probability, possessions,
            points, points_allowed, turnovers_forced, ortg, drtg, net_rtg
        )
        SELECT player_ids, team_id, opponent_team_id, NULL::bigint, NULL::bigint,
               v_league_key, p_season, competition, 1.0,
               greatest(off_possessions, def_possessions),
               points, points_allowed, turnovers_forced,
               100.0 * points / nullif(off_possessions, 0),
               100.0 * points_allowed / nullif(def_possessions, 0),
               100.0 * points / nullif(off_possessions, 0)
                 - 100.0 * points_allowed / nullif(def_possessions, 0)
          FROM combined
         WHERE greatest(off_possessions, def_possessions) > 0;
    END IF;

    SELECT count(*) INTO v_event_rows
      FROM "AI_Source"."SimulationPbpEvents"
     WHERE league_key=v_league_key AND season=p_season;
    SELECT count(*) INTO v_lineup_rows
      FROM "AI_Source"."SimulationLineupStints"
     WHERE league_key=v_league_key AND season=p_season;
    SELECT count(*) INTO v_shot_rows
      FROM "AI_Source"."SimulationShotProfiles"
     WHERE league_key=v_league_key AND season=p_season;
    SELECT count(*) INTO v_play_rows
      FROM "AI_Source"."SimulationPlayTypeStats"
     WHERE league_key=v_league_key AND season=p_season;

    v_status := CASE
        WHEN v_stage_rows = 0 THEN 'unavailable'
        WHEN v_event_rows = 0 OR v_lineup_rows = 0 OR v_shot_rows = 0 OR v_play_rows = 0 THEN 'partial'
        ELSE 'available'
    END;
    v_details := format(
        'PBP rows=%s; probabilistic defender exposures=%s; lineup aggregates=%s; shot-profile rows=%s; tactical play-type rows=%s. Shot locations are unknown without coordinates. Tactical labels require a separate recognized source taxonomy. SimulationCausalPanel remains empty until a causal identification design is approved.',
        v_stage_rows, v_event_rows, v_lineup_rows, v_shot_rows, v_play_rows
    );

    INSERT INTO "AI_Source"."SimulationSourceRefreshState" (
        "LeagueKey", "Season", "RefreshedAt", "Status", "PbpRows",
        "PbpEventRows", "LineupRows", "ShotProfileRows", "PlayTypeRows", "Details"
    )
    VALUES (v_league_key, p_season, now(), v_status, v_stage_rows,
            v_event_rows, v_lineup_rows, v_shot_rows, v_play_rows, v_details)
    ON CONFLICT ("LeagueKey", "Season") DO UPDATE SET
        "RefreshedAt"=EXCLUDED."RefreshedAt", "Status"=EXCLUDED."Status",
        "PbpRows"=EXCLUDED."PbpRows", "PbpEventRows"=EXCLUDED."PbpEventRows",
        "LineupRows"=EXCLUDED."LineupRows", "ShotProfileRows"=EXCLUDED."ShotProfileRows",
        "PlayTypeRows"=EXCLUDED."PlayTypeRows", "Details"=EXCLUDED."Details";
END;
$refresh_simulation$;

CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshScenarioServing"(
    p_league_key text,
    p_season integer,
    p_competition text
)
LANGUAGE plpgsql
AS $$
DECLARE normalized_competition text := upper(btrim(p_competition));
        affected_rows bigint;
BEGIN
    IF p_league_key IS NULL OR p_season IS NULL OR normalized_competition = '' THEN
        RAISE EXCEPTION 'league_key, season and competition are required';
    END IF;

    -- Simulation source tables are owned and refreshed by this SQL. Some feeds
    -- may remain empty when their source taxonomy/data is unavailable; we never
    -- manufacture tactical play types or causal labels from raw event codes.
    DELETE FROM "AI_Source"."ScenarioPlayTypeStats"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    IF to_regclass('"AI_Source"."SimulationPlayTypeStats"') IS NOT NULL THEN
        EXECUTE $copy_play_type$
            INSERT INTO "AI_Source"."ScenarioPlayTypeStats" (
                player_id, team_id, opponent_team_id, league_key, season, competition,
                play_type, possessions, ppp, ppp_allowed
            )
            SELECT player_id, team_id, opponent_team_id, league_key, season, upper(competition),
                   play_type, possessions, ppp, ppp_allowed
              FROM "AI_Source"."SimulationPlayTypeStats"
             WHERE league_key = $1 AND season = $2 AND upper(competition) = $3
        $copy_play_type$ USING p_league_key, p_season, normalized_competition;
        GET DIAGNOSTICS affected_rows = ROW_COUNT;
        IF affected_rows = 0 THEN
            RAISE NOTICE 'ScenarioPlayTypeStats remains empty for %/%/%: SimulationPlayTypeStats has no matching source rows',
                p_league_key, p_season, normalized_competition;
        END IF;
    ELSE
        RAISE NOTICE 'Skipping play-type scenario feed: AI_Source.SimulationPlayTypeStats is missing';
    END IF;

    DELETE FROM "AI_Source"."ScenarioShotProfiles"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    IF to_regclass('"AI_Source"."SimulationShotProfiles"') IS NOT NULL THEN
        EXECUTE $copy_shot_profiles$
            INSERT INTO "AI_Source"."ScenarioShotProfiles" (
                player_id, team_id, league_key, season, competition, zone, attempts, fg_pct
            )
            SELECT player_id, team_id, league_key, season, upper(competition), zone, attempts, fg_pct
              FROM "AI_Source"."SimulationShotProfiles"
             WHERE league_key = $1 AND season = $2 AND upper(competition) = $3
        $copy_shot_profiles$ USING p_league_key, p_season, normalized_competition;
        GET DIAGNOSTICS affected_rows = ROW_COUNT;
        IF affected_rows = 0 THEN
            RAISE NOTICE 'ScenarioShotProfiles remains empty for %/%/%: SimulationShotProfiles has no matching source rows',
                p_league_key, p_season, normalized_competition;
        END IF;
    ELSE
        RAISE NOTICE 'Skipping shot-profile scenario feed: AI_Source.SimulationShotProfiles is missing';
    END IF;

    DELETE FROM "AI_Source"."ScenarioDefenderMatchups"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    IF to_regclass('"AI_Source"."SimulationPbpEvents"') IS NOT NULL THEN
        EXECUTE $copy_defender_matchups$
            INSERT INTO "AI_Source"."ScenarioDefenderMatchups" (
                offensive_player_id, defender_id, team_id, opponent_team_id,
                league_key, season, competition, possessions, assignment_probability,
                points, turnovers_forced
            )
            SELECT offensive_player_id, defender_id, team_id, opponent_team_id,
                   league_key, season, upper(competition), count(*)::double precision,
                   avg(assignment_probability), sum(points),
                   sum(CASE WHEN turnover THEN 1 ELSE 0 END)::double precision
              FROM "AI_Source"."SimulationPbpEvents"
             WHERE league_key = $1 AND season = $2
               AND upper(competition) = $3 AND defender_id IS NOT NULL
             GROUP BY offensive_player_id, defender_id, team_id, opponent_team_id,
                      league_key, season, upper(competition)
        $copy_defender_matchups$ USING p_league_key, p_season, normalized_competition;
        GET DIAGNOSTICS affected_rows = ROW_COUNT;
        IF affected_rows = 0 THEN
            RAISE NOTICE 'ScenarioDefenderMatchups remains empty for %/%/%: SimulationPbpEvents has no matching non-null defender rows',
                p_league_key, p_season, normalized_competition;
        END IF;
    ELSE
        RAISE NOTICE 'Skipping defender matchup feed: AI_Source.SimulationPbpEvents is missing';
    END IF;

    DELETE FROM "AI_Source"."ScenarioLineupStats"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    IF to_regclass('"AI_Source"."SimulationLineupStints"') IS NOT NULL THEN
        EXECUTE $copy_lineup_stats$
            INSERT INTO "AI_Source"."ScenarioLineupStats" (
                player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
                league_key, season, competition, assignment_probability, possessions, points,
                points_allowed, turnovers_forced, ortg, drtg, net_rtg
            )
            SELECT player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
                   league_key, season, upper(competition),
                   avg(assignment_probability), sum(possessions), sum(points), sum(points_allowed),
                   sum(turnovers_forced),
                   sum(ortg * possessions) / NULLIF(sum(possessions), 0),
                   sum(drtg * possessions) / NULLIF(sum(possessions), 0),
                   sum(net_rtg * possessions) / NULLIF(sum(possessions), 0)
              FROM "AI_Source"."SimulationLineupStints"
             WHERE league_key = $1 AND season = $2 AND upper(competition) = $3
             GROUP BY player_ids, team_id, opponent_team_id, offense_player_id,
                      defense_player_id, league_key, season, upper(competition)
        $copy_lineup_stats$ USING p_league_key, p_season, normalized_competition;
        GET DIAGNOSTICS affected_rows = ROW_COUNT;
        IF affected_rows = 0 THEN
            RAISE NOTICE 'ScenarioLineupStats remains empty for %/%/%: SimulationLineupStints has no matching source rows',
                p_league_key, p_season, normalized_competition;
        END IF;
    ELSE
        RAISE NOTICE 'Skipping lineup scenario feed: AI_Source.SimulationLineupStints is missing';
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshTrainingServing"(
    p_league_key text,
    p_season integer,
    p_competition text
)
LANGUAGE plpgsql
AS $refresh_training$
DECLARE
    normalized_competition text := upper(btrim(p_competition));
    v_player_columns text;
    v_team_columns text;
BEGIN
    IF p_league_key IS NULL OR p_season IS NULL OR normalized_competition = '' THEN
        RAISE EXCEPTION 'league_key, season and competition are required';
    END IF;

    -- Build projection lists from the canonical source relation, not SELECT *.
    -- Existing physical stores may contain old columns in a different order.
    SELECT string_agg(format('%I', source.attname), ', ' ORDER BY source.attnum)
      INTO v_player_columns
      FROM pg_catalog.pg_attribute source
     WHERE source.attrelid = to_regclass('"AI_Source"."PlayerCompetitionStats"')
       AND source.attnum > 0
       AND NOT source.attisdropped
       AND EXISTS (
           SELECT 1
             FROM pg_catalog.pg_attribute target
            WHERE target.attrelid = to_regclass('"AI_Source"."TrainingPlayerCompetitionStats"')
              AND target.attname = source.attname
              AND target.attnum > 0
              AND NOT target.attisdropped
       );

    SELECT string_agg(format('%I', source.attname), ', ' ORDER BY source.attnum)
      INTO v_team_columns
      FROM pg_catalog.pg_attribute source
     WHERE source.attrelid = to_regclass('"AI_Source"."TeamCompetitionStats"')
       AND source.attnum > 0
       AND NOT source.attisdropped
       AND EXISTS (
           SELECT 1
             FROM pg_catalog.pg_attribute target
            WHERE target.attrelid = to_regclass('"AI_Source"."TrainingTeamCompetitionStats"')
              AND target.attname = source.attname
              AND target.attnum > 0
              AND NOT target.attisdropped
       );

    IF v_player_columns IS NULL OR v_team_columns IS NULL THEN
        RAISE EXCEPTION 'Training serving schema does not match canonical AI_Source statistics';
    END IF;

    DELETE FROM "AI_Source"."TrainingPlayerCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;
    EXECUTE format(
        'INSERT INTO "AI_Source"."TrainingPlayerCompetitionStats" (%1$s)
         SELECT %1$s FROM "AI_Source"."PlayerCompetitionStats"
          WHERE league_key = $1 AND season = $2 AND upper(competition) = $3',
        v_player_columns
    ) USING p_league_key, p_season, normalized_competition;

    DELETE FROM "AI_Source"."TrainingTeamCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;
    EXECUTE format(
        'INSERT INTO "AI_Source"."TrainingTeamCompetitionStats" (%1$s)
         SELECT %1$s FROM "AI_Source"."TeamCompetitionStats"
          WHERE league_key = $1 AND season = $2 AND upper(competition) = $3',
        v_team_columns
    ) USING p_league_key, p_season, normalized_competition;
END;
$refresh_training$;

-- Canonical ETL entry point: refresh every league and every competition for a
-- season from the current canonical AI_Source statistical tables. Callers use
-- RefreshContextSeasonal; there is no public single-context RefreshContext overload.
CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshContextSeasonal"(p_season integer)
LANGUAGE plpgsql
AS $refresh_season$
DECLARE
    ctx record;
    sim_ctx record;
    processed_contexts integer := 0;
BEGIN
    IF p_season IS NULL THEN
        RAISE EXCEPTION 'season is required';
    END IF;

    -- Rebuild Simulation inputs once per league-season, before refreshing each
    -- individual competition's Scenario* serving table.
    FOR sim_ctx IN
        SELECT contexts.league_key, contexts.season
        FROM (
            SELECT NULLIF(btrim(league_key), '') AS league_key, season
              FROM "AI_Source"."PlayerCompetitionStats"
             WHERE season = p_season
            UNION
            SELECT NULLIF(btrim(league_key), '') AS league_key, season
              FROM "AI_Source"."TeamCompetitionStats"
             WHERE season = p_season
        ) contexts
        WHERE contexts.league_key IS NOT NULL
        ORDER BY contexts.league_key, contexts.season
    LOOP
        CALL "AI_Source"."RefreshSimulationSources"(sim_ctx.league_key, sim_ctx.season);
    END LOOP;

    FOR ctx IN
        SELECT contexts.league_key, contexts.season, contexts.competition
        FROM (
            SELECT
                NULLIF(btrim(league_key), '') AS league_key,
                season,
                upper(btrim(competition)) AS competition
            FROM "AI_Source"."PlayerCompetitionStats"
            WHERE season = p_season

            UNION

            SELECT
                NULLIF(btrim(league_key), '') AS league_key,
                season,
                upper(btrim(competition)) AS competition
            FROM "AI_Source"."TeamCompetitionStats"
            WHERE season = p_season
        ) AS contexts
        WHERE contexts.league_key IS NOT NULL
          AND contexts.competition IS NOT NULL
        ORDER BY contexts.league_key, contexts.season, contexts.competition
    LOOP
        CALL "AI_Source"."RefreshTrainingServing"(
            ctx.league_key, ctx.season, ctx.competition
        );
        CALL "AI_Source"."RefreshScenarioServing"(
            ctx.league_key, ctx.season, ctx.competition
        );

        INSERT INTO "AI_Source"."ServingRefreshState" (
            "LeagueKey", "Season", "Competition", "RefreshedAt",
            "PlayerRows", "TeamRows", "PlayTypeRows", "ShotProfileRows",
            "DefenderMatchupRows", "LineupRows"
        )
        SELECT ctx.league_key, ctx.season, ctx.competition, now(),
            (SELECT count(*) FROM "AI_Source"."TrainingPlayerCompetitionStats"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition),
            (SELECT count(*) FROM "AI_Source"."TrainingTeamCompetitionStats"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition),
            (SELECT count(*) FROM "AI_Source"."ScenarioPlayTypeStats"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition),
            (SELECT count(*) FROM "AI_Source"."ScenarioShotProfiles"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition),
            (SELECT count(*) FROM "AI_Source"."ScenarioDefenderMatchups"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition),
            (SELECT count(*) FROM "AI_Source"."ScenarioLineupStats"
              WHERE league_key = ctx.league_key AND season = ctx.season
                AND competition = ctx.competition)
        ON CONFLICT ("LeagueKey", "Season", "Competition") DO UPDATE SET
            "RefreshedAt" = EXCLUDED."RefreshedAt",
            "PlayerRows" = EXCLUDED."PlayerRows",
            "TeamRows" = EXCLUDED."TeamRows",
            "PlayTypeRows" = EXCLUDED."PlayTypeRows",
            "ShotProfileRows" = EXCLUDED."ShotProfileRows",
            "DefenderMatchupRows" = EXCLUDED."DefenderMatchupRows",
            "LineupRows" = EXCLUDED."LineupRows";

        processed_contexts := processed_contexts + 1;
    END LOOP;

    IF processed_contexts = 0 THEN
        RAISE NOTICE 'No AI_Source contexts found for season %', p_season;
    ELSE
        RAISE NOTICE 'Refreshed % AI_Source contexts for season %',
            processed_contexts, p_season;
    END IF;
END;
$refresh_season$;

-- Remove legacy overloads when this SQL is reapplied to an existing database.
DROP PROCEDURE IF EXISTS "AI_Source"."RefreshContext"(text, integer, text);
DROP PROCEDURE IF EXISTS "AI_Source"."RefreshContext"(integer);

-- Administrative/runtime read access is re-applied on every serving rebuild.
DO $grant_ai_source_serving$
DECLARE
    requested_role text := NULLIF(
        btrim(current_setting('app.ai_source_read_role', true)),
        ''
    );
    read_role text;
BEGIN
    FOREACH read_role IN ARRAY ARRAY[
        'ai_model',
        'hm_admin',
        'hm_local_workers',
        requested_role
    ]
    LOOP
        CONTINUE WHEN read_role IS NULL OR btrim(read_role) = '';

        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_roles
            WHERE rolname = read_role
        ) THEN
            EXECUTE format(
                'GRANT USAGE ON SCHEMA "AI_Source" TO %I',
                read_role
            );
            EXECUTE format(
                'GRANT SELECT ON TABLE
                    "AI_Source"."TrainingPlayerCompetitionStats",
                    "AI_Source"."TrainingTeamCompetitionStats",
                    "AI_Source"."ScenarioPlayTypeStats",
                    "AI_Source"."ScenarioShotProfiles",
                    "AI_Source"."ScenarioDefenderMatchups",
                    "AI_Source"."ScenarioLineupStats",
                    "AI_Source"."SimulationPbpEvents",
                    "AI_Source"."SimulationLineupStints",
                    "AI_Source"."SimulationPlayTypeStats",
                    "AI_Source"."SimulationShotProfiles",
                    "AI_Source"."SimulationCausalPanel",
                    "AI_Source"."SimulationSourceRefreshState",
                    "AI_Source"."ServingRefreshState"
                 TO %I',
                read_role
            );
            EXECUTE format(
                'GRANT EXECUTE ON PROCEDURE "AI_Source"."RefreshSimulationSources"(text, integer),
                                         "AI_Source"."RefreshScenarioServing"(text, integer, text),
                                         "AI_Source"."RefreshTrainingServing"(text, integer, text),
                                         "AI_Source"."RefreshContextSeasonal"(integer)
                 TO %I',
                read_role
            );
            RAISE NOTICE
                'AI_Source serving read/execute access granted to role %',
                read_role;
        END IF;
    END LOOP;
END;
$grant_ai_source_serving$;



-- Persistent scheduler for repeatable AI_Model jobs. One-time schema/diagnostic
-- scripts are deliberately excluded; only idempotent refresh procedures and
-- explicit model lifecycle tasks are schedulable.
CREATE TABLE IF NOT EXISTS "AI_Source"."AIModelScheduledJob" (
    job_id bigserial PRIMARY KEY,
    name text NOT NULL,
    task_key text NOT NULL CHECK (task_key IN (
        'refresh_context_seasonal',
        'refresh_simulation_sources',
        'refresh_scenario_serving',
        'refresh_training_serving',
        'bb_prediction_training',
        'bb_rating_calibration',
        'future_performance_training'
    )),
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    interval_minutes integer NOT NULL CHECK (interval_minutes >= 60),
    enabled boolean NOT NULL DEFAULT true,
    next_run_at timestamptz NOT NULL,
    last_run_at timestamptz,
    last_status text NOT NULL DEFAULT 'scheduled',
    last_error text,
    created_by text NOT NULL DEFAULT 'admin',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_ai_model_scheduled_job_due
    ON "AI_Source"."AIModelScheduledJob" (enabled, next_run_at);

CREATE TABLE IF NOT EXISTS "AI_Source"."AIModelJobRun" (
    run_id bigserial PRIMARY KEY,
    job_id bigint REFERENCES "AI_Source"."AIModelScheduledJob"(job_id) ON DELETE SET NULL,
    task_key text NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL CHECK (status IN ('running','complete','submitted','failed')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    result jsonb,
    error text
);
CREATE INDEX IF NOT EXISTS ix_ai_model_job_run_started
    ON "AI_Source"."AIModelJobRun" (started_at DESC);

COMMIT;

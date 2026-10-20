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

    -- Simulation feeds are optional external inputs. ai_source_full.sql must
    -- not drop them, but a database without a feed must still be refreshable.
    -- Clear stale derived rows first, then copy only when each source exists.
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
AS $$
DECLARE normalized_competition text := upper(btrim(p_competition));
BEGIN
    IF p_league_key IS NULL OR p_season IS NULL OR normalized_competition = '' THEN
        RAISE EXCEPTION 'league_key, season and competition are required';
    END IF;
    DELETE FROM "AI_Source"."TrainingPlayerCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND competition = normalized_competition;
    INSERT INTO "AI_Source"."TrainingPlayerCompetitionStats"
    SELECT * FROM "AI_Source"."PlayerCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM "AI_Source"."TrainingTeamCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND competition = normalized_competition;
    INSERT INTO "AI_Source"."TrainingTeamCompetitionStats"
    SELECT * FROM "AI_Source"."TeamCompetitionStats"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;
END;
$$;

-- Canonical ETL entry point: refresh every league and every competition for a
-- season from the current canonical AI_Source statistical tables. Callers use
-- RefreshContextSeasonal; there is no public single-context RefreshContext overload.
CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshContextSeasonal"(p_season integer)
LANGUAGE plpgsql
AS $refresh_season$
DECLARE
    ctx record;
    processed_contexts integer := 0;
BEGIN
    IF p_season IS NULL THEN
        RAISE EXCEPTION 'season is required';
    END IF;

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
                    "AI_Source"."ServingRefreshState"
                 TO %I',
                read_role
            );
            EXECUTE format(
                'GRANT EXECUTE ON PROCEDURE "AI_Source"."RefreshScenarioServing"(text, integer, text),
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

COMMIT;

-- Scalable scenario-serving layer.
-- Run after ai_source_full.sql, then refresh only the contexts changed by ETL:
--   CALL "AI_Source"."RefreshScenarioServing"('ITA1', 2025, 'RS');

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
BEGIN
    IF p_league_key IS NULL OR p_season IS NULL OR normalized_competition = '' THEN
        RAISE EXCEPTION 'league_key, season and competition are required';
    END IF;

    DELETE FROM "AI_Source"."ScenarioPlayTypeStats"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO "AI_Source"."ScenarioPlayTypeStats"
    SELECT player_id, team_id, opponent_team_id, league_key, season, upper(competition),
           play_type, possessions, ppp, ppp_allowed
      FROM "AI_Source"."SimulationPlayTypeStats"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM "AI_Source"."ScenarioShotProfiles"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO "AI_Source"."ScenarioShotProfiles"
    SELECT player_id, team_id, league_key, season, upper(competition), zone, attempts, fg_pct
      FROM "AI_Source"."SimulationShotProfiles"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM "AI_Source"."ScenarioDefenderMatchups"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO "AI_Source"."ScenarioDefenderMatchups"
    SELECT offensive_player_id, defender_id, team_id, opponent_team_id,
           league_key, season, upper(competition), count(*)::double precision,
           avg(assignment_probability), sum(points),
           sum(CASE WHEN turnover THEN 1 ELSE 0 END)::double precision
      FROM "AI_Source"."SimulationPbpEvents"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition AND defender_id IS NOT NULL
     GROUP BY offensive_player_id, defender_id, team_id, opponent_team_id,
              league_key, season, upper(competition);

    DELETE FROM "AI_Source"."ScenarioLineupStats"
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO "AI_Source"."ScenarioLineupStats"
    SELECT player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
           league_key, season, upper(competition),
           avg(assignment_probability), sum(possessions), sum(points), sum(points_allowed),
           sum(turnovers_forced),
           sum(ortg * possessions) / NULLIF(sum(possessions), 0),
           sum(drtg * possessions) / NULLIF(sum(possessions), 0),
           sum(net_rtg * possessions) / NULLIF(sum(possessions), 0)
      FROM "AI_Source"."SimulationLineupStints"
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition
     GROUP BY player_ids, team_id, opponent_team_id, offense_player_id,
              defense_player_id, league_key, season, upper(competition);
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

-- Single ETL hook. Call it after HoopmetricsEngine/AdvanceStats has committed
-- every changed league-season-competition context, including same-day updates.
CREATE OR REPLACE PROCEDURE "AI_Source"."RefreshContext"(
    p_league_key text,
    p_season integer,
    p_competition text
)
LANGUAGE plpgsql
AS $$
DECLARE normalized_competition text := upper(btrim(p_competition));
BEGIN
    CALL "AI_Source"."RefreshTrainingServing"(
        p_league_key, p_season, normalized_competition
    );
    CALL "AI_Source"."RefreshScenarioServing"(
        p_league_key, p_season, normalized_competition
    );

    INSERT INTO "AI_Source"."ServingRefreshState" (
        "LeagueKey", "Season", "Competition", "RefreshedAt",
        "PlayerRows", "TeamRows", "PlayTypeRows", "ShotProfileRows",
        "DefenderMatchupRows", "LineupRows"
    )
    SELECT p_league_key, p_season, normalized_competition, now(),
        (SELECT count(*) FROM "AI_Source"."TrainingPlayerCompetitionStats"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition),
        (SELECT count(*) FROM "AI_Source"."TrainingTeamCompetitionStats"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition),
        (SELECT count(*) FROM "AI_Source"."ScenarioPlayTypeStats"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition),
        (SELECT count(*) FROM "AI_Source"."ScenarioShotProfiles"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition),
        (SELECT count(*) FROM "AI_Source"."ScenarioDefenderMatchups"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition),
        (SELECT count(*) FROM "AI_Source"."ScenarioLineupStats"
          WHERE league_key = p_league_key AND season = p_season
            AND competition = normalized_competition)
    ON CONFLICT ("LeagueKey", "Season", "Competition") DO UPDATE SET
        "RefreshedAt" = EXCLUDED."RefreshedAt",
        "PlayerRows" = EXCLUDED."PlayerRows",
        "TeamRows" = EXCLUDED."TeamRows",
        "PlayTypeRows" = EXCLUDED."PlayTypeRows",
        "ShotProfileRows" = EXCLUDED."ShotProfileRows",
        "DefenderMatchupRows" = EXCLUDED."DefenderMatchupRows",
        "LineupRows" = EXCLUDED."LineupRows";
END;
$$;

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
                                         "AI_Source"."RefreshContext"(text, integer, text)
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

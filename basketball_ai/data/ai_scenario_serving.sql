-- Scalable scenario-serving layer.
-- Run after ai_source_simulation.sql, then refresh only the contexts changed by ETL:
--   CALL ai_source.refresh_scenario_serving('ITA1', 2025, 'RS');

BEGIN;

-- Physical training feature store. This prevents every snapshot build from
-- recomputing the canonical views over the complete boxscore history.
CREATE TABLE IF NOT EXISTS ai_source.training_player_competition_stats
    AS SELECT * FROM ai_source.player_competition_stats WITH NO DATA;
CREATE TABLE IF NOT EXISTS ai_source.training_team_competition_stats
    AS SELECT * FROM ai_source.team_competition_stats WITH NO DATA;
CREATE UNIQUE INDEX IF NOT EXISTS ux_training_player_context
    ON ai_source.training_player_competition_stats
       (player_id, league_id, season, competition);
CREATE UNIQUE INDEX IF NOT EXISTS ux_training_team_context
    ON ai_source.training_team_competition_stats
       (team_id, league_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_training_player_partition
    ON ai_source.training_player_competition_stats
       (league_key, season, competition);
CREATE INDEX IF NOT EXISTS ix_training_team_partition
    ON ai_source.training_team_competition_stats
       (league_key, season, competition);

CREATE TABLE IF NOT EXISTS ai_source.scenario_play_type_stats (
    player_id bigint, team_id bigint, opponent_team_id bigint,
    league_key text NOT NULL, season integer NOT NULL, competition text NOT NULL,
    play_type text NOT NULL, possessions double precision NOT NULL,
    ppp double precision, ppp_allowed double precision
);

CREATE TABLE IF NOT EXISTS ai_source.scenario_shot_profiles (
    player_id bigint NOT NULL, team_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL, zone text NOT NULL,
    attempts double precision NOT NULL, fg_pct double precision
);

CREATE TABLE IF NOT EXISTS ai_source.scenario_defender_matchups (
    offensive_player_id bigint NOT NULL, defender_id bigint NOT NULL,
    team_id bigint, opponent_team_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL,
    possessions double precision NOT NULL, assignment_probability double precision,
    points double precision, turnovers_forced double precision
);

CREATE TABLE IF NOT EXISTS ai_source.scenario_lineup_stats (
    player_ids text NOT NULL, team_id bigint, opponent_team_id bigint,
    offense_player_id bigint, defense_player_id bigint, league_key text NOT NULL,
    season integer NOT NULL, competition text NOT NULL,
    assignment_probability double precision, possessions double precision NOT NULL,
    points double precision, points_allowed double precision,
    turnovers_forced double precision, ortg double precision, drtg double precision,
    net_rtg double precision
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_play_type_context
    ON ai_source.scenario_play_type_stats (
        league_key, season, competition,
        COALESCE(player_id, -1), COALESCE(team_id, -1),
        COALESCE(opponent_team_id, -1), play_type
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_shot_context
    ON ai_source.scenario_shot_profiles (
        league_key, season, competition, player_id, COALESCE(team_id, -1), zone
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_matchup_context
    ON ai_source.scenario_defender_matchups (
        league_key, season, competition, offensive_player_id, defender_id,
        COALESCE(team_id, -1), COALESCE(opponent_team_id, -1)
    );
CREATE UNIQUE INDEX IF NOT EXISTS ux_scenario_lineup_context
    ON ai_source.scenario_lineup_stats (
        league_key, season, competition, player_ids,
        COALESCE(team_id, -1), COALESCE(opponent_team_id, -1),
        COALESCE(offense_player_id, -1), COALESCE(defense_player_id, -1)
    );

CREATE INDEX IF NOT EXISTS ix_scenario_play_type_player
    ON ai_source.scenario_play_type_stats (player_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_play_type_team
    ON ai_source.scenario_play_type_stats (team_id, opponent_team_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_shot_player
    ON ai_source.scenario_shot_profiles (player_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_matchup_pair
    ON ai_source.scenario_defender_matchups (offensive_player_id, defender_id, season, competition);
CREATE INDEX IF NOT EXISTS ix_scenario_lineup_team
    ON ai_source.scenario_lineup_stats (team_id, season, competition);

CREATE OR REPLACE PROCEDURE ai_source.refresh_scenario_serving(
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

    DELETE FROM ai_source.scenario_play_type_stats
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO ai_source.scenario_play_type_stats
    SELECT player_id, team_id, opponent_team_id, league_key, season, upper(competition),
           play_type, possessions, ppp, ppp_allowed
      FROM ai_source.simulation_play_type_stats
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM ai_source.scenario_shot_profiles
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO ai_source.scenario_shot_profiles
    SELECT player_id, team_id, league_key, season, upper(competition), zone, attempts, fg_pct
      FROM ai_source.simulation_shot_profiles
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM ai_source.scenario_defender_matchups
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO ai_source.scenario_defender_matchups
    SELECT offensive_player_id, defender_id, team_id, opponent_team_id,
           league_key, season, upper(competition), count(*)::double precision,
           avg(assignment_probability), sum(points),
           sum(CASE WHEN turnover THEN 1 ELSE 0 END)::double precision
      FROM ai_source.simulation_pbp_events
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition AND defender_id IS NOT NULL
     GROUP BY offensive_player_id, defender_id, team_id, opponent_team_id,
              league_key, season, upper(competition);

    DELETE FROM ai_source.scenario_lineup_stats
     WHERE league_key = p_league_key AND season = p_season AND competition = normalized_competition;
    INSERT INTO ai_source.scenario_lineup_stats
    SELECT player_ids, team_id, opponent_team_id, offense_player_id, defense_player_id,
           league_key, season, upper(competition),
           avg(assignment_probability), sum(possessions), sum(points), sum(points_allowed),
           sum(turnovers_forced),
           sum(ortg * possessions) / NULLIF(sum(possessions), 0),
           sum(drtg * possessions) / NULLIF(sum(possessions), 0),
           sum(net_rtg * possessions) / NULLIF(sum(possessions), 0)
      FROM ai_source.simulation_lineup_stints
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition
     GROUP BY player_ids, team_id, opponent_team_id, offense_player_id,
              defense_player_id, league_key, season, upper(competition);
END;
$$;

CREATE OR REPLACE PROCEDURE ai_source.refresh_training_serving(
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
    DELETE FROM ai_source.training_player_competition_stats
     WHERE league_key = p_league_key AND season = p_season
       AND competition = normalized_competition;
    INSERT INTO ai_source.training_player_competition_stats
    SELECT * FROM ai_source.player_competition_stats
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;

    DELETE FROM ai_source.training_team_competition_stats
     WHERE league_key = p_league_key AND season = p_season
       AND competition = normalized_competition;
    INSERT INTO ai_source.training_team_competition_stats
    SELECT * FROM ai_source.team_competition_stats
     WHERE league_key = p_league_key AND season = p_season
       AND upper(competition) = normalized_competition;
END;
$$;

COMMIT;

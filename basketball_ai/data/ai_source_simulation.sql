-- Optional possession-level source contract for the Simulation & Causal Engine.
-- Run after ai_source_schema.sql and ai_source_competition.sql.
-- Physical source tables are read-only; missing feeds produce empty views.

BEGIN;

DROP VIEW IF EXISTS "AI_Source"."SimulationCausalPanel";
DROP VIEW IF EXISTS "AI_Source"."SimulationShotProfiles";
DROP VIEW IF EXISTS "AI_Source"."SimulationPlayTypeStats";
DROP VIEW IF EXISTS "AI_Source"."SimulationLineupStints";
DROP VIEW IF EXISTS "AI_Source"."SimulationPbpEvents";
DROP VIEW IF EXISTS "AI_Source"."SimulationLineupRawInternal";
DROP VIEW IF EXISTS "AI_Source"."SimulationPbpRawInternal";

CREATE OR REPLACE FUNCTION "AI_Source"."SimulationPlayerId"(
    requested_league text, requested_season integer, requested_source_id text
)
RETURNS bigint
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT "AI_Source"."StableId"('player', r.global_id)
    FROM "AI_Source"."CompetitionPlayerRegistryInternal" r
    WHERE lower(r.league_key) = lower(requested_league)
      AND r.season = requested_season
      AND btrim(r.source_player_id) = btrim(requested_source_id)
    LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."SimulationTeamId"(
    requested_league text, requested_season integer, requested_source_id text
)
RETURNS bigint
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT "AI_Source"."StableId"('team', r.global_id)
    FROM "AI_Source"."CompetitionTeamRegistryInternal" r
    WHERE lower(r.league_key) = lower(requested_league)
      AND r.season = requested_season
      AND btrim(r.source_team_id) = btrim(requested_source_id)
    LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."SimulationLineupIds"(
    requested_league text, requested_season integer, source_lineup text
)
RETURNS text
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT string_agg("AI_Source"."SimulationPlayerId"(
        requested_league, requested_season, btrim(piece)
    )::text, ',' ORDER BY "AI_Source"."SimulationPlayerId"(
        requested_league, requested_season, btrim(piece)
    ))
    FROM regexp_split_to_table(COALESCE(source_lineup, ''), '\s*-\s*|\s*,\s*') piece
    WHERE btrim(piece) <> ''
      AND "AI_Source"."SimulationPlayerId"(
          requested_league, requested_season, btrim(piece)
      ) IS NOT NULL;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."SimulationPlayType"(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE
        WHEN lower(COALESCE(value, '')) ~ 'pick.*roll.*ball|p.?r.*handler' THEN 'pick_and_roll_ball_handler'
        WHEN lower(COALESCE(value, '')) ~ 'pick.*roll.*roll|roll.?man' THEN 'pick_and_roll_roll_man'
        WHEN lower(COALESCE(value, '')) ~ 'isolat' THEN 'isolation'
        WHEN lower(COALESCE(value, '')) ~ 'post.?up' THEN 'post_up'
        WHEN lower(COALESCE(value, '')) ~ 'spot.?up' THEN 'spot_up'
        WHEN lower(COALESCE(value, '')) ~ 'handoff' THEN 'handoff'
        WHEN lower(COALESCE(value, '')) ~ 'cut' THEN 'cut'
        WHEN lower(COALESCE(value, '')) ~ 'off.?screen' THEN 'off_screen'
        WHEN lower(COALESCE(value, '')) ~ 'transition' THEN 'transition'
        WHEN lower(COALESCE(value, '')) ~ 'putback' THEN 'putback'
        ELSE NULL
    END;
$$;

-- Event feeds (Pbp.<LEAGUE>).
DO $$
DECLARE r record; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'pbp'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) = 'player')
        ORDER BY c.table_name
    LOOP
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
                   "AI_Source"."TextValue"(j.doc, 'gamecode', 'game', 'idgame') AS game_id,
                   "AI_Source"."IntValue"(j.doc, 'quarter', 'period')::integer AS period,
                   "AI_Source"."TextValue"(j.doc, 'player', 'playerid', 'idplayer') AS source_player_id,
                   "AI_Source"."TextValue"(j.doc, 'teamid', 'idteam') AS source_team_id,
                   "AI_Source"."TextValue"(j.doc, 'opponentteamid', 'oppteamid') AS source_opponent_team_id,
                   "AI_Source"."TextValue"(j.doc, 'defenderid', 'defensiveplayerid', 'matcheddefenderid') AS source_defender_id,
                   "AI_Source"."NumericValue"(j.doc, 'assignmentprobability', 'matchupprobability') AS assignment_probability,
                   "AI_Source"."TextValue"(j.doc, 'synergyplaytype', 'possessiontype', 'actiontype') AS tactical_play_type,
                   "AI_Source"."TextValue"(j.doc, 'playtype', 'eventtype') AS event_type,
                   "AI_Source"."NumericValue"(j.doc, 'points', 'pts') AS explicit_points,
                   "AI_Source"."TextValue"(j.doc, 'side_area_zone', 'shotzone', 'zone') AS shot_zone,
                   "AI_Source"."NumericValue"(j.doc, 'x') AS x,
                   "AI_Source"."NumericValue"(j.doc, 'y') AS y
            FROM %I.%I src
            CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
            WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
              AND "AI_Source"."TextValue"(j.doc, 'player', 'playerid', 'idplayer') IS NOT NULL
        $sql$, r.table_name, r.table_schema, r.table_name);
    END LOOP;
    IF body = '' THEN
        body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text competition, NULL::text game_id, NULL::integer period, NULL::text source_player_id, NULL::text source_team_id, NULL::text source_opponent_team_id, NULL::text source_defender_id, NULL::double precision assignment_probability, NULL::text tactical_play_type, NULL::text event_type, NULL::double precision explicit_points, NULL::text shot_zone, NULL::double precision x, NULL::double precision y WHERE false';
    END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."SimulationPbpRawInternal" AS ' || body;
END;
$$;

CREATE VIEW "AI_Source"."SimulationPbpEvents" AS
SELECT
    p.league_key,
    p.season,
    p.competition,
    p.game_id,
    p.period,
    "AI_Source"."SimulationPlayerId"(p.league_key, p.season, p.source_player_id) AS offensive_player_id,
    "AI_Source"."SimulationTeamId"(p.league_key, p.season, p.source_team_id) AS team_id,
    "AI_Source"."SimulationTeamId"(p.league_key, p.season, p.source_opponent_team_id) AS opponent_team_id,
    "AI_Source"."SimulationPlayerId"(p.league_key, p.season, p.source_defender_id) AS defender_id,
    LEAST(1.0, GREATEST(0.0, COALESCE(p.assignment_probability, CASE WHEN p.source_defender_id IS NULL THEN 0 ELSE 1 END))) AS assignment_probability,
    "AI_Source"."SimulationPlayType"(p.tactical_play_type) AS play_type,
    COALESCE(
        p.explicit_points,
        CASE WHEN upper(p.event_type) LIKE '%3FGM%' THEN 3
             WHEN upper(p.event_type) LIKE '%2FGM%' THEN 2 ELSE 0 END
    ) AS points,
    CASE WHEN upper(COALESCE(p.event_type, '')) ~ '(^|[^A-Z])(TO|TURNOVER)([^A-Z]|$)' THEN true ELSE false END AS turnover,
    p.event_type,
    p.shot_zone,
    p.x,
    p.y
FROM "AI_Source"."SimulationPbpRawInternal" p
WHERE "AI_Source"."SimulationPlayerId"(p.league_key, p.season, p.source_player_id) IS NOT NULL;

-- Five-player stint feeds (Analisi.AdvancedStats_Lineups_Quarter_<LEAGUE>).
DO $$
DECLARE r record; league_key text; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(c.table_name) LIKE 'advancedstats_lineups_quarter_%'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) = 'lineup')
        ORDER BY c.table_name
    LOOP
        league_key := regexp_replace(r.table_name, '^AdvancedStats_Lineups_Quarter_', '', 'i');
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
                   "AI_Source"."TextValue"(j.doc, 'lineup') AS source_lineup,
                   "AI_Source"."TextValue"(j.doc, 'teamid', 'idteam') AS source_team_id,
                   "AI_Source"."TextValue"(j.doc, 'opponentteamid', 'oppteamid') AS source_opponent_team_id,
                   "AI_Source"."TextValue"(j.doc, 'offensiveplayerid', 'offenseplayerid') AS source_offense_player_id,
                   "AI_Source"."TextValue"(j.doc, 'defensiveplayerid', 'defenderid') AS source_defense_player_id,
                   "AI_Source"."NumericValue"(j.doc, 'assignmentprobability', 'matchupprobability') AS assignment_probability,
                   "AI_Source"."NumericValue"(j.doc, 'offposs', 'possessions', 'poss') AS off_possessions,
                   "AI_Source"."NumericValue"(j.doc, 'defposs') AS def_possessions,
                   "AI_Source"."NumericValue"(j.doc, 'pts', 'points') AS points,
                   "AI_Source"."NumericValue"(j.doc, 'ptsallowed', 'pointsallowed') AS points_allowed,
                   "AI_Source"."NumericValue"(j.doc, 'opptov', 'turnoversforced') AS turnovers_forced,
                   "AI_Source"."NumericValue"(j.doc, 'ortg') AS ortg,
                   "AI_Source"."NumericValue"(j.doc, 'drtg') AS drtg,
                   "AI_Source"."NumericValue"(j.doc, 'netrtg') AS net_rtg
            FROM %I.%I src
            CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
            WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
              AND "AI_Source"."TextValue"(j.doc, 'lineup') IS NOT NULL
        $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;
    IF body = '' THEN
        body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text competition, NULL::text source_lineup, NULL::text source_team_id, NULL::text source_opponent_team_id, NULL::text source_offense_player_id, NULL::text source_defense_player_id, NULL::double precision assignment_probability, NULL::double precision off_possessions, NULL::double precision def_possessions, NULL::double precision points, NULL::double precision points_allowed, NULL::double precision turnovers_forced, NULL::double precision ortg, NULL::double precision drtg, NULL::double precision net_rtg WHERE false';
    END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."SimulationLineupRawInternal" AS ' || body;
END;
$$;

CREATE VIEW "AI_Source"."SimulationLineupStints" AS
SELECT
    l.league_key,
    l.season,
    l.competition,
    "AI_Source"."SimulationLineupIds"(l.league_key, l.season, l.source_lineup) AS player_ids,
    "AI_Source"."SimulationTeamId"(l.league_key, l.season, l.source_team_id) AS team_id,
    "AI_Source"."SimulationTeamId"(l.league_key, l.season, l.source_opponent_team_id) AS opponent_team_id,
    "AI_Source"."SimulationPlayerId"(l.league_key, l.season, l.source_offense_player_id) AS offense_player_id,
    "AI_Source"."SimulationPlayerId"(l.league_key, l.season, l.source_defense_player_id) AS defense_player_id,
    LEAST(1.0, GREATEST(0.0, COALESCE(l.assignment_probability, 0.0))) AS assignment_probability,
    GREATEST(COALESCE(l.off_possessions, 0), COALESCE(l.def_possessions, 0)) AS possessions,
    l.points,
    l.points_allowed,
    l.turnovers_forced,
    l.ortg,
    l.drtg,
    COALESCE(l.net_rtg, l.ortg - l.drtg) AS net_rtg
FROM "AI_Source"."SimulationLineupRawInternal" l
WHERE "AI_Source"."SimulationLineupIds"(l.league_key, l.season, l.source_lineup) IS NOT NULL;

CREATE VIEW "AI_Source"."SimulationPlayTypeStats" AS
SELECT
    offensive_player_id AS player_id,
    team_id,
    opponent_team_id,
    league_key,
    season,
    competition,
    play_type,
    count(*)::double precision AS possessions,
    sum(points) / NULLIF(count(*), 0)::double precision AS ppp,
    NULL::double precision AS ppp_allowed
FROM "AI_Source"."SimulationPbpEvents"
WHERE play_type IS NOT NULL
GROUP BY offensive_player_id, team_id, opponent_team_id, league_key, season, competition, play_type;

CREATE VIEW "AI_Source"."SimulationShotProfiles" AS
SELECT
    offensive_player_id AS player_id,
    team_id,
    league_key,
    season,
    competition,
    lower(regexp_replace(COALESCE(NULLIF(shot_zone, ''), 'unknown'), '[^a-zA-Z0-9]+', '_', 'g')) AS zone,
    count(*)::double precision AS attempts,
    avg(CASE WHEN points IN (2, 3) THEN 1.0 ELSE 0.0 END) AS fg_pct
FROM "AI_Source"."SimulationPbpEvents"
WHERE upper(COALESCE(event_type, '')) ~ '(2FG|3FG)'
GROUP BY offensive_player_id, team_id, league_key, season, competition,
         lower(regexp_replace(COALESCE(NULLIF(shot_zone, ''), 'unknown'), '[^a-zA-Z0-9]+', '_', 'g'));

-- Explicit causal panels are supplied by a separately governed ETL. The empty
-- view keeps deployment additive and prevents accidental causal claims.
CREATE VIEW "AI_Source"."SimulationCausalPanel" AS
SELECT NULL::double precision AS treatment,
       NULL::double precision AS next_outcome
WHERE false;

COMMIT;

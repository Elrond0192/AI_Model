-- BBallstat AI_Model canonical PostgreSQL source contract
--
-- Idempotent adapter from the current unified BBallstat PostgreSQL schemas to
-- the five read-only views consumed by AI_Model.  Source tables are NEVER
-- modified. Re-run this file after adding a league or changing source schema.
--
-- Discovered physical tables:
--   Anagrafiche.<LEAGUE>
--   Anagrafiche.Team_<LEAGUE>
--   Analisi.AdvancedStats_Player_<LEAGUE>
--   Analisi.AdvancedStatsTeam_<LEAGUE>                 optional
--   Analisi.PlayerRoles_<LEAGUE>                       optional
--   Analisi.AdvancedStatsOnOffCourt_<LEAGUE>           optional
--   Analisi.AdvancedStats_Clutch_<LEAGUE>              optional
--   Boxscore.<LEAGUE>                                   optional
--
-- Requirements:
--   * unified tables contain numeric Season
--   * IdGlobal is preferred for cross-season identity; a league-local fallback
--     is generated only when IdGlobal is missing
--   * PostgreSQL 12+ (hashtextextended, JSONB)

BEGIN;

CREATE SCHEMA IF NOT EXISTS "AI_Source";

-- -------------------------------------------------------------------------
-- Helpers. to_jsonb(row) + lower-case keys makes the adapter tolerant to
-- PostgreSQL tables that preserved historical SQL Server column casing.
-- -------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION "AI_Source"."LowerKeys"(doc jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE(jsonb_object_agg(lower(key), value), '{}'::jsonb)
    FROM jsonb_each(doc);
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."TextValue"(doc jsonb, VARIADIC candidates text[])
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    candidate text;
    value text;
BEGIN
    FOREACH candidate IN ARRAY candidates LOOP
        value := doc ->> lower(candidate);
        IF value IS NOT NULL AND btrim(value) <> '' THEN
            RETURN btrim(value);
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."NumericValue"(doc jsonb, VARIADIC candidates text[])
RETURNS double precision
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    value text;
BEGIN
    value := "AI_Source"."TextValue"(doc, VARIADIC candidates);
    IF value IS NULL THEN
        RETURN NULL;
    END IF;
    value := replace(replace(value, '%', ''), ',', '.');
    IF value ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$' THEN
        RETURN value::double precision;
    END IF;
    RETURN NULL;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."IntValue"(doc jsonb, VARIADIC candidates text[])
RETURNS bigint
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    value double precision;
BEGIN
    value := "AI_Source"."NumericValue"(doc, VARIADIC candidates);
    IF value IS NULL
       OR value > 9223372036854775807::double precision
       OR value < -9223372036854775808::double precision THEN
        RETURN NULL;
    END IF;
    RETURN trunc(value)::bigint;
END;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."DateValue"(doc jsonb, VARIADIC candidates text[])
RETURNS date
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    value text;
BEGIN
    value := "AI_Source"."TextValue"(doc, VARIADIC candidates);
    IF value IS NULL THEN
        RETURN NULL;
    END IF;

    BEGIN
        IF value ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN
            RETURN substr(value, 1, 10)::date;
        ELSIF value ~ '^[0-9]{2}/[0-9]{2}/[0-9]{4}$' THEN
            RETURN to_date(value, 'DD/MM/YYYY');
        ELSIF value ~ '^[0-9]{2}-[0-9]{2}-[0-9]{4}$' THEN
            RETURN to_date(value, 'DD-MM-YYYY');
        ELSIF value ~ '^[0-9]{4}/[0-9]{2}/[0-9]{2}$' THEN
            RETURN to_date(value, 'YYYY/MM/DD');
        END IF;
    EXCEPTION WHEN OTHERS THEN
        RETURN NULL;
    END;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."Truthy"(value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT lower(btrim(COALESCE(value, ''))) IN ('1', 't', 'true', 'y', 'yes');
$$;

CREATE OR REPLACE FUNCTION "AI_Source"."StableId"(namespace text, value text)
RETURNS bigint
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE
        WHEN value IS NULL OR btrim(value) = '' THEN NULL
        ELSE hashtextextended(COALESCE(namespace, '') || ':' || btrim(value), 0)
             & 9223372036854775807::bigint
    END;
$$;

-- -------------------------------------------------------------------------
-- Drop adapter views only. CASCADE is intentionally avoided so accidental
-- external dependencies are visible instead of being silently destroyed.
-- -------------------------------------------------------------------------

-- Additive simulation views depend on the registries rebuilt below. Remove
-- only these known adapter objects first; ai_source_simulation.sql recreates
-- them after the competition contract is installed.
DROP VIEW IF EXISTS "AI_Source"."SimulationCausalPanel";
DROP VIEW IF EXISTS "AI_Source"."SimulationShotProfiles";
DROP VIEW IF EXISTS "AI_Source"."SimulationPlayTypeStats";
DROP VIEW IF EXISTS "AI_Source"."SimulationLineupStints";
DROP VIEW IF EXISTS "AI_Source"."SimulationPbpEvents";
DROP VIEW IF EXISTS "AI_Source"."SimulationLineupRawInternal";
DROP VIEW IF EXISTS "AI_Source"."SimulationPbpRawInternal";
DROP FUNCTION IF EXISTS "AI_Source"."SimulationLineupIds"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayerId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationTeamId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayType"(text);

DROP VIEW IF EXISTS "AI_Source"."Leagues";
DROP VIEW IF EXISTS "AI_Source"."PlayerStats";
DROP VIEW IF EXISTS "AI_Source"."Players";
DROP VIEW IF EXISTS "AI_Source"."Teams";
DROP VIEW IF EXISTS "AI_Source"."TeamPlayerRelations";
DROP VIEW IF EXISTS "AI_Source"."BoxscoreRawInternal";
DROP VIEW IF EXISTS "AI_Source"."ClutchRawInternal";
DROP VIEW IF EXISTS "AI_Source"."OnOffRawInternal";
DROP VIEW IF EXISTS "AI_Source"."RolesRawInternal";
DROP VIEW IF EXISTS "AI_Source"."TeamStatsRawInternal";
DROP VIEW IF EXISTS "AI_Source"."StatsRawInternal";
DROP VIEW IF EXISTS "AI_Source"."TeamRegistryInternal";
DROP VIEW IF EXISTS "AI_Source"."PlayerRegistryInternal";

-- -------------------------------------------------------------------------
-- Physical player registry
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'anagrafiche'
          AND lower(left(c.table_name, 5)) <> 'team_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) = 'id')
        ORDER BY c.table_name
    LOOP
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id') AS source_player_id,
                    COALESCE(
                        "AI_Source"."TextValue"(j.doc, 'idglobal', 'globalid'),
                        %L || ':' || "AI_Source"."TextValue"(j.doc, 'id')
                    ) AS global_id,
                    COALESCE(
                        "AI_Source"."TextValue"(j.doc, 'normalizedplayername', 'playername', 'name'),
                        "AI_Source"."TextValue"(j.doc, 'id')
                    ) AS name,
                    "AI_Source"."DateValue"(j.doc, 'birthdate', 'dateofbirth', 'dob') AS birth_date,
                    "AI_Source"."IntValue"(j.doc, 'age')::integer AS age,
                    COALESCE("AI_Source"."TextValue"(j.doc, 'pos', 'position'), 'PG') AS position,
                    COALESCE("AI_Source"."TextValue"(j.doc, 'nat', 'nationality'), '') AS nationality,
                    "AI_Source"."NumericValue"(j.doc, 'cm', 'heightcm', 'height_cm') AS height_cm,
                    "AI_Source"."NumericValue"(j.doc, 'weight', 'weightkg', 'weight_kg') AS weight_kg,
                    "AI_Source"."IntValue"(j.doc, 'shirtnumber', 'jerseynumber')::integer AS jersey_number,
                    "AI_Source"."TextValue"(j.doc, 'teamname') AS team_name
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc
                ) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id') IS NOT NULL
            $sql$, r.table_name, r.table_name, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS global_id,
                   NULL::text AS name, NULL::date AS birth_date,
                   NULL::integer AS age, NULL::text AS position,
                   NULL::text AS nationality, NULL::double precision AS height_cm,
                   NULL::double precision AS weight_kg, NULL::integer AS jersey_number,
                   NULL::text AS team_name
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."PlayerRegistryInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Physical team registry
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'anagrafiche'
          AND lower(left(c.table_name, 5)) = 'team_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) = 'id')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 6);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id') AS source_team_id,
                    COALESCE(
                        "AI_Source"."TextValue"(j.doc, 'idglobal', 'globalid'),
                        %L || ':' || "AI_Source"."TextValue"(j.doc, 'id')
                    ) AS global_id,
                    COALESCE(
                        "AI_Source"."TextValue"(j.doc, 'teamname', 'name', 'shortname'),
                        "AI_Source"."TextValue"(j.doc, 'id')
                    ) AS name,
                    "AI_Source"."TextValue"(j.doc, 'shortname', 'short_name') AS short_name
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc
                ) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id') IS NOT NULL
            $sql$, league_key, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_team_id, NULL::text AS global_id,
                   NULL::text AS name, NULL::text AS short_name
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."TeamRegistryInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Player season aggregates
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(left(c.table_name, 21)) = 'advancedstats_player_'
          AND lower(right(c.table_name, 5)) <> '_game'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) IN ('id', 'playerid', 'idplayer'))
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 22);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id', 'playerid', 'idplayer') AS source_player_id,
                    "AI_Source"."TextValue"(j.doc, 'teamid', 'idteam') AS source_team_id,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition,
                    "AI_Source"."NumericValue"(j.doc, 'games', 'gamesplayed') AS games,
                    "AI_Source"."NumericValue"(j.doc, 'min', 'minutes') AS minutes_total,
                    "AI_Source"."NumericValue"(j.doc, 'pts', 'points') AS points_total,
                    "AI_Source"."NumericValue"(j.doc, 'tr', 'reb', 'rebounds') AS rebounds_total,
                    "AI_Source"."NumericValue"(j.doc, 'or', 'orb', 'offensiverebounds') AS offensive_rebounds_total,
                    "AI_Source"."NumericValue"(j.doc, 'dr', 'drb', 'defensiverebounds') AS defensive_rebounds_total,
                    "AI_Source"."NumericValue"(j.doc, 'ast', 'assists') AS assists_total,
                    "AI_Source"."NumericValue"(j.doc, 'stl', 'steals') AS steals_total,
                    "AI_Source"."NumericValue"(j.doc, 'blk', 'blocks') AS blocks_total,
                    "AI_Source"."NumericValue"(j.doc, 'to', 'tov', 'turnovers') AS turnovers_total,
                    "AI_Source"."NumericValue"(j.doc, 'pf', 'fouls', 'personalfouls') AS fouls_total,
                    "AI_Source"."NumericValue"(j.doc, 'efgpct', 'fgpct', 'fg_pct') AS fg_pct,
                    "AI_Source"."NumericValue"(j.doc, 'fg3pct', 'threepointpct', 'three_point_pct') AS three_point_pct,
                    "AI_Source"."NumericValue"(j.doc, 'ftpct', 'ft_pct') AS ft_pct,
                    "AI_Source"."NumericValue"(j.doc, 'plusminus', 'plus_minus') AS plus_minus,
                    "AI_Source"."NumericValue"(j.doc, 'pie', 'per') AS per,
                    "AI_Source"."NumericValue"(j.doc, 'tspct', 'ts_pct') AS ts_pct,
                    "AI_Source"."NumericValue"(j.doc, 'usgpct', 'usg_pct') AS usg_pct,
                    "AI_Source"."NumericValue"(j.doc, 'bpm') AS bpm,
                    "AI_Source"."NumericValue"(j.doc, 'vorp') AS vorp,
                    "AI_Source"."NumericValue"(j.doc, 'ws', 'winshares') AS win_shares,
                    "AI_Source"."NumericValue"(j.doc, 'astratio', 'ast_ratio') AS ast_ratio,
                    "AI_Source"."NumericValue"(j.doc, 'rebpct', 'reb_pct') AS reb_pct,
                    "AI_Source"."NumericValue"(j.doc, 'vallegapergame', 'rating') AS rating,
                    "AI_Source"."NumericValue"(j.doc, 'ortg') AS ortg,
                    "AI_Source"."NumericValue"(j.doc, 'drtg') AS drtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg', 'net_rating') AS net_rtg,
                    "AI_Source"."TextValue"(j.doc, 'ruolooffensivo') AS ruolo_offensivo,
                    "AI_Source"."TextValue"(j.doc, 'ruolodifensivo') AS ruolo_difensivo,
                    "AI_Source"."TextValue"(j.doc, 'ruolocombinato') AS ruolo_combinato,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_on', 'onnetrtg') AS on_net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_off', 'offnetrtg') AS off_net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_diff') AS net_rtg_diff,
                    "AI_Source"."NumericValue"(j.doc, 'spm') AS spm,
                    "AI_Source"."NumericValue"(j.doc, 'obpm') AS obpm,
                    "AI_Source"."NumericValue"(j.doc, 'dbpm') AS dbpm,
                    "AI_Source"."NumericValue"(j.doc, 'gmsc', 'gm_sc') AS gm_sc,
                    "AI_Source"."NumericValue"(j.doc, 'fic') AS fic,
                    "AI_Source"."NumericValue"(j.doc, 'ows') AS ows,
                    "AI_Source"."NumericValue"(j.doc, 'dws') AS dws,
                    "AI_Source"."NumericValue"(j.doc, 'raptoroff', 'raptor_off') AS raptor_off,
                    "AI_Source"."NumericValue"(j.doc, 'raptordef', 'raptor_def') AS raptor_def,
                    "AI_Source"."NumericValue"(j.doc, 'raptortotal', 'raptor_total') AS raptor_total,
                    "AI_Source"."NumericValue"(j.doc, 'lebronoff', 'lebron_off') AS lebron_off,
                    "AI_Source"."NumericValue"(j.doc, 'lebrondef', 'lebron_def') AS lebron_def,
                    "AI_Source"."NumericValue"(j.doc, 'lebrontotal', 'lebron_total') AS lebron_total,
                    "AI_Source"."NumericValue"(j.doc, 'scoringefficiency', 'scoring_efficiency') AS scoring_efficiency,
                    "AI_Source"."NumericValue"(j.doc, 'ppsa') AS ppsa,
                    "AI_Source"."NumericValue"(j.doc, 'fg2pct', 'twopointpct') AS two_point_pct,
                    "AI_Source"."NumericValue"(j.doc, 'tovpct', 'tov_pct') AS tov_pct,
                    "AI_Source"."NumericValue"(j.doc, 'astpct', 'ast_pct') AS ast_pct,
                    "AI_Source"."NumericValue"(j.doc, 'stlpct', 'stl_pct') AS stl_pct,
                    "AI_Source"."NumericValue"(j.doc, 'blkpct', 'blk_pct') AS blk_pct,
                    "AI_Source"."NumericValue"(j.doc, 'orebpct', 'orbpct', 'orb_pct') AS orb_pct,
                    "AI_Source"."NumericValue"(j.doc, 'drebpct', 'drbpct', 'drb_pct') AS drb_pct,
                    "AI_Source"."NumericValue"(j.doc, 'threepar', 'three_par') AS three_par,
                    "AI_Source"."NumericValue"(j.doc, 'tusgpct', 'trueusgpct', 'true_usg_pct') AS true_usg_pct,
                    "AI_Source"."NumericValue"(j.doc, 'fouldrawingrate', 'foul_drawing_rate') AS foul_drawing_rate,
                    "AI_Source"."NumericValue"(j.doc, 'rfpergame', 'rf_per_game') AS rf_per_game,
                    "AI_Source"."NumericValue"(j.doc, 'hustleindex', 'hustle_index') AS hustle_index,
                    "AI_Source"."NumericValue"(j.doc, 'ptsper40', 'pts_per_40') AS pts_per_40,
                    "AI_Source"."NumericValue"(j.doc, 'astper40', 'ast_per_40') AS ast_per_40,
                    "AI_Source"."NumericValue"(j.doc, 'trper40', 'rebper40', 'tr_per_40') AS tr_per_40,
                    "AI_Source"."NumericValue"(j.doc, 'stlper40', 'stl_per_40') AS stl_per_40,
                    "AI_Source"."NumericValue"(j.doc, 'blkper40', 'blk_per_40') AS blk_per_40
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc
                ) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id', 'playerid', 'idplayer') IS NOT NULL
            $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS source_team_id,
                   NULL::text AS competition,
                   NULL::double precision AS games, NULL::double precision AS minutes_total,
                   NULL::double precision AS points_total, NULL::double precision AS rebounds_total,
                   NULL::double precision AS offensive_rebounds_total, NULL::double precision AS defensive_rebounds_total,
                   NULL::double precision AS assists_total, NULL::double precision AS steals_total,
                   NULL::double precision AS blocks_total, NULL::double precision AS turnovers_total,
                   NULL::double precision AS fouls_total, NULL::double precision AS fg_pct,
                   NULL::double precision AS three_point_pct, NULL::double precision AS ft_pct,
                   NULL::double precision AS plus_minus, NULL::double precision AS per,
                   NULL::double precision AS ts_pct, NULL::double precision AS usg_pct,
                   NULL::double precision AS bpm, NULL::double precision AS vorp,
                   NULL::double precision AS win_shares, NULL::double precision AS ast_ratio,
                   NULL::double precision AS reb_pct, NULL::double precision AS rating,
                   NULL::double precision AS ortg, NULL::double precision AS drtg,
                   NULL::double precision AS net_rtg,
                   NULL::text AS ruolo_offensivo, NULL::text AS ruolo_difensivo,
                   NULL::text AS ruolo_combinato,
                   NULL::double precision AS on_net_rtg, NULL::double precision AS off_net_rtg,
                   NULL::double precision AS net_rtg_diff, NULL::double precision AS spm,
                   NULL::double precision AS obpm, NULL::double precision AS dbpm,
                   NULL::double precision AS gm_sc, NULL::double precision AS fic,
                   NULL::double precision AS ows, NULL::double precision AS dws,
                   NULL::double precision AS raptor_off, NULL::double precision AS raptor_def,
                   NULL::double precision AS raptor_total, NULL::double precision AS lebron_off,
                   NULL::double precision AS lebron_def, NULL::double precision AS lebron_total,
                   NULL::double precision AS scoring_efficiency, NULL::double precision AS ppsa,
                   NULL::double precision AS two_point_pct, NULL::double precision AS tov_pct,
                   NULL::double precision AS ast_pct, NULL::double precision AS stl_pct,
                   NULL::double precision AS blk_pct, NULL::double precision AS orb_pct,
                   NULL::double precision AS drb_pct, NULL::double precision AS three_par,
                   NULL::double precision AS true_usg_pct, NULL::double precision AS foul_drawing_rate,
                   NULL::double precision AS rf_per_game, NULL::double precision AS hustle_index,
                   NULL::double precision AS pts_per_40, NULL::double precision AS ast_per_40,
                   NULL::double precision AS tr_per_40, NULL::double precision AS stl_per_40,
                   NULL::double precision AS blk_per_40
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."StatsRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Optional team season aggregates
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(left(c.table_name, 18)) = 'advancedstatsteam_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 19);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'teamid', 'id') AS source_team_id,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition,
                    "AI_Source"."NumericValue"(j.doc, 'games') AS games,
                    "AI_Source"."NumericValue"(j.doc, 'pace') AS pace,
                    "AI_Source"."NumericValue"(j.doc, 'ortg') AS ortg,
                    "AI_Source"."NumericValue"(j.doc, 'drtg') AS drtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg') AS net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'threepar', 'fg3rate') AS three_par,
                    "AI_Source"."NumericValue"(j.doc, 'fg3a') AS fg3a,
                    "AI_Source"."NumericValue"(j.doc, 'fg2a') AS fg2a,
                    "AI_Source"."NumericValue"(j.doc, 'ast') AS ast_total,
                    "AI_Source"."NumericValue"(j.doc, 'astpergame') AS ast_per_game
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc
                ) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'teamid', 'id') IS NOT NULL
            $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_team_id, NULL::text AS competition,
                   NULL::double precision AS games, NULL::double precision AS pace,
                   NULL::double precision AS ortg, NULL::double precision AS drtg,
                   NULL::double precision AS net_rtg, NULL::double precision AS three_par,
                   NULL::double precision AS fg3a, NULL::double precision AS fg2a,
                   NULL::double precision AS ast_total, NULL::double precision AS ast_per_game
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."TeamStatsRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Optional role rows
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(left(c.table_name, 12)) = 'playerroles_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 13);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition,
                    "AI_Source"."TextValue"(j.doc, 'ruolooffensivo') AS ruolo_offensivo,
                    "AI_Source"."TextValue"(j.doc, 'ruolodifensivo') AS ruolo_difensivo,
                    "AI_Source"."TextValue"(j.doc, 'ruolocombinato') AS ruolo_combinato
                FROM %I.%I src
                CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id', 'playerid') IS NOT NULL
            $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS competition,
                   NULL::text AS ruolo_offensivo, NULL::text AS ruolo_difensivo,
                   NULL::text AS ruolo_combinato
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."RolesRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Optional on/off rows
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(left(c.table_name, 24)) = 'advancedstatsonoffcourt_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 25);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_on') AS on_net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_off') AS off_net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'netrtg_diff') AS net_rtg_diff,
                    "AI_Source"."NumericValue"(j.doc, 'ortg_on') AS ortg_on,
                    "AI_Source"."NumericValue"(j.doc, 'ortg_off') AS ortg_off,
                    "AI_Source"."NumericValue"(j.doc, 'ortg_diff') AS ortg_diff
                FROM %I.%I src
                CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id', 'playerid') IS NOT NULL
            $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS competition,
                   NULL::double precision AS on_net_rtg, NULL::double precision AS off_net_rtg,
                   NULL::double precision AS net_rtg_diff, NULL::double precision AS ortg_on,
                   NULL::double precision AS ortg_off, NULL::double precision AS ortg_diff
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."OnOffRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Optional clutch rows
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    league_key text;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi'
          AND lower(left(c.table_name, 21)) = 'advancedstats_clutch_'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 22);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition,
                    "AI_Source"."NumericValue"(j.doc, 'clutchgames', 'games') AS clutch_games,
                    "AI_Source"."NumericValue"(j.doc, 'clutchpts', 'pts') AS clutch_pts,
                    "AI_Source"."NumericValue"(j.doc, 'clutchtspct', 'tspct') AS clutch_ts_pct,
                    "AI_Source"."NumericValue"(j.doc, 'asttotovratio', 'clutchasttotov') AS clutch_ast_to_tov,
                    "AI_Source"."NumericValue"(j.doc, 'clutchnetrtg', 'netrtg') AS clutch_net_rtg,
                    "AI_Source"."NumericValue"(j.doc, 'clutchefgpct', 'efgpct') AS clutch_efg_pct
                FROM %I.%I src
                CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id', 'playerid') IS NOT NULL
            $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS competition,
                   NULL::double precision AS clutch_games, NULL::double precision AS clutch_pts,
                   NULL::double precision AS clutch_ts_pct, NULL::double precision AS clutch_ast_to_tov,
                   NULL::double precision AS clutch_net_rtg, NULL::double precision AS clutch_efg_pct
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."ClutchRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Optional Boxscore rows. They are used for roster relations and starter %.
-- -------------------------------------------------------------------------

DO $$
DECLARE
    r record;
    body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'boxscore'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) IN ('id', 'idplayer', 'playerid'))
        ORDER BY c.table_name
    LOOP
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END ||
            format($sql$
                SELECT
                    %L::text AS league_key,
                    "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                    "AI_Source"."TextValue"(j.doc, 'id', 'idplayer', 'playerid') AS source_player_id,
                    "AI_Source"."TextValue"(j.doc, 'teamid', 'idteam') AS source_team_id,
                    "AI_Source"."TextValue"(j.doc, 'game', 'idgame', 'gamecode') AS game_id,
                    "AI_Source"."TextValue"(j.doc, 'sf', 'starter', 'isstarter') AS started_raw,
                    upper(COALESCE("AI_Source"."TextValue"(j.doc, 'competition'), 'RS')) AS competition
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc
                ) j
                WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
                  AND "AI_Source"."TextValue"(j.doc, 'id', 'idplayer', 'playerid') IS NOT NULL
            $sql$, r.table_name, r.table_schema, r.table_name);
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT NULL::text AS league_key, NULL::integer AS season,
                   NULL::text AS source_player_id, NULL::text AS source_team_id,
                   NULL::text AS game_id, NULL::text AS started_raw,
                   NULL::text AS competition
            WHERE false
        $empty$;
    END IF;

    EXECUTE 'CREATE VIEW "AI_Source"."BoxscoreRawInternal" AS ' || body;
END;
$$;

-- -------------------------------------------------------------------------
-- Public relation contract. Future roster seasons are preserved even when no
-- Boxscore/stats exist yet by also matching Anagrafiche.TeamName.
-- -------------------------------------------------------------------------

CREATE VIEW "AI_Source"."TeamPlayerRelations" AS
WITH relation_source AS (
    SELECT DISTINCT league_key, season, source_player_id, source_team_id
    FROM "AI_Source"."BoxscoreRawInternal"
    WHERE source_team_id IS NOT NULL

    UNION

    SELECT DISTINCT league_key, season, source_player_id, source_team_id
    FROM "AI_Source"."StatsRawInternal"
    WHERE source_team_id IS NOT NULL

    UNION

    SELECT DISTINCT
        pr.league_key,
        pr.season,
        pr.source_player_id,
        tr.source_team_id
    FROM "AI_Source"."PlayerRegistryInternal" pr
    JOIN "AI_Source"."TeamRegistryInternal" tr
      ON tr.league_key = pr.league_key
     AND tr.season = pr.season
     AND pr.team_name IS NOT NULL
     AND (
          lower(btrim(pr.team_name)) = lower(btrim(tr.name))
          OR lower(btrim(pr.team_name)) = lower(btrim(COALESCE(tr.short_name, '')))
     )
),
mapped AS (
    SELECT
        "AI_Source"."StableId"('team', tr.global_id) AS team_id,
        "AI_Source"."StableId"('player', pr.global_id) AS player_id,
        rs.season,
        COALESCE(pr.position, '') AS role,
        pr.jersey_number,
        row_number() OVER (
            PARTITION BY tr.global_id, pr.global_id, rs.season
            ORDER BY rs.league_key
        ) AS rn
    FROM relation_source rs
    JOIN "AI_Source"."PlayerRegistryInternal" pr
      ON pr.league_key = rs.league_key
     AND pr.season = rs.season
     AND pr.source_player_id = rs.source_player_id
    JOIN "AI_Source"."TeamRegistryInternal" tr
      ON tr.league_key = rs.league_key
     AND tr.season = rs.season
     AND tr.source_team_id = rs.source_team_id
)
SELECT team_id, player_id, season, role, jersey_number
FROM mapped
WHERE rn = 1
  AND team_id IS NOT NULL
  AND player_id IS NOT NULL;

-- -------------------------------------------------------------------------
-- Public players contract: one row per global player, latest registry state.
-- -------------------------------------------------------------------------

CREATE VIEW "AI_Source"."Players" AS
WITH registry_ranked AS (
    SELECT
        pr.*,
        row_number() OVER (
            PARTITION BY pr.global_id
            ORDER BY pr.season DESC, pr.league_key
        ) AS rn
    FROM "AI_Source"."PlayerRegistryInternal" pr
),
current_relation AS (
    SELECT
        rel.player_id,
        rel.team_id,
        rel.season,
        row_number() OVER (
            PARTITION BY rel.player_id
            ORDER BY rel.season DESC, rel.team_id
        ) AS rn
    FROM "AI_Source"."TeamPlayerRelations" rel
)
SELECT
    "AI_Source"."StableId"('player', p.global_id) AS id,
    p.global_id,
    p.name,
    COALESCE(
        p.age,
        CASE
            WHEN p.birth_date IS NOT NULL
                THEN p.season - extract(year FROM p.birth_date)::integer
            ELSE 0
        END
    )::integer AS age,
    COALESCE(NULLIF(p.position, ''), 'PG') AS position,
    COALESCE(p.nationality, '') AS nationality,
    p.height_cm,
    p.weight_kg,
    'R'::text AS dominant_hand,
    cr.team_id AS current_team_id,
    "AI_Source"."StableId"('league', p.league_key) AS current_league_id,
    NULL::integer AS draft_year,
    NULL::integer AS draft_pick,
    p.birth_date,
    p.league_key AS current_league_key
FROM registry_ranked p
LEFT JOIN current_relation cr
  ON cr.player_id = "AI_Source"."StableId"('player', p.global_id)
 AND cr.rn = 1
WHERE p.rn = 1;

-- -------------------------------------------------------------------------
-- Public teams contract: one row per global team, latest registry state.
-- -------------------------------------------------------------------------

CREATE VIEW "AI_Source"."Teams" AS
WITH registry_ranked AS (
    SELECT
        tr.*,
        row_number() OVER (
            PARTITION BY tr.global_id
            ORDER BY tr.season DESC, tr.league_key
        ) AS rn
    FROM "AI_Source"."TeamRegistryInternal" tr
),
team_stats_ranked AS (
    SELECT
        ts.*,
        row_number() OVER (
            PARTITION BY ts.league_key, ts.season, ts.source_team_id
            ORDER BY
                CASE upper(ts.competition)
                    WHEN 'TOT' THEN 0
                    WHEN 'RS' THEN 1
                    ELSE 2
                END,
                ts.competition
        ) AS rn
    FROM "AI_Source"."TeamStatsRawInternal" ts
)
SELECT
    "AI_Source"."StableId"('team', tr.global_id) AS id,
    tr.global_id,
    tr.name,
    "AI_Source"."StableId"('league', tr.league_key) AS league_id,
    ''::text AS playing_style,
    ''::text AS formation,
    COALESCE(ts.pace, 75.0) AS pace,
    COALESCE(ts.ortg, 110.0) AS offensive_rating,
    COALESCE(ts.drtg, 110.0) AS defensive_rating,
    COALESCE(
        ts.three_par,
        CASE
            WHEN COALESCE(ts.fg3a, 0) + COALESCE(ts.fg2a, 0) > 0
                THEN ts.fg3a / (ts.fg3a + ts.fg2a)
            ELSE NULL
        END,
        0.35
    ) AS three_point_attempt_rate,
    COALESCE(
        ts.ast_per_game,
        CASE WHEN COALESCE(ts.games, 0) > 0 THEN ts.ast_total / ts.games ELSE NULL END,
        20.0
    ) AS assists_per_game,
    0.25::double precision AS star_player_usage,
    1::integer AS league_tier,
    COALESCE(tr.short_name, '') AS short_name,
    COALESCE(ts.net_rtg, COALESCE(ts.ortg, 110.0) - COALESCE(ts.drtg, 110.0)) AS net_rtg,
    tr.league_key,
    tr.season AS latest_season
FROM registry_ranked tr
LEFT JOIN team_stats_ranked ts
  ON ts.league_key = tr.league_key
 AND ts.season = tr.season
 AND ts.source_team_id = tr.source_team_id
 AND ts.rn = 1
WHERE tr.rn = 1;

-- -------------------------------------------------------------------------
-- Public player_stats contract.
--
-- CRITICAL: exactly ONE observation per global player + season. A player can
-- have domestic + European rows and each physical table can contain RS/PO/TOT
-- or team-specific rows. Those are not consecutive seasons and must not become
-- t -> t targets. Ranking prefers all-team aggregates, TOT/RS, then the row
-- with most games.
-- -------------------------------------------------------------------------

CREATE VIEW "AI_Source"."PlayerStats" AS
WITH resolved AS (
    SELECT
        s.*,
        pr.global_id AS player_global_id,
        direct_team.global_id AS direct_team_global_id,
        row_number() OVER (
            PARTITION BY pr.global_id, s.season
            ORDER BY
                CASE WHEN s.source_team_id IS NULL THEN 0 ELSE 1 END,
                CASE upper(s.competition)
                    WHEN 'TOT' THEN 0
                    WHEN 'RS' THEN 1
                    ELSE 2
                END,
                COALESCE(s.games, 0) DESC,
                s.league_key,
                s.source_team_id NULLS LAST
        ) AS rn
    FROM "AI_Source"."StatsRawInternal" s
    JOIN "AI_Source"."PlayerRegistryInternal" pr
      ON pr.league_key = s.league_key
     AND pr.season = s.season
     AND pr.source_player_id = s.source_player_id
    LEFT JOIN "AI_Source"."TeamRegistryInternal" direct_team
      ON direct_team.league_key = s.league_key
     AND direct_team.season = s.season
     AND direct_team.source_team_id = s.source_team_id
    WHERE s.rating IS NOT NULL
),
role_ranked AS (
    SELECT
        r.*,
        row_number() OVER (
            PARTITION BY r.league_key, r.season, r.source_player_id
            ORDER BY CASE upper(r.competition) WHEN 'TOT' THEN 0 WHEN 'RS' THEN 1 ELSE 2 END,
                     r.competition
        ) AS rn
    FROM "AI_Source"."RolesRawInternal" r
),
onoff_ranked AS (
    SELECT
        o.*,
        row_number() OVER (
            PARTITION BY o.league_key, o.season, o.source_player_id
            ORDER BY CASE upper(o.competition) WHEN 'TOT' THEN 0 WHEN 'RS' THEN 1 ELSE 2 END,
                     o.competition
        ) AS rn
    FROM "AI_Source"."OnOffRawInternal" o
),
clutch_ranked AS (
    SELECT
        c.*,
        row_number() OVER (
            PARTITION BY c.league_key, c.season, c.source_player_id, c.competition
            ORDER BY CASE upper(c.competition) WHEN 'TOT' THEN 0 WHEN 'RS' THEN 1 ELSE 2 END,
                     c.competition
        ) AS rn
    FROM "AI_Source"."ClutchRawInternal" c
),
single_relation AS (
    SELECT
        player_id,
        season,
        CASE WHEN count(DISTINCT team_id) = 1 THEN min(team_id) ELSE NULL END AS team_id
    FROM "AI_Source"."TeamPlayerRelations"
    GROUP BY player_id, season
),
starts AS (
    SELECT
        b.league_key,
        b.season,
        b.source_player_id,
        count(DISTINCT b.game_id) FILTER (WHERE "AI_Source"."Truthy"(b.started_raw))::double precision AS games_started
    FROM "AI_Source"."BoxscoreRawInternal" b
    GROUP BY b.league_key, b.season, b.source_player_id
)
SELECT
    "AI_Source"."StableId"('player', s.player_global_id) AS player_id,
    s.season,
    COALESCE(
        "AI_Source"."StableId"('team', s.direct_team_global_id),
        sr.team_id
    ) AS team_id,
    "AI_Source"."StableId"('league', s.league_key) AS league_id,
    COALESCE(s.games, 0)::integer AS games_played,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.minutes_total, 0) / s.games ELSE 0 END AS minutes_per_game,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.points_total, 0) / s.games ELSE 0 END AS points,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.rebounds_total, 0) / s.games ELSE 0 END AS rebounds,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.offensive_rebounds_total, 0) / s.games ELSE 0 END AS offensive_rebounds,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.defensive_rebounds_total, 0) / s.games ELSE 0 END AS defensive_rebounds,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.assists_total, 0) / s.games ELSE 0 END AS assists,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.steals_total, 0) / s.games ELSE 0 END AS steals,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.blocks_total, 0) / s.games ELSE 0 END AS blocks,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.turnovers_total, 0) / s.games ELSE 0 END AS turnovers,
    CASE WHEN COALESCE(s.games, 0) > 0 THEN COALESCE(s.fouls_total, 0) / s.games ELSE 0 END AS personal_fouls,
    COALESCE(s.fg_pct, 0) AS fg_pct,
    COALESCE(s.three_point_pct, 0) AS three_point_pct,
    COALESCE(s.ft_pct, 0) AS ft_pct,
    COALESCE(s.plus_minus, 0) AS plus_minus,
    COALESCE(s.per, 0) AS per,
    COALESCE(s.ts_pct, 0) AS ts_pct,
    COALESCE(s.usg_pct, 0) AS usg_pct,
    COALESCE(s.bpm, 0) AS bpm,
    COALESCE(s.vorp, 0) AS vorp,
    COALESCE(s.win_shares, 0) AS win_shares,
    COALESCE(s.ast_ratio, 0) AS ast_ratio,
    COALESCE(s.reb_pct, 0) AS reb_pct,
    s.rating,
    COALESCE(s.ortg, 0) AS ortg,
    COALESCE(s.drtg, 0) AS drtg,
    COALESCE(s.net_rtg, 0) AS net_rtg,
    COALESCE(s.ruolo_offensivo, rr.ruolo_offensivo, '') AS ruolo_offensivo,
    COALESCE(s.ruolo_difensivo, rr.ruolo_difensivo, '') AS ruolo_difensivo,
    COALESCE(s.ruolo_combinato, rr.ruolo_combinato, '') AS ruolo_combinato,
    COALESCE(s.on_net_rtg, oo.on_net_rtg, 0) AS on_net_rtg,
    COALESCE(s.off_net_rtg, oo.off_net_rtg, 0) AS off_net_rtg,
    COALESCE(s.net_rtg_diff, oo.net_rtg_diff, 0) AS net_rtg_diff,
    CASE WHEN upper(s.competition) = 'TOT' THEN 'RS' ELSE upper(s.competition) END AS competition,
    COALESCE(s.spm, 0) AS spm,
    COALESCE(s.obpm, 0) AS obpm,
    COALESCE(s.dbpm, 0) AS dbpm,
    COALESCE(s.gm_sc, 0) AS gm_sc,
    COALESCE(s.fic, 0) AS fic,
    COALESCE(s.ows, 0) AS ows,
    COALESCE(s.dws, 0) AS dws,
    COALESCE(s.raptor_off, 0) AS raptor_off,
    COALESCE(s.raptor_def, 0) AS raptor_def,
    COALESCE(s.raptor_total, COALESCE(s.raptor_off, 0) + COALESCE(s.raptor_def, 0)) AS raptor_total,
    COALESCE(s.lebron_off, 0) AS lebron_off,
    COALESCE(s.lebron_def, 0) AS lebron_def,
    COALESCE(s.lebron_total, COALESCE(s.lebron_off, 0) + COALESCE(s.lebron_def, 0)) AS lebron_total,
    COALESCE(s.scoring_efficiency, 0) AS scoring_efficiency,
    COALESCE(s.ppsa, 0) AS ppsa,
    COALESCE(s.two_point_pct, 0) AS two_point_pct,
    COALESCE(s.tov_pct, 0) AS tov_pct,
    COALESCE(s.ast_pct, 0) AS ast_pct,
    COALESCE(s.stl_pct, 0) AS stl_pct,
    COALESCE(s.blk_pct, 0) AS blk_pct,
    COALESCE(s.orb_pct, 0) AS orb_pct,
    COALESCE(s.drb_pct, 0) AS drb_pct,
    COALESCE(s.three_par, 0) AS three_par,
    COALESCE(s.true_usg_pct, 0) AS true_usg_pct,
    COALESCE(s.foul_drawing_rate, 0) AS foul_drawing_rate,
    COALESCE(s.rf_per_game, 0) AS rf_per_game,
    COALESCE(s.hustle_index, 0) AS hustle_index,
    COALESCE(s.pts_per_40, 0) AS pts_per_40,
    COALESCE(s.ast_per_40, 0) AS ast_per_40,
    COALESCE(s.tr_per_40, 0) AS tr_per_40,
    COALESCE(s.stl_per_40, 0) AS stl_per_40,
    COALESCE(s.blk_per_40, 0) AS blk_per_40,
    COALESCE(cr.clutch_games, 0) AS clutch_games,
    COALESCE(cr.clutch_pts, 0) AS clutch_pts,
    COALESCE(cr.clutch_ts_pct, 0) AS clutch_ts_pct,
    COALESCE(cr.clutch_ast_to_tov, 0) AS clutch_ast_to_tov,
    COALESCE(cr.clutch_net_rtg, 0) AS clutch_net_rtg,
    COALESCE(cr.clutch_efg_pct, 0) AS clutch_efg_pct,
    COALESCE(oo.ortg_on, 0) AS ortg_on,
    COALESCE(oo.ortg_off, 0) AS ortg_off,
    COALESCE(oo.ortg_diff, 0) AS ortg_diff,
    COALESCE(st.games_started, 0)::integer AS games_started,
    CASE
        WHEN COALESCE(s.games, 0) > 0
            THEN LEAST(1.0, COALESCE(st.games_started, 0) / s.games)
        ELSE 0
    END AS starter_pct,
    s.league_key
FROM resolved s
LEFT JOIN single_relation sr
  ON sr.player_id = "AI_Source"."StableId"('player', s.player_global_id)
 AND sr.season = s.season
LEFT JOIN role_ranked rr
  ON rr.league_key = s.league_key
 AND rr.season = s.season
 AND rr.source_player_id = s.source_player_id
 AND rr.rn = 1
LEFT JOIN onoff_ranked oo
  ON oo.league_key = s.league_key
 AND oo.season = s.season
 AND oo.source_player_id = s.source_player_id
 AND oo.rn = 1
LEFT JOIN clutch_ranked cr
 ON cr.league_key = s.league_key
 AND cr.season = s.season
 AND cr.source_player_id = s.source_player_id
 AND upper(cr.competition) = upper(s.competition)
 AND cr.rn = 1
LEFT JOIN starts st
  ON st.league_key = s.league_key
 AND st.season = s.season
 AND st.source_player_id = s.source_player_id
WHERE s.rn = 1;

-- -------------------------------------------------------------------------
-- Public leagues contract. max_games is observed from canonical player rows,
-- avoiding an NBA-specific 82-game assumption.
-- -------------------------------------------------------------------------

CREATE VIEW "AI_Source"."Leagues" AS
WITH keys AS (
    SELECT DISTINCT league_key FROM "AI_Source"."PlayerRegistryInternal"
    UNION
    SELECT DISTINCT league_key FROM "AI_Source"."TeamRegistryInternal"
    UNION
    SELECT DISTINCT league_key FROM "AI_Source"."StatsRawInternal"
),
observed AS (
    SELECT league_id, max(games_played)::integer AS max_games
    FROM "AI_Source"."PlayerStats"
    GROUP BY league_id
),
team_average AS (
    SELECT
        league_id,
        avg(pace) AS avg_pace,
        avg(offensive_rating) AS avg_offensive_rating
    FROM "AI_Source"."Teams"
    GROUP BY league_id
)
SELECT
    "AI_Source"."StableId"('league', k.league_key) AS id,
    k.league_key AS name,
    CASE
        WHEN k.league_key LIKE 'ITA%' THEN 'IT'
        WHEN k.league_key LIKE 'ESP%' THEN 'ES'
        WHEN k.league_key LIKE 'GER%' THEN 'DE'
        WHEN k.league_key LIKE 'FRA%' THEN 'FR'
        WHEN k.league_key LIKE 'GRC%' THEN 'GR'
        WHEN k.league_key LIKE 'TUR%' THEN 'TR'
        WHEN k.league_key LIKE 'ISR%' THEN 'IL'
        WHEN k.league_key LIKE 'LTU%' THEN 'LT'
        WHEN k.league_key LIKE 'CRO%' THEN 'HR'
        WHEN k.league_key LIKE 'SRB%' THEN 'RS'
        WHEN k.league_key LIKE 'POR%' THEN 'PT'
        ELSE regexp_replace(k.league_key, '[0-9]+$', '')
    END AS country,
    1::integer AS tier,
    1.0::double precision AS competitiveness_score,
    COALESCE(ta.avg_pace, 75.0) AS avg_pace,
    COALESCE(ta.avg_offensive_rating, 110.0) AS avg_offensive_rating,
    GREATEST(COALESCE(o.max_games, 1), 1)::integer AS max_games,
    k.league_key
FROM keys k
LEFT JOIN observed o
  ON o.league_id = "AI_Source"."StableId"('league', k.league_key)
LEFT JOIN team_average ta
  ON ta.league_id = "AI_Source"."StableId"('league', k.league_key)
WHERE k.league_key IS NOT NULL
  AND btrim(k.league_key) <> '';

COMMIT;

-- -------------------------------------------------------------------------
-- Suggested verification after installation
-- -------------------------------------------------------------------------
-- SELECT name, max_games FROM "AI_Source"."Leagues" ORDER BY name;
-- SELECT count(*) FROM "AI_Source"."Players";
-- SELECT count(*) FROM "AI_Source"."Teams";
-- SELECT min(season), max(season), count(*) FROM "AI_Source"."PlayerStats";
-- SELECT player_id, season, count(*)
--   FROM "AI_Source"."PlayerStats" GROUP BY player_id, season HAVING count(*) > 1;
-- SELECT * FROM "AI_Source"."TeamPlayerRelations" LIMIT 20;

-- =========================================================================
-- AI_Source — SCRIPT UNICO auto-adattivo
--
-- Fa tutto quello che facevano prima ai_source_schema.sql +
-- ai_source_parallel_safety.sql + ai_source_competition.sql +
-- ai_source_simulation.sql + il patch di ottimizzazione, IN UN SOLO FILE
-- IDEMPOTENTE.
--
-- QUANDO AGGIUNGI UNA LEGA: non tocchi nulla qui dentro. Carichi le nuove
-- tabelle fisiche (Anagrafiche.<LEAGUE>, Analisi.AdvancedStats_Player_<LEAGUE>,
-- ecc. — stessa convenzione di nomi di sempre) e rilanci semplicemente
-- QUESTO file. La discovery via information_schema trova le nuove tabelle
-- da sola; il livello RawInternal (ora materializzato per le performance)
-- viene smontato e ricostruito automaticamente ad ogni run, quindi non
-- serve nessun passaggio manuale di drop/rollback.
--
-- Esecuzione: psql -f ai_source_full.sql -d <tuo_db>
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS "AI_Source";

-- -------------------------------------------------------------------------
-- Helper di smontaggio: usato al posto dei "DROP VIEW IF EXISTS" originali
-- sui SOLI oggetti *RawInternal, cioe' quelli che questo script materializza
-- alla fine. Capisce da solo se l'oggetto e' attualmente una VIEW o una
-- MATERIALIZED VIEW (dal run precedente) e lo droppa nel modo corretto,
-- cosi' la rigenerazione dinamica sotto non fallisce mai.
-- -------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION "AI_Source"."DropRawObject"(obj_name text)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname = 'AI_Source' AND matviewname = obj_name) THEN
        EXECUTE format('DROP MATERIALIZED VIEW IF EXISTS "AI_Source".%I CASCADE', obj_name);
    ELSE
        EXECUTE format('DROP VIEW IF EXISTS "AI_Source".%I CASCADE', obj_name);
    END IF;
    -- residuo di un run precedente (vista rinominata prima di materializzare)
    EXECUTE format('DROP VIEW IF EXISTS "AI_Source".%I CASCADE', obj_name || '_Def');
END;
$$;

-- =========================================================================
-- SEZIONE 1 — Contratto base (ex ai_source_schema.sql)
-- =========================================================================

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
SELECT "AI_Source"."DropRawObject"('SimulationLineupRawInternal');
SELECT "AI_Source"."DropRawObject"('SimulationPbpRawInternal');
DROP FUNCTION IF EXISTS "AI_Source"."SimulationLineupIds"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayerId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationTeamId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayType"(text);

DROP VIEW IF EXISTS "AI_Source"."Leagues";
DROP VIEW IF EXISTS "AI_Source"."PlayerStats";
DROP VIEW IF EXISTS "AI_Source"."Players";
DROP VIEW IF EXISTS "AI_Source"."Teams";
DROP VIEW IF EXISTS "AI_Source"."TeamPlayerRelations";
SELECT "AI_Source"."DropRawObject"('BoxscoreRawInternal');
SELECT "AI_Source"."DropRawObject"('ClutchRawInternal');
SELECT "AI_Source"."DropRawObject"('OnOffRawInternal');
SELECT "AI_Source"."DropRawObject"('RolesRawInternal');
SELECT "AI_Source"."DropRawObject"('TeamStatsRawInternal');
SELECT "AI_Source"."DropRawObject"('StatsRawInternal');
SELECT "AI_Source"."DropRawObject"('TeamRegistryInternal');
SELECT "AI_Source"."DropRawObject"('PlayerRegistryInternal');

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

-- =========================================================================
-- SEZIONE 2 — Fix parallel-safety (ex ai_source_parallel_safety.sql)
-- Necessario perche' NumericValue()/IntValue()/DateValue() contengono
-- blocchi EXCEPTION (implementati con subtransaction, incompatibili con i
-- parallel worker). Vedi errore "cannot start subtransactions during a
-- parallel operation".
-- =========================================================================

-- PostgreSQL parallel-safety correction for "AI_Source" parsing helpers.
--
-- _num() and _date() contain EXCEPTION handlers. PostgreSQL implements those
-- handlers with subtransactions, which cannot execute inside parallel workers.
-- _int() calls _num(), so it must inherit the same restriction.
--
-- Run this immediately after ai_source_schema.sql. It is idempotent.

BEGIN;

ALTER FUNCTION "AI_Source"."NumericValue"(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION "AI_Source"."IntValue"(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION "AI_Source"."DateValue"(jsonb, text[]) PARALLEL UNSAFE;

COMMIT;

-- =========================================================================
-- SEZIONE 3 — Contratto competition-preserving (ex ai_source_competition.sql)
-- =========================================================================

-- Competition-preserving PostgreSQL contract for AI_Model.
-- Run after ai_source_schema.sql. This adapter discovers the physical source
-- tables independently, so re-running the base adapter does not destroy or
-- depend on these views.

BEGIN;

CREATE OR REPLACE FUNCTION "AI_Source"."CompetitionCode"(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE upper(regexp_replace(btrim(COALESCE(value, 'RS')), '[[:space:]_-]+', ' ', 'g'))
        WHEN '' THEN 'RS'
        WHEN 'REGULAR' THEN 'RS'
        WHEN 'REGULAR SEASON' THEN 'RS'
        WHEN 'REG SEASON' THEN 'RS'
        WHEN 'RS' THEN 'RS'
        WHEN 'PLAYOFF' THEN 'PO'
        WHEN 'PLAYOFFS' THEN 'PO'
        WHEN 'POSTSEASON' THEN 'PO'
        WHEN 'PO' THEN 'PO'
        WHEN 'TOTAL' THEN 'TOT'
        WHEN 'ALL' THEN 'TOT'
        WHEN 'TOT' THEN 'TOT'
        WHEN 'SUPER CUP' THEN 'SUPERCUP'
        WHEN 'SUPERCUP' THEN 'SUPERCUP'
        WHEN 'CUP' THEN 'CUP'
        ELSE regexp_replace(
            upper(regexp_replace(btrim(COALESCE(value, 'RS')), '[[:space:]-]+', '_', 'g')),
            '[^A-Z0-9_]', '', 'g'
        )
    END;
$$;

DROP VIEW IF EXISTS "AI_Source"."TeamCompetitionStats";
DROP VIEW IF EXISTS "AI_Source"."PlayerCompetitionStats";
SELECT "AI_Source"."DropRawObject"('CompetitionBoxscoreRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionClutchRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionOnOffRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionRolesRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionTeamStatsRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionStatsRawInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionTeamRegistryInternal');
SELECT "AI_Source"."DropRawObject"('CompetitionPlayerRegistryInternal');

-- Player identity by league/season.
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
                   NULL::text AS team_name
            WHERE false
        $empty$;
    END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionPlayerRegistryInternal" AS ' || body;
END;
$$;

-- Team identity by league/season.
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
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionTeamRegistryInternal" AS ' || body;
END;
$$;

-- Player aggregate rows. Competition is preserved instead of collapsing to TOT/RS.
-- Missing advanced metrics remain NULL in the canonical competition contract so the
-- model can distinguish "metric unavailable" from a measured value of zero.
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
                    "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
                    "AI_Source"."NumericValue"(j.doc, 'games', 'gamesplayed') AS games,
                    "AI_Source"."NumericValue"(j.doc, 'gamesstarted', 'games_started') AS games_started,
                    "AI_Source"."NumericValue"(j.doc, 'starterpct', 'starter_pct') AS starter_pct,
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
                   NULL::text AS competition, NULL::double precision AS games,
                   NULL::double precision AS games_started, NULL::double precision AS starter_pct,
                   NULL::double precision AS minutes_total, NULL::double precision AS points_total,
                   NULL::double precision AS rebounds_total, NULL::double precision AS offensive_rebounds_total,
                   NULL::double precision AS defensive_rebounds_total, NULL::double precision AS assists_total,
                   NULL::double precision AS steals_total, NULL::double precision AS blocks_total,
                   NULL::double precision AS turnovers_total, NULL::double precision AS fouls_total,
                   NULL::double precision AS fg_pct, NULL::double precision AS three_point_pct,
                   NULL::double precision AS ft_pct, NULL::double precision AS plus_minus,
                   NULL::double precision AS per, NULL::double precision AS ts_pct,
                   NULL::double precision AS usg_pct, NULL::double precision AS bpm,
                   NULL::double precision AS vorp, NULL::double precision AS win_shares,
                   NULL::double precision AS ast_ratio, NULL::double precision AS reb_pct,
                   NULL::double precision AS rating, NULL::double precision AS ortg,
                   NULL::double precision AS drtg, NULL::double precision AS net_rtg,
                   NULL::text AS ruolo_offensivo, NULL::text AS ruolo_difensivo,
                   NULL::text AS ruolo_combinato, NULL::double precision AS on_net_rtg,
                   NULL::double precision AS off_net_rtg, NULL::double precision AS net_rtg_diff,
                   NULL::double precision AS spm, NULL::double precision AS obpm,
                   NULL::double precision AS dbpm, NULL::double precision AS gm_sc,
                   NULL::double precision AS fic, NULL::double precision AS ows,
                   NULL::double precision AS dws, NULL::double precision AS raptor_off,
                   NULL::double precision AS raptor_def, NULL::double precision AS raptor_total,
                   NULL::double precision AS lebron_off, NULL::double precision AS lebron_def,
                   NULL::double precision AS lebron_total, NULL::double precision AS scoring_efficiency,
                   NULL::double precision AS ppsa, NULL::double precision AS two_point_pct,
                   NULL::double precision AS tov_pct, NULL::double precision AS ast_pct,
                   NULL::double precision AS stl_pct, NULL::double precision AS blk_pct,
                   NULL::double precision AS orb_pct, NULL::double precision AS drb_pct,
                   NULL::double precision AS three_par, NULL::double precision AS true_usg_pct,
                   NULL::double precision AS foul_drawing_rate, NULL::double precision AS rf_per_game,
                   NULL::double precision AS hustle_index, NULL::double precision AS pts_per_40,
                   NULL::double precision AS ast_per_40, NULL::double precision AS tr_per_40,
                   NULL::double precision AS stl_per_40, NULL::double precision AS blk_per_40
            WHERE false
        $empty$;
    END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionStatsRawInternal" AS ' || body;
END;
$$;

-- Optional competition-specific role/on-off/clutch rows.
DO $$
DECLARE r record; league_key text; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi' AND lower(left(c.table_name, 12)) = 'playerroles_'
        GROUP BY c.table_schema, c.table_name HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 13);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
                   "AI_Source"."TextValue"(j.doc, 'ruolooffensivo') AS ruolo_offensivo,
                   "AI_Source"."TextValue"(j.doc, 'ruolodifensivo') AS ruolo_difensivo,
                   "AI_Source"."TextValue"(j.doc, 'ruolocombinato') AS ruolo_combinato
            FROM %I.%I src
            CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
            WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
              AND "AI_Source"."TextValue"(j.doc, 'id', 'playerid') IS NOT NULL
        $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;
    IF body = '' THEN body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_player_id, NULL::text competition, NULL::text ruolo_offensivo, NULL::text ruolo_difensivo, NULL::text ruolo_combinato WHERE false'; END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionRolesRawInternal" AS ' || body;
END;
$$;

DO $$
DECLARE r record; league_key text; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi' AND lower(left(c.table_name, 24)) = 'advancedstatsonoffcourt_'
        GROUP BY c.table_schema, c.table_name HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 25);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
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
    IF body = '' THEN body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_player_id, NULL::text competition, NULL::double precision on_net_rtg, NULL::double precision off_net_rtg, NULL::double precision net_rtg_diff, NULL::double precision ortg_on, NULL::double precision ortg_off, NULL::double precision ortg_diff WHERE false'; END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionOnOffRawInternal" AS ' || body;
END;
$$;

DO $$
DECLARE r record; league_key text; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi' AND lower(left(c.table_name, 21)) = 'advancedstats_clutch_'
        GROUP BY c.table_schema, c.table_name HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 22);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."TextValue"(j.doc, 'id', 'playerid') AS source_player_id,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
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
    IF body = '' THEN body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_player_id, NULL::text competition, NULL::double precision clutch_games, NULL::double precision clutch_pts, NULL::double precision clutch_ts_pct, NULL::double precision clutch_ast_to_tov, NULL::double precision clutch_net_rtg, NULL::double precision clutch_efg_pct WHERE false'; END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionClutchRawInternal" AS ' || body;
END;
$$;

-- Boxscore is used only for competition-specific starter counts.
DO $$
DECLARE r record; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'boxscore'
        GROUP BY c.table_schema, c.table_name
        HAVING bool_or(lower(c.column_name) = 'season')
           AND bool_or(lower(c.column_name) IN ('id', 'idplayer', 'playerid'))
        ORDER BY c.table_name
    LOOP
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."TextValue"(j.doc, 'id', 'idplayer', 'playerid') AS source_player_id,
                   "AI_Source"."TextValue"(j.doc, 'teamid', 'idteam') AS source_team_id,
                   "AI_Source"."TextValue"(j.doc, 'game', 'idgame', 'gamecode') AS game_id,
                   "AI_Source"."TextValue"(j.doc, 'sf', 'starter', 'isstarter') AS started_raw,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition
            FROM %I.%I src
            CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
            WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
              AND "AI_Source"."TextValue"(j.doc, 'id', 'idplayer', 'playerid') IS NOT NULL
        $sql$, r.table_name, r.table_schema, r.table_name);
    END LOOP;
    IF body = '' THEN body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_player_id, NULL::text source_team_id, NULL::text game_id, NULL::text started_raw, NULL::text competition WHERE false'; END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionBoxscoreRawInternal" AS ' || body;
END;
$$;

-- Team aggregate rows by competition.
DO $$
DECLARE r record; league_key text; body text := '';
BEGIN
    FOR r IN
        SELECT c.table_schema, c.table_name FROM information_schema.columns c
        WHERE lower(c.table_schema) = 'analisi' AND lower(left(c.table_name, 18)) = 'advancedstatsteam_'
        GROUP BY c.table_schema, c.table_name HAVING bool_or(lower(c.column_name) = 'season')
        ORDER BY c.table_name
    LOOP
        league_key := substr(r.table_name, 19);
        body := body || CASE WHEN body = '' THEN '' ELSE E'\nUNION ALL\n' END || format($sql$
            SELECT %L::text AS league_key,
                   "AI_Source"."IntValue"(j.doc, 'season')::integer AS season,
                   "AI_Source"."TextValue"(j.doc, 'teamid', 'id') AS source_team_id,
                   "AI_Source"."CompetitionCode"("AI_Source"."TextValue"(j.doc, 'competition')) AS competition,
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
            CROSS JOIN LATERAL (SELECT "AI_Source"."LowerKeys"(to_jsonb(src)) AS doc) j
            WHERE "AI_Source"."IntValue"(j.doc, 'season') IS NOT NULL
              AND "AI_Source"."TextValue"(j.doc, 'teamid', 'id') IS NOT NULL
        $sql$, league_key, r.table_schema, r.table_name);
    END LOOP;
    IF body = '' THEN body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_team_id, NULL::text competition, NULL::double precision games, NULL::double precision pace, NULL::double precision ortg, NULL::double precision drtg, NULL::double precision net_rtg, NULL::double precision three_par, NULL::double precision fg3a, NULL::double precision fg2a, NULL::double precision ast_total, NULL::double precision ast_per_game WHERE false'; END IF;
    EXECUTE 'CREATE VIEW "AI_Source"."CompetitionTeamStatsRawInternal" AS ' || body;
END;
$$;

CREATE VIEW "AI_Source"."PlayerCompetitionStats" AS
WITH resolved AS (
    SELECT s.*, pr.global_id AS player_global_id,
           direct_team.global_id AS direct_team_global_id,
           row_number() OVER (
               PARTITION BY pr.global_id, s.league_key, s.season, s.competition
               ORDER BY CASE WHEN s.source_team_id IS NULL THEN 0 ELSE 1 END,
                        COALESCE(s.games, 0) DESC,
                        s.source_team_id NULLS LAST
           ) AS rn
    FROM "AI_Source"."CompetitionStatsRawInternal" s
    JOIN "AI_Source"."CompetitionPlayerRegistryInternal" pr
      ON pr.league_key = s.league_key
     AND pr.season = s.season
     AND pr.source_player_id = s.source_player_id
    LEFT JOIN "AI_Source"."CompetitionTeamRegistryInternal" direct_team
      ON direct_team.league_key = s.league_key
     AND direct_team.season = s.season
     AND direct_team.source_team_id = s.source_team_id
    WHERE s.rating IS NOT NULL
),
scored AS (
    SELECT r.*,
           CASE
             WHEN count(*) OVER (
                    PARTITION BY r.league_key, r.season, r.competition
                  ) = 1 THEN 5.5
             ELSE 1.0 + 9.0 * percent_rank() OVER (
                    PARTITION BY r.league_key, r.season, r.competition
                    ORDER BY r.rating
                  )
           END AS rating_0_10
    FROM resolved r
    WHERE r.rn = 1
),
role_ranked AS (
    SELECT r.*, row_number() OVER (
        PARTITION BY r.league_key, r.season, r.source_player_id, r.competition
        ORDER BY r.source_player_id
    ) AS rn FROM "AI_Source"."CompetitionRolesRawInternal" r
),
onoff_ranked AS (
    SELECT o.*, row_number() OVER (
        PARTITION BY o.league_key, o.season, o.source_player_id, o.competition
        ORDER BY o.source_player_id
    ) AS rn FROM "AI_Source"."CompetitionOnOffRawInternal" o
),
clutch_ranked AS (
    SELECT c.*, row_number() OVER (
        PARTITION BY c.league_key, c.season, c.source_player_id, c.competition
        ORDER BY c.source_player_id
    ) AS rn FROM "AI_Source"."CompetitionClutchRawInternal" c
),
boxscore_starts AS (
    SELECT league_key, season, source_player_id, competition,
           count(DISTINCT game_id) FILTER (WHERE "AI_Source"."Truthy"(started_raw))::double precision AS games_started
    FROM "AI_Source"."CompetitionBoxscoreRawInternal"
    GROUP BY league_key, season, source_player_id, competition
),
roster_fallback AS (
    SELECT pr.global_id AS player_global_id, pr.league_key, pr.season,
           CASE WHEN count(DISTINCT tr.global_id) = 1 THEN min(tr.global_id) ELSE NULL END AS team_global_id
    FROM "AI_Source"."CompetitionPlayerRegistryInternal" pr
    LEFT JOIN "AI_Source"."CompetitionTeamRegistryInternal" tr
      ON tr.league_key = pr.league_key
     AND tr.season = pr.season
     AND pr.team_name IS NOT NULL
     AND (
         lower(btrim(pr.team_name)) = lower(btrim(tr.name))
         OR lower(btrim(pr.team_name)) = lower(btrim(COALESCE(tr.short_name, '')))
     )
    GROUP BY pr.global_id, pr.league_key, pr.season
)
SELECT
    "AI_Source"."StableId"('player', s.player_global_id) AS player_id,
    s.season,
    COALESCE(
        "AI_Source"."StableId"('team', s.direct_team_global_id),
        "AI_Source"."StableId"('team', rf.team_global_id)
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
    s.per AS per,
    s.ts_pct AS ts_pct,
    s.usg_pct AS usg_pct,
    s.bpm AS bpm,
    s.vorp AS vorp,
    s.win_shares AS win_shares,
    s.ast_ratio AS ast_ratio,
    s.reb_pct AS reb_pct,
    s.rating_0_10 AS rating,
    s.ortg AS ortg,
    s.drtg AS drtg,
    s.net_rtg AS net_rtg,
    COALESCE(s.ruolo_offensivo, rr.ruolo_offensivo, '') AS ruolo_offensivo,
    COALESCE(s.ruolo_difensivo, rr.ruolo_difensivo, '') AS ruolo_difensivo,
    COALESCE(s.ruolo_combinato, rr.ruolo_combinato, '') AS ruolo_combinato,
    COALESCE(s.on_net_rtg, oo.on_net_rtg, 0) AS on_net_rtg,
    COALESCE(s.off_net_rtg, oo.off_net_rtg, 0) AS off_net_rtg,
    COALESCE(s.net_rtg_diff, oo.net_rtg_diff, 0) AS net_rtg_diff,
    s.competition,
    s.spm AS spm,
    s.obpm AS obpm,
    s.dbpm AS dbpm,
    s.gm_sc AS gm_sc,
    s.fic AS fic,
    s.ows AS ows,
    s.dws AS dws,
    s.raptor_off AS raptor_off,
    s.raptor_def AS raptor_def,
    s.raptor_total AS raptor_total,
    s.lebron_off AS lebron_off,
    s.lebron_def AS lebron_def,
    s.lebron_total AS lebron_total,
    s.scoring_efficiency AS scoring_efficiency,
    s.ppsa AS ppsa,
    s.two_point_pct AS two_point_pct,
    s.tov_pct AS tov_pct,
    s.ast_pct AS ast_pct,
    s.stl_pct AS stl_pct,
    s.blk_pct AS blk_pct,
    s.orb_pct AS orb_pct,
    s.drb_pct AS drb_pct,
    s.three_par AS three_par,
    s.true_usg_pct AS true_usg_pct,
    s.foul_drawing_rate AS foul_drawing_rate,
    s.rf_per_game AS rf_per_game,
    s.hustle_index AS hustle_index,
    s.pts_per_40 AS pts_per_40,
    s.ast_per_40 AS ast_per_40,
    s.tr_per_40 AS tr_per_40,
    s.stl_per_40 AS stl_per_40,
    s.blk_per_40 AS blk_per_40,
    COALESCE(cr.clutch_games, 0) AS clutch_games,
    COALESCE(cr.clutch_pts, 0) AS clutch_pts,
    COALESCE(cr.clutch_ts_pct, 0) AS clutch_ts_pct,
    COALESCE(cr.clutch_ast_to_tov, 0) AS clutch_ast_to_tov,
    COALESCE(cr.clutch_net_rtg, 0) AS clutch_net_rtg,
    COALESCE(cr.clutch_efg_pct, 0) AS clutch_efg_pct,
    COALESCE(oo.ortg_on, 0) AS ortg_on,
    COALESCE(oo.ortg_off, 0) AS ortg_off,
    COALESCE(oo.ortg_diff, 0) AS ortg_diff,
    COALESCE(s.games_started, bs.games_started, 0)::integer AS games_started,
    COALESCE(
        s.starter_pct,
        CASE WHEN COALESCE(s.games, 0) > 0
             THEN LEAST(1.0, COALESCE(bs.games_started, 0) / s.games)
             ELSE 0 END
    ) AS starter_pct,
    s.league_key
FROM scored s
LEFT JOIN roster_fallback rf
  ON rf.player_global_id = s.player_global_id
 AND rf.league_key = s.league_key
 AND rf.season = s.season
LEFT JOIN role_ranked rr
  ON rr.league_key = s.league_key AND rr.season = s.season
 AND rr.source_player_id = s.source_player_id AND rr.competition = s.competition AND rr.rn = 1
LEFT JOIN onoff_ranked oo
  ON oo.league_key = s.league_key AND oo.season = s.season
 AND oo.source_player_id = s.source_player_id AND oo.competition = s.competition AND oo.rn = 1
LEFT JOIN clutch_ranked cr
  ON cr.league_key = s.league_key AND cr.season = s.season
 AND cr.source_player_id = s.source_player_id AND cr.competition = s.competition AND cr.rn = 1
LEFT JOIN boxscore_starts bs
  ON bs.league_key = s.league_key AND bs.season = s.season
 AND bs.source_player_id = s.source_player_id AND bs.competition = s.competition
;

CREATE VIEW "AI_Source"."TeamCompetitionStats" AS
WITH resolved AS (
    SELECT ts.*, tr.global_id, tr.name, tr.short_name,
           row_number() OVER (
               PARTITION BY tr.global_id, ts.league_key, ts.season, ts.competition
               ORDER BY COALESCE(ts.games, 0) DESC, ts.source_team_id
           ) AS rn
    FROM "AI_Source"."CompetitionTeamStatsRawInternal" ts
    JOIN "AI_Source"."CompetitionTeamRegistryInternal" tr
      ON tr.league_key = ts.league_key
     AND tr.season = ts.season
     AND tr.source_team_id = ts.source_team_id
),
star_usage AS (
    SELECT team_id, league_id, season, competition,
           LEAST(
               0.60,
               GREATEST(
                   0.10,
                   CASE WHEN max(usg_pct) > 1.5 THEN max(usg_pct) / 100.0 ELSE max(usg_pct) END
               )
           ) AS star_player_usage
    FROM "AI_Source"."PlayerCompetitionStats"
    WHERE team_id IS NOT NULL AND usg_pct IS NOT NULL
    GROUP BY team_id, league_id, season, competition
)
SELECT
    "AI_Source"."StableId"('team', r.global_id) AS team_id,
    r.global_id,
    r.name,
    "AI_Source"."StableId"('league', r.league_key) AS league_id,
    r.league_key,
    r.season,
    r.competition,
    COALESCE(r.games, 0)::integer AS games_played,
    COALESCE(r.pace, 75.0) AS pace,
    COALESCE(r.ortg, 110.0) AS offensive_rating,
    COALESCE(r.drtg, 110.0) AS defensive_rating,
    COALESCE(
        r.three_par,
        CASE WHEN COALESCE(r.fg3a, 0) + COALESCE(r.fg2a, 0) > 0
             THEN r.fg3a / (r.fg3a + r.fg2a) ELSE NULL END,
        0.35
    ) AS three_point_attempt_rate,
    COALESCE(
        r.ast_per_game,
        CASE WHEN COALESCE(r.games, 0) > 0 THEN r.ast_total / r.games ELSE NULL END,
        20.0
    ) AS assists_per_game,
    COALESCE(
        r.net_rtg,
        COALESCE(r.ortg, 110.0) - COALESCE(r.drtg, 110.0)
    ) AS net_rtg,
    COALESCE(su.star_player_usage, 0.25) AS star_player_usage,
    COALESCE(r.short_name, '') AS short_name
FROM resolved r
LEFT JOIN star_usage su
  ON su.team_id = "AI_Source"."StableId"('team', r.global_id)
 AND su.league_id = "AI_Source"."StableId"('league', r.league_key)
 AND su.season = r.season
 AND su.competition = r.competition
WHERE r.rn = 1;

COMMIT;

-- Verification:
-- SELECT competition, count(*) FROM "AI_Source"."PlayerCompetitionStats" GROUP BY competition ORDER BY competition;
-- SELECT competition, count(*) FROM "AI_Source"."TeamCompetitionStats" GROUP BY competition ORDER BY competition;
-- SELECT player_id, league_id, season, competition, count(*)
-- FROM "AI_Source"."PlayerCompetitionStats"
-- GROUP BY player_id, league_id, season, competition HAVING count(*) > 1;
-- SELECT team_id, league_id, season, competition, count(*)
-- FROM "AI_Source"."TeamCompetitionStats"
-- GROUP BY team_id, league_id, season, competition HAVING count(*) > 1;

-- =========================================================================
-- SEZIONE 4 — Contratto simulation/causal (ex ai_source_simulation.sql)
-- =========================================================================

-- Optional possession-level source contract for the Simulation & Causal Engine.
-- Run after ai_source_schema.sql and ai_source_competition.sql.
-- Physical source tables are read-only; missing feeds produce empty views.

BEGIN;

DROP VIEW IF EXISTS "AI_Source"."SimulationCausalPanel";
DROP VIEW IF EXISTS "AI_Source"."SimulationShotProfiles";
DROP VIEW IF EXISTS "AI_Source"."SimulationPlayTypeStats";
DROP VIEW IF EXISTS "AI_Source"."SimulationLineupStints";
DROP VIEW IF EXISTS "AI_Source"."SimulationPbpEvents";
SELECT "AI_Source"."DropRawObject"('SimulationLineupRawInternal');
SELECT "AI_Source"."DropRawObject"('SimulationPbpRawInternal');

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

-- =========================================================================
-- SEZIONE 5 — Materializzazione del livello *RawInternal (performance)
--
-- A questo punto tutte le viste *RawInternal (base + competition +
-- simulation) sono state RIGENERATE come VIEW normali dalla discovery
-- dinamica sopra (leghe nuove incluse). Le trasformiamo in MATERIALIZED
-- VIEW indicizzate cosi' il parsing JSONB/regex si fa una volta sola qui,
-- non ad ogni query su Players/PlayerStats/PlayerCompetitionStats/
-- SimulationPbpEvents ecc.
-- =========================================================================

DO $$
DECLARE
    v text;
    names text[] := ARRAY[
        'PlayerRegistryInternal', 'TeamRegistryInternal', 'StatsRawInternal',
        'TeamStatsRawInternal', 'RolesRawInternal', 'OnOffRawInternal',
        'ClutchRawInternal', 'BoxscoreRawInternal',
        'CompetitionPlayerRegistryInternal', 'CompetitionTeamRegistryInternal',
        'CompetitionStatsRawInternal', 'CompetitionTeamStatsRawInternal',
        'CompetitionRolesRawInternal', 'CompetitionOnOffRawInternal',
        'CompetitionClutchRawInternal', 'CompetitionBoxscoreRawInternal',
        'SimulationPbpRawInternal', 'SimulationLineupRawInternal'
    ];
BEGIN
    FOREACH v IN ARRAY names LOOP
        -- puo' non esistere se una tabella sorgente opzionale (es. Clutch)
        -- non e' presente per nessuna lega: la discovery produce comunque
        -- una vista vuota, quindi esiste sempre come VIEW a questo punto.
        IF EXISTS (SELECT 1 FROM pg_views WHERE schemaname = 'AI_Source' AND viewname = v) THEN
            EXECUTE format('ALTER VIEW "AI_Source".%I RENAME TO %I', v, v || '_Def');
            EXECUTE format(
                'CREATE MATERIALIZED VIEW "AI_Source".%I AS SELECT * FROM "AI_Source".%I',
                v, v || '_Def'
            );
        END IF;
    END LOOP;
END;
$$;

-- -------------------------------------------------------------------------
-- Indici mirati sulle chiavi usate nei JOIN/ROW_NUMBER/funzioni per-riga a
-- valle. CREATE INDEX IF NOT EXISTS: le matview sono nuove ad ogni run,
-- quindi in pratica vengono sempre create, ma la clausola le rende sicure
-- anche in caso di run parziali.
-- -------------------------------------------------------------------------

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='PlayerRegistryInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_playerreg_lookup ON "AI_Source"."PlayerRegistryInternal" (league_key, season, source_player_id);
        CREATE INDEX IF NOT EXISTS ix_playerreg_global  ON "AI_Source"."PlayerRegistryInternal" (global_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='TeamRegistryInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_teamreg_lookup ON "AI_Source"."TeamRegistryInternal" (league_key, season, source_team_id);
        CREATE INDEX IF NOT EXISTS ix_teamreg_global  ON "AI_Source"."TeamRegistryInternal" (global_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='StatsRawInternal') THEN
        -- Tentativo di indice UNIQUE (abilita REFRESH CONCURRENTLY, piu' veloce
        -- e non bloccante). Un giocatore trasferito a meta' stagione puo' avere
        -- piu' righe per la stessa (league_key, season, source_player_id,
        -- competition), una per squadra: in quel caso l'indice fallisce e si
        -- ricade su un indice normale, senza bloccare lo script.
        BEGIN
            CREATE UNIQUE INDEX IF NOT EXISTS ux_stats_refresh ON "AI_Source"."StatsRawInternal"
                (league_key, season, source_player_id, competition);
        EXCEPTION WHEN OTHERS THEN
            DROP INDEX IF EXISTS "AI_Source".ux_stats_refresh;
            CREATE INDEX IF NOT EXISTS ix_stats_dedupe ON "AI_Source"."StatsRawInternal"
                (league_key, season, source_player_id, competition);
        END;
        CREATE INDEX IF NOT EXISTS ix_stats_team ON "AI_Source"."StatsRawInternal" (league_key, season, source_team_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='TeamStatsRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_teamstats_lookup ON "AI_Source"."TeamStatsRawInternal" (league_key, season, source_team_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='RolesRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_roles_lookup ON "AI_Source"."RolesRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='OnOffRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_onoff_lookup ON "AI_Source"."OnOffRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='ClutchRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_clutch_lookup ON "AI_Source"."ClutchRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='BoxscoreRawInternal') THEN
        BEGIN
            CREATE UNIQUE INDEX IF NOT EXISTS ux_boxscore_refresh ON "AI_Source"."BoxscoreRawInternal"
                (league_key, season, source_player_id, game_id, competition);
        EXCEPTION WHEN OTHERS THEN
            DROP INDEX IF EXISTS "AI_Source".ux_boxscore_refresh;
            CREATE INDEX IF NOT EXISTS ix_boxscore_dedupe ON "AI_Source"."BoxscoreRawInternal"
                (league_key, season, source_player_id, game_id, competition);
        END;
        CREATE INDEX IF NOT EXISTS ix_boxscore_team ON "AI_Source"."BoxscoreRawInternal" (league_key, season, source_team_id);
    END IF;

    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionPlayerRegistryInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_cplayerreg_lookup ON "AI_Source"."CompetitionPlayerRegistryInternal" (league_key, season, source_player_id);
        CREATE INDEX IF NOT EXISTS ix_cplayerreg_sim     ON "AI_Source"."CompetitionPlayerRegistryInternal" (lower(league_key), season, btrim(source_player_id));
        CREATE INDEX IF NOT EXISTS ix_cplayerreg_global   ON "AI_Source"."CompetitionPlayerRegistryInternal" (global_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionTeamRegistryInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_cteamreg_lookup ON "AI_Source"."CompetitionTeamRegistryInternal" (league_key, season, source_team_id);
        CREATE INDEX IF NOT EXISTS ix_cteamreg_sim     ON "AI_Source"."CompetitionTeamRegistryInternal" (lower(league_key), season, btrim(source_team_id));
        CREATE INDEX IF NOT EXISTS ix_cteamreg_global   ON "AI_Source"."CompetitionTeamRegistryInternal" (global_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionStatsRawInternal') THEN
        BEGIN
            CREATE UNIQUE INDEX IF NOT EXISTS ux_cstats_refresh ON "AI_Source"."CompetitionStatsRawInternal"
                (league_key, season, source_player_id, competition);
        EXCEPTION WHEN OTHERS THEN
            DROP INDEX IF EXISTS "AI_Source".ux_cstats_refresh;
            CREATE INDEX IF NOT EXISTS ix_cstats_dedupe ON "AI_Source"."CompetitionStatsRawInternal"
                (league_key, season, source_player_id, competition);
        END;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionTeamStatsRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_cteamstats_lookup ON "AI_Source"."CompetitionTeamStatsRawInternal" (league_key, season, source_team_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionRolesRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_croles_lookup ON "AI_Source"."CompetitionRolesRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionOnOffRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_conoff_lookup ON "AI_Source"."CompetitionOnOffRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionClutchRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_cclutch_lookup ON "AI_Source"."CompetitionClutchRawInternal" (league_key, season, source_player_id, competition);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='CompetitionBoxscoreRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_cboxscore_lookup ON "AI_Source"."CompetitionBoxscoreRawInternal" (league_key, season, source_player_id, competition);
    END IF;

    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='SimulationPbpRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_simpbp_lookup ON "AI_Source"."SimulationPbpRawInternal" (league_key, season, competition, game_id);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname='SimulationLineupRawInternal') THEN
        CREATE INDEX IF NOT EXISTS ix_simlineup_lookup ON "AI_Source"."SimulationLineupRawInternal" (league_key, season, competition);
    END IF;
END;
$$;

ANALYZE;

-- -------------------------------------------------------------------------
-- Funzione di refresh leggero: da usare quando cambiano SOLO i dati (nuovo
-- import di una lega gia' esistente, nuova giornata di boxscore) e NON la
-- struttura. In quel caso non serve rilanciare l'intero script: basta
-- SELECT "AI_Source"."RefreshRawLayer"();
-- Quando invece aggiungi una lega/tabella nuova, rilancia questo script
-- intero: la discovery dinamica la trova da sola.
-- -------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION "AI_Source"."RefreshRawLayer"()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    v text;
    names text[] := ARRAY[
        'PlayerRegistryInternal', 'TeamRegistryInternal', 'StatsRawInternal',
        'TeamStatsRawInternal', 'RolesRawInternal', 'OnOffRawInternal',
        'ClutchRawInternal', 'BoxscoreRawInternal',
        'CompetitionPlayerRegistryInternal', 'CompetitionTeamRegistryInternal',
        'CompetitionStatsRawInternal', 'CompetitionTeamStatsRawInternal',
        'CompetitionRolesRawInternal', 'CompetitionOnOffRawInternal',
        'CompetitionClutchRawInternal', 'CompetitionBoxscoreRawInternal',
        'SimulationPbpRawInternal', 'SimulationLineupRawInternal'
    ];
BEGIN
    FOREACH v IN ARRAY names LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_matviews WHERE schemaname='AI_Source' AND matviewname=v) THEN
            CONTINUE;
        END IF;
        BEGIN
            EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY "AI_Source".%I', v);
        EXCEPTION WHEN OTHERS THEN
            EXECUTE format('REFRESH MATERIALIZED VIEW "AI_Source".%I', v);
        END;
    END LOOP;
    RAISE NOTICE 'AI_Source raw layer refreshed at %', clock_timestamp();
END;
$$;

-- Uso quotidiano dopo un caricamento dati (stessa lega, stessa struttura):
--   SELECT "AI_Source"."RefreshRawLayer"();
-- Con pg_cron (se disponibile):
--   SELECT cron.schedule('ai_source_refresh', '*/15 * * * *',
--          $$SELECT "AI_Source"."RefreshRawLayer"()$$);
--
-- Aggiunta di una nuova lega/tabella: rilancia semplicemente questo intero
-- file (ai_source_full.sql). Nessun passaggio manuale necessario.

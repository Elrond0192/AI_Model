-- ============================================================================
-- AI_Source — minimal, PostgreSQL-native canonical cache
--
-- This file owns ONLY the six canonical datasets consumed by AI_Model:
--   Leagues
--   Teams
--   Players
--   TeamPlayerRelations
--   PlayerCompetitionStats
--   TeamCompetitionStats
--
-- There are intentionally NO RawInternal views, no simulation feeds, no
-- persistent helper functions and no JSONB row parsing.
--
-- Source data is normalized once into physical canonical tables with CTAS.
-- This is important because CREATE TABLE AS can use a parallel plan for the
-- underlying SELECT when PostgreSQL decides that parallel execution is useful.
--
-- Re-run this file after adding/changing source leagues. Daily serving data
-- refreshes should be handled by the serving layer, not by rebuilding this
-- bootstrap contract.
-- ============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS "AI_Source";

-- ---------------------------------------------------------------------------
-- Remove only objects owned by the old ai_source_full implementation.
-- Serving objects (Training*, Scenario*, ServingRefreshState, etc.) are kept.
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS
    "AI_Source"."Leagues",
    "AI_Source"."Teams",
    "AI_Source"."Players",
    "AI_Source"."TeamPlayerRelations",
    "AI_Source"."PlayerCompetitionStats",
    "AI_Source"."TeamCompetitionStats"
CASCADE;

DROP VIEW IF EXISTS
    "AI_Source"."Leagues",
    "AI_Source"."Teams",
    "AI_Source"."Players",
    "AI_Source"."TeamPlayerRelations",
    "AI_Source"."PlayerCompetitionStats",
    "AI_Source"."TeamCompetitionStats",
    "AI_Source"."PlayerStats"
CASCADE;

DO $cleanup$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT c.relkind, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'AI_Source'
          AND (
              c.relname LIKE '%RawInternal'
              OR c.relname LIKE 'Simulation%'
          )
          AND c.relkind IN ('r', 'p', 'v', 'm')
    LOOP
        IF r.relkind = 'm' THEN
            EXECUTE format(
                'DROP MATERIALIZED VIEW IF EXISTS "AI_Source".%I CASCADE',
                r.relname
            );
        ELSIF r.relkind IN ('v') THEN
            EXECUTE format(
                'DROP VIEW IF EXISTS "AI_Source".%I CASCADE',
                r.relname
            );
        ELSE
            EXECUTE format(
                'DROP TABLE IF EXISTS "AI_Source".%I CASCADE',
                r.relname
            );
        END IF;
    END LOOP;
END;
$cleanup$;

-- Old persistent helpers are no longer part of AI_Source. Drop exact
-- signatures without touching the independent serving procedures/functions.
DROP FUNCTION IF EXISTS "AI_Source"."DropRawObject"(text);
DROP FUNCTION IF EXISTS "AI_Source"."LowerKeys"(jsonb);
DROP FUNCTION IF EXISTS "AI_Source"."TextValue"(jsonb, text[]);
DROP FUNCTION IF EXISTS "AI_Source"."NumericValue"(jsonb, text[]);
DROP FUNCTION IF EXISTS "AI_Source"."IntValue"(jsonb, text[]);
DROP FUNCTION IF EXISTS "AI_Source"."DateValue"(jsonb, text[]);
DROP FUNCTION IF EXISTS "AI_Source"."Truthy"(text);
DROP FUNCTION IF EXISTS "AI_Source"."StableId"(text, text);
DROP FUNCTION IF EXISTS "AI_Source"."CompetitionCode"(text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayerId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationTeamId"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationLineupIds"(text, integer, text);
DROP FUNCTION IF EXISTS "AI_Source"."SimulationPlayType"(text);
DROP FUNCTION IF EXISTS "AI_Source"."RefreshRawLayer"();

-- ---------------------------------------------------------------------------
-- Temporary code-generation helpers.
--
-- These functions live in pg_temp and disappear automatically with the psql
-- session. They are executed only while generating SQL, never once per source
-- row. The data path therefore contains direct column references and native
-- PostgreSQL casts instead of to_jsonb/jsonb_each/PLpgSQL parsing.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pg_temp.ai_find_column(
    p_schema text,
    p_table text,
    p_candidates text[]
)
RETURNS text
LANGUAGE plpgsql
AS $fn$
DECLARE
    result text;
BEGIN
    SELECT a.attname
      INTO result
    FROM pg_catalog.pg_attribute a
    JOIN pg_catalog.pg_class c
      ON c.oid = a.attrelid
    JOIN pg_catalog.pg_namespace n
      ON n.oid = c.relnamespace
    WHERE n.nspname = p_schema
      AND c.relname = p_table
      AND a.attnum > 0
      AND NOT a.attisdropped
      AND lower(a.attname) = ANY (p_candidates)
    ORDER BY array_position(p_candidates, lower(a.attname)), a.attnum
    LIMIT 1;

    RETURN result;
END;
$fn$;

CREATE OR REPLACE FUNCTION pg_temp.ai_expr(
    p_schema text,
    p_table text,
    p_alias text,
    p_kind text,
    p_candidates text[]
)
RETURNS text
LANGUAGE plpgsql
AS $fn$
DECLARE
    att text;
    typcategory "char";
    typname text;
    q text;
BEGIN
    SELECT a.attname, t.typcategory, t.typname
      INTO att, typcategory, typname
    FROM pg_catalog.pg_attribute a
    JOIN pg_catalog.pg_class c
      ON c.oid = a.attrelid
    JOIN pg_catalog.pg_namespace n
      ON n.oid = c.relnamespace
    JOIN pg_catalog.pg_type t
      ON t.oid = a.atttypid
    WHERE n.nspname = p_schema
      AND c.relname = p_table
      AND a.attnum > 0
      AND NOT a.attisdropped
      AND lower(a.attname) = ANY (p_candidates)
    ORDER BY array_position(p_candidates, lower(a.attname)), a.attnum
    LIMIT 1;

    IF att IS NULL THEN
        IF p_kind = 'text' THEN
            RETURN 'NULL::text';
        ELSIF p_kind = 'numeric' THEN
            RETURN 'NULL::double precision';
        ELSIF p_kind = 'int' THEN
            RETURN 'NULL::bigint';
        ELSIF p_kind = 'date' THEN
            RETURN 'NULL::date';
        ELSIF p_kind = 'bool' THEN
            RETURN 'NULL::boolean';
        END IF;
        RAISE EXCEPTION 'Unsupported ai_expr kind: %', p_kind;
    END IF;

    q := format('%I.%I', p_alias, att);

    IF p_kind = 'text' THEN
        RETURN format('NULLIF(btrim(%s::text), '''')', q);

    ELSIF p_kind = 'numeric' THEN
        IF typcategory = 'N' THEN
            RETURN format('%s::double precision', q);
        END IF;

        RETURN format(
            '(CASE
                WHEN NULLIF(btrim(%1$s::text), '''') ~
                     ''^[+-]?([0-9]+([.,][0-9]*)?|[.,][0-9]+)([eE][+-]?[0-9]+)?%%?$''
                THEN replace(
                         replace(NULLIF(btrim(%1$s::text), ''''), ''%%'', ''''),
                         '','', ''.''
                     )::double precision
                ELSE NULL::double precision
              END)',
            q
        );

    ELSIF p_kind = 'int' THEN
        IF typcategory = 'N' THEN
            RETURN format('trunc(%s::double precision)::bigint', q);
        END IF;

        RETURN format(
            '(CASE
                WHEN NULLIF(btrim(%1$s::text), '''') ~
                     ''^[+-]?[0-9]+([.][0-9]+)?$''
                THEN btrim(%1$s::text)::bigint
                ELSE NULL::bigint
              END)',
            q
        );

    ELSIF p_kind = 'date' THEN
        IF typname IN ('date', 'timestamp', 'timestamptz') THEN
            RETURN format('%s::date', q);
        END IF;

        RETURN format(
            '(CASE
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{4}-[0-9]{2}-[0-9]{2}''
                    THEN substr(btrim(%1$s::text), 1, 10)::date
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{4}/[0-9]{2}/[0-9]{2}$''
                    THEN to_date(btrim(%1$s::text), ''YYYY/MM/DD'')
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{2}/[0-9]{2}/[0-9]{4}$''
                    THEN to_date(btrim(%1$s::text), ''DD/MM/YYYY'')
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{2}-[0-9]{2}-[0-9]{4}$''
                    THEN to_date(btrim(%1$s::text), ''DD-MM-YYYY'')
                ELSE NULL::date
              END)',
            q
        );

    ELSIF p_kind = 'bool' THEN
        IF typcategory = 'B' THEN
            RETURN format('%s::boolean', q);
        END IF;

        RETURN format(
            '(lower(btrim(coalesce(%1$s::text, ''''))) IN
                (''1'', ''t'', ''true'', ''y'', ''yes''))',
            q
        );
    END IF;

    RAISE EXCEPTION 'Unsupported ai_expr kind: %', p_kind;
END;
$fn$;

-- Generates the canonical competition code. Fast-paths known values and only
-- invokes regexp_replace for genuinely non-standard labels.
CREATE OR REPLACE FUNCTION pg_temp.ai_comp_expr(
    p_schema text,
    p_table text,
    p_alias text,
    p_candidates text[]
)
RETURNS text
LANGUAGE plpgsql
AS $fn$
DECLARE
    v text;
BEGIN
    v := pg_temp.ai_expr(p_schema, p_table, p_alias, 'text', p_candidates);

    RETURN format(
        '(CASE upper(coalesce(%1$s, ''RS''))
            WHEN '''' THEN ''RS''
            WHEN ''REGULAR'' THEN ''RS''
            WHEN ''REGULAR SEASON'' THEN ''RS''
            WHEN ''REG SEASON'' THEN ''RS''
            WHEN ''RS'' THEN ''RS''
            WHEN ''PLAYOFF'' THEN ''PO''
            WHEN ''PLAYOFFS'' THEN ''PO''
            WHEN ''POSTSEASON'' THEN ''PO''
            WHEN ''PO'' THEN ''PO''
            WHEN ''TOTAL'' THEN ''TOT''
            WHEN ''ALL'' THEN ''TOT''
            WHEN ''TOT'' THEN ''TOT''
            WHEN ''SUPER CUP'' THEN ''SUPERCUP''
            WHEN ''SUPERCUP'' THEN ''SUPERCUP''
            WHEN ''CUP'' THEN ''CUP''
            ELSE regexp_replace(
                upper(regexp_replace(btrim(%1$s), ''[[:space:]_-]+'', ''_'', ''g'')),
                ''[^A-Z0-9_]'', '''', ''g''
            )
         END)',
        v
    );
END;
$fn$;

-- ---------------------------------------------------------------------------
-- Dynamic source metadata and canonical expressions.
-- No source table is copied wholesale; only columns actually consumed by the
-- model are selected.
-- ---------------------------------------------------------------------------

DO $build$
DECLARE
    r record;
    body text := '';
    league_key text;
    season_expr text;
    player_id_expr text;
    team_id_expr text;
    comp_expr text;
    rating_expr text;
    direct_team_global_expr text;
    player_global_expr text;
    p_team_name_expr text;

    -- Optional joins are emitted only when the main stats table does not
    -- already contain the corresponding fields.
    role_join text := '';
    onoff_join text := '';
    clutch_join text := '';
    boxscore_join text := '';

    role_off_expr text;
    role_def_expr text;
    role_combo_expr text;
    on_net_expr text;
    off_net_expr text;
    net_diff_expr text;
    ortg_on_expr text;
    ortg_off_expr text;
    ortg_diff_expr text;
    clutch_games_expr text;
    clutch_pts_expr text;
    clutch_ts_expr text;
    clutch_ast_expr text;
    clutch_net_expr text;
    clutch_efg_expr text;
    games_started_expr text;
    starter_pct_expr text;
    box_started_expr text;

    player_table text;
    role_table text;
    onoff_table text;
    clutch_table text;
    boxscore_table text;
BEGIN
    FOR r IN
        SELECT
            n.nspname AS table_schema,
            c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n
          ON n.oid = c.relnamespace
        WHERE n.nspname = 'Analisi'
          AND lower(c.relname) LIKE 'advancedstats_player_%'
          AND c.relkind IN ('r', 'p')
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid
                AND a.attnum > 0
                AND NOT a.attisdropped
                AND lower(a.attname) = 'season'
          )
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid
                AND a.attnum > 0
                AND NOT a.attisdropped
                AND lower(a.attname) IN ('id', 'playerid', 'idplayer')
          )
        ORDER BY c.relname
    LOOP
        league_key := regexp_replace(r.table_name, '^AdvancedStats_Player_', '', 'i');
        player_table := league_key;

        season_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'int', ARRAY['season']);
        player_id_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['id', 'playerid', 'idplayer']);
        team_id_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['teamid', 'idteam']);
        comp_expr := pg_temp.ai_comp_expr(r.table_schema, r.table_name, 's', ARRAY['competition']);
        rating_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['vallegapergame', 'rating']);
        p_team_name_expr := pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['teamname']);

        player_global_expr :=
            format(
                'coalesce(%1$s, %2$L::text || '':'' || %3$s)',
                pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['idglobal', 'globalid']),
                league_key,
                pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id'])
            );

        direct_team_global_expr :=
            pg_temp.ai_expr('Anagrafiche', 'Team_' || league_key, 'dt', 'text', ARRAY['idglobal', 'globalid']);

        role_off_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolooffensivo']);
        role_def_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolodifensivo']);
        role_combo_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolocombinato']);

        IF pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolooffensivo']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolodifensivo']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolocombinato']) IS NULL
        THEN
            role_table := 'PlayerRoles_' || league_key;
            IF EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'Analisi'
                  AND c.relname = role_table
                  AND c.relkind IN ('r', 'p')
            ) THEN
                role_join := format(
                    'LEFT JOIN (
                        SELECT DISTINCT ON (season, source_player_id, competition)
                               season, source_player_id, competition,
                               ruolo_offensivo, ruolo_difensivo, ruolo_combinato
                        FROM (
                            SELECT
                                %1$s AS season,
                                %2$s AS source_player_id,
                                %3$s AS competition,
                                %4$s AS ruolo_offensivo,
                                %5$s AS ruolo_difensivo,
                                %6$s AS ruolo_combinato
                            FROM %7$I.%8$I rr0
                            WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                        ) rr1
                        ORDER BY season, source_player_id, competition
                    ) rr
                      ON rr.season = %9$s
                     AND rr.source_player_id = %10$s
                     AND rr.competition = %11$s',
                    pg_temp.ai_expr('Analisi', role_table, 'rr0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Analisi', role_table, 'rr0', 'text', ARRAY['id', 'playerid', 'idplayer']),
                    pg_temp.ai_comp_expr('Analisi', role_table, 'rr0', ARRAY['competition']),
                    pg_temp.ai_expr('Analisi', role_table, 'rr0', 'text', ARRAY['ruolooffensivo']),
                    pg_temp.ai_expr('Analisi', role_table, 'rr0', 'text', ARRAY['ruolodifensivo']),
                    pg_temp.ai_expr('Analisi', role_table, 'rr0', 'text', ARRAY['ruolocombinato']),
                    'Analisi', role_table,
                    season_expr, player_id_expr, comp_expr
                );
                role_off_expr := 'rr.ruolo_offensivo';
                role_def_expr := 'rr.ruolo_difensivo';
                role_combo_expr := 'rr.ruolo_combinato';
            END IF;
        END IF;

        on_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_on', 'onnetrtg']);
        off_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_off', 'offnetrtg']);
        net_diff_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_diff']);
        ortg_on_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_on']);
        ortg_off_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_off']);
        ortg_diff_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_diff']);

        IF pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['netrtg_on', 'onnetrtg']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['netrtg_off', 'offnetrtg']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['netrtg_diff']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ortg_on']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ortg_off']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ortg_diff']) IS NULL
        THEN
            onoff_table := 'AdvancedStatsOnOffCourt_' || league_key;
            IF EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'Analisi'
                  AND c.relname = onoff_table
                  AND c.relkind IN ('r', 'p')
            ) THEN
                onoff_join := format(
                    'LEFT JOIN (
                        SELECT DISTINCT ON (season, source_player_id, competition)
                               season, source_player_id, competition,
                               on_net_rtg, off_net_rtg, net_rtg_diff,
                               ortg_on, ortg_off, ortg_diff
                        FROM (
                            SELECT
                                %1$s AS season,
                                %2$s AS source_player_id,
                                %3$s AS competition,
                                %4$s AS on_net_rtg,
                                %5$s AS off_net_rtg,
                                %6$s AS net_rtg_diff,
                                %7$s AS ortg_on,
                                %8$s AS ortg_off,
                                %9$s AS ortg_diff
                            FROM %10$I.%11$I oo0
                            WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                        ) oo1
                        ORDER BY season, source_player_id, competition
                    ) oo
                      ON oo.season = %12$s
                     AND oo.source_player_id = %13$s
                     AND oo.competition = %14$s',
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'text', ARRAY['id', 'playerid', 'idplayer']),
                    pg_temp.ai_comp_expr('Analisi', onoff_table, 'oo0', ARRAY['competition']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['netrtg_on']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['netrtg_off']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['netrtg_diff']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['ortg_on']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['ortg_off']),
                    pg_temp.ai_expr('Analisi', onoff_table, 'oo0', 'numeric', ARRAY['ortg_diff']),
                    'Analisi', onoff_table,
                    season_expr, player_id_expr, comp_expr
                );
                on_net_expr := 'oo.on_net_rtg';
                off_net_expr := 'oo.off_net_rtg';
                net_diff_expr := 'oo.net_rtg_diff';
                ortg_on_expr := 'oo.ortg_on';
                ortg_off_expr := 'oo.ortg_off';
                ortg_diff_expr := 'oo.ortg_diff';
            END IF;
        END IF;

        clutch_games_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchgames', 'games']);
        clutch_pts_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchpts', 'pts']);
        clutch_ts_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchtspct', 'tspct']);
        clutch_ast_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['asttotovratio', 'clutchasttotov']);
        clutch_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchnetrtg', 'netrtg']);
        clutch_efg_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchefgpct', 'efgpct']);

        IF pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchgames', 'games']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchpts', 'pts']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchtspct', 'tspct']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['asttotovratio', 'clutchasttotov']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchnetrtg', 'netrtg']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchefgpct', 'efgpct']) IS NULL
        THEN
            clutch_table := 'AdvancedStats_Clutch_' || league_key;
            IF EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'Analisi'
                  AND c.relname = clutch_table
                  AND c.relkind IN ('r', 'p')
            ) THEN
                clutch_join := format(
                    'LEFT JOIN (
                        SELECT DISTINCT ON (season, source_player_id, competition)
                               season, source_player_id, competition,
                               clutch_games, clutch_pts, clutch_ts_pct,
                               clutch_ast_to_tov, clutch_net_rtg, clutch_efg_pct
                        FROM (
                            SELECT
                                %1$s AS season,
                                %2$s AS source_player_id,
                                %3$s AS competition,
                                %4$s AS clutch_games,
                                %5$s AS clutch_pts,
                                %6$s AS clutch_ts_pct,
                                %7$s AS clutch_ast_to_tov,
                                %8$s AS clutch_net_rtg,
                                %9$s AS clutch_efg_pct
                            FROM %10$I.%11$I cl0
                            WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                        ) cl1
                        ORDER BY season, source_player_id, competition
                    ) cl
                      ON cl.season = %12$s
                     AND cl.source_player_id = %13$s
                     AND cl.competition = %14$s',
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'text', ARRAY['id', 'playerid', 'idplayer']),
                    pg_temp.ai_comp_expr('Analisi', clutch_table, 'cl0', ARRAY['competition']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['clutchgames', 'games']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['clutchpts', 'pts']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['clutchtspct', 'tspct']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['asttotovratio', 'clutchasttotov']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['clutchnetrtg', 'netrtg']),
                    pg_temp.ai_expr('Analisi', clutch_table, 'cl0', 'numeric', ARRAY['clutchefgpct', 'efgpct']),
                    'Analisi', clutch_table,
                    season_expr, player_id_expr, comp_expr
                );
                clutch_games_expr := 'cl.clutch_games';
                clutch_pts_expr := 'cl.clutch_pts';
                clutch_ts_expr := 'cl.clutch_ts_pct';
                clutch_ast_expr := 'cl.clutch_ast_to_tov';
                clutch_net_expr := 'cl.clutch_net_rtg';
                clutch_efg_expr := 'cl.clutch_efg_pct';
            END IF;
        END IF;

        games_started_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['gamesstarted', 'games_started']);
        starter_pct_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['starterpct', 'starter_pct']);

        IF pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['gamesstarted', 'games_started']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['starterpct', 'starter_pct']) IS NULL
        THEN
            boxscore_table := league_key;
            IF EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'Boxscore'
                  AND c.relname = boxscore_table
                  AND c.relkind IN ('r', 'p')
            ) THEN
                box_started_expr :=
                    format(
                        '(SELECT count(DISTINCT bx_game_id)
                            FROM (
                                SELECT %1$s AS bx_game_id
                                FROM %2$I.%3$I b0
                                WHERE %4$s = %5$s
                                  AND %6$s = %7$s
                                  AND %8$s
                                  AND %9$s
                            ) bx
                           WHERE bx_game_id IS NOT NULL)',
                        pg_temp.ai_expr('Boxscore', boxscore_table, 'b0', 'text', ARRAY['game', 'idgame', 'gamecode']),
                        'Boxscore', boxscore_table,
                        pg_temp.ai_expr('Boxscore', boxscore_table, 'b0', 'int', ARRAY['season']),
                        season_expr,
                        pg_temp.ai_expr('Boxscore', boxscore_table, 'b0', 'text', ARRAY['id', 'idplayer', 'playerid']),
                        player_id_expr,
                        pg_temp.ai_expr('Boxscore', boxscore_table, 'b0', 'bool', ARRAY['sf', 'starter', 'isstarter']),
                        pg_temp.ai_expr('Boxscore', boxscore_table, 'b0', 'bool', ARRAY['sf', 'starter', 'isstarter'])
                    );
                boxscore_join := format(
                    'LEFT JOIN LATERAL (
                        SELECT
                            %1$s AS games_started
                        FROM (SELECT %2$s AS games_started) bx0
                    ) bs ON true',
                    box_started_expr,
                    box_started_expr
                );
                games_started_expr := 'bs.games_started';
                starter_pct_expr := format(
                    '(CASE WHEN coalesce(%1$s, 0) > 0
                           THEN least(1.0, coalesce(%2$s, 0) / %1$s)
                           ELSE 0 END)',
                    pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['games', 'gamesplayed']),
                    box_started_expr
                );
            END IF;
        END IF;

        body := body ||
            CASE WHEN body = '' THEN '' ELSE E'
UNION ALL
' END ||
            format($q$
                SELECT
                    %1$L::text AS league_key,
                    %2$s::integer AS season,
                    %3$s AS source_player_id,
                    %4$s AS source_team_id,
                    %5$s AS competition,
                    %6$s AS games,
                    %7$s AS games_started,
                    %8$s AS starter_pct,
                    %9$s AS minutes_total,
                    %10$s AS points_total,
                    %11$s AS rebounds_total,
                    %12$s AS offensive_rebounds_total,
                    %13$s AS defensive_rebounds_total,
                    %14$s AS assists_total,
                    %15$s AS steals_total,
                    %16$s AS blocks_total,
                    %17$s AS turnovers_total,
                    %18$s AS fouls_total,
                    %19$s AS fg_pct,
                    %20$s AS three_point_pct,
                    %21$s AS ft_pct,
                    %22$s AS plus_minus,
                    %23$s AS per,
                    %24$s AS ts_pct,
                    %25$s AS usg_pct,
                    %26$s AS bpm,
                    %27$s AS vorp,
                    %28$s AS win_shares,
                    %29$s AS ast_ratio,
                    %30$s AS reb_pct,
                    %31$s AS rating,
                    %32$s AS ortg,
                    %33$s AS drtg,
                    %34$s AS net_rtg,
                    %35$s AS ruolo_offensivo,
                    %36$s AS ruolo_difensivo,
                    %37$s AS ruolo_combinato,
                    %38$s AS on_net_rtg,
                    %39$s AS off_net_rtg,
                    %40$s AS net_rtg_diff,
                    %41$s AS spm,
                    %42$s AS obpm,
                    %43$s AS dbpm,
                    %44$s AS gm_sc,
                    %45$s AS fic,
                    %46$s AS ows,
                    %47$s AS dws,
                    %48$s AS raptor_off,
                    %49$s AS raptor_def,
                    %50$s AS raptor_total,
                    %51$s AS lebron_off,
                    %52$s AS lebron_def,
                    %53$s AS lebron_total,
                    %54$s AS scoring_efficiency,
                    %55$s AS ppsa,
                    %56$s AS two_point_pct,
                    %57$s AS tov_pct,
                    %58$s AS ast_pct,
                    %59$s AS stl_pct,
                    %60$s AS blk_pct,
                    %61$s AS orb_pct,
                    %62$s AS drb_pct,
                    %63$s AS three_par,
                    %64$s AS true_usg_pct,
                    %65$s AS foul_drawing_rate,
                    %66$s AS rf_per_game,
                    %67$s AS hustle_index,
                    %68$s AS pts_per_40,
                    %69$s AS ast_per_40,
                    %70$s AS tr_per_40,
                    %71$s AS stl_per_40,
                    %72$s AS blk_per_40,
                    %73$s AS clutch_games,
                    %74$s AS clutch_pts,
                    %75$s AS clutch_ts_pct,
                    %76$s AS clutch_ast_to_tov,
                    %77$s AS clutch_net_rtg,
                    %78$s AS clutch_efg_pct,
                    %79$s AS ortg_on,
                    %80$s AS ortg_off,
                    %81$s AS ortg_diff
                FROM %82$I.%83$I s
                JOIN %84$I.%85$I p
                  ON %86$s = %87$s
                 AND %88$s = %89$s
                LEFT JOIN %90$I.%91$I dt
                  ON %92$s = %93$s
                 AND %94$s = %95$s
                LEFT JOIN %96$I.%97$I nt
                  ON %98$s = %99$s
                 AND (
                    lower(btrim(%100$s)) = lower(btrim(%101$s))
                    OR lower(btrim(%100$s)) = lower(btrim(%102$s))
                 )
                %103$s
                %104$s
                %105$s
                %106$s
                WHERE %2$s IS NOT NULL
                  AND %3$s IS NOT NULL
                  AND %31$s IS NOT NULL
            $q$,
            league_key,
            season_expr,
            player_id_expr,
            team_id_expr,
            comp_expr,
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['games', 'gamesplayed']),
            games_started_expr,
            starter_pct_expr,
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['min', 'minutes']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pts', 'points']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tr', 'reb', 'rebounds']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['or', 'orb', 'offensiverebounds']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dr', 'drb', 'defensiverebounds']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ast', 'assists']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stl', 'steals']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blk', 'blocks']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['to', 'tov', 'turnovers']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pf', 'fouls', 'personalfouls']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['efgpct', 'fgpct', 'fg_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fg3pct', 'threepointpct', 'three_point_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ftpct', 'ft_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['plusminus', 'plus_minus']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pie', 'per']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tspct', 'ts_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['usgpct', 'usg_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['bpm']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['vorp']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ws', 'winshares']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astratio', 'ast_ratio']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['rebpct', 'reb_pct']),
            rating_expr,
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drtg']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg', 'net_rating']),
            role_off_expr,
            role_def_expr,
            role_combo_expr,
            on_net_expr,
            off_net_expr,
            net_diff_expr,
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['spm']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['obpm']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dbpm']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['gmsc', 'gm_sc']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fic']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ows']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dws']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptoroff', 'raptor_off']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptordef', 'raptor_def']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptortotal', 'raptor_total']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebronoff', 'lebron_off']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebrondef', 'lebron_def']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebrontotal', 'lebron_total']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['scoringefficiency', 'scoring_efficiency']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ppsa']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fg2pct', 'twopointpct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tovpct', 'tov_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astpct', 'ast_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stlpct', 'stl_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blkpct', 'blk_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['orebpct', 'orbpct', 'orb_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drebpct', 'drbpct', 'drb_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['threepar', 'three_par']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tusgpct', 'trueusgpct', 'true_usg_pct']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fouldrawingrate', 'foul_drawing_rate']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['rfpergame', 'rf_per_game']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['hustleindex', 'hustle_index']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ptsper40', 'pts_per_40']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astper40', 'ast_per_40']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['trper40', 'rebper40', 'tr_per_40']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stlper40', 'stl_per_40']),
            pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blkper40', 'blk_per_40']),
            clutch_games_expr,
            clutch_pts_expr,
            clutch_ts_expr,
            clutch_ast_expr,
            clutch_net_expr,
            clutch_efg_expr,
            ortg_on_expr,
            ortg_off_expr,
            ortg_diff_expr,
            r.table_schema,
            r.table_name,
            'Anagrafiche',
            player_table,
            player_id_expr,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id']),
            season_expr,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['season']),
            'Anagrafiche',
            'Team_' || league_key,
            season_expr,
            pg_temp.ai_expr('Anagrafiche', 'Team_' || league_key, 'dt', 'int', ARRAY['season']),
            team_id_expr,
            pg_temp.ai_expr('Anagrafiche', 'Team_' || league_key, 'dt', 'text', ARRAY['id']),
            'Anagrafiche',
            'Team_' || league_key,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['teamname']),
            pg_temp.ai_expr('Anagrafiche', 'Team_' || league_key, 'nt', 'text', ARRAY['teamname', 'name']),
            pg_temp.ai_expr('Anagrafiche', 'Team_' || league_key, 'nt', 'text', ARRAY['shortname', 'short_name']),
            role_join,
            onoff_join,
            clutch_join,
            boxscore_join
        );
    END LOOP;

    IF body = '' THEN
        RAISE NOTICE 'No AdvancedStats_Player_* source tables found; creating an empty canonical PlayerCompetitionStats table.';
        body := $empty$
            SELECT
                NULL::text AS league_key,
                NULL::integer AS season,
                NULL::text AS source_player_id,
                NULL::text AS source_team_id,
                NULL::text AS competition,
                NULL::double precision AS games,
                NULL::double precision AS games_started,
                NULL::double precision AS starter_pct,
                NULL::double precision AS minutes_total,
                NULL::double precision AS points_total,
                NULL::double precision AS rebounds_total,
                NULL::double precision AS offensive_rebounds_total,
                NULL::double precision AS defensive_rebounds_total,
                NULL::double precision AS assists_total,
                NULL::double precision AS steals_total,
                NULL::double precision AS blocks_total,
                NULL::double precision AS turnovers_total,
                NULL::double precision AS fouls_total,
                NULL::double precision AS fg_pct,
                NULL::double precision AS three_point_pct,
                NULL::double precision AS ft_pct,
                NULL::double precision AS plus_minus,
                NULL::double precision AS per,
                NULL::double precision AS ts_pct,
                NULL::double precision AS usg_pct,
                NULL::double precision AS bpm,
                NULL::double precision AS vorp,
                NULL::double precision AS win_shares,
                NULL::double precision AS ast_ratio,
                NULL::double precision AS reb_pct,
                NULL::double precision AS rating,
                NULL::double precision AS ortg,
                NULL::double precision AS drtg,
                NULL::double precision AS net_rtg,
                NULL::text AS ruolo_offensivo,
                NULL::text AS ruolo_difensivo,
                NULL::text AS ruolo_combinato,
                NULL::double precision AS on_net_rtg,
                NULL::double precision AS off_net_rtg,
                NULL::double precision AS net_rtg_diff,
                NULL::double precision AS spm,
                NULL::double precision AS obpm,
                NULL::double precision AS dbpm,
                NULL::double precision AS gm_sc,
                NULL::double precision AS fic,
                NULL::double precision AS ows,
                NULL::double precision AS dws,
                NULL::double precision AS raptor_off,
                NULL::double precision AS raptor_def,
                NULL::double precision AS raptor_total,
                NULL::double precision AS lebron_off,
                NULL::double precision AS lebron_def,
                NULL::double precision AS lebron_total,
                NULL::double precision AS scoring_efficiency,
                NULL::double precision AS ppsa,
                NULL::double precision AS two_point_pct,
                NULL::double precision AS tov_pct,
                NULL::double precision AS ast_pct,
                NULL::double precision AS stl_pct,
                NULL::double precision AS blk_pct,
                NULL::double precision AS orb_pct,
                NULL::double precision AS drb_pct,
                NULL::double precision AS three_par,
                NULL::double precision AS true_usg_pct,
                NULL::double precision AS foul_drawing_rate,
                NULL::double precision AS rf_per_game,
                NULL::double precision AS hustle_index,
                NULL::double precision AS pts_per_40,
                NULL::double precision AS ast_per_40,
                NULL::double precision AS tr_per_40,
                NULL::double precision AS stl_per_40,
                NULL::double precision AS blk_per_40,
                NULL::double precision AS clutch_games,
                NULL::double precision AS clutch_pts,
                NULL::double precision AS clutch_ts_pct,
                NULL::double precision AS clutch_ast_to_tov,
                NULL::double precision AS clutch_net_rtg,
                NULL::double precision AS clutch_efg_pct,
                NULL::double precision AS ortg_on,
                NULL::double precision AS ortg_off,
                NULL::double precision AS ortg_diff
            WHERE false
        $empty$;
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."_PlayerCompetitionRaw" AS ' ||
        'WITH raw AS (' || body || '),
        resolved AS (
            SELECT r.*,
                   row_number() OVER (
                       PARTITION BY r.source_player_id, r.league_key, r.season, r.competition
                       ORDER BY CASE WHEN r.source_team_id IS NULL THEN 0 ELSE 1 END,
                                coalesce(r.games, 0) DESC,
                                r.source_team_id NULLS LAST
                   ) AS rn
            FROM raw r
            WHERE r.rating IS NOT NULL
        ),
        scored AS (
            SELECT r.*,
                   CASE
                       WHEN count(*) OVER (
                           PARTITION BY r.league_key, r.season, r.competition
                       ) = 1
                           THEN 5.5
                       ELSE 1.0 + 9.0 * percent_rank() OVER (
                           PARTITION BY r.league_key, r.season, r.competition
                           ORDER BY r.rating
                       )
                   END AS rating_0_10
            FROM resolved r
            WHERE r.rn = 1
        )
        SELECT * FROM scored';

    -- A single canonical conversion from source-local IDs to stable IDs.
    -- This intermediate table is immediately indexed/renamed and never exposed.
END;
$build$;

-- The temporary raw table is only a build artifact in this transaction.
-- Keep its definition physical so PostgreSQL can optimize the CTAS above.
ALTER TABLE "AI_Source"."_PlayerCompetitionRaw" RENAME TO "PlayerCompetitionStats";

-- Final canonical player-stat shape: stable IDs, model defaults, no source-local
-- identifiers exposed.
CREATE TABLE "AI_Source"."_PlayerCompetitionStatsFinal" AS
SELECT
    CASE
        WHEN player_global_id IS NULL OR btrim(player_global_id) = '' THEN NULL::bigint
        ELSE hashtextextended('player:' || btrim(player_global_id), 0) & 9223372036854775807::bigint
    END AS player_id,
    season,
    CASE
        WHEN coalesce(direct_team_global_id, fallback_team_global_id) IS NULL
          OR btrim(coalesce(direct_team_global_id, fallback_team_global_id)) = ''
            THEN NULL::bigint
        ELSE hashtextextended(
            'team:' || btrim(coalesce(direct_team_global_id, fallback_team_global_id)),
            0
        ) & 9223372036854775807::bigint
    END AS team_id,
    CASE
        WHEN league_key IS NULL OR btrim(league_key) = '' THEN NULL::bigint
        ELSE hashtextextended('league:' || btrim(league_key), 0) & 9223372036854775807::bigint
    END AS league_id,
    coalesce(games, 0)::integer AS games_played,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(minutes_total, 0) / games ELSE 0 END AS minutes_per_game,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(points_total, 0) / games ELSE 0 END AS points,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(rebounds_total, 0) / games ELSE 0 END AS rebounds,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(offensive_rebounds_total, 0) / games ELSE 0 END AS offensive_rebounds,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(defensive_rebounds_total, 0) / games ELSE 0 END AS defensive_rebounds,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(assists_total, 0) / games ELSE 0 END AS assists,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(steals_total, 0) / games ELSE 0 END AS steals,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(blocks_total, 0) / games ELSE 0 END AS blocks,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(turnovers_total, 0) / games ELSE 0 END AS turnovers,
    CASE WHEN coalesce(games, 0) > 0 THEN coalesce(fouls_total, 0) / games ELSE 0 END AS personal_fouls,
    coalesce(fg_pct, 0) AS fg_pct,
    coalesce(three_point_pct, 0) AS three_point_pct,
    coalesce(ft_pct, 0) AS ft_pct,
    coalesce(plus_minus, 0) AS plus_minus,
    per,
    ts_pct,
    usg_pct,
    bpm,
    vorp,
    win_shares,
    ast_ratio,
    reb_pct,
    rating_0_10 AS rating,
    coalesce(ortg, 0) AS ortg,
    coalesce(drtg, 0) AS drtg,
    coalesce(net_rtg, 0) AS net_rtg,
    coalesce(ruolo_offensivo, '') AS ruolo_offensivo,
    coalesce(ruolo_difensivo, '') AS ruolo_difensivo,
    coalesce(ruolo_combinato, '') AS ruolo_combinato,
    coalesce(on_net_rtg, 0) AS on_net_rtg,
    coalesce(off_net_rtg, 0) AS off_net_rtg,
    coalesce(net_rtg_diff, 0) AS net_rtg_diff,
    competition,
    spm,
    obpm,
    dbpm,
    gm_sc,
    fic,
    ows,
    dws,
    raptor_off,
    raptor_def,
    raptor_total,
    lebron_off,
    lebron_def,
    lebron_total,
    scoring_efficiency,
    ppsa,
    two_point_pct,
    tov_pct,
    ast_pct,
    stl_pct,
    blk_pct,
    orb_pct,
    drb_pct,
    three_par,
    true_usg_pct,
    foul_drawing_rate,
    rf_per_game,
    hustle_index,
    pts_per_40,
    ast_per_40,
    tr_per_40,
    stl_per_40,
    blk_per_40,
    coalesce(clutch_games, 0) AS clutch_games,
    coalesce(clutch_pts, 0) AS clutch_pts,
    coalesce(clutch_ts_pct, 0) AS clutch_ts_pct,
    coalesce(clutch_ast_to_tov, 0) AS clutch_ast_to_tov,
    coalesce(clutch_net_rtg, 0) AS clutch_net_rtg,
    coalesce(clutch_efg_pct, 0) AS clutch_efg_pct,
    coalesce(ortg_on, 0) AS ortg_on,
    coalesce(ortg_off, 0) AS ortg_off,
    coalesce(ortg_diff, 0) AS ortg_diff,
    coalesce(games_started, 0)::integer AS games_started,
    coalesce(
        starter_pct,
        CASE WHEN coalesce(games, 0) > 0
             THEN least(1.0, coalesce(games_started, 0) / games)
             ELSE 0
        END
    ) AS starter_pct,
    league_key
FROM "AI_Source"."PlayerCompetitionStats";

DROP TABLE "AI_Source"."PlayerCompetitionStats";
ALTER TABLE "AI_Source"."_PlayerCompetitionStatsFinal" RENAME TO "PlayerCompetitionStats";

-- ---------------------------------------------------------------------------
-- Team competition stats — one canonical physical table.
-- ---------------------------------------------------------------------------

DO $build_team$
DECLARE
    r record;
    body text := '';
    league_key text;
    team_table text;
    season_expr text;
    team_id_expr text;
    comp_expr text;
BEGIN
    FOR r IN
        SELECT
            n.nspname AS table_schema,
            c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n
          ON n.oid = c.relnamespace
        WHERE n.nspname = 'Analisi'
          AND lower(c.relname) LIKE 'advancedstatsteam_%'
          AND c.relkind IN ('r', 'p')
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid
                AND a.attnum > 0
                AND NOT a.attisdropped
                AND lower(a.attname) = 'season'
          )
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid
                AND a.attnum > 0
                AND NOT a.attisdropped
                AND lower(a.attname) IN ('teamid', 'id')
          )
        ORDER BY c.relname
    LOOP
        league_key := regexp_replace(r.table_name, '^AdvancedStatsTeam_', '', 'i');
        team_table := 'Team_' || league_key;
        season_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'int', ARRAY['season']);
        team_id_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'text', ARRAY['teamid', 'id']);
        comp_expr := pg_temp.ai_comp_expr(r.table_schema, r.table_name, 't', ARRAY['competition']);

        body := body ||
            CASE WHEN body = '' THEN '' ELSE E'
UNION ALL
' END ||
            format($q$
                SELECT
                    %1$L::text AS league_key,
                    %2$s::integer AS season,
                    %3$s AS source_team_id,
                    %4$s AS competition,
                    %5$s AS games,
                    %6$s AS pace,
                    %7$s AS ortg,
                    %8$s AS drtg,
                    %9$s AS net_rtg,
                    %10$s AS three_par,
                    %11$s AS fg3a,
                    %12$s AS fg2a,
                    %13$s AS ast_total,
                    %14$s AS ast_per_game,
                    %15$s AS team_global_id,
                    %16$s AS team_name,
                    %17$s AS short_name
                FROM %18$I.%19$I t
                LEFT JOIN %20$I.%21$I tr
                  ON %2$s = %22$s
                 AND %3$s = %23$s
                WHERE %2$s IS NOT NULL
                  AND %3$s IS NOT NULL
            $q$,
                league_key,
                season_expr,
                team_id_expr,
                comp_expr,
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['games']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['pace']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['ortg']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['drtg']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['netrtg', 'net_rating']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['threepar', 'fg3rate']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['fg3a']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['fg2a']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['ast']),
                pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['astpergame']),
                pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['idglobal', 'globalid']),
                pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['teamname', 'name', 'shortname']),
                pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['shortname', 'short_name']),
                r.table_schema,
                r.table_name,
                'Anagrafiche',
                team_table,
                pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'int', ARRAY['season']),
                pg_temp.ai_expr('Anagrafiche', team_table, 'text', ARRAY['id'])
            );
    END LOOP;

    IF body = '' THEN
        body := $empty$
            SELECT
                NULL::text AS league_key,
                NULL::integer AS season,
                NULL::text AS source_team_id,
                NULL::text AS competition,
                NULL::double precision AS games,
                NULL::double precision AS pace,
                NULL::double precision AS ortg,
                NULL::double precision AS drtg,
                NULL::double precision AS net_rtg,
                NULL::double precision AS three_par,
                NULL::double precision AS fg3a,
                NULL::double precision AS fg2a,
                NULL::double precision AS ast_total,
                NULL::double precision AS ast_per_game,
                NULL::text AS team_global_id,
                NULL::text AS team_name,
                NULL::text AS short_name
            WHERE false
        $empty$;
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."TeamCompetitionStats" AS
         WITH raw AS (' || body || '),
         resolved AS (
             SELECT r.*,
                    row_number() OVER (
                        PARTITION BY r.source_team_id, r.league_key, r.season, r.competition
                        ORDER BY coalesce(r.games, 0) DESC, r.source_team_id
                    ) AS rn
             FROM raw r
         )
         SELECT
             CASE WHEN team_global_id IS NULL OR btrim(team_global_id) = '''' THEN NULL::bigint
                  ELSE hashtextextended(''team:'' || btrim(team_global_id), 0) & 9223372036854775807::bigint END AS team_id,
             team_global_id AS global_id,
             coalesce(team_name, '''') AS name,
             CASE WHEN league_key IS NULL OR btrim(league_key) = '''' THEN NULL::bigint
                  ELSE hashtextextended(''league:'' || btrim(league_key), 0) & 9223372036854775807::bigint END AS league_id,
             league_key,
             season,
             competition,
             coalesce(games, 0)::integer AS games_played,
             coalesce(pace, 75.0) AS pace,
             coalesce(ortg, 110.0) AS offensive_rating,
             coalesce(drtg, 110.0) AS defensive_rating,
             coalesce(
                 three_par,
                 CASE WHEN coalesce(fg3a, 0) + coalesce(fg2a, 0) > 0
                      THEN fg3a / (fg3a + fg2a) ELSE NULL END,
                 0.35
             ) AS three_point_attempt_rate,
             coalesce(
                 ast_per_game,
                 CASE WHEN coalesce(games, 0) > 0 THEN ast_total / games ELSE NULL END,
                 20.0
             ) AS assists_per_game,
             coalesce(net_rtg, coalesce(ortg, 110.0) - coalesce(drtg, 110.0)) AS net_rtg,
             coalesce(short_name, '''') AS short_name
         FROM resolved r
         WHERE rn = 1';
END;
$build_team$;

-- Populate star usage after both canonical stat tables exist.
ALTER TABLE "AI_Source"."TeamCompetitionStats"
    ADD COLUMN IF NOT EXISTS star_player_usage double precision;

UPDATE "AI_Source"."TeamCompetitionStats" t
SET star_player_usage = coalesce(s.star_player_usage, 0.25)
FROM (
    SELECT
        team_id,
        league_id,
        season,
        competition,
        least(
            0.60,
            greatest(
                0.10,
                CASE
                    WHEN max(usg_pct) > 1.5 THEN max(usg_pct) / 100.0
                    ELSE max(usg_pct)
                END
            )
        ) AS star_player_usage
    FROM "AI_Source"."PlayerCompetitionStats"
    WHERE team_id IS NOT NULL
      AND usg_pct IS NOT NULL
    GROUP BY team_id, league_id, season, competition
) s
WHERE t.team_id = s.team_id
  AND t.league_id = s.league_id
  AND t.season = s.season
  AND t.competition = s.competition;

-- ---------------------------------------------------------------------------
-- Team-player relations — no PBP dependency.
-- Uses stats team IDs plus Anagrafiche TeamName mapping so future roster rows
-- with no stats are still visible to the model.
-- ---------------------------------------------------------------------------

DO $build_rel$
DECLARE
    r record;
    body text := '';
    league_key text;
    player_table text;
    team_table text;
BEGIN
    FOR r IN
        SELECT
            n.nspname AS table_schema,
            c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'Anagrafiche'
          AND lower(c.relname) <> ALL (ARRAY[]::text[])
          AND c.relkind IN ('r', 'p')
          AND lower(left(c.relname, 5)) <> 'team_'
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
                AND lower(a.attname)='season'
          )
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
                AND lower(a.attname)='id'
          )
        ORDER BY c.relname
    LOOP
        league_key := r.table_name;
        player_table := league_key;
        team_table := 'Team_' || league_key;

        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='Anagrafiche' AND c.relname=team_table
              AND c.relkind IN ('r','p')
        ) THEN
            CONTINUE;
        END IF;

        body := body ||
            CASE WHEN body='' THEN '' ELSE E'
UNION ALL
' END ||
            format($q$
                SELECT
                    CASE WHEN %1$s IS NULL OR btrim(%1$s)='' THEN NULL::bigint
                         ELSE hashtextextended(''player:'' || btrim(%1$s),0) & 9223372036854775807::bigint END AS player_id,
                    CASE WHEN %2$s IS NULL OR btrim(%2$s)='' THEN NULL::bigint
                         ELSE hashtextextended(''team:'' || btrim(%2$s),0) & 9223372036854775807::bigint END AS team_id,
                    %3$s::integer AS season,
                    coalesce(%4$s, '''') AS role,
                    %5$s::integer AS jersey_number
                FROM %6$I.%7$I p
                JOIN %8$I.%9$I t
                  ON %3$s = %10$s
                 AND lower(btrim(%11$s)) = lower(btrim(%12$s))
                WHERE %3$s IS NOT NULL
                  AND %1$s IS NOT NULL
                  AND %2$s IS NOT NULL
            $q$,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['idglobal','globalid']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['idglobal','globalid']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['season']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['pos','position']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['shirtnumber','jerseynumber']),
            'Anagrafiche', player_table,
            'Anagrafiche', team_table,
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'int', ARRAY['season']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['id']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['teamname']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['teamname','name','shortname'])
            );

        body := body ||
            CASE WHEN body='' THEN '' ELSE E'
UNION ALL
' END ||
            format(
                'SELECT player_id, team_id, season, role, jersey_number
                   FROM "AI_Source"."PlayerCompetitionStats"
                  WHERE league_key = %L
                    AND player_id IS NOT NULL
                    AND team_id IS NOT NULL',
                league_key
            );
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::bigint player_id, NULL::bigint team_id, NULL::integer season, NULL::text role, NULL::integer jersey_number WHERE false';
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."TeamPlayerRelations" AS
         SELECT DISTINCT player_id, team_id, season, role, jersey_number
         FROM (' || body || ') q
         WHERE player_id IS NOT NULL AND team_id IS NOT NULL AND season IS NOT NULL';
END;
$build_rel$;

-- ---------------------------------------------------------------------------
-- Players — one row per global player, latest known registry state.
-- ---------------------------------------------------------------------------

DO $build_players$
DECLARE
    r record;
    body text := '';
    league_key text;
    player_table text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='Anagrafiche'
          AND lower(left(c.relname,5)) <> 'team_'
          AND c.relkind IN ('r','p')
        ORDER BY c.relname
    LOOP
        league_key := r.table_name;
        player_table := league_key;
        IF pg_temp.ai_find_column('Anagrafiche', player_table, ARRAY['season']) IS NULL
           OR pg_temp.ai_find_column('Anagrafiche', player_table, ARRAY['id']) IS NULL
        THEN
            CONTINUE;
        END IF;

        body := body ||
            CASE WHEN body='' THEN '' ELSE E'
UNION ALL
' END ||
            format($q$
                SELECT
                    CASE WHEN %1$s IS NULL OR btrim(%1$s)='' THEN NULL::bigint
                         ELSE hashtextextended(''player:'' || btrim(%1$s),0) & 9223372036854775807::bigint END AS id,
                    %1$s AS global_id,
                    coalesce(%2$s, %3$s) AS name,
                    coalesce(%4$s,
                             CASE WHEN %5$s IS NOT NULL THEN %6$s - extract(year FROM %5$s)::integer ELSE NULL END) AS age,
                    coalesce(nullif(%7$s,''''), ''PG'') AS position,
                    coalesce(%8$s, '''') AS nationality,
                    %9$s AS height_cm,
                    %10$s AS weight_kg,
                    ''R''::text AS dominant_hand,
                    ''$'' AS current_team_placeholder,
                    0::bigint AS current_league_id,
                    NULL::integer AS draft_year,
                    NULL::integer AS draft_pick,
                    %5$s AS birth_date,
                    %6$s::integer AS registry_season,
                    %11$L::text AS league_key
                FROM %12$I.%13$I p
                WHERE %6$s IS NOT NULL AND %14$s IS NOT NULL
            $q$,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['idglobal','globalid']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['normalizedplayername','playername','name']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['age']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'date', ARRAY['birthdate','dateofbirth','dob']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['season']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['pos','position']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['nat','nationality']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'numeric', ARRAY['cm','heightcm','height_cm']),
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'numeric', ARRAY['weight','weightkg','weight_kg']),
            league_key,
            'Anagrafiche', player_table,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id'])
        );
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::bigint id, NULL::text global_id, NULL::text name, NULL::integer age, NULL::text position, NULL::text nationality, NULL::double precision height_cm, NULL::double precision weight_kg, NULL::text dominant_hand, NULL::text current_team_placeholder, NULL::bigint current_league_id, NULL::integer draft_year, NULL::integer draft_pick, NULL::date birth_date, NULL::integer registry_season, NULL::text league_key WHERE false';
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."_PlayersRegistry" AS
         SELECT * FROM (' || body || ') q';

    EXECUTE
        'CREATE TABLE "AI_Source"."Players" AS
         WITH ranked AS (
             SELECT r.*,
                    row_number() OVER (
                        PARTITION BY r.global_id
                        ORDER BY r.registry_season DESC, r.league_key
                    ) AS rn
             FROM "AI_Source"."_PlayersRegistry" r
             WHERE r.id IS NOT NULL
         ),
         current_relation AS (
             SELECT DISTINCT ON (player_id)
                    player_id, team_id, season
             FROM "AI_Source"."TeamPlayerRelations"
             ORDER BY player_id, season DESC, team_id DESC
         )
         SELECT
             p.id,
             p.global_id,
             p.name,
             coalesce(p.age, 0)::integer AS age,
             coalesce(nullif(p.position,''), ''PG'') AS position,
             coalesce(p.nationality, '''') AS nationality,
             p.height_cm,
             p.weight_kg,
             p.dominant_hand,
             cr.team_id AS current_team_id,
             CASE
                 WHEN p.league_key IS NULL OR btrim(p.league_key)='' THEN NULL::bigint
                 ELSE hashtextextended(''league:'' || btrim(p.league_key),0) & 9223372036854775807::bigint
             END AS current_league_id,
             p.draft_year,
             p.draft_pick,
             p.birth_date,
             p.league_key AS current_league_key
         FROM ranked p
         LEFT JOIN current_relation cr ON cr.player_id=p.id
         WHERE p.rn=1';

    DROP TABLE "AI_Source"."_PlayersRegistry";
END;
$build_players$;

-- ---------------------------------------------------------------------------
-- Teams — one row per global team, latest registry state.
-- ---------------------------------------------------------------------------

DO $build_teams$
DECLARE
    r record;
    body text := '';
    league_key text;
    team_table text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='Anagrafiche'
          AND lower(left(c.relname,5))='team_'
          AND c.relkind IN ('r','p')
        ORDER BY c.relname
    LOOP
        league_key := regexp_replace(r.table_name, '^Team_', '', 'i');
        team_table := r.table_name;
        IF pg_temp.ai_find_column('Anagrafiche', team_table, ARRAY['season']) IS NULL
           OR pg_temp.ai_find_column('Anagrafiche', team_table, ARRAY['id']) IS NULL
        THEN
            CONTINUE;
        END IF;

        body := body ||
            CASE WHEN body='' THEN '' ELSE E'
UNION ALL
' END ||
            format($q$
                SELECT
                    CASE WHEN %1$s IS NULL OR btrim(%1$s)='' THEN NULL::bigint
                         ELSE hashtextextended(''team:'' || btrim(%1$s),0) & 9223372036854775807::bigint END AS id,
                    %1$s AS global_id,
                    coalesce(%2$s, %3$s) AS name,
                    CASE WHEN %4$L IS NULL OR %5$s='' THEN NULL::bigint
                         ELSE hashtextextended(''league:'' || %4$L,0) & 9223372036854775807::bigint END AS league_id,
                    ''''::text AS playing_style,
                    ''''::text AS formation,
                    NULL::double precision AS pace,
                    NULL::double precision AS offensive_rating,
                    NULL::double precision AS defensive_rating,
                    0.35::double precision AS three_point_attempt_rate,
                    20.0::double precision AS assists_per_game,
                    0.25::double precision AS star_player_usage,
                    1::integer AS league_tier,
                    coalesce(%6$s,'''') AS short_name,
                    NULL::double precision AS net_rtg,
                    %4$L::text AS league_key,
                    %7$s::integer AS latest_season
                FROM %8$I.%9$I t
                WHERE %7$s IS NOT NULL AND %10$s IS NOT NULL
            $q$,
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['idglobal','globalid']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['teamname','name','shortname']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['id']),
            league_key,
            league_key,
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['shortname','short_name']),
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'int', ARRAY['season']),
            'Anagrafiche', team_table,
            pg_temp.ai_expr('Anagrafiche', team_table, 't', 'text', ARRAY['id'])
        );
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::bigint id, NULL::text global_id, NULL::text name, NULL::bigint league_id, NULL::text playing_style, NULL::text formation, NULL::double precision pace, NULL::double precision offensive_rating, NULL::double precision defensive_rating, NULL::double precision three_point_attempt_rate, NULL::double precision assists_per_game, NULL::double precision star_player_usage, NULL::integer league_tier, NULL::text short_name, NULL::double precision net_rtg, NULL::text league_key, NULL::integer latest_season WHERE false';
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."_TeamsRegistry" AS
         SELECT * FROM (' || body || ') q';

    EXECUTE
        'CREATE TABLE "AI_Source"."_TeamsCurrent" AS
         WITH ranked AS (
             SELECT r.*,
                    row_number() OVER (
                        PARTITION BY r.global_id
                        ORDER BY r.latest_season DESC, r.league_key
                    ) AS rn
             FROM "AI_Source"."_TeamsRegistry" r
             WHERE r.id IS NOT NULL
         ),
         preferred_stats AS (
             SELECT DISTINCT ON (team_id, league_id, season)
                    team_id, league_id, season,
                    pace, offensive_rating, defensive_rating,
                    three_point_attempt_rate, assists_per_game, net_rtg,
                    short_name
             FROM "AI_Source"."TeamCompetitionStats"
             ORDER BY team_id, league_id, season,
                      CASE upper(competition)
                          WHEN ''TOT'' THEN 0
                          WHEN ''RS'' THEN 1
                          ELSE 2
                      END,
                      competition
         )
         SELECT
             r.id,
             r.global_id,
             r.name,
             r.league_id,
             r.playing_style,
             r.formation,
             coalesce(s.pace, r.pace, 75.0) AS pace,
             coalesce(s.offensive_rating, r.offensive_rating, 110.0) AS offensive_rating,
             coalesce(s.defensive_rating, r.defensive_rating, 110.0) AS defensive_rating,
             coalesce(s.three_point_attempt_rate, r.three_point_attempt_rate, 0.35) AS three_point_attempt_rate,
             coalesce(s.assists_per_game, r.assists_per_game, 20.0) AS assists_per_game,
             coalesce(s.star_player_usage, r.star_player_usage, 0.25) AS star_player_usage,
             r.league_tier,
             coalesce(s.short_name, r.short_name, '''') AS short_name,
             coalesce(s.net_rtg, r.net_rtg, coalesce(s.offensive_rating,110.0)-coalesce(s.defensive_rating,110.0)) AS net_rtg,
             r.league_key,
             r.latest_season
         FROM ranked r
         LEFT JOIN preferred_stats s
           ON s.team_id=r.id
          AND s.league_id=r.league_id
          AND s.season=r.latest_season
         WHERE r.rn=1';

    DROP TABLE "AI_Source"."_TeamsRegistry";
    ALTER TABLE "AI_Source"."_TeamsCurrent" RENAME TO "Teams";
END;
$build_teams$;

-- Leagues — only keys represented by the canonical player/team datasets.
CREATE TABLE "AI_Source"."Leagues" AS
WITH keys AS (
    SELECT DISTINCT league_key FROM "AI_Source"."Players"
    UNION
    SELECT DISTINCT league_key FROM "AI_Source"."Teams"
    UNION
    SELECT DISTINCT league_key FROM "AI_Source"."PlayerCompetitionStats"
    UNION
    SELECT DISTINCT league_key FROM "AI_Source"."TeamCompetitionStats"
),
observed AS (
    SELECT league_id, max(games_played)::integer AS max_games
    FROM "AI_Source"."PlayerCompetitionStats"
    GROUP BY league_id
),
team_average AS (
    SELECT league_id,
           avg(pace) AS avg_pace,
           avg(offensive_rating) AS avg_offensive_rating
    FROM "AI_Source"."Teams"
    GROUP BY league_id
)
SELECT
    hashtextextended('league:' || btrim(k.league_key),0) & 9223372036854775807::bigint AS id,
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
    coalesce(ta.avg_pace,75.0) AS avg_pace,
    coalesce(ta.avg_offensive_rating,110.0) AS avg_offensive_rating,
    greatest(coalesce(o.max_games,1),1)::integer AS max_games,
    k.league_key
FROM keys k
LEFT JOIN observed o ON o.league_id=hashtextextended('league:' || btrim(k.league_key),0) & 9223372036854775807::bigint
LEFT JOIN team_average ta ON ta.league_id=hashtextextended('league:' || btrim(k.league_key),0) & 9223372036854775807::bigint
WHERE k.league_key IS NOT NULL AND btrim(k.league_key) <> '';

-- ---------------------------------------------------------------------------
-- Final canonical indexes. Only indexes useful to the model/API contract are
-- created; there are no indexes on temporary/build-only artifacts.
-- ---------------------------------------------------------------------------

CREATE UNIQUE INDEX "ux_PlayerCompetitionStats_key"
    ON "AI_Source"."PlayerCompetitionStats"
       (player_id, league_id, season, competition);

CREATE INDEX "ix_PlayerCompetitionStats_player_time"
    ON "AI_Source"."PlayerCompetitionStats"
       (player_id, season, competition);

CREATE INDEX "ix_PlayerCompetitionStats_league_time"
    ON "AI_Source"."PlayerCompetitionStats"
       (league_id, season, competition);

CREATE UNIQUE INDEX "ux_TeamCompetitionStats_key"
    ON "AI_Source"."TeamCompetitionStats"
       (team_id, league_id, season, competition);

CREATE INDEX "ix_TeamCompetitionStats_league_time"
    ON "AI_Source"."TeamCompetitionStats"
       (league_id, season, competition);

CREATE UNIQUE INDEX "ux_TeamPlayerRelations_key"
    ON "AI_Source"."TeamPlayerRelations"
       (player_id, team_id, season);

CREATE INDEX "ix_TeamPlayerRelations_player_time"
    ON "AI_Source"."TeamPlayerRelations"
       (player_id, season);

CREATE UNIQUE INDEX "ux_Players_id"
    ON "AI_Source"."Players"(id);

CREATE UNIQUE INDEX "ux_Teams_id"
    ON "AI_Source"."Teams"(id);

CREATE UNIQUE INDEX "ux_Leagues_id"
    ON "AI_Source"."Leagues"(id);

ANALYZE "AI_Source"."PlayerCompetitionStats";
ANALYZE "AI_Source"."TeamCompetitionStats";
ANALYZE "AI_Source"."TeamPlayerRelations";
ANALYZE "AI_Source"."Players";
ANALYZE "AI_Source"."Teams";
ANALYZE "AI_Source"."Leagues";

COMMIT;

-- Verification:
-- SELECT count(*) FROM "AI_Source"."PlayerCompetitionStats";
-- SELECT competition, count(*) FROM "AI_Source"."PlayerCompetitionStats" GROUP BY competition ORDER BY competition;
-- SELECT count(*) FROM "AI_Source"."TeamCompetitionStats";
-- SELECT name, max_games FROM "AI_Source"."Leagues" ORDER BY name;

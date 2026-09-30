-- ============================================================================
-- AI_Source — minimal PostgreSQL-native canonical cache
--
-- Persistent objects owned by this file:
--   Leagues
--   Teams
--   Players
--   TeamPlayerRelations
--   PlayerCompetitionStats
--   TeamCompetitionStats
--
-- Design:
--   * no RawInternal views
--   * no simulation/PBP objects
--   * no persistent helper functions in AI_Source
--   * no to_jsonb/jsonb_each parsing
--   * no row-by-row PL/pgSQL data transformation
--   * dynamic discovery uses pg_catalog only
--   * heavy datasets are built with CREATE TABLE AS, allowing PostgreSQL to
--     choose a parallel plan for the underlying SELECT when appropriate
--   * only columns consumed by AI_Model are selected
--
-- Re-run this file after adding/changing source leagues. The resulting
-- canonical tables are self-contained and are read directly by postgres_loader.
-- ============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS "AI_Source";

-- ---------------------------------------------------------------------------
-- Remove only the canonical objects owned by this bootstrap plus legacy
-- RawInternal/simulation objects from previous versions. Serving tables
-- (Training*, Scenario*, ServingRefreshState, etc.) are intentionally kept.
-- ---------------------------------------------------------------------------

DO $cleanup$
DECLARE
    object_name text;
    relkind "char";
BEGIN
    FOREACH object_name IN ARRAY ARRAY[
        'PlayerCompetitionStats',
        'TeamCompetitionStats',
        'TeamPlayerRelations',
        'Players',
        'Teams',
        'Leagues',
        'PlayerStats'
    ]
    LOOP
        SELECT c.relkind
          INTO relkind
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='AI_Source'
          AND c.relname=object_name
        LIMIT 1;

        IF relkind = 'm' THEN
            EXECUTE format(
                'DROP MATERIALIZED VIEW "AI_Source".%I CASCADE',
                object_name
            );
        ELSIF relkind = 'v' THEN
            EXECUTE format(
                'DROP VIEW "AI_Source".%I CASCADE',
                object_name
            );
        ELSIF relkind IN ('r','p') THEN
            EXECUTE format(
                'DROP TABLE "AI_Source".%I CASCADE',
                object_name
            );
        END IF;
    END LOOP;
END;
$cleanup$;

DO $cleanup_legacy$
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
        ELSIF r.relkind = 'v' THEN
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
$cleanup_legacy$;

DROP FUNCTION IF EXISTS "AI_Source"."DropRawObject"(text);
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
-- These are pg_temp functions: they exist only for this connection and are
-- executed during SQL generation, never once per data row.
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
    JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
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
    JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
    WHERE n.nspname = p_schema
      AND c.relname = p_table
      AND a.attnum > 0
      AND NOT a.attisdropped
      AND lower(a.attname) = ANY (p_candidates)
    ORDER BY array_position(p_candidates, lower(a.attname)), a.attnum
    LIMIT 1;

    IF att IS NULL THEN
        CASE p_kind
            WHEN 'text'    THEN RETURN 'NULL::text';
            WHEN 'numeric' THEN RETURN 'NULL::double precision';
            WHEN 'int'     THEN RETURN 'NULL::bigint';
            WHEN 'date'    THEN RETURN 'NULL::date';
            WHEN 'bool'    THEN RETURN 'NULL::boolean';
            ELSE RAISE EXCEPTION 'Unsupported ai_expr kind: %', p_kind;
        END CASE;
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
                     ''^[+-]?([0-9]+([.,][0-9]*)?|[.,][0-9]+)([eE][+-]?[0-9]+)%%?$''
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
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[+-]?[0-9]+([.][0-9]+)?$''
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
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{1,2}-[0-9]{1,2}-[0-9]{4}$''
                    THEN to_date(btrim(%1$s::text), ''DD-MM-YYYY'')
                WHEN NULLIF(btrim(%1$s::text), '''') ~ ''^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$''
                    THEN to_date(btrim(%1$s::text), ''DD/MM/YYYY'')
                ELSE NULL::date
              END)',
            q
        );
    ELSIF p_kind = 'bool' THEN
        IF typcategory = 'B' THEN
            RETURN format('%s::boolean', q);
        ELSIF typcategory = 'N' THEN
            RETURN format(
                '(CASE WHEN %1$s IS NULL THEN NULL
                       ELSE %1$s::double precision <> 0 END)',
                q
            );
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

CREATE OR REPLACE FUNCTION pg_temp.ai_id_key(
    value_expr text
)
RETURNS text
LANGUAGE plpgsql
AS $fn$
BEGIN
    RETURN format(
        '(CASE
            WHEN NULLIF(btrim(%1$s::text), '''') IS NULL THEN NULL::text
            WHEN btrim(%1$s::text) ~ ''^[+-]?[0-9]+[.]0+$''
                THEN regexp_replace(btrim(%1$s::text), ''[.]0+$'', '''')
            ELSE lower(btrim(%1$s::text))
          END)',
        value_expr
    );
END;
$fn$;

CREATE OR REPLACE FUNCTION pg_temp.ai_stable_id(
    namespace text,
    value_expr text
)
RETURNS text
LANGUAGE plpgsql
AS $fn$
BEGIN
    RETURN format(
        '(CASE
            WHEN %1$s IS NULL OR btrim(%1$s::text) = '''' THEN NULL::bigint
            ELSE hashtextextended(%2$L || '':'' || btrim(%1$s::text), 0)
                 & 9223372036854775807::bigint
          END)',
        value_expr,
        namespace
    );
END;
$fn$;

-- Stable IDs are deliberately null-preserving. The format template below
-- expands hashtextextended(%2$L || ':' || btrim(%1$s::text), 0).
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

-- Competition normalization contract: when 'total' then 'tot', when 'all' then 'tot',
-- when 'tot' then 'tot', when 'playoffs' then 'po', when 'regular season' then 'rs'.
-- ---------------------------------------------------------------------------
-- Canonical PlayerCompetitionStats
-- ---------------------------------------------------------------------------

DO $player_stats$
DECLARE
    r record;
    body text := '';
    league_key text;
    player_table text;
    team_table text;
    season_expr text;
    player_local_expr text;
    team_local_expr text;
    comp_expr text;
    rating_expr text;
    player_global_expr text;
    p_team_local_expr text;
    p_team_name_expr text;
    team_source_expr text;
    direct_team_global_expr text := 'NULL::text';
    fallback_team_global_expr text := 'NULL::text';

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

    role_join text := '';
    onoff_join text := '';
    clutch_join text := '';
    boxscore_join text := '';

    opt_table text;
    box_started_expr text;
    games_expr text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'Analisi'
          AND lower(c.relname) LIKE 'advancedstats_player_%'
          AND c.relkind IN ('r', 'p')
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid AND a.attnum > 0
                AND NOT a.attisdropped AND lower(a.attname) = 'season'
          )
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid AND a.attnum > 0
                AND NOT a.attisdropped
                AND lower(a.attname) IN ('id', 'playerid', 'idplayer')
          )
        ORDER BY c.relname
    LOOP
        -- Reset all per-league state. A previous league must never leak an
        -- optional join/expression into the next branch.
        role_join := '';
        onoff_join := '';
        clutch_join := '';
        boxscore_join := '';
        direct_team_global_expr := 'NULL::text';
        fallback_team_global_expr := 'NULL::text';

        role_off_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolooffensivo']);
        role_def_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolodifensivo']);
        role_combo_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['ruolocombinato']);

        on_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_on', 'onnetrtg']);
        off_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_off', 'offnetrtg']);
        net_diff_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg_diff']);
        ortg_on_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_on']);
        ortg_off_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_off']);
        ortg_diff_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg_diff']);

        clutch_games_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchgames']);
        clutch_pts_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchpts']);
        clutch_ts_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchtspct']);
        clutch_ast_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['asttotovratio', 'clutchasttotov']);
        clutch_net_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchnetrtg']);
        clutch_efg_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['clutchefgpct']);

        games_started_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['gamesstarted', 'games_started']);
        starter_pct_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['starterpct', 'starter_pct']);

        league_key := regexp_replace(r.table_name, '^AdvancedStats_Player_', '', 'i');
        player_table := league_key;
        team_table := 'Team_' || league_key;

        IF pg_temp.ai_find_column('Anagrafiche', player_table, ARRAY['season']) IS NULL
           OR pg_temp.ai_find_column('Anagrafiche', player_table, ARRAY['id']) IS NULL
        THEN
            CONTINUE;
        END IF;

        season_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'int', ARRAY['season']);
        player_local_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['id', 'playerid', 'idplayer']);
        team_local_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'text', ARRAY['teamid', 'idteam']);
        p_team_local_expr := pg_temp.ai_expr(
            'Anagrafiche', player_table, 'p', 'text', ARRAY['team', 'teamid', 'idteam']
        );
        team_source_expr := format('coalesce(%s, %s)', team_local_expr, p_team_local_expr);
        comp_expr := pg_temp.ai_comp_expr(r.table_schema, r.table_name, 's', ARRAY['competition']);
        rating_expr := pg_temp.ai_expr(
            r.table_schema, r.table_name, 's', 'numeric', ARRAY['vallegapergame', 'rating']
        );

        player_global_expr := format(
            'coalesce(%1$s, %2$L || '':'' || %3$s)',
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['idglobal', 'globalid']),
            league_key,
            pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id'])
        );
        p_team_name_expr := pg_temp.ai_expr(
            'Anagrafiche', player_table, 'p', 'text', ARRAY['teamname', 'team', 'name']
        );

        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'Anagrafiche'
              AND c.relname = team_table
              AND c.relkind IN ('r', 'p')
        ) THEN
            direct_team_global_expr := format(
                'coalesce(%1$s, %2$L || '':'' || %3$s)',
                pg_temp.ai_expr('Anagrafiche', team_table, 'dt', 'text', ARRAY['idglobal', 'globalid']),
                league_key,
                pg_temp.ai_expr('Anagrafiche', team_table, 'dt', 'text', ARRAY['id'])
            );
            fallback_team_global_expr := format(
                'coalesce(%1$s, %2$L || '':'' || %3$s)',
                pg_temp.ai_expr('Anagrafiche', team_table, 'nt', 'text', ARRAY['idglobal', 'globalid']),
                league_key,
                pg_temp.ai_expr('Anagrafiche', team_table, 'nt', 'text', ARRAY['id'])
            );
        ELSE
            CONTINUE;
        END IF;

        -- Only touch optional role source when at least one role field is not
        -- already present in AdvancedStats_Player_<LEAGUE>.
        IF (
            pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolooffensivo']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolodifensivo']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['ruolocombinato']) IS NULL
        ) THEN
            opt_table := 'PlayerRoles_' || league_key;
            IF EXISTS (
                SELECT 1 FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname='Analisi' AND c.relname=opt_table
                  AND c.relkind IN ('r','p')
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
                            FROM "Analisi".%7$I rr0
                            WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                        ) q
                        ORDER BY season, source_player_id, competition
                    ) rr
                      ON rr.season = %8$s
                     AND rr.source_player_id = %9$s
                     AND rr.competition = %10$s',
                    pg_temp.ai_expr('Analisi', opt_table, 'rr0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Analisi', opt_table, 'rr0', 'text', ARRAY['id','playerid','idplayer']),
                    pg_temp.ai_comp_expr('Analisi', opt_table, 'rr0', ARRAY['competition']),
                    pg_temp.ai_expr('Analisi', opt_table, 'rr0', 'text', ARRAY['ruolooffensivo']),
                    pg_temp.ai_expr('Analisi', opt_table, 'rr0', 'text', ARRAY['ruolodifensivo']),
                    pg_temp.ai_expr('Analisi', opt_table, 'rr0', 'text', ARRAY['ruolocombinato']),
                    opt_table,
                    season_expr, player_local_expr, comp_expr
                );
                role_off_expr := 'coalesce(' || role_off_expr || ', rr.ruolo_offensivo)';
                role_def_expr := 'coalesce(' || role_def_expr || ', rr.ruolo_difensivo)';
                role_combo_expr := 'coalesce(' || role_combo_expr || ', rr.ruolo_combinato)';
            END IF;
        END IF;

        -- On/Off data is additive and may coexist with the same columns in
        -- AdvancedStats_Player_<LEAGUE>. The player table can contain zero-filled
        -- placeholders, so column existence alone is not a valid reason to skip
        -- AdvancedStatsOnOffCourt_<LEAGUE>. When the authoritative On/Off source
        -- has a real observation, it replaces only a NULL/zero primary value.
        opt_table := 'AdvancedStatsOnOffCourt_' || league_key;
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='Analisi' AND c.relname=opt_table
              AND c.relkind IN ('r','p')
        ) THEN
            onoff_join := format(
                'LEFT JOIN (
                    SELECT DISTINCT ON (
                        season,
                        source_player_id,
                        competition,
                        source_team_id
                    )
                           season,
                           source_player_id,
                           source_team_id,
                           competition,
                           on_net_rtg, off_net_rtg, net_rtg_diff,
                           ortg_on, ortg_off, ortg_diff
                    FROM (
                        SELECT
                            %1$s AS season,
                            %2$s AS source_player_id,
                            %3$s AS source_team_id,
                            %4$s AS competition,
                            %5$s AS on_net_rtg,
                            %6$s AS off_net_rtg,
                            %7$s AS net_rtg_diff,
                            %8$s AS ortg_on,
                            %9$s AS ortg_off,
                            %10$s AS ortg_diff
                        FROM "Analisi".%11$I oo0
                        WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                    ) q
                    ORDER BY
                        season,
                        source_player_id,
                        competition,
                        source_team_id
                ) oo
                  ON oo.season = %12$s
                 AND %13$s = %14$s
                 AND oo.competition = %15$s
                 AND (
                      (
                          oo.source_team_id IS NOT NULL
                          AND pg_temp.ai_id_key(oo.source_team_id)
                              =
                              pg_temp.ai_id_key(%16$s)
                      )
                      OR
                      (
                          oo.source_team_id IS NULL
                          AND pg_temp.ai_id_key(%16$s) IS NULL
                      )
                 )',
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'int', ARRAY['season']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'text', ARRAY['id','playerid','idplayer']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'text', ARRAY['teamid','idteam']),
                pg_temp.ai_comp_expr('Analisi', opt_table, 'oo0', ARRAY['competition']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['netrtg_on']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['netrtg_off']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['netrtg_diff']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['ortg_on']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['ortg_off']),
                pg_temp.ai_expr('Analisi', opt_table, 'oo0', 'numeric', ARRAY['ortg_diff']),
                opt_table,
                season_expr,
                pg_temp.ai_id_key('oo.source_player_id'),
                pg_temp.ai_id_key(player_local_expr),
                comp_expr,
                team_local_expr
            );
            on_net_expr := '(CASE WHEN ' || on_net_expr || ' IS NULL OR ' || on_net_expr || ' = 0
                                  THEN coalesce(oo.on_net_rtg, ' || on_net_expr || ')
                                  ELSE ' || on_net_expr || ' END)';
            off_net_expr := '(CASE WHEN ' || off_net_expr || ' IS NULL OR ' || off_net_expr || ' = 0
                                   THEN coalesce(oo.off_net_rtg, ' || off_net_expr || ')
                                   ELSE ' || off_net_expr || ' END)';
            net_diff_expr := '(CASE WHEN ' || net_diff_expr || ' IS NULL OR ' || net_diff_expr || ' = 0
                                    THEN coalesce(oo.net_rtg_diff, ' || net_diff_expr || ')
                                    ELSE ' || net_diff_expr || ' END)';
            ortg_on_expr := '(CASE WHEN ' || ortg_on_expr || ' IS NULL OR ' || ortg_on_expr || ' = 0
                                   THEN coalesce(oo.ortg_on, ' || ortg_on_expr || ')
                                   ELSE ' || ortg_on_expr || ' END)';
            ortg_off_expr := '(CASE WHEN ' || ortg_off_expr || ' IS NULL OR ' || ortg_off_expr || ' = 0
                                    THEN coalesce(oo.ortg_off, ' || ortg_off_expr || ')
                                    ELSE ' || ortg_off_expr || ' END)';
            ortg_diff_expr := '(CASE WHEN ' || ortg_diff_expr || ' IS NULL OR ' || ortg_diff_expr || ' = 0
                                     THEN coalesce(oo.ortg_diff, ' || ortg_diff_expr || ')
                                     ELSE ' || ortg_diff_expr || ' END)';
        END IF;

        IF (
            pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchgames']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchpts']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchtspct']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['asttotovratio','clutchasttotov']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchnetrtg']) IS NULL
            OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['clutchefgpct']) IS NULL
        ) THEN
            opt_table := 'AdvancedStats_Clutch_' || league_key;
            IF EXISTS (
                SELECT 1 FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname='Analisi' AND c.relname=opt_table
                  AND c.relkind IN ('r','p')
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
                            FROM "Analisi".%10$I cl0
                            WHERE %1$s IS NOT NULL AND %2$s IS NOT NULL
                        ) q
                        ORDER BY season, source_player_id, competition
                    ) cl
                      ON cl.season = %11$s
                     AND cl.source_player_id = %12$s
                     AND cl.competition = %13$s',
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'text', ARRAY['id','playerid','idplayer']),
                    pg_temp.ai_comp_expr('Analisi', opt_table, 'cl0', ARRAY['competition']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['clutchgames','games']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['clutchpts','pts']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['clutchtspct','tspct']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['asttotovratio','clutchasttotov']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['clutchnetrtg','netrtg']),
                    pg_temp.ai_expr('Analisi', opt_table, 'cl0', 'numeric', ARRAY['clutchefgpct','efgpct']),
                    opt_table,
                    season_expr, player_local_expr, comp_expr
                );
                clutch_games_expr := 'coalesce(' || clutch_games_expr || ', cl.clutch_games)';
                clutch_pts_expr := 'coalesce(' || clutch_pts_expr || ', cl.clutch_pts)';
                clutch_ts_expr := 'coalesce(' || clutch_ts_expr || ', cl.clutch_ts_pct)';
                clutch_ast_expr := 'coalesce(' || clutch_ast_expr || ', cl.clutch_ast_to_tov)';
                clutch_net_expr := 'coalesce(' || clutch_net_expr || ', cl.clutch_net_rtg)';
                clutch_efg_expr := 'coalesce(' || clutch_efg_expr || ', cl.clutch_efg_pct)';
            END IF;
        END IF;

        -- Boxscore is used only when AdvancedStats does not already contain
        -- BOTH starter fields. This avoids scanning Boxscore unnecessarily.
        IF pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['gamesstarted','games_started']) IS NULL
           OR pg_temp.ai_find_column(r.table_schema, r.table_name, ARRAY['starterpct','starter_pct']) IS NULL
        THEN
            opt_table := league_key;
            IF EXISTS (
                SELECT 1 FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname='Boxscore' AND c.relname=opt_table
                  AND c.relkind IN ('r','p')
            ) THEN
                -- Pre-aggregate Boxscore once per league. The previous
                -- correlated count(DISTINCT game) scanned Boxscore repeatedly
                -- for every AdvancedStats row.
                boxscore_join := format(
                    'LEFT JOIN (
                        SELECT bx.season,
                               bx.source_player_id,
                               bx.competition,
                               count(DISTINCT bx.game_id)::double precision AS games_started
                        FROM (
                            SELECT %1$s AS season,
                                   %2$s AS source_player_id,
                                   %3$s AS game_id,
                                   %4$s AS competition
                            FROM "Boxscore".%5$I b0
                            WHERE %1$s IS NOT NULL
                              AND %2$s IS NOT NULL
                              AND %3$s IS NOT NULL
                              AND %6$s
                        ) bx
                        GROUP BY bx.season, bx.source_player_id, bx.competition
                    ) bs
                      ON bs.season = %7$s
                     AND bs.source_player_id = %8$s
                     AND bs.competition = %9$s',
                    pg_temp.ai_expr('Boxscore', opt_table, 'b0', 'int', ARRAY['season']),
                    pg_temp.ai_expr('Boxscore', opt_table, 'b0', 'text', ARRAY['id','idplayer','playerid']),
                    pg_temp.ai_expr('Boxscore', opt_table, 'b0', 'text', ARRAY['game','idgame','gamecode']),
                    pg_temp.ai_comp_expr('Boxscore', opt_table, 'b0', ARRAY['competition']),
                    opt_table,
                    pg_temp.ai_expr('Boxscore', opt_table, 'b0', 'bool', ARRAY['sf','starter','isstarter']),
                    season_expr,
                    player_local_expr,
                    comp_expr
                );
                games_started_expr := 'bs.games_started';
                games_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['games','gamesplayed']);
                starter_pct_expr := format(
                    '(CASE WHEN coalesce(%1$s, 0) > 0
                            THEN least(1.0, coalesce(%2$s, 0) / %1$s)
                            ELSE 0 END)',
                    games_expr,
                    games_started_expr
                );
            END IF;
        END IF;

        body := body
            || CASE WHEN body='' THEN '' ELSE chr(10) || 'UNION ALL' || chr(10) END
            || 'SELECT '
            || quote_literal(league_key) || '::text AS league_key, '
            || season_expr || '::integer AS season, '
            || player_local_expr || ' AS source_player_id, '
            || team_local_expr || ' AS source_team_id, '
            || player_global_expr || ' AS player_global_id, '
            || direct_team_global_expr || ' AS direct_team_global_id, '
            || fallback_team_global_expr || ' AS fallback_team_global_id, '
            || comp_expr || ' AS competition, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['games','gamesplayed']) || ' AS games, '
            || games_started_expr || ' AS games_started, '
            || starter_pct_expr || ' AS starter_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['min','minutes']) || ' AS minutes_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pts','points']) || ' AS points_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tr','reb','rebounds']) || ' AS rebounds_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['or','orb','offensiverebounds']) || ' AS offensive_rebounds_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dr','drb','defensiverebounds']) || ' AS defensive_rebounds_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ast','assists']) || ' AS assists_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stl','steals']) || ' AS steals_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blk','blocks']) || ' AS blocks_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['to','tov','turnovers']) || ' AS turnovers_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pf','fouls','personalfouls']) || ' AS fouls_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['efgpct','fgpct','fg_pct']) || ' AS fg_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fg3pct','threepointpct','three_point_pct']) || ' AS three_point_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ftpct','ft_pct']) || ' AS ft_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['plusminus','plus_minus']) || ' AS plus_minus, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['pie','per']) || ' AS per, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tspct','ts_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tspct','ts_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tspct','ts_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tspct','ts_pct']) || ' END AS ts_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['usgpct','usg_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['usgpct','usg_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['usgpct','usg_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['usgpct','usg_pct']) || ' END AS usg_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['bpm']) || ' AS bpm, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['vorp']) || ' AS vorp, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ws','winshares']) || ' AS win_shares, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astratio','ast_ratio']) || ' AS ast_ratio, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['rebpct','reb_pct']) || ' AS reb_pct, '
            || rating_expr || ' AS rating_source, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ortg']) || ' AS ortg, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drtg']) || ' AS drtg, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['netrtg','net_rating']) || ' AS net_rtg, '
            || role_off_expr || ' AS ruolo_offensivo, '
            || role_def_expr || ' AS ruolo_difensivo, '
            || role_combo_expr || ' AS ruolo_combinato, '
            || on_net_expr || ' AS on_net_rtg, '
            || off_net_expr || ' AS off_net_rtg, '
            || net_diff_expr || ' AS net_rtg_diff, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['spm']) || ' AS spm, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['obpm']) || ' AS obpm, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dbpm']) || ' AS dbpm, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['gmsc','gm_sc']) || ' AS gm_sc, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fic']) || ' AS fic, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ows']) || ' AS ows, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['dws']) || ' AS dws, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptoroff','raptor_off']) || ' AS raptor_off, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptordef','raptor_def']) || ' AS raptor_def, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['raptortotal','raptor_total']) || ' AS raptor_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebronoff','lebron_off']) || ' AS lebron_off, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebrondef','lebron_def']) || ' AS lebron_def, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['lebrontotal','lebron_total']) || ' AS lebron_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['scoringefficiency','scoring_efficiency']) || ' AS scoring_efficiency, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ppsa']) || ' AS ppsa, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fg2pct','twopointpct']) || ' AS two_point_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tovpct','tov_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tovpct','tov_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tovpct','tov_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tovpct','tov_pct']) || ' END AS tov_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astpct','ast_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astpct','ast_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astpct','ast_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astpct','ast_pct']) || ' END AS ast_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stlpct','stl_pct']) || ' AS stl_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blkpct','blk_pct']) || ' AS blk_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['orebpct','orbpct','orb_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['orebpct','orbpct','orb_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['orebpct','orbpct','orb_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['orebpct','orbpct','orb_pct']) || ' END AS orb_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drebpct','drbpct','drb_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drebpct','drbpct','drb_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drebpct','drbpct','drb_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['drebpct','drbpct','drb_pct']) || ' END AS drb_pct, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['threepar','three_par']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['threepar','three_par']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['threepar','three_par']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['threepar','three_par']) || ' END AS three_par, '
            || 'CASE WHEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tusgpct','trueusgpct','true_usg_pct']) || ' IS NULL THEN NULL WHEN abs(' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tusgpct','trueusgpct','true_usg_pct']) || ') > 1.0 THEN ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tusgpct','trueusgpct','true_usg_pct']) || ' / 100.0 ELSE ' || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['tusgpct','trueusgpct','true_usg_pct']) || ' END AS true_usg_pct, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['fouldrawingrate','foul_drawing_rate']) || ' AS foul_drawing_rate, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['rfpergame','rf_per_game']) || ' AS rf_per_game, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['hustleindex','hustle_index']) || ' AS hustle_index, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['ptsper40','pts_per_40']) || ' AS pts_per_40, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['astper40','ast_per_40']) || ' AS ast_per_40, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['trper40','rebper40','tr_per_40']) || ' AS tr_per_40, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['stlper40','stl_per_40']) || ' AS stl_per_40, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 's', 'numeric', ARRAY['blkper40','blk_per_40']) || ' AS blk_per_40, '
            || clutch_games_expr || ' AS clutch_games, '
            || clutch_pts_expr || ' AS clutch_pts, '
            || clutch_ts_expr || ' AS clutch_ts_pct, '
            || clutch_ast_expr || ' AS clutch_ast_to_tov, '
            || clutch_net_expr || ' AS clutch_net_rtg, '
            || clutch_efg_expr || ' AS clutch_efg_pct, '
            || ortg_on_expr || ' AS ortg_on, '
            || ortg_off_expr || ' AS ortg_off, '
            || ortg_diff_expr || ' AS ortg_diff '
            || 'FROM "'
            || replace(r.table_schema, '"', '""')
            || '"."'
            || replace(r.table_name, '"', '""')
            || '" s '
            || 'JOIN "Anagrafiche"."'
            || replace(player_table, '"', '""')
            || '" p ON '
            || pg_temp.ai_id_key(player_local_expr) || ' = '
            || pg_temp.ai_id_key(
                pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'text', ARRAY['id'])
            )
            || ' AND '
            || season_expr || ' = '
            || pg_temp.ai_expr('Anagrafiche', player_table, 'p', 'int', ARRAY['season'])
            || ' LEFT JOIN "Anagrafiche"."'
            || replace(team_table, '"', '""')
            || '" dt ON '
            || season_expr || ' = '
            || pg_temp.ai_expr('Anagrafiche', team_table, 'dt', 'int', ARRAY['season'])
            || ' AND '
            || pg_temp.ai_id_key(team_source_expr) || ' = '
            || pg_temp.ai_id_key(
                pg_temp.ai_expr('Anagrafiche', team_table, 'dt', 'text', ARRAY['id'])
            )
            || ' LEFT JOIN "Anagrafiche"."'
            || replace(team_table, '"', '""')
            || '" nt ON '
            || season_expr || ' = '
            || pg_temp.ai_expr('Anagrafiche', team_table, 'nt', 'int', ARRAY['season'])
            || ' AND lower(btrim('
            || p_team_name_expr
            || '::text)) = lower(btrim('
            || pg_temp.ai_expr('Anagrafiche', team_table, 'nt', 'text', ARRAY['teamname','name','shortname'])
            || '::text)) '
            || role_join || onoff_join || clutch_join || boxscore_join
            || ' WHERE ' || season_expr || ' IS NOT NULL'
            || ' AND ' || player_local_expr || ' IS NOT NULL'
            || ' AND ' || rating_expr || ' IS NOT NULL';
    END LOOP;

    IF body = '' THEN
        body := 'SELECT NULL::text AS league_key, NULL::integer AS season, NULL::text AS source_player_id, NULL::text AS source_team_id, NULL::text AS player_global_id, NULL::text AS direct_team_global_id, NULL::text AS fallback_team_global_id, NULL::text AS competition, NULL::double precision AS games, NULL::double precision AS games_started, NULL::double precision AS starter_pct, NULL::double precision AS minutes_total, NULL::double precision AS points_total, NULL::double precision AS rebounds_total, NULL::double precision AS offensive_rebounds_total, NULL::double precision AS defensive_rebounds_total, NULL::double precision AS assists_total, NULL::double precision AS steals_total, NULL::double precision AS blocks_total, NULL::double precision AS turnovers_total, NULL::double precision AS fouls_total, NULL::double precision AS fg_pct, NULL::double precision AS three_point_pct, NULL::double precision AS ft_pct, NULL::double precision AS plus_minus, NULL::double precision AS per, NULL::double precision AS ts_pct, NULL::double precision AS usg_pct, NULL::double precision AS bpm, NULL::double precision AS vorp, NULL::double precision AS win_shares, NULL::double precision AS ast_ratio, NULL::double precision AS reb_pct, NULL::double precision AS rating_source, NULL::double precision AS ortg, NULL::double precision AS drtg, NULL::double precision AS net_rtg, NULL::text AS ruolo_offensivo, NULL::text AS ruolo_difensivo, NULL::text AS ruolo_combinato, NULL::double precision AS on_net_rtg, NULL::double precision AS off_net_rtg, NULL::double precision AS net_rtg_diff, NULL::double precision AS spm, NULL::double precision AS obpm, NULL::double precision AS dbpm, NULL::double precision AS gm_sc, NULL::double precision AS fic, NULL::double precision AS ows, NULL::double precision AS dws, NULL::double precision AS raptor_off, NULL::double precision AS raptor_def, NULL::double precision AS raptor_total, NULL::double precision AS lebron_off, NULL::double precision AS lebron_def, NULL::double precision AS lebron_total, NULL::double precision AS scoring_efficiency, NULL::double precision AS ppsa, NULL::double precision AS two_point_pct, NULL::double precision AS tov_pct, NULL::double precision AS ast_pct, NULL::double precision AS stl_pct, NULL::double precision AS blk_pct, NULL::double precision AS orb_pct, NULL::double precision AS drb_pct, NULL::double precision AS three_par, NULL::double precision AS true_usg_pct, NULL::double precision AS foul_drawing_rate, NULL::double precision AS rf_per_game, NULL::double precision AS hustle_index, NULL::double precision AS pts_per_40, NULL::double precision AS ast_per_40, NULL::double precision AS tr_per_40, NULL::double precision AS stl_per_40, NULL::double precision AS blk_per_40, NULL::double precision AS clutch_games, NULL::double precision AS clutch_pts, NULL::double precision AS clutch_ts_pct, NULL::double precision AS clutch_ast_to_tov, NULL::double precision AS clutch_net_rtg, NULL::double precision AS clutch_efg_pct, NULL::double precision AS ortg_on, NULL::double precision AS ortg_off, NULL::double precision AS ortg_diff WHERE false';
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."PlayerCompetitionStats" AS
         WITH raw AS (' || body || '),
         resolved AS (
             SELECT r.*,
                    row_number() OVER (
                        PARTITION BY r.player_global_id, r.league_key, r.season, r.competition
                        ORDER BY CASE WHEN r.source_team_id IS NULL THEN 0 ELSE 1 END,
                                 coalesce(r.games, 0) DESC,
                                 r.source_team_id NULLS LAST
                    ) AS rn
             FROM raw r
             WHERE r.rating_source IS NOT NULL
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
                            ORDER BY r.rating_source
                        )
                    END AS rating_0_10
             FROM resolved r
             WHERE r.rn = 1
         )
         SELECT
             ' || pg_temp.ai_stable_id('player','player_global_id') || ' AS player_id,
             season,
             ' || pg_temp.ai_stable_id('team','coalesce(direct_team_global_id, fallback_team_global_id)') || ' AS team_id,
             ' || pg_temp.ai_stable_id('league','league_key') || ' AS league_id,
             coalesce(games, 0)::integer AS games_played,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(minutes_total,0)/games ELSE 0 END AS minutes_per_game,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(points_total,0)/games ELSE 0 END AS points,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(rebounds_total,0)/games ELSE 0 END AS rebounds,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(offensive_rebounds_total,0)/games ELSE 0 END AS offensive_rebounds,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(defensive_rebounds_total,0)/games ELSE 0 END AS defensive_rebounds,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(assists_total,0)/games ELSE 0 END AS assists,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(steals_total,0)/games ELSE 0 END AS steals,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(blocks_total,0)/games ELSE 0 END AS blocks,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(turnovers_total,0)/games ELSE 0 END AS turnovers,
             CASE WHEN coalesce(games,0)>0 THEN coalesce(fouls_total,0)/games ELSE 0 END AS personal_fouls,
             coalesce(fg_pct,0) AS fg_pct,
             coalesce(three_point_pct,0) AS three_point_pct,
             coalesce(ft_pct,0) AS ft_pct,
             coalesce(plus_minus,0) AS plus_minus,
             per, ts_pct, usg_pct, bpm, vorp, win_shares, ast_ratio, reb_pct,
             rating_0_10 AS rating,
             coalesce(ortg,0) AS ortg,
             coalesce(drtg,0) AS drtg,
             coalesce(net_rtg,0) AS net_rtg,
             coalesce(ruolo_offensivo,'''') AS ruolo_offensivo,
             coalesce(ruolo_difensivo,'''') AS ruolo_difensivo,
             coalesce(ruolo_combinato,'''') AS ruolo_combinato,
             coalesce(on_net_rtg,0) AS on_net_rtg,
             coalesce(off_net_rtg,0) AS off_net_rtg,
             coalesce(net_rtg_diff,0) AS net_rtg_diff,
             competition,
             spm, obpm, dbpm, gm_sc, fic, ows, dws,
             raptor_off, raptor_def, raptor_total,
             lebron_off, lebron_def, lebron_total,
             scoring_efficiency, ppsa, two_point_pct,
             tov_pct, ast_pct, stl_pct, blk_pct, orb_pct, drb_pct,
             three_par, true_usg_pct, foul_drawing_rate, rf_per_game,
             hustle_index, pts_per_40, ast_per_40, tr_per_40,
             stl_per_40, blk_per_40,
             coalesce(clutch_games,0) AS clutch_games,
             coalesce(clutch_pts,0) AS clutch_pts,
             coalesce(clutch_ts_pct,0) AS clutch_ts_pct,
             coalesce(clutch_ast_to_tov,0) AS clutch_ast_to_tov,
             coalesce(clutch_net_rtg,0) AS clutch_net_rtg,
             coalesce(clutch_efg_pct,0) AS clutch_efg_pct,
             coalesce(ortg_on,0) AS ortg_on,
             coalesce(ortg_off,0) AS ortg_off,
             coalesce(ortg_diff,0) AS ortg_diff,
             coalesce(games_started,0)::integer AS games_started,
             coalesce(
                 starter_pct,
                 CASE WHEN coalesce(games,0)>0
                      THEN least(1.0, coalesce(games_started,0)/games)
                      ELSE 0 END
             ) AS starter_pct,
             league_key
         FROM scored'
    ;
END;
$player_stats$;

-- ---------------------------------------------------------------------------
-- Canonical TeamCompetitionStats
-- ---------------------------------------------------------------------------

DO $team_stats$
DECLARE
    r record;
    body text := '';
    league_key text;
    team_table text;
    season_expr text;
    team_local_expr text;
    comp_expr text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname='Analisi'
          AND lower(c.relname) LIKE 'advancedstatsteam_%'
          AND c.relkind IN ('r','p')
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
                AND lower(a.attname)='season'
          )
        ORDER BY c.relname
    LOOP
        league_key := regexp_replace(r.table_name, '^AdvancedStatsTeam_', '', 'i');
        team_table := 'Team_' || league_key;

        IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='Anagrafiche' AND c.relname=team_table
              AND c.relkind IN ('r','p')
        ) THEN
            CONTINUE;
        END IF;

        season_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'int', ARRAY['season']);
        team_local_expr := pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'text', ARRAY['teamid','id']);
        comp_expr := pg_temp.ai_comp_expr(r.table_schema, r.table_name, 't', ARRAY['competition']);

        body := body
            || CASE WHEN body='' THEN '' ELSE chr(10) || 'UNION ALL' || chr(10) END
            || 'SELECT '
            || quote_literal(league_key) || '::text AS league_key, '
            || season_expr || '::integer AS season, '
            || team_local_expr || ' AS source_team_id, '
            || comp_expr || ' AS competition, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['games']) || ' AS games, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['pace']) || ' AS pace, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['ortg']) || ' AS ortg, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['drtg']) || ' AS drtg, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['netrtg','net_rating']) || ' AS net_rtg, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['threepar','fg3rate']) || ' AS three_par, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['fg3a']) || ' AS fg3a, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['fg2a']) || ' AS fg2a, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['ast']) || ' AS ast_total, '
            || pg_temp.ai_expr(r.table_schema, r.table_name, 't', 'numeric', ARRAY['astpergame']) || ' AS ast_per_game, '
            || 'coalesce('
            || pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['idglobal','globalid'])
            || ', ' || quote_literal(league_key) || ' || ' || pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['id']) || ') AS team_global_id, '
            || pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['teamname','name','shortname']) || ' AS team_name, '
            || pg_temp.ai_expr('Anagrafiche', team_table, 'tr', 'text', ARRAY['shortname','short_name']) || ' AS short_name '
            || 'FROM "' || replace(r.table_schema,'"','""') || '"."' || replace(r.table_name,'"','""') || '" t '
            || 'LEFT JOIN "Anagrafiche"."'
            || replace(team_table,'"','""')
            || '" tr ON ' || season_expr || ' = '
            || pg_temp.ai_expr('Anagrafiche',team_table,'tr','int',ARRAY['season'])
            || ' AND ' || pg_temp.ai_id_key(team_local_expr) || ' = '
            || pg_temp.ai_id_key(
                pg_temp.ai_expr('Anagrafiche',team_table,'tr','text',ARRAY['id'])
            )
            || ' WHERE ' || season_expr || ' IS NOT NULL AND ' || team_local_expr || ' IS NOT NULL';
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::text league_key, NULL::integer season, NULL::text source_team_id, NULL::text competition, NULL::double precision games, NULL::double precision pace, NULL::double precision ortg, NULL::double precision drtg, NULL::double precision net_rtg, NULL::double precision three_par, NULL::double precision fg3a, NULL::double precision fg2a, NULL::double precision ast_total, NULL::double precision ast_per_game, NULL::text team_global_id, NULL::text team_name, NULL::text short_name WHERE false';
    END IF;

    EXECUTE
        'CREATE TABLE "AI_Source"."TeamCompetitionStats" AS
         WITH raw AS (' || body || '),
         resolved AS (
             SELECT r.*,
                    row_number() OVER (
                        PARTITION BY r.team_global_id, r.league_key, r.season, r.competition
                        ORDER BY coalesce(r.games,0) DESC, r.source_team_id
                    ) AS rn
             FROM raw r
         )
         SELECT
             ' || pg_temp.ai_stable_id('team','team_global_id') || ' AS team_id,
             team_global_id AS global_id,
             coalesce(team_name,'''') AS name,
             ' || pg_temp.ai_stable_id('league','league_key') || ' AS league_id,
             league_key,
             season,
             competition,
             coalesce(games,0)::integer AS games_played,
             coalesce(pace,75.0) AS pace,
             coalesce(ortg,110.0) AS offensive_rating,
             coalesce(drtg,110.0) AS defensive_rating,
             coalesce(
                 three_par,
                 CASE WHEN coalesce(fg3a,0)+coalesce(fg2a,0)>0
                      THEN fg3a/(fg3a+fg2a) ELSE NULL END,
                 0.35
             ) AS three_point_attempt_rate,
             coalesce(
                 ast_per_game,
                 CASE WHEN coalesce(games,0)>0 THEN ast_total/games ELSE NULL END,
                 20.0
             ) AS assists_per_game,
             coalesce(net_rtg,coalesce(ortg,110.0)-coalesce(drtg,110.0)) AS net_rtg,
             0.25::double precision AS star_player_usage,
             coalesce(short_name,'''') AS short_name
         FROM resolved r
         WHERE rn=1'
    ;
END;
$team_stats$;

UPDATE "AI_Source"."TeamCompetitionStats" t
SET star_player_usage = coalesce(s.star_player_usage, 0.25)
FROM (
    SELECT
        team_id, league_id, season, competition,
        least(
            0.60,
            greatest(
                0.10,
                CASE WHEN max(usg_pct) > 1.5
                     THEN max(usg_pct)/100.0
                     ELSE max(usg_pct)
                END
            )
        ) AS star_player_usage
    FROM "AI_Source"."PlayerCompetitionStats"
    WHERE team_id IS NOT NULL
      AND usg_pct IS NOT NULL
    GROUP BY team_id, league_id, season, competition
) s
WHERE t.team_id=s.team_id
  AND t.league_id=s.league_id
  AND t.season=s.season
  AND t.competition=s.competition;

-- ---------------------------------------------------------------------------
-- Player registry staging (TEMP only, never persistent).
-- ---------------------------------------------------------------------------

DO $players_registry$
DECLARE
    r record;
    body text := '';
    league_key text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='Anagrafiche'
          AND lower(left(c.relname,5)) <> 'team_'
          AND c.relkind IN ('r','p')
          AND pg_temp.ai_find_column('Anagrafiche',c.relname,ARRAY['season']) IS NOT NULL
          AND pg_temp.ai_find_column('Anagrafiche',c.relname,ARRAY['id']) IS NOT NULL
        ORDER BY c.relname
    LOOP
        league_key := r.table_name;
        body := body
            || CASE WHEN body='' THEN '' ELSE chr(10) || 'UNION ALL' || chr(10) END
            || 'SELECT '
            || pg_temp.ai_stable_id('player',
                format(
                    'coalesce(%s, %L || '':'' || %s)',
                    pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['idglobal','globalid']),
                    league_key,
                    pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['id'])
                )
              )
            || ' AS player_id, '
            || 'coalesce('
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['idglobal','globalid'])
            || ', ' || quote_literal(league_key) || ' || ' ||
            pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['id'])
            || ') AS global_id, '
            || 'coalesce('
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['normalizedplayername'])
            || ', ' || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['playername'])
            || ', ' || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['name'])
            || ') AS name, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','date',ARRAY['birthdate','dateofbirth','dob']) || ' AS birth_date, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','int',ARRAY['age']) || ' AS source_age, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['pos','position']) || ' AS position, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['nat','nationality']) || ' AS nationality, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','numeric',ARRAY['cm','heightcm','height_cm']) || ' AS height_cm, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','numeric',ARRAY['weight','weightkg','weight_kg']) || ' AS weight_kg, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','int',ARRAY['shirtnumber','jerseynumber']) || ' AS jersey_number, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['team','teamid','idteam']) || ' AS source_team_id, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','text',ARRAY['teamname','team_name']) || ' AS team_name, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'p','int',ARRAY['season']) || ' AS season, '
            || quote_literal(league_key) || '::text AS league_key '
            || 'FROM "Anagrafiche"."'
            || replace(r.table_name,'"','""')
            || '" p';
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::bigint player_id, NULL::text global_id, NULL::text name, NULL::date birth_date, NULL::bigint source_age, NULL::text position, NULL::text nationality, NULL::double precision height_cm, NULL::double precision weight_kg, NULL::bigint jersey_number, NULL::text source_team_id, NULL::text team_name, NULL::integer season, NULL::text league_key WHERE false';
    END IF;

    EXECUTE
        'CREATE TEMP TABLE pg_temp.ai_players_registry ON COMMIT DROP AS
         SELECT DISTINCT ON (player_id, season, league_key) *
         FROM (' || body || ') q
         WHERE player_id IS NOT NULL AND season IS NOT NULL
         ORDER BY
             player_id,
             season,
             league_key,
             CASE WHEN NULLIF(btrim(name), '') IS NULL THEN 1 ELSE 0 END,
             CASE WHEN birth_date IS NULL THEN 1 ELSE 0 END,
             CASE WHEN source_age IS NULL THEN 1 ELSE 0 END,
             CASE WHEN NULLIF(btrim(position), '') IS NULL THEN 1 ELSE 0 END,
             CASE WHEN NULLIF(btrim(nationality), '') IS NULL THEN 1 ELSE 0 END,
             CASE WHEN NULLIF(btrim(source_team_id), '') IS NULL THEN 1 ELSE 0 END,
             CASE WHEN jersey_number IS NULL THEN 1 ELSE 0 END,
             name';

    CREATE INDEX ai_players_registry_key
        ON pg_temp.ai_players_registry (player_id, season, league_key);

    CREATE INDEX ai_players_registry_team
        ON pg_temp.ai_players_registry (league_key, season, source_team_id, team_name);
END;
$players_registry$;

-- ---------------------------------------------------------------------------
-- Team registry staging (TEMP only).
-- ---------------------------------------------------------------------------

DO $teams_registry$
DECLARE
    r record;
    body text := '';
    league_key text;
BEGIN
    FOR r IN
        SELECT n.nspname AS table_schema, c.relname AS table_name
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='Anagrafiche'
          AND lower(left(c.relname,5))='team_'
          AND c.relkind IN ('r','p')
          AND pg_temp.ai_find_column('Anagrafiche',c.relname,ARRAY['season']) IS NOT NULL
          AND pg_temp.ai_find_column('Anagrafiche',c.relname,ARRAY['id']) IS NOT NULL
        ORDER BY c.relname
    LOOP
        league_key := regexp_replace(r.table_name,'^Team_','','i');
        body := body
            || CASE WHEN body='' THEN '' ELSE chr(10) || 'UNION ALL' || chr(10) END
            || 'SELECT '
            || pg_temp.ai_stable_id('team',
                format(
                    'coalesce(%s, %L || '':'' || %s)',
                    pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['idglobal','globalid']),
                    league_key,
                    pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['id'])
                )
              )
            || ' AS team_id, '
            || 'coalesce('
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['idglobal','globalid'])
            || ', ' || quote_literal(league_key) || ' || ' ||
            pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['id'])
            || ') AS global_id, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['id']) || ' AS source_team_id, '
            || 'coalesce('
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['teamname'])
            || ', ' || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['name'])
            || ', ' || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['shortname'])
            || ') AS name, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','text',ARRAY['shortname','short_name']) || ' AS short_name, '
            || pg_temp.ai_expr('Anagrafiche',r.table_name,'t','int',ARRAY['season']) || ' AS season, '
            || quote_literal(league_key) || '::text AS league_key '
            || 'FROM "Anagrafiche"."'
            || replace(r.table_name,'"','""')
            || '" t';
    END LOOP;

    IF body='' THEN
        body := 'SELECT NULL::bigint team_id, NULL::text global_id, NULL::text source_team_id, NULL::text name, NULL::text short_name, NULL::integer season, NULL::text league_key WHERE false';
    END IF;

    EXECUTE
        'CREATE TEMP TABLE pg_temp.ai_teams_registry ON COMMIT DROP AS
         SELECT DISTINCT ON (team_id, season, league_key) *
         FROM (' || body || ') q
         WHERE team_id IS NOT NULL AND season IS NOT NULL
         ORDER BY team_id, season, league_key';

    CREATE INDEX ai_teams_registry_key
        ON pg_temp.ai_teams_registry (team_id, season, league_key);

    CREATE INDEX ai_teams_registry_name
        ON pg_temp.ai_teams_registry (league_key, season, name);
END;
$teams_registry$;

-- ---------------------------------------------------------------------------
-- TeamPlayerRelations — derived without any PBP dependency.
--
-- 1) direct team ids from canonical player competition stats
-- 2) historical roster/team-name matches for seasons where no stats exist
-- ---------------------------------------------------------------------------

CREATE TABLE "AI_Source"."TeamPlayerRelations" AS
WITH stats_rel AS (
    SELECT
        s.player_id,
        s.team_id,
        s.season,
        coalesce(p.position, '') AS role,
        p.jersey_number,
        1 AS relation_priority
    FROM "AI_Source"."PlayerCompetitionStats" s
    JOIN pg_temp.ai_players_registry p
      ON p.player_id=s.player_id
     AND p.season=s.season
     AND p.league_key=s.league_key
    WHERE s.player_id IS NOT NULL
      AND s.team_id IS NOT NULL
),
roster_rel AS (
    SELECT
        p.player_id,
        t.team_id,
        p.season,
        coalesce(p.position, '') AS role,
        p.jersey_number,
        2 AS relation_priority
    FROM pg_temp.ai_players_registry p
    JOIN pg_temp.ai_teams_registry t
      ON t.league_key=p.league_key
     AND t.season=p.season
     AND (
         (
             p.source_team_id IS NOT NULL
             AND t.source_team_id IS NOT NULL
             AND btrim(p.source_team_id) = btrim(t.source_team_id)
         )
         OR
         (
             p.source_team_id IS NULL
             AND p.team_name IS NOT NULL
             AND (
                 lower(btrim(p.team_name)) = lower(btrim(t.name))
                 OR lower(btrim(p.team_name)) = lower(btrim(coalesce(t.short_name,'')))
             )
         )
     )
    WHERE (p.source_team_id IS NOT NULL OR p.team_name IS NOT NULL)
      AND p.player_id IS NOT NULL
      AND t.team_id IS NOT NULL
),
deduped AS (
    SELECT DISTINCT ON (player_id, team_id, season)
           player_id,
           team_id,
           season,
           role,
           jersey_number
    FROM (
        SELECT * FROM stats_rel
        UNION ALL
        SELECT * FROM roster_rel
    ) q
    WHERE player_id IS NOT NULL
      AND team_id IS NOT NULL
      AND season IS NOT NULL
    ORDER BY
        player_id,
        team_id,
        season,
        relation_priority,
        CASE WHEN role = '' THEN 1 ELSE 0 END,
        CASE WHEN jersey_number IS NULL THEN 1 ELSE 0 END,
        role,
        jersey_number
)
SELECT player_id, team_id, season, role, jersey_number
FROM deduped;

-- ---------------------------------------------------------------------------
-- Players — latest registry state + latest known historical relation.
-- ---------------------------------------------------------------------------

CREATE TABLE "AI_Source"."Players" AS
WITH ranked AS (
    SELECT p.*,
           row_number() OVER (
               PARTITION BY p.player_id
               ORDER BY p.season DESC, p.league_key
           ) AS rn
    FROM pg_temp.ai_players_registry p
),
current_relation AS (
    SELECT DISTINCT ON (player_id)
           player_id, team_id
    FROM "AI_Source"."TeamPlayerRelations"
    ORDER BY player_id, season DESC, team_id DESC
)
SELECT
    p.player_id AS id,
    p.global_id,
    p.name AS name,
    CASE
        WHEN p.source_age BETWEEN 14 AND 44
            THEN p.source_age
        WHEN p.birth_date IS NOT NULL
            THEN p.season - extract(year FROM p.birth_date)::integer
        ELSE NULL
    END::integer AS age,
    NULLIF(p.position,'') AS position,
    NULLIF(p.nationality,'') AS nationality,
    p.height_cm,
    p.weight_kg,
    NULL::text AS dominant_hand,
    cr.team_id AS current_team_id,
    CASE
        WHEN p.league_key IS NULL OR btrim(p.league_key)=''
            THEN NULL::bigint
        ELSE hashtextextended('league:' || btrim(p.league_key),0)
             & 9223372036854775807::bigint
    END AS current_league_id,
    NULL::integer AS draft_year,
    NULL::integer AS draft_pick,
    p.birth_date,
    p.season AS age_reference_season,
    p.league_key AS current_league_key
FROM ranked p
LEFT JOIN current_relation cr ON cr.player_id=p.player_id
WHERE p.rn=1;

-- ---------------------------------------------------------------------------
-- Teams — latest registry state + preferred latest team-season stats.
-- ---------------------------------------------------------------------------

CREATE TABLE "AI_Source"."Teams" AS
WITH ranked AS (
    SELECT t.*,
           row_number() OVER (
               PARTITION BY t.team_id
               ORDER BY t.season DESC, t.league_key
           ) AS rn
    FROM pg_temp.ai_teams_registry t
),
preferred_stats AS (
    SELECT DISTINCT ON (team_id, league_id, season)
           team_id, league_id, season,
           pace, offensive_rating, defensive_rating,
           three_point_attempt_rate, assists_per_game,
           net_rtg, star_player_usage, short_name
    FROM "AI_Source"."TeamCompetitionStats"
    ORDER BY team_id, league_id, season,
             CASE upper(competition)
                 WHEN 'TOT' THEN 0
                 WHEN 'RS' THEN 1
                 ELSE 2
             END,
             competition
)
SELECT
    r.team_id AS id,
    r.global_id,
    coalesce(r.name,r.global_id) AS name,
    CASE
        WHEN r.league_key IS NULL OR btrim(r.league_key)=''
            THEN NULL::bigint
        ELSE hashtextextended('league:' || btrim(r.league_key),0)
             & 9223372036854775807::bigint
    END AS league_id,
    ''::text AS playing_style,
    ''::text AS formation,
    coalesce(s.pace,75.0) AS pace,
    coalesce(s.offensive_rating,110.0) AS offensive_rating,
    coalesce(s.defensive_rating,110.0) AS defensive_rating,
    coalesce(s.three_point_attempt_rate,0.35) AS three_point_attempt_rate,
    coalesce(s.assists_per_game,20.0) AS assists_per_game,
    coalesce(s.star_player_usage,0.25) AS star_player_usage,
    1::integer AS league_tier,
    coalesce(s.short_name,r.short_name,'') AS short_name,
    coalesce(
        s.net_rtg,
        coalesce(s.offensive_rating,110.0)-coalesce(s.defensive_rating,110.0)
    ) AS net_rtg,
    r.league_key,
    r.season AS latest_season
FROM ranked r
LEFT JOIN preferred_stats s
  ON s.team_id=r.team_id
 AND s.league_id=CASE
        WHEN r.league_key IS NULL OR btrim(r.league_key)=''
            THEN NULL::bigint
        ELSE hashtextextended('league:' || btrim(r.league_key),0)
             & 9223372036854775807::bigint
    END
 AND s.season=r.season
WHERE r.rn=1;

-- ---------------------------------------------------------------------------
-- Leagues — only league keys present in canonical entities/statistics.
-- ---------------------------------------------------------------------------

CREATE TABLE "AI_Source"."Leagues" AS
WITH keys AS (
    SELECT DISTINCT current_league_key AS league_key FROM "AI_Source"."Players"
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
    hashtextextended('league:' || btrim(k.league_key),0)
        & 9223372036854775807::bigint AS id,
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
        ELSE regexp_replace(k.league_key,'[0-9]+$','')
    END AS country,
    1::integer AS tier,
    1.0::double precision AS competitiveness_score,
    coalesce(ta.avg_pace,75.0) AS avg_pace,
    coalesce(ta.avg_offensive_rating,110.0) AS avg_offensive_rating,
    greatest(coalesce(o.max_games,1),1)::integer AS max_games,
    k.league_key
FROM keys k
LEFT JOIN observed o
  ON o.league_id =
     hashtextextended('league:' || btrim(k.league_key),0)
     & 9223372036854775807::bigint
LEFT JOIN team_average ta
  ON ta.league_id =
     hashtextextended('league:' || btrim(k.league_key),0)
     & 9223372036854775807::bigint
WHERE k.league_key IS NOT NULL
  AND btrim(k.league_key) <> '';

-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Read access for the model runtime and administrative account.
--
-- The canonical tables are recreated by this script, so grants must be
-- re-applied after every rebuild. "ai_model" is the application runtime role;
-- "hm_admin" is the permanent administrative/read role used for diagnostics,
-- BO operations and database maintenance.
-- ---------------------------------------------------------------------------

DO $grant_ai_source$
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
                    "AI_Source"."Leagues",
                    "AI_Source"."Teams",
                    "AI_Source"."Players",
                    "AI_Source"."TeamPlayerRelations",
                    "AI_Source"."PlayerCompetitionStats",
                    "AI_Source"."TeamCompetitionStats"
                 TO %I',
                read_role
            );
            RAISE NOTICE
                'AI_Source canonical read access granted to role %',
                read_role;
        ELSE
            RAISE NOTICE
                'AI_Source read role "%" does not exist; no grant applied',
                read_role;
        END IF;
    END LOOP;
END;
$grant_ai_source$;

-- Only final canonical indexes. The model mostly bulk-loads these tables, so
-- indexes are limited to identity/time access paths actually used downstream.
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
-- SELECT count(*) FROM "AI_Source"."Players";
-- SELECT count(*) FROM "AI_Source"."Teams";
-- SELECT count(*) FROM "AI_Source"."TeamPlayerRelations";
-- SELECT competition, count(*) FROM "AI_Source"."PlayerCompetitionStats"
--   GROUP BY competition ORDER BY competition;
-- SELECT competition, count(*) FROM "AI_Source"."TeamCompetitionStats"

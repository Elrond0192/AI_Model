-- Historical team-season context for leakage-free ML training/evaluation.
-- Run after basketball_ai/data/ai_source_schema.sql.
--
-- IMPORTANT: this adapter discovers physical tables independently instead of
-- depending on ai_source._team_registry / _team_stats_raw. That keeps the base
-- ai_source_schema.sql idempotent: it can rebuild its own internal views without
-- being blocked by dependencies from team_season_stats.

BEGIN;

DROP VIEW IF EXISTS ai_source.team_season_stats;
DROP VIEW IF EXISTS ai_source._team_season_stats_raw;
DROP VIEW IF EXISTS ai_source._team_season_registry;

-- One physical team registry row per league/season/source ID.
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
                    ai_source._int(j.doc, 'season')::integer AS season,
                    ai_source._text(j.doc, 'id') AS source_team_id,
                    COALESCE(
                        ai_source._text(j.doc, 'idglobal', 'globalid'),
                        %L || ':' || ai_source._text(j.doc, 'id')
                    ) AS global_id,
                    COALESCE(
                        ai_source._text(j.doc, 'teamname', 'name', 'shortname'),
                        ai_source._text(j.doc, 'id')
                    ) AS name,
                    ai_source._text(j.doc, 'shortname', 'short_name') AS short_name
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT ai_source._lower_keys(to_jsonb(src)) AS doc
                ) j
                WHERE ai_source._int(j.doc, 'season') IS NOT NULL
                  AND ai_source._text(j.doc, 'id') IS NOT NULL
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

    EXECUTE 'CREATE VIEW ai_source._team_season_registry AS ' || body;
END;
$$;

-- Historical team metrics from the physical AdvancedStatsTeam tables.
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
                    ai_source._int(j.doc, 'season')::integer AS season,
                    ai_source._text(j.doc, 'teamid', 'id') AS source_team_id,
                    upper(COALESCE(ai_source._text(j.doc, 'competition'), 'RS')) AS competition,
                    ai_source._num(j.doc, 'games') AS games,
                    ai_source._num(j.doc, 'pace') AS pace,
                    ai_source._num(j.doc, 'ortg') AS ortg,
                    ai_source._num(j.doc, 'drtg') AS drtg,
                    ai_source._num(j.doc, 'netrtg') AS net_rtg,
                    ai_source._num(j.doc, 'threepar', 'fg3rate') AS three_par,
                    ai_source._num(j.doc, 'fg3a') AS fg3a,
                    ai_source._num(j.doc, 'fg2a') AS fg2a,
                    ai_source._num(j.doc, 'ast') AS ast_total,
                    ai_source._num(j.doc, 'astpergame') AS ast_per_game
                FROM %I.%I src
                CROSS JOIN LATERAL (
                    SELECT ai_source._lower_keys(to_jsonb(src)) AS doc
                ) j
                WHERE ai_source._int(j.doc, 'season') IS NOT NULL
                  AND ai_source._text(j.doc, 'teamid', 'id') IS NOT NULL
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

    EXECUTE 'CREATE VIEW ai_source._team_season_stats_raw AS ' || body;
END;
$$;

CREATE VIEW ai_source.team_season_stats AS
WITH registry_ranked AS (
    SELECT
        tr.*,
        row_number() OVER (
            PARTITION BY tr.global_id, tr.season
            ORDER BY tr.league_key, tr.source_team_id
        ) AS rn
    FROM ai_source._team_season_registry tr
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
                COALESCE(ts.games, 0) DESC,
                ts.competition
        ) AS rn
    FROM ai_source._team_season_stats_raw ts
)
SELECT
    ai_source._stable_id('team', tr.global_id) AS team_id,
    tr.global_id,
    tr.name,
    ai_source._stable_id('league', tr.league_key) AS league_id,
    tr.league_key,
    tr.season,
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
        CASE
            WHEN COALESCE(ts.games, 0) > 0 THEN ts.ast_total / ts.games
            ELSE NULL
        END,
        20.0
    ) AS assists_per_game,
    COALESCE(
        ts.net_rtg,
        COALESCE(ts.ortg, 110.0) - COALESCE(ts.drtg, 110.0)
    ) AS net_rtg,
    0.25::double precision AS star_player_usage,
    COALESCE(tr.short_name, '') AS short_name
FROM registry_ranked tr
LEFT JOIN team_stats_ranked ts
  ON ts.league_key = tr.league_key
 AND ts.season = tr.season
 AND ts.source_team_id = tr.source_team_id
 AND ts.rn = 1
WHERE tr.rn = 1;

COMMIT;

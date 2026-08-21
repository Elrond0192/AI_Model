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
DROP VIEW IF EXISTS "AI_Source"."CompetitionBoxscoreRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionClutchRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionOnOffRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionRolesRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionTeamStatsRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionStatsRawInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionTeamRegistryInternal";
DROP VIEW IF EXISTS "AI_Source"."CompetitionPlayerRegistryInternal";

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
    s.competition,
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
    COALESCE(s.games_started, bs.games_started, 0)::integer AS games_started,
    COALESCE(
        s.starter_pct,
        CASE WHEN COALESCE(s.games, 0) > 0
             THEN LEAST(1.0, COALESCE(bs.games_started, 0) / s.games)
             ELSE 0 END
    ) AS starter_pct,
    s.league_key
FROM resolved s
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
WHERE s.rn = 1;

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

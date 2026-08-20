-- Historical team-season context for leakage-free ML training/evaluation.
-- Run after basketball_ai/data/ai_source_schema.sql.
-- Source schemas are read-only; this view depends only on ai_source adapter views.

BEGIN;

DROP VIEW IF EXISTS ai_source.team_season_stats;

CREATE VIEW ai_source.team_season_stats AS
WITH registry_ranked AS (
    SELECT
        tr.*,
        row_number() OVER (
            PARTITION BY tr.global_id, tr.season
            ORDER BY tr.league_key, tr.source_team_id
        ) AS rn
    FROM ai_source._team_registry tr
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
    FROM ai_source._team_stats_raw ts
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
        CASE WHEN COALESCE(ts.games, 0) > 0 THEN ts.ast_total / ts.games ELSE NULL END,
        20.0
    ) AS assists_per_game,
    COALESCE(ts.net_rtg, COALESCE(ts.ortg, 110.0) - COALESCE(ts.drtg, 110.0)) AS net_rtg,
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

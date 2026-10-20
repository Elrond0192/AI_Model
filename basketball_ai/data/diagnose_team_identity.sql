-- Read-only diagnostics for canonical team identity gaps.
-- Run after ai_source_full.sql has been applied on the target database.
-- This file contains SELECT statements only; it does not alter production data.

-- 1) For consecutive-season transfers into GRC1/LIT1, show which side is missing.
WITH rs AS (
    SELECT player_id, season, league_key, league_id, team_id
    FROM "AI_Source"."PlayerCompetitionStats"
    WHERE competition = 'RS'
      AND season BETWEEN 2018 AND 2025
)
SELECT
    s.season AS source_season,
    t.season AS destination_season,
    s.league_key AS source_league,
    t.league_key AS destination_league,
    count(*) AS player_season_pairs,
    count(*) FILTER (WHERE s.team_id IS NULL) AS source_team_missing,
    count(*) FILTER (WHERE t.team_id IS NULL) AS destination_team_missing,
    count(*) FILTER (WHERE s.team_id IS NULL AND t.team_id IS NULL) AS both_teams_missing,
    count(*) FILTER (
        WHERE s.team_id IS NOT NULL AND t.team_id IS NOT NULL AND s.team_id = t.team_id
    ) AS same_team_id,
    count(*) FILTER (
        WHERE s.team_id IS NOT NULL AND t.team_id IS NOT NULL AND s.team_id <> t.team_id
    ) AS different_team_id
FROM rs s
JOIN rs t
  ON t.player_id = s.player_id
 AND t.season = s.season + 1
WHERE t.league_key IN ('GRC1', 'LIT1')
GROUP BY s.season, t.season, s.league_key, t.league_key
ORDER BY t.league_key, s.season, s.league_key;

-- 2) Detail each missing regular-season assignment and determine whether
-- TeamPlayerRelations + TeamCompetitionStats offer exactly one safe candidate.
WITH missing AS (
    SELECT
        s.player_id,
        s.league_id,
        s.league_key,
        s.season,
        s.competition,
        s.games_played,
        p.global_id AS player_global_id,
        p.name AS player_name
    FROM "AI_Source"."PlayerCompetitionStats" s
    LEFT JOIN "AI_Source"."Players" p ON p.id = s.player_id
    WHERE s.season BETWEEN 2018 AND 2026
      AND s.league_key IN ('GRC1', 'LIT1')
      AND s.competition = 'RS'
      AND s.team_id IS NULL
),
candidates AS (
    SELECT
        m.player_id,
        m.league_id,
        m.season,
        m.league_key,
        count(DISTINCT t.team_id) FILTER (WHERE t.team_id IS NOT NULL) AS candidate_team_count,
        array_agg(DISTINCT t.global_id) FILTER (WHERE t.global_id IS NOT NULL) AS candidate_global_ids,
        array_agg(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL) AS candidate_team_names
    FROM missing m
    LEFT JOIN "AI_Source"."TeamPlayerRelations" r
      ON r.player_id = m.player_id
     AND r.season = m.season
    LEFT JOIN "AI_Source"."TeamCompetitionStats" t
      ON t.team_id = r.team_id
     AND t.league_id = m.league_id
     AND t.season = m.season
    GROUP BY m.player_id, m.league_id, m.season, m.league_key
)
SELECT
    m.league_key,
    m.season,
    m.player_global_id,
    m.player_name,
    m.games_played,
    coalesce(c.candidate_team_count, 0) AS candidate_team_count,
    c.candidate_global_ids,
    c.candidate_team_names,
    CASE
        WHEN coalesce(c.candidate_team_count, 0) = 0 THEN 'no_matching_roster_team'
        WHEN c.candidate_team_count = 1 THEN 'one_safe_candidate_should_be_reconciled'
        ELSE 'multiple_teams_do_not_guess'
    END AS diagnosis
FROM missing m
LEFT JOIN candidates c
  ON c.player_id = m.player_id
 AND c.league_id = m.league_id
 AND c.season = m.season
 AND c.league_key = m.league_key
ORDER BY m.league_key, m.season, diagnosis, m.player_name;

-- 3) Establish whether gaps originate mainly in player stats or in team-season
-- coverage. Compare row counts and null team identities by season/competition.
SELECT
    league_key,
    season,
    competition,
    count(*) AS player_stat_rows,
    count(*) FILTER (WHERE team_id IS NULL) AS player_rows_without_team,
    round(
        100.0 * count(*) FILTER (WHERE team_id IS NULL) / nullif(count(*), 0),
        2
    ) AS pct_player_rows_without_team
FROM "AI_Source"."PlayerCompetitionStats"
WHERE season BETWEEN 2018 AND 2026
  AND league_key IN ('GRC1', 'LIT1')
GROUP BY league_key, season, competition
ORDER BY league_key, season, competition;

SELECT
    league_key,
    season,
    competition,
    count(*) AS team_stat_rows,
    count(DISTINCT team_id) AS distinct_team_ids,
    count(*) FILTER (
        WHERE team_id IS NULL OR global_id IS NULL OR btrim(global_id) = ''
    ) AS rows_without_stable_team_identity,
    count(*) FILTER (WHERE coalesce(games_played, 0) = 0) AS rows_without_games
FROM "AI_Source"."TeamCompetitionStats"
WHERE season BETWEEN 2018 AND 2026
  AND league_key IN ('GRC1', 'LIT1')
GROUP BY league_key, season, competition
ORDER BY league_key, season, competition;

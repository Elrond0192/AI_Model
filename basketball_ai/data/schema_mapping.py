from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class ColumnDef:
    db_col: Optional[str]
    logical_col: str
    description: str
    default: Any = None
    compute: Optional[Callable[[pd.DataFrame], pd.Series]] = field(default=None, repr=False)


TABLE_LEAGUES: List[ColumnDef] = [
    ColumnDef("Id", "id", "League identifier from Configuration.Championship_ids."),
    ColumnDef("Nome", "name", "League display name from Configuration.Championship_ids."),
    ColumnDef("Nazione", "country", "League country code from Configuration.Championship_ids."),
    ColumnDef(None, "tier", "Default competitive tier.", default=1),
    ColumnDef(None, "competitiveness_score", "Default competitiveness score.", default=1.0),
    ColumnDef(None, "avg_pace", "Default league pace.", default=75.0),
    ColumnDef(None, "avg_offensive_rating", "Default league offensive rating.", default=110.0),
]


TABLE_TEAMS: List[ColumnDef] = [
    ColumnDef("Id", "id", "Team identifier from Anagrafiche.Team_* tables."),
    ColumnDef("TeamName", "name", "Full team name."),
    ColumnDef("ShortName", "short_name", "Short team name.", default=""),
    ColumnDef("Competition", "league_id", "Competition code from team stats.", default="RS"),
    ColumnDef(None, "playing_style", "Derived from team stats (pace/3PA/assists/DRtg).", default=None),
    ColumnDef(None, "formation", "Default formation.", default=""),
    ColumnDef("Pace", "pace", "Team pace from Analisi.AdvancedStatsTeam_*.", default=75.0),
    ColumnDef("ORtg", "offensive_rating", "Team offensive rating.", default=110.0),
    ColumnDef("DRtg", "defensive_rating", "Team defensive rating.", default=110.0),
    ColumnDef(
        None,
        "three_point_attempt_rate",
        "Derived 3PA rate from team field-goal attempts.",
        default=0.0,
        compute=lambda df: _compute_three_point_attempt_rate(df),
    ),
    ColumnDef(
        None,
        "assists_per_game",
        "Derived assists per 40-minute team game.",
        default=0.0,
        compute=lambda df: _compute_team_assists_per_game(df),
    ),
    ColumnDef(None, "star_player_usage", "Max usg_pct/100 of team's top player.", default=None),
    ColumnDef(None, "league_tier", "Default league tier.", default=1),
    ColumnDef("NetRtg", "net_rtg", "Team net rating.", default=0.0),
]


TABLE_PLAYERS: List[ColumnDef] = [
    ColumnDef("Id", "id", "Player identifier from Anagrafiche.*."),
    ColumnDef("PlayerName", "name", "Player display name."),
    ColumnDef(
        "Age",
        "age",
        "Age from advanced stats, with BirthDate fallback for direct table reads.",
        default=0,
        compute=lambda df: _compute_player_age(df),
    ),
    ColumnDef("Pos", "position", "Player position.", default=""),
    ColumnDef("Nat", "nationality", "Player nationality.", default=""),
    ColumnDef("Cm", "height_cm", "Player height in centimeters.", default=0),
    ColumnDef("Weight", "weight_kg", "Player weight in kilograms.", default=0),
    ColumnDef(None, "dominant_hand", "Default dominant hand.", default="R"),
    ColumnDef(None, "current_team_id", "Current team id placeholder.", default=None),
    ColumnDef(None, "current_league_id", "Current league id placeholder.", default=None),
    ColumnDef(None, "draft_year", "Draft year placeholder.", default=None),
    ColumnDef(None, "draft_pick", "Draft pick placeholder.", default=None),
]


TABLE_PLAYER_STATS: List[ColumnDef] = [
    ColumnDef("Id", "player_id", "Player identifier from season advanced stats."),
    ColumnDef(None, "season", "Static season label.", default="2024"),
    ColumnDef("TeamId", "team_id", "Most recent team identifier from Boxscore.*.", default=None),
    ColumnDef("Competition", "competition", "Competition code from season stats.", default="RS"),
    ColumnDef("Games", "games_played", "Games played.", default=0),
    ColumnDef("Min", "minutes_per_game", "Minutes per game.", default=0.0, compute=lambda df: _per_game(df, "Min")),
    ColumnDef("Pts", "points", "Points per game.", default=0.0, compute=lambda df: _per_game(df, "Pts")),
    ColumnDef("Tr", "rebounds", "Rebounds per game.", default=0.0, compute=lambda df: _per_game(df, "Tr")),
    ColumnDef("Or", "offensive_rebounds", "Offensive rebounds per game.", default=0.0, compute=lambda df: _per_game(df, "Or")),
    ColumnDef("Dr", "defensive_rebounds", "Defensive rebounds per game.", default=0.0, compute=lambda df: _per_game(df, "Dr")),
    ColumnDef("Ast", "assists", "Assists per game.", default=0.0, compute=lambda df: _per_game(df, "Ast")),
    ColumnDef("Stl", "steals", "Steals per game.", default=0.0, compute=lambda df: _per_game(df, "Stl")),
    ColumnDef("Blk", "blocks", "Blocks per game.", default=0.0, compute=lambda df: _per_game(df, "Blk")),
    ColumnDef("To", "turnovers", "Turnovers per game.", default=0.0, compute=lambda df: _per_game(df, "To")),
    ColumnDef("Pf", "personal_fouls", "Personal fouls per game.", default=0.0, compute=lambda df: _per_game(df, "Pf")),
    ColumnDef("EfgPct", "fg_pct", "Effective field-goal percentage.", default=0.0),
    ColumnDef("Fg3Pct", "three_point_pct", "Three-point percentage.", default=0.0),
    ColumnDef("FtPct", "ft_pct", "Free-throw percentage.", default=0.0),
    ColumnDef(None, "plus_minus", "Season plus/minus is not available in source stats.", default=0.0),
    ColumnDef("Pie", "per", "PIE mapped to logical PER.", default=0.0),
    ColumnDef("TsPct", "ts_pct", "True shooting percentage.", default=0.0),
    ColumnDef("UsgPct", "usg_pct", "Usage percentage.", default=0.0),
    ColumnDef("Bpm", "bpm", "Box plus/minus.", default=0.0),
    ColumnDef("Vorp", "vorp", "Value over replacement player.", default=0.0),
    ColumnDef("Ws", "win_shares", "Win shares.", default=0.0),
    ColumnDef("AstRatio", "ast_ratio", "Assist ratio.", default=0.0),
    ColumnDef("RebPct", "reb_pct", "Rebound percentage.", default=0.0),
    ColumnDef("ValLegaPerGame", "rating", "League rating per game.", default=0.0),
    ColumnDef("ORtg", "ortg", "Offensive rating.", default=0.0),
    ColumnDef("DRtg", "drtg", "Defensive rating.", default=0.0),
    ColumnDef("NetRtg", "net_rtg", "Net rating.", default=0.0),
    ColumnDef("RuoloOffensivo", "ruolo_offensivo", "Offensive role label.", default=""),
    ColumnDef("RuoloDifensivo", "ruolo_difensivo", "Defensive role label.", default=""),
    ColumnDef("RuoloCombinato", "ruolo_combinato", "Combined role label.", default=""),
    ColumnDef("NetRtg_On", "on_net_rtg", "On-court net rating.", default=0.0),
    ColumnDef("NetRtg_Off", "off_net_rtg", "Off-court net rating.", default=0.0),
    ColumnDef("NetRtg_Diff", "net_rtg_diff", "On/off net rating differential.", default=0.0),
    # --- Advanced metrics added from DB schema (schema_db.sql) ----------------
    ColumnDef("Spm", "spm", "Statistical Plus/Minus.", default=0.0),
    ColumnDef("OBpm", "obpm", "Offensive Box Plus/Minus.", default=0.0),
    ColumnDef("DBpm", "dbpm", "Defensive Box Plus/Minus.", default=0.0),
    ColumnDef("GmSc", "gm_sc", "Game Score (Hollinger).", default=0.0),
    ColumnDef("Fic", "fic", "Floor Impact Counter.", default=0.0),
    ColumnDef("OWS", "ows", "Offensive Win Shares.", default=0.0),
    ColumnDef("DWS", "dws", "Defensive Win Shares.", default=0.0),
    ColumnDef("RaptorOff", "raptor_off", "RAPTOR offensive rating.", default=0.0),
    ColumnDef("RaptorDef", "raptor_def", "RAPTOR defensive rating.", default=0.0),
    ColumnDef("RaptorTotal", "raptor_total", "RAPTOR total rating.", default=0.0),
    ColumnDef("LebronOff", "lebron_off", "LEBRON offensive rating.", default=0.0),
    ColumnDef("LebronDef", "lebron_def", "LEBRON defensive rating.", default=0.0),
    ColumnDef("LebronTotal", "lebron_total", "LEBRON total rating.", default=0.0),
    ColumnDef("ScoringEfficiency", "scoring_efficiency", "Scoring efficiency index.", default=0.0),
    ColumnDef("Ppsa", "ppsa", "Points per shot attempt.", default=0.0),
    ColumnDef("Fg2Pct", "two_point_pct", "Two-point field goal %.", default=0.0),
    ColumnDef("TovPct", "tov_pct", "Turnover %.", default=0.0),
    ColumnDef("AstPct", "ast_pct", "Assist %.", default=0.0),
    ColumnDef("StlPct", "stl_pct", "Steal %.", default=0.0),
    ColumnDef("BlkPct", "blk_pct", "Block %.", default=0.0),
    ColumnDef("OrebPct", "orb_pct", "Offensive rebound %.", default=0.0),
    ColumnDef("DrebPct", "drb_pct", "Defensive rebound %.", default=0.0),
    ColumnDef("ThreePAr", "three_par", "Three-point attempt rate.", default=0.0),
    ColumnDef("TusgPct", "true_usg_pct", "True usage percentage.", default=0.0),
    ColumnDef("FoulDrawingRate", "foul_drawing_rate", "Foul drawing rate.", default=0.0),
    ColumnDef("RfPerGame", "rf_per_game", "Referee fouls drawn per game.", default=0.0),
    ColumnDef("HustleIndex", "hustle_index", "Hustle index.", default=0.0),
    ColumnDef("PtsPer40", "pts_per_40", "Points per 40 minutes.", default=0.0),
    ColumnDef("AstPer40", "ast_per_40", "Assists per 40 minutes.", default=0.0),
    ColumnDef("TrPer40", "tr_per_40", "Rebounds per 40 minutes.", default=0.0),
    ColumnDef("StlPer40", "stl_per_40", "Steals per 40 minutes.", default=0.0),
    ColumnDef("BlkPer40", "blk_per_40", "Blocks per 40 minutes.", default=0.0),
    # --- Clutch stats from Analisi.AdvancedStats_Clutch_* (joined at query time) -
    ColumnDef("ClutchGames", "clutch_games", "Games with clutch situations.", default=0),
    ColumnDef("ClutchPts", "clutch_pts", "Points per game in clutch situations.", default=0.0),
    ColumnDef("ClutchTsPct", "clutch_ts_pct", "True shooting % in clutch.", default=0.0),
    ColumnDef("ClutchAstToTov", "clutch_ast_to_tov", "Ast/Tov ratio in clutch.", default=0.0),
    ColumnDef("ClutchNetRtg", "clutch_net_rtg", "Net rating in clutch situations.", default=0.0),
    ColumnDef("ClutchEfgPct", "clutch_efg_pct", "eFG% in clutch situations.", default=0.0),
    # --- On/Off offensive rating from Analisi.AdvancedStatsOnOffCourt_* --------
    ColumnDef("ORtg_On", "ortg_on", "Offensive rating when on court.", default=0.0),
    ColumnDef("ORtg_Off", "ortg_off", "Offensive rating when off court.", default=0.0),
    ColumnDef("ORtg_Diff", "ortg_diff", "ORtg on/off differential.", default=0.0),
]


TABLE_TEAM_PLAYER_RELATIONS: List[ColumnDef] = [
    ColumnDef("TeamId", "team_id", "Team identifier from Boxscore.*.", default=None),
    ColumnDef("Id", "player_id", "Player identifier from Boxscore.*."),
    ColumnDef(None, "season", "Static season label.", default="2024"),
    ColumnDef("Pos", "role", "Player role / position from Anagrafiche.*.", default=""),
    ColumnDef("ShirtNumber", "jersey_number", "Player jersey number.", default=0),
]


# ---------------------------------------------------------------------------
# Dynamic query generation — built from discovered table names at runtime
# ---------------------------------------------------------------------------

LEAGUES_DISCOVERY_QUERY: str = """
SELECT DISTINCT
    LEFT(TABLE_NAME, LEN(TABLE_NAME) - 5) AS id,
    LEFT(TABLE_NAME, LEN(TABLE_NAME) - 5) AS name,
    LEFT(
        LEFT(TABLE_NAME, LEN(TABLE_NAME) - 5),
        PATINDEX('%[0-9]%', LEFT(TABLE_NAME, LEN(TABLE_NAME) - 5) + '0') - 1
    ) AS country,
    RIGHT(TABLE_NAME, 4) AS season,
    CAST(1 AS int) AS tier,
    CAST(1.0 AS float) AS competitiveness_score,
    CAST(75.0 AS float) AS avg_pace,
    CAST(110.0 AS float) AS avg_offensive_rating
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'Anagrafiche'
  AND TABLE_NAME NOT LIKE 'Team[_]%'
  AND TABLE_NAME LIKE '%[_][0-9][0-9][0-9][0-9]'
"""


def _teams_block(league: str, season: str) -> str:
    tag = f"{league}_{season}"
    return f"""
SELECT
    CAST(t.Id AS nvarchar(100)) AS id,
    t.TeamName AS name,
    ISNULL(t.ShortName, t.TeamName) AS short_name,
    '{league}' AS league_id,
    CAST(NULL AS nvarchar(20)) AS playing_style,
    '' AS formation,
    ISNULL(CAST(s.Pace AS float), 75.0) AS pace,
    ISNULL(CAST(s.ORtg AS float), 110.0) AS offensive_rating,
    ISNULL(CAST(s.DRtg AS float), 110.0) AS defensive_rating,
    ISNULL(CAST(s.[3Fga] AS float) / NULLIF(CAST(s.[2Fga] + s.[3Fga] AS float), 0.0), 0.0) AS three_point_attempt_rate,
    ISNULL(CAST(s.Ast AS float) / NULLIF(CAST(s.Min AS float) / 40.0, 0.0), 0.0) AS assists_per_game,
    CAST(NULL AS float) AS star_player_usage,
    CAST(1 AS int) AS league_tier,
    ISNULL(CAST(s.NetRtg AS float), 0.0) AS net_rtg
FROM Anagrafiche.Team_{tag} AS t
LEFT JOIN (
    SELECT
        CAST(TeamId AS nvarchar(100)) AS team_id,
        [2Fga], [3Fga], Ast, Min, Pace, ORtg, DRtg, NetRtg,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(TeamId AS nvarchar(100))
            ORDER BY CASE WHEN Competition = 'RS' THEN 0 ELSE 1 END
        ) AS rn
    FROM Analisi.AdvancedStatsTeam_{tag}
) AS s ON CAST(t.Id AS nvarchar(100)) = s.team_id AND s.rn = 1"""


def _players_block(league: str, season: str, has_normalized_name: bool = False) -> str:
    tag = f"{league}_{season}"
    name_expr = (
        "ISNULL(NULLIF(p.NormalizedPlayerName, ''), p.PlayerName)"
        if has_normalized_name
        else "p.PlayerName"
    )
    return f"""
SELECT
    CAST(p.Id AS nvarchar(100)) AS id,
    {name_expr} AS name,
    ISNULL(CAST(adv.Age AS int), 0) AS age,
    ISNULL(adv.Position, p.Pos) AS position,
    ISNULL(p.Nat, '') AS nationality,
    ISNULL(TRY_CAST(p.Cm AS int), 0) AS height_cm,
    ISNULL(TRY_CAST(p.Weight AS int), 0) AS weight_kg,
    'R' AS dominant_hand,
    CAST(NULL AS nvarchar(100)) AS current_team_id,
    CAST(NULL AS nvarchar(100)) AS current_league_id,
    CAST(NULL AS int) AS draft_year,
    CAST(NULL AS int) AS draft_pick
FROM Anagrafiche.{tag} AS p
LEFT JOIN (
    SELECT
        CAST(Id AS nvarchar(100)) AS player_id,
        Age, Position,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(Id AS nvarchar(100))
            ORDER BY CASE WHEN Competition = 'RS' THEN 0 ELSE 1 END, ISNULL(Games, 0) DESC
        ) AS rn
    FROM Analisi.AdvancedStats_Player_{tag}
) AS adv ON CAST(p.Id AS nvarchar(100)) = adv.player_id AND adv.rn = 1"""


def _player_stats_cte(league: str, season: str, has_clutch: bool = True) -> str:
    """Returns CTE definitions (no WITH keyword) for one league/season.

    Parameters
    ----------
    has_clutch:
        When ``True`` (default) the clutch CTE reads from the real
        ``Analisi.AdvancedStats_Clutch_{tag}`` table.  When ``False`` the
        table is absent and an empty stub CTE is emitted instead so that the
        LEFT JOIN in the SELECT still compiles and simply produces NULL
        values (which the ISNULL() calls in the SELECT normalise to zeros).
    """
    tag = f"{league}_{season}"
    slug = tag.lower()

    if has_clutch:
        clutch_cte = f"""{slug}_clutch AS (
    SELECT
        CAST(c.Id AS nvarchar(100)) AS player_id,
        ISNULL(CAST(c.ClutchGames AS int), 0) AS ClutchGames,
        ISNULL(CAST(c.Pts AS float), 0.0) AS ClutchPts,
        ISNULL(CAST(c.TsPct AS float), 0.0) AS ClutchTsPct,
        ISNULL(CAST(c.AstToTovRatio AS float), 0.0) AS ClutchAstToTov,
        ISNULL(CAST(c.NetRtg AS float), 0.0) AS ClutchNetRtg,
        ISNULL(CAST(c.EfgPct AS float), 0.0) AS ClutchEfgPct,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(c.Id AS nvarchar(100))
            ORDER BY CASE WHEN c.Competition = 'RS' THEN 0 ELSE 1 END, ISNULL(c.ClutchGames, 0) DESC
        ) AS rn
    FROM Analisi.AdvancedStats_Clutch_{tag} AS c
)"""
    else:
        # Table does not exist — emit an empty stub so the LEFT JOIN compiles.
        clutch_cte = f"""{slug}_clutch AS (
    SELECT
        CAST(NULL AS nvarchar(100)) AS player_id,
        CAST(0 AS int)   AS ClutchGames,
        CAST(0.0 AS float) AS ClutchPts,
        CAST(0.0 AS float) AS ClutchTsPct,
        CAST(0.0 AS float) AS ClutchAstToTov,
        CAST(0.0 AS float) AS ClutchNetRtg,
        CAST(0.0 AS float) AS ClutchEfgPct,
        CAST(1 AS int)   AS rn
    WHERE 1 = 0
)"""

    return f"""{slug}_box AS (
    SELECT
        CAST(b.Id AS nvarchar(100)) AS player_id,
        CAST(b.TeamId AS nvarchar(100)) AS team_id,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(b.Id AS nvarchar(100))
            ORDER BY b.[Timestamp] DESC, b.Game DESC
        ) AS rn
    FROM Boxscore.{tag} AS b
),
{slug}_roles AS (
    SELECT
        CAST(r.Id AS nvarchar(100)) AS player_id,
        r.RuoloOffensivo, r.RuoloDifensivo, r.RuoloCombinato,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(r.Id AS nvarchar(100))
            ORDER BY CASE WHEN r.Competition = 'RS' THEN 0 ELSE 1 END, ISNULL(r.GamesPlayed, 0) DESC
        ) AS rn
    FROM Analisi.PlayerRoles_{tag} AS r
),
{slug}_onoff AS (
    SELECT
        CAST(o.Player AS nvarchar(100)) AS player_id,
        o.NetRtg_On, o.NetRtg_Off, o.NetRtg_Diff,
        o.ORtg_On, o.ORtg_Off, o.ORtg_Diff,
        ROW_NUMBER() OVER (
            PARTITION BY CAST(o.Player AS nvarchar(100))
            ORDER BY CASE WHEN o.Competition = 'RS' THEN 0 ELSE 1 END
        ) AS rn
    FROM Analisi.AdvancedStatsOnOffCourt_{tag} AS o
),
{clutch_cte}"""


def _player_stats_select(league: str, season: str) -> str:
    tag = f"{league}_{season}"
    slug = tag.lower()
    return f"""SELECT
    CAST(s.Id AS nvarchar(100)) AS player_id,
    '{season}' AS season,
    b.team_id AS team_id,
    '{league}' AS league_id,
    ISNULL(CAST(s.Games AS int), 0) AS games_played,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Min, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS minutes_per_game,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Pts, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS points,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Tr, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS rebounds,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.[Or], 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS offensive_rebounds,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Dr, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS defensive_rebounds,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Ast, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS assists,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Stl, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS steals,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Blk, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS blocks,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.[To], 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS turnovers,
    CASE WHEN ISNULL(s.Games, 0) > 0 THEN CAST(ISNULL(s.Pf, 0) AS float) / CAST(s.Games AS float) ELSE 0.0 END AS personal_fouls,
    ISNULL(CAST(s.EfgPct AS float), 0.0) AS fg_pct,
    ISNULL(CAST(s.Fg3Pct AS float), 0.0) AS three_point_pct,
    ISNULL(CAST(s.FtPct AS float), 0.0) AS ft_pct,
    CAST(0.0 AS float) AS plus_minus,
    ISNULL(CAST(s.Pie AS float), 0.0) AS per,
    ISNULL(CAST(s.TsPct AS float), 0.0) AS ts_pct,
    ISNULL(CAST(s.UsgPct AS float), 0.0) AS usg_pct,
    ISNULL(CAST(s.Bpm AS float), 0.0) AS bpm,
    ISNULL(CAST(s.Vorp AS float), 0.0) AS vorp,
    ISNULL(CAST(s.Ws AS float), 0.0) AS win_shares,
    ISNULL(CAST(s.AstRatio AS float), 0.0) AS ast_ratio,
    ISNULL(CAST(s.RebPct AS float), 0.0) AS reb_pct,
    ISNULL(CAST(s.ValLegaPerGame AS float), 0.0) AS rating,
    ISNULL(CAST(s.ORtg AS float), 0.0) AS ortg,
    ISNULL(CAST(s.DRtg AS float), 0.0) AS drtg,
    ISNULL(CAST(s.NetRtg AS float), 0.0) AS net_rtg,
    ISNULL(r.RuoloOffensivo, '') AS ruolo_offensivo,
    ISNULL(r.RuoloDifensivo, '') AS ruolo_difensivo,
    ISNULL(r.RuoloCombinato, '') AS ruolo_combinato,
    ISNULL(CAST(o.NetRtg_On AS float), 0.0) AS on_net_rtg,
    ISNULL(CAST(o.NetRtg_Off AS float), 0.0) AS off_net_rtg,
    ISNULL(CAST(o.NetRtg_Diff AS float), 0.0) AS net_rtg_diff,
    ISNULL(s.Competition, 'RS') AS competition,
    -- Advanced metrics (schema_db.sql: Analisi.AdvancedStats_Player_*)
    ISNULL(CAST(s.Spm AS float), 0.0) AS spm,
    ISNULL(CAST(s.OBpm AS float), 0.0) AS obpm,
    ISNULL(CAST(s.DBpm AS float), 0.0) AS dbpm,
    ISNULL(CAST(s.GmSc AS float), 0.0) AS gm_sc,
    ISNULL(CAST(s.Fic AS float), 0.0) AS fic,
    ISNULL(CAST(s.OWS AS float), 0.0) AS ows,
    ISNULL(CAST(s.DWS AS float), 0.0) AS dws,
    ISNULL(CAST(s.RaptorOff AS float), 0.0) AS raptor_off,
    ISNULL(CAST(s.RaptorDef AS float), 0.0) AS raptor_def,
    ISNULL(CAST(s.RaptorTotal AS float), 0.0) AS raptor_total,
    ISNULL(CAST(s.LebronOff AS float), 0.0) AS lebron_off,
    ISNULL(CAST(s.LebronDef AS float), 0.0) AS lebron_def,
    ISNULL(CAST(s.LebronTotal AS float), 0.0) AS lebron_total,
    ISNULL(CAST(s.ScoringEfficiency AS float), 0.0) AS scoring_efficiency,
    ISNULL(CAST(s.Ppsa AS float), 0.0) AS ppsa,
    ISNULL(CAST(s.Fg2Pct AS float), 0.0) AS two_point_pct,
    ISNULL(CAST(s.TovPct AS float), 0.0) AS tov_pct,
    ISNULL(CAST(s.AstPct AS float), 0.0) AS ast_pct,
    ISNULL(CAST(s.StlPct AS float), 0.0) AS stl_pct,
    ISNULL(CAST(s.BlkPct AS float), 0.0) AS blk_pct,
    ISNULL(CAST(s.OrebPct AS float), 0.0) AS orb_pct,
    ISNULL(CAST(s.DrebPct AS float), 0.0) AS drb_pct,
    ISNULL(CAST(s.ThreePAr AS float), 0.0) AS three_par,
    ISNULL(CAST(s.TusgPct AS float), 0.0) AS true_usg_pct,
    ISNULL(CAST(s.FoulDrawingRate AS float), 0.0) AS foul_drawing_rate,
    ISNULL(CAST(s.RfPerGame AS float), 0.0) AS rf_per_game,
    ISNULL(CAST(s.HustleIndex AS float), 0.0) AS hustle_index,
    ISNULL(CAST(s.PtsPer40 AS float), 0.0) AS pts_per_40,
    ISNULL(CAST(s.AstPer40 AS float), 0.0) AS ast_per_40,
    ISNULL(CAST(s.TrPer40 AS float), 0.0) AS tr_per_40,
    ISNULL(CAST(s.StlPer40 AS float), 0.0) AS stl_per_40,
    ISNULL(CAST(s.BlkPer40 AS float), 0.0) AS blk_per_40,
    -- Clutch stats (schema_db.sql: Analisi.AdvancedStats_Clutch_*)
    ISNULL(c.ClutchGames, 0) AS clutch_games,
    ISNULL(c.ClutchPts, 0.0) AS clutch_pts,
    ISNULL(c.ClutchTsPct, 0.0) AS clutch_ts_pct,
    ISNULL(c.ClutchAstToTov, 0.0) AS clutch_ast_to_tov,
    ISNULL(c.ClutchNetRtg, 0.0) AS clutch_net_rtg,
    ISNULL(c.ClutchEfgPct, 0.0) AS clutch_efg_pct,
    -- On/Off offensive rating (schema_db.sql: Analisi.AdvancedStatsOnOffCourt_*)
    ISNULL(CAST(o.ORtg_On AS float), 0.0) AS ortg_on,
    ISNULL(CAST(o.ORtg_Off AS float), 0.0) AS ortg_off,
    ISNULL(CAST(o.ORtg_Diff AS float), 0.0) AS ortg_diff
FROM Analisi.AdvancedStats_Player_{tag} AS s
LEFT JOIN {slug}_box    AS b ON CAST(s.Id AS nvarchar(100)) = b.player_id AND b.rn = 1
LEFT JOIN {slug}_roles  AS r ON CAST(s.Id AS nvarchar(100)) = r.player_id AND r.rn = 1
LEFT JOIN {slug}_onoff  AS o ON CAST(s.Id AS nvarchar(100)) = o.player_id AND o.rn = 1
LEFT JOIN {slug}_clutch AS c ON CAST(s.Id AS nvarchar(100)) = c.player_id AND c.rn = 1
WHERE s.Competition NOT IN ('TOT')
-- 'TOT' rows are aggregated totals across multiple teams in the same season;
-- we exclude them to avoid double-counting when a player changed teams mid-season."""


def _team_player_relations_block(league: str, season: str) -> str:
    tag = f"{league}_{season}"
    return f"""SELECT DISTINCT
    CAST(b.TeamId AS nvarchar(100)) AS team_id,
    CAST(b.Id AS nvarchar(100)) AS player_id,
    '{season}' AS season,
    ISNULL(p.Pos, '') AS role,
    ISNULL(TRY_CAST(p.ShirtNumber AS int), 0) AS jersey_number
FROM Boxscore.{tag} AS b
LEFT JOIN Anagrafiche.{tag} AS p
    ON CAST(b.Id AS nvarchar(100)) = CAST(p.Id AS nvarchar(100))"""


def get_table_queries(
    league_seasons: List[tuple],
    tags_with_normalized_name: Optional[set] = None,
    tags_with_clutch: Optional[set] = None,
) -> Dict[str, str]:
    """Build SQL queries for all discovered (league, season) pairs.

    Each logical table becomes a UNION ALL across all discovered leagues.
    The leagues query returns one row per unique (league, season) pair found
    by scanning Anagrafiche schema table names — no dependency on
    Configuration.Championship_ids.

    Parameters
    ----------
    league_seasons:
        List of ``(league_id, season)`` tuples discovered from the database.
    tags_with_normalized_name:
        Optional set of ``"<league>_<season>"`` tags whose Anagrafiche player
        table contains the ``NormalizedPlayerName`` column.  When a tag is in
        this set the generated SQL prefers ``NormalizedPlayerName`` over the
        raw ``PlayerName``.
    tags_with_clutch:
        Optional set of ``"<league>_<season>"`` tags for which the
        ``Analisi.AdvancedStats_Clutch_*`` table exists.  When a tag is
        **not** in this set an empty stub CTE is used so the query compiles
        even when the table is absent.  When ``None`` (default) the clutch
        table is assumed to exist for all tags (backwards-compatible).
    """
    if not league_seasons:
        return {}

    _norm_tags = tags_with_normalized_name or set()
    # None means "assume all tags have clutch" for backwards compatibility.
    _clutch_tags: Optional[set] = tags_with_clutch
    teams_sql = "\nUNION ALL\n".join(_teams_block(l, s) for l, s in league_seasons)
    players_sql = "\nUNION ALL\n".join(
        _players_block(l, s, has_normalized_name=(f"{l}_{s}" in _norm_tags))
        for l, s in league_seasons
    )

    cte_parts = ",\n".join(
        _player_stats_cte(
            l, s,
            has_clutch=(_clutch_tags is None or f"{l}_{s}" in _clutch_tags),
        )
        for l, s in league_seasons
    )
    stats_selects = "\nUNION ALL\n".join(_player_stats_select(l, s) for l, s in league_seasons)
    stats_sql = f"WITH\n{cte_parts}\n{stats_selects}"

    rels_sql = "\nUNION ALL\n".join(_team_player_relations_block(l, s) for l, s in league_seasons)

    return {
        "leagues": LEAGUES_DISCOVERY_QUERY,
        "teams": teams_sql,
        "players": players_sql,
        "player_stats": stats_sql,
        "team_player_relations": rels_sql,
    }


TABLE_QUERIES: Dict[str, str] = {}  # Replaced at runtime by get_table_queries()

ALL_TABLES: Dict[str, List[ColumnDef]] = {
    "leagues": TABLE_LEAGUES,
    "teams": TABLE_TEAMS,
    "players": TABLE_PLAYERS,
    "player_stats": TABLE_PLAYER_STATS,
    "team_player_relations": TABLE_TEAM_PLAYER_RELATIONS,
}


def get_rename_map(logical: str) -> Dict[str, str]:
    return {
        c.db_col: c.logical_col
        for c in ALL_TABLES.get(logical, [])
        if c.db_col is not None
    }


def get_defaults(logical: str) -> Dict[str, Any]:
    return {
        c.logical_col: c.default
        for c in ALL_TABLES.get(logical, [])
        if c.db_col is None or c.default is not None
    }


def apply_computes(df: pd.DataFrame, logical: str) -> pd.DataFrame:
    for col_def in ALL_TABLES.get(logical, []):
        if col_def.compute is None:
            continue
        try:
            series = col_def.compute(df)
        except Exception:
            continue
        if series is not None and len(series) == len(df):
            df[col_def.logical_col] = series
    return df


def _compute_age(birthdate_val: Any) -> Optional[int]:
    if pd.isna(birthdate_val) or not isinstance(birthdate_val, str):
        return None
    birthdate = pd.to_datetime(birthdate_val, dayfirst=True, errors="coerce")
    if pd.isna(birthdate):
        return None
    return int((pd.Timestamp.now() - birthdate).days / 365.25)


def _compute_player_age(df: pd.DataFrame) -> pd.Series:
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        if age.notna().any():
            return age.fillna(0)
    if "BirthDate" in df.columns:
        return df["BirthDate"].apply(_compute_age).fillna(0)
    return pd.Series([0] * len(df), index=df.index)


def _per_game(df: pd.DataFrame, raw_col: str) -> pd.Series:
    games_col = "games_played" if "games_played" in df.columns else "Games"
    if raw_col not in df.columns or games_col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    games = pd.to_numeric(df[games_col], errors="coerce").replace(0, pd.NA)
    values = pd.to_numeric(df[raw_col], errors="coerce")
    return (values / games).fillna(0.0)


def _compute_three_point_attempt_rate(df: pd.DataFrame) -> pd.Series:
    if "3Fga" not in df.columns or "2Fga" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    three_fga = pd.to_numeric(df["3Fga"], errors="coerce")
    total_fga = pd.to_numeric(df["2Fga"], errors="coerce") + three_fga
    return (three_fga / total_fga.replace(0, pd.NA)).fillna(0.0)


def _compute_team_assists_per_game(df: pd.DataFrame) -> pd.Series:
    if "Ast" not in df.columns or "Min" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    assists = pd.to_numeric(df["Ast"], errors="coerce")
    minutes = pd.to_numeric(df["Min"], errors="coerce")
    return (assists / ((minutes / 40.0).replace(0, pd.NA))).fillna(0.0)
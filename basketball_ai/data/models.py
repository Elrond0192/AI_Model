"""Basketball data models using Python dataclasses.

Positions support single and hybrid roles:
  Pure:   PG, SG, SF, PF, C
  Hybrid: PG/SG, SG/SF, SF/PF, PF/C, SG/PF (stretch), PG/SF (wing-guard)

Extended with all columns available in the SQL Server DB schema (schema_db.sql):
  - Advanced metrics: SPM, RAPTOR, LEBRON, OBpm/DBpm, GmSc, FIC, etc.
  - Clutch performance stats from Analisi.AdvancedStats_Clutch_*
  - On/Off court differential stats from Analisi.AdvancedStatsOnOffCourt_*
  - Player role labels from Analisi.PlayerRoles_*
  - Per-40 minute stats for cross-context comparisons
"""
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class League:
    id: int
    name: str
    country: str
    tier: int
    competitiveness_score: float
    avg_pace: float
    avg_offensive_rating: float

@dataclass
class Team:
    id: int
    name: str
    league_id: int
    playing_style: str
    formation: str
    pace: float
    offensive_rating: float
    defensive_rating: float
    three_point_attempt_rate: float
    assists_per_game: float
    star_player_usage: float
    league_tier: int
    short_name: str = field(default="")
    net_rtg: float = field(default=0.0)

@dataclass
class Player:
    id: int
    name: str
    age: int
    position: str
    nationality: str
    height_cm: int
    weight_kg: int
    dominant_hand: str
    current_team_id: Optional[int]
    current_league_id: Optional[int]
    draft_year: Optional[int]
    draft_pick: Optional[int]

@dataclass
class PlayerStats:
    player_id: int
    season: str
    team_id: int
    league_id: int
    games_played: int
    minutes_per_game: float
    points: float
    rebounds: float
    offensive_rebounds: float
    defensive_rebounds: float
    assists: float
    steals: float
    blocks: float
    turnovers: float
    personal_fouls: float
    fg_pct: float
    three_point_pct: float
    ft_pct: float
    plus_minus: float
    per: float
    ts_pct: float
    usg_pct: float
    bpm: float
    vorp: float
    win_shares: float
    ast_ratio: float
    reb_pct: float
    rating: float
    ortg: float = field(default=0.0)
    drtg: float = field(default=0.0)
    net_rtg: float = field(default=0.0)
    ruolo_offensivo: str = field(default="")
    ruolo_difensivo: str = field(default="")
    ruolo_combinato: str = field(default="")
    on_net_rtg: float = field(default=0.0)
    off_net_rtg: float = field(default=0.0)
    net_rtg_diff: float = field(default=0.0)
    competition: str = field(default="RS")  # RS | PO | CUP | SUPERCUP

    # --- Advanced metrics from AdvancedStats_Player_* -------------------------
    # Statistical / composite ratings
    spm: float = field(default=0.0)              # Statistical Plus/Minus
    obpm: float = field(default=0.0)             # Offensive Box Plus/Minus
    dbpm: float = field(default=0.0)             # Defensive Box Plus/Minus
    gm_sc: float = field(default=0.0)            # Game Score (Hollinger)
    fic: float = field(default=0.0)              # Floor Impact Counter
    ows: float = field(default=0.0)              # Offensive Win Shares
    dws: float = field(default=0.0)              # Defensive Win Shares

    # RAPTOR model (FiveThirtyEight)
    raptor_off: float = field(default=0.0)       # RAPTOR offensive
    raptor_def: float = field(default=0.0)       # RAPTOR defensive
    raptor_total: float = field(default=0.0)     # RAPTOR total

    # LEBRON model
    lebron_off: float = field(default=0.0)       # LEBRON offensive
    lebron_def: float = field(default=0.0)       # LEBRON defensive
    lebron_total: float = field(default=0.0)     # LEBRON total

    # Efficiency / scoring metrics
    scoring_efficiency: float = field(default=0.0)  # Scoring efficiency index
    ppsa: float = field(default=0.0)             # Points per shot attempt
    two_point_pct: float = field(default=0.0)    # Two-point field goal %
    tov_pct: float = field(default=0.0)          # Turnover %
    ast_pct: float = field(default=0.0)          # Assist %
    stl_pct: float = field(default=0.0)          # Steal %
    blk_pct: float = field(default=0.0)          # Block %
    orb_pct: float = field(default=0.0)          # Offensive rebound %
    drb_pct: float = field(default=0.0)          # Defensive rebound %
    three_par: float = field(default=0.0)        # Three-point attempt rate
    true_usg_pct: float = field(default=0.0)     # True usage percentage
    foul_drawing_rate: float = field(default=0.0)  # Foul drawing rate
    rf_per_game: float = field(default=0.0)      # Referee fouls drawn per game
    hustle_index: float = field(default=0.0)     # Hustle index

    # Per-40-minute stats (for cross-context comparisons)
    pts_per_40: float = field(default=0.0)
    ast_per_40: float = field(default=0.0)
    tr_per_40: float = field(default=0.0)
    stl_per_40: float = field(default=0.0)
    blk_per_40: float = field(default=0.0)

    # --- Clutch stats from Analisi.AdvancedStats_Clutch_* --------------------
    clutch_games: int = field(default=0)
    clutch_pts: float = field(default=0.0)
    clutch_ts_pct: float = field(default=0.0)
    clutch_ast_to_tov: float = field(default=0.0)
    clutch_net_rtg: float = field(default=0.0)
    clutch_efg_pct: float = field(default=0.0)

    # --- On/Off differential from Analisi.AdvancedStatsOnOffCourt_* ----------
    # (on_net_rtg, off_net_rtg, net_rtg_diff are already above)
    ortg_on: float = field(default=0.0)          # Offensive rating when player on court
    ortg_off: float = field(default=0.0)         # Offensive rating when player off court
    ortg_diff: float = field(default=0.0)        # ORtg on-court minus off-court delta

    # --- Starter status from Boxscore.SF ------------------------------------
    games_started: int = field(default=0)           # Games in starting five
    starter_pct: float = field(default=0.0)         # Fraction of games started (0–1)


@dataclass
class TeamPlayerRelation:
    team_id: int
    player_id: int
    season: str
    role: str
    jersey_number: int

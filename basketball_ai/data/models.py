"""Basketball data models using Python dataclasses.

Positions support single and hybrid roles:
  Pure:   PG, SG, SF, PF, C
  Hybrid: PG/SG, SG/SF, SF/PF, PF/C, SG/PF (stretch), PG/SF (wing-guard)
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

@dataclass
class TeamPlayerRelation:
    team_id: int
    player_id: int
    season: str
    role: str
    jersey_number: int

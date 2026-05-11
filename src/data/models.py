"""Data models (dataclasses) for the football performance prediction system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Player:
    """Represents a football player."""

    id: int
    name: str
    age: int
    position: str        # GK / CB / FB / CM / AM / W / ST
    nationality: str
    foot: str            # right / left / both
    height: float        # cm
    weight: float        # kg
    current_team_id: int
    current_league_id: int


@dataclass
class PlayerStats:
    """Season statistics for a single player."""

    player_id: int
    season: int
    team_id: int
    league_id: int
    goals: float
    assists: float
    matches_played: int
    minutes: float
    pass_accuracy: float        # 0–100 %
    dribbles: float             # per 90
    tackles: float              # per 90
    interceptions: float        # per 90
    aerial_duels_won: float     # per 90
    rating: float               # 0–10
    xG: float
    xA: float
    progressive_passes: float   # per 90
    key_passes: float           # per 90


@dataclass
class Team:
    """Represents a football club."""

    id: int
    name: str
    league_id: int
    playing_style: str      # possession / counter / high_press / direct
    formation: str
    avg_possession: float   # %
    pressing_intensity: float   # 0–10
    defensive_line: float       # 0–10
    passing_tempo: float        # 0–10
    league_tier: int


@dataclass
class League:
    """Represents a football league / competition."""

    id: int
    name: str
    country: str
    tier: int                       # 1–5
    competitiveness_score: float    # 0–1


@dataclass
class TeamPlayerRelation:
    """A player's role at a club in a specific season."""

    team_id: int
    player_id: int
    season: int
    role: str           # starter / rotation / bench
    jersey_number: int

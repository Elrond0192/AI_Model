"""Pydantic v2 schemas for all API request / response types."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class APIResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: str = "ok"


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    age: int
    position: str
    nationality: str
    foot: str
    height: float
    weight: float
    current_team_id: int
    current_league_id: int


class PlayerStatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    player_id: int
    season: int
    team_id: int
    league_id: int
    goals: float
    assists: float
    matches_played: int
    minutes: float
    pass_accuracy: float
    dribbles: float
    tackles: float
    interceptions: float
    aerial_duels_won: float
    rating: float
    xG: float
    xA: float
    progressive_passes: float
    key_passes: float


class PlayerProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    player: PlayerOut
    latest_stats: Optional[PlayerStatOut] = None
    form_score: float
    consistency_score: float
    peak_rating: float
    career_trajectory: float
    position_group: str


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    league_id: int
    playing_style: str
    formation: str
    avg_possession: float
    pressing_intensity: float
    defensive_line: float
    passing_tempo: float
    league_tier: int


class TeamAnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    team: TeamOut
    avg_squad_rating: float
    top_performers: List[PlayerOut]
    style_strengths: List[str]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    player_id: int
    team_id: int
    season: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    base_rating: float
    age_factor: float
    compatibility_factor: float
    league_factor: float
    context_adjustment: float
    shap_values: Dict[str, float] = Field(default_factory=dict)
    explanation: str


class TrajectoryPointOut(BaseModel):
    age: int
    season: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    age_factor: float


class PeakPredictionOut(BaseModel):
    player_id: int
    player_name: str
    current_age: int
    peak_age: int
    current_rating: float
    peak_rating: float
    seasons_to_peak: int
    peak_window_start: int
    peak_window_end: int


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class WhatIfRequest(BaseModel):
    player_id: int
    team_id: int
    season: int = 2024


class CompareRequest(BaseModel):
    player_id: int
    team_ids: List[int] = Field(..., min_length=2)
    season: int = 2024


class TransferImpactRequest(BaseModel):
    player_id: int
    from_team_id: int
    to_team_id: int
    season: int = 2024


class WhatIfTeammatesRequest(BaseModel):
    player_id: int
    team_id: int
    hypothetical_avg_rating: float = Field(..., ge=4.0, le=10.0)
    season: int = 2024


class ScenarioOut(BaseModel):
    team_id: int
    team_name: str
    league_name: str
    rating: float
    confidence_low: float
    confidence_high: float
    explanation: str


class CompareOut(BaseModel):
    player_id: int
    scenarios: List[ScenarioOut]
    best_scenario: ScenarioOut


class TeamFitOut(BaseModel):
    team_id: int
    team_name: str
    league_name: str
    predicted_rating: float
    compatibility_score: float
    position_fit: float
    rank: int


class PlayerFitOut(BaseModel):
    player_id: int
    player_name: str
    position: str
    predicted_rating: float
    current_team: str
    rank: int


class TransferImpactOut(BaseModel):
    player_id: int
    from_team_id: int
    to_team_id: int
    rating_before: float
    rating_after: float
    rating_delta: float
    adaptation_factor: float
    recommendation: str

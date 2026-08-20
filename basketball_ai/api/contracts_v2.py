"""Stable prediction contract consumed by external server-side clients."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from basketball_ai.models.competition_training import normalize_competition


class PredictionContextV2(BaseModel):
    model_run_id: str
    model_version: str
    feature_version: str
    data_cutoff: date


class CompetitionSupportV2(BaseModel):
    mode: Literal["isolated"] = "isolated"
    competition: str
    source_seasons: int = Field(ge=0)
    source_games: int = Field(ge=0)
    exact_source_games: int = Field(ge=0)
    calibration_scope: Literal["competition", "global"]
    calibration_samples: int = Field(ge=0)


class PlayerTeamPredictionRequestV2(BaseModel):
    player_global_id: str = Field(min_length=1, max_length=128)
    team_global_id: str = Field(min_length=1, max_length=128)
    league: str = Field(min_length=1, max_length=32)
    season: int = Field(ge=2000, le=2100)
    competition: str = Field(default="RS", min_length=1, max_length=32)

    @field_validator("competition")
    @classmethod
    def normalise_competition(cls, value: str) -> str:
        competition = normalize_competition(value)
        if not competition:
            raise ValueError("competition is required")
        return competition

    @field_validator("league")
    @classmethod
    def normalise_league(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("league is required")
        return value


class PlayerTeamPredictionV2(PredictionContextV2):
    intent: Literal["forecast_t_plus_1", "team_fit"] = "forecast_t_plus_1"
    player_global_id: str
    team_global_id: str
    league: str
    season: int
    competition: str
    target_season: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    competition_support: CompetitionSupportV2
    generated_at: datetime
    explanation: dict[str, Any] = Field(default_factory=dict)

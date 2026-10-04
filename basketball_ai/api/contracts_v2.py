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


class CompatibilityPlayerProfileV2(BaseModel):
    position: str
    usg_pct: float
    ts_pct: float
    points: float
    three_par: float
    dbpm: float


class CompatibilityTeamProfileV2(BaseModel):
    team_global_id: str | None = None
    team_name: str
    season: int
    competition: str
    pace: float
    three_point_attempt_rate: float
    assists_per_game: float
    star_player_usage: float
    offensive_rating: float
    defensive_rating: float


class CompatibilityComparisonV2(BaseModel):
    selected_team_score: float = Field(ge=0.0, le=1.0)
    real_team_score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_delta_vs_real_team: float | None = None
    score_delta_vs_neutral: float
    neutral_score: float = 0.5
    selected_team: CompatibilityTeamProfileV2
    real_team: CompatibilityTeamProfileV2 | None = None
    player_profile: CompatibilityPlayerProfileV2
    methodology: Literal["competition_temporal_knn"] = "competition_temporal_knn"


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
    predicted_rating_100: float | None = None
    confidence_low_100: float | None = None
    confidence_high_100: float | None = None
    prediction_calibration_version: str | None = None
    competition_support: CompetitionSupportV2
    compatibility: CompatibilityComparisonV2
    generated_at: datetime
    explanation: dict[str, Any] = Field(default_factory=dict)

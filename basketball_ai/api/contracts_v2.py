"""Stable contract consumed by WordPress Chat V3."""
from __future__ import annotations
from datetime import date, datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
class PredictionContextV2(BaseModel):
    model_run_id: str; model_version: str; feature_version: str; data_cutoff: date
class PlayerTeamPredictionRequestV2(BaseModel):
    player_global_id: str = Field(min_length=1, max_length=128)
    team_global_id: str = Field(min_length=1, max_length=128)
    league: str = Field(min_length=2, max_length=32)
    season: int = Field(ge=2000, le=2100)
    competition: str = Field(default="RS", min_length=1, max_length=16)
class PlayerTeamPredictionV2(PredictionContextV2):
    intent: Literal["forecast_t_plus_1", "team_fit"] = "forecast_t_plus_1"
    player_global_id: str; team_global_id: str; league: str; season: int; competition: str; target_season: int
    predicted_rating: float; confidence_low: float; confidence_high: float; generated_at: datetime
    explanation: dict[str, Any] = Field(default_factory=dict)

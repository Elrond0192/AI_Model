"""API contracts for the Player Future Performance Model."""
from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class FuturePerformancePlayerRequestV2(BaseModel):
    player_global_id: str
    league: str
    season: Union[int, str]
    competition: str = "RS"


class FuturePerformancePlayerResponseV2(BaseModel):
    future_performance_version: str
    feature_version: str
    player_global_id: str | None
    player_name: str | None
    league: str
    source_season: int
    target_season: int
    competition: str
    targets: dict[str, dict]
    model_status: str = Field(default="production")

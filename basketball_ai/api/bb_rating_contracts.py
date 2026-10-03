"""API contracts for the contextual BB-Rating engine."""
from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class BBRatingPlayerRequestV2(BaseModel):
    player_global_id: str
    league: str
    season: Union[int, str]
    phase: str
    include_history: bool = False
    history_limit: int = Field(default=8, ge=2, le=8)


class BBRatingPlayerResponseV2(BaseModel):
    player_global_id: str
    player_name: str | None
    league: str
    season: int
    phase: str
    bb_rating: int = Field(ge=1, le=100)
    score_band: str
    dimensions: dict[str, int]
    metrics: dict[str, dict]
    strengths: list[str]
    limitations: list[str]
    explanation: str
    peer_group: dict
    quality: str
    metric_coverage: float = Field(ge=0.0, le=1.0)
    bb_rating_version: str
    uncertainty: dict | None = None
    history: list[dict] = Field(default_factory=list)

"""Direct future BB-Rating prediction API."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from basketball_ai.models.competition_training import normalize_competition

router = APIRouter(prefix="/api/v2/bb-rating-prediction", tags=["bb-rating-prediction"])


class BBRatingFuturePredictionRequest(BaseModel):
    player_global_id: str = Field(min_length=1, max_length=128)
    source_season: int = Field(ge=2000, le=2100)
    target_league: str = Field(min_length=1, max_length=32)
    target_competition: str = Field(default="RS", min_length=1, max_length=32)
    source_league: str | None = Field(default=None, max_length=32)
    source_competition: str = Field(default="RS", min_length=1, max_length=32)
    target_team_global_id: str | None = Field(default=None, max_length=128)

    @field_validator("target_league")
    @classmethod
    def normalize_league(cls, value: str) -> str:
        result = value.strip().upper()
        if not result:
            raise ValueError("target_league is required")
        return result

    @field_validator("source_league")
    @classmethod
    def normalize_source_league(cls, value: str | None) -> str | None:
        return value.strip().upper() if value and value.strip() else None

    @field_validator("source_competition", "target_competition")
    @classmethod
    def normalize_competition(cls, value: str) -> str:
        result = normalize_competition(value)
        if not result:
            raise ValueError("target_competition is required")
        return result


@router.post("/player")
def predict_future_bb_rating(
    body: BBRatingFuturePredictionRequest, request: Request
) -> dict[str, Any]:
    model = getattr(request.app.state, "bb_rating_future_prediction_model", None)
    data = getattr(request.app.state, "data", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Direct BB-Rating prediction model is not installed as a validated release",
        )
    if not data:
        raise HTTPException(status_code=503, detail="Canonical player data is unavailable")
    try:
        result = model.predict_from_data(
            data,
            player_global_id=body.player_global_id,
            source_season=body.source_season,
            target_league=body.target_league,
            target_competition=body.target_competition,
            target_team_global_id=body.target_team_global_id,
            source_league=body.source_league,
            source_competition=body.source_competition,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "player_global_id": body.player_global_id,
        "source_season": body.source_season,
        "target_season": body.source_season + 1,
        "source_league": result.get("source_league"),
        "source_competition": result.get("source_competition"),
        "target_team_global_id": body.target_team_global_id,
        "method": "direct_bb_rating_t_plus_1",
        "prediction": result,
        "explanation": (
            "Previsione diretta del BB-Rating ufficiale nella scala 1-100. "
            "L'intervallo è stimato dagli errori out-of-fold storici; "
            "la calibrazione indica il livello di supporto del contesto."
        ),
    }

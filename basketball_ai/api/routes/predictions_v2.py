"""Version 2 typed prediction endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.contracts_v2 import (
    PlayerTeamPredictionRequestV2,
    PlayerTeamPredictionV2,
)
from basketball_ai.models.strict_production import build_historical_snapshot
from basketball_ai.scenarios.engine import WhatIfEngine

router = APIRouter(prefix="/api/v2/predictions", tags=["predictions-v2"])


@router.post("/player-team", response_model=PlayerTeamPredictionV2)
async def player_team(body: PlayerTeamPredictionRequestV2, request: Request):
    engine, data = request.app.state.engine, request.app.state.data
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    player = next(
        (
            value
            for value in data.get("player_dict", {}).values()
            if str(value.get("global_id", "")) == body.player_global_id
        ),
        None,
    )
    team = next(
        (
            value
            for value in data.get("team_dict", {}).values()
            if str(value.get("global_id", "")) == body.team_global_id
        ),
        None,
    )
    if not player or not team:
        raise HTTPException(
            404,
            "Resolved player or team is unavailable for the requested context",
        )

    try:
        snapshot = build_historical_snapshot(data, int(body.season))
    except Exception as exc:
        raise HTTPException(422, f"Historical context is unavailable: {exc}") from exc

    player_id = int(player["id"])
    team_id = int(team["id"])
    if player_id not in snapshot.get("player_dict", {}) or team_id not in snapshot.get("team_dict", {}):
        raise HTTPException(
            404,
            "Resolved player or team has no state at the requested source season",
        )

    historical_engine = WhatIfEngine(engine.ensemble, snapshot)
    try:
        result = historical_engine.predict_in_team(
            player_id,
            team_id,
            int(body.season),
            competition=body.competition,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc

    return PlayerTeamPredictionV2(
        **request.app.state.model_metadata,
        player_global_id=body.player_global_id,
        team_global_id=body.team_global_id,
        league=body.league,
        season=body.season,
        competition=body.competition,
        target_season=body.season + 1,
        predicted_rating=result.predicted_rating,
        confidence_low=result.confidence_low,
        confidence_high=result.confidence_high,
        generated_at=datetime.now(timezone.utc),
        explanation={
            "method": "forecast_t_plus_1",
            "context": "historical_as_of_source_season",
        },
    )

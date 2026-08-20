"""Version 2 typed prediction endpoints."""
from __future__ import annotations
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from basketball_ai.api.contracts_v2 import PlayerTeamPredictionRequestV2, PlayerTeamPredictionV2
router = APIRouter(prefix="/api/v2/predictions", tags=["predictions-v2"])
@router.post("/player-team", response_model=PlayerTeamPredictionV2)
async def player_team(body: PlayerTeamPredictionRequestV2, request: Request):
    engine, data = request.app.state.engine, request.app.state.data
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    player = next((p for p in data.get("player_dict", {}).values() if str(p.get("global_id", "")) == body.player_global_id), None)
    team = next((t for t in data.get("team_dict", {}).values() if str(t.get("global_id", "")) == body.team_global_id), None)
    if not player or not team:
        raise HTTPException(404, "Resolved player or team is unavailable for the requested context")
    result = engine.predict_in_team(int(player["id"]), int(team["id"]), body.season)
    return PlayerTeamPredictionV2(**request.app.state.model_metadata, player_global_id=body.player_global_id, team_global_id=body.team_global_id, league=body.league, season=body.season, competition=body.competition, target_season=body.season + 1, predicted_rating=result.predicted_rating, confidence_low=result.confidence_low, confidence_high=result.confidence_high, generated_at=datetime.now(timezone.utc), explanation={"method":"forecast_t_plus_1"})

"""Player Future Performance v2 API route.

The route exposes the independent player-centric season-ahead performance
forecast. It does not alter Prediction Model or BB-Rating outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.future_performance_contracts import (
    FuturePerformancePlayerRequestV2,
    FuturePerformancePlayerResponseV2,
)
from basketball_ai.models.competition_training import normalize_competition, resolve_league_id

router = APIRouter(
    prefix="/api/v2/future-performance",
    tags=["future-performance-v2"],
)


@router.post("/player", response_model=FuturePerformancePlayerResponseV2)
async def future_player(
    body: FuturePerformancePlayerRequestV2,
    request: Request,
):
    model = getattr(request.app.state, "future_performance_model", None)
    data = getattr(request.app.state, "data", {})
    if model is None:
        raise HTTPException(503, "Player Future Performance Model is not loaded")

    source_season = int(str(body.season).split("-")[0])
    competition = normalize_competition(body.competition)

    try:
        league_id = resolve_league_id(data, body.league)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    canonical_league = str(body.league).strip().upper()
    leagues = data.get("leagues")
    if leagues is not None and not getattr(leagues, "empty", True):
        for row in leagues.to_dict("records"):
            try:
                if int(row.get("id")) == int(league_id):
                    canonical_league = str(
                        row.get("league_key") or row.get("name") or canonical_league
                    ).strip().upper()
                    break
            except (TypeError, ValueError):
                continue

    player = next(
        (
            value
            for value in data.get("player_dict", {}).values()
            if str(value.get("global_id", "")) == body.player_global_id
        ),
        None,
    )
    if player is None:
        raise HTTPException(404, "Player is unavailable in the active data contract")

    try:
        result = model.predict_player(
            data,
            player_id=int(player["id"]),
            league_key=canonical_league,
            season=source_season,
            competition=competition,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    # resolve_league_id validates the requested league. The numeric ID remains
    # internal to the AI service and is not emitted by this public contract.
    result["league"] = canonical_league
    result["generated_at"] = datetime.now(timezone.utc)
    return FuturePerformancePlayerResponseV2(**result)

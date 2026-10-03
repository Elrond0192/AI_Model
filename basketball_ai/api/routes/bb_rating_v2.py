"""BB-Rating v2 API route.

This endpoint exposes the descriptive/contextual rating layer. It does not
call or modify the season-ahead prediction model.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.bb_rating_contracts import (
    BBRatingPlayerRequestV2,
    BBRatingPlayerResponseV2,
)

router = APIRouter(prefix="/api/v2/bb-rating", tags=["bb-rating-v2"])


@router.post("/player", response_model=BBRatingPlayerResponseV2)
async def rate_player(body: BBRatingPlayerRequestV2, request: Request):
    engine = getattr(request.app.state, "bb_rating_engine", None)
    if engine is None:
        raise HTTPException(
            503,
            "BB-Rating engine is not loaded",
        )

    season = int(str(body.season).split("-")[0])
    try:
        result = engine.rate_player(
            body.player_global_id,
            league=body.league,
            season=season,
            phase=body.phase,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    payload = result.to_dict()

    if body.include_history:
        frame = getattr(engine, "frame", None)
        history = []
        if frame is not None and not getattr(frame, "empty", True):
            try:
                years = (
                    frame.loc[
                        (frame["player_global_id"].astype(str) == str(body.player_global_id))
                        & (frame["league_key"].astype(str).str.upper() == str(body.league).strip().upper())
                        & (frame["competition"].astype(str).str.upper() == str(body.phase).strip().upper())
                    ]["season"]
                    .map(lambda value: int(str(value).split("-")[0]))
                )
                seasons = sorted({int(year) for year in years if int(year) <= season}, reverse=True)[: body.history_limit]
                for historical_season in reversed(seasons):
                    try:
                        historical = engine.rate_player(
                            body.player_global_id,
                            league=body.league,
                            season=historical_season,
                            phase=body.phase,
                        )
                        history.append({
                            "season": historical_season,
                            "bb_rating": int(historical.score),
                        })
                    except ValueError:
                        continue
            except (KeyError, TypeError, ValueError):
                history = []
        payload["history"] = history

    uncertainty = getattr(request.app.state, "bb_rating_uncertainty", None)
    if uncertainty is None:
        payload["uncertainty"] = {
            "available": False,
            "reason": "uncertainty_artifact_unavailable",
            "calibration_version": None,
            "bb_rating_version": str(payload.get("bb_rating_version", "1.8")),
        }
    else:
        payload["uncertainty"] = uncertainty.for_player(
            engine.frame,
            player_global_id=body.player_global_id,
            league=body.league,
            season=season,
            phase=body.phase,
            score=result.score,
        )

    return BBRatingPlayerResponseV2(**payload)

"""Prediction API routes."""

from __future__ import annotations

import asyncio
import os as _os
import time as _ptime
from typing import Any as _CAny, Dict as _CDict, List, Optional, Tuple as _CTuple

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from basketball_ai.api.schemas import PredictionOut, TrajectoryPointOut, PeakPredictionOut
from basketball_ai.api.limiter import limiter, RATE_LIMIT_PREDICTIONS

router = APIRouter(prefix="/predictions", tags=["predictions"])

# ---------------------------------------------------------------------------
# In-process prediction cache (TTL-based dict)
# ---------------------------------------------------------------------------

_PRED_CACHE: _CDict[_CTuple, tuple] = {}
_PRED_CACHE_TTL = int(_os.environ.get("PREDICTION_CACHE_TTL_SECONDS", str(5 * 60)))  # 5 min
_PRED_CACHE_MAX  = int(_os.environ.get("PREDICTION_CACHE_SIZE", "1024"))


def _cache_get(key: _CTuple) -> _CAny:
    entry = _PRED_CACHE.get(key)
    if entry is None:
        return None
    value, exp = entry
    if _ptime.monotonic() > exp:
        del _PRED_CACHE[key]
        return None
    return value


def _cache_set(key: _CTuple, value: _CAny) -> None:
    if len(_PRED_CACHE) >= _PRED_CACHE_MAX:
        # Evict oldest 10%
        to_del = list(_PRED_CACHE.keys())[: max(1, _PRED_CACHE_MAX // 10)]
        for k in to_del:
            _PRED_CACHE.pop(k, None)
    _PRED_CACHE[key] = (value, _ptime.monotonic() + _PRED_CACHE_TTL)


def _get_engine():
    from basketball_ai.api.main import app_state
    return app_state["engine"]


def _get_data():
    from basketball_ai.api.main import app_state
    return app_state["data"]


@router.post("/player/{player_id}/team/{team_id}", response_model=PredictionOut)
@limiter.limit(RATE_LIMIT_PREDICTIONS)
async def predict_player_in_team(
    request: Request,
    player_id: int,
    team_id: int,
    season: int = Query(2024, ge=2015, le=2035),
):
    """Predict how a player would perform at a specific team."""
    engine = _get_engine()
    data = _get_data()

    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    if team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    cache_key = (player_id, team_id, season)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.predict_in_team, player_id, team_id, season)
    out = PredictionOut(
        player_id=result.player_id,
        team_id=result.team_id,
        season=result.season,
        predicted_rating=result.predicted_rating,
        confidence_low=result.confidence_low,
        confidence_high=result.confidence_high,
        base_rating=result.base_rating,
        age_factor=result.age_factor,
        compatibility_factor=result.compatibility_factor,
        league_factor=result.league_factor,
        context_adjustment=result.context_adjustment,
        shap_values=result.shap_values,
        explanation=result.explanation,
    )
    _cache_set(cache_key, out)
    return out


@router.get("/player/{player_id}/trajectory", response_model=List[TrajectoryPointOut])
async def get_trajectory(
    request: Request,
    player_id: int,
    team_id: Optional[int] = Query(None),
    age_from: int = Query(None),
    age_to: int = Query(None),
):
    """Get predicted rating trajectory across ages."""
    engine = _get_engine()
    data = _get_data()

    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    age_range = None
    if age_from is not None and age_to is not None:
        age_range = (age_from, age_to)

    loop = asyncio.get_event_loop()
    traj = await loop.run_in_executor(
        None, lambda: engine.predict_age_trajectory(player_id, age_range=age_range, team_id=team_id)
    )
    return [
        TrajectoryPointOut(
            age=p.age,
            season=p.season,
            predicted_rating=p.predicted_rating,
            confidence_low=p.confidence_low,
            confidence_high=p.confidence_high,
            age_factor=p.age_factor,
        )
        for p in traj
    ]


@router.get("/player/{player_id}/peak", response_model=PeakPredictionOut)
async def get_peak_prediction(
    request: Request,
    player_id: int,
    team_id: Optional[int] = Query(None),
):
    """Predict a player's career peak."""
    engine = _get_engine()
    data = _get_data()

    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    loop = asyncio.get_event_loop()
    peak = await loop.run_in_executor(None, lambda: engine.predict_peak(player_id, team_id=team_id))
    return PeakPredictionOut(
        player_id=peak.player_id,
        player_name=peak.player_name,
        current_age=peak.current_age,
        peak_age=peak.peak_age,
        current_rating=peak.current_rating,
        peak_rating=peak.peak_rating,
        seasons_to_peak=peak.seasons_to_peak,
        peak_window_start=peak.peak_window[0],
        peak_window_end=peak.peak_window[1],
    )


# ---------------------------------------------------------------------------
# Batch prediction endpoint
# ---------------------------------------------------------------------------

class BatchPredictionRequest(BaseModel):
    items: List[dict]  # each: {player_id, team_id, season?}


class BatchPredictionResponse(BaseModel):
    results: List[dict]
    errors:  List[dict]


@router.post("/batch", response_model=BatchPredictionResponse)
@limiter.limit(RATE_LIMIT_PREDICTIONS)
async def predict_batch(
    request: Request,
    body: BatchPredictionRequest,
):
    """Predict player performance for multiple player/team pairs in one call.

    Body: ``{"items": [{"player_id": 1, "team_id": 2, "season": 2024}, ...]}``
    Up to 50 items per request.
    """
    from basketball_ai.api._deps import get_data, get_engine

    if len(body.items) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 items per batch request")

    engine = get_engine(request)
    data   = get_data(request)

    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    errors  = []

    for item in body.items:
        pid  = item.get("player_id")
        tid  = item.get("team_id")
        seas = item.get("season", 2024)
        if pid is None or tid is None:
            errors.append({"item": item, "error": "player_id and team_id are required"})
            continue
        if pid not in data.get("player_dict", {}):
            errors.append({"item": item, "error": f"Player {pid} not found"})
            continue
        if tid not in data.get("team_dict", {}):
            errors.append({"item": item, "error": f"Team {tid} not found"})
            continue
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, engine.predict_in_team, pid, tid, seas)
            results.append({
                "player_id":        pid,
                "team_id":          tid,
                "season":           seas,
                "predicted_rating": result.predicted_rating,
                "confidence_low":   result.confidence_low,
                "confidence_high":  result.confidence_high,
            })
        except Exception as exc:
            errors.append({"item": item, "error": str(exc)})

    return BatchPredictionResponse(results=results, errors=errors)

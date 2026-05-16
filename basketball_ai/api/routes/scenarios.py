"""Scenario API routes."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from basketball_ai.api.schemas import (
    WhatIfRequest,
    CompareRequest,
    TransferImpactRequest,
    WhatIfTeammatesRequest,
    WhatIfLineupRequest,
    PredictionOut,
    CompareOut,
    ScenarioOut,
    TeamFitOut,
    PlayerFitOut,
    TransferImpactOut,
    LineupMemberProfileOut,
    LineupAnalysisOut,
)
from basketball_ai.api._deps import get_engine, get_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scenarios", tags=["scenarios"])

# AP9 – Idempotency cache: stores response keyed by Idempotency-Key header.
# In-process only; safe for single-worker deployments.
_IDEMPOTENCY_CACHE: Dict[str, Any] = {}
_IDEMPOTENCY_CACHE_MAX = 1000   # evict oldest entry when this limit is reached
_IDEMPOTENCY_LOCK = threading.Lock()


def _idempotency_key(request: Request) -> Optional[str]:
    """Return the Idempotency-Key header value, or None if absent."""
    return request.headers.get("Idempotency-Key") or request.headers.get("X-Idempotency-Key")


def _check_idempotency(key: str) -> Optional[Any]:
    """Return cached result for *key*, or None if not yet seen."""
    with _IDEMPOTENCY_LOCK:
        return _IDEMPOTENCY_CACHE.get(key)


def _store_idempotency(key: str, result: Any) -> None:
    """Cache *result* for *key*, evicting the oldest entry if the cache is full."""
    with _IDEMPOTENCY_LOCK:
        if len(_IDEMPOTENCY_CACHE) >= _IDEMPOTENCY_CACHE_MAX:
            oldest = next(iter(_IDEMPOTENCY_CACHE))
            del _IDEMPOTENCY_CACHE[oldest]
        _IDEMPOTENCY_CACHE[key] = result


@router.post("/what-if", response_model=PredictionOut)
async def what_if_scenario(request: Request, req: WhatIfRequest):
    """Predict performance for a specific player-team combination."""
    # AP9 – Idempotency-Key: replay cached response for duplicate requests.
    ikey = _idempotency_key(request)
    if ikey:
        cached = _check_idempotency(ikey)
        if cached is not None:
            logger.debug("[AP9] Replaying idempotent response for key %s", ikey)
            return cached

    engine = get_engine(request)
    data = get_data(request)
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    if req.team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.predict_in_team, req.player_id, req.team_id, req.season)
    out = PredictionOut(**result.__dict__)

    if ikey:
        _store_idempotency(ikey, out)
    return out


@router.get("/best-teams/{player_id}", response_model=List[TeamFitOut])
async def best_teams_for_player(
    request: Request,
    player_id: int,
    league_id: Optional[int] = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    """Find the best-fitting teams for a player."""
    engine = get_engine(request)
    data = get_data(request)
    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    loop = asyncio.get_event_loop()
    fits = await loop.run_in_executor(
        None, lambda: engine.best_team_fit(player_id, league_id=league_id, top_n=top_n)
    )
    return [
        TeamFitOut(
            team_id=f.team_id,
            team_name=f.team_name,
            league_name=f.league_name,
            predicted_rating=f.predicted_rating,
            compatibility_score=f.compatibility_score,
            position_fit=f.position_fit,
            rank=f.rank,
        )
        for f in fits
    ]


@router.get("/best-players/{team_id}", response_model=List[PlayerFitOut])
async def best_players_for_team(
    request: Request,
    team_id: int,
    position: Optional[str] = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    """Find the best-fitting players for a team (by position)."""
    engine = get_engine(request)
    data = get_data(request)
    if team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    loop    = asyncio.get_event_loop()
    players = await loop.run_in_executor(
        None, lambda: engine.best_player_for_team(team_id, position=position, top_n=top_n)
    )
    return [
        PlayerFitOut(
            player_id=p.player_id,
            player_name=p.player_name,
            position=p.position,
            predicted_rating=p.predicted_rating,
            current_team=p.current_team,
            rank=p.rank,
        )
        for p in players
    ]


@router.post("/compare", response_model=CompareOut)
async def compare_scenarios(request: Request, req: CompareRequest):
    """Compare player performance across multiple teams."""
    engine = get_engine(request)
    data = get_data(request)
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.compare_scenarios(req.player_id, req.team_ids, req.season)
    )
    scenarios_out = [
        ScenarioOut(
            team_id=s["team_id"],
            team_name=s["team_name"],
            league_name=s["league_name"],
            rating=s["rating"],
            confidence_low=s["confidence_low"],
            confidence_high=s["confidence_high"],
            explanation=s["explanation"],
        )
        for s in result.scenarios
    ]
    best = result.best_scenario
    best_out = ScenarioOut(
        team_id=best["team_id"],
        team_name=best["team_name"],
        league_name=best["league_name"],
        rating=best["rating"],
        confidence_low=best["confidence_low"],
        confidence_high=best["confidence_high"],
        explanation=best["explanation"],
    ) if best else scenarios_out[0]
    return CompareOut(
        player_id=result.player_id,
        scenarios=scenarios_out,
        best_scenario=best_out,
    )


@router.post("/transfer-impact", response_model=TransferImpactOut)
async def transfer_impact(request: Request, req: TransferImpactRequest):
    """Simulate transfer impact on player performance."""
    engine = get_engine(request)
    data = get_data(request)
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.simulate_transfer(req.player_id, req.from_team_id, req.to_team_id, req.season)
    )
    return TransferImpactOut(**result.__dict__)


@router.post("/what-if-teammates", response_model=PredictionOut)
async def what_if_teammates(request: Request, req: WhatIfTeammatesRequest):
    """What if the player had hypothetical-quality teammates?"""
    engine = get_engine(request)
    data = get_data(request)
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.what_if_teammates(
            req.player_id, req.team_id, req.hypothetical_avg_rating, req.season
        )
    )
    return PredictionOut(**result.__dict__)


@router.post("/what-if-lineup", response_model=LineupAnalysisOut)
async def what_if_lineup(request: Request, req: WhatIfLineupRequest):
    """Predict player performance with a specific named 5-man lineup."""
    engine = get_engine(request)
    data = get_data(request)
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    if req.team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.what_if_lineup(
            req.player_id, req.team_id, req.lineup_player_ids, req.season
        )
    )
    return LineupAnalysisOut(
        player_id=result.player_id,
        player_name=result.player_name,
        team_id=result.team_id,
        predicted_rating=result.predicted_rating,
        confidence_low=result.confidence_low,
        confidence_high=result.confidence_high,
        avg_lineup_rating=result.avg_lineup_rating,
        positions_covered=result.positions_covered,
        missing_positions=result.missing_positions,
        position_overlaps=result.position_overlaps,
        lineup_profiles=[
            LineupMemberProfileOut(
                player_id=p.player_id,
                player_name=p.player_name,
                position=p.position,
                predicted_rating=p.predicted_rating,
                role=p.role,
                style_compat=p.style_compat,
            )
            for p in result.lineup_profiles
        ],
        explanation=result.explanation,
    )

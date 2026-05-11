"""Scenario API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

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

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


def _get_engine():
    from basketball_ai.api.main import app_state
    return app_state["engine"]


def _get_data():
    from basketball_ai.api.main import app_state
    return app_state["data"]


@router.post("/what-if", response_model=PredictionOut)
def what_if_scenario(req: WhatIfRequest):
    """Predict performance for a specific player-team combination."""
    engine = _get_engine()
    data = _get_data()
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    if req.team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    result = engine.predict_in_team(req.player_id, req.team_id, req.season)
    return PredictionOut(**result.__dict__)


@router.get("/best-teams/{player_id}", response_model=List[TeamFitOut])
def best_teams_for_player(
    player_id: int,
    league_id: Optional[int] = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    """Find the best-fitting teams for a player."""
    engine = _get_engine()
    data = _get_data()
    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    fits = engine.best_team_fit(player_id, league_id=league_id, top_n=top_n)
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
def best_players_for_team(
    team_id: int,
    position: Optional[str] = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    """Find the best-fitting players for a team (by position)."""
    engine = _get_engine()
    data = _get_data()
    if team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    players = engine.best_player_for_team(team_id, position=position, top_n=top_n)
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
def compare_scenarios(req: CompareRequest):
    """Compare player performance across multiple teams."""
    engine = _get_engine()
    data = _get_data()
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    result = engine.compare_scenarios(req.player_id, req.team_ids, req.season)
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
def transfer_impact(req: TransferImpactRequest):
    """Simulate transfer impact on player performance."""
    engine = _get_engine()
    data = _get_data()
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    result = engine.simulate_transfer(
        req.player_id, req.from_team_id, req.to_team_id, req.season
    )
    return TransferImpactOut(**result.__dict__)


@router.post("/what-if-teammates", response_model=PredictionOut)
def what_if_teammates(req: WhatIfTeammatesRequest):
    """What if the player had hypothetical-quality teammates?"""
    engine = _get_engine()
    data = _get_data()
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    result = engine.what_if_teammates(
        req.player_id, req.team_id,
        req.hypothetical_avg_rating, req.season,
    )
    return PredictionOut(**result.__dict__)


@router.post("/what-if-lineup", response_model=LineupAnalysisOut)
def what_if_lineup(req: WhatIfLineupRequest):
    """Predict player performance with a specific named 5-man lineup."""
    engine = _get_engine()
    data = _get_data()
    if req.player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail="Player not found")
    if req.team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail="Team not found")
    result = engine.what_if_lineup(
        req.player_id, req.team_id, req.lineup_player_ids, req.season
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

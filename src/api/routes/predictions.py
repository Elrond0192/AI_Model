"""Prediction API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import PredictionOut, TrajectoryPointOut, PeakPredictionOut

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _get_engine():
    from src.api.main import app_state
    return app_state["engine"]


def _get_data():
    from src.api.main import app_state
    return app_state["data"]


@router.post("/player/{player_id}/team/{team_id}", response_model=PredictionOut)
def predict_player_in_team(
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

    result = engine.predict_in_team(player_id, team_id, season)
    return PredictionOut(
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


@router.get("/player/{player_id}/trajectory", response_model=List[TrajectoryPointOut])
def get_trajectory(
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

    traj = engine.predict_age_trajectory(player_id, age_range=age_range, team_id=team_id)
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
def get_peak_prediction(player_id: int, team_id: Optional[int] = Query(None)):
    """Predict a player's career peak."""
    engine = _get_engine()
    data = _get_data()

    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    peak = engine.predict_peak(player_id, team_id=team_id)
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

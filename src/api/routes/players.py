"""Player API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import PlayerOut, PlayerStatOut, PlayerProfileOut
from src.features.player_features import (
    compute_form_score,
    compute_consistency_score,
    compute_career_trajectory,
    _default_player_features,
)
from src.utils.helpers import position_group

router = APIRouter(prefix="/players", tags=["players"])


def _get_data():
    """Lazily import app_state to avoid circular imports."""
    from src.api.main import app_state
    return app_state["data"]


@router.get("", response_model=List[PlayerOut])
def list_players(
    position: Optional[str] = Query(None),
    nationality: Optional[str] = Query(None),
    min_age: Optional[int] = Query(None),
    max_age: Optional[int] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
):
    """List players with optional filters."""
    data = _get_data()
    df = data["players"].copy()

    if position:
        df = df[df["position"] == position.upper()]
    if nationality:
        df = df[df["nationality"].str.lower() == nationality.lower()]
    if min_age is not None:
        df = df[df["age"] >= min_age]
    if max_age is not None:
        df = df[df["age"] <= max_age]

    subset = df.iloc[offset: offset + limit]
    return [PlayerOut(**row.to_dict()) for _, row in subset.iterrows()]


@router.get("/{player_id}", response_model=PlayerOut)
def get_player(player_id: int):
    """Get a single player by ID."""
    data = _get_data()
    player = data["player_dict"].get(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return PlayerOut(**player)


@router.get("/{player_id}/stats", response_model=List[PlayerStatOut])
def get_player_stats(player_id: int):
    """Get all season stats for a player."""
    data = _get_data()
    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    stats = data["player_stats"][data["player_stats"]["player_id"] == player_id]
    return [PlayerStatOut(**row.to_dict()) for _, row in stats.iterrows()]


@router.get("/{player_id}/profile", response_model=PlayerProfileOut)
def get_player_profile(player_id: int):
    """Get an enriched player profile with computed metrics."""
    data = _get_data()
    player = data["player_dict"].get(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    history = data["player_stats"][
        data["player_stats"]["player_id"] == player_id
    ].sort_values("season")

    form = compute_form_score(history)
    consistency = compute_consistency_score(history)
    peak = float(history["rating"].max()) if not history.empty else 6.5
    trajectory = compute_career_trajectory(history)

    latest_stat = None
    if not history.empty:
        last_row = history.iloc[-1]
        latest_stat = PlayerStatOut(**last_row.to_dict())

    return PlayerProfileOut(
        player=PlayerOut(**player),
        latest_stats=latest_stat,
        form_score=round(form, 3),
        consistency_score=round(consistency, 3),
        peak_rating=round(peak, 3),
        career_trajectory=round(trajectory, 4),
        position_group=position_group(str(player.get("position", "CM"))),
    )

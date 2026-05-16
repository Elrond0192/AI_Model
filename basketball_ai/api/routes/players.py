"""Basketball player API routes."""
from __future__ import annotations
import hashlib
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Request, Response
from basketball_ai.api.schemas import PlayerOut, PlayerStatOut, PlayerProfileOut
from basketball_ai.features.player_features import compute_form_score, compute_consistency_score, compute_career_trajectory
from basketball_ai.utils.helpers import position_group

router = APIRouter(prefix="/players", tags=["players"])


def _get_data():
    from basketball_ai.api.main import app_state
    return app_state["data"]


@router.get("", response_model=List[PlayerOut])
def list_players(
    response: Response,
    position: Optional[str] = Query(None, description="e.g. PG, SG/SF, PF/C"),
    nationality: Optional[str] = Query(None),
    min_age: Optional[int] = Query(None),
    max_age: Optional[int] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    """List basketball players with optional filters."""
    data = _get_data()
    df   = data["players"].copy()

    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if nationality:
        df = df[df["nationality"].str.lower() == nationality.lower()]
    if min_age is not None:
        df = df[df["age"] >= min_age]
    if max_age is not None:
        df = df[df["age"] <= max_age]

    total = len(df)
    response.headers["X-Total-Count"] = str(total)
    subset = df.iloc[offset: offset + limit]
    results = []
    for _, row in subset.iterrows():
        d = row.to_dict()
        for col in ["current_team_id", "current_league_id"]:
            import math
            if col in d and (d[col] is None or (isinstance(d[col], float) and math.isnan(d[col]))):
                d[col] = None
        results.append(PlayerOut(**d))
    return results


@router.get("/{player_id}", response_model=PlayerOut)
def get_player(player_id: int, request: Request, response: Response):
    data   = _get_data()
    player = data["player_dict"].get(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    # AP5 – ETag for deterministic GET
    etag = '"' + hashlib.sha256(json.dumps(player, sort_keys=True, default=str).encode()).hexdigest() + '"'
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304)
    response.headers["ETag"] = etag
    return PlayerOut(**player)


@router.get("/{player_id}/stats", response_model=List[PlayerStatOut])
def get_player_stats(player_id: int):
    data = _get_data()
    if player_id not in data["player_dict"]:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    stats = data["player_stats"][data["player_stats"]["player_id"] == player_id]
    return [PlayerStatOut(**row.to_dict()) for _, row in stats.iterrows()]


@router.get("/{player_id}/profile", response_model=PlayerProfileOut)
def get_player_profile(player_id: int):
    data   = _get_data()
    player = data["player_dict"].get(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    history = data["player_stats"][
        data["player_stats"]["player_id"] == player_id
    ].sort_values("season")

    form        = compute_form_score(history)
    consistency = compute_consistency_score(history)
    peak        = float(history["rating"].max()) if not history.empty else 5.0
    trajectory  = compute_career_trajectory(history)

    latest_stat = None
    if not history.empty:
        last_row    = history.iloc[-1]
        latest_stat = PlayerStatOut(**last_row.to_dict())

    return PlayerProfileOut(
        player=PlayerOut(**player),
        latest_stats=latest_stat,
        form_score=round(form, 3),
        consistency_score=round(consistency, 3),
        peak_rating=round(peak, 3),
        career_trajectory=round(trajectory, 4),
        position_group=position_group(str(player.get("position", "PG"))),
    )

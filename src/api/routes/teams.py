"""Team API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import TeamOut, TeamAnalysisOut, PlayerOut

router = APIRouter(prefix="/teams", tags=["teams"])


def _get_data():
    from src.api.main import app_state
    return app_state["data"]


@router.get("", response_model=List[TeamOut])
def list_teams(
    league_id: Optional[int] = Query(None),
    tier: Optional[int] = Query(None),
    style: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
):
    """List teams with optional filters."""
    data = _get_data()
    df = data["teams"].copy()

    if league_id is not None:
        df = df[df["league_id"] == league_id]
    if tier is not None:
        df = df[df["league_tier"] == tier]
    if style:
        df = df[df["playing_style"] == style.lower()]

    subset = df.iloc[offset: offset + limit]
    return [TeamOut(**row.to_dict()) for _, row in subset.iterrows()]


@router.get("/{team_id}", response_model=TeamOut)
def get_team(team_id: int):
    """Get a single team by ID."""
    data = _get_data()
    team = data["team_dict"].get(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return TeamOut(**team)


@router.get("/{team_id}/analysis", response_model=TeamAnalysisOut)
def get_team_analysis(team_id: int):
    """Return squad analysis for a team."""
    data = _get_data()
    team = data["team_dict"].get(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Players currently at this team
    current_rels = data["team_player_relations"][
        (data["team_player_relations"]["team_id"] == team_id) &
        (data["team_player_relations"]["season"] == 2024)
    ]
    player_ids = current_rels["player_id"].tolist()
    squad = data["players"][data["players"]["id"].isin(player_ids)]

    # Average squad rating from recent stats
    recent_stats = data["player_stats"][
        (data["player_stats"]["team_id"] == team_id) &
        (data["player_stats"]["season"] >= 2022)
    ]
    avg_rating = float(recent_stats["rating"].mean()) if not recent_stats.empty else 6.5

    # Top 5 performers
    if not recent_stats.empty:
        top_pids = (
            recent_stats.groupby("player_id")["rating"]
            .mean()
            .nlargest(5)
            .index.tolist()
        )
    else:
        top_pids = player_ids[:5]

    top_players = data["players"][data["players"]["id"].isin(top_pids)]
    top_out = [PlayerOut(**row.to_dict()) for _, row in top_players.iterrows()]

    # Style strengths
    style = str(team.get("playing_style", "possession"))
    style_strengths = {
        "possession": ["Short passing", "Ball retention", "Positional play"],
        "high_press": ["Press intensity", "High defensive line", "Quick transitions"],
        "counter": ["Fast breaks", "Wing play", "Direct ball"],
        "direct": ["Long balls", "Set pieces", "Physical play"],
    }.get(style, ["Balanced"])

    return TeamAnalysisOut(
        team=TeamOut(**team),
        avg_squad_rating=round(avg_rating, 2),
        top_performers=top_out,
        style_strengths=style_strengths,
    )

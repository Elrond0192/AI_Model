"""Basketball team API routes."""
from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from basketball_ai.api.schemas import TeamOut, TeamAnalysisOut, PlayerOut

router = APIRouter(prefix="/teams", tags=["teams"])


def _get_data():
    from basketball_ai.api.main import app_state
    return app_state["data"]


@router.get("", response_model=List[TeamOut])
def list_teams(
    league_id: Optional[int] = Query(None),
    tier: Optional[int] = Query(None),
    style: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
):
    data = _get_data()
    df   = data["teams"].copy()
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
    data = _get_data()
    team = data["team_dict"].get(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return TeamOut(**team)


@router.get("/{team_id}/roster", response_model=List[PlayerOut])
def get_team_roster(team_id: int, season: str = Query("2023-24")):
    data = _get_data()
    if team_id not in data["team_dict"]:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    rels    = data["team_player_relations"]
    pids    = rels[(rels["team_id"] == team_id) & (rels["season"] == season)]["player_id"].tolist()
    players = data["players"][data["players"]["id"].isin(pids)]
    results = []
    for _, row in players.iterrows():
        d = row.to_dict()
        import math
        for col in ["current_team_id", "current_league_id"]:
            if col in d and (d[col] is None or (isinstance(d[col], float) and math.isnan(d[col]))):
                d[col] = None
        results.append(PlayerOut(**d))
    return results


@router.get("/{team_id}/analysis", response_model=TeamAnalysisOut)
def get_team_analysis(team_id: int):
    data = _get_data()
    team = data["team_dict"].get(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    rels = data["team_player_relations"]
    current_rels = rels[(rels["team_id"] == team_id) & (rels["season"] == "2023-24")]
    player_ids   = current_rels["player_id"].tolist()

    recent_stats = data["player_stats"][
        (data["player_stats"]["team_id"] == team_id) &
        (data["player_stats"]["season"] >= "2021-22")
    ]
    avg_rating = float(recent_stats["rating"].mean()) if not recent_stats.empty else 6.0

    if not recent_stats.empty:
        top_pids = (
            recent_stats.groupby("player_id")["rating"]
            .mean().nlargest(5).index.tolist()
        )
    else:
        top_pids = player_ids[:5]

    top_players = data["players"][data["players"]["id"].isin(top_pids)]
    top_out = []
    for _, row in top_players.iterrows():
        d = row.to_dict()
        import math
        for col in ["current_team_id", "current_league_id"]:
            if col in d and (d[col] is None or (isinstance(d[col], float) and math.isnan(d[col]))):
                d[col] = None
        top_out.append(PlayerOut(**d))

    style = str(team.get("playing_style", "motion_offense"))
    style_strengths = {
        "pace_and_space": ["High pace", "3-point shooting", "Spacing", "Transition offense"],
        "pick_and_roll":  ["Pick-and-roll execution", "Screen and roll", "Mid-range game"],
        "isolation":      ["Star-driven offense", "1-on-1 scoring", "Late-shot creation"],
        "defensive":      ["Rim protection", "Low pace", "Physical defense", "Half-court sets"],
        "motion_offense": ["Ball movement", "Off-ball cuts", "Team scoring", "3-point shooting"],
        "post_up":        ["Inside scoring", "Post moves", "Paint dominance", "Free throws"],
    }.get(style, ["Balanced offense", "Team defense"])

    return TeamAnalysisOut(
        team=TeamOut(**team),
        avg_squad_rating=round(avg_rating, 2),
        top_performers=top_out,
        style_strengths=style_strengths,
    )

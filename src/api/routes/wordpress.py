"""WordPress-optimised API routes.

Endpoint
--------
GET /api/v1/wordpress/player-card/{player_id}

Returns a pre-formatted JSON object ready for consumption by the WordPress
WPGetAPI plugin (or any custom shortcode / REST block) without the client
needing to perform any data transformation.

The response deliberately mirrors what a WordPress template would display:
name, position, current rating, peak forecast, confidence interval, and the
top-3 best team fits.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.api.schemas import PlayerCardOut, TeamFitSummary

router = APIRouter(prefix="/wordpress", tags=["wordpress"])


def _get_engine():
    from src.api.main import app_state
    return app_state.get("engine")


def _get_data():
    from src.api.main import app_state
    return app_state.get("data")


@router.get("/player-card/{player_id}", response_model=PlayerCardOut)
def get_player_card(player_id: int):
    """Return a WordPress-ready player card for *player_id*.

    Suitable for use as::

        [basketball_player_card id="42"]

    or fetched via ``WPGetAPI`` and rendered with a custom template.
    """
    engine = _get_engine()
    data   = _get_data()

    if not data or player_id not in data.get("player_dict", {}):
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    player_row = data["player_dict"][player_id]
    team_dict  = data["team_dict"]

    # Current team name
    cur_tid      = player_row.get("current_team_id")
    current_team = str(team_dict.get(cur_tid, {}).get("name", "—")) if cur_tid else "—"

    # Current rating (prediction at current team)
    current_rating      = None
    confidence_low      = None
    confidence_high     = None
    if engine and cur_tid and cur_tid in team_dict:
        try:
            pred           = engine.predict_in_team(player_id, cur_tid)
            current_rating = pred.predicted_rating
            confidence_low = pred.confidence_low
            confidence_high = pred.confidence_high
        except Exception:
            pass

    # Peak prediction
    peak_rating = None
    peak_age    = None
    if engine:
        try:
            peak        = engine.predict_peak(player_id)
            peak_rating = peak.peak_rating
            peak_age    = peak.peak_age
        except Exception:
            pass

    # Top-3 team fits
    top_teams: list[TeamFitSummary] = []
    if engine:
        try:
            fits = engine.best_team_fit(player_id, top_n=3)
            top_teams = [
                TeamFitSummary(
                    rank=f.rank,
                    team_name=f.team_name,
                    league_name=f.league_name,
                    predicted_rating=f.predicted_rating,
                )
                for f in fits
            ]
        except Exception:
            pass

    return PlayerCardOut(
        player_id=player_id,
        name=str(player_row.get("name", "")),
        position=str(player_row.get("position", "")),
        age=int(player_row.get("age", 0)),
        current_team=current_team,
        current_rating=current_rating,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        peak_rating=peak_rating,
        peak_age=peak_age,
        top_teams=top_teams,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

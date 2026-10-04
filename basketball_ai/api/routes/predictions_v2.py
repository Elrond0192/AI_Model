"""Version 2 typed prediction endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.contracts_v2 import (
    CompetitionSupportV2,
    PlayerTeamPredictionRequestV2,
    PlayerTeamPredictionV2,
)
from basketball_ai.models.competition_training import (
    normalize_competition,
    resolve_league_id,
    scope_prediction_context,
)
from basketball_ai.models.strict_production import (
    StrictWhatIfEngine,
    build_historical_snapshot,
)

router = APIRouter(prefix="/api/v2/predictions", tags=["predictions-v2"])


@router.post("/player-team", response_model=PlayerTeamPredictionV2)
async def player_team(body: PlayerTeamPredictionRequestV2, request: Request):
    engine, data = request.app.state.engine, request.app.state.data
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    player = next(
        (
            value
            for value in data.get("player_dict", {}).values()
            if str(value.get("global_id", "")) == body.player_global_id
        ),
        None,
    )
    team = next(
        (
            value
            for value in data.get("team_dict", {}).values()
            if str(value.get("global_id", "")) == body.team_global_id
        ),
        None,
    )
    if not player or not team:
        raise HTTPException(
            404,
            "Resolved player or team is unavailable for the requested context",
        )

    player_id = int(player["id"])
    team_id = int(team["id"])
    competition = normalize_competition(body.competition)

    try:
        league_id = resolve_league_id(data, body.league)
        snapshot = build_historical_snapshot(data, int(body.season))
        scoped = scope_prediction_context(
            snapshot,
            player_id,
            team_id,
            league_id,
            competition,
            int(body.season),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"Historical competition context is unavailable: {exc}") from exc

    resolved_team_id = int(scoped.get("_prediction_team_id", team_id))
    if player_id not in scoped.get("player_dict", {}) or resolved_team_id not in scoped.get("team_dict", {}):
        raise HTTPException(
            404,
            "Resolved player or team has no state in the requested league/competition",
        )

    historical_engine = StrictWhatIfEngine(engine.ensemble, scoped)
    try:
        result = historical_engine.predict_in_team(
            player_id,
            resolved_team_id,
            int(body.season),
            competition=competition,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc

    ensemble = engine.ensemble
    support = dict(scoped.get("_competition_support", {}))
    competition_intervals = getattr(
        ensemble, "_conformal_by_competition", {}
    ) or {}
    calibration_samples = getattr(
        ensemble, "_conformal_samples_by_competition", {}
    ) or {}
    support.update(
        {
            "calibration_scope": (
                "competition" if competition in competition_intervals else "global"
            ),
            "calibration_samples": int(calibration_samples.get(competition, 0)),
        }
    )

    calibration = getattr(request.app.state, "prediction_calibration", None)
    predicted_rating_100 = None
    confidence_low_100 = None
    confidence_high_100 = None
    calibration_version = None
    if calibration:
        from basketball_ai.prediction_calibration import native_to_100

        predicted_rating_100 = native_to_100(result.predicted_rating, calibration["fit"])
        confidence_low_100 = native_to_100(result.confidence_low, calibration["fit"])
        confidence_high_100 = native_to_100(result.confidence_high, calibration["fit"])
        calibration_version = calibration.get("calibration_version")

    return PlayerTeamPredictionV2(
        **request.app.state.model_metadata,
        player_global_id=body.player_global_id,
        team_global_id=body.team_global_id,
        league=body.league,
        season=body.season,
        competition=competition,
        target_season=body.season + 1,
        predicted_rating=result.predicted_rating,
        predicted_rating_100=predicted_rating_100,
        confidence_low=result.confidence_low,
        confidence_high=result.confidence_high,
        confidence_low_100=confidence_low_100,
        confidence_high_100=confidence_high_100,
        prediction_calibration_version=calibration_version,
        competition_support=CompetitionSupportV2(**support),
        generated_at=datetime.now(timezone.utc),
        explanation={
            "method": "forecast_t_plus_1",
            "context": "isolated_league_competition_as_of_source_season",
        },
    )

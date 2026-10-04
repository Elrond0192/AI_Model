"""Version 2 typed prediction endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.contracts_v2 import (
    CompatibilityComparisonV2,
    CompatibilityPlayerProfileV2,
    CompatibilityTeamProfileV2,
    CompetitionSupportV2,
    PlayerTeamPredictionRequestV2,
    PlayerTeamPredictionV2,
)
from basketball_ai.models.competition_training import (
    normalize_competition,
    resolve_league_id,
    scope_prediction_context,
)
from basketball_ai.models.performance_model import _canonical_source_value
from basketball_ai.data.loader import _to_int
from basketball_ai.models.production_training import _numeric_seasons
from basketball_ai.models.strict_production import (
    StrictWhatIfEngine,
    build_historical_snapshot,
)

router = APIRouter(prefix="/api/v2/predictions", tags=["predictions-v2"])


_PLAYER_COMPAT_PERCENT_COLUMNS = {
    "usg_pct",
    "ts_pct",
    "tov_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "orb_pct",
    "drb_pct",
    "three_par",
    "true_usg_pct",
}


def _compat_mean(stats: pd.DataFrame, column: str, fallback: float) -> float:
    if stats.empty or column not in stats.columns:
        return fallback
    values = pd.to_numeric(stats[column], errors="coerce")
    if column in _PLAYER_COMPAT_PERCENT_COLUMNS:
        values = values.map(lambda value: _canonical_source_value(column, value))
    values = values.dropna()
    return float(values.mean()) if not values.empty else fallback


def _compat_player_profile(
    data: dict[str, Any],
    player_id: int,
) -> CompatibilityPlayerProfileV2:
    player = data.get("player_dict", {}).get(_to_int(player_id), {})
    stats = data.get("player_stats", pd.DataFrame())
    player_ids = (
        stats["player_id"].map(_to_int)
        if "player_id" in stats.columns
        else pd.Series(dtype="Int64")
    )
    p_stats = stats.loc[player_ids == _to_int(player_id)].copy()

    return CompatibilityPlayerProfileV2(
        position=str(player.get("position", "PG") or "PG"),
        usg_pct=_compat_mean(p_stats, "usg_pct", 0.18),
        ts_pct=_compat_mean(p_stats, "ts_pct", 0.52),
        points=_compat_mean(p_stats, "points", 12.0),
        three_par=_compat_mean(p_stats, "three_par", 0.30),
        dbpm=_compat_mean(p_stats, "dbpm", 0.0),
    )


def _compat_team_profile(
    compat_model: Any,
    data: dict[str, Any],
    team_id: int,
    season: int,
    league_id: int,
    competition: str,
) -> Optional[CompatibilityTeamProfileV2]:
    finder = getattr(compat_model, "_team_row_context", None)
    row = (
        finder(
            team_id,
            data,
            int(season),
            int(league_id),
            competition,
            strict_before=False,
        )
        if callable(finder)
        else None
    )
    if row is None:
        return None
    season_values = _numeric_seasons(pd.DataFrame([row]))
    season_value = season_values.iloc[0]
    return CompatibilityTeamProfileV2(
        team_global_id=str(row.get("global_id", "") or "") or None,
        team_name=str(row.get("name", f"Team {team_id}") or f"Team {team_id}"),
        season=int(season_value) if pd.notna(season_value) else int(season),
        competition=competition,
        pace=float(row.get("pace", 75.0)),
        three_point_attempt_rate=float(row.get("three_point_attempt_rate", 0.35)),
        assists_per_game=float(row.get("assists_per_game", 20.0)),
        star_player_usage=float(row.get("star_player_usage", 0.25)),
        offensive_rating=float(row.get("offensive_rating", 110.0)),
        defensive_rating=float(row.get("defensive_rating", 110.0)),
    )


def _source_team_id(
    data: dict[str, Any],
    player_id: int,
    season: int,
    league_id: int,
    competition: str,
) -> Optional[int]:
    stats = data.get("player_stats", pd.DataFrame())
    if stats.empty or "player_id" not in stats.columns:
        return None
    rows = stats[stats["player_id"].map(_to_int) == _to_int(player_id)].copy()
    if "league_id" in rows.columns:
        rows = rows[rows["league_id"].map(_to_int) == _to_int(league_id)]
    if "competition" in rows.columns:
        rows = rows[
            rows["competition"].map(normalize_competition) == competition
        ]
    if rows.empty or "team_id" not in rows.columns:
        return None
    rows["_season_year"] = _numeric_seasons(rows)
    rows = rows[rows["_season_year"] == int(season)].dropna(subset=["team_id"])
    if rows.empty:
        return None
    return _to_int(rows.sort_values("_season_year").iloc[-1]["team_id"])


def _compatibility_comparison(
    snapshot: dict[str, Any],
    scoped: dict[str, Any],
    player_id: int,
    selected_team_id: int,
    league_id: int,
    competition: str,
    season: int,
    compat_model: Any,
) -> CompatibilityComparisonV2:
    selected_score = float(compat_model.score(player_id, selected_team_id, scoped))
    player_profile = _compat_player_profile(scoped, player_id)
    selected_profile = _compat_team_profile(
        compat_model, scoped, selected_team_id, season, league_id, competition
    )
    if selected_profile is None:
        raise ValueError("No compatibility context exists for the selected team")

    real_score: Optional[float] = None
    real_profile: Optional[CompatibilityTeamProfileV2] = None
    actual_team_raw = _source_team_id(
        snapshot, player_id, season, league_id, competition
    )

    if actual_team_raw is not None:
        try:
            actual_scoped = scope_prediction_context(
                snapshot,
                player_id,
                actual_team_raw,
                league_id,
                competition,
                season,
            )
            actual_team_id = _to_int(
                actual_scoped.get("_prediction_team_id", actual_team_raw)
            )
            real_score = float(
                compat_model.score(player_id, actual_team_id, actual_scoped)
            )
            real_profile = _compat_team_profile(
                compat_model,
                actual_scoped,
                actual_team_id,
                season,
                league_id,
                competition,
            )
        except (ValueError, RuntimeError):
            real_score = None
            real_profile = None

    return CompatibilityComparisonV2(
        selected_team_score=float(max(0.0, min(1.0, selected_score))),
        real_team_score=(
            float(max(0.0, min(1.0, real_score)))
            if real_score is not None
            else None
        ),
        score_delta_vs_real_team=(
            float(selected_score - real_score)
            if real_score is not None
            else None
        ),
        score_delta_vs_neutral=float(selected_score - 0.5),
        selected_team=selected_profile,
        real_team=real_profile,
        player_profile=player_profile,
    )


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

    compatibility = _compatibility_comparison(
        snapshot,
        scoped,
        player_id,
        resolved_team_id,
        league_id,
        competition,
        int(body.season),
        engine.ensemble.compat_model,
    )

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
        compatibility=compatibility,
        generated_at=datetime.now(timezone.utc),
        explanation={
            "method": "forecast_t_plus_1",
            "context": "isolated_league_competition_as_of_source_season",
        },
    )

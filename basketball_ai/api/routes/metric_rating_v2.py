"""FASE I — Metric Rating Engine API routes.

WordPress / Chat consume the precomputed, versioned distributions
(``AI.MetricDistribution``) — the percentile is never recomputed per request
and no percentile/tier logic lives in WordPress.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.metric_rating_contracts import (
    MetricRatingPlayerRequestV2,
    MetricRatingPlayerResponseV2,
    MetricRatingRateRequestV2,
    MetricRatingRateResponseV2,
)
from basketball_ai.metric_rating.definitions import get_metric_definition
from basketball_ai.metric_rating.features import (
    build_distribution_lookup,
    player_rating_snapshot,
)
from basketball_ai.metric_rating.rating import MetricRatingEngine

router = APIRouter(prefix="/api/v2/metric-rating", tags=["metric-rating-v2"])

DEFAULT_METRICS = ("RAPTOR", "LEBRON", "VORP")


def _season_year(value: int | str) -> int:
    """Normalize ``2025`` / ``"2025"`` / ``"2025-26"`` to the start year."""
    return int(str(value).split("-")[0])


def _distributions(request: Request) -> list[dict]:
    distributions = getattr(request.app.state, "metric_distributions", None)
    if not distributions:
        raise HTTPException(
            503,
            "Metric distributions are not loaded; run the FASE G backfill "
            "and restart the API",
        )
    return distributions


@router.post("/rate", response_model=MetricRatingRateResponseV2)
async def rate_metric(body: MetricRatingRateRequestV2, request: Request):
    try:
        definition = get_metric_definition(body.metric)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not math.isfinite(float(body.value)):
        raise HTTPException(422, "value must be finite")

    lookup = build_distribution_lookup(_distributions(request))
    key = (
        definition.metric,
        str(body.league).strip().upper(),
        _season_year(body.season),
        str(body.phase).strip().upper(),
    )
    distribution = lookup.get(key)
    if distribution is None:
        raise HTTPException(
            422,
            f"No stored distribution for metric={definition.metric} "
            f"league={key[1]} season={key[2]} phase={key[3]}",
        )

    engine = MetricRatingEngine(distribution_version=distribution.distribution_version)
    rating = engine.rate_from_distribution(definition.metric, float(body.value), distribution)
    return MetricRatingRateResponseV2(**rating.to_dict())


@router.post("/player-snapshot", response_model=MetricRatingPlayerResponseV2)
async def player_snapshot(body: MetricRatingPlayerRequestV2, request: Request):
    data = request.app.state.data
    frame = data.get("player_stats") if data else None
    if frame is None or getattr(frame, "empty", True):
        raise HTTPException(503, "Player data is not loaded")

    player = next(
        (
            value
            for value in data.get("player_dict", {}).values()
            if str(value.get("global_id", "")) == body.player_global_id
        ),
        None,
    )
    if player is None:
        raise HTTPException(404, "Resolved player is unavailable")

    metrics = [str(metric).strip().upper() for metric in (body.metrics or DEFAULT_METRICS)]
    try:
        ratings = player_rating_snapshot(
            frame,
            _distributions(request),
            player_global_id=body.player_global_id,
            player_id=int(player["id"]),
            league=body.league,
            season=_season_year(body.season),
            phase=body.phase,
            metrics=metrics,
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc

    return MetricRatingPlayerResponseV2(
        **request.app.state.model_metadata,
        player_global_id=body.player_global_id,
        league=str(body.league).strip().upper(),
        season=_season_year(body.season),
        phase=str(body.phase).strip().upper(),
        ratings=ratings,
        generated_at=datetime.now(timezone.utc),
    )

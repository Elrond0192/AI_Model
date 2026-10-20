"""Typed composable scenario endpoint for Chat V3."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from basketball_ai.api.scenario_contracts_v2 import (
    ScenarioRequestV2,
    ScenarioResponseV2,
)
from basketball_ai.data.postgres_loader import load_scenario_feeds
from basketball_ai.models.competition_training import normalize_competition, resolve_league_id
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine

router = APIRouter(prefix="/api/v2/scenarios", tags=["scenarios-v2"])
logger = logging.getLogger(__name__)

# Native/source rating values have meaning only inside the modeling pipeline.
# Public scenario responses may include the canonical BB-Rating (1–100) and
# calibrated forecasts, but must never leak the statistical 0–10 rating.
_PUBLIC_RAW_RATING_KEYS = {
    "rating", "current_rating", "previous_rating", "rating_delta",
    "rating_change", "predicted_rating_delta", "peak_rating", "historical_peak",
    "rating_trend_per_season", "trend_slope_per_season", "rating_mean",
    "rating_median", "rating_std", "recent_rating", "peak_premium",
    "change_vs_previous_season", "first_rating", "second_rating",
    "bb_rating_source_score", "source_rating", "raw_rating",
    "statistical_rating", "statistical_rating_delta",
    "previous_statistical_rating", "current_statistical_rating",
    "predicted_rating", "confidence_low", "confidence_high", "base_rating",
    "delta_vs_observed", "baseline_rating", "scenario_rating",
    "rating_with_pair_context", "observed_rating", "expected_rating",
    "actual_rating", "forecast_rating",
    # These acquisition fields are native player-rating comparisons, not
    # BB-Rating or team Net Rating. Never leak them through the public scenario API.
    "projected_player_rating", "projected_rating", "incoming_rating", "outgoing_rating",
    "empirical_transfer_delta", "league_transfer_delta", "league_prior_delta", "team_fit_delta",
    "predicted_rating_delta", "observed_rating_delta",
    "rotation_rating_before", "rotation_rating_after", "rotation_delta",
    "replacement_level_rating", "rotation_quality_delta",
}
_PUBLIC_RAW_METRIC_NAMES = {"rating", "source_rating", "statistical_rating"}
_DROP_PUBLIC_FIELD = object()


def _sanitize_public_scenario_payload(value):
    """Strip native/raw rating fields at the public API response boundary."""
    if isinstance(value, dict):
        metric_name = str(value.get("metric", "")).strip().lower()
        if metric_name in _PUBLIC_RAW_METRIC_NAMES:
            return _DROP_PUBLIC_FIELD
        if str(value.get("label", "")).strip().lower() in {
            "meaningful_positive_change",
            "meaningful_negative_change",
            "limited_change",
        }:
            # These labels are derived from native prediction-vs-source-rating
            # deltas; do not expose them without a calibrated comparable score.
            return _DROP_PUBLIC_FIELD
        sanitized = {}
        for key, item in value.items():
            if str(key).strip().lower() in _PUBLIC_RAW_RATING_KEYS:
                continue
            clean_item = _sanitize_public_scenario_payload(item)
            if clean_item is not _DROP_PUBLIC_FIELD:
                sanitized[key] = clean_item
        return sanitized
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        for item in value:
            clean_item = _sanitize_public_scenario_payload(item)
            if clean_item is not _DROP_PUBLIC_FIELD:
                sanitized_items.append(clean_item)
        return sanitized_items
    return value


def _season_year_for_feed(value) -> int | None:
    try:
        return int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None


def _lineup_feed_scope(
    data: dict,
    player_ids: list[int],
    requested_team_ids: list[int],
    source_league_id: int | None,
    season: int,
    competition: str,
) -> list[int]:
    """Bound player-attribution lineup reads to the player's recent team history."""
    team_ids = set(requested_team_ids)
    stats = data.get("player_stats")
    if stats is None or getattr(stats, "empty", True):
        return sorted(team_ids)
    if not {"player_id", "team_id", "season", "competition"}.issubset(stats.columns):
        return sorted(team_ids)
    for row in stats.to_dict("records"):
        try:
            player_id = int(row.get("player_id"))
            team_id = int(row.get("team_id"))
        except (TypeError, ValueError):
            continue
        year = _season_year_for_feed(row.get("season"))
        if player_id not in set(player_ids) or year is None:
            continue
        if year < max(2000, int(season) - 3) or year > int(season):
            continue
        if normalize_competition(row.get("competition")) != normalize_competition(competition):
            continue
        if source_league_id is not None and "league_id" in row:
            try:
                if int(row.get("league_id")) != int(source_league_id):
                    continue
            except (TypeError, ValueError):
                continue
        team_ids.add(team_id)
    return sorted(team_ids)


def _resolve_global_ids(data: dict, values: list[str], entity: str) -> list[int]:
    dictionary = data.get("player_dict" if entity == "player" else "team_dict", {})
    by_global = {
        str(row.get("global_id", "")): int(row["id"])
        for row in dictionary.values()
        if row.get("global_id") and row.get("id") is not None
    }
    resolved: list[int] = []
    for value in values:
        internal = by_global.get(str(value))
        if internal is None:
            raise HTTPException(404, f"Resolved {entity} is unavailable")
        resolved.append(internal)
    return resolved


@router.post("/evaluate", response_model=ScenarioResponseV2)
async def evaluate_scenario(body: ScenarioRequestV2, request: Request):
    runtime = request.app.state.engine
    data = request.app.state.data
    if runtime is None:
        raise HTTPException(503, "Model not loaded")

    logger.info(
        "[scenario] request scenario=%s league=%s target_league=%s season=%s competition=%s players=%d teams=%d",
        body.scenario, body.source_league, body.target_league, body.season, body.competition,
        len(body.player_global_ids), len(body.team_global_ids),
    )
    try:
        player_ids = _resolve_global_ids(data, body.player_global_ids, "player")
        team_ids = _resolve_global_ids(data, body.team_global_ids, "team")
        source_league_id = (
            resolve_league_id(data, body.source_league)
            if body.source_league is not None
            else None
        )
        target_league_id = (
            resolve_league_id(data, body.target_league)
            if body.target_league is not None
            else None
        )
        league_keys = []
        for league_id in (source_league_id, target_league_id):
            row = data.get("league_dict", {}).get(league_id, {})
            key = row.get("league_key")
            if key:
                league_keys.append(str(key))
        # Player/team attribution needs lineup stints for the player's own
        # recent team-seasons. Load them only for the relevant question families;
        # ordinary Player Intelligence requests avoid an unnecessary PBP query.
        feed_scenario = body.scenario
        needs_lineup_context = body.scenario == "causal_team_effect"
        if body.scenario == "player_intelligence":
            parameters = body.parameters or {}
            raw_keys = parameters.get("question_keys")
            if not isinstance(raw_keys, (list, tuple)):
                raw_keys = [parameters.get("question_key", "current_level")]
            normalized_keys = {str(key).strip().lower() for key in raw_keys}
            needs_lineup_context = bool(
                normalized_keys & {"performance_attribution", "causal_team_effect"}
            )
            if needs_lineup_context:
                feed_scenario = "player_intelligence_team_attribution"
        feed_team_ids = (
            _lineup_feed_scope(
                data,
                player_ids,
                team_ids,
                source_league_id,
                body.season,
                body.competition,
            )
            if needs_lineup_context else team_ids
        )
        feeds = await asyncio.to_thread(
            load_scenario_feeds,
            feed_scenario,
            player_ids=player_ids,
            team_ids=feed_team_ids,
            league_keys=league_keys,
            season=body.season,
            competition=body.competition,
        )
        scenario_data = {**data, **feeds}
        ensemble = runtime.ensemble
        engine = ChatScenarioEngine(
            ensemble,
            scenario_data,
            bb_rating_engine=getattr(request.app.state, "bb_rating_engine", None),
            bb_rating_uncertainty=getattr(request.app.state, "bb_rating_uncertainty", None),
            bb_rating_prediction_model=getattr(request.app.state, "bb_rating_future_prediction_model", None),
            future_performance_model=getattr(request.app.state, "future_performance_model", None),
        )
        payload = engine.evaluate(
            body.model_dump(),
            player_ids,
            team_ids,
            source_league_id,
            target_league_id,
        )
        payload = _sanitize_public_scenario_payload(payload)
        logger.info(
            "[scenario] success scenario=%s result_keys=%s evidence=%d",
            body.scenario, sorted(payload.get("result", {}).keys()), len(payload.get("evidence", [])),
        )
    except HTTPException:
        raise
    except (ValueError, RuntimeError, KeyError) as exc:
        logger.exception(
            "[scenario] 422 scenario=%s league=%s season=%s competition=%s error=%s",
            body.scenario, body.source_league, body.season, body.competition, exc,
        )
        raise HTTPException(422, str(exc)) from exc

    return ScenarioResponseV2(
        **request.app.state.model_metadata,
        scenario=body.scenario,
        generated_at=datetime.now(UTC),
        result=payload.get("result", {}),
        evidence=payload.get("evidence", []),
        support=payload.get("support", {}),
        limitations=payload.get("limitations", []),
        provenance=payload.get("provenance", {}),
    )

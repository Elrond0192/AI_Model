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
from basketball_ai.models.competition_training import resolve_league_id
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine

router = APIRouter(prefix="/api/v2/scenarios", tags=["scenarios-v2"])
logger = logging.getLogger(__name__)


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
        feeds = await asyncio.to_thread(
            load_scenario_feeds,
            body.scenario,
            player_ids=player_ids,
            team_ids=team_ids,
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
            future_performance_model=getattr(request.app.state, "future_performance_model", None),
        )
        payload = engine.evaluate(
            body.model_dump(),
            player_ids,
            team_ids,
            source_league_id,
            target_league_id,
        )
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
    )

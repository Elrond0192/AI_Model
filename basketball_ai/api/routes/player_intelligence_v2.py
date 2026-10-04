"""Player Intelligence capability discovery endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from basketball_ai.api.question_analysis_registry import (
    capability_dicts,
    registry_summary,
)


router = APIRouter(prefix="/api/v2/player-intelligence", tags=["player-intelligence"])


@router.get("/capabilities")
def player_intelligence_capabilities() -> dict:
    """Return the current question-to-analysis capability map.

    This is metadata only. It does not run a model and never changes
    Prediction Model or BB-Rating outputs.
    """
    return {
        "version": "1.0",
        "summary": registry_summary(),
        "capabilities": capability_dicts(),
    }

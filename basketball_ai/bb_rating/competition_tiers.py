"""Competition-tier context shared by BB-Rating and Player Intelligence.

Tiers are a contextual classification for transferability and cross-competition
analysis. They are deliberately NOT a fixed quality multiplier and must not
penalise or inflate a player's within-league BB-Rating by themselves.
"""
from __future__ import annotations

from typing import Optional

COMPETITION_TIER_DEFINITIONS: dict[int, dict[str, str]] = {
    1: {
        "label": "European elite",
        "description": "Top European club competitions represented by EL and EC.",
    },
    2: {
        "label": "National first division",
        "description": "Primary national first divisions in the current BBallstat pool.",
    },
    3: {
        "label": "National second division",
        "description": "Second-tier national competition, currently ITA2.",
    },
}

# This is a modelling context classification, not an official ranking of leagues.
# BEL1 is intentionally grouped with the existing national first divisions.
LEAGUE_TO_COMPETITION_TIER: dict[str, int] = {
    "EL": 1,
    "EC": 1,
    "ESP1": 2,
    "FRA1": 2,
    "GER1": 2,
    "GRC1": 2,
    "ISR1": 2,
    "ITA1": 2,
    "LIT1": 2,
    "TUR1": 2,
    "BEL1": 2,
    "ITA2": 3,
}


def competition_tier(league: str) -> Optional[int]:
    """Return the configured modelling tier for a league code."""
    return LEAGUE_TO_COMPETITION_TIER.get(str(league).strip().upper())


def competition_tier_label(league: str) -> Optional[str]:
    """Return the human-readable tier label for a league code."""
    return competition_tier_name(competition_tier(league))


def competition_tier_name(tier: int | None) -> Optional[str]:
    """Return the human-readable label for a numeric modelling tier."""
    try:
        tier_value = int(tier) if tier is not None else None
    except (TypeError, ValueError):
        tier_value = None
    definition = COMPETITION_TIER_DEFINITIONS.get(tier_value)
    return definition["label"] if definition else None


def competition_tier_metadata(league: str) -> dict[str, object]:
    """Return stable tier metadata suitable for API/model evidence."""
    tier = competition_tier(league)
    definition = COMPETITION_TIER_DEFINITIONS.get(tier)
    return {
        "league": str(league).strip().upper(),
        "competition_tier": tier,
        "competition_tier_label": definition["label"] if definition else None,
        "competition_tier_description": definition["description"] if definition else None,
    }


__all__ = [
    "COMPETITION_TIER_DEFINITIONS",
    "LEAGUE_TO_COMPETITION_TIER",
    "competition_tier",
    "competition_tier_label",
    "competition_tier_name",
    "competition_tier_metadata",
]

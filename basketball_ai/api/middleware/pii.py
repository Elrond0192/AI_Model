"""PII masking utilities for API responses.

Masks sensitive player fields (birth_date, weight_kg, height_cm) in API
responses when the caller's role is not 'analyst' or 'admin'.

Usage::

    from basketball_ai.api.middleware.pii import mask_player_pii

    player_dict = mask_player_pii(player_dict, role="viewer")
"""
from __future__ import annotations

from typing import Any, Dict, List

#: Roles that may see full PII
_PII_ALLOWED_ROLES = {"admin", "analyst"}

#: Fields to mask for non-PII roles
_PII_FIELDS = ("birth_date", "height_cm", "weight_kg", "nationality")


def mask_player_pii(player: Dict[str, Any], role: str = "") -> Dict[str, Any]:
    """Return a copy of *player* with PII fields masked unless role allows it.

    Args:
        player: Player dict from ``player_dict``.
        role:   Caller's role string (e.g. ``"viewer"``).

    Returns:
        Dict with sensitive fields replaced by ``None`` for non-PII roles.
    """
    if role in _PII_ALLOWED_ROLES:
        return player
    masked = dict(player)
    for field in _PII_FIELDS:
        if field in masked:
            masked[field] = None
    return masked


def mask_players_pii(players: List[Dict[str, Any]], role: str = "") -> List[Dict[str, Any]]:
    """Apply ``mask_player_pii`` to a list of player dicts."""
    return [mask_player_pii(p, role) for p in players]

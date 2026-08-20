"""Utility helper functions for the Basketball Performance AI system."""
from __future__ import annotations
import logging
import os
from typing import Any, Dict

# ---------------------------------------------------------------------------
# ID normalisation (single canonical implementation, imported everywhere)
# ---------------------------------------------------------------------------

def normalize_id(value: Any) -> Any:
    """Normalize an ID to int when possible, keep as-is for non-numeric strings.

    Handles ``int``, ``float``, decimal strings (``"42"``), and hex strings
    such as ``"0000009B"``.  Returns ``None`` when *value* is ``None``.

    >>> normalize_id(3.0)
    3
    >>> normalize_id("0000009B")
    155
    >>> normalize_id("GRC1")
    'GRC1'
    """
    if value is None:
        return None
    if isinstance(value, float):
        return int(value)
    if isinstance(value, int):
        return value
    v = str(value).strip()
    try:
        return int(v, 10)
    except (ValueError, TypeError):
        try:
            return int(v, 16)
        except (ValueError, TypeError):
            return v

# ---------------------------------------------------------------------------
# Team display-name helpers
# ---------------------------------------------------------------------------

#: Env-var that stores per-country field preferences, e.g. "IT:name,ES:short_name"
TEAM_DISPLAY_FIELD_MAP_ENV = "TEAM_DISPLAY_FIELD_MAP"


def parse_team_display_map(raw: str = "") -> Dict[str, str]:
    """Parse a ``"KEY:field,KEY:field,…"`` string into a ``{key: field}`` dict.

    *raw* defaults to the value of :data:`TEAM_DISPLAY_FIELD_MAP_ENV`.
    Valid *field* values are ``"name"`` and ``"short_name"``; anything else is
    ignored and falls back to the default behaviour.

    Keys can be either **league codes** (e.g. ``"ITA1"``, ``"GRC1"``) or
    **country codes** (e.g. ``"IT"``, ``"ES"``).  League codes take priority
    over country codes in :func:`team_display_name`.

    Example::

        parse_team_display_map("ITA1:name,ITA2:short_name,GRC1:name")
        # → {"ITA1": "name", "ITA2": "short_name", "GRC1": "name"}
    """
    if not raw:
        raw = os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, "")
    result: Dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if ":" not in token:
            continue
        key, _, field = token.partition(":")
        key = key.strip().upper()
        field = field.strip().lower()
        if key and field in ("name", "short_name"):
            result[key] = field
    return result


def team_display_name(
    team: dict,
    fallback: str = "Unknown",
    country: str = "",
    league_id: str = "",
    display_map: Dict[str, str] | None = None,
) -> str:
    """Return the display name for a team.

    The *display_map* (``{key: "name"|"short_name"}``) controls which DB field
    is used.  Keys can be league codes (e.g. ``"ITA1"``) or country codes
    (e.g. ``"IT"``).

    Lookup priority:
    1. **league_id** — exact league code (e.g. ``"ITA1"``, ``"GRC1"``).
       This allows per-league configuration within the same country.
    2. **country** — country code as fallback (e.g. ``"ITA"``, ``"IT"``).
    3. **default** — prefer *short_name* when non-empty, otherwise *name*.

    Args:
        team:        Team dict with at least ``"name"`` and ``"short_name"`` keys.
        fallback:    Value returned when neither field yields a non-empty string.
        country:     Country code for this team's league (e.g. ``"ITA"``).
        league_id:   League code (e.g. ``"ITA1"``). Has higher priority than
                     *country* in the display map lookup.
        display_map: Mapping produced by :func:`parse_team_display_map`.
                     When ``None`` the global env-var is read automatically.
    """
    if display_map is None:
        display_map = parse_team_display_map()

    # 1. League code (highest priority)
    lid = (league_id or "").strip().upper()
    if lid and lid in display_map:
        field = display_map[lid]
        val = str(team.get(field, "") or "").strip()
        return val if val else str(team.get("name", fallback))

    # 2. Country code fallback
    cc = (country or "").strip().upper()
    if cc and cc in display_map:
        field = display_map[cc]
        val = str(team.get(field, "") or "").strip()
        return val if val else str(team.get("name", fallback))

    # 3. Default: prefer short_name when available
    sn = str(team.get("short_name", "") or "").strip()
    return sn if sn else str(team.get("name", fallback))


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    return logging.getLogger("basketball_ai")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def position_group(position: str) -> str:
    """Return positional group for a basketball position (incl. hybrid roles)."""
    primary = position.split("/")[0]
    groups = {
        "PG": "guard", "SG": "guard",
        "SF": "wing",
        "PF": "big",  "C": "big",
    }
    return groups.get(primary, "wing")


def tier_to_multiplier(tier: int) -> float:
    multipliers = {1: 1.00, 2: 0.88, 3: 0.76, 4: 0.64, 5: 0.52}
    return multipliers.get(tier, 0.70)


def format_prediction_output(
    player_name: str, team_name: str, rating: float,
    confidence_low: float, confidence_high: float,
    factors: Dict[str, float],
) -> str:
    lines = [
        f"\n{'='*60}",
        f"  Player: {player_name}",
        f"  Team:   {team_name}",
        f"  Predicted Rating: {rating:.2f}/10  "
        f"(95% CI: {confidence_low:.2f} – {confidence_high:.2f})",
        "  Contributing Factors:",
    ]
    for factor, value in factors.items():
        lines.append(f"    • {factor:<30} {value:+.3f}")
    lines.append("=" * 60)
    return "\n".join(lines)

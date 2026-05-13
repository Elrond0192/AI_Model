"""Utility helper functions for the Basketball Performance AI system."""
from __future__ import annotations
import logging
from typing import Dict


def team_display_name(team: dict, fallback: str = "Unknown") -> str:
    """Return the display name for a team, preferring short_name over name.

    ShortName is the translated/localised abbreviation stored in the DB.
    It is used unconditionally when non-empty (even when longer than 6 chars).
    """
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

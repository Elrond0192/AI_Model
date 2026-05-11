"""Utility helper functions for the football performance AI system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    return logging.getLogger("football_ai")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero division."""
    if denominator == 0:
        return default
    return numerator / denominator


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def position_group(position: str) -> str:
    """Return positional group for a position."""
    groups = {
        "GK": "goalkeeper",
        "CB": "defender",
        "FB": "defender",
        "CM": "midfielder",
        "AM": "midfielder",
        "W": "attacker",
        "ST": "attacker",
    }
    return groups.get(position, "midfielder")


def tier_to_multiplier(tier: int) -> float:
    """Convert league tier to a quality multiplier."""
    multipliers = {1: 1.00, 2: 0.90, 3: 0.80, 4: 0.70, 5: 0.60}
    return multipliers.get(tier, 0.75)


def format_prediction_output(player_name: str, team_name: str, rating: float,
                               confidence_low: float, confidence_high: float,
                               factors: Dict[str, float]) -> str:
    """Format a prediction result for human-readable output."""
    lines = [
        f"\n{'='*60}",
        f"  Player: {player_name}",
        f"  Team:   {team_name}",
        f"  Predicted Rating: {rating:.2f}/10  "
        f"(95% CI: {confidence_low:.2f} – {confidence_high:.2f})",
        f"  Contributing Factors:",
    ]
    for factor, value in factors.items():
        lines.append(f"    • {factor:<28} {value:+.3f}")
    lines.append("=" * 60)
    return "\n".join(lines)

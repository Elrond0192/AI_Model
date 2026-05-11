"""Basketball age development curves by position.

Uses an asymmetric Gaussian: the rise to peak can be faster or slower
than the post-peak decline, calibrated per position group.
"""
from __future__ import annotations

import numpy as np

# Peak age per position (pure and hybrid)
PEAK_AGES: dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}

# Sigma before peak (wider = slower development)
_SIGMA_BEFORE: dict[str, float] = {
    "PG": 4.5, "SG": 4.0, "SF": 4.5, "PF": 5.0, "C": 5.5,
    "PG/SG": 4.2, "SG/SF": 4.2, "SF/PF": 4.8, "PF/C": 5.2, "SG/PF": 4.5,
}

# Sigma after peak (narrower = faster decline)
_SIGMA_AFTER: dict[str, float] = {
    "PG": 4.0, "SG": 3.5, "SF": 4.0, "PF": 4.5, "C": 5.0,
    "PG/SG": 3.8, "SG/SF": 3.7, "SF/PF": 4.2, "PF/C": 4.8, "SG/PF": 4.0,
}


def _primary_pos(pos: str) -> str:
    return pos.split("/")[0]


def age_performance_factor(age: int, position: str) -> float:
    """Return a 0–1 performance multiplier based on age vs positional peak.

    Args:
        age:      Player age in years.
        position: Position string, e.g. "PG", "PF/C".

    Returns:
        Float in [0.40, 1.00].
    """
    primary = _primary_pos(position)
    peak   = PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))
    diff   = age - peak
    if diff <= 0:
        sigma = _SIGMA_BEFORE.get(position, _SIGMA_BEFORE.get(primary, 4.5))
    else:
        sigma = _SIGMA_AFTER.get(position, _SIGMA_AFTER.get(primary, 4.0))

    factor = float(np.exp(-0.5 * (diff / sigma) ** 2))
    return float(np.clip(factor, 0.40, 1.00))


def age_trajectory(
    position: str,
    age_start: int = 18,
    age_end: int = 40,
) -> list[tuple[int, float]]:
    """Return (age, factor) pairs across a career range."""
    return [(age, age_performance_factor(age, position)) for age in range(age_start, age_end + 1)]


def peak_age_window(position: str, threshold: float = 0.95) -> tuple[int, int]:
    """Return (start_age, end_age) when factor >= threshold."""
    start: int | None = None
    end:   int | None = None
    for age in range(18, 41):
        if age_performance_factor(age, position) >= threshold:
            if start is None:
                start = age
            end = age
    primary = _primary_pos(position)
    default = PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))
    return (start or default, end or default)

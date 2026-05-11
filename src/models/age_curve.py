"""Age curve model: Gaussian-like performance multiplier by position."""

from __future__ import annotations

import numpy as np

# Peak age per position
PEAK_AGES: dict[str, int] = {
    "GK": 31, "CB": 29, "FB": 26, "CM": 28, "AM": 27, "W": 25, "ST": 27,
}

# Width parameters (broader = slower decline)
_SIGMA_BEFORE: dict[str, float] = {
    "GK": 6.0, "CB": 5.5, "FB": 4.5, "CM": 5.0, "AM": 4.5, "W": 4.0, "ST": 4.5,
}
_SIGMA_AFTER: dict[str, float] = {
    "GK": 5.0, "CB": 4.5, "FB": 3.5, "CM": 4.0, "AM": 3.5, "W": 3.0, "ST": 3.5,
}


def age_performance_factor(age: int, position: str) -> float:
    """Return a 0–1 performance multiplier based on age vs positional peak.

    Uses an asymmetric Gaussian: faster decline after the peak for
    physically-demanding positions (W, FB), slower for GK / CB.

    Args:
        age: Player's age in years.
        position: Position code (GK/CB/FB/CM/AM/W/ST).

    Returns:
        Float in [0.40, 1.00].
    """
    peak = PEAK_AGES.get(position, 27)
    diff = age - peak
    if diff <= 0:
        sigma = _SIGMA_BEFORE.get(position, 5.0)
    else:
        sigma = _SIGMA_AFTER.get(position, 4.0)

    factor = float(np.exp(-0.5 * (diff / sigma) ** 2))
    return float(np.clip(factor, 0.40, 1.00))


def age_trajectory(position: str, age_start: int = 16, age_end: int = 40) -> list[tuple[int, float]]:
    """Return (age, factor) pairs over a career range."""
    return [(age, age_performance_factor(age, position)) for age in range(age_start, age_end + 1)]


def peak_age_window(position: str, threshold: float = 0.95) -> tuple[int, int]:
    """Return (start_age, end_age) when factor >= threshold."""
    start, end = None, None
    for age in range(16, 41):
        f = age_performance_factor(age, position)
        if f >= threshold:
            if start is None:
                start = age
            end = age
    return (start or PEAK_AGES[position], end or PEAK_AGES[position])

"""Basketball age development curves by position.

Uses an asymmetric Gaussian: the rise to peak can be faster or slower
than the post-peak decline, calibrated per position group.

Empirical fit
-------------
When historical data is available (a dict of ``{position: [(age, rating), ...]}``
pairs), ``fit_from_data()`` can derive position-specific peak age and sigmas by
optimising the asymmetric Gaussian to the data.  The hardcoded defaults serve as
fallback when no data is supplied or when the fit fails.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Peak age per position (pure and hybrid)
PEAK_AGES: Dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}

# Sigma before peak (wider = slower development)
_SIGMA_BEFORE: Dict[str, float] = {
    "PG": 4.5, "SG": 4.0, "SF": 4.5, "PF": 5.0, "C": 5.5,
    "PG/SG": 4.2, "SG/SF": 4.2, "SF/PF": 4.8, "PF/C": 5.2, "SG/PF": 4.5,
}

# Sigma after peak (narrower = faster decline)
_SIGMA_AFTER: Dict[str, float] = {
    "PG": 4.0, "SG": 3.5, "SF": 4.0, "PF": 4.5, "C": 5.0,
    "PG/SG": 3.8, "SG/SF": 3.7, "SF/PF": 4.2, "PF/C": 4.8, "SG/PF": 4.0,
}

# Empirically fitted parameters (overrides defaults when set via fit_from_data)
_fitted_peak_ages:    Dict[str, float] = {}
_fitted_sigma_before: Dict[str, float] = {}
_fitted_sigma_after:  Dict[str, float] = {}


def _primary_pos(pos: str) -> str:
    return pos.split("/")[0]


def age_performance_factor(age: int, position: str) -> float:
    """Return a 0–1 performance multiplier based on age vs positional peak.

    Uses empirically fitted parameters when available, falls back to
    hardcoded defaults otherwise.

    Args:
        age:      Player age in years.
        position: Position string, e.g. "PG", "PF/C".

    Returns:
        Float in [0.40, 1.00].
    """
    primary = _primary_pos(position)

    # Prefer empirically fitted values if available
    peak = (
        _fitted_peak_ages.get(position, _fitted_peak_ages.get(primary))
        or PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))
    )
    diff   = age - peak
    if diff <= 0:
        sigma = (
            _fitted_sigma_before.get(position, _fitted_sigma_before.get(primary))
            or _SIGMA_BEFORE.get(position, _SIGMA_BEFORE.get(primary, 4.5))
        )
    else:
        sigma = (
            _fitted_sigma_after.get(position, _fitted_sigma_after.get(primary))
            or _SIGMA_AFTER.get(position, _SIGMA_AFTER.get(primary, 4.0))
        )

    factor = float(np.exp(-0.5 * (diff / sigma) ** 2))
    return float(np.clip(factor, 0.40, 1.00))


def fit_from_data(
    data: Dict[str, List[Tuple[int, float]]],
    min_samples: int = 20,
) -> None:
    """Fit the asymmetric Gaussian parameters from historical (age, rating) data.

    Modifies the module-level ``_fitted_*`` dicts in-place.  Positions with
    fewer than *min_samples* data points are skipped (defaults remain).

    Args:
        data: Dict mapping position string → list of (age, rating) tuples.
              Ratings should be comparable (e.g. normalised 0-10 scale).
        min_samples: Minimum data points required to attempt fitting.

    Example::

        from basketball_ai.models.age_curve import fit_from_data
        historical = {
            "PG": [(22, 7.1), (24, 7.8), (26, 8.2), (28, 7.9), (31, 7.0)],
            ...
        }
        fit_from_data(historical)
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except ImportError:
        logger.warning(
            "[AgeCurve] scipy not installed – empirical fit skipped. "
            "pip install scipy to enable."
        )
        return

    def _asym_gauss(age_arr, peak_age, sigma_before, sigma_after):
        """Asymmetric Gaussian evaluated at a numpy array of ages."""
        result = np.empty_like(age_arr, dtype=float)
        for i, a in enumerate(age_arr):
            d = a - peak_age
            s = sigma_before if d <= 0 else sigma_after
            result[i] = np.exp(-0.5 * (d / s) ** 2)
        return result

    fitted = 0
    for position, points in data.items():
        if len(points) < min_samples:
            continue
        ages    = np.array([p[0] for p in points], dtype=float)
        ratings = np.array([p[1] for p in points], dtype=float)
        # Normalise ratings to [0, 1] for the fit
        r_min, r_max = ratings.min(), ratings.max()
        if r_max - r_min < 0.5:
            continue
        y = (ratings - r_min) / (r_max - r_min)
        primary = _primary_pos(position)
        p0 = [
            float(PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))),
            float(_SIGMA_BEFORE.get(position, _SIGMA_BEFORE.get(primary, 4.5))),
            float(_SIGMA_AFTER.get(position, _SIGMA_AFTER.get(primary, 4.0))),
        ]
        bounds = ([18, 1.0, 1.0], [38, 12.0, 12.0])
        try:
            popt, _ = curve_fit(_asym_gauss, ages, y, p0=p0, bounds=bounds, maxfev=5000)
            _fitted_peak_ages[position]    = float(round(popt[0], 1))
            _fitted_sigma_before[position] = float(round(popt[1], 2))
            _fitted_sigma_after[position]  = float(round(popt[2], 2))
            fitted += 1
            logger.info(
                "[AgeCurve] Fitted %s: peak_age=%.1f  sigma_before=%.2f  sigma_after=%.2f",
                position, popt[0], popt[1], popt[2],
            )
        except Exception as exc:
            logger.warning("[AgeCurve] Fit failed for %s: %s", position, exc)

    if fitted:
        logger.info("[AgeCurve] Empirical fit applied for %d position(s).", fitted)


def reset_fitted_params() -> None:
    """Clear any empirically fitted parameters, reverting to hardcoded defaults."""
    _fitted_peak_ages.clear()
    _fitted_sigma_before.clear()
    _fitted_sigma_after.clear()
    logger.info("[AgeCurve] Empirical parameters cleared; using defaults.")


def age_trajectory(
    position: str,
    age_start: int = 18,
    age_end: int = 40,
) -> list:
    """Return (age, factor) pairs across a career range."""
    return [(age, age_performance_factor(age, position)) for age in range(age_start, age_end + 1)]


def peak_age_window(position: str, threshold: float = 0.95) -> Tuple[int, int]:
    """Return (start_age, end_age) when factor >= threshold."""
    start: Optional[int] = None
    end:   Optional[int] = None
    for age in range(18, 41):
        if age_performance_factor(age, position) >= threshold:
            if start is None:
                start = age
            end = age
    primary = _primary_pos(position)
    default = PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))
    return (start or default, end or default)

"""Distribution engine for the Metric Rating Engine.

Builds a compact, request-independent description of a population's metric
values: sample size, quantiles, min/max, mean and population stddev. The
percentile of an arbitrary value is computed by linear interpolation over the
stored quantile function, so a stored :class:`DistributionStats` is fully
self-contained and can be persisted and versioned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# Quantiles (in percentage points) stored in the distribution.
_QUANTILE_POINTS: tuple[float, ...] = (10.0, 25.0, 40.0, 50.0, 60.0, 75.0, 90.0, 97.5)

# Quality thresholds: (quality, min_sample_size). First match wins, applied on
# a descending sample-size scale; anything below the last threshold is
# "insufficient".
DEFAULT_QUALITY_THRESHOLDS: tuple[tuple[str, int], ...] = (
    ("high", 200),
    ("medium", 100),
    ("low", 50),
)

MIN_QUALITY = "insufficient"


def quality_for_sample_size(
    sample_size: int,
    thresholds: Sequence[Tuple[str, int]] = DEFAULT_QUALITY_THRESHOLDS,
) -> str:
    """Return the quality label for a sample size (configurable thresholds)."""
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}")
    for quality, minimum in sorted(thresholds, key=lambda item: item[1], reverse=True):
        if sample_size >= minimum:
            return quality
    return MIN_QUALITY


@dataclass(frozen=True)
class DistributionStats:
    """Compact, versioned description of one population distribution."""

    sample_size: int
    min: float
    p10: float
    p25: float
    p40: float
    p50: float
    p60: float
    p75: float
    p90: float
    p97_5: float
    max: float
    mean: float
    stddev: float
    quality: str
    population_source: str = "unknown"
    population_type: str = "qualified_players"
    population_key: Tuple[str, ...] = ()
    distribution_version: str = "1.0"

    @property
    def quantile_points(self) -> list[tuple[float, float]]:
        """Ordered ``(percentile_points, value)`` pairs, including min/max."""
        return [
            (0.0, self.min),
            (10.0, self.p10),
            (25.0, self.p25),
            (40.0, self.p40),
            (50.0, self.p50),
            (60.0, self.p60),
            (75.0, self.p75),
            (90.0, self.p90),
            (97.5, self.p97_5),
            (100.0, self.max),
        ]

    def summary(self) -> dict[str, object]:
        """Plain dict with every stored statistic (for reports/storage)."""
        return {
            "sample_size": self.sample_size,
            "min": self.min,
            "p10": self.p10,
            "p25": self.p25,
            "p40": self.p40,
            "p50": self.p50,
            "p60": self.p60,
            "p75": self.p75,
            "p90": self.p90,
            "p97_5": self.p97_5,
            "max": self.max,
            "mean": self.mean,
            "stddev": self.stddev,
            "quality": self.quality,
            "population_source": self.population_source,
            "population_type": self.population_type,
            "population_key": list(self.population_key),
            "distribution_version": self.distribution_version,
        }


def build_distribution(
    values: Iterable[float],
    *,
    population_source: str = "unknown",
    population_type: str = "qualified_players",
    population_key: Tuple[str, ...] = (),
    distribution_version: str = "1.0",
    quality_thresholds: Sequence[Tuple[str, int]] = DEFAULT_QUALITY_THRESHOLDS,
) -> DistributionStats:
    """Compute a :class:`DistributionStats` from raw metric values.

    Raises ``ValueError`` on an empty input. Non-finite values are rejected
    (the population builder drops them before calling this).
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot build a distribution from an empty population")
    if not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be finite")
    array = np.sort(array)
    quantiles = {
        point: float(np.percentile(array, point, method="linear"))
        for point in _QUANTILE_POINTS
    }
    return DistributionStats(
        sample_size=int(array.size),
        min=float(array[0]),
        p10=quantiles[10.0],
        p25=quantiles[25.0],
        p40=quantiles[40.0],
        p50=quantiles[50.0],
        p60=quantiles[60.0],
        p75=quantiles[75.0],
        p90=quantiles[90.0],
        p97_5=quantiles[97.5],
        max=float(array[-1]),
        mean=float(np.mean(array)),
        stddev=float(np.std(array, ddof=0)),
        quality=quality_for_sample_size(int(array.size), quality_thresholds),
        population_source=population_source,
        population_type=population_type,
        population_key=tuple(population_key),
        distribution_version=distribution_version,
    )


def percentile_from_distribution(
    distribution: DistributionStats,
    value: float,
) -> float:
    """Percentile (fraction in ``[0, 1]``) of *value* in the distribution.

    Linear interpolation over the stored quantile function:
    ``(0, min) ... (97.5, p97.5) (100, max)``. ``value <= min`` → ``0.0``,
    ``value >= max`` → ``1.0``. A degenerate distribution (``min == max``)
    maps the single value to ``0.5``. Handles ties (repeated quantile values)
    without division by zero.
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    points = distribution.quantile_points
    if points[0][1] == points[-1][1]:
        # Degenerate distribution: every sample equals the same value.
        if value == points[0][1]:
            return 0.5
        return 0.0 if value < points[0][1] else 1.0
    if value < points[0][1]:
        return 0.0
    if value >= points[-1][1]:
        return 1.0

    previous_pct, previous_value = points[0]
    for pct, point_value in points[1:]:
        if point_value > previous_value:
            if value <= point_value:
                if value == previous_value:
                    return previous_pct / 100.0
                fraction = (value - previous_value) / (point_value - previous_value)
                return (previous_pct + (pct - previous_pct) * fraction) / 100.0
            previous_pct, previous_value = pct, point_value
        else:
            # Duplicate quantile value: keep the highest percentile for it.
            previous_pct = pct
    # value >= max is handled above; defensive fallback.
    return 1.0


def zscore_from_distribution(distribution: DistributionStats, value: float) -> Optional[float]:
    """``(value - mean) / stddev``; ``None`` when stddev is zero (never NaN/inf)."""
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    if distribution.stddev == 0.0:
        return None
    return (value - distribution.mean) / distribution.stddev

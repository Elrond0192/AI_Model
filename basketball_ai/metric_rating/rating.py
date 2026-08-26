"""Metric Rating Engine — the single rating entry point.

Transforms a raw advanced-metric value into an interpretable rating
(percentile, z-score, tier, label) relative to a contextual, versioned
population distribution. The engine is metric-agnostic: RAPTOR, LEBRON and
VORP (and future metrics) all flow through the same code path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from basketball_ai.metric_rating.definitions import (
    METRIC_DEFINITIONS,
    MetricDefinition,
    get_metric_definition,
)
from basketball_ai.metric_rating.distribution import (
    DEFAULT_QUALITY_THRESHOLDS,
    DistributionStats,
    build_distribution,
    percentile_from_distribution,
    zscore_from_distribution,
)
from basketball_ai.metric_rating.population import (
    PopulationBuilder,
    PopulationConfig,
)
from basketball_ai.metric_rating.tiers import (
    DEFAULT_TIERS,
    TierSpec,
    tier_for_percentile,
)

DEFAULT_DISTRIBUTION_VERSION = "1.0"
DEFAULT_RATING_VERSION = "1.0"


@dataclass(frozen=True)
class MetricRating:
    """Output of :meth:`MetricRatingEngine.rate`."""

    metric: str
    value: Optional[float]
    percentile: Optional[float]
    zscore: Optional[float]
    tier: Optional[int]
    label: Optional[str]
    sample_size: int
    quality: str
    population_source: str
    fallback_used: bool
    population_type: str
    population_key: Tuple[str, ...]
    distribution_version: str
    rating_version: str
    above_reference: Optional[bool] = None
    below_reference: Optional[bool] = None
    warnings: Tuple[str, ...] = ()

    def to_dict(self, *, rounded: bool = True) -> dict[str, Any]:
        """Plain dict; ``rounded=True`` (API default) trims float noise."""
        out: dict[str, Any] = {
            "metric": self.metric,
            "value": _fmt(self.value, 4, rounded),
            "percentile": _fmt(self.percentile, 4, rounded),
            "zscore": _fmt(self.zscore, 3, rounded),
            "tier": self.tier,
            "label": self.label,
            "sample_size": self.sample_size,
            "quality": self.quality,
            "population_source": self.population_source,
            "fallback_used": self.fallback_used,
            "population_type": self.population_type,
            "population_key": list(self.population_key),
            "distribution_version": self.distribution_version,
            "rating_version": self.rating_version,
            "above_reference": self.above_reference,
            "below_reference": self.below_reference,
            "warnings": list(self.warnings),
        }
        return out


def _fmt(value: Optional[float], ndigits: int, rounded: bool) -> Optional[float]:
    if not rounded or value is None:
        return value
    return round(float(value), ndigits)


class MetricRatingEngine:
    """Single rating engine for every registered metric."""

    def __init__(
        self,
        *,
        definitions: Optional[Mapping[str, MetricDefinition]] = None,
        tiers: Sequence[TierSpec] = DEFAULT_TIERS,
        quality_thresholds: Sequence[Tuple[str, int]] = DEFAULT_QUALITY_THRESHOLDS,
        population_config: Optional[PopulationConfig] = None,
        distribution_version: str = DEFAULT_DISTRIBUTION_VERSION,
        rating_version: str = DEFAULT_RATING_VERSION,
    ) -> None:
        self.definitions = dict(definitions or METRIC_DEFINITIONS)
        self.tiers = tuple(tiers)
        self.quality_thresholds = tuple(quality_thresholds)
        self.population_builder = PopulationBuilder(population_config)
        self.distribution_version = distribution_version
        self.rating_version = rating_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rate(
        self,
        metric: str,
        value: float,
        *,
        league: str,
        season: Any,
        phase: str,
        frame: Optional[pd.DataFrame] = None,
        distribution: Optional[DistributionStats] = None,
    ) -> MetricRating:
        """Rate a raw metric value.

        ``frame``: canonical stats DataFrame (``"AI_Source"`` contract
        columns) used to build the population on the fly.

        ``distribution``: a precomputed :class:`DistributionStats` (stored /
        versioned). Exactly one of the two must be provided.
        """
        definition = self._definition(metric)
        if not math.isfinite(float(value)):
            raise ValueError(f"value must be finite, got {value!r}")

        if distribution is not None:
            return self._rate_against(distribution, definition, value)

        if frame is None:
            raise ValueError("either frame or distribution must be provided")

        population = self.population_builder.build(
            frame,
            source_column=definition.source_column,
            league=league,
            season=season,
            phase=phase,
        )
        if population.sample_size == 0:
            return self._insufficient(
                definition,
                value,
                population_source="none",
                fallback_used=False,
                population_type=population.population_type,
                population_key=(),
                sample_size=0,
                warning="no population rows found for the requested context",
            )

        distribution = build_distribution(
            population.values,
            population_source=population.population_source,
            population_type=population.population_type,
            population_key=population.population_key,
            distribution_version=self.distribution_version,
            quality_thresholds=self.quality_thresholds,
        )
        return self._rate_against(
            distribution, definition, value, fallback_used=population.fallback_used
        )

    def rate_from_distribution(
        self,
        metric: str,
        value: float,
        distribution: DistributionStats,
    ) -> MetricRating:
        """Rate a value against a precomputed, versioned distribution."""
        return self.rate(
            metric, value, league="", season=0, phase="",
            distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _definition(self, metric: str) -> MetricDefinition:
        key = str(metric).strip().upper()
        definition = self.definitions.get(key)
        if definition is None:
            definition = get_metric_definition(metric)  # raises for unknown metrics
        if not definition.rating_enabled:
            raise ValueError(f"Rating is disabled for metric {metric!r}")
        return definition

    def _rate_against(
        self,
        distribution: DistributionStats,
        definition: MetricDefinition,
        value: float,
        *,
        fallback_used: Optional[bool] = None,
    ) -> MetricRating:
        if fallback_used is None:
            # Derived from the distribution provenance so a stored/versioned
            # distribution keeps reporting the fallback state truthfully.
            fallback_used = distribution.population_source != "exact"
        if distribution.quality == "insufficient":
            return self._insufficient(
                definition,
                value,
                population_source=distribution.population_source,
                fallback_used=fallback_used,
                population_type=distribution.population_type,
                population_key=distribution.population_key,
                sample_size=distribution.sample_size,
                warning="population below the minimum sample size; "
                "no precise rating is produced",
            )

        percentile = percentile_from_distribution(distribution, float(value))
        if definition.direction == "lower_better":
            percentile = 1.0 - percentile
        zscore = zscore_from_distribution(distribution, float(value))
        tier = tier_for_percentile(percentile, self.tiers)

        above = below = None
        if definition.replacement_reference is not None:
            above = definition.is_above_reference(float(value))
            below = definition.is_below_reference(float(value))

        return MetricRating(
            metric=definition.metric,
            value=float(value),
            percentile=percentile,
            zscore=zscore,
            tier=tier[0] if tier is not None else None,
            label=tier[1] if tier is not None else None,
            sample_size=distribution.sample_size,
            quality=distribution.quality,
            population_source=distribution.population_source,
            fallback_used=fallback_used,
            population_type=distribution.population_type,
            population_key=distribution.population_key,
            distribution_version=distribution.distribution_version,
            rating_version=self.rating_version,
            above_reference=above,
            below_reference=below,
        )

    def _insufficient(
        self,
        definition: MetricDefinition,
        value: float,
        *,
        population_source: str,
        fallback_used: bool,
        population_type: str,
        population_key: Tuple[str, ...],
        sample_size: int,
        warning: str,
    ) -> MetricRating:
        return MetricRating(
            metric=definition.metric,
            value=float(value),
            percentile=None,
            zscore=None,
            tier=None,
            label=None,
            sample_size=sample_size,
            quality="insufficient",
            population_source=population_source,
            fallback_used=fallback_used,
            population_type=population_type,
            population_key=tuple(population_key),
            distribution_version=self.distribution_version,
            rating_version=self.rating_version,
            warnings=(warning,),
        )

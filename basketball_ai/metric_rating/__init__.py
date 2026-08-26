"""Metric Rating Engine.

Centralized, metric-agnostic interpretation layer that transforms advanced
metrics already produced by HoopmetricsEngine (RAPTOR, LEBRON, VORP, …) into
contextual, versioned ratings: percentile, z-score, tier and label.

Public API::

    from basketball_ai.metric_rating import MetricRatingEngine

    engine = MetricRatingEngine()
    rating = engine.rate(
        metric="RAPTOR", value=5.82,
        league="ITA1", season="2025-26", phase="RS",
        frame=player_stats_frame,
    )
"""
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
    quality_for_sample_size,
    zscore_from_distribution,
)
from basketball_ai.metric_rating.population import (
    DEFAULT_COMPETITION_FAMILIES,
    DEFAULT_FALLBACK_ORDER,
    PopulationBuilder,
    PopulationConfig,
    PopulationResult,
)
from basketball_ai.metric_rating.rating import (
    DEFAULT_DISTRIBUTION_VERSION,
    DEFAULT_RATING_VERSION,
    MetricRating,
    MetricRatingEngine,
)
from basketball_ai.metric_rating.tiers import (
    DEFAULT_TIERS,
    TierSpec,
    tier_for_percentile,
    validate_tiers,
)

__all__ = [
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "get_metric_definition",
    "DEFAULT_QUALITY_THRESHOLDS",
    "DistributionStats",
    "build_distribution",
    "percentile_from_distribution",
    "quality_for_sample_size",
    "zscore_from_distribution",
    "DEFAULT_COMPETITION_FAMILIES",
    "DEFAULT_FALLBACK_ORDER",
    "PopulationBuilder",
    "PopulationConfig",
    "PopulationResult",
    "DEFAULT_DISTRIBUTION_VERSION",
    "DEFAULT_RATING_VERSION",
    "MetricRating",
    "MetricRatingEngine",
    "DEFAULT_TIERS",
    "TierSpec",
    "tier_for_percentile",
    "validate_tiers",
]

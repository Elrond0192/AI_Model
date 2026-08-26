"""FASE H — rating feature provider (vectorized).

Adds ``<metric>_value`` / ``<metric>_percentile`` / ``<metric>_zscore`` /
``<metric>_tier`` columns to a player-stats frame using the stored, versioned
``AI.MetricDistribution`` rows. The original metric columns are **never
replaced**; contexts without a distribution yield NaN (missing).

Vectorized: percentile via ``np.interp`` over the stored quantile function,
z-score via (value - mean) / stddev (NaN when stddev is zero), tier via
``searchsorted`` on the tier table — equivalent to the scalar engine.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from basketball_ai.metric_rating.definitions import METRIC_DEFINITIONS
from basketball_ai.metric_rating.distribution import DistributionStats
from basketball_ai.metric_rating.tiers import DEFAULT_TIERS, TierSpec

VALUE_SUFFIX = "_value"
PERCENTILE_SUFFIX = "_percentile"
ZSCORE_SUFFIX = "_zscore"
TIER_SUFFIX = "_tier"

DEFAULT_METRICS: tuple[str, ...] = ("RAPTOR", "LEBRON", "VORP")


def distribution_from_row(row: Mapping[str, Any]) -> DistributionStats:
    """Build a DistributionStats from a metric_distribution row (dict/Series)."""
    return DistributionStats(
        sample_size=int(row["sample_size"]),
        min=float(row["min"]),
        p10=float(row["p10"]),
        p25=float(row["p25"]),
        p40=float(row["p40"]),
        p50=float(row["p50"]),
        p60=float(row["p60"]),
        p75=float(row["p75"]),
        p90=float(row["p90"]),
        p97_5=float(row["p97_5"]),
        max=float(row["max"]),
        mean=float(row["mean"]),
        stddev=float(row["stddev"]),
        quality=str(row["quality"]),
        population_source=str(row.get("population_source", "exact")),
        population_type=str(row.get("population_type", "qualified_players")),
        population_key=(),
        distribution_version=str(row.get("distribution_version", "1.0")),
    )


def build_distribution_lookup(
    distribution_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str], DistributionStats]:
    """Key ``(metric, league_key, season, competition)`` -> DistributionStats."""
    lookup: dict[tuple[str, str, int, str], DistributionStats] = {}
    for row in distribution_rows:
        key = (
            str(row["metric"]).strip().upper(),
            str(row["league_key"]).strip().upper(),
            int(row["season"]),
            str(row["competition"]).strip().upper(),
        )
        lookup[key] = distribution_from_row(row)
    return lookup


def _interp_tables(dist: DistributionStats) -> tuple[np.ndarray, np.ndarray]:
    """Strictly-increasing quantile table; ties keep the highest percentile."""
    xp: list[float] = []
    fp: list[float] = []
    for pct, value in dist.quantile_points:
        if xp and value == xp[-1]:
            fp[-1] = pct
        else:
            xp.append(value)
            fp.append(pct)
    return np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)


def percentile_series(dist: DistributionStats, values: pd.Series) -> pd.Series:
    """Percentile in [0, 1] per finite value (NaN stays NaN)."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if dist.min == dist.max:
        out = np.where(np.isnan(arr), np.nan,
                       np.where(arr == dist.min, 0.5,
                                np.where(arr < dist.min, 0.0, 1.0)))
        return pd.Series(out, index=values.index, dtype=float)
    xp, fp = _interp_tables(dist)
    pct = np.interp(arr, xp, fp) / 100.0
    return pd.Series(pct, index=values.index, dtype=float)


def zscore_series(dist: DistributionStats, values: pd.Series) -> pd.Series:
    """(value - mean) / stddev; NaN when stddev is zero (never inf)."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if dist.stddev == 0.0:
        return pd.Series(np.full(len(arr), np.nan), index=values.index, dtype=float)
    return pd.Series((arr - dist.mean) / dist.stddev, index=values.index, dtype=float)


def tier_series(
    percentiles: pd.Series,
    tiers: Sequence[TierSpec] = DEFAULT_TIERS,
) -> tuple[pd.Series, pd.Series]:
    """(tier_number, label) per percentile; NaN percentile -> NaN/None."""
    highs = np.asarray([tier.high for tier in tiers], dtype=float)
    labels = [tier.label for tier in tiers]
    arr = percentiles.to_numpy(dtype=float)
    index = np.searchsorted(highs, arr, side="right")  # 0-based tier
    # percentile == 1.0 falls into the last (closed) tier.
    index = np.minimum(index, len(tiers) - 1)
    numbers = index.astype(float) + 1.0
    numbers = np.where(np.isnan(arr), np.nan, numbers)
    label_array = np.array(
        [labels[i] if 0 <= i < len(labels) else "" for i in index], dtype=object
    )
    label_array = np.where(np.isnan(arr), None, label_array)
    return (
        pd.Series(numbers, index=percentiles.index, dtype=float),
        pd.Series(label_array, index=percentiles.index, dtype=object),
    )


def add_rating_features(
    frame: pd.DataFrame,
    distribution_rows: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    tiers: Sequence[TierSpec] = DEFAULT_TIERS,
    inplace: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Add ``<metric>_value/_percentile/_zscore/_tier`` columns to *frame*.

    Returns ``(augmented_frame, added_columns)``. Metrics whose source column
    is missing, and contexts without a stored distribution, produce NaN.
    """
    out = frame if inplace else frame.copy()
    if out.empty:
        return out, []
    if "competition" not in out.columns:
        out["competition"] = "RS"
    lookup = build_distribution_lookup(distribution_rows)
    league_upper = out["league_key"].astype(str).str.upper()
    season_num = pd.to_numeric(out["season"], errors="coerce")
    competition_upper = out["competition"].astype(str).str.upper()

    added: list[str] = []
    for metric in metrics:
        definition = METRIC_DEFINITIONS.get(str(metric).strip().upper())
        if definition is None:
            continue
        source_column = definition.source_column
        if source_column not in out.columns:
            continue
        key_metric = metric.upper()
        value_column = f"{key_metric}{VALUE_SUFFIX}"
        pct_column = f"{key_metric}{PERCENTILE_SUFFIX}"
        z_column = f"{key_metric}{ZSCORE_SUFFIX}"
        tier_column = f"{key_metric}{TIER_SUFFIX}"

        out[value_column] = pd.to_numeric(out[source_column], errors="coerce")
        pct = pd.Series(np.nan, index=out.index, dtype=float)
        z = pd.Series(np.nan, index=out.index, dtype=float)
        for (metric_key, league_key, season, competition), dist in lookup.items():
            if metric_key != key_metric:
                continue
            mask = (
                (league_upper == league_key)
                & (season_num == float(season))
                & (competition_upper == competition)
            )
            if not mask.any():
                continue
            pct.loc[mask] = percentile_series(dist, out.loc[mask, source_column])
            z.loc[mask] = zscore_series(dist, out.loc[mask, source_column])
        if definition.direction == "lower_better":
            pct = 1.0 - pct  # low raw value -> high percentile (matches engine)
        out[pct_column] = pct
        out[z_column] = z
        tier_numbers, _ = tier_series(pct, tiers)
        out[tier_column] = tier_numbers
        added.extend([value_column, pct_column, z_column, tier_column])
    return out, added


def player_rating_snapshot(
    frame: pd.DataFrame,
    distribution_rows: Iterable[Mapping[str, Any]],
    *,
    player_global_id: str,
    player_id: Optional[Any] = None,
    league: str,
    season: Any,
    phase: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, dict[str, Any]]:
    """Per-metric rating snapshot for one player/context (Chat-friendly).

    Returns ``{metric: {value, percentile, zscore, tier, label, quality,
    population_source, fallback_used, distribution_version}}`` — raw values
    are always reported, rating fields are None when no distribution exists.

    Matching: by the ``player_global_id`` column when present, otherwise by
    ``player_id`` (the API frame exposes only the internal id).
    """
    frame = frame.copy()
    if "competition" not in frame.columns:
        frame["competition"] = "RS"
    league_upper = frame["league_key"].astype(str).str.upper()
    season_num = pd.to_numeric(frame["season"], errors="coerce")
    competition_upper = frame["competition"].astype(str).str.upper()
    season_year = int(str(season).split("-")[0])

    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        definition = METRIC_DEFINITIONS.get(metric.upper())
        if definition is None:
            continue
        mask = (
            (league_upper == str(league).upper())
            & (season_num == float(season_year))
            & (competition_upper == str(phase).upper())
        )
        if "player_global_id" in frame.columns:
            mask &= frame["player_global_id"].astype(str) == str(player_global_id)
        elif player_id is not None:
            mask &= frame["player_id"].astype(str) == str(player_id)
        else:
            mask &= frame["player_id"].astype(str) == str(player_global_id)
        rows = frame.loc[mask]
        if rows.empty:
            out[metric.upper()] = {"value": None, "reason": "no_row_for_context"}
            continue
        value = rows.iloc[0].get(definition.source_column)
        if value is None or pd.isna(value):
            out[metric.upper()] = {"value": None, "reason": "no_value"}
            continue

        dist_key = (
            metric.upper(),
            str(league).upper(),
            season_year,
            str(phase).upper(),
        )
        lookup = build_distribution_lookup(distribution_rows)
        dist = lookup.get(dist_key)
        if dist is None:
            out[metric.upper()] = {
                "value": float(value),
                "percentile": None,
                "zscore": None,
                "tier": None,
                "label": None,
                "quality": "insufficient",
                "population_source": "none",
                "fallback_used": False,
                "distribution_version": "",
                "reason": "no_distribution_for_context",
            }
            continue

        from basketball_ai.metric_rating.rating import MetricRatingEngine

        engine = MetricRatingEngine(
            distribution_version=dist.distribution_version,
        )
        rating = engine.rate_from_distribution(metric, float(value), dist)
        payload = rating.to_dict(rounded=True)
        out[metric.upper()] = {
            key: payload[key]
            for key in (
                "value", "percentile", "zscore", "tier", "label", "quality",
                "population_source", "fallback_used", "distribution_version",
            )
        }
    return out

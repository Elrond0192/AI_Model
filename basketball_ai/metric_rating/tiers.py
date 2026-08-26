"""Percentile tier table for the Metric Rating Engine.

Tiers are **percentile** thresholds, not raw metric thresholds, and are
configured centrally. Ranges are half-open ``[low, high)`` on the percentile
fraction domain; the last tier includes ``1.0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class TierSpec:
    """One tier: percentile fraction in ``[low, high)`` (last tier closed)."""

    label: str
    low: float
    high: float


DEFAULT_TIERS: tuple[TierSpec, ...] = (
    TierSpec("Very Poor", 0.00, 0.10),
    TierSpec("Poor", 0.10, 0.25),
    TierSpec("Below Average", 0.25, 0.40),
    TierSpec("Average", 0.40, 0.60),
    TierSpec("Good", 0.60, 0.75),
    TierSpec("Very Good", 0.75, 0.90),
    TierSpec("Elite", 0.90, 0.975),
    TierSpec("Superstar", 0.975, 1.00),
)


def validate_tiers(tiers: Sequence[TierSpec]) -> None:
    """Raise ``ValueError`` when the tier table is not a valid partition of
    ``[0, 1]``: non-empty, sorted, contiguous, starting at 0, ending at 1."""
    if not tiers:
        raise ValueError("tiers must not be empty")
    previous_high = 0.0
    for index, tier in enumerate(tiers):
        if tier.high <= tier.low:
            raise ValueError(
                f"tier[{index}] ({tier.label!r}) has invalid range "
                f"[{tier.low}, {tier.high})"
            )
        if abs(tier.low - previous_high) > 1e-9:
            raise ValueError(
                f"tiers are not contiguous at tier[{index}] ({tier.label!r}): "
                f"expected low={previous_high}, got {tier.low}"
            )
        previous_high = tier.high
    if abs(previous_high - 1.0) > 1e-9:
        raise ValueError(
            f"tiers must end at 1.0, got {previous_high}"
        )


def tier_for_percentile(
    percentile: Optional[float],
    tiers: Sequence[TierSpec] = DEFAULT_TIERS,
) -> Optional[tuple[int, str]]:
    """Map a percentile fraction in ``[0, 1]`` to ``(tier_index_1based, label)``.

    Returns ``None`` when the percentile is ``None`` (e.g. insufficient
    population). ``None`` input yields ``None`` so rating output can be
    computed without branching on quality.
    """
    if percentile is None:
        return None
    if not 0.0 <= float(percentile) <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {percentile!r}")
    validate_tiers(tiers)
    value = float(percentile)
    for index, tier in enumerate(tiers, start=1):
        if value < tier.high:
            return index, tier.label
    # value == 1.0 falls into the last (closed) tier.
    return len(tiers), tiers[-1].label


def tier_boundaries(tiers: Sequence[TierSpec] = DEFAULT_TIERS) -> list[float]:
    """Return the percentile fractions at which a new tier starts."""
    return [tier.low for tier in tiers]

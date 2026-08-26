"""Centralised metric definitions for the Metric Rating Engine.

A metric is fully described by a :class:`MetricDefinition`: its source column
in the ``"AI_Source"."PlayerCompetitionStats"`` contract, its direction, the
interpretable zero reference and optional above/below-replacement semantics.

The first version registers RAPTOR, LEBRON and VORP. Adding a metric is a
one-line registration here plus a value source mapping in the SQL contract —
never a new algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable definition of one rating metric."""

    metric: str
    source_column: str
    direction: str = "higher_better"  # "higher_better" | "lower_better"
    zero_reference: float = 0.0
    rating_enabled: bool = True
    replacement_reference: Optional[float] = None  # VORP: 0.0 (above/below replacement)
    description: str = ""
    value_label: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("higher_better", "lower_better"):
            raise ValueError(
                f"Invalid direction {self.direction!r} for {self.metric}; "
                "must be 'higher_better' or 'lower_better'"
            )

    def is_above_reference(self, value: float) -> bool:
        """True when *value* is at or above the replacement reference."""
        if self.replacement_reference is None:
            raise ValueError(
                f"{self.metric} has no replacement_reference configured"
            )
        return bool(value >= self.replacement_reference)

    def is_below_reference(self, value: float) -> bool:
        """True when *value* is strictly below the replacement reference."""
        if self.replacement_reference is None:
            raise ValueError(
                f"{self.metric} has no replacement_reference configured"
            )
        return bool(value < self.replacement_reference)


# ---------------------------------------------------------------------------
# Registered metrics (first version)
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "RAPTOR": MetricDefinition(
        metric="RAPTOR",
        source_column="raptor_total",
        direction="higher_better",
        zero_reference=0.0,
        description="RAPTOR total (FiveThirtyEight)",
        value_label="RAPTOR",
    ),
    "LEBRON": MetricDefinition(
        metric="LEBRON",
        source_column="lebron_total",
        direction="higher_better",
        zero_reference=0.0,
        description="LEBRON impact model",
        value_label="LEBRON",
    ),
    "VORP": MetricDefinition(
        metric="VORP",
        source_column="vorp",
        direction="higher_better",
        zero_reference=0.0,
        replacement_reference=0.0,
        description="Value Over Replacement Player",
        value_label="VORP",
    ),
}


def get_metric_definition(metric: str) -> MetricDefinition:
    """Return the registered definition for *metric* (case-insensitive)."""
    key = str(metric).strip().upper()
    try:
        return METRIC_DEFINITIONS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown metric {metric!r}; registered: {sorted(METRIC_DEFINITIONS)}"
        ) from exc

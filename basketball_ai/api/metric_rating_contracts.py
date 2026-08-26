"""FASE I — Metric Rating Engine API contracts (WordPress consumes only)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class MetricRatingRateRequestV2(BaseModel):
    """Rate a raw advanced-metric value against the stored population.

    ``season`` accepts an integer start year (``2025``) or a label like
    ``"2025-26"``; ``phase`` is the competition code (``RS``, ``PO``, ``TOT``,
    ``CUP``, ``SUPERCUP``, …).
    """

    metric: str
    value: float
    league: str
    season: Union[int, str]
    phase: str


class MetricRatingRateResponseV2(BaseModel):
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
    distribution_version: str
    rating_version: str
    above_reference: Optional[bool] = None
    below_reference: Optional[bool] = None
    warnings: list[str] = Field(default_factory=list)


class MetricRatingPlayerRequestV2(BaseModel):
    """Per-metric rating snapshot for one player in a context (Chat use)."""

    player_global_id: str
    league: str
    season: Union[int, str]
    phase: str
    metrics: Optional[list[str]] = None  # default RAPTOR, LEBRON, VORP


class MetricRatingPlayerResponseV2(BaseModel):
    model_run_id: str
    model_version: str
    feature_version: str
    data_cutoff: str
    player_global_id: str
    league: str
    season: int
    phase: str
    ratings: dict[str, Any]
    generated_at: datetime

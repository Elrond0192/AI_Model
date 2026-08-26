"""Population builder for the Metric Rating Engine.

Selects the set of players a metric value is compared against. The population
is contextualized by metric + league + season + phase + population type, with
configurable qualification rules (existing columns only: ``games_played`` and
``minutes_per_game``) and an explicit, ordered fallback chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Default compatible competition families used by the fallback chain.
# A request for phase P also accepts the members of family(P) when the exact
# phase has an insufficient population.
DEFAULT_COMPETITION_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "RS": ("RS", "TOT"),
    "PO": ("PO", "TOT"),
    "CUP": ("CUP", "TOT"),
    "SUPERCUP": ("SUPERCUP", "TOT"),
    "TOT": ("TOT", "RS", "PO", "CUP", "SUPERCUP"),
}

# Fallback order, least to most permissive (FASE 8).
DEFAULT_FALLBACK_ORDER: Tuple[str, ...] = (
    "exact",
    "league_multi_season",
    "competition_family",
    "global",
)

# Canonical column names in "AI_Source"."PlayerCompetitionStats".
LEAGUE_COLUMN = "league_key"
SEASON_COLUMN = "season"
COMPETITION_COLUMN = "competition"
GAMES_COLUMN = "games_played"
MINUTES_COLUMN = "minutes_per_game"

_REQUIRED_COLUMNS = (
    LEAGUE_COLUMN,
    SEASON_COLUMN,
    COMPETITION_COLUMN,
    GAMES_COLUMN,
    MINUTES_COLUMN,
)


@dataclass(frozen=True)
class PopulationConfig:
    """Configurable population rules. Thresholds are calibrated on real data
    (FASE D POC) and are never invented columns."""

    min_games: int = 10
    min_minutes_per_game: float = 0.0
    min_samples: int = 50
    population_type: str = "qualified_players"
    exclude_non_finite: bool = True
    competition_families: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_COMPETITION_FAMILIES)
    )
    fallback_order: Tuple[str, ...] = DEFAULT_FALLBACK_ORDER


@dataclass(frozen=True)
class PopulationResult:
    """The winning population and its provenance."""

    values: tuple[float, ...]
    sample_size: int
    population_source: str  # exact | league_multi_season | competition_family | global
    fallback_used: bool
    population_type: str
    population_key: Tuple[str, ...]
    filters: Tuple[str, ...] = ()


def _to_year(season: Any) -> int:
    """Normalize ``2025`` / ``"2025"`` / ``"2025-26"`` to the start year 2025."""
    if season is None or (isinstance(season, float) and np.isnan(season)):
        raise ValueError("season is null")
    text = str(season).strip()
    return int(text.split("-")[0])


def _metric_values(
    frame: pd.DataFrame,
    source_column: str,
    *,
    exclude_non_finite: bool,
) -> pd.Series:
    if source_column not in frame.columns:
        raise ValueError(
            f"Missing metric source column {source_column!r} in the population "
            "frame; the 'AI_Source' contract must expose it."
        )
    values = frame[source_column]
    if exclude_non_finite:
        numeric = pd.to_numeric(values, errors="coerce")
        values = numeric[np.isfinite(numeric)]
    else:
        values = pd.to_numeric(values, errors="coerce")
    return values.dropna()


class PopulationBuilder:
    """Build contextual populations from a canonical stats frame."""

    def __init__(self, config: Optional[PopulationConfig] = None) -> None:
        self.config = config or PopulationConfig()

    def build(
        self,
        frame: pd.DataFrame,
        *,
        source_column: str,
        league: str,
        season: Any,
        phase: str,
        fallback: bool = True,
    ) -> PopulationResult:
        """Return the first population in the fallback chain that reaches
        ``min_samples``; otherwise the most permissive candidate (with its
        actual sample size, so the engine can report ``insufficient``).

        With ``fallback=False`` only the exact context (first level) is
        evaluated — used by validation/POC reports that must show the exact
        population without relaxation."""
        if frame is None or frame.empty:
            return PopulationResult(
                values=(), sample_size=0, population_source="none",
                fallback_used=False, population_type=self.config.population_type,
                population_key=(), filters=(),
            )
        missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Population frame is missing canonical columns: {sorted(missing)}"
            )
        year = _to_year(season)
        league_key = str(league).strip().upper()
        phase = str(phase).strip().upper()

        metrics = _metric_values(
            frame,
            source_column,
            exclude_non_finite=self.config.exclude_non_finite,
        )
        rows = frame.loc[metrics.index].copy()
        rows["_metric"] = metrics

        levels = (
            self.config.fallback_order
            if fallback
            else self.config.fallback_order[:1]
        )
        candidates: list[PopulationResult] = []
        for level in levels:
            candidates.append(
                self._candidate(rows, source_column, level, league_key, year, phase)
            )
        for result in candidates:
            if result.sample_size >= self.config.min_samples:
                return result
        # No level reached min_samples: report the best effort (most permissive).
        return candidates[-1]

    # ------------------------------------------------------------------

    def _candidate(
        self,
        rows: pd.DataFrame,
        source_column: str,
        level: str,
        league: str,
        season: int,
        phase: str,
    ) -> PopulationResult:
        mask = self._level_mask(rows, level, league, season, phase)
        filtered = rows[mask]
        if self.config.population_type == "qualified_players":
            qualified = (
                (filtered[GAMES_COLUMN].fillna(0) >= self.config.min_games)
                & (
                    filtered[MINUTES_COLUMN].fillna(0)
                    >= self.config.min_minutes_per_game
                )
            )
            filtered = filtered[qualified]
        values = filtered["_metric"].dropna()
        if not isinstance(values.dtype.type, (np.floating, np.integer)):
            values = pd.to_numeric(values, errors="coerce").dropna()
        clean = tuple(float(value) for value in values if np.isfinite(value))
        return PopulationResult(
            values=clean,
            sample_size=len(clean),
            population_source=level,
            fallback_used=level != "exact",
            population_type=self.config.population_type,
            population_key=(source_column, league, str(season), phase),
            filters=self._level_filters(level, league, season, phase),
        )

    def _level_mask(
        self,
        rows: pd.DataFrame,
        level: str,
        league: str,
        season: int,
        phase: str,
    ) -> pd.Series:
        season_year = pd.to_numeric(
            rows[SEASON_COLUMN].map(_to_year), errors="coerce"
        )
        if level == "exact":
            return (
                (rows[LEAGUE_COLUMN].astype(str).str.upper() == league)
                & (season_year == season)
                & (rows[COMPETITION_COLUMN].astype(str).str.upper() == phase)
            )
        if level == "league_multi_season":
            return (
                (rows[LEAGUE_COLUMN].astype(str).str.upper() == league)
                & (rows[COMPETITION_COLUMN].astype(str).str.upper() == phase)
            )
        if level == "competition_family":
            family = self.config.competition_families.get(phase, (phase,))
            return (
                (rows[LEAGUE_COLUMN].astype(str).str.upper() == league)
                & (season_year == season)
                & (rows[COMPETITION_COLUMN].astype(str).str.upper().isin(family))
            )
        if level == "global":
            return pd.Series(True, index=rows.index)
        raise ValueError(f"Unknown fallback level {level!r}")

    @staticmethod
    def _level_filters(level: str, league: str, season: int, phase: str) -> Tuple[str, ...]:
        if level == "exact":
            return (f"league={league}", f"season={season}", f"phase={phase}")
        if level == "league_multi_season":
            return (f"league={league}", f"phase={phase}", "all_seasons")
        if level == "competition_family":
            return (f"league={league}", f"season={season}", "phase_family")
        if level == "global":
            return ("all_leagues", "all_seasons", "all_phases")
        return (level,)

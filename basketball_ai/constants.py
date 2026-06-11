"""Shared domain constants and pure helpers used across basketball_ai modules.

Centralising these definitions avoids duplication between
``features/player_features.py``, ``models/performance_model.py``, and
``models/age_curve.py``.  This module has **no module-level internal imports**
so it can be imported by any other module without risk of circular
dependencies.  Functions in this module may use *lazy* internal imports
(inside the function body) where needed to avoid circular dependency issues at
import time.
"""
from __future__ import annotations

import re
from typing import Dict

# ---------------------------------------------------------------------------
# Positional peak ages – empirical approximations used as a default prior
# when fitting the age curve to historical data is not possible.
# ---------------------------------------------------------------------------

POSITIONAL_PEAK_AGES: Dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}

# ---------------------------------------------------------------------------
# League maximum regular-season game counts – used to normalise
# ``durability_score`` correctly.  NBA = 82, European leagues play 22-46.
# Key is the league name exactly as generated in ``data/generator.py``.
# ---------------------------------------------------------------------------

LEAGUE_MAX_GAMES_BY_NAME: Dict[str, int] = {
    "NBA":           82,
    "EuroLeague":    34,
    "BCL":           32,
    "ACB":           32,
    "Bundesliga":    32,
    "Lega Basket":   30,
    "BSL":           34,
    "LNB Pro A":     32,
    "NBL":           28,
    "VTB League":    30,
    "Adriatic":      28,
    "Liga ACB B":    30,
    "Pro B":         30,
    "Serie A2":      30,
    "BSL B":         30,
    "Pro B France":  30,
    "NBL1":          22,
    "FIBA EuroCup":  22,
    "CBA":           46,
    "Liga Nacional": 30,
}

#: Default when a league is not found in LEAGUE_MAX_GAMES_BY_NAME.
LEAGUE_MAX_GAMES_DEFAULT: int = 82


# ---------------------------------------------------------------------------
# Pure position helpers
# ---------------------------------------------------------------------------

def _primary_pos(pos: str) -> str:
    """Return the first component of a (potentially hybrid) position string.

    Examples::

        _primary_pos("PG")     → "PG"
        _primary_pos("PF/C")   → "PF"
        _primary_pos("SG/PF")  → "SG"
    """
    return pos.split("/")[0]


def _peak_age(pos: str) -> int:
    """Return the positional peak age for *pos*.

    Falls back to the primary position if the hybrid is not in the table,
    then to 26 as a global default.
    """
    return POSITIONAL_PEAK_AGES.get(pos, POSITIONAL_PEAK_AGES.get(_primary_pos(pos), 26))


def get_peak_age(pos: str) -> int:
    """Return best available peak age for *pos*: fitted > static prior.

    Uses a lazy import of ``age_curve._fitted_peak_ages`` to avoid circular
    dependencies (constants.py must remain import-free of internal modules).
    Falls back to the hardcoded prior when no empirical fit is available.
    """
    try:
        from basketball_ai.models.age_curve import _fitted_peak_ages  # lazy import

        primary = _primary_pos(pos)
        fitted = _fitted_peak_ages.get(pos)
        if fitted is None:
            fitted = _fitted_peak_ages.get(primary)
        if fitted is not None:
            return int(round(fitted))
    except ImportError:
        pass
    return _peak_age(pos)


# ---------------------------------------------------------------------------
# Supported leagues and seasons – canonical registry (no DB lookup required)
# ---------------------------------------------------------------------------

#: All supported league codes. Used to validate dynamic table names.
SUPPORTED_LEAGUES: tuple[str, ...] = (
    "ITA1", "ITA2", "GRC1", "GRC2", "ESP1", "ESP2",
    "DEU1", "FRA1", "TUR1", "SRB1", "AUS1",
)

#: All supported season years (integer) in chronological order.
SUPPORTED_SEASONS: tuple[int, ...] = (
    2019, 2020, 2021, 2022, 2023, 2024, 2025,
)

#: Regex for safe SQL table-name identifiers (alphanumeric + underscore only).
_SAFE_IDENTIFIER_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,127}$')


def is_safe_identifier(name: str) -> bool:
    """Return True if *name* is safe to embed as a SQL identifier."""
    return bool(_SAFE_IDENTIFIER_RE.match(name))

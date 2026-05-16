"""Shared domain constants and pure helpers used across basketball_ai modules.

Centralising these definitions avoids duplication between
``features/player_features.py``, ``models/performance_model.py``, and
``models/age_curve.py``.  This module has **no internal imports** so it
can be imported by any other module without risk of circular dependencies.
"""
from __future__ import annotations

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

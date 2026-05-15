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

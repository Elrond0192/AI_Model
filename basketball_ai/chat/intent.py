"""Intent detection for the basketball chat engine.

Each incoming message is mapped to one of the supported intents using
ordered regular-expression patterns (most-specific first).
"""
from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    PREDICT      = "predict"        # "How good is X at Y?"
    TRAJECTORY   = "trajectory"     # "What's X's career arc?"
    PEAK         = "peak"           # "When will X peak?"
    TRANSFER     = "transfer"       # "What if X moved from A to B?"
    LINEUP       = "lineup"         # "Player X at team Y with lineup [A, B, C, D]"
    COMPARE      = "compare"        # "Compare X across teams"
    BEST_TEAMS   = "best_teams"     # "Best teams for X?"
    BEST_PLAYERS = "best_players"   # "Best players for team Y?"
    TEAMMATES    = "teammates"      # "What if X had elite teammates?"
    CLUTCH       = "clutch"         # "How is X in clutch/pressure situations?"
    MARKET_VALUE = "market_value"   # "What is X's market value?"
    INDIVIDUAL   = "individual"     # "What are X's individual stats?"
    ROLE_FIT     = "role_fit"       # "Is X good as a false 9 / point forward / ...?"
    HELP         = "help"           # "What can you do?"
    UNKNOWN      = "unknown"


# Ordered list: most-specific patterns checked first.
# Each entry is (Intent, compiled-regex).
_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (
        Intent.TRANSFER,
        re.compile(
            r"\btransfer|traded?|moved?\s+from\b|from\s+\S+\s+to\s",
            re.IGNORECASE,
        ),
    ),
    # LINEUP must come BEFORE TEAMMATES to capture "with [named players]" first
    (
        Intent.LINEUP,
        re.compile(
            r"\blineup\b|quintett[oi]\b|starting[\s_]five\b|formazione\b"
            r"|roster\s+with\b"
            r"|composto\s+d[ai]\b|composed?\s+(of|by|with)\b"
            r"|\bwith\b.+\band\b.+\b(at|in|on)\b"   # "with A and B at team"
            r"|\b(at|in|on)\b.+\bwith\b.+\band\b",   # "at team with A and B"
            re.IGNORECASE,
        ),
    ),
    (
        Intent.TEAMMATES,
        re.compile(
            r"\bteammate|with\s+(better|worse|elite|weaker|stronger|star|great)\s+team",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CLUTCH,
        re.compile(
            r"\bclutch\b|pressure\s+situation|decisive\s+moment|crunch\s+time"
            r"|final\s+(seconds?|minutes?|quarter)|ultimo\s+quarto"
            r"|situazion[ie]\s+decisive|nei\s+moment[io]\s+(important|critic|chiave|crucial)"
            r"|playoffs?\s+performer|big\s+game",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.MARKET_VALUE,
        re.compile(
            r"\bmarket\s+value|valore\s+di\s+mercato|worth|stipendio|salary"
            r"|quanto\s+(vale|costerebbe|cost[ae])|price\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.ROLE_FIT,
        re.compile(
            r"\brole\b|ruolo\b|false\s+9|punto\s+forward|stretch\s+(big|4|five)"
            r"|small\s+ball|lead\s+guard|spacer|dual[\s-]threat"
            r"|adatto\b|si\s+adatta|pu[oò]\s+fare\s+(il|da)\b|fit\s+(as|for)\s+(a|the)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.INDIVIDUAL,
        re.compile(
            r"\bindividual|statistich[ae]\s+individual|da\s+solo|isolation"
            r"|solo\s+stats?|player\s+profile|profilo\s+giocatore"
            r"|what\s+are\s+\w+[\''s]*\s+stats|dimmi\s+di\b|raccontami\s+di\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.COMPARE,
        re.compile(
            r"\bcompar|vs\.?\b|versus|which\s+team\s+is\s+best|where\s+should\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.PEAK,
        re.compile(
            r"\bpeak|prime\b|career\s+(high|best)|best\s+age|when.{1,25}best\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.TRAJECTORY,
        re.compile(
            r"\btrajectory|over\s+time|career\s+arc|age\s+curve|evolv|age\s+\d+",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.BEST_TEAMS,
        re.compile(
            r"\bbest\s+team|top\s+team|which\s+team|where\s+(would|will|should)\b|best\s+fit",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.BEST_PLAYERS,
        re.compile(
            r"\bbest\s+player|who\s+(fits?|would|will)\b|top\s+player",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.PREDICT,
        re.compile(
            r"\bpredict|rating\b|how\s+good|how\s+well|perform|rate\b|score\b"
            r"|\bquanto\s+segner[aà]|\bquante\s+partite|\bmetriche",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.HELP,
        re.compile(
            r"\bhelp\b|what\s+can\s+you|how\s+do\s+i\b|what\s+do\s+you\b|commands?\b",
            re.IGNORECASE,
        ),
    ),
]


def detect_intent(text: str) -> Intent:
    """Return the most specific matching Intent for *text*."""
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent
    return Intent.UNKNOWN

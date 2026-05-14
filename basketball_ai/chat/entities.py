"""Entity extraction for the basketball chat engine.

Extracts player names, team names and numeric values from free-text using
a combination of:
  1. Case-insensitive substring matching (fast, exact).
  2. difflib fuzzy matching (handles typos / partial names).

No external NLP libraries are required.
"""
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    return s.lower().strip()


# Minimum similarity ratio for fuzzy name matching (0–1).
# Higher values reduce false positives; 0.72 balances recall vs. precision
# for typical team/player name typos (e.g. "Panathinaikos" vs "Panathinaikos BC").
_FUZZY_CUTOFF = 0.72


def _token_ngrams(text: str, max_n: int = 3) -> List[str]:
    """Return all word n-grams (1..max_n) from *text* as lowercase strings.

    Used by fuzzy-matching fallbacks so that a long message like
    "Diego Flaccadori statistiche nel Panathinaikos" yields individual
    token windows that can fuzzy-match a short name like "Panathinaikos BC".
    """
    tokens = text.split()
    ngrams: List[str] = []
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def find_player(
    text: str,
    player_dict: Dict[int, dict],
) -> Optional[Tuple[int, str]]:
    """Return ``(player_id, player_name)`` for the best name match, or None.

    Search order:
    1. Exact case-insensitive substring.
    2. difflib fuzzy match against individual token n-grams of the message
       (handles typos and partial names without being confused by unrelated
       words in long messages).
    """
    names: Dict[int, str] = {pid: str(d.get("name", "")) for pid, d in player_dict.items()}
    low_text = _norm(text)

    # 1. Substring
    for pid, name in names.items():
        if _norm(name) and (_norm(name) in low_text or low_text in _norm(name)):
            return pid, name

    # 2. Fuzzy — try each token n-gram as the candidate query
    name_list = [n for n in names.values() if n]
    norm_to_orig = {_norm(n): n for n in name_list}
    for candidate in _token_ngrams(low_text, max_n=4):
        matches = get_close_matches(candidate, list(norm_to_orig.keys()), n=1, cutoff=_FUZZY_CUTOFF)
        if matches:
            orig_name = norm_to_orig[matches[0]]
            for pid, name in names.items():
                if name == orig_name:
                    return pid, name

    return None


def find_team(
    text: str,
    team_dict: Dict[int, dict],
) -> Optional[Tuple[int, str]]:
    """Return ``(team_id, canonical_name)`` for the best name match, or None.

    Checks both ``name`` and ``short_name`` fields so that "Panathinaikos"
    matches a team stored as "Panathinaikos BC" (short_name = "Panathinaikos").
    Falls back to token n-gram fuzzy matching for typos and partial mentions.
    """
    low_text = _norm(text)

    # Build a list of (team_id, canonical_name, norm_alias) for all aliases
    aliases: List[Tuple[int, str, str]] = []
    for tid, d in team_dict.items():
        name = str(d.get("name", ""))
        short = str(d.get("short_name", "") or "")
        canonical = name  # always use full name as the returned display value
        if _norm(name):
            aliases.append((tid, canonical, _norm(name)))
        if _norm(short) and _norm(short) != _norm(name):
            aliases.append((tid, canonical, _norm(short)))

    # 1. Substring — check all aliases
    for tid, canonical, alias_norm in aliases:
        if alias_norm in low_text or low_text in alias_norm:
            return tid, canonical

    # 2. Fuzzy — try each token n-gram against all alias strings.
    # Build a list-of-entries per normalized alias to avoid collisions when
    # two teams share the same normalized alias string (e.g. both have
    # short_name "BC"); the first match wins in iteration order.
    alias_norms = [a for _, _, a in aliases]
    norm_to_entries: Dict[str, List[Tuple[int, str]]] = {}
    for tid, c, a in aliases:
        norm_to_entries.setdefault(a, []).append((tid, c))
    for candidate in _token_ngrams(low_text, max_n=3):
        matches = get_close_matches(candidate, alias_norms, n=1, cutoff=_FUZZY_CUTOFF)
        if matches:
            tid, canonical = norm_to_entries[matches[0]][0]
            return tid, canonical

    return None


def find_all_teams(
    text: str,
    team_dict: Dict[int, dict],
) -> List[Tuple[int, str]]:
    """Return all team references found in *text* (used for transfer detection).

    Checks both ``name`` and ``short_name`` so partial names like
    "Panathinaikos" correctly match "Panathinaikos BC".
    """
    low_text = _norm(text)
    found: List[Tuple[int, str]] = []
    seen: set = set()
    for tid, d in team_dict.items():
        name = str(d.get("name", ""))
        short = str(d.get("short_name", "") or "")
        norm_name = _norm(name)
        norm_short = _norm(short)
        matched = (
            (norm_name and norm_name in low_text)
            or (norm_short and norm_short != norm_name and norm_short in low_text)
        )
        if matched and tid not in seen:
            found.append((tid, name))
            seen.add(tid)
    return found


def find_all_players(
    text: str,
    player_dict: Dict[int, dict],
    max_n: int = 6,
) -> List[Tuple[int, str]]:
    """Find ALL player name mentions in *text*, ordered by their position in the string.

    Used for lineup/quintetto intent where the message contains several player names,
    e.g. *"Player X in Team Y with Player A, Player B, Player C and Player D"*.

    Returns up to *max_n* ``(player_id, player_name)`` pairs, de-duplicated.
    """
    names: Dict[int, str] = {pid: str(d.get("name", "")) for pid, d in player_dict.items()}
    low_text = _norm(text)

    # Each entry: (player_id, player_name, first_occurrence_index)
    found: list[tuple[int, str, int]] = []
    seen_pids: set[int] = set()

    for pid, name in names.items():
        if not _norm(name):
            continue
        idx = low_text.find(_norm(name))
        if idx >= 0 and pid not in seen_pids:
            found.append((pid, name, idx))
            seen_pids.add(pid)

    found.sort(key=lambda x: x[2])
    return [(pid, name) for pid, name, _ in found[:max_n]]


def extract_number(text: str) -> Optional[float]:
    """Return the first numeric value found in *text*, or None."""
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None

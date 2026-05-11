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


def find_player(
    text: str,
    player_dict: Dict[int, dict],
) -> Optional[Tuple[int, str]]:
    """Return ``(player_id, player_name)`` for the best name match, or None.

    Search order:
    1. Exact case-insensitive substring.
    2. difflib fuzzy match (cutoff 0.60).
    """
    names: Dict[int, str] = {pid: str(d.get("name", "")) for pid, d in player_dict.items()}
    low_text = _norm(text)

    # 1. Substring
    for pid, name in names.items():
        if _norm(name) and (_norm(name) in low_text or low_text in _norm(name)):
            return pid, name

    # 2. Fuzzy
    name_list = [n for n in names.values() if n]
    matches = get_close_matches(text, name_list, n=1, cutoff=0.60)
    if matches:
        for pid, name in names.items():
            if name == matches[0]:
                return pid, name

    return None


def find_team(
    text: str,
    team_dict: Dict[int, dict],
) -> Optional[Tuple[int, str]]:
    """Return ``(team_id, team_name)`` for the best name match, or None."""
    names: Dict[int, str] = {tid: str(d.get("name", "")) for tid, d in team_dict.items()}
    low_text = _norm(text)

    for tid, name in names.items():
        if _norm(name) and (_norm(name) in low_text or low_text in _norm(name)):
            return tid, name

    name_list = [n for n in names.values() if n]
    matches = get_close_matches(text, name_list, n=1, cutoff=0.60)
    if matches:
        for tid, name in names.items():
            if name == matches[0]:
                return tid, name

    return None


def find_all_teams(
    text: str,
    team_dict: Dict[int, dict],
) -> List[Tuple[int, str]]:
    """Return all team references found in *text* (used for transfer detection)."""
    names: Dict[int, str] = {tid: str(d.get("name", "")) for tid, d in team_dict.items()}
    low_text = _norm(text)
    found: List[Tuple[int, str]] = []
    for tid, name in names.items():
        if _norm(name) and _norm(name) in low_text:
            found.append((tid, name))
    return found


def extract_number(text: str) -> Optional[float]:
    """Return the first numeric value found in *text*, or None."""
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None

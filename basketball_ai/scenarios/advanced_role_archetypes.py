"""Mapping from HoopmetricsEngine's authoritative advanced roles to scouting archetypes.

The three role columns are calculated upstream by HoopmetricsEngine and exposed
through AI_Source. This module maps those existing labels to scouting concepts;
it does not recalculate or overwrite the upstream classifications.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Offensive labels emitted by HoopmetricsEngine.classify_all_roles().
OFFENSIVE_ROLE_ARCHETYPES: dict[str, tuple[str, ...]] = {
    "PRIMARY BALL HANDLER": ("playmaker", "primary_creator"),
    "PRIMARY SHOT CREATOR": ("primary_creator", "isolation_scorer", "scorer"),
    "PASS-FIRST PLAYER": ("facilitator", "connector"),
    "SECONDARY BALL HANDLER": ("secondary_creator", "combo_guard"),
    "SECONDARY SHOT CREATOR": ("secondary_creator", "scorer"),
    "SLASHER": ("slasher", "rim_finisher"),
    "SCORER": ("scorer", "volume_scorer"),
    "PURE SHOOTER": ("shooter", "catch_and_shoot"),
    "SPOT-UP SHOOTER": ("shooter", "low_usage_shooter"),
    "DOUBLE THREAT BIG": ("playmaking_big", "stretch_big"),
    "POST-UP CREATOR": ("post_scorer", "playmaking_big"),
    "ROLL & POP": ("roll_man", "stretch_big"),
    "POST & 3": ("post_big", "stretch_big"),
    "PICK & ROLLER": ("roll_man", "rim_runner"),
    "POST-UP SCORER": ("post_scorer", "post_big"),
    "STRETCH BIG": ("stretch_big", "low_usage_shooter"),
    "INTERIOR SCORER": ("rim_finisher", "post_big"),
    "OFF-BALL PLAYER": ("off_ball_scorer", "connector"),
    "MARGINAL OFFENSIVE ROLE": ("high_impact_low_usage",),
}

DEFENSIVE_ROLE_ARCHETYPES: dict[str, tuple[str, ...]] = {
    "RIM PROTECTOR": ("rim_protector",),
    "INTERIOR DEFENDER": ("team_defender", "versatile_defender"),
    "AVERAGE INTERIOR DEFENDER": ("team_defender",),
    "DEFENSIVE REBOUNDER": ("defensive_rebounder", "rebounder"),
    "SWITCH DEFENDER": ("switchable_defender", "versatile_defender"),
    "PERIMETER DEFENDER": ("perimeter_defender", "point_of_attack_defender"),
    "AVERAGE PERIMETER DEFENDER": ("perimeter_defender",),
    "NON-DEFENDER": (),
}

COMBINED_ROLE_ARCHETYPES: dict[str, tuple[str, ...]] = {
    "ELITE 2-WAY PG": ("playmaker", "two_way"),
    "TWO-WAY PG": ("playmaker", "two_way"),
    "OFFENSIVE PG": ("playmaker",),
    "ELITE 2-WAY SCORER": ("scorer", "two_way"),
    "TWO-WAY SCORER": ("scorer", "two_way"),
    "ISO SCORER": ("isolation_scorer", "scorer"),
    "2-WAY FACILITATOR": ("facilitator", "two_way"),
    "FACILITATOR": ("facilitator",),
    "PASS-FIRST PG": ("facilitator", "playmaker"),
    "2-WAY 2nd CREATOR": ("secondary_creator", "two_way"),
    "SOLID 2nd CREATOR": ("secondary_creator",),
    "2nd CREATOR": ("secondary_creator",),
    "2-WAY 2nd SCORER": ("scorer", "two_way"),
    "SOLID 2nd SCORER": ("scorer",),
    "2nd SCORER": ("scorer",),
    "2-WAY SLASHER": ("slasher", "two_way"),
    "SLASHER": ("slasher",),
    "SCORING SLASHER": ("slasher", "scorer"),
    "SCORING SPECIALIST": ("scorer",),
    "OFF. SPECIALIST": ("scorer",),
    "ELITE 3&D": ("three_and_d", "two_way"),
    "3&D SHOOTER": ("three_and_d",),
    "3&D PLAYER": ("three_and_d",),
    "SPOT-UP 3&D": ("three_and_d",),
    "3PT SPECIALIST": ("shooter",),
    "SPOT-UP SHOOTER": ("shooter",),
    "UNICORN BIG": ("playmaking_big", "stretch_big", "two_way"),
    "ELITE MODERN BIG": ("stretch_big", "two_way"),
    "MODERN BIG": ("stretch_big",),
    "ELITE INTERIOR ANCHOR": ("post_big", "rim_protector", "two_way"),
    "2-WAY POST CREATOR": ("post_scorer", "two_way"),
    "POST CREATOR": ("post_scorer",),
    "2-WAY STRETCH BIG": ("stretch_big", "two_way"),
    "ROLL & POP BIG": ("roll_man", "stretch_big"),
    "ROLL & POP": ("roll_man", "stretch_big"),
    "2-WAY COMBO BIG": ("post_big", "stretch_big", "two_way"),
    "COMBO BIG": ("post_big", "stretch_big"),
    "ELITE VERTICAL BIG": ("rim_runner", "rim_protector", "two_way"),
    "2-WAY ROLL MAN": ("roll_man", "two_way"),
    "ROLL MAN": ("roll_man",),
    "2-WAY POST SCORER": ("post_scorer", "two_way"),
    "POST SCORER": ("post_scorer",),
    "3&D BIG": ("stretch_big", "three_and_d"),
    "TRADITIONAL CENTER": ("post_big", "rim_protector"),
    "2-WAY INT. BIG": ("post_big", "two_way"),
    "INTERIOR BIG": ("post_big",),
    "2-WAY ROLE PLAYER": ("high_impact_low_usage", "two_way"),
    "ROLE PLAYER": ("high_impact_low_usage",),
    "DEF SPECIALIST": ("team_defender", "high_impact_low_usage"),
    "FRINGE PLAYER": (),
}


def _normalise_role(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def map_advanced_roles(
    offensive_role: Any,
    defensive_role: Any,
    combined_role: Any,
) -> dict[str, Any]:
    """Return upstream roles unchanged plus mapped archetypes and mapping coverage."""
    offensive = _normalise_role(offensive_role)
    defensive = _normalise_role(defensive_role)
    combined = _normalise_role(combined_role)
    ordered: list[str] = []
    sources: dict[str, list[str]] = {}
    for source, role, mapping in (
        ("offensive", offensive, OFFENSIVE_ROLE_ARCHETYPES),
        ("defensive", defensive, DEFENSIVE_ROLE_ARCHETYPES),
        ("combined", combined, COMBINED_ROLE_ARCHETYPES),
    ):
        mapped = list(mapping.get(role, ()))
        sources[source] = mapped
        for archetype in mapped:
            if archetype not in ordered:
                ordered.append(archetype)
    available = [name for name, role in (
        ("offensive", offensive), ("defensive", defensive), ("combined", combined)
    ) if role]
    mapped_count = sum(bool(sources[name]) for name in available)
    return {
        "offensive": offensive_role if offensive else None,
        "defensive": defensive_role if defensive else None,
        "combined": combined_role if combined else None,
        "archetypes": ordered,
        "archetypes_by_dimension": sources,
        "source": "hoopmetrics_engine_advanced_roles",
        "mapping_coverage": round(mapped_count / len(available), 3) if available else 0.0,
        "unmapped_roles": [
            {"dimension": name, "role": role}
            for name, role in (("offensive", offensive), ("defensive", defensive), ("combined", combined))
            if role and not sources[name]
        ],
    }

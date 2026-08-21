"""Leakage-safe basketball player/team context feature engineering."""
from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from basketball_ai.features.team_features import get_style_position_compat
from basketball_ai.utils.helpers import normalize_id as _normalize_id


def compute_league_adaptation(
    from_tier: int,
    to_tier: int,
    same_country: bool = False,
) -> float:
    """Return a bounded adaptation multiplier between source/target leagues."""
    tier_diff = to_tier - from_tier
    if tier_diff < 0:
        factor = 1.0 - abs(tier_diff) * 0.06
    else:
        factor = 1.0 + tier_diff * 0.02
    if same_country:
        factor += 0.02
    return float(np.clip(factor, 0.70, 1.05))


def _season_year(value: Any) -> int | None:
    try:
        return int(str(value).split("-")[0])
    except (TypeError, ValueError, IndexError):
        return None


def _style_for_team(team_id: int, team_row: Dict[str, Any], data: Dict[str, Any]) -> str:
    """Use a style derived from the current snapshot, never a future cached label."""
    if data.get("_as_of_season") is None:
        return str(team_row.get("playing_style", "motion_offense") or "motion_offense")

    style_map = data.get("_historical_style_map")
    if style_map is None:
        from basketball_ai.data.loader import _derive_playing_style

        teams = data.get("teams", pd.DataFrame()).copy()
        _derive_playing_style(teams)
        style_map = {
            _normalize_id(row["id"]): str(row.get("playing_style", "motion_offense") or "motion_offense")
            for row in teams.to_dict("records")
            if pd.notna(row.get("id"))
        }
        data["_historical_style_map"] = style_map
    return str(style_map.get(_normalize_id(team_id), "motion_offense"))


def _team_roster_at_context(team_id: int, data: Dict[str, Any]) -> pd.DataFrame:
    rels = data.get("team_player_relations", pd.DataFrame())
    if rels.empty or "team_id" not in rels.columns:
        return rels.iloc[0:0]
    team_rels = rels[rels["team_id"].map(_normalize_id) == _normalize_id(team_id)].copy()
    if team_rels.empty or "season" not in team_rels.columns:
        return team_rels

    team_rels["_season_year"] = team_rels["season"].map(_season_year)
    team_rels = team_rels[team_rels["_season_year"].notna()]
    if team_rels.empty:
        return team_rels

    as_of = data.get("_as_of_season")
    if as_of is not None:
        team_rels = team_rels[team_rels["_season_year"] <= int(as_of)]
    if team_rels.empty:
        return team_rels
    roster_season = int(team_rels["_season_year"].max())
    return team_rels[team_rels["_season_year"] == roster_season]


def _same_position_competition(
    player_id: int,
    team_id: int,
    primary_position: str,
    data: Dict[str, Any],
) -> int:
    """Count same-position roster competitors in the relevant source-season roster."""
    roster = _team_roster_at_context(team_id, data)
    if roster.empty:
        return 0
    roster = roster[roster["player_id"].map(_normalize_id) != _normalize_id(player_id)]
    if roster.empty:
        return 0

    players = data.get("player_dict", {})
    pattern = rf"(?:^|/){re.escape(primary_position)}(?:$|/)"
    if "role" in roster.columns:
        role_matches = (
            roster["role"].fillna("").astype(str).str.upper().str.contains(pattern, regex=True)
        )
    else:
        role_matches = pd.Series(False, index=roster.index)
    teammate_positions = roster["player_id"].map(
        lambda value: str(
            players.get(_normalize_id(value), {}).get("position", "") or ""
        ).upper()
    )
    position_matches = teammate_positions.str.contains(pattern, regex=True)
    return int((role_matches | position_matches).sum())


def compute_context_features(
    player_id: int,
    team_id: int,
    data: Dict[str, Any],
) -> Dict[str, float]:
    """Compute player/team fit using only the state present in ``data``.

    Historical callers pass an as-of snapshot, so every lookup below is bounded
    to the requested source season. No hardcoded roster season is used.
    """
    player_row = data["player_dict"].get(_normalize_id(player_id))
    team_row = data["team_dict"].get(_normalize_id(team_id))
    if player_row is None or team_row is None:
        return _default_context_features()

    position = str(player_row.get("position", "PG") or "PG")
    primary_pos = position.split("/")[0].upper()
    style = _style_for_team(team_id, team_row, data)

    style_compat = get_style_position_compat(style, position)

    same_pos_count = _same_position_competition(
        player_id, team_id, primary_pos, data
    )
    role_opportunity = float(
        np.clip(1.0 - same_pos_count * 0.15, 0.35, 1.0)
    )
    position_fit = float(
        np.clip(style_compat * 0.60 + role_opportunity * 0.40, 0.0, 1.0)
    )

    from_league_id = player_row.get("current_league_id")
    to_league_id = _normalize_id(team_row.get("league_id"))
    from_league = data["league_dict"].get(_normalize_id(from_league_id), {})
    to_league = data["league_dict"].get(to_league_id, {})
    from_tier = int(from_league.get("tier", 1))
    to_tier = int(to_league.get("tier", 1))
    same_country = (
        from_league.get("country", "") == to_league.get("country", "")
    )
    adaptation = compute_league_adaptation(from_tier, to_tier, same_country)

    spacing_fit = 0.5
    if style in ("pace_and_space", "motion_offense"):
        stats_df = data["player_stats"]
        p_stats = stats_df[
            stats_df["player_id"].map(_normalize_id) == _normalize_id(player_id)
        ].copy()
        if not p_stats.empty:
            if "season" in p_stats.columns:
                p_stats["_season_year"] = p_stats["season"].map(_season_year)
                p_stats = p_stats.sort_values("_season_year")
            tpp = float(p_stats.iloc[-1].get("three_point_pct", 0.33) or 0.33)
            spacing_fit = float(np.clip(tpp / 0.40, 0, 1))

    return {
        "position_team_fit": round(position_fit, 4),
        "style_compatibility": round(style_compat, 4),
        "role_opportunity": round(role_opportunity, 4),
        "league_adaptation_factor": round(adaptation, 4),
        "spacing_fit": round(spacing_fit, 4),
    }


def _default_context_features() -> Dict[str, float]:
    return {
        "position_team_fit": 0.80,
        "style_compatibility": 0.80,
        "role_opportunity": 0.80,
        "league_adaptation_factor": 1.00,
        "spacing_fit": 0.50,
    }

"""Export pre-computed AI predictions for the HoopMetrics local chat backend.

Produces a gzip-compressed JSON file (``chat_model.json.gz``) consumed by the
WordPress ``hoopmetrics-chat`` plugin's **local** backend (Scenario C in the
paywall TODO §6.1).

Usage (CLI)::

    python main.py --mode export-chat-model \\
        [--source {file|sql}] \\
        [--data-dir data/sample] \\
        [--model-dir models_saved] \\
        [--out chat_model.json.gz] \\
        [--top-players 500] \\
        [--top-teams 100]

Output schema (``schema_version: 1``)::

    {
      "schema_version": 1,
      "model_version": "sha256-<hex>",
      "generated_at":  "2026-01-15T00:00:00Z",
      "predictions": [
        {
          "intent":     "predict",
          "player_hm":  "hm_p000123",
          "team_hm":    "hm_t000789",
          "season":     "2024-25",
          "payload":    { "predicted_rating": 7.42, "confidence": 0.81 }
        },
        {
          "intent":     "trajectory",
          "player_hm":  "hm_p000123",
          "season":     "2024-25",
          "payload":    { "ages": [20, 21, ...], "ratings": [6.1, 6.5, ...] }
        },
        { "intent": "peak", ... },
        { "intent": "best_team", ... },
        { "intent": "best_player", ... }
      ]
    }
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from basketball_ai.utils.helpers import (
    team_display_name as _team_display_name,
    normalize_id      as _normalize_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constant – bump only on breaking schema changes
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _player_name(data: Dict[str, Any], player_id: int) -> str:
    """Return the display name of a player, or a fallback string."""
    try:
        pdf = data["players"]
        row = pdf[pdf["id"] == player_id]
        if not row.empty:
            return str(row.iloc[0]["name"])
    except Exception:
        pass
    return f"Player {player_id}"


def _team_name_from_data(data: Dict[str, Any], team_id: int) -> str:
    """Return the display name of a team, or a fallback string."""
    try:
        team = data["team_dict"].get(team_id, {})
        return _team_display_name(team, f"Team {team_id}",
                                  league_id=str(team.get("league_id", "") or ""))
    except Exception:
        return f"Team {team_id}"

# ---------------------------------------------------------------------------
# HM-ID helpers
# ---------------------------------------------------------------------------

def _hm_player_id(player_id: int) -> str:
    """Convert numeric player ID to stable hm_* token."""
    return f"hm_p{player_id:06d}"


def _hm_team_id(team_id: int) -> str:
    """Convert numeric team ID to stable hm_* token."""
    return f"hm_t{team_id:06d}"


def _season_label(year: int) -> str:
    """Return human-readable season label, e.g. ``2024`` → ``"2024-25"``."""
    return f"{year}-{str(year + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Per-intent generators
# ---------------------------------------------------------------------------

def _gen_predict(
    engine: Any,
    player_id: int,
    team_id: int,
    season: int,
) -> Optional[Dict[str, Any]]:
    """Generate a 'predict' entry for one player × team combination."""
    try:
        pred = engine.predict_in_team(player_id, team_id, season)
        ci = round((pred.confidence_high - pred.confidence_low) / 2.0, 3)
        return {
            "intent":    "predict",
            "player_hm": _hm_player_id(player_id),
            "team_hm":   _hm_team_id(team_id),
            "season":    _season_label(season),
            "payload": {
                "predicted_rating": round(pred.predicted_rating, 3),
                "confidence":       round(max(0.0, 1.0 - ci), 3),
                "confidence_low":   round(pred.confidence_low, 3),
                "confidence_high":  round(pred.confidence_high, 3),
            },
        }
    except Exception as exc:
        logger.debug("[export] predict skip pid=%s tid=%s: %s", player_id, team_id, exc)
        return None


def _gen_trajectory(
    engine: Any,
    player_id: int,
    season_base: int,
) -> Optional[Dict[str, Any]]:
    """Generate a 'trajectory' entry for one player."""
    try:
        traj = engine.predict_age_trajectory(player_id, season_base=season_base)
        if not traj:
            return None
        return {
            "intent":    "trajectory",
            "player_hm": _hm_player_id(player_id),
            "season":    _season_label(season_base),
            "payload": {
                "ages":    [pt.age for pt in traj],
                "ratings": [round(pt.predicted_rating, 3) for pt in traj],
                "conf_low":  [round(pt.confidence_low, 3) for pt in traj],
                "conf_high": [round(pt.confidence_high, 3) for pt in traj],
            },
        }
    except Exception as exc:
        logger.debug("[export] trajectory skip pid=%s: %s", player_id, exc)
        return None


def _gen_peak(
    engine: Any,
    player_id: int,
) -> Optional[Dict[str, Any]]:
    """Generate a 'peak' entry for one player."""
    try:
        peak = engine.predict_peak(player_id)
        return {
            "intent":    "peak",
            "player_hm": _hm_player_id(player_id),
            "payload": {
                "current_age":    peak.current_age,
                "peak_age":       peak.peak_age,
                "current_rating": round(peak.current_rating, 3),
                "peak_rating":    round(peak.peak_rating, 3),
                "seasons_to_peak": peak.seasons_to_peak,
                "peak_window":    list(peak.peak_window),
            },
        }
    except Exception as exc:
        logger.debug("[export] peak skip pid=%s: %s", player_id, exc)
        return None


def _gen_best_teams(
    engine: Any,
    player_id: int,
    season: int,
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """Generate a 'best_team' entry for one player (top N fits)."""
    try:
        fits = engine.best_team_fit(player_id, top_n=top_n, season=season)
        if not fits:
            return None
        return {
            "intent":    "best_team",
            "player_hm": _hm_player_id(player_id),
            "season":    _season_label(season),
            "payload": {
                "top_teams": [
                    {
                        "rank":             f.rank,
                        "team_hm":          _hm_team_id(f.team_id),
                        "team_name":        f.team_name,
                        "league_name":      f.league_name,
                        "predicted_rating": round(f.predicted_rating, 3),
                    }
                    for f in fits
                ],
            },
        }
    except Exception as exc:
        logger.debug("[export] best_team skip pid=%s: %s", player_id, exc)
        return None


def _gen_best_players(
    engine: Any,
    team_id: int,
    season: int,
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """Generate a 'best_player' entry for one team (top N player fits)."""
    try:
        fits = engine.best_player_for_team(team_id, top_n=top_n, season=season)
        if not fits:
            return None
        return {
            "intent":  "best_player",
            "team_hm": _hm_team_id(team_id),
            "season":  _season_label(season),
            "payload": {
                "top_players": [
                    {
                        "rank":             f.rank,
                        "player_hm":        _hm_player_id(f.player_id),
                        "player_name":      f.player_name,
                        "position":         f.position,
                        "predicted_rating": round(f.predicted_rating, 3),
                    }
                    for f in fits
                ],
            },
        }
    except Exception as exc:
        logger.debug("[export] best_player skip tid=%s: %s", team_id, exc)
        return None


def _gen_what_if_transfer(
    engine: Any,
    data: Dict[str, Any],
    player_id: int,
    to_team_id: int,
    season: int,
) -> Optional[Dict[str, Any]]:
    """Generate a 'what_if' entry via simulate_transfer() for one player → team.

    Uses the player's current team as *from_team*.  Skips when the target team
    is the same as the current team.
    """
    try:
        player_info  = data["player_dict"].get(_normalize_id(player_id), {})
        from_team_id = _normalize_id(player_info.get("current_team_id")) if player_info.get("current_team_id") else None
        if not from_team_id or from_team_id == _normalize_id(to_team_id):
            return None

        impact = engine.simulate_transfer(player_id, from_team_id, to_team_id, season)

        pname       = _player_name(data, player_id)
        from_t_name = _team_name_from_data(data, from_team_id)
        to_t_name   = _team_name_from_data(data, to_team_id)

        return {
            "intent":      "what_if",
            "player_hm":   _hm_player_id(player_id),
            "team_hm":     _hm_team_id(to_team_id),
            "player_name": pname,
            "team_name":   to_t_name,
            "season":      _season_label(season),
            "payload": {
                "player_name":       pname,
                "from_team":         from_t_name,
                "to_team":           to_t_name,
                "rating_before":     round(impact.rating_before, 3),
                "rating_after":      round(impact.rating_after, 3),
                "rating_delta":      round(impact.rating_delta, 3),
                "adaptation_factor": round(impact.adaptation_factor, 3),
                "recommendation":    impact.recommendation,
            },
        }
    except Exception as exc:
        logger.debug("[export] what_if skip pid=%s tid=%s: %s", player_id, to_team_id, exc)
        return None


def _gen_player_index(
    data: Dict[str, Any],
    player_id: int,
    season: int,
) -> Optional[Dict[str, Any]]:
    """Generate a 'player_index' entry for name→ID resolution in the PHP backend."""
    try:
        pdf      = data["players"]
        row      = pdf[pdf["id"] == player_id]
        if row.empty:
            return None
        row      = row.iloc[0]
        pname    = str(row.get("name", ""))
        position = str(row.get("position", ""))
        cur_tid  = _normalize_id(row.get("current_team_id"))
        t_name   = _team_name_from_data(data, cur_tid) if cur_tid else ""

        return {
            "intent":      "player_index",
            "player_hm":   _hm_player_id(player_id),
            "player_name": pname,
            "season":      _season_label(season),
            "payload": {
                "player_name":            pname,
                "player_name_lower":      pname.lower(),
                "team_name":              t_name,
                "position":               position,
            },
        }
    except Exception as exc:
        logger.debug("[export] player_index skip pid=%s: %s", player_id, exc)
        return None


# ---------------------------------------------------------------------------
# B1 – Teammate-quality what-if scenarios
# ---------------------------------------------------------------------------

_TEAMMATE_QUALITY_LEVELS: Dict[str, float] = {
    "scarsi": 5.5,
    "medi":   6.5,
    "forti":  7.5,
    "elite":  8.5,
}


def _gen_what_if_teammates(
    engine: Any,
    player_id: int,
    team_id: int,
    season: int,
) -> Optional[Dict[str, Any]]:
    """4 teammate-quality scenarios in one payload entry."""
    scenarios: Dict[str, Any] = {}
    for label, avg_rating in _TEAMMATE_QUALITY_LEVELS.items():
        try:
            pred = engine.what_if_teammates(player_id, team_id, avg_rating, season)
            scenarios[label] = {
                "hypothetical_avg_rating": avg_rating,
                "predicted_rating":        round(pred.predicted_rating, 3),
                "confidence_low":          round(pred.confidence_low,   3),
                "confidence_high":         round(pred.confidence_high,  3),
            }
        except Exception as exc:
            logger.debug("[export] what_if_teammates skip pid=%s tid=%s label=%s: %s",
                         player_id, team_id, label, exc)
    if not scenarios:
        return None
    return {
        "intent":    "what_if_teammates",
        "player_hm": _hm_player_id(player_id),
        "team_hm":   _hm_team_id(team_id),
        "season":    _season_label(season),
        "payload":   {"scenarios": scenarios},
    }


# ---------------------------------------------------------------------------
# B2 – Role-archetype lineup scenarios
# ---------------------------------------------------------------------------

# Each value is an ordered list of role slots for a 4-man lineup archetype.
_ROLE_ARCHETYPES_SPEC: Dict[str, List[str]] = {
    "offensiva":   ["Playmaker",          "Primary Scorer",       "Primary Scorer",       "3pt Specialist"],
    "difensiva":   ["Defender",           "Defender",             "Two-way / Role Player", "Paint Scorer / Big"],
    "bilanciata":  ["Playmaker",          "Primary Scorer",       "Defender",             "Two-way / Role Player"],
    "spacing_3pt": ["Playmaker",          "3pt Specialist",       "3pt Specialist",       "Defender"],
}


def _build_role_archetypes(
    engine: Any,
    data: Dict[str, Any],
    latest_season: int,
) -> Dict[str, List[int]]:
    """Select representative player IDs per role-archetype from the dataset."""
    import pandas as pd
    from basketball_ai.features.player_features import compute_player_features

    players_df: pd.DataFrame   = data["players"]
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())

    # player_id → latest rating
    rating_map: Dict[int, float] = {}
    if not player_stats.empty and "rating" in player_stats.columns:
        latest_st = (
            player_stats.sort_values("season")
            .groupby("player_id").last().reset_index()
        )
        for _, r in latest_st.iterrows():
            rating_map[int(r["player_id"])] = float(r.get("rating") or 6.5)

    # Classify each player into a role bucket
    role_buckets: Dict[str, List[tuple]] = {}
    for _, row in players_df.iterrows():
        pid = int(row["id"])
        try:
            feats = compute_player_features(pid, data)
            role  = engine._assign_role(feats)
        except Exception:
            role = "Two-way / Role Player"
        role_buckets.setdefault(role, []).append((pid, rating_map.get(pid, 6.5)))

    for role in role_buckets:
        role_buckets[role].sort(key=lambda x: x[1], reverse=True)

    def _pick(role: str, idx: int = 0) -> Optional[int]:
        lst = role_buckets.get(role, [])
        return lst[idx][0] if len(lst) > idx else None

    archetypes: Dict[str, List[int]] = {}
    for arch_label, role_slots in _ROLE_ARCHETYPES_SPEC.items():
        role_usage: Dict[str, int] = {}
        lineup: List[int] = []
        for role in role_slots:
            idx = role_usage.get(role, 0)
            pid = _pick(role, idx)
            if pid is not None:
                lineup.append(pid)
                role_usage[role] = idx + 1
        if len(lineup) >= 3:
            archetypes[arch_label] = lineup

    logger.info("[export] Role archetypes built: %s",
                {k: len(v) for k, v in archetypes.items()})
    return archetypes


def _gen_what_if_roles(
    engine: Any,
    player_id: int,
    team_id: int,
    season: int,
    archetypes: Dict[str, List[int]],
) -> Optional[Dict[str, Any]]:
    """One payload entry with results for every role-archetype lineup."""
    results: Dict[str, Any] = {}
    for label, lineup_pids in archetypes.items():
        filtered = [p for p in lineup_pids if p != player_id]
        if not filtered:
            continue
        try:
            res = engine.what_if_lineup(player_id, team_id, filtered, season)
            results[label] = {
                "lineup_label":     label,
                "lineup_members":   [
                    {
                        "player_hm":        _hm_player_id(m.player_id),
                        "player_name":      m.player_name,
                        "position":         m.position,
                        "role":             m.role,
                        "predicted_rating": m.predicted_rating,
                    }
                    for m in res.lineup_profiles
                ],
                "predicted_rating":  round(res.predicted_rating, 3),
                "avg_lineup_rating": round(res.avg_lineup_rating, 3),
                "missing_positions": res.missing_positions,
            }
        except Exception as exc:
            logger.debug("[export] what_if_roles skip pid=%s tid=%s arch=%s: %s",
                         player_id, team_id, label, exc)
    if not results:
        return None
    return {
        "intent":    "what_if_roles",
        "player_hm": _hm_player_id(player_id),
        "team_hm":   _hm_team_id(team_id),
        "season":    _season_label(season),
        "payload":   {"archetypes": results},
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_chat_model(
    engine: Any,
    data: Dict[str, Any],
    out_path: Path,
    top_players: Optional[int] = None,
    top_teams: Optional[int] = None,
    seasons: Optional[List[int]] = None,
) -> Path:
    """Build and write the pre-computed predictions file.

    Parameters
    ----------
    engine:
        A :class:`~basketball_ai.scenarios.engine.WhatIfEngine` instance.
    data:
        The loaded data dict (as returned by ``load_all_data()``).
    out_path:
        Destination file path.  Compressed with gzip regardless of suffix.
    top_players:
        Maximum number of players to include, ranked by latest season rating.
        ``None`` (default) exports **all** players in the dataset.
    top_teams:
        Maximum number of teams to include.
        ``None`` (default) exports **all** teams in the dataset.
    seasons:
        List of season years (int) to export.  ``None`` (default) auto-detects
        all seasons present in ``data['player_stats']``.

    Returns
    -------
    Path
        The resolved path to the written file.
    """
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    players_df: pd.DataFrame = data["players"]
    teams_df:   pd.DataFrame = data["teams"]

    # ---- Select players (all by default, or top-N by rating) -----------
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    if top_players is not None and not player_stats.empty and "rating" in player_stats.columns:
        latest = (
            player_stats
            .sort_values("season")
            .groupby("player_id")
            .last()
            .reset_index()
        )
        top_pids_series = latest.nlargest(top_players, "rating")["player_id"]
        top_pids: List[int] = [int(p) for p in top_pids_series.tolist()]
    elif top_players is not None:
        top_pids = [int(row["id"]) for _, row in players_df.head(top_players).iterrows()]
    else:
        # No filter: all players in the dataset
        top_pids = [int(row["id"]) for _, row in players_df.iterrows()]

    # ---- Select teams (all by default, or top-N by tier then name) ------
    sort_cols = [c for c in ("league_tier", "name") if c in teams_df.columns]
    teams_sorted = teams_df.sort_values(sort_cols) if sort_cols else teams_df
    if top_teams is not None:
        teams_sorted = teams_sorted.head(top_teams)
    top_tids: List[int] = [int(row["id"]) for _, row in teams_sorted.iterrows()]

    # ---- Detect seasons from data if not provided -----------------------
    _ps: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    if seasons is None:
        if not _ps.empty and "season" in _ps.columns:
            seasons = sorted(int(s) for s in _ps["season"].dropna().unique())
        else:
            seasons = [2024]
    latest_season: int = max(seasons)
    # A: always include the next (forecast) season for predict / what_if loops
    seasons_ext: List[int] = sorted(set(seasons) | {latest_season + 1})

    total_predict = len(top_pids) * len(top_tids) * len(seasons_ext)
    logger.info(
        "[export] Seasons: %s (+ forecast) | Players: %d | Teams: %d | predict entries: %d",
        seasons_ext, len(top_pids), len(top_tids), total_predict,
    )

    predictions: List[Dict[str, Any]] = []

    # --- predict (player × team × season, incl. forecast season) ----------
    for season in seasons_ext:
        done = 0
        for pid in top_pids:
            for tid in top_tids:
                entry = _gen_predict(engine, pid, tid, season)
                if entry:
                    entry["player_name"] = _player_name(data, pid)
                    predictions.append(entry)
            done += 1
            if done % 50 == 0:
                logger.info("[export] predict s=%d progress: %d / %d players", season, done, len(top_pids))

    # --- trajectory (per player, from latest season forward) -------------
    for pid in top_pids:
        entry = _gen_trajectory(engine, pid, latest_season)
        if entry:
            entry["player_name"] = _player_name(data, pid)
            predictions.append(entry)

    # --- peak (per player, lifetime) -------------------------------------
    for pid in top_pids:
        entry = _gen_peak(engine, pid)
        if entry:
            entry["player_name"] = _player_name(data, pid)
            predictions.append(entry)

    # --- best_team (player × season) -------------------------------------
    for season in seasons_ext:
        for pid in top_pids:
            entry = _gen_best_teams(engine, pid, season)
            if entry:
                entry["player_name"] = _player_name(data, pid)
                predictions.append(entry)

    # --- best_player (team × season) -------------------------------------
    for season in seasons_ext:
        for tid in top_tids:
            entry = _gen_best_players(engine, tid, season)
            if entry:
                predictions.append(entry)

    # --- what_if / transfer impact (player × team × season) -------------
    for season in seasons_ext:
        done = 0
        for pid in top_pids:
            for tid in top_tids:
                entry = _gen_what_if_transfer(engine, data, pid, tid, season)
                if entry:
                    predictions.append(entry)
            done += 1
            if done % 50 == 0:
                logger.info("[export] what_if s=%d progress: %d / %d players", season, done, len(top_pids))

    # --- player_index (name → ID, once per player using latest season) ---
    for pid in top_pids:
        entry = _gen_player_index(data, pid, latest_season)
        if entry:
            predictions.append(entry)

    # --- B1: what_if_teammates (quality scenarios, latest + forecast season) ---
    _seasons_b: List[int] = [latest_season, latest_season + 1]
    _wt_done = 0
    for season in _seasons_b:
        for pid in top_pids:
            for tid in top_tids:
                entry = _gen_what_if_teammates(engine, pid, tid, season)
                if entry:
                    entry["player_name"] = _player_name(data, pid)
                    predictions.append(entry)
            _wt_done += 1
            if _wt_done % 100 == 0:
                logger.info("[export] what_if_teammates s=%d progress: %d / %d players",
                            season, _wt_done % len(top_pids) or len(top_pids), len(top_pids))

    # --- B2: what_if_roles (role-archetype lineups, latest + forecast season) ---
    _archetypes = _build_role_archetypes(engine, data, latest_season)
    if _archetypes:
        _wr_done = 0
        for season in _seasons_b:
            for pid in top_pids:
                for tid in top_tids:
                    entry = _gen_what_if_roles(engine, pid, tid, season, _archetypes)
                    if entry:
                        entry["player_name"] = _player_name(data, pid)
                        predictions.append(entry)
                _wr_done += 1
                if _wr_done % 100 == 0:
                    logger.info("[export] what_if_roles s=%d progress: %d / %d players",
                                season, _wr_done % len(top_pids) or len(top_pids), len(top_pids))

    # ---- Assemble payload -----------------------------------------------
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_bytes    = json.dumps(
        {"schema_version": SCHEMA_VERSION, "predictions": predictions},
        ensure_ascii=False,
    ).encode("utf-8")
    model_version = "sha256-" + hashlib.sha256(raw_bytes).hexdigest()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version":  model_version,
        "generated_at":   generated_at,
        "top_players":    top_players if top_players is not None else len(top_pids),
        "top_teams":      top_teams   if top_teams   is not None else len(top_tids),
        "seasons":        [_season_label(s) for s in seasons_ext],
        "predictions":    predictions,
    }

    # ---- Write gzip JSON ------------------------------------------------
    out_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with gzip.open(out_path, "wb", compresslevel=9) as fh:
        fh.write(out_bytes)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(
        "[export] Wrote %d predictions to %s (%.1f MB compressed)",
        len(predictions), out_path, size_mb,
    )
    print(
        f"  Predictions : {len(predictions):,}\n"
        f"  Output      : {out_path}\n"
        f"  Size        : {size_mb:.1f} MB\n"
        f"  Version     : {model_version}\n"
        f"  Generated at: {generated_at}"
    )
    return out_path

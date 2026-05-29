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
# Main export function
# ---------------------------------------------------------------------------

def export_chat_model(
    engine: Any,
    data: Dict[str, Any],
    out_path: Path,
    top_players: int = 500,
    top_teams: int = 100,
    season: int = 2024,
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
        Number of highest-rated players to include (by latest season rating).
    top_teams:
        Number of teams to include (by league tier then alphabetically).
    season:
        Base season year for predictions.

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

    # ---- Select top players by latest rating ----------------------------
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    if not player_stats.empty and "rating" in player_stats.columns:
        latest = (
            player_stats
            .sort_values("season")
            .groupby("player_id")
            .last()
            .reset_index()
        )
        top_pids_series = (
            latest.nlargest(top_players, "rating")["player_id"]
        )
        top_pids: List[int] = [int(p) for p in top_pids_series.tolist()]
    else:
        top_pids = [int(row["id"]) for _, row in players_df.head(top_players).iterrows()]

    # ---- Select top teams (tier 1 first, then by name) ------------------
    sort_cols = [c for c in ("league_tier", "name") if c in teams_df.columns]
    if sort_cols:
        teams_sorted = teams_df.sort_values(sort_cols)
    else:
        teams_sorted = teams_df
    top_tids: List[int] = [int(row["id"]) for _, row in teams_sorted.head(top_teams).iterrows()]

    total_predict = len(top_pids) * len(top_tids)
    logger.info(
        "[export] Generating %d predict + %d trajectory + %d peak + %d best_team + %d best_player + %d what_if + %d player_index entries …",
        total_predict, len(top_pids), len(top_pids), len(top_pids), len(top_tids),
        total_predict, len(top_pids),
    )

    predictions: List[Dict[str, Any]] = []
    done = 0

    # --- predict (player × team) -----------------------------------------
    for pid in top_pids:
        for tid in top_tids:
            entry = _gen_predict(engine, pid, tid, season)
            if entry:
                # Annotate with player name for importer
                entry["player_name"] = _player_name(data, pid)
                predictions.append(entry)
        done += 1
        if done % 50 == 0:
            logger.info("[export] predict progress: %d / %d players", done, len(top_pids))

    # --- trajectory (per player) -----------------------------------------
    for pid in top_pids:
        entry = _gen_trajectory(engine, pid, season)
        if entry:
            entry["player_name"] = _player_name(data, pid)
            predictions.append(entry)

    # --- peak (per player) -----------------------------------------------
    for pid in top_pids:
        entry = _gen_peak(engine, pid)
        if entry:
            entry["player_name"] = _player_name(data, pid)
            predictions.append(entry)

    # --- best_team (per player) ------------------------------------------
    for pid in top_pids:
        entry = _gen_best_teams(engine, pid, season)
        if entry:
            entry["player_name"] = _player_name(data, pid)
            predictions.append(entry)

    # --- best_player (per team) ------------------------------------------
    for tid in top_tids:
        entry = _gen_best_players(engine, tid, season)
        if entry:
            predictions.append(entry)

    # --- what_if / transfer impact (player × team) -----------------------
    done = 0
    for pid in top_pids:
        for tid in top_tids:
            entry = _gen_what_if_transfer(engine, data, pid, tid, season)
            if entry:
                predictions.append(entry)
        done += 1
        if done % 50 == 0:
            logger.info("[export] what_if progress: %d / %d players", done, len(top_pids))

    # --- player_index (name → ID) ----------------------------------------
    for pid in top_pids:
        entry = _gen_player_index(data, pid, season)
        if entry:
            predictions.append(entry)

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
        "top_players":    top_players,
        "top_teams":      top_teams,
        "season":         _season_label(season),
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

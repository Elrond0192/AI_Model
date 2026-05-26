"""Export pre-computed chat model predictions for the Local backend.

Generates a gzip-compressed JSON file (``chat_model.json.gz``) consumed by
the WordPress ``hoopmetrics-chat`` plugin's **Local backend** (Scenario C).

The output contains pre-computed predictions for the supported chat intents
so that the plugin can answer questions without calling the FastAPI server.

Intents exported
----------------
- ``predict``    – player rating at a specific team / season
- ``trajectory`` – age-curve rating across a player's career
- ``peak``       – career peak prediction (age, rating, window)
- ``best_team``  – top-N teams for a given player
- ``best_player``– top-N players for a given team

Usage
-----
::

    python main.py --mode export-chat-model --out chat_model.json.gz

The ``--top-players`` / ``--top-teams`` flags control the sample size
(defaults: 500 players, 100 teams).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# JSON null is used instead of None when a prediction fails.
_SKIP = None

# Season label helper: integer 2024 → "2024-25"
def _season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _hm_player(player_id: int) -> str:
    """Canonical HoopMetrics player identifier (``hm_p<id>``)."""
    return f"hm_p{player_id}"


def _hm_team(team_id: int) -> str:
    """Canonical HoopMetrics team identifier (``hm_t<id>``)."""
    return f"hm_t{team_id}"


# ---------------------------------------------------------------------------
# Per-intent generators
# ---------------------------------------------------------------------------

def _export_predict(
    engine: Any,
    player_id: int,
    team_id: int,
    season: int,
) -> Optional[Dict[str, Any]]:
    """Single predict record: player × team × season."""
    try:
        pred = engine.predict_in_team(player_id, team_id, season=season)
        return {
            "intent":    "predict",
            "player_hm": _hm_player(player_id),
            "team_hm":   _hm_team(team_id),
            "season":    _season_label(season),
            "payload": {
                "predicted_rating": round(pred.predicted_rating, 3),
                "confidence_low":   round(pred.confidence_low, 3),
                "confidence_high":  round(pred.confidence_high, 3),
                "confidence":       round(
                    (pred.confidence_high - pred.confidence_low) / 10.0, 3
                ),
                "explanation":      pred.explanation,
            },
        }
    except Exception as exc:
        logger.debug(
            "[export] predict failed player_id=%s team_id=%s: %s",
            player_id, team_id, exc,
        )
        return _SKIP


def _export_trajectory(
    engine: Any,
    player_id: int,
    season_base: int,
) -> Optional[Dict[str, Any]]:
    """Trajectory record: full age-curve for a player."""
    try:
        points = engine.predict_age_trajectory(player_id, season_base=season_base)
        return {
            "intent":    "trajectory",
            "player_hm": _hm_player(player_id),
            "season":    _season_label(season_base),
            "payload": {
                "ages":    [p.age for p in points],
                "ratings": [round(p.predicted_rating, 3) for p in points],
                "conf_lo": [round(p.confidence_low, 3)   for p in points],
                "conf_hi": [round(p.confidence_high, 3)  for p in points],
            },
        }
    except Exception as exc:
        logger.debug(
            "[export] trajectory failed player_id=%s: %s", player_id, exc
        )
        return _SKIP


def _export_peak(
    engine: Any,
    player_id: int,
) -> Optional[Dict[str, Any]]:
    """Peak record: career-peak prediction for a player."""
    try:
        peak = engine.predict_peak(player_id)
        return {
            "intent":    "peak",
            "player_hm": _hm_player(player_id),
            "payload": {
                "player_name":    peak.player_name,
                "current_age":    peak.current_age,
                "current_rating": round(peak.current_rating, 3),
                "peak_age":       peak.peak_age,
                "peak_rating":    round(peak.peak_rating, 3),
                "seasons_to_peak": peak.seasons_to_peak,
                "peak_window_start": peak.peak_window[0],
                "peak_window_end":   peak.peak_window[1],
            },
        }
    except Exception as exc:
        logger.debug(
            "[export] peak failed player_id=%s: %s", player_id, exc
        )
        return _SKIP


def _export_best_team(
    engine: Any,
    player_id: int,
    season: int,
    top_n: int = 5,
) -> Optional[Dict[str, Any]]:
    """Best-team record: top-N team fits for a player."""
    try:
        fits = engine.best_team_fit(player_id, top_n=top_n, season=season)
        return {
            "intent":    "best_team",
            "player_hm": _hm_player(player_id),
            "season":    _season_label(season),
            "payload": {
                "teams": [
                    {
                        "rank":             f.rank,
                        "team_hm":          _hm_team(f.team_id),
                        "team_name":        f.team_name,
                        "league_name":      f.league_name,
                        "predicted_rating": round(f.predicted_rating, 3),
                        "compatibility":    round(f.compatibility_score, 3),
                    }
                    for f in fits
                ]
            },
        }
    except Exception as exc:
        logger.debug(
            "[export] best_team failed player_id=%s: %s", player_id, exc
        )
        return _SKIP


def _export_best_player(
    engine: Any,
    team_id: int,
    season: int,
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """Best-player record: top-N player fits for a team."""
    try:
        fits = engine.best_player_for_team(team_id, top_n=top_n, season=season)
        return {
            "intent":  "best_player",
            "team_hm": _hm_team(team_id),
            "season":  _season_label(season),
            "payload": {
                "players": [
                    {
                        "rank":             f.rank,
                        "player_hm":        _hm_player(f.player_id),
                        "player_name":      f.player_name,
                        "position":         f.position,
                        "current_team":     f.current_team,
                        "predicted_rating": round(f.predicted_rating, 3),
                    }
                    for f in fits
                ]
            },
        }
    except Exception as exc:
        logger.debug(
            "[export] best_player failed team_id=%s: %s", team_id, exc
        )
        return _SKIP


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_chat_model(
    engine: Any,
    data: Dict[str, Any],
    out_path: str = "chat_model.json.gz",
    season: int = 2024,
    top_players: int = 500,
    top_teams: int = 100,
    top_n_best_team: int = 5,
    top_n_best_player: int = 10,
    model_version: Optional[str] = None,
) -> str:
    """Generate and write the chat model export file.

    Parameters
    ----------
    engine:
        A ``WhatIfEngine`` instance (already loaded with data and model).
    data:
        The data dict as returned by ``load_all_data()``.
    out_path:
        Destination file path.  Should end with ``.json.gz``.
    season:
        Base season year (integer) used for predictions.
    top_players:
        Number of players to include (sorted by rating desc).
    top_teams:
        Number of teams to include for cross-product predict exports.
    top_n_best_team / top_n_best_player:
        Top-N limits for the best_team / best_player intents.
    model_version:
        Optional version string; defaults to SHA-256 of the serialised model
        registry when available.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    import pandas as pd
    from pathlib import Path as _Path

    players_df: pd.DataFrame = data["players"]
    teams_df:   pd.DataFrame = data["teams"]

    # ------------------------------------------------------------------ #
    # Select representative players & teams
    # ------------------------------------------------------------------ #
    # Sort players by most recent rating (desc) to prioritise top talent
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    if not player_stats.empty and "rating" in player_stats.columns:
        latest_ratings = (
            player_stats
            .sort_values("season")
            .groupby("player_id")["rating"]
            .last()
            .reset_index()
            .rename(columns={"rating": "_sort_rating"})
        )
        merged = players_df.merge(
            latest_ratings,
            left_on="id", right_on="player_id",
            how="left",
        ).fillna({"_sort_rating": 0})
        merged = merged.sort_values("_sort_rating", ascending=False)
        selected_players = merged.head(top_players)
    else:
        selected_players = players_df.head(top_players)

    # Teams: prefer top-tier leagues
    if "league_tier" in teams_df.columns:
        teams_sorted = teams_df.sort_values("league_tier")
    else:
        teams_sorted = teams_df
    selected_teams = teams_sorted.head(top_teams)

    player_ids: List[int] = [int(r["id"]) for _, r in selected_players.iterrows()]
    team_ids:   List[int] = [int(r["id"]) for _, r in selected_teams.iterrows()]

    # ------------------------------------------------------------------ #
    # Generate predictions
    # ------------------------------------------------------------------ #
    predictions: List[Dict[str, Any]] = []
    total = 0

    n_players = len(player_ids)
    n_teams   = len(team_ids)

    logger.info(
        "[export] Generating predictions: %d players × %d teams …",
        n_players, n_teams,
    )
    print(
        f"[export-chat-model] Players: {n_players}  |  Teams: {n_teams}  |  "
        f"Season base: {_season_label(season)}"
    )

    # -- predict (player × current team only to avoid O(N×M) explosion) -
    print("[export-chat-model] 1/5 Generating 'predict' records …")
    player_team_map = {
        int(r["id"]): int(r["current_team_id"])
        for _, r in players_df.iterrows()
        if r.get("current_team_id")
    }
    for pid in player_ids:
        tid = player_team_map.get(pid)
        if tid is None:
            continue
        rec = _export_predict(engine, pid, tid, season)
        if rec:
            predictions.append(rec)
            total += 1

    # -- predict cross-product for top-50 players × top-20 teams
    print("[export-chat-model] 2/5 Generating cross-product 'predict' records …")
    cross_players = player_ids[:50]
    cross_teams   = team_ids[:20]
    for pid in cross_players:
        for tid in cross_teams:
            # skip if already exported (current team)
            if player_team_map.get(pid) == tid:
                continue
            rec = _export_predict(engine, pid, tid, season)
            if rec:
                predictions.append(rec)
                total += 1

    # -- trajectory
    print("[export-chat-model] 3/5 Generating 'trajectory' records …")
    for pid in player_ids:
        rec = _export_trajectory(engine, pid, season_base=season)
        if rec:
            predictions.append(rec)
            total += 1

    # -- peak
    print("[export-chat-model] 4/5 Generating 'peak' records …")
    for pid in player_ids:
        rec = _export_peak(engine, pid)
        if rec:
            predictions.append(rec)
            total += 1

    # -- best_team
    print("[export-chat-model] 5a/5 Generating 'best_team' records …")
    for pid in player_ids:
        rec = _export_best_team(engine, pid, season, top_n=top_n_best_team)
        if rec:
            predictions.append(rec)
            total += 1

    # -- best_player (for selected teams only – O(N) not O(N×M))
    print("[export-chat-model] 5b/5 Generating 'best_player' records …")
    for tid in team_ids[:top_teams]:
        rec = _export_best_player(engine, tid, season, top_n=top_n_best_player)
        if rec:
            predictions.append(rec)
            total += 1

    # ------------------------------------------------------------------ #
    # Compute model version (SHA-256 of sorted prediction payload)
    # ------------------------------------------------------------------ #
    if model_version is None:
        digest_src = json.dumps(
            predictions[:100], sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        model_version = "sha256-" + hashlib.sha256(digest_src).hexdigest()[:16]

    generated_at = datetime.now(timezone.utc).isoformat()

    document: Dict[str, Any] = {
        "schema_version": 1,
        "model_version":  model_version,
        "generated_at":   generated_at,
        "season_base":    _season_label(season),
        "stats": {
            "total_records":   total,
            "players_included": n_players,
            "teams_included":   n_teams,
        },
        "predictions": predictions,
    }

    # ------------------------------------------------------------------ #
    # Write gzip JSON
    # ------------------------------------------------------------------ #
    out = _Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(str(out), "wb", compresslevel=9) as fh:
        fh.write(raw_bytes)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(
        f"\n[export-chat-model] ✓ Written {total:,} records → {out}  "
        f"({size_mb:.1f} MB compressed)"
    )
    print(f"[export-chat-model] model_version = {model_version}")
    print(f"[export-chat-model] generated_at  = {generated_at}")
    return str(out.resolve())

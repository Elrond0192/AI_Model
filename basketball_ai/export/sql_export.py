"""Export pre-computed AI predictions directly to Azure SQL Server.

This module complements :mod:`basketball_ai.export.chat_model_export` by writing
predictions to an Azure SQL table instead of a gzip-compressed JSON file.
Both exports can be run independently or together.

The target table ``[{schema}].[{table}]`` is created automatically if it does
not exist.  Existing rows are merged (UPSERT) on the natural key
``(intent, player_hm, team_hm, season)``.

Usage (CLI)::

    python main.py --mode export-sql \\
        [--source {file|sql}] \\
        [--top-players 500] \\
        [--top-teams 100] \\
        [--sql-schema Predizioni] \\
        [--sql-table hm_predictions]

The Azure SQL connection string is read from the environment variable
``AZURE_SQL_CONNECTION_STRING`` or built from the individual
``AZURE_SQL_*`` variables (same as the data loader).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_CREATE_SCHEMA = (
    "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{schema}') "
    "EXEC('CREATE SCHEMA [{schema}]')"
)

_DDL_CREATE_TABLE = """
IF OBJECT_ID(N'[{schema}].[{table}]', N'U') IS NULL
BEGIN
    CREATE TABLE [{schema}].[{table}] (
        [id]            BIGINT        IDENTITY(1,1) NOT NULL,
        [intent]        NVARCHAR(64)  NOT NULL,
        [player_hm]     NVARCHAR(20)  NULL,
        [team_hm]       NVARCHAR(20)  NULL,
        [player_name]   NVARCHAR(200) NULL,
        [season]        NVARCHAR(10)  NULL,
        [payload]       NVARCHAR(MAX) NOT NULL,
        [model_version] NVARCHAR(80)  NOT NULL,
        [computed_at]   DATETIME2     NOT NULL,
        CONSTRAINT [PK_{table}] PRIMARY KEY CLUSTERED ([id] ASC)
    );
    CREATE NONCLUSTERED INDEX [IX_{table}_lookup]
        ON [{schema}].[{table}] ([intent], [player_hm], [team_hm], [season]);
END
"""

# MERGE keyed on (intent, player_hm, team_hm, season) with NULL-safe comparison.
# Parameter order:  intent, player_hm(×2), team_hm(×2), season(×2)  → UPDATE params
#                   intent, player_hm, team_hm, player_name, season, payload, model_version → INSERT params
_MERGE_SQL = """
MERGE [{schema}].[{table}] WITH (HOLDLOCK) AS tgt
USING (SELECT ? AS intent, ? AS player_hm, ? AS team_hm, ? AS season) AS src
    ON  tgt.intent    =  src.intent
    AND (tgt.player_hm = src.player_hm OR (tgt.player_hm IS NULL AND src.player_hm IS NULL))
    AND (tgt.team_hm   = src.team_hm   OR (tgt.team_hm   IS NULL AND src.team_hm   IS NULL))
    AND (tgt.season    = src.season     OR (tgt.season    IS NULL AND src.season    IS NULL))
WHEN MATCHED THEN
    UPDATE SET
        [payload]       = ?,
        [player_name]   = ?,
        [model_version] = ?,
        [computed_at]   = GETUTCDATE()
WHEN NOT MATCHED THEN
    INSERT ([intent],[player_hm],[team_hm],[player_name],[season],[payload],[model_version],[computed_at])
    VALUES (?,?,?,?,?,?,?,GETUTCDATE());
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _build_conn_str(override: Optional[str] = None) -> str:
    """Return a pyodbc connection string from env vars or the provided override."""
    if override:
        return override

    env = os.environ.get("AZURE_SQL_CONNECTION_STRING", "")
    if env:
        # If it's a SQLAlchemy URL, extract the pyodbc portion
        if "odbc_connect=" in env:
            from urllib.parse import unquote_plus
            return unquote_plus(env.split("odbc_connect=", 1)[1])
        if not env.startswith("mssql"):
            return env  # Already a pyodbc string

    server   = os.environ.get("AZURE_SQL_SERVER",   "")
    database = os.environ.get("AZURE_SQL_DATABASE", "")
    user     = (os.environ.get("AZURE_SQL_USER",     "")
                or os.environ.get("AZURE_SQL_USERNAME", ""))
    password = os.environ.get("AZURE_SQL_PASSWORD", "")
    driver   = os.environ.get("AZURE_SQL_DRIVER",   "ODBC Driver 18 for SQL Server")

    if not server or not database:
        raise ValueError(
            "Azure SQL connection not configured. "
            "Set AZURE_SQL_CONNECTION_STRING or "
            "AZURE_SQL_SERVER + AZURE_SQL_DATABASE + AZURE_SQL_USER + AZURE_SQL_PASSWORD."
        )
    return (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};Encrypt=yes;TrustServerCertificate=no"
    )


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_chat_model_to_sql(
    engine: Any,
    data: Dict[str, Any],
    conn_str: Optional[str] = None,
    schema: str = "Predizioni",
    table: str = "hm_predictions",
    top_players: Optional[int] = None,
    top_teams: Optional[int] = None,
    seasons: Optional[List[int]] = None,
    batch_size: int = 500,
) -> int:
    """Build all predictions and upsert them into Azure SQL Server.

    Parameters
    ----------
    engine:
        A :class:`~basketball_ai.scenarios.engine.WhatIfEngine` instance.
    data:
        The loaded data dict (as returned by ``load_all_data()``).
    conn_str:
        A pyodbc connection string.  Falls back to ``AZURE_SQL_CONNECTION_STRING``
        environment variable when omitted.
    schema:
        Target SQL schema (default: ``"Predizioni"``).
    table:
        Target table name (default: ``"hm_predictions"``).
    top_players, top_teams, seasons:
        Forwarded to :func:`~basketball_ai.export.chat_model_export.export_chat_model`.
    batch_size:
        Rows committed per transaction (affects progress logging).

    Returns
    -------
    int
        Total number of rows upserted.
    """
    try:
        import pyodbc
    except ImportError as exc:
        raise ImportError(
            "pyodbc is required for SQL export.  "
            "Install it with:  pip install pyodbc"
        ) from exc

    from basketball_ai.export.chat_model_export import export_chat_model

    # ------------------------------------------------------------------ #
    # Step 1 – Generate predictions to a temporary gzip file             #
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as fh:
        tmp_path = Path(fh.name)

    try:
        logger.info("[sql_export] Generating predictions → %s …", tmp_path)
        export_chat_model(
            engine, data, tmp_path,
            top_players=top_players,
            top_teams=top_teams,
            seasons=seasons,
        )
        with gzip.open(tmp_path, "rt", encoding="utf-8") as fh:
            doc: Dict[str, Any] = json.load(fh)
    finally:
        tmp_path.unlink(missing_ok=True)

    predictions: List[Dict[str, Any]] = doc["predictions"]
    model_version: str                = doc["model_version"]

    if not predictions:
        logger.warning("[sql_export] No predictions generated — nothing to write.")
        return 0

    logger.info("[sql_export] %d predictions to upsert → [%s].[%s]",
                len(predictions), schema, table)

    # ------------------------------------------------------------------ #
    # Step 2 – Connect and ensure schema + table exist                   #
    # ------------------------------------------------------------------ #
    resolved_conn = _build_conn_str(conn_str)
    conn = pyodbc.connect(resolved_conn, autocommit=False)
    cur  = conn.cursor()
    cur.execute("SET NOCOUNT ON")

    cur.execute(_DDL_CREATE_SCHEMA.format(schema=schema))
    conn.commit()
    cur.execute(_DDL_CREATE_TABLE.format(schema=schema, table=table))
    conn.commit()

    merge_sql = _MERGE_SQL.format(schema=schema, table=table)

    # ------------------------------------------------------------------ #
    # Step 3 – Upsert in batches                                         #
    # ------------------------------------------------------------------ #
    upserted = 0
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i : i + batch_size]
        for pred in batch:
            intent      = pred.get("intent", "")
            player_hm   = pred.get("player_hm")   or None
            team_hm     = pred.get("team_hm")     or None
            player_name = pred.get("player_name") or None
            season      = pred.get("season")      or None
            payload_str = json.dumps(pred.get("payload", {}), ensure_ascii=False)

            cur.execute(
                merge_sql,
                # USING key
                intent, player_hm, team_hm, season,
                # UPDATE SET
                payload_str, player_name, model_version,
                # INSERT values
                intent, player_hm, team_hm, player_name, season,
                payload_str, model_version,
            )
            upserted += 1

        conn.commit()
        logger.info(
            "[sql_export] Progress: %d / %d",
            min(i + batch_size, len(predictions)), len(predictions),
        )

    cur.close()
    conn.close()

    logger.info(
        "[sql_export] Done: %d rows → [%s].[%s]  model_version=%s",
        upserted, schema, table, model_version,
    )
    print(
        f"  Rows upserted : {upserted:,}\n"
        f"  Target        : [{schema}].[{table}]\n"
        f"  Model version : {model_version}"
    )

    # Also build per-player profiles (denormalised, for DeepSeek RAG)
    export_player_profiles_to_sql(
        engine=engine,
        data=data,
        conn_str=conn_str,
        schema=schema,
        model_version=model_version,
        top_players=top_players,
        seasons=seasons,
        batch_size=batch_size,
    )

    return upserted


# ---------------------------------------------------------------------------
# Per-player profile export
# ---------------------------------------------------------------------------

_DDL_CREATE_PROFILES_TABLE = """
IF OBJECT_ID(N'[{schema}].[hm_player_profiles]', N'U') IS NULL
BEGIN
    CREATE TABLE [{schema}].[hm_player_profiles] (
        [id]            BIGINT        IDENTITY(1,1) NOT NULL,
        [player_hm]     NVARCHAR(20)  NOT NULL,
        [player_name]   NVARCHAR(200) NOT NULL,
        [season]        NVARCHAR(10)  NOT NULL,
        [profile_json]  NVARCHAR(MAX) NOT NULL,
        [model_version] NVARCHAR(80)  NOT NULL,
        [computed_at]   DATETIME2     NOT NULL,
        CONSTRAINT [PK_hm_player_profiles] PRIMARY KEY CLUSTERED ([id] ASC)
    );
    CREATE NONCLUSTERED INDEX [IX_hm_player_profiles_lookup]
        ON [{schema}].[hm_player_profiles] ([player_hm], [season]);
    CREATE NONCLUSTERED INDEX [IX_hm_player_profiles_name]
        ON [{schema}].[hm_player_profiles] ([player_name]);
END
"""

_MERGE_PROFILES_SQL = """
MERGE [{schema}].[hm_player_profiles] WITH (HOLDLOCK) AS tgt
USING (SELECT ? AS player_hm, ? AS season) AS src
    ON tgt.player_hm = src.player_hm AND tgt.season = src.season
WHEN MATCHED THEN
    UPDATE SET
        [player_name]   = ?,
        [profile_json]  = ?,
        [model_version] = ?,
        [computed_at]   = GETUTCDATE()
WHEN NOT MATCHED THEN
    INSERT ([player_hm],[player_name],[season],[profile_json],[model_version],[computed_at])
    VALUES (?,?,?,?,?,GETUTCDATE());
"""


def export_player_profiles_to_sql(
    engine: Any,
    data: Dict[str, Any],
    conn_str: Optional[str] = None,
    schema: str = "Predizioni",
    model_version: str = "unknown",
    top_players: Optional[int] = None,
    seasons: Optional[List[int]] = None,
    batch_size: int = 200,
) -> int:
    """Build one denormalised profile row per player and upsert to SQL.

    Each row contains a ``profile_json`` column with all AI insights for the
    player (current rating, peak, trajectory, best teams, teammate quality
    impact).  This is the primary data source for the DeepSeek RAG backend.

    Must be called *after* :func:`export_chat_model_to_sql` (or at least
    after the models are trained and ``engine`` is ready).

    Returns the number of profile rows upserted.
    """
    try:
        import pyodbc
    except ImportError as exc:
        raise ImportError("pyodbc is required. pip install pyodbc") from exc

    import pandas as pd
    from basketball_ai.data.loader import _to_int as _id_to_int
    from basketball_ai.export.chat_model_export import (
        _hm_player_id, _hm_team_id, _player_name, _season_label,
    )

    players_df: pd.DataFrame   = data["players"]
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    team_dict: dict            = data.get("team_dict", {})

    # ---- Resolve player list (same logic as export_chat_model) ----------
    if top_players is not None and not player_stats.empty and "rating" in player_stats.columns:
        latest = (
            player_stats
            .sort_values("season")
            .groupby("player_id").last()
            .reset_index()
        )
        top_pids: List[int] = [
            _id_to_int(p)
            for p in latest.nlargest(top_players, "rating")["player_id"].tolist()
        ]
    elif top_players is not None:
        top_pids = [_id_to_int(r["id"]) for _, r in players_df.head(top_players).iterrows()]
    else:
        top_pids = [_id_to_int(r["id"]) for _, r in players_df.iterrows()]

    # ---- Detect latest season -------------------------------------------
    if seasons is None:
        if not player_stats.empty and "season" in player_stats.columns:
            seasons = sorted(int(s) for s in player_stats["season"].dropna().unique())
        else:
            seasons = [2024]
    latest_season: int = max(seasons)
    season_label: str  = _season_label(latest_season)

    # ---- Connect ---------------------------------------------------------
    resolved_conn = _build_conn_str(conn_str)
    conn = pyodbc.connect(resolved_conn, autocommit=False)
    cur  = conn.cursor()
    cur.execute("SET NOCOUNT ON")
    cur.execute(_DDL_CREATE_PROFILES_TABLE.format(schema=schema))
    conn.commit()

    merge_sql = _MERGE_PROFILES_SQL.format(schema=schema)

    # ---- Build & upsert profiles ----------------------------------------
    _TEAMMATE_QUALITIES = [
        ("scarsi", 5.0),
        ("medi",   6.5),
        ("forti",  7.5),
        ("elite",  8.5),
    ]

    upserted = 0
    for i, pid in enumerate(top_pids):
        player_name_str = _player_name(data, pid)
        player_hm       = _hm_player_id(pid)

        # Resolve current team
        p_row = players_df[players_df["id"].apply(_id_to_int) == pid]
        cur_tid: Optional[int] = None
        if not p_row.empty:
            raw_tid = p_row.iloc[0].get("current_team_id")
            if raw_tid and str(raw_tid) not in ("", "nan", "None"):
                try:
                    cur_tid = _id_to_int(raw_tid)
                except Exception:
                    pass
        if cur_tid is None and top_pids:
            # Fallback: first team in dataset
            try:
                cur_tid = _id_to_int(data["teams"].iloc[0]["id"])
            except Exception:
                cur_tid = 1

        profile: Dict[str, Any] = {
            "player_hm":   player_hm,
            "player_name": player_name_str,
            "season":      season_label,
        }

        # -- Basic info from players_df
        if not p_row.empty:
            pr = p_row.iloc[0]
            profile["position"]     = str(pr.get("position", ""))
            profile["age"]          = int(pr.get("age", 0) or 0)
            profile["current_team"] = str(
                team_dict.get(cur_tid, {}).get("name", "") if cur_tid else ""
            )

        # -- Current rating (predict in current team)
        try:
            pred = engine.predict_in_team(pid, cur_tid, latest_season)
            profile["current_rating"]   = round(pred.predicted_rating, 3)
            profile["confidence"]       = round(max(0.0, 1.0 - (pred.confidence_high - pred.confidence_low) / 2.0), 3)
            profile["confidence_low"]   = round(pred.confidence_low,  3)
            profile["confidence_high"]  = round(pred.confidence_high, 3)
        except Exception as exc:
            logger.debug("[profiles] predict skip pid=%s: %s", pid, exc)

        # -- Peak prediction
        try:
            peak = engine.predict_peak(pid)
            profile["peak_age"]        = peak.peak_age
            profile["peak_rating"]     = round(peak.peak_rating, 3)
            profile["current_age"]     = peak.current_age
            profile["seasons_to_peak"] = peak.seasons_to_peak
            profile["peak_window"]     = list(peak.peak_window)
        except Exception as exc:
            logger.debug("[profiles] peak skip pid=%s: %s", pid, exc)

        # -- Trajectory
        try:
            traj = engine.predict_age_trajectory(pid, season_base=latest_season)
            if traj:
                profile["trajectory"] = [
                    {
                        "age":       pt.age,
                        "rating":    round(pt.predicted_rating, 3),
                        "conf_low":  round(pt.confidence_low,   3),
                        "conf_high": round(pt.confidence_high,  3),
                    }
                    for pt in traj
                ]
        except Exception as exc:
            logger.debug("[profiles] trajectory skip pid=%s: %s", pid, exc)

        # -- Best team fits (top 5)
        try:
            fits = engine.best_team_fit(pid, top_n=5, season=latest_season)
            if fits:
                profile["best_teams"] = [
                    {
                        "rank":    f.rank,
                        "team":    f.team_name,
                        "league":  f.league_name,
                        "rating":  round(f.predicted_rating, 3),
                    }
                    for f in fits
                ]
        except Exception as exc:
            logger.debug("[profiles] best_team skip pid=%s: %s", pid, exc)

        # -- Teammate quality scenarios
        teammate_impact: Dict[str, Any] = {}
        for label, avg_q in _TEAMMATE_QUALITIES:
            try:
                res = engine.what_if_teammates(pid, cur_tid, avg_q, season=latest_season)
                teammate_impact[label] = {
                    "avg_teammate_quality": avg_q,
                    "predicted_rating":     round(res.predicted_rating, 3),
                }
            except Exception as exc:
                logger.debug("[profiles] teammates skip pid=%s q=%s: %s", pid, avg_q, exc)
        if teammate_impact:
            profile["teammate_quality_impact"] = teammate_impact

        profile_str = json.dumps(profile, ensure_ascii=False)

        cur.execute(
            merge_sql,
            # USING key
            player_hm, season_label,
            # UPDATE
            player_name_str, profile_str, model_version,
            # INSERT
            player_hm, player_name_str, season_label, profile_str, model_version,
        )
        upserted += 1

        if (i + 1) % batch_size == 0:
            conn.commit()
            logger.info("[profiles] %d / %d players upserted", i + 1, len(top_pids))

    conn.commit()
    cur.close()
    conn.close()

    logger.info("[profiles] Done: %d profile rows → [%s].[hm_player_profiles]", upserted, schema)
    print(f"  Player profiles : {upserted:,} → [{schema}].[hm_player_profiles]")
    return upserted

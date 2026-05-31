"""Basketball XGBoost performance model.

Predicts player rating (0-10) from basketball-specific features:
per-36 stats, advanced metrics (PER, BPM, TS%, USG%), age, position,
DB role labels (ruolo_combinato / offensivo / difensivo), RAPTOR, LEBRON,
OWS/DWS, FIC, interaction features, and career trajectory signals.
"""
from __future__ import annotations

import json as _json
from datetime import datetime as _dt, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from basketball_ai.data.loader import _to_int

logger = logging.getLogger(__name__)


def _xgb_device() -> str:
    """Return 'cuda' when an NVIDIA GPU with CUDA is available, else 'cpu'."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_XGB_DEVICE = _xgb_device()
if _XGB_DEVICE == "cuda":
    logger.info("XGBoost: GPU (CUDA) accelerazione attiva.")
else:
    logger.info("XGBoost: nessuna GPU rilevata, uso CPU.")
from basketball_ai.constants import (
    LEAGUE_MAX_GAMES_BY_NAME,
    LEAGUE_MAX_GAMES_DEFAULT,
    _primary_pos,
    _peak_age,
)


def _get_git_sha() -> str:
    """Return the current git commit SHA (short), or empty string if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _safe_league_id(lid: Any) -> int:
    """Convert a raw league_id value to ``int``, returning 0 on failure."""
    if lid is None:
        return 0
    if isinstance(lid, float) and np.isnan(lid):
        return 0
    try:
        return int(lid)
    except (TypeError, ValueError):
        return 0

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _compute_data_signature(X: pd.DataFrame, y: np.ndarray) -> str:
    """MD5 hash of feature matrix + targets for reproducibility."""
    combined = pd.concat([X, pd.Series(y, name="_target")], axis=1)
    return hashlib.md5(
        pd.util.hash_pandas_object(combined, index=False).values.tobytes()
    ).hexdigest()


def compute_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Fit simple baseline models and return their validation RMSEs."""
    results: Dict[str, float] = {}
    for name, model in [
        ("baseline_mean_val_rmse",   DummyRegressor(strategy="mean")),
        ("baseline_linear_val_rmse", LinearRegression()),
        ("baseline_rf_val_rmse",     RandomForestRegressor(n_estimators=50, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results[name] = float(np.sqrt(np.mean((preds - y_val) ** 2)))
    return results


# Competition-type encoding used as a training feature and prediction context
COMPETITION_ENCODING: Dict[str, int] = {
    "RS": 0,          # Regular Season
    "PO": 1,          # Playoffs
    "CUP": 2,         # Cup competition
    "SUPERCUP": 3,    # Super Cup / preseason
}


def compute_po_features(player_stats_df: "pd.DataFrame") -> Dict[str, float]:
    """Return competition-specific features for a player from their full stats DataFrame.

    Args:
        player_stats_df: Rows from ``player_stats`` for a single player.

    Returns:
        Dict with keys ``po_vs_rs_delta`` (mean PO rating − mean RS rating; 0.0 if
        one side is missing) and ``po_games_played`` (total PO games played).
    """
    if "competition" not in player_stats_df.columns:
        return {"po_vs_rs_delta": 0.0, "po_games_played": 0.0, "has_po_history": 0.0}

    rs_hist = player_stats_df[player_stats_df["competition"] == "RS"]
    po_hist = player_stats_df[player_stats_df["competition"] == "PO"]

    rs_mean = float(rs_hist["rating"].mean()) if not rs_hist.empty else np.nan
    po_mean = float(po_hist["rating"].mean()) if not po_hist.empty else np.nan

    if not (np.isnan(rs_mean) or np.isnan(po_mean)):
        po_vs_rs_delta = po_mean - rs_mean
    else:
        po_vs_rs_delta = 0.0

    po_gp = (
        float(po_hist["games_played"].sum())
        if (not po_hist.empty and "games_played" in po_hist.columns)
        else 0.0
    )

    return {
        "po_vs_rs_delta":  po_vs_rs_delta,
        "po_games_played": po_gp,
        "has_po_history":  1.0 if po_gp > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Feature columns – the full feature set used by the XGBoost model.
# ---------------------------------------------------------------------------
FEATURE_COLS: List[str] = [
    # --- Identity / context ---------------------------------------------------
    "age",
    # One-hot position encoding (replaces ordinal to eliminate positional bias)
    "pos_PG",
    "pos_SG",
    "pos_SF",
    "pos_PF",
    "pos_C",
    "pos_hybrid",
    "competition_enc",
    # --- DB role labels (all three dimensions) --------------------------------
    "role_enc",             # ruolo_combinato (combined role)
    "role_off_enc",         # ruolo_offensivo (offensive role)
    "role_def_enc",         # ruolo_difensivo (defensive role)
    # --- Per-36 volume stats --------------------------------------------------
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
    "stl_per_36",
    "blk_per_36",
    # --- Core advanced metrics (historical averages) --------------------------
    "avg_per",
    "avg_ts_pct",
    "avg_usg_pct",
    # avg_bpm removed: BPM = OBPM + DBPM (perfect multicollinearity)
    "avg_obpm",
    "avg_dbpm",
    # --- Career signals -------------------------------------------------------
    "form_score",
    "consistency_score",
    "career_trajectory",
    "age_vs_peak_age",
    "ts_efficiency_trend",  # per-season slope of TS% over career
    "durability_score",     # avg games_played / 82 (or league max)
    # --- Competition context --------------------------------------------------
    "po_vs_rs_delta",
    "po_games_played",
    "has_po_history",      # 1.0 if player has any PO history, 0.0 otherwise
    # --- Composite / model-based ratings (DB schema) --------------------------
    "avg_spm",
    "avg_raptor_off",       # RAPTOR offensive component
    "avg_raptor_def",       # RAPTOR defensive component
    # avg_raptor_total removed: equals raptor_off + raptor_def (perfect multicollinearity)
    "avg_lebron_off",       # LEBRON offensive component
    "avg_lebron_def",       # LEBRON defensive component
    # avg_lebron_total removed: equals lebron_off + lebron_def (perfect multicollinearity)
    "avg_gm_sc",
    "avg_fic",              # Floor Impact Counter
    "avg_ows",              # Offensive Win Shares
    "avg_dws",              # Defensive Win Shares
    # --- Efficiency / hustle --------------------------------------------------
    "avg_scoring_efficiency",
    "avg_hustle_index",
    "avg_foul_drawing_rate",
    # --- Rate stats (normalised percentages) ----------------------------------
    "avg_tov_pct",
    "avg_ast_pct",
    "avg_orb_pct",
    "avg_drb_pct",
    # --- Clutch performance ---------------------------------------------------
    "clutch_pts_per_36",
    "avg_clutch_ts_pct",
    "avg_clutch_net_rtg",
    # --- On/Off impact --------------------------------------------------------
    "avg_net_rtg_diff",
    "avg_ortg_diff",
    # --- Per-40 projection features -------------------------------------------
    "avg_pts_per_40",
    "avg_ast_per_40",
    # --- Starter status (from Boxscore.SF) ------------------------------------
    "avg_starter_pct",      # Fraction of games started (0–1) – role signal
    # --- Engineered interaction features (domain-specific) --------------------
    "obpm_x_usg",           # Offensive production at high usage (OBPM × USG%)
    "dbpm_x_reb",           # Defensive impact via rebounding (DBPM × REB%)
    "two_way_score",        # RAPTOR_off + |RAPTOR_def| (balanced two-way value)
]

# Non-metric identity/target columns to exclude from METRIC_CATALOG selection
_IDENTITY_COLS = {"player_id", "season", "team_id", "league_id", "rating", "minutes_per_game"}


class _MetricInfo(NamedTuple):
    label: str
    description: str
    use_per36: bool
    gruppo: str


# Catalog of all possible player_stats columns with metadata
METRIC_CATALOG: Dict[str, _MetricInfo] = {
    "points":               _MetricInfo("Punti",               "Punti per partita",                          True,  "Attacco"),
    "assists":              _MetricInfo("Assist",               "Assist per partita",                         True,  "Playmaking"),
    "rebounds":             _MetricInfo("Rimbalzi totali",      "Rimbalzi per partita",                       True,  "Rimbalzi"),
    "offensive_rebounds":   _MetricInfo("Rim. offensivi",       "Rimbalzi offensivi per partita",             True,  "Rimbalzi"),
    "defensive_rebounds":   _MetricInfo("Rim. difensivi",       "Rimbalzi difensivi per partita",             True,  "Rimbalzi"),
    "steals":               _MetricInfo("Palle rubate",         "Palle rubate per partita",                   True,  "Difesa"),
    "blocks":               _MetricInfo("Stoppate",             "Stoppate per partita",                       True,  "Difesa"),
    "turnovers":            _MetricInfo("Palle perse",          "Palle perse per partita",                    True,  "Efficienza"),
    "fouls":                _MetricInfo("Falli",                "Falli per partita",                          True,  "Disciplina"),
    "fg_pct":               _MetricInfo("FG%",                  "Percentuale tiri dal campo (effettivo)",      False, "Efficienza"),
    "three_point_pct":      _MetricInfo("3P%",                  "Percentuale tiri da tre punti",              False, "Efficienza"),
    "three_point_attempts": _MetricInfo("3PA/36",               "Tentativi da tre per 36 minuti",             True,  "Attacco"),
    "three_point_made":     _MetricInfo("3PM/36",               "Canestri da tre per 36 minuti",              True,  "Attacco"),
    "free_throw_pct":       _MetricInfo("FT%",                  "Percentuale tiri liberi",                    False, "Efficienza"),
    "free_throw_attempts":  _MetricInfo("FTA/36",               "Tiri liberi tentati per 36 minuti",          True,  "Efficienza"),
    "per":                  _MetricInfo("PIE",                  "Player Impact Estimate (PIE)",               False, "Avanzate"),
    "ts_pct":               _MetricInfo("TS%",                  "True Shooting Percentage",                   False, "Avanzate"),
    "usg_pct":              _MetricInfo("USG%",                 "Usage Rate",                                 False, "Avanzate"),
    "bpm":                  _MetricInfo("BPM",                  "Box Plus/Minus",                             False, "Avanzate"),
    "vorp":                 _MetricInfo("VORP",                 "Value Over Replacement Player",              False, "Avanzate"),
    "ws":                   _MetricInfo("Win Shares",           "Vittorie attribuite al giocatore",           False, "Avanzate"),
    "ws_per_48":            _MetricInfo("WS/48",                "Win Shares per 48 minuti",                   False, "Avanzate"),
    "obpm":                 _MetricInfo("OBPM",                 "Offensive Box Plus/Minus",                   False, "Avanzate"),
    "dbpm":                 _MetricInfo("DBPM",                 "Defensive Box Plus/Minus",                   False, "Avanzate"),
    "plus_minus":           _MetricInfo("+/-",                  "Plus/Minus grezzo",                          False, "Team"),
    "net_rating":           _MetricInfo("Net Rating",           "Net rating in campo",                        False, "Team"),
    "games_played":         _MetricInfo("Partite giocate",      "Numero di partite disputate",                False, "Utilizzo"),
    "games_started":        _MetricInfo("Da titolare",          "Partite giocate da titolare",                False, "Utilizzo"),
    "efg_pct":              _MetricInfo("eFG%",                 "Effective Field Goal Percentage",            False, "Efficienza"),
    "two_point_pct":        _MetricInfo("2P%",                  "Percentuale tiri da due punti",              False, "Efficienza"),
    "ast_pct":              _MetricInfo("AST%",                 "Assist Rate",                                False, "Playmaking"),
    "reb_pct":              _MetricInfo("REB%",                 "Rebound Rate",                               False, "Rimbalzi"),
    "tov_pct":              _MetricInfo("TOV%",                 "Turnover Rate",                              False, "Efficienza"),
    "stl_pct":              _MetricInfo("STL%",                 "Steal Rate",                                 False, "Difesa"),
    "blk_pct":              _MetricInfo("BLK%",                 "Block Rate",                                 False, "Difesa"),
    "orb_pct":              _MetricInfo("ORB%",                 "Offensive Rebound Rate",                     False, "Rimbalzi"),
    "drb_pct":              _MetricInfo("DRB%",                 "Defensive Rebound Rate",                     False, "Rimbalzi"),
    "ortg":                 _MetricInfo("ORtg",                 "Offensive Rating per 100 possessi",          False, "Team"),
    "drtg":                 _MetricInfo("DRtg",                 "Defensive Rating per 100 possessi",          False, "Team"),
    "net_rtg":              _MetricInfo("NetRtg",               "Net rating per 100 possessi",                False, "Team"),
    "on_net_rtg":           _MetricInfo("NetRtg On",            "Net rating quando in campo",                 False, "Team"),
    "net_rtg_diff":         _MetricInfo("NetRtg Diff",          "Differenziale on/off net rating",            False, "Team"),
    # --- New: Advanced rating models (DB schema) ----------------------------
    "spm":                  _MetricInfo("SPM",                  "Statistical Plus/Minus",                     False, "Avanzate"),
    "raptor_total":         _MetricInfo("RAPTOR",               "RAPTOR total (FiveThirtyEight)",             False, "Avanzate"),
    "raptor_off":           _MetricInfo("RAPTOR Off",           "RAPTOR offensive rating",                    False, "Avanzate"),
    "raptor_def":           _MetricInfo("RAPTOR Def",           "RAPTOR defensive rating",                    False, "Avanzate"),
    "lebron_total":         _MetricInfo("LEBRON",               "LEBRON impact model",                        False, "Avanzate"),
    "lebron_off":           _MetricInfo("LEBRON Off",           "LEBRON offensive component",                 False, "Avanzate"),
    "lebron_def":           _MetricInfo("LEBRON Def",           "LEBRON defensive component",                 False, "Avanzate"),
    "gm_sc":                _MetricInfo("GmSc",                 "Game Score per partita",                     False, "Avanzate"),
    "fic":                  _MetricInfo("FIC",                  "Floor Impact Counter",                       False, "Avanzate"),
    "ows":                  _MetricInfo("OWS",                  "Offensive Win Shares",                       False, "Avanzate"),
    "dws":                  _MetricInfo("DWS",                  "Defensive Win Shares",                       False, "Avanzate"),
    # --- New: Efficiency / hustle -------------------------------------------
    "scoring_efficiency":   _MetricInfo("Sc. Eff.",             "Scoring efficiency index",                   False, "Efficienza"),
    "ppsa":                 _MetricInfo("PPSA",                 "Points per shot attempt",                    False, "Efficienza"),
    "true_usg_pct":         _MetricInfo("TUSG%",                "True usage percentage",                      False, "Avanzate"),
    "foul_drawing_rate":    _MetricInfo("FDR",                  "Foul drawing rate",                          False, "Avanzate"),
    "rf_per_game":          _MetricInfo("RF/G",                 "Referee fouls drawn per game",               False, "Disciplina"),
    "hustle_index":         _MetricInfo("Hustle",               "Hustle index (aggressività)",                False, "Avanzate"),
    "three_par":            _MetricInfo("3PAr",                 "Three-point attempt rate",                   False, "Attacco"),
    # --- New: Clutch stats --------------------------------------------------
    "clutch_pts":           _MetricInfo("Clutch Pts",           "Punti in situazioni clutch",                 True,  "Clutch"),
    "clutch_ts_pct":        _MetricInfo("Clutch TS%",           "True shooting % in clutch",                  False, "Clutch"),
    "clutch_net_rtg":       _MetricInfo("Clutch NetRtg",        "Net rating in situazioni clutch",            False, "Clutch"),
    "clutch_efg_pct":       _MetricInfo("Clutch eFG%",          "eFG% in situazioni clutch",                  False, "Clutch"),
    "clutch_ast_to_tov":    _MetricInfo("Clutch A/T",           "Assist/turnover ratio in clutch",            False, "Clutch"),
    # --- New: Per-40 minute stats --------------------------------------------
    "pts_per_40":           _MetricInfo("Pts/40",               "Punti per 40 minuti",                        False, "Utilizzo"),
    "ast_per_40":           _MetricInfo("Ast/40",               "Assist per 40 minuti",                       False, "Utilizzo"),
    "tr_per_40":            _MetricInfo("Reb/40",               "Rimbalzi per 40 minuti",                     False, "Utilizzo"),
    "stl_per_40":           _MetricInfo("Stl/40",               "Palle rubate per 40 minuti",                 False, "Utilizzo"),
    "blk_per_40":           _MetricInfo("Blk/40",               "Stoppate per 40 minuti",                     False, "Utilizzo"),
}

# Columns that are already covered by the base FEATURE_COLS (per-36 computed
# or historical averages already baked in)
_BASE_COVERED_COLS = {
    "points", "assists", "rebounds", "steals", "blocks",
    "per", "ts_pct", "usg_pct", "bpm",
    # New base features (already in FEATURE_COLS as avg_*)
    "spm", "raptor_total", "raptor_off", "raptor_def",
    "lebron_total", "lebron_off", "lebron_def",
    "obpm", "dbpm",
    "gm_sc", "fic", "ows", "dws",
    "scoring_efficiency", "hustle_index", "foul_drawing_rate",
    "tov_pct", "ast_pct", "orb_pct", "drb_pct",
    "clutch_pts", "clutch_ts_pct", "clutch_net_rtg",
    "net_rtg_diff", "ortg_diff",
    "pts_per_40", "ast_per_40",
}


def get_available_metrics(player_stats_df: pd.DataFrame) -> List[str]:
    """Return catalog metric columns present in *player_stats_df*.

    Excludes identity/target columns, non-numeric columns, and columns already
    fully covered by the base FEATURE_COLS set.
    """
    available: List[str] = []
    numeric_cols = set(player_stats_df.select_dtypes(include=[np.number]).columns)
    for col in METRIC_CATALOG:
        if col in _IDENTITY_COLS:
            continue
        if col in _BASE_COVERED_COLS:
            continue
        if col not in player_stats_df.columns:
            continue
        if col not in numeric_cols:
            continue
        available.append(col)
    return available

# Encoding for all position strings (pure + hybrid)
# Kept for backward compatibility with saved models; new code uses one-hot columns.
POSITION_ENCODING: Dict[str, int] = {
    "PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4,
    "PG/SG": 5, "SG/SF": 6, "SF/PF": 7, "PF/C": 8, "SG/PF": 9,
}

# Primary positions for one-hot encoding
_PRIMARY_POSITIONS = ("PG", "SG", "SF", "PF", "C")



class PerformanceModel:
    """XGBoost player rating predictor for basketball."""

    def __init__(self) -> None:
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
            device=_XGB_DEVICE,
        )
        self.feature_names: List[str] = FEATURE_COLS.copy()
        self.is_trained: bool = False
        self._shap_explainer: Optional[Any] = None
        self.data_signature: str = ""
        #: Maps ruolo_combinato / ruolo_offensivo / ruolo_difensivo → ordinal int.
        #: Built during prepare_features(); unknown roles at inference default to 0.
        self.role_encoding:     Dict[str, int] = {}
        self.role_off_encoding: Dict[str, int] = {}
        self.role_def_encoding: Dict[str, int] = {}
        self._last_metrics: Optional[dict] = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        stat_row: pd.Series,
        age: int,
        position: str,
        player_stats_history: pd.DataFrame,
        extra_metrics: Optional[List[str]] = None,
        league_max_games: Optional[Dict[int, int]] = None,
    ) -> Dict[str, float]:
        """Build a feature row from a player-stats row.

        Includes all base FEATURE_COLS (per-36 volume, career signals, advanced
        DB-schema metrics, all three DB role encodings, interaction features)
        plus any extra_metrics requested by the caller.

        Args:
            stat_row:             A single player-season stats row.
            age:                  Player age for this season.
            position:             Player position string (e.g. ``"PG"``).
            player_stats_history: All stats rows for this player up to (and
                                  including) the current season.
            extra_metrics:        Additional METRIC_CATALOG columns to include.
            league_max_games:     Optional ``{league_id: max_games}`` lookup for
                                  league-aware ``durability_score`` normalisation.
        """
        if extra_metrics is None:
            extra_metrics = []
        _mpg_raw = stat_row.get("minutes_per_game", 0)
        mpg = float(_mpg_raw) if not pd.isna(_mpg_raw) and _mpg_raw > 0 else 1.0

        def per36(col: str) -> float:
            v = stat_row.get(col, 0)
            return (float(v) if not pd.isna(v) else 0.0) / mpg * 36

        def hist_avg(col: str, fallback: float = 0.0) -> float:
            """Historical average of *col* from career so far; fall back to current row."""
            if not player_stats_history.empty and col in player_stats_history.columns:
                vals = player_stats_history[col].dropna().tolist()
                if vals:
                    return float(np.mean(vals))
            v = stat_row.get(col, fallback)
            return fallback if pd.isna(v) else float(v)

        # Career stats up to this season
        ratings = player_stats_history["rating"].tolist()
        last3 = ratings[-3:]
        weights = ([0.2, 0.3, 0.5] if len(last3) == 3
                   else ([0.4, 0.6] if len(last3) == 2 else [1.0]))
        form_score = sum(r * w for r, w in zip(last3, weights))
        mean_r = float(np.mean(ratings)) if ratings else 6.5
        std_r  = float(np.std(ratings))  if len(ratings) > 1 else 0.0
        consistency = float(np.clip(1.0 - (std_r / mean_r) if mean_r > 0 else 0.0, 0, 1))
        trajectory = (
            float(np.polyfit(np.arange(len(ratings), dtype=float), ratings, 1)[0])
            if len(ratings) >= 2 else 0.0
        )

        # TS% efficiency trend over career (slope of ts_pct per season)
        if (
            not player_stats_history.empty
            and "ts_pct" in player_stats_history.columns
            and len(player_stats_history) >= 2
        ):
            ts_vals = player_stats_history["ts_pct"].dropna().tolist()
            ts_trend = float(np.polyfit(np.arange(len(ts_vals), dtype=float), ts_vals, 1)[0]) if len(ts_vals) >= 2 else 0.0
        else:
            ts_trend = 0.0

        # Durability: average fraction of league max games played (league-aware)
        if not player_stats_history.empty and "games_played" in player_stats_history.columns:
            gp_vals = player_stats_history["games_played"].dropna()
            if league_max_games and "league_id" in player_stats_history.columns:
                dur_list = [
                    gp / max(1, league_max_games.get(
                        _safe_league_id(lid),
                        LEAGUE_MAX_GAMES_DEFAULT,
                    ))
                    for gp, lid in zip(
                        gp_vals.tolist(),
                        player_stats_history.loc[gp_vals.index, "league_id"].tolist(),
                    )
                ]
                durability = float(np.clip(np.mean(dur_list), 0.0, 1.0)) if dur_list else 0.5
            else:
                durability = float(np.clip(np.mean(gp_vals.tolist()) / LEAGUE_MAX_GAMES_DEFAULT, 0.0, 1.0)) if not gp_vals.empty else 0.5
        else:
            gp_cur = float(stat_row.get("games_played", 40))
            lid_cur = stat_row.get("league_id", 0)
            max_g = (league_max_games or {}).get(
                _safe_league_id(lid_cur),
                LEAGUE_MAX_GAMES_DEFAULT,
            )
            durability = float(np.clip(gp_cur / max(1, max_g), 0.0, 1.0))

        avg_per  = hist_avg("per",     12.0)
        avg_ts   = hist_avg("ts_pct",   0.52)
        avg_usg  = hist_avg("usg_pct", 18.0)
# avg_bpm omitted from features (= obpm + dbpm)
        avg_obpm = hist_avg("obpm",     0.0)
        avg_dbpm = hist_avg("dbpm",     0.0)

        # Engineered interaction features
        avg_reb_pct = hist_avg("reb_pct", 5.0)
        obpm_x_usg  = avg_obpm * avg_usg / 100.0
        dbpm_x_reb  = avg_dbpm * avg_reb_pct / 100.0

        avg_raptor_off = hist_avg("raptor_off", 0.0)
        avg_raptor_def = hist_avg("raptor_def", 0.0)
        # Two-way score: offensive value + magnitude of defensive contribution
        two_way_score = avg_raptor_off + abs(avg_raptor_def)

        peak_age = _peak_age(position)
        # One-hot position encoding
        _prim = _primary_pos(position)
        _is_hybrid = int(_prim != position)

        row: Dict[str, float] = {
            # --- Identity / context ------------------------------------------
            "age":                  float(age),
            # One-hot position columns
            "pos_PG":               1.0 if _prim == "PG" else 0.0,
            "pos_SG":               1.0 if _prim == "SG" else 0.0,
            "pos_SF":               1.0 if _prim == "SF" else 0.0,
            "pos_PF":               1.0 if _prim == "PF" else 0.0,
            "pos_C":                1.0 if _prim == "C"  else 0.0,
            "pos_hybrid":           float(_is_hybrid),
            "competition_enc":      float(COMPETITION_ENCODING.get(
                str(stat_row.get("competition", "RS")), 0
            )),
            # --- DB role encodings (all three dimensions) --------------------
            "role_enc":             float(self.role_encoding.get(
                str(stat_row.get("ruolo_combinato", "") or "").strip(), 0
            )),
            "role_off_enc":         float(self.role_off_encoding.get(
                str(stat_row.get("ruolo_offensivo", "") or "").strip(), 0
            )),
            "role_def_enc":         float(self.role_def_encoding.get(
                str(stat_row.get("ruolo_difensivo", "") or "").strip(), 0
            )),
            # --- Per-36 volume stats -----------------------------------------
            "pts_per_36":           per36("points"),
            "ast_per_36":           per36("assists"),
            "reb_per_36":           per36("rebounds"),
            "stl_per_36":           per36("steals"),
            "blk_per_36":           per36("blocks"),
            # --- Core advanced metrics ---------------------------------------
            "avg_per":              avg_per,
            "avg_ts_pct":           avg_ts,
            "avg_usg_pct":          avg_usg,
            # avg_bpm omitted (= avg_obpm + avg_dbpm)
            "avg_obpm":             avg_obpm,
            "avg_dbpm":             avg_dbpm,
            # --- Career signals ----------------------------------------------
            "form_score":           form_score,
            "consistency_score":    consistency,
            "career_trajectory":    trajectory,
            "age_vs_peak_age":      float(age - peak_age),
            "ts_efficiency_trend":  ts_trend,
            "durability_score":     durability,
            # --- Composite / model-based ratings (DB schema) -----------------
            "avg_spm":              hist_avg("spm", 0.0),
            "avg_raptor_off":       avg_raptor_off,
            "avg_raptor_def":       avg_raptor_def,
            # avg_raptor_total omitted (= raptor_off + raptor_def)
            "avg_lebron_off":       hist_avg("lebron_off", 0.0),
            "avg_lebron_def":       hist_avg("lebron_def", 0.0),
            # avg_lebron_total omitted (= lebron_off + lebron_def)
            "avg_gm_sc":            hist_avg("gm_sc", 0.0),
            "avg_fic":              hist_avg("fic", 0.0),
            "avg_ows":              hist_avg("ows", 0.0),
            "avg_dws":              hist_avg("dws", 0.0),
            # --- Efficiency / hustle -----------------------------------------
            "avg_scoring_efficiency": hist_avg("scoring_efficiency", 0.0),
            "avg_hustle_index":       hist_avg("hustle_index", 0.0),
            "avg_foul_drawing_rate":  hist_avg("foul_drawing_rate", 0.0),
            # --- Rate stats (normalised percentages) -------------------------
            "avg_tov_pct":          hist_avg("tov_pct", 10.0),
            "avg_ast_pct":          hist_avg("ast_pct", 10.0),
            "avg_orb_pct":          hist_avg("orb_pct", 3.0),
            "avg_drb_pct":          hist_avg("drb_pct", 12.0),
            # --- Clutch performance ------------------------------------------
            "clutch_pts_per_36":    per36("clutch_pts"),
            "avg_clutch_ts_pct":    hist_avg("clutch_ts_pct", 0.0),
            "avg_clutch_net_rtg":   hist_avg("clutch_net_rtg", 0.0),
            # --- On/Off impact -----------------------------------------------
            "avg_net_rtg_diff":     hist_avg("net_rtg_diff", 0.0),
            "avg_ortg_diff":        hist_avg("ortg_diff", 0.0),
            # --- Per-40 projection features ----------------------------------
            "avg_pts_per_40":       hist_avg("pts_per_40", 0.0),
            "avg_ast_per_40":       hist_avg("ast_per_40", 0.0),
            # --- Starter status (from Boxscore.SF) ----------------------------
            # NOTE: Was previously missing from _build_row, causing a reverse
            # training/inference skew (training always 0, inference non-zero).
            "avg_starter_pct":      hist_avg("starter_pct", 0.5),
            # --- Engineered interaction features -----------------------------
            "obpm_x_usg":           obpm_x_usg,
            "dbpm_x_reb":           dbpm_x_reb,
            "two_way_score":        two_way_score,
        }

        # PO vs RS delta and PO games played from historical data
        po_feats = compute_po_features(player_stats_history)
        row["po_vs_rs_delta"]  = po_feats["po_vs_rs_delta"]
        row["po_games_played"] = po_feats["po_games_played"]
        row["has_po_history"]  = po_feats["has_po_history"]

        # Extra metrics requested by the caller
        for col in extra_metrics:
            info = METRIC_CATALOG.get(col)
            if info is None:
                continue
            if col not in stat_row.index:
                row[f"{col}_per_36" if info.use_per36 else f"avg_{col}"] = 0.0
                continue
            if info.use_per36:
                row[f"{col}_per_36"] = per36(col)
            else:
                if col in player_stats_history.columns:
                    vals = player_stats_history[col].dropna().tolist()
                    row[f"avg_{col}"] = float(np.mean(vals)) if vals else float(stat_row.get(col, 0))
                else:
                    row[f"avg_{col}"] = float(stat_row.get(col, 0))

        return row

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build training (X, y) from all player-season data.

        Rebuilds all three role encodings (``ruolo_combinato``,
        ``ruolo_offensivo``, ``ruolo_difensivo``) so codes are consistent
        between training and inference.

        Args:
            data:          dict with keys ``player_stats`` and ``players``.
            extra_metrics: additional columns from METRIC_CATALOG to include
                           as features (on top of the base FEATURE_COLS).
            split_season:  if provided (e.g. ``"2022-23"``), only rows from
                           seasons ≤ split_season are used for training.
        """
        if extra_metrics is None:
            extra_metrics = []
        player_stats = data["player_stats"]
        players      = data["players"]

        # Apply temporal filter if split_season is provided
        if split_season is not None:
            player_stats = player_stats[player_stats["season"] <= split_season]

        # --- Build role encodings for all three DB role columns ---------------
        def _build_encoding(col: str) -> Dict[str, int]:
            if col not in player_stats.columns:
                return {}
            unique = sorted({
                str(v).strip()
                for v in player_stats[col].dropna()
                if str(v).strip()
            })
            return {role: idx + 1 for idx, role in enumerate(unique)}

        self.role_encoding     = _build_encoding("ruolo_combinato")
        self.role_off_encoding = _build_encoding("ruolo_offensivo")
        self.role_def_encoding = _build_encoding("ruolo_difensivo")

        if self.role_encoding:
            logger.info(
                "[PerformanceModel] Role encoding – combinato: %d, offensivo: %d, difensivo: %d",
                len(self.role_encoding), len(self.role_off_encoding), len(self.role_def_encoding),
            )

        # --- Extra metric feature names ---------------------------------------
        extra_feature_names: List[str] = []
        for col in extra_metrics:
            info = METRIC_CATALOG.get(col)
            if info is None:
                continue
            feat_name = f"{col}_per_36" if info.use_per36 else f"avg_{col}"
            if feat_name not in extra_feature_names:
                extra_feature_names.append(feat_name)

        all_feature_names = FEATURE_COLS + extra_feature_names

        if extra_metrics:
            logger.info("[PerformanceModel] Metriche extra (%d): %s", len(extra_metrics), extra_metrics)

        birth_year_map: Dict[int, int] = {
            _to_int(row["id"]): 2024 - int(row["age"])
            for _, row in players.iterrows()
            if row.get("age") is not None and not (isinstance(row.get("age"), float) and np.isnan(row["age"]))
        }
        position_map: Dict[int, str] = {
            _to_int(row["id"]): str(row["position"])
            for _, row in players.iterrows()
        }

        rows:    List[Dict[str, float]] = []
        targets: List[float]            = []
        season_years: List[int]         = []

        # Build a {league_id: max_games} lookup for league-aware durability_score.
        # Uses the ``max_games`` column when present (generated data) or falls back
        # to the LEAGUE_MAX_GAMES_BY_NAME constant dict keyed by league name.
        league_max_games: Optional[Dict[int, int]] = None
        leagues_df = data.get("leagues")
        if leagues_df is not None and not leagues_df.empty and "id" in leagues_df.columns:
            if "max_games" in leagues_df.columns:
                league_max_games = {
                    int(_to_int(r["id"])): int(r["max_games"])
                    for _, r in leagues_df.iterrows()
                    if r.get("max_games") and not (isinstance(r["max_games"], float) and np.isnan(r["max_games"]))
                }
            elif "name" in leagues_df.columns:
                league_max_games = {
                    int(_to_int(r["id"])): LEAGUE_MAX_GAMES_BY_NAME.get(
                        str(r["name"]).strip(), LEAGUE_MAX_GAMES_DEFAULT
                    )
                    for _, r in leagues_df.iterrows()
                }

        for pid, grp in player_stats.groupby("player_id"):
            grp = grp.sort_values("season")
            pid = _to_int(pid)
            pos = position_map.get(pid, "PG")
            by  = birth_year_map.get(pid)
            if by is None:
                continue
            for _, stat in grp.iterrows():
                season_str = str(stat["season"])
                try:
                    year = int(season_str.split("-")[0])
                except Exception:
                    year = 2024
                age = year - by
                if age < 14 or age > 45:
                    continue
                history_so_far = grp[grp["season"] < stat["season"]]
                rows.append(self._build_row(stat, age, pos, history_so_far, extra_metrics=extra_metrics, league_max_games=league_max_games))
                targets.append(float(stat["rating"]))
                season_years.append(year)

        X = pd.DataFrame(rows, columns=all_feature_names)
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.debug("[PerformanceModel] fillna(0) applied to %d column(s): %s", len(nan_cols), nan_cols)
        X = X.fillna(0.0)
        y = np.array(targets, dtype=float)
        # Store season years for chronological splitting in train()
        self._last_season_years: List[int] = season_years
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        cv_folds: int = 0,
    ) -> Dict[str, Any]:
        """Train the model and return RMSE metrics.

        Uses a **chronological three-way split** to prevent temporal data
        leakage in conformal calibration:

        * **Train (70%)** – oldest seasons; used to fit XGBoost weights.
        * **Val (15%)** – next chronological block; used for early-stopping.
        * **Conformal holdout (15%)** – most-recent seasons; reserved for
          computing empirical residuals for prediction-interval calibration.

        Args:
            data:          dict with keys ``player_stats`` and ``players``.
            extra_metrics: additional METRIC_CATALOG columns to include as features.
            cv_folds:      when > 1, run TimeSeriesSplit cross-validation **in
                           addition** to the standard split.  CV scores are
                           returned as ``cv_mean_rmse`` / ``cv_std_rmse`` but
                           the final model is always re-fitted on the training
                           portion.
        """
        if extra_metrics is None:
            extra_metrics = []
        logger.info("[PerformanceModel] Building feature matrix …")
        X, y = self.prepare_features(data, extra_metrics=extra_metrics)
        self.feature_names = list(X.columns)
        logger.info("[PerformanceModel] Training on %d samples, %d features …", len(X), len(self.feature_names))

        # --- Chronological split (A3) ----------------------------------------
        # Sort rows by season year to prevent temporal leakage.
        season_years = np.array(getattr(self, "_last_season_years", [0] * len(X)))
        sort_idx = np.argsort(season_years, kind="stable")
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y[sort_idx]

        n = len(X)
        if n >= 40:
            # Conformal holdout: most-recent 15%; val: next 15%; train: oldest 70%
            conf_start = int(n * 0.85)
            val_start  = int(n * 0.70)
            X_train, y_train     = X.iloc[:val_start],   y[:val_start]
            X_val,   y_val       = X.iloc[val_start:conf_start], y[val_start:conf_start]
            X_conformal          = X.iloc[conf_start:]
            y_conformal          = y[conf_start:]
        else:
            # Very small datasets: skip the conformal split to avoid tiny sets
            X_conformal = y_conformal = None
            split = max(1, int(n * 0.85))
            X_train, y_train = X.iloc[:split],  y[:split]
            X_val,   y_val   = X.iloc[split:],  y[split:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_tr_pred = self.model.predict(X_train)
        y_va_pred = self.model.predict(X_val)

        train_rmse = float(np.sqrt(np.mean((y_tr_pred - y_train) ** 2)))
        val_rmse   = float(np.sqrt(np.mean((y_va_pred - y_val)   ** 2)))
        val_mae    = float(np.mean(np.abs(y_va_pred - y_val)))
        # R² = 1 − SS_res / SS_tot
        ss_res = float(np.sum((y_va_pred - y_val) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self.is_trained = True
        logger.info(
            "[PerformanceModel] Train RMSE=%.4f  Val RMSE=%.4f  Val MAE=%.4f  Val R²=%.4f",
            train_rmse, val_rmse, val_mae, val_r2,
        )

        metrics: Dict[str, Any] = {
            "train_rmse": train_rmse,
            "val_rmse":   val_rmse,
            "val_mae":    val_mae,
            "val_r2":     val_r2,
        }

        # --- Baseline models (A4) --------------------------------------------
        baselines = compute_baselines(X_tr_sc, y_train, X_va_sc, y_val)
        metrics.update(baselines)
        metrics["xgboost_vs_baseline_delta"] = baselines["baseline_mean_val_rmse"] - val_rmse

        # --- Dataset signature (A17) -----------------------------------------
        sig = _compute_data_signature(X, y)
        self.data_signature = sig
        metrics["data_signature"] = sig

        # Compute conformal residuals on the true holdout (never seen by model).
        if X_conformal is not None and len(X_conformal) >= 10:
            y_conf_pred = self.model.predict(X_conformal)
            metrics["conformal_residuals"] = np.abs(y_conf_pred - y_conformal).tolist()
            logger.info(
                "[PerformanceModel] Conformal holdout: %d samples  "
                "mean_residual=%.4f",
                len(X_conformal),
                float(np.mean(np.abs(y_conf_pred - y_conformal))),
            )

        # --- TimeSeriesSplit cross-validation (A3) ----------------------------
        if cv_folds > 1:
            logger.info("[PerformanceModel] Running %d-fold TimeSeriesSplit CV …", cv_folds)
            tss = TimeSeriesSplit(n_splits=cv_folds)
            X_np = X.values.astype(float)
            best_n_estimators = int(getattr(self.model, "best_iteration", 300) or 300)
            cv_rmses: List[float] = []
            for fold, (tr_idx, va_idx) in enumerate(tss.split(X_np), 1):
                Xtr, Xva = X_np[tr_idx], X_np[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                _m = XGBRegressor(
                    n_estimators=best_n_estimators, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
                    device=_XGB_DEVICE,
                )
                _m.fit(Xtr, ytr, verbose=False)
                fold_rmse = float(np.sqrt(np.mean((_m.predict(Xva) - yva) ** 2)))
                cv_rmses.append(fold_rmse)
                logger.info("  Fold %d/%d  RMSE=%.4f  (n_estimators=%d)", fold, cv_folds, fold_rmse, best_n_estimators)
            metrics["cv_mean_rmse"] = float(np.mean(cv_rmses))
            metrics["cv_std_rmse"]  = float(np.std(cv_rmses))
            logger.info(
                "[PerformanceModel] CV RMSE=%.4f ±%.4f  (n_estimators=%d)",
                metrics["cv_mean_rmse"], metrics["cv_std_rmse"], best_n_estimators,
            )

        # --- Data lineage (D6) -----------------------------------------------
        lineage = {
            "trained_at":     _dt.now(timezone.utc).isoformat(),
            "n_samples":      len(X),
            "n_features":     len(self.feature_names),
            "data_signature": metrics.get("data_signature", ""),
        }
        metrics["lineage"] = lineage
        logger.info("[PerformanceModel] Lineage: %s", lineage)

        self._last_metrics = metrics
        return metrics

    def predict_from_features(self, feature_dict: Dict[str, float]) -> float:
        """Predict rating from an already-engineered feature dict."""
        if not self.is_trained:
            return float(np.clip(feature_dict.get("form_score", 6.5), 4.0, 10.0))
        logger.debug("[PerformanceModel] predict_from_features called; stored data_signature=%s", self.data_signature)
        arr = np.array([[feature_dict.get(c, 0.0) for c in self.feature_names]], dtype=float)
        return float(np.clip(self.model.predict(arr)[0], 3.5, 10.0))

    def get_shap_values(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        if not self.is_trained or not _SHAP_AVAILABLE:
            return {}
        try:
            if self._shap_explainer is None:
                self._shap_explainer = _shap.TreeExplainer(self.model)
            arr = np.array([[feature_dict.get(c, 0.0) for c in self.feature_names]], dtype=float)
            sv = self._shap_explainer.shap_values(arr)
            return {name: float(sv[0][i]) for i, name in enumerate(self.feature_names)}
        except Exception:
            return {}

    def feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _update_registry(self, model_path: str, metrics: Optional[dict] = None) -> None:
        """Append an entry to the model registry JSON in the model directory."""
        registry_path = Path(model_path).parent / "registry.json"
        try:
            if registry_path.exists():
                registry = _json.loads(registry_path.read_text())
            else:
                registry = []
            entry = {
                "path":             str(Path(model_path).name),
                "saved_at":         _dt.now(timezone.utc).isoformat(),
                "metrics":          metrics or {},
                "git_sha":          _get_git_sha(),
                "feature_list":     list(self.feature_names) if self.feature_names else [],
                "metrics_by_league": self._last_metrics.get("by_league", {}) if self._last_metrics else {},
                "metrics_by_role":   self._last_metrics.get("by_role", {}) if self._last_metrics else {},
            }
            registry.append(entry)
            registry_path.write_text(_json.dumps(registry, indent=2))
        except Exception as exc:
            logger.warning("[O5] Could not update model registry: %s", exc)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":              self.model,
            "feature_names":      self.feature_names,
            "role_encoding":      self.role_encoding,
            "role_off_encoding":  self.role_off_encoding,
            "role_def_encoding":  self.role_def_encoding,
            "data_signature":     self.data_signature,
        }, path)
        logger.info("[PerformanceModel] Saved to %s", path)
        self._update_registry(path, self._last_metrics)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.model              = payload["model"]
        self.feature_names      = payload["feature_names"]
        self.role_encoding      = payload.get("role_encoding", {})
        self.role_off_encoding  = payload.get("role_off_encoding", {})
        self.role_def_encoding  = payload.get("role_def_encoding", {})
        self.data_signature     = payload.get("data_signature", "")
        self.is_trained         = True
        self._shap_explainer    = None
        logger.info("[PerformanceModel] Loaded from %s", path)


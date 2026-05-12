"""Basketball XGBoost performance model.

Predicts player rating (0-10) from basketball-specific features:
per-36 stats, advanced metrics (PER, BPM, TS%, USG%), age, position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Competition-type encoding used as a training feature and prediction context
COMPETITION_ENCODING: Dict[str, int] = {
    "RS": 0,          # Regular Season
    "PO": 1,          # Playoffs
    "CUP": 2,         # Cup competition
    "SUPERCUP": 3,    # Super Cup / preseason
}

# Feature columns (order matters for scaling)
FEATURE_COLS: List[str] = [
    "age",
    "position_enc",
    "competition_enc",
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
    "stl_per_36",
    "blk_per_36",
    "avg_per",
    "avg_ts_pct",
    "avg_usg_pct",
    "avg_bpm",
    "form_score",
    "consistency_score",
    "career_trajectory",
    "age_vs_peak_age",
    "po_vs_rs_delta",
    "po_games_played",
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
    "fg_pct":               _MetricInfo("FG%",                  "Percentuale tiri dal campo",                 False, "Efficienza"),
    "three_point_pct":      _MetricInfo("3P%",                  "Percentuale tiri da tre punti",              False, "Efficienza"),
    "three_point_attempts": _MetricInfo("3PA/36",               "Tentativi da tre per 36 minuti",             True,  "Attacco"),
    "three_point_made":     _MetricInfo("3PM/36",               "Canestri da tre per 36 minuti",              True,  "Attacco"),
    "free_throw_pct":       _MetricInfo("FT%",                  "Percentuale tiri liberi",                    False, "Efficienza"),
    "free_throw_attempts":  _MetricInfo("FTA/36",               "Tiri liberi tentati per 36 minuti",          True,  "Efficienza"),
    "per":                  _MetricInfo("PER",                  "Player Efficiency Rating",                   False, "Avanzate"),
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
    "ortg":                 _MetricInfo("ORtg",                 "Offensive Rating per 100 possessi",          False, "Team"),
    "drtg":                 _MetricInfo("DRtg",                 "Defensive Rating per 100 possessi",          False, "Team"),
}

# Columns that are already covered by the base FEATURE_COLS (per-36 computed)
_BASE_COVERED_COLS = {"points", "assists", "rebounds", "steals", "blocks",
                      "per", "ts_pct", "usg_pct", "bpm"}


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
POSITION_ENCODING: Dict[str, int] = {
    "PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4,
    "PG/SG": 5, "SG/SF": 6, "SF/PF": 7, "PF/C": 8, "SG/PF": 9,
}

POSITIONAL_PEAK_AGES: Dict[str, int] = {
    "PG": 26, "SG": 25, "SF": 26, "PF": 27, "C": 28,
    "PG/SG": 25, "SG/SF": 25, "SF/PF": 26, "PF/C": 27, "SG/PF": 26,
}


def _primary_pos(pos: str) -> str:
    return pos.split("/")[0]


def _peak_age(pos: str) -> int:
    return POSITIONAL_PEAK_AGES.get(pos, POSITIONAL_PEAK_AGES.get(_primary_pos(pos), 26))


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
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = FEATURE_COLS.copy()
        self.is_trained: bool = False
        self._shap_explainer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        stat_row: pd.Series,
        age: int,
        position: str,
        player_stats_history: pd.DataFrame,
        extra_metrics: List[str] = [],
    ) -> Dict[str, float]:
        """Build a feature row from a player-stats row."""
        mpg = float(stat_row.get("minutes_per_game", 0))
        if mpg <= 0:
            mpg = 1.0

        def per36(col: str) -> float:
            return float(stat_row.get(col, 0)) / mpg * 36

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

        avg_per = float(np.mean(player_stats_history["per"].tolist())) if not player_stats_history.empty else float(stat_row.get("per", 12))
        avg_ts  = float(np.mean(player_stats_history["ts_pct"].tolist())) if not player_stats_history.empty else float(stat_row.get("ts_pct", 0.52))
        avg_usg = float(np.mean(player_stats_history["usg_pct"].tolist())) if not player_stats_history.empty else float(stat_row.get("usg_pct", 18))
        avg_bpm = float(np.mean(player_stats_history["bpm"].tolist())) if not player_stats_history.empty else float(stat_row.get("bpm", -1))

        pos_enc = float(POSITION_ENCODING.get(position, POSITION_ENCODING.get(_primary_pos(position), 0)))
        peak_age = _peak_age(position)

        row: Dict[str, float] = {
            "age":                float(age),
            "position_enc":       pos_enc,
            "competition_enc":    float(COMPETITION_ENCODING.get(
                str(stat_row.get("competition", "RS")), 0
            )),
            "pts_per_36":         per36("points"),
            "ast_per_36":         per36("assists"),
            "reb_per_36":         per36("rebounds"),
            "stl_per_36":         per36("steals"),
            "blk_per_36":         per36("blocks"),
            "avg_per":            avg_per,
            "avg_ts_pct":         avg_ts,
            "avg_usg_pct":        avg_usg,
            "avg_bpm":            avg_bpm,
            "form_score":         form_score,
            "consistency_score":  consistency,
            "career_trajectory":  trajectory,
            "age_vs_peak_age":    float(age - peak_age),
        }

        # PO vs RS delta and PO games played from historical data
        if "competition" in player_stats_history.columns:
            rs_hist = player_stats_history[player_stats_history["competition"] == "RS"]
            po_hist = player_stats_history[player_stats_history["competition"] == "PO"]
            rs_mean = float(rs_hist["rating"].mean()) if not rs_hist.empty else float("nan")
            po_mean = float(po_hist["rating"].mean()) if not po_hist.empty else float("nan")
            if not (np.isnan(rs_mean) or np.isnan(po_mean)):
                po_vs_rs_delta = po_mean - rs_mean
            else:
                po_vs_rs_delta = 0.0
            po_gp = (
                float(po_hist["games_played"].sum())
                if (not po_hist.empty and "games_played" in po_hist.columns)
                else 0.0
            )
        else:
            po_vs_rs_delta = 0.0
            po_gp = 0.0
        row["po_vs_rs_delta"]  = po_vs_rs_delta
        row["po_games_played"] = po_gp

        # Extra metrics requested by the caller
        for col in extra_metrics:
            info = METRIC_CATALOG.get(col)
            if info is None:
                continue
            if col not in stat_row.index:
                print(f"[PerformanceModel] Warning: column '{col}' not found in stat_row, using 0.0")
                if info.use_per36:
                    row[f"{col}_per_36"] = 0.0
                else:
                    row[f"avg_{col}"] = 0.0
                continue
            if info.use_per36:
                row[f"{col}_per_36"] = per36(col)
            else:
                # Use historical average if available, otherwise current value
                if col in player_stats_history.columns:
                    vals = player_stats_history[col].dropna().tolist()
                    row[f"avg_{col}"] = float(np.mean(vals)) if vals else float(stat_row.get(col, 0))
                else:
                    row[f"avg_{col}"] = float(stat_row.get(col, 0))

        return row

    def prepare_features(
        self, data: Dict[str, Any], extra_metrics: List[str] = []
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build training (X, y) from all player-season data.

        Args:
            data: dict with keys ``player_stats`` and ``players``.
            extra_metrics: additional columns from METRIC_CATALOG to include
                as features (on top of the base FEATURE_COLS).
        """
        player_stats = data["player_stats"]
        players      = data["players"]

        # Determine the full feature column list for this run
        extra_feature_names: List[str] = []
        for col in extra_metrics:
            info = METRIC_CATALOG.get(col)
            if info is None:
                continue
            feat_name = f"{col}_per_36" if info.use_per36 else f"avg_{col}"
            if feat_name not in extra_feature_names:
                extra_feature_names.append(feat_name)

        all_feature_names = FEATURE_COLS + extra_feature_names
        self.feature_names = all_feature_names

        if extra_metrics:
            print(f"[PerformanceModel] Metriche extra ({len(extra_metrics)}): {extra_metrics}")

        birth_year_map: Dict[int, int] = {
            int(row["id"]): 2024 - int(row["age"])
            for _, row in players.iterrows()
        }
        position_map: Dict[int, str] = {
            int(row["id"]): str(row["position"])
            for _, row in players.iterrows()
        }

        rows:    List[Dict[str, float]] = []
        targets: List[float]            = []

        for pid, grp in player_stats.groupby("player_id"):
            grp = grp.sort_values("season")
            pid = int(pid)
            pos = position_map.get(pid, "PG")
            by  = birth_year_map.get(pid)
            if by is None:
                continue
            for idx, stat in grp.iterrows():
                season_str = str(stat["season"])
                # approximate year from season string like "2023-24"
                try:
                    year = int(season_str.split("-")[0])
                except Exception:
                    year = 2024
                age = year - by
                if age < 14 or age > 45:
                    continue
                history_so_far = grp[grp["season"] <= stat["season"]]
                rows.append(self._build_row(stat, age, pos, history_so_far, extra_metrics=extra_metrics))
                targets.append(float(stat["rating"]))

        X = pd.DataFrame(rows, columns=self.feature_names)
        y = np.array(targets, dtype=float)
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any], extra_metrics: List[str] = []) -> Dict[str, float]:
        """Train the model and return RMSE metrics.

        Args:
            data: dict with keys ``player_stats`` and ``players``.
            extra_metrics: additional METRIC_CATALOG columns to include as features.
        """
        print("[PerformanceModel] Building feature matrix …")
        X, y = self.prepare_features(data, extra_metrics=extra_metrics)
        print(f"[PerformanceModel] Training on {len(X):,} samples …")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        X_tr_sc = self.scaler.fit_transform(X_train)
        X_va_sc = self.scaler.transform(X_val)

        self.model.fit(X_tr_sc, y_train, eval_set=[(X_va_sc, y_val)], verbose=False)

        train_rmse = float(np.sqrt(np.mean((self.model.predict(X_tr_sc) - y_train) ** 2)))
        val_rmse   = float(np.sqrt(np.mean((self.model.predict(X_va_sc) - y_val) ** 2)))
        self.is_trained = True
        print(f"[PerformanceModel] Train RMSE={train_rmse:.4f}  Val RMSE={val_rmse:.4f}")
        return {"train_rmse": train_rmse, "val_rmse": val_rmse}

    def predict_from_features(self, feature_dict: Dict[str, float]) -> float:
        """Predict rating from an already-engineered feature dict."""
        if not self.is_trained:
            return float(np.clip(feature_dict.get("form_score", 6.5), 4.0, 10.0))
        row = pd.DataFrame(
            [[feature_dict.get(c, 0.0) for c in self.feature_names]],
            columns=self.feature_names,
        )
        scaled = self.scaler.transform(row)
        return float(np.clip(self.model.predict(scaled)[0], 3.5, 10.0))

    def get_shap_values(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        if not self.is_trained or not _SHAP_AVAILABLE:
            return {}
        try:
            if self._shap_explainer is None:
                self._shap_explainer = _shap.TreeExplainer(self.model)
            row = pd.DataFrame(
                [[feature_dict.get(c, 0.0) for c in self.feature_names]],
                columns=self.feature_names,
            )
            scaled = self.scaler.transform(row)
            sv = self._shap_explainer.shap_values(scaled)
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

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names}, path)
        print(f"[PerformanceModel] Saved to {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.model         = payload["model"]
        self.scaler        = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.is_trained    = True
        self._shap_explainer = None
        print(f"[PerformanceModel] Loaded from {path}")

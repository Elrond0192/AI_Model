"""Basketball Performance AI – Streamlit Training GUI.

Launch with:
    streamlit run gui/app.py

The GUI covers three sections:
  1. Data source  – Azure SQL Server connection string OR CSV upload
  2. Training     – hyper-parameter configuration, feature selection, run training
  3. Results      – metrics (RMSE, MAE, R²), SHAP feature importance chart,
                   model download
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHAP_SAMPLE_SIZE = 50


def _safe_model_dir(user_input: str) -> Path:
    """Sanitize a user-supplied directory path to prevent path traversal.

    Only allows relative paths composed of safe components (no ``..``, no
    leading ``/`` or ``\\``, no drive letters).  Falls back to the default
    ``models_saved`` directory if the result would be empty.
    """
    raw = Path(user_input)
    safe_parts = [
        p for p in raw.parts
        if p not in ("..", "/", "\\") and ":" not in p
    ]
    return Path(*safe_parts) if safe_parts else Path("models_saved")


st.set_page_config(
    page_title="Basketball AI – Training GUI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏀 Basketball Performance AI – Training GUI")

# ---------------------------------------------------------------------------
# Sidebar – Data source
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Data Source")
    source = st.radio(
        "Choose data source",
        ["CSV Files (local)", "Azure SQL Server"],
        index=0,
    )

    data: dict | None = None

    if source == "CSV Files (local)":
        st.subheader("Upload CSV files")
        uploaded = {
            "leagues":               st.file_uploader("leagues.csv",               type="csv"),
            "teams":                 st.file_uploader("teams.csv",                 type="csv"),
            "players":               st.file_uploader("players.csv",               type="csv"),
            "player_stats":          st.file_uploader("player_stats.csv",          type="csv"),
            "team_player_relations": st.file_uploader("team_player_relations.csv", type="csv"),
        }
        all_uploaded = all(v is not None for v in uploaded.values())
        if st.button("Load CSV Data", disabled=not all_uploaded):
            with st.spinner("Loading CSV files …"):
                leagues_df = pd.read_csv(uploaded["leagues"])
                teams_df   = pd.read_csv(uploaded["teams"])
                players_df = pd.read_csv(uploaded["players"])
                stats_df   = pd.read_csv(uploaded["player_stats"])
                rels_df    = pd.read_csv(uploaded["team_player_relations"])

                for col in ["current_team_id", "current_league_id", "draft_year", "draft_pick"]:
                    if col in players_df.columns:
                        players_df[col] = players_df[col].where(players_df[col].notna(), other=None)

                league_dict = {int(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()}
                team_dict   = {int(r["id"]): r.to_dict() for _, r in teams_df.iterrows()}
                player_dict = {int(r["id"]): r.to_dict() for _, r in players_df.iterrows()}
                league_teams: dict = {}
                for _, t in teams_df.iterrows():
                    league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

                data = {
                    "leagues": leagues_df, "teams": teams_df,
                    "players": players_df, "player_stats": stats_df,
                    "team_player_relations": rels_df,
                    "league_dict": league_dict, "team_dict": team_dict,
                    "player_dict": player_dict, "league_teams": league_teams,
                }
                st.session_state["data"] = data
            st.success(
                f"Loaded {len(players_df):,} players, "
                f"{len(stats_df):,} stat rows, "
                f"{len(teams_df):,} teams."
            )

    else:  # Azure SQL Server
        st.subheader("Azure SQL connection")
        conn_str = st.text_input(
            "Connection string",
            value=os.environ.get("AZURE_SQL_CONNECTION_STRING", ""),
            type="password",
            help=(
                "SQLAlchemy URL, e.g.:\n"
                "mssql+pyodbc://user:pass@server.database.windows.net/db"
                "?driver=ODBC+Driver+18+for+SQL+Server"
            ),
        )
        if st.button("Test & Load SQL Data"):
            if not conn_str.strip():
                st.error("Please enter a connection string.")
            else:
                with st.spinner("Connecting to Azure SQL …"):
                    try:
                        os.environ["AZURE_SQL_CONNECTION_STRING"] = conn_str
                        from src.data.sql_loader import load_all_data
                        data = load_all_data()
                        st.session_state["data"] = data
                        st.success(
                            f"Connected! {len(data['players']):,} players loaded."
                        )
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

# Retrieve data from session state
if "data" not in st.session_state:
    st.info("👈 Load data from the sidebar to get started.")
    st.stop()

data = st.session_state["data"]

# ---------------------------------------------------------------------------
# Section 2 – Training parameters
# ---------------------------------------------------------------------------
st.header("2 · Training Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("XGBoost (performance model)")
    xgb_n_estimators = st.slider("n_estimators", 50, 1000, 300, step=50)
    xgb_max_depth    = st.slider("max_depth",     2,   12,   5)
    xgb_lr           = st.slider("learning_rate", 0.005, 0.3, 0.05, step=0.005,
                                  format="%.3f")
    xgb_subsample    = st.slider("subsample",     0.4, 1.0, 0.8, step=0.05)

with col2:
    st.subheader("General")
    seed          = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    test_size     = st.slider("Validation split", 0.05, 0.40, 0.15, step=0.05)
    model_dir_gui = st.text_input("Model save directory", value="models_saved")

st.subheader("Feature selection")
from src.models.performance_model import FEATURE_COLS
all_features = FEATURE_COLS.copy()
selected_features = st.multiselect(
    "Include features (deselect to exclude)",
    options=all_features,
    default=all_features,
)
if len(selected_features) < 3:
    st.warning("Select at least 3 features.")

# ---------------------------------------------------------------------------
# Section 3 – Train
# ---------------------------------------------------------------------------
st.header("3 · Train")

if st.button("🚀 Start Training", type="primary"):
    if len(selected_features) < 3:
        st.error("Please select at least 3 features.")
    else:
        progress = st.progress(0, text="Preparing …")
        log_area = st.empty()
        log_lines: list[str] = []

        def log(msg: str) -> None:
            log_lines.append(msg)
            log_area.code("\n".join(log_lines[-30:]), language="text")

        try:
            from src.models.performance_model import PerformanceModel
            from src.models.compatibility_model import CompatibilityModel
            from src.models.ensemble import EnsembleModel
            from sklearn.model_selection import train_test_split
            import sklearn.metrics as skm

            # --- Performance model with custom hyper-params ---------------
            from xgboost import XGBRegressor
            from sklearn.preprocessing import StandardScaler

            log("Building feature matrix …")
            progress.progress(10, text="Building features …")

            perf_model = PerformanceModel()
            perf_model.feature_names = selected_features
            perf_model.model = XGBRegressor(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_lr,
                subsample=xgb_subsample,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=int(seed),
                verbosity=0,
            )
            perf_model.scaler = StandardScaler()

            X_full, y_full = perf_model.prepare_features(data)
            X = X_full[selected_features]
            log(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

            progress.progress(30, text="Training performance model …")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_full, test_size=float(test_size), random_state=int(seed)
            )
            X_tr_sc = perf_model.scaler.fit_transform(X_train)
            X_va_sc = perf_model.scaler.transform(X_val)
            perf_model.model.fit(
                X_tr_sc, y_train,
                eval_set=[(X_va_sc, y_val)],
                verbose=False,
            )
            perf_model.is_trained = True

            y_pred_val  = perf_model.model.predict(X_va_sc)
            train_rmse  = float(np.sqrt(np.mean((perf_model.model.predict(X_tr_sc) - y_train) ** 2)))
            val_rmse    = float(np.sqrt(np.mean((y_pred_val - y_val) ** 2)))
            val_mae     = float(np.mean(np.abs(y_pred_val - y_val)))
            val_r2      = float(skm.r2_score(y_val, y_pred_val))

            log(f"Performance model – Train RMSE: {train_rmse:.4f}")
            log(f"Performance model – Val   RMSE: {val_rmse:.4f}  MAE: {val_mae:.4f}  R²: {val_r2:.4f}")

            progress.progress(60, text="Training compatibility model …")
            compat_model = CompatibilityModel()
            compat_model.train(data)
            log("Compatibility model trained.")

            progress.progress(80, text="Saving models …")
            ensemble = EnsembleModel(
                performance_model=perf_model,
                compatibility_model=compat_model,
            )
            Path(_safe_model_dir(model_dir_gui)).mkdir(parents=True, exist_ok=True)
            ensemble.save(str(_safe_model_dir(model_dir_gui)))
            log(f"Models saved to '{_safe_model_dir(model_dir_gui)}/'")
            st.session_state["model_dir"] = str(_safe_model_dir(model_dir_gui))
            progress.progress(100, text="Done!")
            st.session_state["ensemble"]   = ensemble
            st.session_state["metrics"]    = {
                "train_rmse": train_rmse,
                "val_rmse":   val_rmse,
                "val_mae":    val_mae,
                "val_r2":     val_r2,
            }
            st.session_state["perf_model"] = perf_model
            st.success("✅ Training complete!")

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Section 4 – Results
# ---------------------------------------------------------------------------
if "metrics" in st.session_state:
    st.header("4 · Results")

    m = st.session_state["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train RMSE", f"{m['train_rmse']:.4f}")
    c2.metric("Val RMSE",   f"{m['val_rmse']:.4f}")
    c3.metric("Val MAE",    f"{m['val_mae']:.4f}")
    c4.metric("Val R²",     f"{m['val_r2']:.4f}")

    # --- Feature importance chart -----------------------------------------
    perf_model = st.session_state.get("perf_model")
    if perf_model and perf_model.is_trained:
        st.subheader("Feature Importance")
        importances = perf_model.feature_importances()
        if importances:
            fi_df = (
                pd.DataFrame(
                    {"feature": list(importances.keys()),
                     "importance": list(importances.values())}
                )
                .sort_values("importance", ascending=False)
            )
            st.bar_chart(fi_df.set_index("feature")["importance"])

        # --- SHAP (if available) ------------------------------------------
        try:
            import shap as _shap
            st.subheader("SHAP Feature Importance (sample)")
            if "data" in st.session_state:
                sample = st.session_state["data"]["player_stats"].sample(
                    min(50, len(st.session_state["data"]["player_stats"])),
                    random_state=42,
                )
                X_sample, _ = perf_model.prepare_features(st.session_state["data"])
                X_sample = X_sample[selected_features].head(50)
                X_sc = perf_model.scaler.transform(X_sample)
                explainer  = _shap.TreeExplainer(perf_model.model)
                shap_vals  = explainer.shap_values(X_sc)
                mean_shap  = np.abs(shap_vals).mean(axis=0)
                shap_df = (
                    pd.DataFrame({"feature": selected_features, "mean_|shap|": mean_shap})
                    .sort_values("mean_|shap|", ascending=False)
                )
                st.bar_chart(shap_df.set_index("feature")["mean_|shap|"])
        except ImportError:
            st.info("Install `shap` for SHAP charts.")
        except Exception:
            pass

    # --- Download model ---------------------------------------------------
    st.subheader("Download trained model")
    # Use the actual save path from the training run, not the (possibly edited) input box
    resolved_dir = Path(
        st.session_state.get("model_dir", str(_safe_model_dir(model_dir_gui)))
    )
    model_path  = resolved_dir / "performance_model.joblib"
    compat_path = resolved_dir / "compatibility_model.joblib"
    if model_path.exists():
        with open(model_path, "rb") as f:
            st.download_button(
                label="⬇️ Download performance_model.joblib",
                data=f.read(),
                file_name="performance_model.joblib",
                mime="application/octet-stream",
            )
    compat_path = resolved_dir / "compatibility_model.joblib"
    if compat_path.exists():
        with open(compat_path, "rb") as f:
            st.download_button(
                label="⬇️ Download compatibility_model.joblib",
                data=f.read(),
                file_name="compatibility_model.joblib",
                mime="application/octet-stream",
            )

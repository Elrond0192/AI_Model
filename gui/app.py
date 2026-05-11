"""Basketball Performance AI – Streamlit GUI.

Launch with:
    streamlit run gui/app.py

Tabs:
  1. 📂 Dati       – data source (CSV / Azure SQL) + interactive data browser
  2. 🏋️ Training    – load pre-trained model OR configure + run training; results
  3. 🎯 Predizioni  – predict player rating at any team + career trajectory
  4. 🔀 Scenari     – transfer impact, best-team fit, best players, lineup, teammates
  5. 💬 Chat        – natural-language assistant backed by the trained model
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import uuid

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_model_dir(user_input: str) -> Path:
    """Sanitize a user-supplied directory path to prevent path traversal."""
    raw = Path(user_input)
    safe_parts = [
        p for p in raw.parts
        if p not in ("..", "/", "\\") and ":" not in p
    ]
    return Path(*safe_parts) if safe_parts else Path("models_saved")


def _require_data():
    """Return data from session_state or show a message and return None."""
    if "data" not in st.session_state:
        st.info("👈 Vai al tab **📂 Dati** e carica i tuoi dati.")
        return None
    return st.session_state["data"]


def _require_engine(data):
    """Return the WhatIfEngine or prompt user and return None."""
    ensemble = st.session_state.get("ensemble")
    if ensemble is None or not ensemble.is_trained:
        st.info("👈 Vai al tab **🏋️ Training** per caricare o addestrare un modello.")
        return None
    if "engine" not in st.session_state:
        from basketball_ai.scenarios.engine import WhatIfEngine
        st.session_state["engine"] = WhatIfEngine(ensemble, data)
    return st.session_state["engine"]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Basketball Performance AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🏀 Basketball Performance AI")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_train, tab_pred, tab_scen, tab_chat = st.tabs([
    "📂 Dati",
    "🏋️ Training",
    "🎯 Predizioni",
    "🔀 Scenari",
    "💬 Chat",
])

# ===========================================================================
# TAB 1 – DATI
# ===========================================================================

with tab_data:
    st.header("📂 Sorgente dati")

    source = st.radio(
        "Scegli la sorgente",
        ["CSV (locale)", "Azure SQL Server"],
        horizontal=True,
        key="data_source",
    )

    if source == "CSV (locale)":
        st.subheader("Carica i file CSV")
        uploaded = {
            "leagues":               st.file_uploader("leagues.csv",               type="csv", key="up_leagues"),
            "teams":                 st.file_uploader("teams.csv",                 type="csv", key="up_teams"),
            "players":               st.file_uploader("players.csv",               type="csv", key="up_players"),
            "player_stats":          st.file_uploader("player_stats.csv",          type="csv", key="up_stats"),
            "team_player_relations": st.file_uploader("team_player_relations.csv", type="csv", key="up_rels"),
        }
        all_uploaded = all(v is not None for v in uploaded.values())
        if st.button("📥 Carica dati CSV", disabled=not all_uploaded, type="primary"):
            with st.spinner("Caricamento file CSV …"):
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

                st.session_state["data"] = {
                    "leagues": leagues_df, "teams": teams_df,
                    "players": players_df, "player_stats": stats_df,
                    "team_player_relations": rels_df,
                    "league_dict": league_dict, "team_dict": team_dict,
                    "player_dict": player_dict, "league_teams": league_teams,
                }
                # Invalidate any existing engine when new data is loaded
                st.session_state.pop("engine", None)
                st.session_state.pop("chat_engine", None)

            st.success(
                f"✅ Caricati {len(players_df):,} giocatori, "
                f"{len(stats_df):,} righe statistiche, "
                f"{len(teams_df):,} squadre."
            )

    else:  # Azure SQL Server
        st.subheader("Connessione Azure SQL")
        conn_str = st.text_input(
            "Connection string",
            value=os.environ.get("AZURE_SQL_CONNECTION_STRING", ""),
            type="password",
            help=(
                "SQLAlchemy URL, es.: "
                "mssql+pyodbc://user:pass@server.database.windows.net/db"
                "?driver=ODBC+Driver+18+for+SQL+Server"
            ),
        )
        if st.button("🔌 Connetti e carica dati SQL", type="primary"):
            if not conn_str.strip():
                st.error("Inserisci la connection string.")
            else:
                with st.spinner("Connessione ad Azure SQL …"):
                    try:
                        os.environ["AZURE_SQL_CONNECTION_STRING"] = conn_str
                        from basketball_ai.data.sql_loader import (
                            load_all_data_from_sql,
                            get_table_mapping,
                            get_engine,
                        )
                        _engine = get_engine()
                        mapping = get_table_mapping(_engine)
                        loaded = load_all_data_from_sql(_engine, table_mapping=mapping)
                        st.session_state["data"] = loaded
                        st.session_state["sql_table_mapping"] = mapping
                        st.session_state.pop("engine", None)
                        st.session_state.pop("chat_engine", None)
                        st.success(f"✅ Connesso! {len(loaded['players']):,} giocatori caricati.")
                    except Exception as exc:
                        st.error(f"Connessione fallita: {exc}")

        if "sql_table_mapping" in st.session_state:
            with st.expander("🔍 Mapping tabelle auto-rilevato", expanded=False):
                for logical, actual in st.session_state["sql_table_mapping"].items():
                    icon = "✅" if logical == actual else "🔀"
                    st.write(f"{icon} **{logical}** → `{actual}`")

    # -----------------------------------------------------------------------
    # Data browser
    # -----------------------------------------------------------------------
    if "data" in st.session_state:
        st.divider()
        st.header("🔎 Browser dati")
        data_b = st.session_state["data"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Giocatori",             f"{len(data_b['players']):,}")
        m2.metric("Squadre",               f"{len(data_b['teams']):,}")
        m3.metric("Statistiche stagionali",f"{len(data_b['player_stats']):,}")
        m4.metric("Leghe",                 f"{len(data_b['leagues']):,}")

        for label, key in [
            ("👥 Giocatori",              "players"),
            ("🏆 Squadre",               "teams"),
            ("📊 Statistiche giocatori", "player_stats"),
            ("🏅 Leghe",                 "leagues"),
        ]:
            with st.expander(label, expanded=False):
                df_show = data_b[key].copy()
                search = st.text_input(
                    f"Filtra {label}", key=f"search_{key}", placeholder="Cerca…"
                )
                if search:
                    mask = df_show.apply(
                        lambda col: col.astype(str).str.contains(search, case=False, na=False)
                    ).any(axis=1)
                    df_show = df_show[mask]
                st.dataframe(df_show, use_container_width=True)
                st.caption(f"{len(df_show):,} righe")


# ===========================================================================
# TAB 2 – TRAINING
# ===========================================================================

with tab_train:
    st.header("🏋️ Modello")

    # -----------------------------------------------------------------------
    # A – Load pre-trained model
    # -----------------------------------------------------------------------
    st.subheader("A · Carica modello pre-addestrato")
    load_dir = st.text_input("Directory modello salvato", value="models_saved", key="load_dir")

    if st.button("📂 Carica modello", key="load_model_btn"):
        resolved = _safe_model_dir(load_dir)
        perf_path   = resolved / "performance_model.joblib"
        compat_path = resolved / "compatibility_model.joblib"
        meta_path   = resolved / "metadata.json"
        if not perf_path.exists() or not compat_path.exists():
            st.error(f"File modello non trovati in '{resolved}/'")
        else:
            try:
                from basketball_ai.models.ensemble import EnsembleModel
                ens = EnsembleModel()
                ens.load(str(resolved))
                st.session_state["ensemble"] = ens
                st.session_state.pop("engine", None)
                st.session_state.pop("chat_engine", None)
                meta_info = ""
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta_info = (
                        f"  \n🗓 Addestrato: `{meta.get('trained_at', '?')}`"
                        f"  \n📈 Val RMSE: `{meta.get('val_rmse', '?')}`"
                        f"  R²: `{meta.get('val_r2', '?')}`"
                    )
                st.success(f"✅ Modello caricato da `{resolved}/`{meta_info}")
            except Exception as exc:
                st.error(f"Caricamento fallito: {exc}")

    st.divider()

    # -----------------------------------------------------------------------
    # B – Train new model
    # -----------------------------------------------------------------------
    st.subheader("B · Addestra nuovo modello")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        xgb_n_estimators = st.slider("n_estimators", 50, 1000, 300, step=50,    key="t_nest")
        xgb_max_depth    = st.slider("max_depth",     2,   12,   5,             key="t_depth")
        xgb_lr           = st.slider("learning_rate", 0.005, 0.3, 0.05,
                                     step=0.005, format="%.3f",                key="t_lr")
        xgb_subsample    = st.slider("subsample",     0.4, 1.0, 0.8, step=0.05, key="t_sub")
    with col2:
        st.markdown("**Generale**")
        seed          = st.number_input("Random seed",         min_value=0, max_value=99999, value=42, key="t_seed")
        test_size     = st.slider("Validation split",          0.05, 0.40, 0.15, step=0.05,            key="t_split")
        model_dir_gui = st.text_input("Directory salvataggio", value="models_saved",                    key="t_dir")

    from basketball_ai.models.performance_model import FEATURE_COLS
    all_features = FEATURE_COLS.copy()
    st.markdown("**Feature selection**")
    selected_features = st.multiselect(
        "Feature da includere (deseleziona per escludere)",
        options=all_features,
        default=all_features,
        key="t_feats",
    )
    if len(selected_features) < 3:
        st.warning("Seleziona almeno 3 feature.")

    if st.button("🚀 Avvia Training", type="primary", key="t_run"):
        if "data" not in st.session_state:
            st.error("Carica prima i dati nel tab 📂 Dati.")
        elif len(selected_features) < 3:
            st.error("Seleziona almeno 3 feature.")
        else:
            data_t   = st.session_state["data"]
            progress = st.progress(0, text="Preparazione …")
            log_area = st.empty()
            log_lines: list[str] = []

            def log(msg: str) -> None:
                log_lines.append(msg)
                log_area.code("\n".join(log_lines[-30:]), language="text")

            try:
                from basketball_ai.models.performance_model import PerformanceModel
                from basketball_ai.models.compatibility_model import CompatibilityModel
                from basketball_ai.models.ensemble import EnsembleModel
                from sklearn.model_selection import train_test_split
                import sklearn.metrics as skm
                from xgboost import XGBRegressor
                from sklearn.preprocessing import StandardScaler

                log("Costruzione feature matrix …")
                progress.progress(10, text="Costruzione feature …")

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

                X_full, y_full = perf_model.prepare_features(data_t)
                X = X_full[selected_features]
                log(f"Feature matrix: {X.shape[0]:,} righe × {X.shape[1]} feature")

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

                y_pred_val = perf_model.model.predict(X_va_sc)
                train_rmse = float(np.sqrt(np.mean((perf_model.model.predict(X_tr_sc) - y_train) ** 2)))
                val_rmse   = float(np.sqrt(np.mean((y_pred_val - y_val) ** 2)))
                val_mae    = float(np.mean(np.abs(y_pred_val - y_val)))
                val_r2     = float(skm.r2_score(y_val, y_pred_val))

                log(f"Performance model – Train RMSE: {train_rmse:.4f}")
                log(f"Performance model – Val   RMSE: {val_rmse:.4f}  MAE: {val_mae:.4f}  R²: {val_r2:.4f}")

                progress.progress(60, text="Training compatibility model …")
                compat_model = CompatibilityModel()
                compat_model.train(data_t)
                log("Compatibility model addestrato.")

                progress.progress(80, text="Salvataggio modelli …")
                ensemble = EnsembleModel(
                    performance_model=perf_model,
                    compatibility_model=compat_model,
                )
                save_dir = _safe_model_dir(model_dir_gui)
                save_dir.mkdir(parents=True, exist_ok=True)
                metrics_dict = {
                    "train_rmse": train_rmse,
                    "val_rmse":   val_rmse,
                    "val_mae":    val_mae,
                    "val_r2":     val_r2,
                }
                ensemble.save(str(save_dir), metrics=metrics_dict)
                log(f"Modelli salvati in '{save_dir}/'")

                st.session_state["model_dir"]        = str(save_dir)
                st.session_state["ensemble"]         = ensemble
                st.session_state["metrics"]          = metrics_dict
                st.session_state["perf_model"]       = perf_model
                st.session_state["selected_features"] = selected_features
                st.session_state.pop("engine", None)
                st.session_state.pop("chat_engine", None)

                progress.progress(100, text="Completato!")
                st.success("✅ Training completato!")

            except Exception as exc:
                st.error(f"Training fallito: {exc}")

    # -----------------------------------------------------------------------
    # C – Results + download
    # -----------------------------------------------------------------------
    if "metrics" in st.session_state:
        st.divider()
        st.subheader("C · Risultati")
        m = st.session_state["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train RMSE", f"{m['train_rmse']:.4f}")
        c2.metric("Val RMSE",   f"{m['val_rmse']:.4f}")
        c3.metric("Val MAE",    f"{m['val_mae']:.4f}")
        c4.metric("Val R²",     f"{m['val_r2']:.4f}")

        perf_model_r = st.session_state.get("perf_model")
        if perf_model_r and perf_model_r.is_trained:
            importances = perf_model_r.feature_importances()
            if importances:
                st.subheader("Feature Importance (XGBoost)")
                fi_df = (
                    pd.DataFrame({
                        "feature":    list(importances.keys()),
                        "importance": list(importances.values()),
                    })
                    .sort_values("importance", ascending=False)
                )
                st.bar_chart(fi_df.set_index("feature")["importance"])

            try:
                import shap as _shap
                st.subheader("SHAP Feature Importance (campione)")
                if "data" in st.session_state:
                    feat_names = st.session_state.get("selected_features", FEATURE_COLS)
                    X_sample, _ = perf_model_r.prepare_features(st.session_state["data"])
                    available   = [f for f in feat_names if f in X_sample.columns]
                    X_sample    = X_sample[available].head(50)
                    X_sc        = perf_model_r.scaler.transform(X_sample)
                    explainer   = _shap.TreeExplainer(perf_model_r.model)
                    shap_vals   = explainer.shap_values(X_sc)
                    mean_shap   = np.abs(shap_vals).mean(axis=0)
                    shap_df = (
                        pd.DataFrame({"feature": available, "mean_|shap|": mean_shap})
                        .sort_values("mean_|shap|", ascending=False)
                    )
                    st.bar_chart(shap_df.set_index("feature")["mean_|shap|"])
            except ImportError:
                st.info("Installa `shap` per i grafici SHAP.")
            except Exception:
                pass

        st.subheader("⬇️ Download modello")
        resolved_dir = Path(st.session_state.get("model_dir", "models_saved"))
        for fname in ["performance_model.joblib", "compatibility_model.joblib", "metadata.json"]:
            fpath = resolved_dir / fname
            if fpath.exists():
                with open(fpath, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {fname}",
                        data=f.read(),
                        file_name=fname,
                        mime="application/octet-stream",
                        key=f"dl_{fname}",
                    )


# ===========================================================================
# TAB 3 – PREDIZIONI
# ===========================================================================

with tab_pred:
    st.header("🎯 Predizioni")
    data_p = _require_data()
    if data_p is not None:
        engine_p = _require_engine(data_p)
        if engine_p is not None:
            players_df_p = data_p["players"]
            teams_df_p   = data_p["teams"]

            player_opts_p = {
                int(r["id"]): f"{r['name']} ({r['position']}, {r['age']}a)"
                for _, r in players_df_p.iterrows()
            }
            team_opts_p = {
                int(r["id"]): str(r["name"])
                for _, r in teams_df_p.iterrows()
            }

            # --- Single prediction ----------------------------------------
            st.subheader("🔮 Predici rating")
            pc1, pc2 = st.columns(2)
            with pc1:
                sel_player_id = st.selectbox(
                    "🏀 Giocatore",
                    options=list(player_opts_p.keys()),
                    format_func=lambda x: player_opts_p[x],
                    key="pred_player",
                )
            with pc2:
                sel_team_id = st.selectbox(
                    "🏆 Squadra",
                    options=list(team_opts_p.keys()),
                    format_func=lambda x: team_opts_p[x],
                    key="pred_team",
                )
            pred_season = st.number_input(
                "Stagione", min_value=2000, max_value=2040, value=2024, key="pred_season"
            )

            if st.button("🔮 Predici", type="primary", key="pred_run"):
                try:
                    result = engine_p.predict_in_team(
                        sel_player_id, sel_team_id, season=int(pred_season)
                    )
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Rating predetto", f"{result.predicted_rating:.2f} / 10")
                    r2.metric("CI basso",         f"{result.confidence_low:.2f}")
                    r3.metric("CI alto",          f"{result.confidence_high:.2f}")

                    st.subheader("📊 Breakdown")
                    breakdown_df = pd.DataFrame({
                        "Componente": [
                            "Base XGBoost", "Age factor",
                            "Compatibility", "League factor", "Context adj.",
                        ],
                        "Valore": [
                            result.base_rating,     result.age_factor,
                            result.compatibility_factor, result.league_factor,
                            result.context_adjustment,
                        ],
                    })
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    st.caption(f"_{result.explanation}_")

                    if result.shap_values:
                        shap_df = (
                            pd.DataFrame({
                                "feature": list(result.shap_values.keys()),
                                "shap":    list(result.shap_values.values()),
                            })
                            .sort_values("shap", key=abs, ascending=False)
                        )
                        st.subheader("SHAP values")
                        st.bar_chart(shap_df.set_index("feature")["shap"])

                except Exception as exc:
                    st.error(f"Predizione fallita: {exc}")

            st.divider()

            # --- Trajectory -----------------------------------------------
            st.subheader("📈 Traiettoria per età")
            tr1, tr2 = st.columns(2)
            with tr1:
                traj_player = st.selectbox(
                    "Giocatore",
                    options=list(player_opts_p.keys()),
                    format_func=lambda x: player_opts_p[x],
                    key="traj_player",
                )
            with tr2:
                traj_team = st.selectbox(
                    "Squadra (contesto)",
                    options=list(team_opts_p.keys()),
                    format_func=lambda x: team_opts_p[x],
                    key="traj_team",
                )
            ta1, ta2 = st.columns(2)
            age_from = ta1.number_input("Età da", min_value=16, max_value=44, value=18, key="traj_from")
            age_to   = ta2.number_input("Età a",  min_value=17, max_value=45, value=38, key="traj_to")

            if st.button("📈 Calcola traiettoria", key="traj_run"):
                if int(age_from) >= int(age_to):
                    st.error("'Età da' deve essere minore di 'Età a'.")
                else:
                    try:
                        traj = engine_p.predict_age_trajectory(
                            traj_player,
                            age_range=(int(age_from), int(age_to)),
                            team_id=traj_team,
                        )
                        traj_df = pd.DataFrame([
                            {
                                "Età":      pt.age,
                                "Rating":   pt.predicted_rating,
                                "CI basso": pt.confidence_low,
                                "CI alto":  pt.confidence_high,
                            }
                            for pt in traj
                        ]).set_index("Età")
                        st.line_chart(traj_df[["Rating", "CI basso", "CI alto"]])
                        with st.expander("Dati completi traiettoria", expanded=False):
                            st.dataframe(traj_df, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Errore traiettoria: {exc}")


# ===========================================================================
# TAB 4 – SCENARI
# ===========================================================================

with tab_scen:
    st.header("🔀 Scenari")
    data_s = _require_data()
    if data_s is not None:
        engine_s = _require_engine(data_s)
        if engine_s is not None:
            players_df_s = data_s["players"]
            teams_df_s   = data_s["teams"]

            player_opts_s = {
                int(r["id"]): f"{r['name']} ({r['position']})"
                for _, r in players_df_s.iterrows()
            }
            team_opts_s = {
                int(r["id"]): str(r["name"])
                for _, r in teams_df_s.iterrows()
            }

            scen_tabs = st.tabs([
                "🔁 Trasferimento",
                "🏆 Migliori squadre",
                "👥 Migliori giocatori",
                "📋 Quintetto",
                "🤝 Compagni",
            ])

            # ---- Transfer -------------------------------------------------
            with scen_tabs[0]:
                st.subheader("🔁 Simulazione trasferimento")
                tr_player = st.selectbox(
                    "Giocatore", list(player_opts_s.keys()),
                    format_func=lambda x: player_opts_s[x], key="tr_player",
                )
                tr_c1, tr_c2 = st.columns(2)
                from_t = tr_c1.selectbox(
                    "Da squadra", list(team_opts_s.keys()),
                    format_func=lambda x: team_opts_s[x], key="tr_from",
                )
                to_t = tr_c2.selectbox(
                    "A squadra", list(team_opts_s.keys()),
                    format_func=lambda x: team_opts_s[x], key="tr_to",
                )
                tr_season = st.number_input(
                    "Stagione", 2000, 2040, 2024, key="tr_season"
                )
                if st.button("🔁 Simula trasferimento", type="primary", key="tr_run"):
                    if from_t == to_t:
                        st.warning("Seleziona due squadre diverse.")
                    else:
                        try:
                            res = engine_s.simulate_transfer(
                                tr_player, from_t, to_t, season=int(tr_season)
                            )
                            sign = "+" if res.rating_delta >= 0 else ""
                            tm1, tm2, tm3 = st.columns(3)
                            tm1.metric("Rating attuale",             f"{res.rating_before:.2f}")
                            tm2.metric("Rating dopo trasferimento",  f"{res.rating_after:.2f}")
                            tm3.metric("Variazione",
                                       f"{sign}{res.rating_delta:.2f}",
                                       delta=round(res.rating_delta, 3))
                            st.info(f"**Verdetto:** {res.recommendation}")
                        except Exception as exc:
                            st.error(f"Errore: {exc}")

            # ---- Best teams -----------------------------------------------
            with scen_tabs[1]:
                st.subheader("🏆 Migliori squadre per un giocatore")
                bt_player = st.selectbox(
                    "Giocatore", list(player_opts_s.keys()),
                    format_func=lambda x: player_opts_s[x], key="bt_player",
                )
                bt_n      = st.slider("Top N squadre", 3, 20, 10, key="bt_n")
                bt_season = st.number_input("Stagione", 2000, 2040, 2024, key="bt_season")
                if st.button("🏆 Trova migliori squadre", type="primary", key="bt_run"):
                    try:
                        fits = engine_s.best_team_fit(
                            bt_player, top_n=int(bt_n), season=int(bt_season)
                        )
                        bt_df = pd.DataFrame([{
                            "Rank":           f.rank,
                            "Squadra":        f.team_name,
                            "Lega":           f.league_name,
                            "Rating predetto": round(f.predicted_rating, 3),
                            "Compatibilità":  round(f.compatibility_score, 3),
                            "Fit posizione":  round(f.position_fit, 3),
                        } for f in fits])
                        st.dataframe(bt_df, use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"Errore: {exc}")

            # ---- Best players ---------------------------------------------
            with scen_tabs[2]:
                st.subheader("👥 Migliori giocatori per una squadra")
                bp_team = st.selectbox(
                    "Squadra", list(team_opts_s.keys()),
                    format_func=lambda x: team_opts_s[x], key="bp_team",
                )
                bp_pos  = st.selectbox(
                    "Posizione (opzionale)",
                    ["Tutte", "PG", "SG", "SF", "PF", "C"],
                    key="bp_pos",
                )
                bp_n      = st.slider("Top N giocatori", 3, 20, 10, key="bp_n")
                bp_season = st.number_input("Stagione", 2000, 2040, 2024, key="bp_season")
                if st.button("👥 Trova migliori giocatori", type="primary", key="bp_run"):
                    try:
                        position_filter = None if bp_pos == "Tutte" else bp_pos
                        players_fit = engine_s.best_player_for_team(
                            bp_team,
                            position=position_filter,
                            top_n=int(bp_n),
                            season=int(bp_season),
                        )
                        bp_df = pd.DataFrame([{
                            "Rank":            p.rank,
                            "Giocatore":       p.player_name,
                            "Posizione":       p.position,
                            "Squadra attuale": p.current_team,
                            "Rating predetto": round(p.predicted_rating, 3),
                        } for p in players_fit])
                        st.dataframe(bp_df, use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"Errore: {exc}")

            # ---- Lineup ---------------------------------------------------
            with scen_tabs[3]:
                st.subheader("📋 Analisi quintetto")
                lu_player = st.selectbox(
                    "Giocatore target", list(player_opts_s.keys()),
                    format_func=lambda x: player_opts_s[x], key="lu_player",
                )
                lu_team = st.selectbox(
                    "Squadra", list(team_opts_s.keys()),
                    format_func=lambda x: team_opts_s[x], key="lu_team",
                )
                other_opts = {k: v for k, v in player_opts_s.items() if k != lu_player}
                lu_lineup = st.multiselect(
                    "Compagni di quintetto (1–4 giocatori)",
                    options=list(other_opts.keys()),
                    format_func=lambda x: other_opts.get(x, str(x)),
                    max_selections=4,
                    key="lu_lineup",
                )
                lu_season = st.number_input("Stagione", 2000, 2040, 2024, key="lu_season")
                if st.button("📋 Analizza quintetto", type="primary", key="lu_run"):
                    if not lu_lineup:
                        st.warning("Seleziona almeno un compagno di quintetto.")
                    else:
                        try:
                            res = engine_s.what_if_lineup(
                                lu_player, lu_team, lu_lineup, season=int(lu_season)
                            )
                            lm1, lm2, lm3 = st.columns(3)
                            lm1.metric("Rating predetto", f"{res.predicted_rating:.2f} / 10")
                            lm2.metric("Media lineup",    f"{res.avg_lineup_rating:.2f}")
                            lm3.metric("Confidenza",
                                       f"[{res.confidence_low:.2f}–{res.confidence_high:.2f}]")

                            st.subheader("Copertura posizioni")
                            lp1, lp2 = st.columns(2)
                            lp1.success(
                                f"Coperte: {', '.join(res.positions_covered) or 'nessuna'}"
                            )
                            if res.missing_positions:
                                lp2.warning(
                                    f"Mancanti: {', '.join(res.missing_positions)}"
                                )
                            if res.position_overlaps:
                                overlaps_str = ", ".join(
                                    f"{p}×{c}" for p, c in res.position_overlaps.items()
                                )
                                st.warning(f"Sovrapposizioni: {overlaps_str}")

                            st.subheader("Profili dei componenti")
                            lu_df = pd.DataFrame([{
                                "Giocatore":      p.player_name,
                                "Posizione":      p.position,
                                "Ruolo":          p.role,
                                "Rating predetto": p.predicted_rating,
                                "Style fit":      p.style_compat,
                            } for p in res.lineup_profiles])
                            st.dataframe(lu_df, use_container_width=True, hide_index=True)
                            st.caption(f"_{res.explanation}_")
                        except Exception as exc:
                            st.error(f"Errore analisi quintetto: {exc}")

            # ---- Teammates ------------------------------------------------
            with scen_tabs[4]:
                st.subheader("🤝 Scenario compagni ipotetici")
                tm_player = st.selectbox(
                    "Giocatore", list(player_opts_s.keys()),
                    format_func=lambda x: player_opts_s[x], key="tm_player",
                )
                tm_team = st.selectbox(
                    "Squadra", list(team_opts_s.keys()),
                    format_func=lambda x: team_opts_s[x], key="tm_team",
                )
                tm_avg    = st.slider(
                    "Rating medio ipotetico dei compagni", 4.0, 10.0, 7.5,
                    step=0.1, key="tm_avg",
                )
                tm_season = st.number_input("Stagione", 2000, 2040, 2024, key="tm_season")
                if st.button("🤝 Calcola scenario", type="primary", key="tm_run"):
                    try:
                        res_base = engine_s.predict_in_team(
                            tm_player, tm_team, season=int(tm_season)
                        )
                        res_hyp = engine_s.what_if_teammates(
                            tm_player, tm_team, float(tm_avg), season=int(tm_season)
                        )
                        delta = res_hyp.predicted_rating - res_base.predicted_rating
                        sign  = "+" if delta >= 0 else ""
                        tc1, tc2, tc3 = st.columns(3)
                        tc1.metric("Rating base", f"{res_base.predicted_rating:.2f}")
                        tc2.metric("Rating con compagni ipotetici",
                                   f"{res_hyp.predicted_rating:.2f}")
                        tc3.metric("Variazione", f"{sign}{delta:.2f}",
                                   delta=round(delta, 3))
                    except Exception as exc:
                        st.error(f"Errore: {exc}")


# ===========================================================================
# TAB 5 – CHAT
# ===========================================================================

with tab_chat:
    st.header("💬 Assistente AI")
    data_c = _require_data()
    if data_c is not None:
        engine_c = _require_engine(data_c)
        if engine_c is not None:
            # Ensure session ID and history exist
            if "chat_session_id" not in st.session_state:
                st.session_state["chat_session_id"] = str(uuid.uuid4())
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            # Lazy ChatEngine construction
            if "chat_engine" not in st.session_state:
                from basketball_ai.chat.engine import ChatEngine
                st.session_state["chat_engine"] = ChatEngine(engine_c, data_c)

            chat_engine = st.session_state["chat_engine"]

            # Display conversation history
            for turn in st.session_state["chat_history"]:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

            # Input
            user_msg = st.chat_input(
                "Chiedi qualcosa… es. 'Quanto vale James ai Lakers?'"
            )
            if user_msg:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_msg}
                )
                with st.chat_message("user"):
                    st.markdown(user_msg)

                resp = None
                with st.chat_message("assistant"):
                    with st.spinner("Elaborazione …"):
                        try:
                            resp  = chat_engine.process(
                                user_msg,
                                session_id=st.session_state["chat_session_id"],
                            )
                            reply = resp.reply
                        except Exception as exc:
                            reply = f"⚠️ Errore: {exc}"
                    st.markdown(reply)
                    if (
                        resp is not None
                        and hasattr(resp, "suggestions")
                        and resp.suggestions
                    ):
                        st.caption(
                            "💡 Suggerimenti: "
                            + " | ".join(f"*{s}*" for s in resp.suggestions)
                        )

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": reply}
                )

            if st.button("🗑️ Nuova conversazione", key="chat_clear"):
                st.session_state["chat_history"]   = []
                st.session_state["chat_session_id"] = str(uuid.uuid4())
                st.session_state.pop("chat_engine", None)
                st.rerun()


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
                        from basketball_ai.data.sql_loader import (
                            load_all_data,
                            get_table_mapping,
                            get_engine,
                        )
                        _engine = get_engine()
                        mapping = get_table_mapping(_engine)
                        data = load_all_data(_engine, table_mapping=mapping)
                        st.session_state["data"] = data
                        st.session_state["sql_table_mapping"] = mapping
                        st.success(
                            f"Connected! {len(data['players']):,} players loaded."
                        )
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

        if "sql_table_mapping" in st.session_state:
            with st.expander("🔍 Auto-detected table mapping", expanded=False):
                for logical, actual in st.session_state["sql_table_mapping"].items():
                    match_icon = "✅" if logical == actual else "🔀"
                    st.write(f"{match_icon} **{logical}** → `{actual}`")

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
from basketball_ai.models.performance_model import FEATURE_COLS
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
            from basketball_ai.models.performance_model import PerformanceModel
            from basketball_ai.models.compatibility_model import CompatibilityModel
            from basketball_ai.models.ensemble import EnsembleModel
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
    if compat_path.exists():
        with open(compat_path, "rb") as f:
            st.download_button(
                label="⬇️ Download compatibility_model.joblib",
                data=f.read(),
                file_name="compatibility_model.joblib",
                mime="application/octet-stream",
            )

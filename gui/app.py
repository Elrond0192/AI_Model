"""Basketball Performance AI – Streamlit GUI.

Launch with:
    streamlit run gui/app.py

Tabs:
  1. 📂 Dati       – data source (sub-tabs: CSV locale | Azure SQL Server) + interactive data browser
  2. 🏋️ Training    – load pre-trained model OR configure + run training; results
  3. 🎯 Predizioni  – predict player rating at any team + career trajectory
  4. 🔀 Scenari     – transfer impact, best-team fit, best players, lineup, teammates
  5. 💬 Chat        – natural-language assistant backed by the trained model
"""
from __future__ import annotations

import json
import os
import uuid
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from basketball_ai.data.loader import _to_int
from basketball_ai.utils.helpers import (
    team_display_name as _team_display_name,
    parse_team_display_map,
    TEAM_DISPLAY_FIELD_MAP_ENV,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _save_to_env(key: str, value: str) -> None:
    """Write or update a single KEY=value line in the project .env file."""
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []
    updated = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}\n")
    _ENV_PATH.write_text("".join(lines), encoding="utf-8")

def _safe_model_dir(user_input: str) -> Path:
    """Resolve a user-supplied directory path, preventing '..' path traversal.

    Unlike the previous implementation this function preserves absolute paths
    (e.g. ``/home/user/models`` or ``C:\\Users\\user\\models``) so that the
    user can specify any writable location on disk.  Only ``..`` components are
    removed to neutralise traversal attempts.
    """
    raw = Path((user_input or "models_saved").strip())
    # Walk the parts and drop any '..' that would escape the root
    safe: list[str] = []
    for part in raw.parts:
        if part == "..":
            # Pop last non-root component (never pop the drive/root anchor)
            if safe and safe[-1] not in {"/", "\\", ""} and ":" not in safe[-1]:
                safe.pop()
        else:
            safe.append(part)
    return Path(*safe) if safe else Path("models_saved")


def _require_data():
    """Return data from session_state or show a message and return None."""
    if "data" not in st.session_state:
        st.info("👈 Vai al tab **📂 Dati** e carica i tuoi dati.")
        return None
    return st.session_state["data"]


def _get_team_country_map(data: dict) -> dict:
    """Return a ``{team_id: country_code}`` mapping from loaded data.

    Combines the *teams* → *league_id* and *leagues* → *country* columns.
    Returns an empty dict when the data is missing or incomplete.
    """
    try:
        leagues_df = data.get("leagues")
        teams_df   = data.get("teams")
        if leagues_df is None or teams_df is None:
            return {}
        if "country" not in leagues_df.columns or "id" not in leagues_df.columns:
            return {}
        if "league_id" not in teams_df.columns or "id" not in teams_df.columns:
            return {}
        league_country = {
            _to_int(r["id"]): str(r.get("country", "") or "").strip().upper()
            for _, r in leagues_df.iterrows()
        }
        return {
            _to_int(r["id"]): league_country.get(_to_int(r["league_id"]), "")
            for _, r in teams_df.iterrows()
        }
    except Exception:
        return {}


def _make_team_display_name(team_row, team_country_map: dict, display_map: dict) -> str:
    """Wrapper around :func:`_team_display_name` that injects *country*."""
    team_id = _to_int(team_row.get("id", 0))
    country = team_country_map.get(team_id, "")
    return _team_display_name(team_row, country=country, display_map=display_map)


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
# Helpers – searchable selectors and auto-detection
# ---------------------------------------------------------------------------

def _player_search_select(
    label: str,
    opts: dict,
    key: str,
    container=None,
    placeholder: str = "Digita nome o posizione…",
):
    """Render a search text_input + filtered selectbox for player selection."""
    ctx = container or st
    srch = ctx.text_input(
        "🔍 Cerca giocatore",
        key=f"_s_{key}",
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    q = srch.strip().lower()
    filtered = {k: v for k, v in opts.items() if q in v.lower()} if q else opts
    if not filtered:
        ctx.caption(f"_Nessun risultato per «{srch}»_")
        filtered = opts
    curr = st.session_state.get(key)
    if curr not in filtered and filtered:
        st.session_state[key] = next(iter(filtered))
    return ctx.selectbox(
        label,
        list(filtered.keys()),
        format_func=lambda x: filtered.get(x, str(x)),
        key=key,
    )


def _team_search_select(
    label: str,
    opts: dict,
    key: str,
    container=None,
):
    """Render a search text_input + filtered selectbox for team selection."""
    ctx = container or st
    srch = ctx.text_input(
        "🔍 Cerca squadra",
        key=f"_s_{key}",
        placeholder="Nome squadra…",
        label_visibility="collapsed",
    )
    q = srch.strip().lower()
    filtered = {k: v for k, v in opts.items() if q in v.lower()} if q else opts
    if not filtered:
        ctx.caption(f"_Nessun risultato per «{srch}»_")
        filtered = opts
    curr = st.session_state.get(key)
    if curr not in filtered and filtered:
        st.session_state[key] = next(iter(filtered))
    return ctx.selectbox(
        label,
        list(filtered.keys()),
        format_func=lambda x: filtered.get(x, str(x)),
        key=key,
    )


def _auto_team_for_player(
    player_id,
    data: dict,
    season: int,
    competition: str = "RS",
):
    """Return the team_id the player played for in the given season/competition."""
    if player_id is None:
        return None
    try:
        stats    = data["player_stats"]
        pid      = int(player_id)
        sea      = str(season)
        mask_pid = stats["player_id"] == pid
        mask_sea = (
            stats["season"].astype(str).str.startswith(sea)
            | stats["season"].astype(str).str.startswith(f"{season - 1}-")
        )
        mask_comp = stats["competition"] == competition
        sub = stats[mask_pid & mask_sea & mask_comp]
        if sub.empty:
            sub = stats[mask_pid & mask_sea]
        if sub.empty:
            sub = stats[mask_pid]
        if sub.empty:
            row = data["player_dict"].get(pid, {})
            ct  = row.get("current_team_id")
            return int(ct) if ct else None
        latest = sub.sort_values("season").iloc[-1]
        tid    = latest.get("team_id")
        if tid is not None and not (isinstance(tid, float) and np.isnan(tid)):
            return int(tid)
    except Exception:
        pass
    return None



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

tab_data, tab_train, tab_pred, tab_scen, tab_chat, tab_mapping = st.tabs([
    "📂 Dati",
    "🏋️ Training",
    "🎯 Predizioni",
    "🔀 Scenari",
    "💬 Chat",
    "🗺️ Mapping",
])

# ===========================================================================
# TAB 1 – DATI
# ===========================================================================

with tab_data:
    st.header("📂 Sorgente dati")

    src_csv_tab, src_sql_tab = st.tabs(["📂 CSV (locale)", "🔌 Azure SQL Server"])

    with src_csv_tab:
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
                from basketball_ai.data.loader import (
                    _to_int, _derive_playing_style,
                    _compute_star_player_usage, _fill_current_team_league,
                )
                leagues_df = pd.read_csv(uploaded["leagues"])
                teams_df   = pd.read_csv(uploaded["teams"])
                players_df = pd.read_csv(uploaded["players"])
                stats_df   = pd.read_csv(uploaded["player_stats"])
                rels_df    = pd.read_csv(uploaded["team_player_relations"])

                for col in ["current_team_id", "current_league_id", "draft_year", "draft_pick"]:
                    if col in players_df.columns:
                        players_df[col] = players_df[col].where(players_df[col].notna(), other=None)

                # Convert hex IDs in stats and relations DataFrames
                for col in ["player_id", "team_id", "league_id"]:
                    if col in stats_df.columns:
                        stats_df[col] = stats_df[col].apply(lambda v: None if pd.isna(v) else _to_int(v))
                for col in ["team_id", "player_id"]:
                    if col in rels_df.columns:
                        rels_df[col] = rels_df[col].apply(lambda v: None if pd.isna(v) else _to_int(v))

                # Enrich teams and fill missing player current team/league
                _derive_playing_style(teams_df)
                _compute_star_player_usage(teams_df, stats_df)
                _fill_current_team_league(players_df, stats_df)

                # Ensure competition column exists in uploaded stats
                if "competition" not in stats_df.columns:
                    stats_df["competition"] = "RS"

                league_dict = {_to_int(r["id"]): r.to_dict() for _, r in leagues_df.iterrows()}
                team_dict   = {_to_int(r["id"]): r.to_dict() for _, r in teams_df.iterrows()}
                player_dict = {_to_int(r["id"]): r.to_dict() for _, r in players_df.iterrows()}
                league_teams: dict = {}
                for _, t in teams_df.iterrows():
                    league_teams.setdefault(_to_int(t["league_id"]), []).append(_to_int(t["id"]))

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

    with src_sql_tab:
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
        _save_cs_col, _save_cs_btn_col = st.columns([3, 1])
        _save_cs = _save_cs_col.checkbox(
            "💾 Salva nel file `.env`",
            value=False,
            key="save_conn_to_env",
            help="Sovrascrive `AZURE_SQL_CONNECTION_STRING` nel file `.env` del progetto.",
        )

        if st.button("🔌 Connetti e carica dati SQL", type="primary"):
            if not conn_str.strip():
                st.error("Inserisci la connection string.")
            else:
                with st.spinner("Connessione ad Azure SQL …"):
                    try:
                        os.environ["AZURE_SQL_CONNECTION_STRING"] = conn_str
                        if _save_cs:
                            _save_to_env("AZURE_SQL_CONNECTION_STRING", conn_str)
                            st.toast("✅ Connection string salvata nel file `.env`", icon="💾")
                        # Apply any column-mapping overrides saved in the Mapping tab
                        for env_key, val in st.session_state.get("mapping_env_overrides", {}).items():
                            if val:
                                os.environ[env_key] = val
                            else:
                                os.environ.pop(env_key, None)
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
                        err_str = str(exc)
                        if "08001" in err_str or "timeout" in err_str.lower():
                            st.error(
                                f"Connessione fallita (timeout): {exc}\n\n"
                                "**Suggerimenti:**\n"
                                "- Verifica che il server Azure SQL sia attivo e non in pausa.\n"
                                "- Se usi il tier **Serverless**, il database impiega 60–90 s per "
                                "riattivarsi: aumenta `AZURE_SQL_CONNECT_TIMEOUT` (es. `90`) nel file `.env` "
                                "e riprova.\n"
                                "- Controlla che l'indirizzo del server, le credenziali e il firewall "
                                "Azure consentano la connessione dal tuo IP."
                            )
                        else:
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
                st.dataframe(df_show, width='stretch')
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

    with st.expander("ℹ️ Guida agli iperparametri", expanded=False):
        st.markdown("""
| Parametro | Descrizione | Range consigliato |
|---|---|---|
| `n_estimators` | Numero di alberi XGBoost. Più è alto, più il modello è preciso ma lento. | 200–500 |
| `max_depth` | Profondità massima di ogni albero. Valori alti = rischio overfitting. | 4–7 |
| `learning_rate` | Velocità di apprendimento (eta). Bilanciare con n_estimators: più basso = più alberi necessari. | 0.01–0.1 |
| `subsample` | Frazione dei campioni usata per ogni albero. Riduce overfitting. | 0.7–0.9 |
| `colsample_bytree` | Frazione delle feature usata per ogni albero. Riduce la correlazione tra alberi. | 0.6–0.9 |
| `min_child_weight` | Peso minimo dei campioni in una foglia. Aumentare se il dataset è piccolo. | 2–6 |
| `reg_alpha (L1)` | Regolarizzazione L1: annulla feature poco rilevanti. | 0.0–0.5 |
| `reg_lambda (L2)` | Regolarizzazione L2: riduce smoothly i pesi. | 0.5–2.0 |
| `Validation split` | Frazione dei dati usata per la validazione (non per il training). | 0.10–0.20 |
| `Random seed` | Seme per la riproducibilità dei risultati. | Qualsiasi intero |
        """)

    # -----------------------------------------------------------------------
    # Preset buttons
    # -----------------------------------------------------------------------
    _PRESETS = {
        "🏆 Preciso": dict(
            n_estimators=800, max_depth=7, learning_rate=0.01,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            reg_alpha=0.1, reg_lambda=1.5, test_size=0.15, seed=42,
            label="Alta precisione – addestramento lento (~5–10 min), ottimi risultati",
        ),
        "⚖️ Bilanciato": dict(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, test_size=0.15, seed=42,
            label="Bilanciato – addestramento medio (~1–2 min), buoni risultati (default)",
        ),
        "⚡ Veloce": dict(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.70, colsample_bytree=0.70, min_child_weight=4,
            reg_alpha=0.0, reg_lambda=0.5, test_size=0.20, seed=42,
            label="Veloce – addestramento rapido (<30 s), precisione ridotta",
        ),
    }
    st.markdown("**Preset rapidi**")
    _pcols = st.columns(3)
    for _pi, (_pname, _pvals) in enumerate(_PRESETS.items()):
        with _pcols[_pi]:
            if st.button(_pname, key=f"preset_{_pi}", use_container_width=True,
                         help=_pvals["label"]):
                st.session_state["t_nest"]       = _pvals["n_estimators"]
                st.session_state["t_depth"]      = _pvals["max_depth"]
                st.session_state["t_lr"]         = _pvals["learning_rate"]
                st.session_state["t_sub"]        = _pvals["subsample"]
                st.session_state["t_colsample"]  = _pvals["colsample_bytree"]
                st.session_state["t_mcw"]        = _pvals["min_child_weight"]
                st.session_state["t_alpha"]      = _pvals["reg_alpha"]
                st.session_state["t_lambda"]     = _pvals["reg_lambda"]
                st.session_state["t_split"]      = _pvals["test_size"]
                st.session_state["t_seed"]       = _pvals["seed"]
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        xgb_n_estimators = st.slider(
            "n_estimators", 50, 1000, 300, step=50, key="t_nest",
            help="Numero di alberi. Più è alto, più il modello è preciso ma lento. Range consigliato: 200–500.",
        )
        xgb_max_depth = st.slider(
            "max_depth", 2, 12, 5, key="t_depth",
            help="Profondità massima di ogni albero. Valori alti aumentano il rischio di overfitting. Range consigliato: 4–7.",
        )
        xgb_lr = st.slider(
            "learning_rate", 0.005, 0.3, 0.05, step=0.005, format="%.3f", key="t_lr",
            help="Velocità di apprendimento (eta). Valori bassi richiedono più alberi. Range consigliato: 0.01–0.1.",
        )
        xgb_subsample = st.slider(
            "subsample", 0.4, 1.0, 0.8, step=0.05, key="t_sub",
            help="Frazione dei campioni usata per addestrare ogni albero. Riduce overfitting. Range consigliato: 0.7–0.9.",
        )
        xgb_colsample = st.slider(
            "colsample_bytree", 0.3, 1.0, 0.8, step=0.05, key="t_colsample",
            help="Frazione delle feature usata per ogni albero. Riduce la correlazione tra alberi. Range consigliato: 0.6–0.9.",
        )
        xgb_min_child_weight = st.slider(
            "min_child_weight", 1, 10, 3, key="t_mcw",
            help="Peso minimo dei campioni in una foglia. Aumentare se il dataset è piccolo. Range consigliato: 2–6.",
        )
    with col2:
        st.markdown("**Regolarizzazione**")
        xgb_reg_alpha = st.slider(
            "reg_alpha (L1)", 0.0, 1.0, 0.1, step=0.05, key="t_alpha",
            help="Regolarizzazione L1: annulla feature poco rilevanti. Range consigliato: 0.0–0.5.",
        )
        xgb_reg_lambda = st.slider(
            "reg_lambda (L2)", 0.0, 3.0, 1.0, step=0.1, key="t_lambda",
            help="Regolarizzazione L2: riduce smoothly i pesi. Range consigliato: 0.5–2.0.",
        )
        st.markdown("**Generale**")
        seed = st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42, key="t_seed",
            help="Seme per la riproducibilità dei risultati. Qualsiasi intero va bene.",
        )
        test_size = st.slider(
            "Validation split", 0.05, 0.40, 0.15, step=0.05, key="t_split",
            help="Frazione dei dati usata per la validazione. NON viene usata nel training. Range consigliato: 0.10–0.20.",
        )
        model_dir_gui = st.text_input("Directory salvataggio", value="models_saved", key="t_dir")

    from basketball_ai.models.performance_model import FEATURE_COLS, METRIC_CATALOG, get_available_metrics
    all_features = FEATURE_COLS.copy()
    st.markdown("**Feature selection (base)**")
    selected_features = st.multiselect(
        "Feature da includere (deseleziona per escludere)",
        options=all_features,
        default=all_features,
        key="t_feats",
    )
    if len(selected_features) < 3:
        st.warning("Seleziona almeno 3 feature.")

    # -----------------------------------------------------------------------
    # Metriche aggiuntive dal DB
    # -----------------------------------------------------------------------
    st.subheader("📊 Metriche disponibili dal DB")

    _data_check = st.session_state.get("data")
    if _data_check is None or "player_stats" not in _data_check:
        st.info("Carica prima i dati per vedere le metriche disponibili.")
        extra_metrics: list = []
    else:
        _available_metrics = get_available_metrics(_data_check["player_stats"])
        if not _available_metrics:
            st.info("Nessuna metrica aggiuntiva trovata nelle colonne del dataset.")
            extra_metrics = []
        else:
            st.success(f"✅ {len(_available_metrics)} metriche aggiuntive trovate nel dataset")

            # Group metrics by gruppo
            _groups: dict = {}
            for col in _available_metrics:
                g = METRIC_CATALOG[col].gruppo
                _groups.setdefault(g, []).append(col)

            # Render each group as an expander with two-column checkboxes
            extra_metrics = []
            _GROUP_ICONS = {
                "Attacco": "⚔️", "Playmaking": "🎯", "Rimbalzi": "🏀",
                "Difesa": "🛡️", "Efficienza": "📊", "Disciplina": "📋",
                "Avanzate": "🔬", "Team": "👥", "Utilizzo": "📅",
            }
            for gruppo, group_cols in _groups.items():
                icon = _GROUP_ICONS.get(gruppo, "📌")
                with st.expander(
                    f"{icon} {gruppo} ({len(group_cols)} metriche disponibili)",
                    expanded=(gruppo == "Avanzate"),
                ):
                    gcols = st.columns(2)
                    for i, col_name in enumerate(group_cols):
                        info = METRIC_CATALOG[col_name]
                        checked = gcols[i % 2].checkbox(
                            f"**{info.label}** – {info.description}",
                            value=True,
                            key=f"metric_{col_name}",
                        )
                        if checked:
                            extra_metrics.append(col_name)

            if extra_metrics:
                st.caption(f"🔧 {len(extra_metrics)} metriche extra selezionate per il training")

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
                perf_model.model = XGBRegressor(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_lr,
                    subsample=xgb_subsample,
                    colsample_bytree=xgb_colsample,
                    min_child_weight=xgb_min_child_weight,
                    reg_alpha=xgb_reg_alpha,
                    reg_lambda=xgb_reg_lambda,
                    random_state=int(seed),
                    verbosity=0,
                )
                perf_model.scaler = StandardScaler()

                X_full, y_full = perf_model.prepare_features(data_t, extra_metrics=extra_metrics)
                all_selected = [f for f in selected_features if f in X_full.columns]
                excluded = [f for f in selected_features if f not in X_full.columns]
                if excluded:
                    log(f"Feature escluse (non presenti): {', '.join(excluded)}")
                # Also include any extra-metric feature columns that were added
                for col in extra_metrics:
                    _info = METRIC_CATALOG.get(col)
                    if _info is None:
                        continue
                    feat_name = f"{col}_per_36" if _info.use_per36 else f"avg_{col}"
                    if feat_name in X_full.columns and feat_name not in all_selected:
                        all_selected.append(feat_name)
                perf_model.feature_names = all_selected
                X = X_full[all_selected]
                if extra_metrics:
                    log(f"Metriche extra incluse ({len(extra_metrics)}): {', '.join(extra_metrics)}")
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
                _abs_save = save_dir.resolve()
                log(f"Modelli salvati in: {_abs_save}")
                log("  · performance_model.joblib")
                log("  · compatibility_model.joblib")
                log("  · metadata.json")

                st.session_state["model_dir"]        = str(save_dir)
                st.session_state["ensemble"]         = ensemble
                st.session_state["metrics"]          = metrics_dict
                st.session_state["perf_model"]       = perf_model
                st.session_state["selected_features"] = all_selected
                st.session_state["extra_metrics"]    = extra_metrics
                st.session_state.pop("engine", None)
                st.session_state.pop("chat_engine", None)

                progress.progress(100, text="Completato!")
                st.success(
                    f"✅ Training completato! Modelli salvati in: `{_abs_save}`"
                )

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
                    saved_extra = st.session_state.get("extra_metrics", [])
                    X_full, _   = perf_model_r.prepare_features(st.session_state["data"], extra_metrics=saved_extra)
                    feat_names  = perf_model_r.feature_names
                    # Fill any extra-metric columns not produced by prepare_features with 0
                    for fn in feat_names:
                        if fn not in X_full.columns:
                            X_full[fn] = 0.0
                    X_sample = X_full[feat_names].head(50)
                    X_sc        = perf_model_r.scaler.transform(X_sample)
                    explainer   = _shap.TreeExplainer(perf_model_r.model)
                    shap_vals   = explainer.shap_values(X_sc)
                    mean_shap   = np.abs(shap_vals).mean(axis=0)
                    shap_df = (
                        pd.DataFrame({"feature": feat_names, "mean_|shap|": mean_shap})
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
                _to_int(r["id"]): f"{r['name']} ({r['position']}, {r['age']}a)"
                for _, r in players_df_p.iterrows()
            }
            _display_map_p   = parse_team_display_map()
            _team_country_p  = _get_team_country_map(data_p)
            team_opts_p = {
                _to_int(r["id"]): _make_team_display_name(r, _team_country_p, _display_map_p)
                for _, r in teams_df_p.iterrows()
            }

            # on_change callback: auto-detect team when player or context changes
            def _auto_team_pred_cb():
                pid  = st.session_state.get("pred_player")
                _d   = st.session_state.get("data")
                if pid and _d:
                    auto = _auto_team_for_player(
                        pid, _d,
                        int(st.session_state.get("pred_season", 2024)),
                        st.session_state.get("pred_competition", "RS"),
                    )
                    if auto is not None:
                        st.session_state["pred_team"] = auto

            # --- Context selectors first (season + competition) -----------
            st.subheader("🔮 Predici rating")
            pred_c1, pred_c2 = st.columns(2)
            with pred_c1:
                pred_season = st.number_input(
                    "Stagione", min_value=2000, max_value=2040, value=2024,
                    key="pred_season", on_change=_auto_team_pred_cb,
                )
            with pred_c2:
                sel_competition = st.selectbox(
                    "🏆 Competizione",
                    options=["RS", "PO", "CUP", "SUPERCUP"],
                    index=0,
                    key="pred_competition",
                    on_change=_auto_team_pred_cb,
                    help="**RS** = Regular Season  |  **PO** = Playoff  |  **CUP** = Coppa  |  **SUPERCUP** = Supercoppa",
                )

            # --- Player + team selectors ----------------------------------
            pc1, pc2 = st.columns(2)
            with pc1:
                srch_pp = st.text_input("🔍 Cerca giocatore", key="_s_pred_player",
                                        placeholder="Nome, posizione…",
                                        label_visibility="collapsed")
                _filt_pp = ({k: v for k, v in player_opts_p.items()
                             if srch_pp.strip().lower() in v.lower()}
                            if srch_pp.strip() else player_opts_p)
                if not _filt_pp:
                    st.caption(f"_Nessun risultato per «{srch_pp}»_")
                    _filt_pp = player_opts_p
                _curr_pp = st.session_state.get("pred_player")
                if _curr_pp not in _filt_pp and _filt_pp:
                    st.session_state["pred_player"] = next(iter(_filt_pp))
                sel_player_id = st.selectbox(
                    "🏀 Giocatore", list(_filt_pp.keys()),
                    format_func=lambda x: _filt_pp.get(x, str(x)),
                    key="pred_player", on_change=_auto_team_pred_cb,
                )

            with pc2:
                # Auto-detect team
                _auto_t = _auto_team_for_player(
                    sel_player_id, data_p, int(pred_season), sel_competition
                )
                # Apply staged auto-team selection from previous run (button click)
                if "_pred_team_stage" in st.session_state:
                    st.session_state["pred_team"] = st.session_state["_pred_team_stage"]
                    del st.session_state["_pred_team_stage"]
                elif _auto_t is not None and "pred_team" not in st.session_state:
                    st.session_state["pred_team"] = _auto_t

                srch_pt = st.text_input("🔍 Cerca squadra", key="_s_pred_team",
                                        placeholder="Nome squadra…",
                                        label_visibility="collapsed")
                _filt_pt = ({k: v for k, v in team_opts_p.items()
                             if srch_pt.strip().lower() in v.lower()}
                            if srch_pt.strip() else team_opts_p)
                if not _filt_pt:
                    st.caption(f"_Nessun risultato per «{srch_pt}»_")
                    _filt_pt = team_opts_p
                _curr_pt = st.session_state.get("pred_team")
                if _curr_pt not in _filt_pt and _filt_pt:
                    st.session_state["pred_team"] = next(iter(_filt_pt))
                sel_team_id = st.selectbox(
                    "🏆 Squadra",
                    list(_filt_pt.keys()),
                    format_func=lambda x: _filt_pt.get(x, str(x)),
                    key="pred_team",
                )
                # Show auto-detection info
                if _auto_t is not None:
                    _auto_name = team_opts_p.get(_auto_t, str(_auto_t))
                    if _auto_t == sel_team_id:
                        st.caption(f"🤖 Squadra auto-rilevata: **{_auto_name}**")
                    else:
                        if st.button(f"🤖 Usa squadra auto-rilevata: {_auto_name}",
                                     key="pred_auto_team"):
                            st.session_state["_pred_team_stage"] = _auto_t
                            st.rerun()

            if st.button("🔮 Predici", type="primary", key="pred_run"):
                try:
                    result = engine_p.predict_in_team(
                        sel_player_id, sel_team_id, season=int(pred_season),
                        competition=sel_competition,
                    )
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Rating predetto", f"{result.predicted_rating:.2f} / 10")
                    r2.metric("CI basso",         f"{result.confidence_low:.2f}")
                    r3.metric("CI alto",          f"{result.confidence_high:.2f}")
                    st.caption(f"📋 Competizione: **{result.competition}**")

                    with st.expander("ℹ️ Come interpretare il rating 0–10", expanded=False):
                        st.markdown(
                            "Il **rating** è un punteggio composito che misura il livello di "
                            "rendimento atteso del giocatore in quel contesto squadra/lega:\n\n"
                            "| Fascia | Significato |\n"
                            "|--------|-------------|\n"
                            "| 8.5 – 10 | Superstar / impatto decisivo |\n"
                            "| 7.5 – 8.4 | Titolare di alto livello / All-Star |\n"
                            "| 6.5 – 7.4 | Titolare solido / Rotazione top |\n"
                            "| 5.5 – 6.4 | Buon giocatore di rotazione |\n"
                            "| < 5.5 | Panchina / sviluppo |\n\n"
                            "**Componenti del calcolo:**\n"
                            "- **XGBoost base**: modello addestrato su statistiche storiche "
                            "(forma, consistenza, metriche avanzate SPM/RAPTOR/LEBRON)\n"
                            "- **Compatibilità stile**: quanto lo stile di gioco della squadra "
                            "si adatta al profilo del giocatore (scala 0–1)\n"
                            "- **Fattore lega**: qualità/livello della competizione "
                            "(tier 1 = massima serie = 1.00)\n"
                            "- **Contesto**: adattamento posizione, ruolo, spacing e lega\n"
                            "- **Curva età**: penalizzazione/bonus in base all'età rispetto al picco\n"
                        )

                    st.subheader("📊 Breakdown del rating")
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
                        "Significato": [
                            "Rating grezzo dal modello ML",
                            "Moltiplicatore curva età",
                            "Compatibilità stile squadra (0–1)",
                            "Qualità lega",
                            "Moltiplicatore contesto (posizione+stile+ruolo)",
                        ],
                    })
                    st.dataframe(breakdown_df, width='stretch', hide_index=True)
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

                    # --- Predicted statistics panel -----------------------
                    st.divider()
                    st.subheader("📈 Proiezione statistiche")
                    with st.spinner("Calcolando proiezione statistiche…"):
                        try:
                            stats_proj = engine_p.predict_stats_at_team(
                                sel_player_id, sel_team_id,
                                season=int(pred_season),
                                competition=sel_competition,
                            )
                        except Exception:
                            stats_proj = None
                    if stats_proj:
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.caption(
                            f"Stagione riferimento: **{stats_proj.get('season_reference', '—')}**"
                        )
                        sc2.caption(
                            f"Rating storico medio: **{stats_proj.get('historical_rating', '—')}**"
                        )
                        sc3.caption(
                            f"Fattore scala: **{stats_proj.get('scaling_ratio', 1.0):.2%}**"
                        )
                        _stat_labels = {
                            "points":           "Punti",
                            "rebounds":         "Rimbalzi",
                            "assists":          "Assist",
                            "steals":           "Palle rubate",
                            "blocks":           "Stoppate",
                            "minutes_per_game": "Minuti",
                            "ts_pct":           "TS%",
                            "fg_pct":           "FG%",
                            "three_point_pct":  "3P%",
                            "usg_pct":          "USG%",
                        }
                        _pct_cols = {"ts_pct", "fg_pct", "three_point_pct"}
                        stat_rows = []
                        for col, label in _stat_labels.items():
                            val = stats_proj.get(col)
                            if val is not None:
                                stat_rows.append({
                                    "Statistica": label,
                                    "Valore proiettato": (
                                        f"{val:.1%}" if col in _pct_cols else f"{val:.1f}"
                                    ),
                                })
                        if stat_rows:
                            st.dataframe(
                                pd.DataFrame(stat_rows),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.caption(
                                "⚠️ Le statistiche sono proiezioni scalate dall'ultima stagione "
                                "in base al rapporto rating-predetto / media-storica. "
                                "Non sono previsioni assolute ma stime indicative."
                            )
                    else:
                        st.info(
                            "Non ci sono dati storici sufficienti per proiettare le statistiche "
                            "di questo giocatore."
                        )

                except Exception as exc:
                    st.error(f"Predizione fallita: {exc}")

            st.divider()

            # --- Trajectory -----------------------------------------------
            st.subheader("📈 Traiettoria per età")

            def _auto_team_traj_cb():
                pid = st.session_state.get("traj_player")
                _d  = st.session_state.get("data")
                if pid and _d:
                    auto = _auto_team_for_player(pid, _d, 2024, "RS")
                    if auto is not None:
                        st.session_state["traj_team"] = auto

            tr1, tr2 = st.columns(2)
            with tr1:
                srch_tp = st.text_input("🔍 Cerca giocatore", key="_s_traj_player",
                                        placeholder="Nome, posizione…",
                                        label_visibility="collapsed")
                _filt_tp = ({k: v for k, v in player_opts_p.items()
                             if srch_tp.strip().lower() in v.lower()}
                            if srch_tp.strip() else player_opts_p)
                if not _filt_tp:
                    _filt_tp = player_opts_p
                _curr_tp = st.session_state.get("traj_player")
                if _curr_tp not in _filt_tp and _filt_tp:
                    st.session_state["traj_player"] = next(iter(_filt_tp))
                traj_player = st.selectbox(
                    "Giocatore", list(_filt_tp.keys()),
                    format_func=lambda x: _filt_tp.get(x, str(x)),
                    key="traj_player", on_change=_auto_team_traj_cb,
                )
            with tr2:
                # Auto-detect team for trajectory
                _auto_tr = _auto_team_for_player(traj_player, data_p, 2024, "RS")
                if _auto_tr is not None and "traj_team" not in st.session_state:
                    st.session_state["traj_team"] = _auto_tr
                srch_tt = st.text_input("🔍 Cerca squadra", key="_s_traj_team",
                                        placeholder="Nome squadra…",
                                        label_visibility="collapsed")
                _filt_tt = ({k: v for k, v in team_opts_p.items()
                             if srch_tt.strip().lower() in v.lower()}
                            if srch_tt.strip() else team_opts_p)
                if not _filt_tt:
                    _filt_tt = team_opts_p
                _curr_tt = st.session_state.get("traj_team")
                if _curr_tt not in _filt_tt and _filt_tt:
                    st.session_state["traj_team"] = next(iter(_filt_tt))
                traj_team = st.selectbox(
                    "Squadra (contesto)", list(_filt_tt.keys()),
                    format_func=lambda x: _filt_tt.get(x, str(x)),
                    key="traj_team",
                )
                if _auto_tr is not None and _auto_tr == traj_team:
                    st.caption(f"🤖 Squadra auto-rilevata: **{team_opts_p.get(_auto_tr, '')}**")

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
                            st.dataframe(traj_df, width='stretch')
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
            from basketball_ai.scenarios.engine import (
                ADVANCED_ROLES, ROLE_DIMENSIONS, get_available_roles,
            )
            import numpy as np

            # Default combined-role map used in the manual lineup mode expander.
            _combined_roles = get_available_roles(data_s, dimension="ruolo_combinato")

            players_df_s = data_s["players"]
            teams_df_s   = data_s["teams"]

            player_opts_s = {
                _to_int(r["id"]): f"{r['name']} ({r['position']})"
                for _, r in players_df_s.iterrows()
            }
            _display_map_s  = parse_team_display_map()
            _team_country_s = _get_team_country_map(data_s)
            team_opts_s = {
                _to_int(r["id"]): _make_team_display_name(r, _team_country_s, _display_map_s)
                for _, r in teams_df_s.iterrows()
            }

            def _make_player_selectbox(label, key, exclude_id=None, container=None):
                ctx = container or st
                opts = {k: v for k, v in player_opts_s.items() if k != exclude_id}
                q_raw = ctx.text_input("🔍 Cerca giocatore", key=f"_s_{key}",
                                       placeholder="Nome o posizione…",
                                       label_visibility="collapsed")
                q = q_raw.strip().lower()
                filtered = {k: v for k, v in opts.items() if q in v.lower()} if q else opts
                if not filtered:
                    ctx.caption(f"_Nessun risultato per «{q_raw}»_"); filtered = opts
                curr = st.session_state.get(key)
                if curr not in filtered and filtered:
                    st.session_state[key] = next(iter(filtered))
                return ctx.selectbox(label, list(filtered.keys()),
                                     format_func=lambda x: filtered.get(x, str(x)),
                                     key=key)

            def _make_team_selectbox(label, key, container=None):
                ctx = container or st
                q_raw = ctx.text_input("🔍 Cerca squadra", key=f"_s_{key}",
                                       placeholder="Nome squadra…",
                                       label_visibility="collapsed")
                q = q_raw.strip().lower()
                filtered = ({k: v for k, v in team_opts_s.items() if q in v.lower()}
                            if q else team_opts_s)
                if not filtered:
                    ctx.caption(f"_Nessun risultato per «{q_raw}»_"); filtered = team_opts_s
                curr = st.session_state.get(key)
                if curr not in filtered and filtered:
                    st.session_state[key] = next(iter(filtered))
                return ctx.selectbox(label, list(filtered.keys()),
                                     format_func=lambda x: filtered.get(x, str(x)),
                                     key=key)

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
                tr_player = _make_player_selectbox("Giocatore", "tr_player")
                tr_season = st.number_input("Stagione", 2000, 2040, 2024, key="tr_season")
                tr_c1, tr_c2 = st.columns(2)
                with tr_c1:
                    # Auto-detect current team
                    _auto_from = _auto_team_for_player(tr_player, data_s, int(tr_season), "RS")
                    if _auto_from is not None and "tr_from" not in st.session_state:
                        st.session_state["tr_from"] = _auto_from
                    from_t = _make_team_selectbox("Da squadra", "tr_from", container=tr_c1)
                    if _auto_from is not None and _auto_from == from_t:
                        tr_c1.caption(f"🤖 Auto-rilevata: **{team_opts_s.get(_auto_from, '')}**")
                with tr_c2:
                    to_t = _make_team_selectbox("A squadra", "tr_to", container=tr_c2)

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
                            tm3.metric("Variazione", f"{sign}{res.rating_delta:.2f}",
                                       delta=round(res.rating_delta, 3))
                            st.info(f"**Verdetto:** {res.recommendation}")
                        except Exception as exc:
                            st.error(f"Errore: {exc}")

            # ---- Best teams -----------------------------------------------
            with scen_tabs[1]:
                st.subheader("🏆 Migliori squadre per un giocatore")
                bt_player = _make_player_selectbox("Giocatore", "bt_player")
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
                        st.dataframe(bt_df, width='stretch', hide_index=True)
                    except Exception as exc:
                        st.error(f"Errore: {exc}")

            # ---- Best players ---------------------------------------------
            with scen_tabs[2]:
                st.subheader("👥 Migliori giocatori per una squadra")
                bp_team = _make_team_selectbox("Squadra", "bp_team")
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
                        st.dataframe(bp_df, width='stretch', hide_index=True)
                    except Exception as exc:
                        st.error(f"Errore: {exc}")

            # ---- Lineup ---------------------------------------------------
            with scen_tabs[3]:
                st.subheader("📋 Costruttore quintetto")

                # Shared selectors for both modes
                lu_player = _make_player_selectbox("🏀 Giocatore target", "lu_player")
                lu_team   = _make_team_selectbox("🏆 Squadra contesto", "lu_team")
                lu_season = st.number_input("Stagione", 2000, 2040, 2024, key="lu_season")

                # Mode switcher
                lu_mode = st.radio(
                    "Modalità costruzione quintetto",
                    ["🎯 Per giocatori", "📋 Per ruoli avanzati"],
                    horizontal=True, key="lu_mode",
                )

                if lu_mode == "🎯 Per giocatori":
                    # ---- Manual mode: pick teammates ----------------------
                    with st.expander("ℹ️ Ruoli disponibili (combinato)", expanded=False):
                        for r_key, r_desc in _combined_roles.items():
                            st.markdown(f"**{r_key}** — {r_desc}" if r_key != r_desc else f"**{r_key}**")

                    other_opts = {k: v for k, v in player_opts_s.items() if k != lu_player}
                    # Search for multiselect
                    lu_search = st.text_input("🔍 Filtra compagni", key="_s_lu_lineup",
                                             placeholder="Nome o posizione…",
                                             label_visibility="collapsed")
                    q_lu = lu_search.strip().lower()
                    filtered_lu = ({k: v for k, v in other_opts.items()
                                    if q_lu in v.lower()} if q_lu else other_opts)
                    lu_lineup = st.multiselect(
                        "Compagni di quintetto (1–4 giocatori)",
                        options=list(filtered_lu.keys()),
                        format_func=lambda x: filtered_lu.get(x, str(x)),
                        max_selections=4,
                        key="lu_lineup",
                    )

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
                                st.dataframe(lu_df, width='stretch', hide_index=True)
                                st.caption(f"_{res.explanation}_")

                                # Synergy analysis
                                all_ids = [lu_player] + list(lu_lineup)
                                if len(all_ids) >= 2:
                                    with st.spinner("Calcolando sinergia…"):
                                        syn = engine_s.compute_lineup_synergy(
                                            all_ids, lu_team, season=int(lu_season)
                                        )
                                    st.divider()
                                    st.subheader("⚡ Analisi sinergia")
                                    s1, s2, s3, s4 = st.columns(4)
                                    s1.metric("Sinergia complessiva", f"{syn.overall_synergy:.1f} / 10")
                                    s2.metric("Diversità ruoli",     f"{syn.role_diversity_score:.0%}")
                                    s3.metric("Bilanc. offensivo",   f"{syn.offensive_balance:.0%}")
                                    s4.metric("Score difensivo",     f"{syn.defensive_score:.1f}")
                                    st.info(f"📊 {syn.summary}")
                                    if syn.missing_roles:
                                        st.warning(f"⚠️ Ruoli mancanti: {', '.join(syn.missing_roles)}")

                                    st.markdown("**Distribuzione ruoli avanzati**")
                                    for role, names in syn.role_distribution.items():
                                        st.markdown(f"- **{role}**: {', '.join(names)}")

                                    if syn.pairwise_compat:
                                        with st.expander("🤝 Compatibilità a coppie", expanded=False):
                                            pw_df = pd.DataFrame(
                                                [(pair, f"{score:.0%}")
                                                 for pair, score in syn.pairwise_compat.items()],
                                                columns=["Coppia", "Compatibilità"],
                                            )
                                            st.dataframe(pw_df, width='stretch', hide_index=True)

                            except Exception as exc:
                                st.error(f"Errore analisi quintetto: {exc}")

                else:
                    # ---- Role-based mode ----------------------------------
                    st.markdown("Seleziona i **ruoli** dei compagni che vuoi attorno al giocatore target.")

                    # Dimension selector: which role column to use for filtering
                    _dim_labels = {
                        "ruolo_combinato": "🔄 Ruolo combinato (offensivo + difensivo)",
                        "ruolo_offensivo": "⚡ Solo ruolo offensivo",
                        "ruolo_difensivo": "🛡️ Solo ruolo difensivo",
                    }
                    _sel_dim = st.radio(
                        "Filtra per dimensione ruolo",
                        options=list(_dim_labels.keys()),
                        format_func=lambda x: _dim_labels[x],
                        horizontal=True,
                        key="lu_role_dimension",
                    )
                    # Build available roles from the chosen dimension
                    _available_roles = get_available_roles(data_s, dimension=_sel_dim)
                    _using_db_roles = _available_roles != dict(ADVANCED_ROLES)
                    if _using_db_roles:
                        st.caption(f"ℹ️ Ruoli letti dal database (`{_sel_dim}`).")
                    else:
                        st.caption("ℹ️ Ruoli classificati euristicamente (colonna DB non presente o vuota).")

                    with st.expander("ℹ️ Ruoli disponibili", expanded=True):
                        for r_key, r_desc in _available_roles.items():
                            st.markdown(f"**{r_key}** — {r_desc}" if r_key != r_desc else f"**{r_key}**")

                    desired_roles = st.multiselect(
                        "🎭 Ruoli desiderati (fino a 4)",
                        options=list(_available_roles.keys()),
                        default=list(_available_roles.keys())[:min(3, len(_available_roles))],
                        max_selections=4,
                        key="lu_roles",
                        format_func=lambda x: x,
                    )
                    top_n_role = st.slider("Candidati per ruolo", 3, 10, 5, key="lu_top_n")

                    if st.button("🔍 Trova quintetto per ruoli", type="primary", key="lu_roles_run"):
                        if not desired_roles:
                            st.warning("Seleziona almeno un ruolo.")
                        else:
                            with st.spinner("Analizzando i migliori candidati per ogni ruolo…"):
                                try:
                                    result_r = engine_s.best_lineup_by_roles(
                                        lu_player, lu_team,
                                        desired_roles=desired_roles,
                                        season=int(lu_season),
                                        top_n_per_role=int(top_n_role),
                                        role_dimension=_sel_dim,
                                    )

                                    st.success(
                                        f"✅ Quintetto ottimale per **{result_r.target_player_name}** "
                                        f"— Rating medio: **{result_r.estimated_avg_rating:.2f}**"
                                    )

                                    # Show optimal lineup with all three role columns
                                    if result_r.optimal_lineup:
                                        opt_df = pd.DataFrame([{
                                            "Ruolo target":       p["target_role"],
                                            "Giocatore":          p["player_name"],
                                            "Posizione":          p["position"],
                                            "Squadra attuale":    p["current_team"],
                                            "Rating predetto":    p["predicted_rating"],
                                            "Ruolo combinato":    p.get("ruolo_combinato", ""),
                                            "Ruolo offensivo":    p.get("ruolo_offensivo", ""),
                                            "Ruolo difensivo":    p.get("ruolo_difensivo", ""),
                                        } for p in result_r.optimal_lineup])
                                        st.dataframe(opt_df, width='stretch', hide_index=True)

                                    # Per-role candidates
                                    st.divider()
                                    st.subheader("📋 Candidati per ruolo")
                                    for rc in result_r.role_candidates:
                                        with st.expander(
                                            f"**{rc.role}** — {rc.description}", expanded=False
                                        ):
                                            if rc.candidates:
                                                cand_df = pd.DataFrame([{
                                                    "Giocatore":       c["player_name"],
                                                    "Posizione":       c["position"],
                                                    "Squadra":         c["current_team"],
                                                    "Rating predetto": c["predicted_rating"],
                                                    "Ruolo combinato": c.get("ruolo_combinato", ""),
                                                    "Ruolo offensivo": c.get("ruolo_offensivo", ""),
                                                    "Ruolo difensivo": c.get("ruolo_difensivo", ""),
                                                } for c in rc.candidates])
                                                st.dataframe(cand_df, width='stretch',
                                                             hide_index=True)
                                            else:
                                                st.caption("Nessun candidato trovato per questo ruolo.")

                                    # Synergy of optimal lineup
                                    opt_ids = [lu_player] + [p["player_id"]
                                                              for p in result_r.optimal_lineup]
                                    if len(opt_ids) >= 2:
                                        with st.spinner("Calcolando sinergia quintetto ottimale…"):
                                            syn_r = engine_s.compute_lineup_synergy(
                                                opt_ids, lu_team, season=int(lu_season)
                                            )
                                        st.divider()
                                        st.subheader("⚡ Sinergia quintetto ottimale")
                                        sr1, sr2, sr3, sr4 = st.columns(4)
                                        sr1.metric("Sinergia complessiva", f"{syn_r.overall_synergy:.1f} / 10")
                                        sr2.metric("Diversità ruoli",     f"{syn_r.role_diversity_score:.0%}")
                                        sr3.metric("Bilanc. offensivo",   f"{syn_r.offensive_balance:.0%}")
                                        sr4.metric("Score difensivo",     f"{syn_r.defensive_score:.1f}")
                                        st.info(f"📊 {syn_r.summary}")
                                        if syn_r.missing_roles:
                                            st.warning(
                                                f"⚠️ Ruoli mancanti: {', '.join(syn_r.missing_roles)}"
                                            )
                                        if syn_r.pairwise_compat:
                                            with st.expander("🤝 Compatibilità a coppie", expanded=False):
                                                pwr_df = pd.DataFrame(
                                                    [(pair, f"{score:.0%}")
                                                     for pair, score in syn_r.pairwise_compat.items()],
                                                    columns=["Coppia", "Compatibilità"],
                                                )
                                                st.dataframe(pwr_df, width='stretch', hide_index=True)

                                except Exception as exc:
                                    st.error(f"Errore ricerca per ruoli: {exc}")

            # ---- Teammates ------------------------------------------------
            with scen_tabs[4]:
                st.subheader("🤝 Scenario compagni ipotetici")
                tm_player = _make_player_selectbox("Giocatore", "tm_player")
                tm_team   = _make_team_selectbox("Squadra", "tm_team")
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


with tab_mapping:
    st.header("🗺️ Mapping colonne DB → Modello")

    try:
        from basketball_ai.data.schema_mapping import ALL_TABLES, ColumnDef
        _mapping_available = True
    except ImportError:
        st.error("schema_mapping.py non trovato. Assicurati che il file esista in basketball_ai/data/.")
        _mapping_available = False

    # Env-var names per tabella (devono coincidere con sql_loader)
    _MAPPING_ENV_VARS = {
        "leagues":               "AZURE_SQL_COLUMN_RENAMES_LEAGUES",
        "teams":                 "AZURE_SQL_COLUMN_RENAMES_TEAMS",
        "players":               "AZURE_SQL_COLUMN_RENAMES_PLAYERS",
        "player_stats":          "AZURE_SQL_COLUMN_RENAMES_PLAYER_STATS",
        "team_player_relations": "AZURE_SQL_COLUMN_RENAMES_TEAM_PLAYER_RELATIONS",
    }
    _TABLE_LABELS = {
        "leagues":               "🏅 Leghe  (Configuration.Championship_ids)",
        "teams":                 "🏆 Squadre  (Anagrafiche.Team_*)",
        "players":               "👥 Giocatori  (Anagrafiche.*)",
        "player_stats":          "📊 Statistiche giocatori  (Analisi.AdvancedStats_Player_*)",
        "team_player_relations": "🔗 Relazioni giocatore-squadra  (Anagrafiche.*)",
    }

    _map_col_tab, _map_display_tab = st.tabs([
        "🗂️ Colonne DB",
        "🏷️ Display Nome Squadra",
    ])

    # -----------------------------------------------------------------------
    # Sub-tab 1 – Column mapping
    # -----------------------------------------------------------------------
    with _map_col_tab:
        st.markdown(
            "Configura la corrispondenza tra le colonne del **database reale** e i campi "
            "attesi dal modello. Modifica la colonna **DB Column** con il nome esatto della "
            "colonna nel tuo database. Le modifiche vengono applicate al prossimo caricamento SQL."
        )

        if _mapping_available:
            # Initialise session state with current env-var values on first load
            if "mapping_env_overrides" not in st.session_state:
                st.session_state["mapping_env_overrides"] = {
                    ev: os.environ.get(ev, "")
                    for ev in _MAPPING_ENV_VARS.values()
                }

            # Parse current env-var overrides into {logical: {db_col: logical_col}}
            def _parse_overrides(env_val: str) -> dict:
                result = {}
                for pair in env_val.split(","):
                    pair = pair.strip()
                    if ":" in pair:
                        src, _, dst = pair.partition(":")
                        result[src.strip()] = dst.strip()
                return result

            any_changed = False

            for logical, col_defs in ALL_TABLES.items():
                env_key = _MAPPING_ENV_VARS[logical]
                current_overrides = _parse_overrides(
                    st.session_state["mapping_env_overrides"].get(env_key, "")
                )

                # Build display dataframe
                rows = []
                for cd in col_defs:
                    # DB column: prefer saved override (by logical_col), then schema default
                    db_col_val = ""
                    for db_c, log_c in current_overrides.items():
                        if log_c == cd.logical_col:
                            db_col_val = db_c
                            break
                    if not db_col_val and cd.db_col:
                        db_col_val = cd.db_col

                    rows.append({
                        "Campo modello": cd.logical_col,
                        "Colonna DB": db_col_val,
                        "Default (se assente)": str(cd.default) if cd.default is not None else "",
                        "Derivata": "✓" if cd.compute is not None else "",
                        "Descrizione": cd.description,
                    })

                df_mapping = pd.DataFrame(rows)

                with st.expander(_TABLE_LABELS.get(logical, logical), expanded=False):
                    st.caption(
                        "• **Campo modello** — nome interno usato dall'applicazione (non modificare).  \n"
                        "• **Colonna DB** — nome esatto della colonna nel database (modifica qui).  \n"
                        "• **Default** — valore usato se la colonna è assente nel DB.  \n"
                        "• **Derivata** — ✓ indica una colonna calcolata automaticamente (es. per-game stats)."
                    )

                    edited_df = st.data_editor(
                        df_mapping,
                        key=f"mapping_editor_{logical}",
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Campo modello": st.column_config.TextColumn(
                                "Campo modello", disabled=True, width="medium"
                            ),
                            "Colonna DB": st.column_config.TextColumn(
                                "Colonna DB", width="medium",
                                help="Nome esatto della colonna nel tuo database."
                            ),
                            "Default (se assente)": st.column_config.TextColumn(
                                "Default", disabled=True, width="small"
                            ),
                            "Derivata": st.column_config.TextColumn(
                                "Derivata", disabled=True, width="small"
                            ),
                            "Descrizione": st.column_config.TextColumn(
                                "Descrizione", disabled=True, width="large"
                            ),
                        },
                    )

                    # Check if anything was edited
                    if not edited_df["Colonna DB"].equals(df_mapping["Colonna DB"]):
                        any_changed = True

                    # Store edited mapping back to session state
                    pairs = []
                    for _, row in edited_df.iterrows():
                        db_c = str(row["Colonna DB"]).strip()
                        log_c = str(row["Campo modello"]).strip()
                        if db_c and log_c and db_c != log_c:
                            pairs.append(f"{db_c}:{log_c}")
                    st.session_state["mapping_env_overrides"][env_key] = ",".join(pairs)

            st.divider()
            col_save, col_reset, col_info = st.columns([2, 2, 6])

            with col_save:
                if st.button("💾 Salva e applica mapping", type="primary", key="mapping_save"):
                    for env_key, val in st.session_state["mapping_env_overrides"].items():
                        if val:
                            os.environ[env_key] = val
                        else:
                            os.environ.pop(env_key, None)
                    st.success(
                        "✅ Mapping salvato. Torna al tab **📂 Dati** e ricarica i dati SQL "
                        "per applicare le modifiche."
                    )

            with col_reset:
                if st.button("↩️ Ripristina default", key="mapping_reset"):
                    st.session_state.pop("mapping_env_overrides", None)
                    for ev in _MAPPING_ENV_VARS.values():
                        os.environ.pop(ev, None)
                    st.rerun()

            with col_info:
                if any_changed:
                    st.info("⚠️ Hai modifiche non salvate — premi **Salva e applica mapping**.")

            # Show current env-var values for debugging
            with st.expander("🔧 Variabili d'ambiente correnti (debug)", expanded=False):
                for logical, env_key in _MAPPING_ENV_VARS.items():
                    val = os.environ.get(env_key, "")
                    st.code(f"{env_key}={val or '(non impostata)'}", language="bash")

    # -----------------------------------------------------------------------
    # Sub-tab 2 – Team display-name mapping
    # -----------------------------------------------------------------------
    with _map_display_tab:
        st.markdown(
            "Scegli quale campo usare come **nome visualizzato** delle squadre in ogni nazione.  \n"
            "- **TeamName** — nome completo della squadra (es. *Olimpia Milano*)  \n"
            "- **ShortName** — abbreviazione/acronimo dal DB (es. *MIL*, *OLI*)  \n\n"
            "La configurazione viene salvata nel file `.env` e applicata immediatamente."
        )

        # ----- Auto-populate nations from loaded data -----------------------
        _loaded_nations: list[str] = []
        if "data" in st.session_state:
            _lg_df = st.session_state["data"].get("leagues")
            if _lg_df is not None and "country" in _lg_df.columns:
                _loaded_nations = sorted(
                    {str(c).strip().upper() for c in _lg_df["country"].dropna() if str(c).strip()}
                )

        # ----- Load persisted mapping from env-var --------------------------
        if "team_display_map_rows" not in st.session_state:
            _initial_map = parse_team_display_map(os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, ""))
            # Seed with all nations found in DB (if any), plus any already saved
            _seed_nations = _loaded_nations or sorted(_initial_map.keys())
            st.session_state["team_display_map_rows"] = [
                {
                    "Nazione (codice ISO)": cc,
                    "Campo da visualizzare": _initial_map.get(cc, "short_name"),
                }
                for cc in _seed_nations
            ] or [{"Nazione (codice ISO)": "", "Campo da visualizzare": "short_name"}]
        else:
            # If new nations were loaded from DB, add any missing ones
            _existing_ccs = {
                r["Nazione (codice ISO)"].strip().upper()
                for r in st.session_state["team_display_map_rows"]
                if r.get("Nazione (codice ISO)", "").strip()
            }
            _current_saved = parse_team_display_map(os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, ""))
            for _cc in _loaded_nations:
                if _cc and _cc not in _existing_ccs:
                    st.session_state["team_display_map_rows"].append({
                        "Nazione (codice ISO)": _cc,
                        "Campo da visualizzare": _current_saved.get(_cc, "short_name"),
                    })

        if not _loaded_nations:
            st.info(
                "ℹ️ Nessun dato SQL caricato. Puoi aggiungere manualmente le nazioni oppure "
                "carica prima i dati dal tab **📂 Dati** per auto-popolare la lista."
            )

        _display_df = pd.DataFrame(st.session_state["team_display_map_rows"])

        _edited_display = st.data_editor(
            _display_df,
            key="team_display_map_editor",
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Nazione (codice ISO)": st.column_config.TextColumn(
                    "Nazione (codice ISO)",
                    width="medium",
                    help="Codice ISO a 2 lettere della nazione, es. IT, ES, FR, DE, GR …",
                ),
                "Campo da visualizzare": st.column_config.SelectboxColumn(
                    "Campo da visualizzare",
                    width="medium",
                    options=["short_name", "name"],
                    help=(
                        "**short_name** → usa ShortName dal DB (abbreviazione).  \n"
                        "**name** → usa TeamName dal DB (nome completo)."
                    ),
                    required=True,
                ),
            },
        )

        # ----- Preview ------------------------------------------------------
        if "data" in st.session_state and not _edited_display.empty:
            _prev_map = {
                str(r["Nazione (codice ISO)"]).strip().upper(): r["Campo da visualizzare"]
                for _, r in _edited_display.iterrows()
                if str(r.get("Nazione (codice ISO)", "")).strip()
                   and r.get("Campo da visualizzare") in ("name", "short_name")
            }
            _prev_country_map = _get_team_country_map(st.session_state["data"])
            _teams_preview = st.session_state["data"].get("teams")
            if _teams_preview is not None and not _teams_preview.empty:
                with st.expander("👁️ Anteprima nomi squadre", expanded=True):
                    _prev_rows = []
                    for _, _tr in _teams_preview.iterrows():
                        _tid = _to_int(_tr.get("id", 0))
                        _cc = _prev_country_map.get(_tid, "")
                        _prev_rows.append({
                            "Squadra (nome DB)": str(_tr.get("name", "")),
                            "ShortName DB": str(_tr.get("short_name", "") or ""),
                            "Nazione": _cc,
                            "Visualizzato come": _make_team_display_name(_tr, _prev_country_map, _prev_map),
                        })
                    st.dataframe(
                        pd.DataFrame(_prev_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

        st.divider()
        _dc1, _dc2, _dc3 = st.columns([2, 2, 6])

        with _dc1:
            if st.button("💾 Salva display mapping", type="primary", key="display_map_save"):
                _new_map_str = ",".join(
                    f"{str(r['Nazione (codice ISO)']).strip().upper()}:{r['Campo da visualizzare']}"
                    for _, r in _edited_display.iterrows()
                    if str(r.get("Nazione (codice ISO)", "")).strip()
                       and r.get("Campo da visualizzare") in ("name", "short_name")
                )
                os.environ[TEAM_DISPLAY_FIELD_MAP_ENV] = _new_map_str
                _save_to_env(TEAM_DISPLAY_FIELD_MAP_ENV, _new_map_str)
                # Update session state so subsequent reruns use the new values
                st.session_state["team_display_map_rows"] = _edited_display.to_dict("records")
                st.toast("✅ Display mapping salvato. Aggiornamento in corso…", icon="💾")
                st.rerun()

        with _dc2:
            if st.button("↩️ Ripristina default", key="display_map_reset"):
                st.session_state.pop("team_display_map_rows", None)
                os.environ.pop(TEAM_DISPLAY_FIELD_MAP_ENV, None)
                _save_to_env(TEAM_DISPLAY_FIELD_MAP_ENV, "")
                st.rerun()

        with _dc3:
            _cur_env = os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, "")
            if _cur_env:
                st.caption(f"**Env-var corrente:** `{TEAM_DISPLAY_FIELD_MAP_ENV}={_cur_env}`")
            else:
                st.caption(
                    f"**Env-var:** `{TEAM_DISPLAY_FIELD_MAP_ENV}` non impostata — "
                    "comportamento di default: usa ShortName se disponibile, altrimenti TeamName."
                )

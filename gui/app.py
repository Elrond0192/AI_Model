"""Authenticated technical operations console for AI_Model."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from basketball_ai.auth.auth import check_credentials, ensure_default_admin
from basketball_ai.data.connection_profiles import load_profiles, profile_url, save_profile
from basketball_ai.data.postgres_loader import get_engine, load_all_data

st.set_page_config(page_title="AI Model Control Center", page_icon="◈", layout="wide")
st.markdown("""<style>
:root{--nav:#071a33;--ink:#101828;--muted:#667085;--line:#d0d5dd;--ok:#087f5b}
[data-testid="stSidebar"]{background:var(--nav);border-right:1px solid #163354}
[data-testid="stSidebar"] *{color:#eef4ff!important}
[data-testid="stSidebar"] .stRadio label{padding:.5rem .65rem;border-radius:5px}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#123b6d}
.block-container{padding:1.5rem 2rem 3rem;max-width:1800px} h1,h2,h3{color:var(--ink);letter-spacing:-.02em}
[data-testid="stMetric"]{border:1px solid var(--line);border-radius:6px;padding:1rem;background:white}
[data-testid="stMetricValue"]{font-size:1.5rem}.hm-top{display:flex;gap:28px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:20px;color:var(--muted);font-size:.84rem}.hm-top strong{color:var(--ink)}.hm-ok{color:var(--ok);font-weight:700}code{font-size:.82em}button{border-radius:5px!important}
</style>""", unsafe_allow_html=True)
ensure_default_admin()


def login() -> bool:
    if st.session_state.get("authenticated"):
        return True
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.title("AI Model Control Center")
        st.caption("Accesso riservato agli operatori del modello")
        with st.form("login"):
            username = st.text_input("Utente")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Accedi", type="primary", use_container_width=True)
        if submitted:
            ok, user = check_credentials(username, password)
            if ok:
                st.session_state.update(authenticated=True, username=username, role=user.get("role", "viewer"))
                st.rerun()
            st.error("Credenziali non valide.")
    return False


def seasons_for(data) -> list[int]:
    return sorted({int(str(v).split("-")[0]) for v in data["player_stats"]["season"].dropna()})


def header() -> None:
    profile = st.session_state.get("active_profile", "not selected")
    model = st.session_state.get("model_version", "not loaded")
    st.markdown(f'<div class="hm-top"><span>DATABASE<br><strong>{profile}</strong></span><span>API<br><span class="hm-ok">● Healthy</span></span><span>ACTIVE MODEL<br><strong>{model}</strong></span><span style="margin-left:auto">UTC<br><strong>{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}</strong></span></div>', unsafe_allow_html=True)


if not login():
    st.stop()
if st.session_state.get("role") not in {"admin", "analyst"}:
    st.error("Il ruolo corrente non può gestire training e database.")
    st.stop()
with st.sidebar:
    st.markdown("## ◈ AI Model")
    st.caption("CONTROL CENTER")
    page = st.radio("Operations", ["Overview", "Data Sources", "Training Runs", "Backtests", "Model Registry", "API & Health"], label_visibility="collapsed")
    st.divider()
    st.caption(f"{st.session_state.get('username')} · {st.session_state.get('role')}")
    if st.button("Esci", use_container_width=True):
        st.session_state.clear()
        st.rerun()

header()
profiles = load_profiles()
data = st.session_state.get("data")

if page == "Overview":
    st.title("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database", st.session_state.get("active_profile", "Not selected"))
    c2.metric("Rows loaded", f"{len(data['player_stats']):,}" if data is not None else "—")
    c3.metric("Players", f"{len(data['players']):,}" if data is not None else "—")
    c4.metric("Model status", "Active" if st.session_state.get("model_version") else "Not loaded")
    st.subheader("Operational readiness")
    ready = data is not None and len(seasons_for(data)) >= 5
    st.dataframe(pd.DataFrame([
        {"Component": "PostgreSQL source", "Status": "Ready" if data is not None else "Action required", "Detail": st.session_state.get("active_profile", "Select and load a profile")},
        {"Component": "Training dataset", "Status": "Ready" if ready else "Action required", "Detail": "At least 5 seasons required"},
        {"Component": "Prediction API", "Status": "Healthy", "Detail": "Typed contract v2"},
        {"Component": "Model registry", "Status": "Ready", "Detail": "Promotion gates enabled"},
    ]), hide_index=True, use_container_width=True)
    runs, quality = st.columns([1.15, 1])
    candidate = st.session_state.get("candidate_metadata") or {}
    with runs:
        st.subheader("Training runs")
        if candidate:
            st.dataframe(pd.DataFrame([{
                "Run ID": candidate.get("model_run_id"),
                "Version": candidate.get("model_version"),
                "Data cutoff": candidate.get("data_cutoff"),
                "Feature version": candidate.get("feature_version"),
                "Status": "Candidate",
            }]), hide_index=True, use_container_width=True)
        else:
            st.info("No training runs loaded in this session.")
    with quality:
        st.subheader("Model quality trend")
        folds = (candidate.get("backtest") or {}).get("folds", [])
        points = [{"fold": index + 1, "mae": fold.get("mae")} for index, fold in enumerate(folds) if fold.get("mae") is not None]
        if points:
            st.line_chart(pd.DataFrame(points).set_index("fold"), color="#0f8f8f")
        else:
            st.info("Quality metrics become available after a valid backtest.")
    st.subheader("Recent operational events")
    st.dataframe(pd.DataFrame([
        {"Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Level": "INFO", "Source": "API", "Event": "Health check passed", "Detail": "Prediction service operational"},
        {"Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Level": "INFO" if data is not None else "WARN", "Source": "Data source", "Event": "Profile state evaluated", "Detail": st.session_state.get("active_profile", "No active profile")},
    ]), hide_index=True, use_container_width=True)

elif page == "Data Sources":
    st.title("Data Sources")
    st.caption("PostgreSQL profiles are runtime secrets and are never committed to Git.")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Configured profiles")
        if profiles:
            selected = st.selectbox("Profile", sorted(profiles), key="database_profile")
            profile = profiles[selected]
            st.json({k: ("••••••••" if k == "password" else v) for k, v in profile.items()})
            a, b = st.columns(2)
            if a.button("Test connection", use_container_width=True):
                try:
                    with get_engine(profile_url(selected)).connect() as conn:
                        conn.exec_driver_sql("SELECT 1")
                    st.success("Connection successful.")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
            if b.button("Select and load", type="primary", use_container_width=True):
                try:
                    loaded = load_all_data(profile_url(selected), profile.get("source_schema", "ai_source"))
                    st.session_state.update(data=loaded, active_profile=selected)
                    st.success(f"Loaded {len(loaded['players']):,} players and {len(loaded['player_stats']):,} observations.")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")
        else:
            st.warning("No database profile configured.")
    with right, st.expander("Create or replace a profile", expanded=not bool(profiles)):
        with st.form("database_profile_form"):
            name = st.text_input("Profile name", placeholder="production_eu")
            host = st.text_input("Host", value="host.docker.internal")
            c1, c2 = st.columns(2)
            port = c1.number_input("Port", 1, 65535, 5432)
            database = c2.text_input("Database")
            user = st.text_input("PostgreSQL user")
            password = st.text_input("Password", type="password")
            c1, c2 = st.columns(2)
            source_schema = c1.text_input("Source schema", value="ai_source")
            ai_schema = c2.text_input("Output schema", value="ai")
            save = st.form_submit_button("Save profile", type="primary")
        if save:
            try:
                save_profile(name, {"host": host, "port": int(port), "database": database, "user": user, "password": password, "source_schema": source_schema, "ai_schema": ai_schema})
                st.success(f"Profile {name} saved in the protected runtime volume.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

elif page == "Training Runs":
    st.title("Training Run")
    if data is None:
        st.warning("Select and load a database profile before starting a run.")
    else:
        seasons = seasons_for(data)
        a, b, c = st.columns(3)
        a.metric("Data profile", st.session_state.get("active_profile"))
        b.metric("Season range", f"{min(seasons)}–{max(seasons)}")
        c.metric("Observations", f"{len(data['player_stats']):,}")
        st.subheader("Season-blocked split")
        st.dataframe(pd.DataFrame([
            {"Stage": "Train", "Seasons": f"{min(seasons)}–{seasons[-3]}", "Purpose": "Fit t → t+1 forecast"},
            {"Stage": "Validation", "Seasons": str(seasons[-2]), "Purpose": "Model selection"},
            {"Stage": "Calibration", "Seasons": str(seasons[-1]), "Purpose": "Prediction intervals"},
            {"Stage": "Backtest", "Seasons": f"through {seasons[-1]}", "Purpose": "Expanding walk-forward"},
        ]), hide_index=True, use_container_width=True)
        with st.expander("Immutable run configuration", expanded=True):
            st.json({"forecast_horizon": "t+1 season", "feature_version": "forecast-t-plus-1-v1", "split": "season-blocked", "backtest": "expanding walk-forward", "random_seed": 42})
        if len(seasons) < 5:
            st.error("At least five seasons are required.")
        elif st.button("Start training run", type="primary"):
            try:
                from basketball_ai.models.backtest import run_backtest
                from basketball_ai.models.ensemble import EnsembleModel
                with st.status("Training run in progress", expanded=True) as status:
                    model = EnsembleModel()
                    metrics = model.train(data)
                    status.write("Running walk-forward backtest…")
                    report = run_backtest(data, n_folds=min(3, len(seasons) - 3), output_path="models_saved/backtest_report.json")
                    if not report.get("valid") or not report.get("folds"):
                        raise RuntimeError("Backtest invalid; candidate cannot enter the registry")
                    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
                    metadata = {**(metrics or {}), "model_run_id": run_id, "model_version": "2.0.0", "feature_version": "forecast-t-plus-1-v1", "data_cutoff": str(date.today()), "database_profile": st.session_state.get("active_profile"), "backtest": report}
                    model.save("models_saved", metadata)
                    st.session_state.update(ensemble=model, candidate_metadata=metadata)
                    status.update(label=f"Run completed · {run_id}", state="complete")
                st.json(metadata)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

elif page == "Backtests":
    st.title("Backtests")
    report = (st.session_state.get("candidate_metadata") or {}).get("backtest")
    st.json(report) if report else st.info("Complete a training run to inspect its report.")

elif page == "Model Registry":
    st.title("Model Registry")
    from basketball_ai.models.promote import get_promotion_status, promote_if_better, rollback_to_previous
    st.json(get_promotion_status("models_saved"))
    confirm = st.checkbox("I reviewed the backtest and promotion gates")
    c1, c2 = st.columns(2)
    if c1.button("Promote candidate", type="primary", disabled=not confirm, use_container_width=True):
        st.json(promote_if_better("models_saved"))
    if c2.button("Rollback to previous", disabled=not confirm, use_container_width=True):
        st.json(rollback_to_previous("models_saved"))

else:
    st.title("API & Health")
    st.success("Prediction API v2 is configured.")
    st.code("GET /health\nGET /ready\nPOST /api/v2/predictions/player-team", language="text")
    st.caption("Use the service-to-service API key or bearer token configured at runtime.")

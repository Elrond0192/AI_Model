"""Authenticated technical operations console for AI_Model."""
from __future__ import annotations

import os
from pathlib import Path
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from basketball_ai.auth.auth import check_credentials, ensure_default_admin
from basketball_ai.data.connection_profiles import load_profiles, profile_url, save_profile
from basketball_ai.data.postgres_loader import get_engine, load_all_data
from basketball_ai.models.promote import get_promotion_status

MODEL_ROOT = Path(os.getenv("MODEL_DIR", "models_saved"))

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
                st.session_state.update(
                    authenticated=True,
                    username=username,
                    role=user.get("role", "viewer"),
                )
                st.rerun()
            st.error("Credenziali non valide.")
    return False


def seasons_for(data) -> list[int]:
    return sorted({int(str(value).split("-")[0]) for value in data["player_stats"]["season"].dropna()})


def registry_status() -> dict:
    try:
        return get_promotion_status(str(MODEL_ROOT))
    except Exception:
        return {"production": None, "candidate": None, "previous": None, "history": []}


def header() -> None:
    profile = st.session_state.get("active_profile", "not selected")
    production = registry_status().get("production") or {}
    active = production.get("run_id", "not promoted")
    st.markdown(
        f'<div class="hm-top"><span>DATABASE<br><strong>{profile}</strong></span>'
        f'<span>API<br><span class="hm-ok">● V2</span></span>'
        f'<span>ACTIVE MODEL<br><strong>{active}</strong></span>'
        f'<span style="margin-left:auto">UTC<br><strong>{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}</strong></span></div>',
        unsafe_allow_html=True,
    )


if not login():
    st.stop()
if st.session_state.get("role") not in {"admin", "analyst"}:
    st.error("Il ruolo corrente non può gestire training e database.")
    st.stop()

with st.sidebar:
    st.markdown("## ◈ AI Model")
    st.caption("CONTROL CENTER")
    page = st.radio(
        "Operations",
        ["Overview", "Data Sources", "Training Runs", "Backtests", "Model Registry", "API & Health"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"{st.session_state.get('username')} · {st.session_state.get('role')}")
    if st.button("Esci", use_container_width=True):
        st.session_state.clear()
        st.rerun()

header()
profiles = load_profiles()
data = st.session_state.get("data")
status = registry_status()

if page == "Overview":
    st.title("Overview")
    production = status.get("production") or {}
    candidate = status.get("candidate") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database", st.session_state.get("active_profile", "Not selected"))
    c2.metric("Rows loaded", f"{len(data['player_stats']):,}" if data is not None else "—")
    c3.metric("Players", f"{len(data['players']):,}" if data is not None else "—")
    c4.metric("Production", production.get("run_id", "Not promoted"))
    st.subheader("Operational readiness")
    ready = data is not None and len(seasons_for(data)) >= 5
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": "PostgreSQL source",
                    "Status": "Ready" if data is not None else "Action required",
                    "Detail": st.session_state.get("active_profile", "Select and load a profile"),
                },
                {
                    "Component": "Historical team context",
                    "Status": "Ready" if data is not None and not data.get("team_season_stats", pd.DataFrame()).empty else "Action required",
                    "Detail": "ai_source.team_season_stats required",
                },
                {
                    "Component": "Training dataset",
                    "Status": "Ready" if ready else "Action required",
                    "Detail": "At least 5 seasons required",
                },
                {
                    "Component": "Production model",
                    "Status": "Ready" if production else "Action required",
                    "Detail": "API loads only promoted artifacts",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Registry")
        rows = []
        if production:
            rows.append({"Run ID": production.get("run_id"), "Status": "Production", "OOT RMSE": production.get("overall_rmse")})
        if candidate:
            rows.append({"Run ID": candidate.get("run_id"), "Status": "Candidate", "OOT RMSE": candidate.get("overall_rmse")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True) if rows else st.info("No model runs registered.")
    with right:
        st.subheader("Candidate quality")
        folds = (candidate.get("backtest") or {}).get("folds", [])
        points = [
            {"target_season": fold.get("target_season"), "rmse": fold.get("rmse")}
            for fold in folds
            if fold.get("valid") and fold.get("rmse") is not None
        ]
        if points:
            st.line_chart(pd.DataFrame(points).set_index("target_season"))
        else:
            st.info("Walk-forward metrics appear after a candidate run.")

elif page == "Data Sources":
    st.title("Data Sources")
    st.caption("PostgreSQL profiles are runtime secrets and are never committed to Git.")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Configured profiles")
        if profiles:
            selected = st.selectbox("Profile", sorted(profiles), key="database_profile")
            profile = profiles[selected]
            st.json({key: ("••••••••" if key == "password" else value) for key, value in profile.items()})
            a, b = st.columns(2)
            if a.button("Test connection", use_container_width=True):
                try:
                    with get_engine(profile_url(selected)).connect() as connection:
                        connection.exec_driver_sql("SELECT 1")
                    st.success("Connection successful.")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
            if b.button("Select and load", type="primary", use_container_width=True):
                try:
                    loaded = load_all_data(
                        profile_url(selected), profile.get("source_schema", "ai_source")
                    )
                    st.session_state.update(data=loaded, active_profile=selected)
                    st.success(
                        f"Loaded {len(loaded['players']):,} players, "
                        f"{len(loaded['player_stats']):,} player seasons and "
                        f"{len(loaded['team_season_stats']):,} team seasons."
                    )
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
                save_profile(
                    name,
                    {
                        "host": host,
                        "port": int(port),
                        "database": database,
                        "user": user,
                        "password": password,
                        "source_schema": source_schema,
                        "ai_schema": ai_schema,
                    },
                )
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
        st.subheader("Leakage-safe training contract")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Stage": "Fit", "Seasons": f"through {seasons[-2]}", "Purpose": "Consecutive t → t+1 only; validation selects tree count"},
                    {"Stage": "Calibration", "Seasons": str(seasons[-1]), "Purpose": "Final ensemble intervals; untouched by fitting"},
                    {"Stage": "Backtest", "Seasons": "expanding OOT folds", "Purpose": "Exact production ensemble vs base and persistence"},
                    {"Stage": "Promotion", "Seasons": "all OOT folds", "Purpose": "RMSE, segments and interval coverage gates"},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Immutable run configuration", expanded=True):
            st.json(
                {
                    "forecast_horizon": "exactly t+1 season",
                    "feature_version": "forecast-t-plus-1-v2",
                    "split": "whole target seasons",
                    "team_context": "historical team-season",
                    "calibration": "final ensemble split-conformal",
                    "backtest": "expanding walk-forward full ensemble",
                    "random_seed": 42,
                }
            )
        if len(seasons) < 5:
            st.error("At least five seasons are required.")
        elif st.button("Start training run", type="primary"):
            try:
                from basketball_ai.models.backtest import run_backtest
                from basketball_ai.models.strict_production import StrictProductionEnsembleModel
                from basketball_ai.models.promote import register_candidate

                with st.status("Training run in progress", expanded=True) as run_status:
                    model = StrictProductionEnsembleModel()
                    metrics = model.train(data)
                    run_status.write("Running full-ensemble walk-forward backtest…")
                    report = run_backtest(
                        data,
                        n_folds=min(3, max(1, len(seasons) - 4)),
                        output_path=str(MODEL_ROOT / "backtest_report.json"),
                    )
                    if not report.get("valid") or not report.get("folds"):
                        raise RuntimeError("Backtest invalid; candidate cannot enter the registry")
                    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
                    run_dir = MODEL_ROOT / "runs" / run_id
                    metadata = {
                        **(metrics or {}),
                        "model_run_id": run_id,
                        "model_version": "2.1.0",
                        "feature_version": "forecast-t-plus-1-v2",
                        "data_cutoff": datetime.now(timezone.utc).date().isoformat(),
                        "latest_observed_season": max(seasons),
                        "database_profile": st.session_state.get("active_profile"),
                        "backtest": report,
                    }
                    model.save(str(run_dir), metadata)
                    register_candidate(str(MODEL_ROOT), run_id, run_dir, metadata)
                    st.session_state.update(candidate_metadata=metadata)
                    run_status.update(label=f"Candidate completed · {run_id}", state="complete")
                st.success("Run saved immutably. Production is unchanged until promotion.")
                st.json(metadata)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

elif page == "Backtests":
    st.title("Backtests")
    candidate = registry_status().get("candidate") or {}
    report = candidate.get("backtest") or (st.session_state.get("candidate_metadata") or {}).get("backtest")
    if report:
        overall = report.get("overall") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble RMSE", f"{overall.get('rmse', 0):.4f}" if overall else "—")
        c2.metric("Base RMSE", f"{report.get('base_rmse', 0):.4f}" if report.get("base_rmse") is not None else "—")
        c3.metric("Persistence RMSE", f"{report.get('persistence_rmse', 0):.4f}" if report.get("persistence_rmse") is not None else "—")
        c4.metric("Interval coverage", f"{overall.get('interval_coverage', 0):.1%}" if overall else "—")
        st.json(report)
    else:
        st.info("Complete a training run to inspect its report.")

elif page == "Model Registry":
    st.title("Model Registry")
    from basketball_ai.models.promote import promote_if_better, rollback_to_previous

    st.json(registry_status())
    st.caption("Promotion activates immutable artifacts in models_saved/production only after all OOT gates pass.")
    confirm = st.checkbox("I reviewed the walk-forward report and promotion gates")
    c1, c2 = st.columns(2)
    if c1.button("Promote candidate", type="primary", disabled=not confirm, use_container_width=True):
        result = promote_if_better(str(MODEL_ROOT))
        st.json(result)
        if result.get("promoted"):
            st.success("Production artifacts activated. Restart/reload the API process to load the new run.")
    if c2.button("Rollback to previous", disabled=not confirm, use_container_width=True):
        result = rollback_to_previous(str(MODEL_ROOT))
        st.json(result)
        if result.get("rolled_back"):
            st.success("Previous artifacts restored. Restart/reload the API process.")

else:
    st.title("API & Health")
    production = registry_status().get("production")
    if production:
        st.success(f"Promoted production run: {production.get('run_id')}")
    else:
        st.warning("No production run is promoted; /health/ready must remain unavailable for inference.")
    st.code(
        "GET /health\nGET /health/live\nGET /health/ready\nPOST /api/v2/predictions/player-team",
        language="text",
    )
    st.caption("The API container loads /app/models_saved/production and never a candidate run.")

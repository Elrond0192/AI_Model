"""Authenticated operations console for PostgreSQL-backed AI_Model."""
from __future__ import annotations
import uuid
from datetime import date, datetime, timezone
import streamlit as st
from basketball_ai.auth.auth import check_credentials, ensure_default_admin
from basketball_ai.data.connection_profiles import load_profiles, profile_url, save_profile
from basketball_ai.data.postgres_loader import get_engine, load_all_data

st.set_page_config(page_title="AI_Model Operations", page_icon="🏀", layout="wide")
ensure_default_admin()

def login() -> bool:
    if st.session_state.get("authenticated"): return True
    st.title("AI_Model Operations")
    st.caption("Console amministrativa per connessioni PostgreSQL, training e promozione.")
    with st.form("login"):
        username = st.text_input("Utente"); password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Accedi", type="primary")
    if submitted:
        ok, user = check_credentials(username, password)
        if ok:
            st.session_state.update(authenticated=True, username=username, role=user.get("role", "viewer")); st.rerun()
        st.error("Credenziali non valide.")
    return False

if not login(): st.stop()
if st.session_state.get("role") not in {"admin", "analyst"}:
    st.error("Il ruolo corrente non può gestire training e database."); st.stop()

st.title("AI_Model Operations")
profiles = load_profiles()
tab_db, tab_train, tab_registry = st.tabs(["Database", "Training", "Registry"])

with tab_db:
    st.subheader("Profili PostgreSQL")
    st.info("Il database gira sull'host. In Docker usa `host.docker.internal`; nessun tunnel SSH è gestito dall'applicazione.")
    with st.form("database_profile_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Nome profilo", placeholder="produzione")
        host = c2.text_input("Host", value="host.docker.internal")
        port = c3.number_input("Porta", min_value=1, max_value=65535, value=5432)
        database = c1.text_input("Database"); user = c2.text_input("Utente PostgreSQL"); password = c3.text_input("Password PostgreSQL", type="password")
        source_schema = c1.text_input("Schema viste sorgente", value="ai_source"); ai_schema = c2.text_input("Schema output", value="ai")
        save = st.form_submit_button("Salva profilo", type="primary")
    if save:
        try:
            save_profile(name, {"host":host,"port":int(port),"database":database,"user":user,"password":password,"source_schema":source_schema,"ai_schema":ai_schema})
            st.success(f"Profilo `{name}` salvato nel volume protetto."); st.rerun()
        except Exception as exc: st.error(str(exc))
    if profiles:
        selected = st.selectbox("Database attivo", sorted(profiles), key="database_profile")
        profile = profiles[selected]
        st.caption(f"{profile.get('host')}:{profile.get('port',5432)}/{profile.get('database')} · sorgente `{profile.get('source_schema','ai_source')}`")
        ctest, cload = st.columns(2)
        if ctest.button("Verifica connessione"):
            try:
                with get_engine(profile_url(selected)).connect() as conn: conn.exec_driver_sql("SELECT 1")
                st.success("Connessione PostgreSQL riuscita.")
            except Exception as exc: st.error(f"Connessione fallita: {exc}")
        if cload.button("Carica dati", type="primary"):
            try:
                data = load_all_data(profile_url(selected), profile.get("source_schema", "ai_source"))
                st.session_state["data"] = data; st.session_state["active_profile"] = selected
                st.success(f"Caricati {len(data['players']):,} giocatori e {len(data['player_stats']):,} righe statistiche.")
            except Exception as exc: st.error(f"Caricamento fallito: {exc}")
    else: st.warning("Crea almeno un profilo database.")

with tab_train:
    st.subheader("Training forecast t → t+1")
    data = st.session_state.get("data")
    if data is None: st.warning("Seleziona e carica un database nella scheda Database.")
    else:
        seasons = sorted({int(str(v).split("-")[0]) for v in data["player_stats"]["season"].dropna()})
        st.write({"database_profile":st.session_state.get("active_profile"), "seasons":seasons, "players":len(data["players"]), "observations":len(data["player_stats"])})
        if len(seasons) < 5: st.error("Servono almeno cinque stagioni per training, validation, calibrazione e backtest walk-forward.")
        elif st.button("Avvia training e backtest", type="primary"):
            try:
                from basketball_ai.models.ensemble import EnsembleModel
                from basketball_ai.models.backtest import run_backtest
                with st.status("Training in corso…", expanded=True) as status:
                    model = EnsembleModel(); metrics = model.train(data)
                    status.write("Backtest walk-forward…")
                    report = run_backtest(data, n_folds=min(3, len(seasons)-3), output_path="models_saved/backtest_report.json")
                    if not report.get("valid") or not report.get("folds"): raise RuntimeError("Backtest non valido: il modello non è promuovibile")
                    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
                    metadata = {**(metrics or {}), "model_run_id":run_id, "model_version":"2.0.0", "feature_version":"forecast-t-plus-1-v1", "data_cutoff":str(date.today()), "database_profile":st.session_state.get("active_profile"), "backtest":report}
                    model.save("models_saved", metadata); st.session_state["ensemble"] = model
                    status.update(label=f"Training completato: {run_id}", state="complete")
                st.json(metadata)
            except Exception as exc: st.error(f"Training fallito: {exc}")

with tab_registry:
    st.subheader("Promotion e rollback")
    from basketball_ai.models.promote import get_promotion_status, promote_if_better, rollback_to_previous
    st.json(get_promotion_status("models_saved"))
    c1, c2 = st.columns(2)
    if c1.button("Promuovi candidate", type="primary"): st.json(promote_if_better("models_saved"))
    if c2.button("Rollback"): st.json(rollback_to_previous("models_saved"))

if st.sidebar.button("Esci"): st.session_state.clear(); st.rerun()

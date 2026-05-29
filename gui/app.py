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

import datetime
import json
import os
import uuid
import sys
import contextlib
import logging
from datetime import timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env into os.environ before any widget rendering.
# sql_loader.py also calls load_dotenv(), but it is imported lazily (inside
# button handlers), so os.environ would be empty when the text_input default
# values are evaluated.  Loading here, with an explicit path, makes it
# work regardless of the current working directory.
try:
    from dotenv import load_dotenv as _load_dotenv_app
    _load_dotenv_app(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _st_components
from basketball_ai.data.loader import _to_int
from basketball_ai.utils.helpers import (
    team_display_name as _team_display_name,
    parse_team_display_map,
    TEAM_DISPLAY_FIELD_MAP_ENV,
)
from basketball_ai.auth.auth import (
    check_credentials,
    create_user,
    delete_user,
    change_password,
    reset_password,
    load_users,
    ensure_default_admin,
    ensure_default_roles,
    load_roles,
    save_roles,
    get_role_sections,
    can_access,
    ALL_SECTIONS,
    SECTION_LABELS,
    DEFAULT_ROLES,
    ADMIN_CREDENTIALS_FILE,
    create_session_token,
    validate_session_token,
    revoke_session_token,
    revoke_user_sessions,
)

# ---------------------------------------------------------------------------
# Session cookie settings
# ---------------------------------------------------------------------------

_SESSION_COOKIE_NAME = "ba_session"
_SESSION_TTL_DAYS    = int(os.environ.get("SESSION_COOKIE_TTL_DAYS", "7"))

# S6: Whether to add the Secure flag to cookies.  Enable by setting the env
# var STREAMLIT_SERVER_COOKIE_SECURE=true (appropriate for HTTPS deployments).
_COOKIE_SECURE = os.environ.get("STREAMLIT_SERVER_COOKIE_SECURE", "").lower() in (
    "true", "1", "yes"
)


def _json_dumps_str(s: str) -> str:
    """Return *s* as a JSON string literal (double-quoted, all chars escaped)."""
    return json.dumps(s)


def _js_set_cookie(name: str, value: str, expires_at: datetime.datetime) -> None:
    """Set a browser cookie with SameSite=Lax (and Secure when configured).

    ``extra_streamlit_components.CookieManager`` does not expose the
    ``SameSite`` or ``Secure`` flags.  This helper injects a tiny JavaScript
    snippet via ``st.components.v1.html`` that re-writes the cookie with the
    proper security attributes.

    The cookie value is set via a JS variable (not embedded in a string
    literal) so any character in *value* is handled safely.

    ``HttpOnly`` cannot be set from JavaScript (browser security model). That
    flag must be enforced at the reverse-proxy layer (nginx ``proxy_cookie_flags``
    or a Set-Cookie header from the backend).
    """
    max_age = max(
        0,
        int((expires_at - datetime.datetime.now(timezone.utc)).total_seconds()),
    )
    secure_attr = "Secure;" if _COOKIE_SECURE else ""
    # Pass the value through a JS variable so that any characters in the
    # token (e.g. newlines, quotes) cannot break out of the string literal.
    js = f"""<script>
(function() {{
  var v = {_json_dumps_str(value)};
  document.cookie = "{name}=" + encodeURIComponent(v) + "; Max-Age={max_age}; Path=/; SameSite=Lax; {secure_attr}";
}})();
</script>"""
    _st_components.html(js, height=0, scrolling=False)


def _js_delete_cookie(name: str) -> None:
    """Expire a browser cookie immediately, respecting the same SameSite flag."""
    secure_attr = "Secure;" if _COOKIE_SECURE else ""
    js = (
        f'<script>'
        f'document.cookie = "{name}=; Max-Age=0; Path=/; SameSite=Lax; {secure_attr}";'
        f'</script>'
    )
    _st_components.html(js, height=0, scrolling=False)

# ---------------------------------------------------------------------------
# Page config – must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Basketball Performance AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Cookie manager – single instance for the entire script
# ---------------------------------------------------------------------------
# Authentication gate – must run before any other Streamlit rendering
# ---------------------------------------------------------------------------

# Ensure at least one admin account exists (first-run bootstrap)
_default_pw = ensure_default_admin()
# Ensure role definitions exist on disk
ensure_default_roles()

if not st.session_state.get("authenticated"):
    # ── Try to restore session from a persistent browser cookie ─────────────
    # st.context.cookies is a read-only dict of all browser cookies (Streamlit ≥ 1.37)
    _saved_token = st.context.cookies.get(_SESSION_COOKIE_NAME, "")
    if _saved_token:
        _session = validate_session_token(_saved_token)
        if _session:
            _tok_user, _tok_role = _session
            st.session_state["authenticated"]   = True
            st.session_state["current_user"]    = _tok_user
            st.session_state["current_role"]    = _tok_role
            st.session_state["_session_token"]  = _saved_token
            st.rerun()

    # ── Show login form ──────────────────────────────────────────────────────
    st.title("🏀 Basketball Performance AI")
    st.subheader("Accesso riservato")

    if _default_pw:
        st.warning(
            f"**Primo avvio**: account `admin` creato.\n\n"
            f"Le credenziali sono state scritte in: `{_default_pw}`\n\n"
            "Leggi il file, accedi e cambia la password. "
            "Poi elimina il file delle credenziali.",
            icon="⚠️",
        )

    with st.form("login_form"):
        _username = st.text_input("Username", autocomplete="username")
        _password = st.text_input("Password", type="password", autocomplete="current-password")
        _login_btn = st.form_submit_button("🔐 Accedi", type="primary", width="stretch")

    if _login_btn:
        _ok, _user = check_credentials(_username, _password)
        if _ok:
            _uname  = _username.strip().lower()
            _role   = _user.get("role", "viewer")
            _token  = create_session_token(_uname, _role, _SESSION_TTL_DAYS)
            _expiry = datetime.datetime.now(timezone.utc) + datetime.timedelta(days=_SESSION_TTL_DAYS)
            # S6: set cookie with SameSite=Lax (+ Secure when configured) via JS.
            # HttpOnly cannot be set from JS; enforce it at the reverse proxy.
            _js_set_cookie(_SESSION_COOKIE_NAME, _token, _expiry)
            st.session_state["authenticated"]   = True
            st.session_state["current_user"]    = _uname
            st.session_state["current_role"]    = _role
            st.session_state["_session_token"]  = _token
            st.rerun()
        else:
            if _user.get("locked_until"):
                st.error("Account temporaneamente bloccato per troppi tentativi. Riprova più tardi.")
            else:
                st.error("Credenziali non valide. Riprova.")

    st.stop()


# ---------------------------------------------------------------------------
# Role-based access control helper
# ---------------------------------------------------------------------------

def _can_access(section: str) -> bool:
    """Return True when the logged-in user's role is permitted to access *section*."""
    role = st.session_state.get("current_role", "viewer")
    return can_access(role, section)


# ---------------------------------------------------------------------------
# Error boundary – U10
# ---------------------------------------------------------------------------

_tab_logger = logging.getLogger("gui.tabs")


@contextlib.contextmanager
def _safe_tab(tab, tab_name: str):
    """Context manager that enters a Streamlit tab and catches unhandled exceptions.

    Usage::

        with _safe_tab(tab_data, "Dati"):
            # tab content here

    If the body raises an unexpected exception the user sees a friendly error
    message instead of a raw traceback, and the exception is logged.
    """
    with tab:
        try:
            yield
        except Exception as _tab_exc:
            # `except Exception` intentionally does NOT catch BaseException subclasses
            # like KeyboardInterrupt, SystemExit, or GeneratorExit, which must propagate.
            _tab_logger.exception("Unexpected error in tab '%s'", tab_name)
            st.error(
                f"⚠️ Errore inatteso nel tab **{tab_name}**: `{_tab_exc}`\n\n"
                "Controlla i dati caricati o contatta l'amministratore."
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
    """Return a validated absolute path that stays within the project root (P7)."""
    project_root = Path(__file__).parent.parent.resolve()
    try:
        resolved = Path((user_input or "models_saved").strip()).resolve()
    except (TypeError, ValueError):
        return project_root / "models_saved"
    if not resolved.is_relative_to(project_root):
        return project_root / "models_saved"
    return resolved


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


def _get_team_league_map(data: dict) -> dict:
    """Return a ``{team_id: league_code}`` mapping from loaded data.

    The *league_code* is the string identifier of the league as stored in
    ``leagues.id`` (e.g. ``"ITA1"``, ``"GRC1"`` for SQL data, or the numeric
    id as a string for generated CSV data).  This is used as the primary key
    in the team display-name mapping, taking priority over the country code.
    """
    try:
        leagues_df = data.get("leagues")
        teams_df   = data.get("teams")
        if leagues_df is None or teams_df is None:
            return {}
        if "id" not in teams_df.columns or "league_id" not in teams_df.columns:
            return {}
        # Build {internal_league_id → league_code_string}
        league_code: dict = {}
        if leagues_df is not None and "id" in leagues_df.columns:
            for _, lr in leagues_df.iterrows():
                raw_id = lr["id"]
                lid_int = _to_int(raw_id)
                code = str(raw_id).strip().upper()
                league_code[lid_int] = code
        result = {}
        for _, row in teams_df.iterrows():
            tid  = _to_int(row["id"])
            lid  = _to_int(row.get("league_id"))
            code = league_code.get(lid, str(row.get("league_id", "") or "").strip().upper())
            if code and code not in ("NAN", "NONE", ""):
                result[tid] = code
        return result
    except Exception:
        return {}


def _make_team_display_name(team_row, team_country_map: dict, display_map: dict,
                             team_league_map: dict | None = None) -> str:
    """Wrapper around :func:`_team_display_name` that injects *country* and *league_id*.

    *league_id* (from *team_league_map*) has higher priority than *country*
    in the display-map lookup, allowing per-league name configuration.
    """
    team_id  = _to_int(team_row.get("id", 0))
    country  = team_country_map.get(team_id, "")
    league_id = (team_league_map or {}).get(team_id, "")
    return _team_display_name(team_row, country=country, league_id=league_id,
                              display_map=display_map)


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



# ---------------------------------------------------------------------------
# Sidebar – user info and logout
# ---------------------------------------------------------------------------
with st.sidebar:
    _cur_user = st.session_state.get("current_user", "")
    _cur_role = st.session_state.get("current_role", "")
    st.caption(f"👤 **{_cur_user}** ({_cur_role})")
    if st.button("🚪 Logout", width="stretch"):
        _tok = st.session_state.get("_session_token")
        if _tok:
            revoke_session_token(_tok)
        _js_delete_cookie(_SESSION_COOKIE_NAME)  # S6: clear with SameSite=Lax flag
        st.session_state.clear()
        st.rerun()

st.title("🏀 Basketball Performance AI")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_train, tab_pred, tab_scen, tab_chat, tab_scout, tab_mapping, tab_admin, tab_audit, tab_health, tab_drift = st.tabs([
    "📂 Dati",
    "🏋️ Training",
    "🎯 Predizioni",
    "🔀 Scenari",
    "💬 Chat",
    "🔬 Scouting AI",
    "🗺️ Mapping",
    "👥 Utenti",
    "📋 Audit",
    "🏥 Health",
    "📊 Drift",
])

# ===========================================================================
# TAB 1 – DATI
# ===========================================================================

with _safe_tab(tab_data, "Dati"):
    if not _can_access("data"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
        # U4: Empty state / onboarding for new users
        if not st.session_state.get("model_loaded") and not st.session_state.get("data_loaded"):
            st.info(
                "👋 **Benvenuto in Basketball AI!**\n\n"
                "Per iniziare, segui questi passaggi:\n\n"
                "1. **📂 Dati** ← sei qui – carica i file CSV o configura la connessione Azure SQL\n"
                "2. **🏋️ Training** – allena il modello sui dati caricati\n"
                "3. **🎯 Predizioni** – ottieni predizioni per i giocatori\n\n"
                "Se hai già dati pronti, carica i file CSV qui sotto.",
                icon="ℹ️",
            )

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
                autocomplete="off",
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

with _safe_tab(tab_train, "Training"):
    if not _can_access("training"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
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

        # -----------------------------------------------------------------------
        # A.5 – Quick standalone prediction (no data connection needed)
        # -----------------------------------------------------------------------
        _ens_loaded = st.session_state.get("ensemble")
        if _ens_loaded is not None and _ens_loaded.is_trained:
            st.divider()
            st.subheader("🔮 A.5 · Predizione rapida (senza dati)")
            st.caption(
                "Inserisci le statistiche del giocatore per ottenere una predizione "
                "senza bisogno di collegarti al database o caricare file CSV."
            )
            with st.form("quick_pred_form"):
                _qp_c1, _qp_c2, _qp_c3 = st.columns(3)

                with _qp_c1:
                    st.markdown("**Anagrafica**")
                    _qp_age  = st.number_input("Età", min_value=16, max_value=45, value=25, key="qp_age")
                    _qp_pos  = st.selectbox("Posizione", ["PG", "SG", "SF", "PF", "C"], key="qp_pos")
                    _qp_comp = st.selectbox("Contesto", ["RS", "PO", "CUP"], key="qp_comp")

                    st.markdown("**Statistiche per 36 min**")
                    _qp_pts  = st.number_input("Punti/36",   0.0, 60.0, value=15.0, step=0.5, key="qp_pts")
                    _qp_ast  = st.number_input("Assist/36",  0.0, 25.0, value=4.0,  step=0.5, key="qp_ast")
                    _qp_reb  = st.number_input("Rimbalzi/36",0.0, 25.0, value=5.0,  step=0.5, key="qp_reb")
                    _qp_stl  = st.number_input("Rubate/36",  0.0,  8.0, value=1.0,  step=0.1, key="qp_stl")
                    _qp_blk  = st.number_input("Stoppate/36",0.0,  8.0, value=0.5,  step=0.1, key="qp_blk")

                with _qp_c2:
                    st.markdown("**Efficienza avanzata**")
                    _qp_ts   = st.number_input("TS%",    0.0, 1.0, value=0.55, step=0.01, key="qp_ts",
                                               help="True Shooting % — tipico: 0.50–0.65")
                    _qp_usg  = st.number_input("USG%",   0.0, 50.0, value=20.0, step=0.5, key="qp_usg",
                                               help="Usage Rate — tipico: 15–35")
                    _qp_obpm = st.number_input("OBPM",  -10.0, 15.0, value=0.0, step=0.1, key="qp_obpm",
                                               help="Offensive Box Plus/Minus")
                    _qp_dbpm = st.number_input("DBPM",  -10.0, 10.0, value=0.0, step=0.1, key="qp_dbpm",
                                               help="Defensive Box Plus/Minus")
                    _qp_per  = st.number_input("PIE",    0.0,  0.4, value=0.12, step=0.01, key="qp_per",
                                               help="Player Impact Estimate (0–0.40, media ≈ 0.10)")

                with _qp_c3:
                    st.markdown("**Segnali di carriera**")
                    _qp_form = st.slider("Form recente (0–10)", 0.0, 10.0, value=6.0, step=0.1, key="qp_form",
                                         help="Prestazioni recenti vs media storica")
                    _qp_dur  = st.slider("Durabilità (0–1)", 0.0, 1.0, value=0.75, step=0.05, key="qp_dur",
                                         help="Partite giocate / partite disponibili")
                    _qp_traj = st.slider("Traiettoria carriera", -2.0, 2.0, value=0.0, step=0.1, key="qp_traj",
                                         help="Tendenza recente: >0 in crescita, <0 in calo")
                    st.markdown("**On/Off impact**")
                    _qp_netdiff = st.number_input("Net Rtg Diff", -15.0, 15.0, value=0.0, step=0.5, key="qp_netdiff",
                                                   help="Differenziale on-court vs off-court")
                    _qp_starter = st.slider("Starter %", 0.0, 1.0, value=0.5, step=0.05, key="qp_starter",
                                             help="Frazione partite giocate da titolare")

                _qp_submit = st.form_submit_button("🎯 Calcola predizione", type="primary")

            if _qp_submit:
                try:
                    from basketball_ai.models.performance_model import COMPETITION_ENCODING as _COMP_ENC
                    _pos_one_hot = {
                        "pos_PG": 0.0, "pos_SG": 0.0, "pos_SF": 0.0,
                        "pos_PF": 0.0, "pos_C": 0.0, "pos_hybrid": 0.0,
                    }
                    _pos_key = f"pos_{_qp_pos}"
                    if _pos_key in _pos_one_hot:
                        _pos_one_hot[_pos_key] = 1.0
                    else:
                        _pos_one_hot["pos_hybrid"] = 1.0

                    _qp_features = {
                        "age":                  float(_qp_age),
                        **_pos_one_hot,
                        "competition_enc":      float(_COMP_ENC.get(_qp_comp, 0)),
                        "role_enc":             0.0,
                        "role_off_enc":         0.0,
                        "role_def_enc":         0.0,
                        "pts_per_36":           float(_qp_pts),
                        "ast_per_36":           float(_qp_ast),
                        "reb_per_36":           float(_qp_reb),
                        "stl_per_36":           float(_qp_stl),
                        "blk_per_36":           float(_qp_blk),
                        "avg_per":              float(_qp_per),
                        "avg_ts_pct":           float(_qp_ts),
                        "avg_usg_pct":          float(_qp_usg),
                        "avg_obpm":             float(_qp_obpm),
                        "avg_dbpm":             float(_qp_dbpm),
                        "form_score":           float(_qp_form),
                        "consistency_score":    float(_qp_form) * 0.8,
                        "career_trajectory":    float(_qp_traj),
                        "age_vs_peak_age":      float(_qp_age) - 27.0,
                        "ts_efficiency_trend":  0.0,
                        "durability_score":     float(_qp_dur),
                        "po_vs_rs_delta":       0.0,
                        "po_games_played":      0.0,
                        "has_po_history":       0.0,
                        "avg_net_rtg_diff":     float(_qp_netdiff),
                        "avg_ortg_diff":        float(_qp_netdiff) * 0.6,
                        "avg_starter_pct":      float(_qp_starter),
                        # Interaction features computed from entered values
                        "obpm_x_usg":           float(_qp_obpm) * float(_qp_usg),
                        "dbpm_x_reb":           float(_qp_dbpm) * float(_qp_reb) / 10.0,
                        "two_way_score":        0.0,
                        # Remaining features not entered: default 0
                    }

                    _qp_rating = _ens_loaded.perf_model.predict_from_features(_qp_features)
                    _qp_ci_lo  = round(max(3.5, _qp_rating - 0.5), 2)
                    _qp_ci_hi  = round(min(10.0, _qp_rating + 0.5), 2)

                    _qpc1, _qpc2, _qpc3 = st.columns(3)
                    _qpc1.metric("⭐ Rating previsto", f"{_qp_rating:.2f} / 10")
                    _qpc2.metric("CI basso",  f"{_qp_ci_lo}")
                    _qpc3.metric("CI alto",   f"{_qp_ci_hi}")
                    st.caption(
                        "⚠️ Predizione semplificata: feature avanzate (RAPTOR, LEBRON, SPM, clutch, "
                        "on/off) impostate a 0. Per una predizione completa carica i dati nel tab 📂 Dati."
                    )
                except Exception as _qp_exc:
                    st.error(f"Errore nella predizione: {_qp_exc}")

        st.divider()

        # -----------------------------------------------------------------------
        # B – Train new model
        # -----------------------------------------------------------------------
        st.subheader("B · Addestra nuovo modello")

        with st.expander("ℹ️ Guida agli iperparametri", expanded=False):
            st.markdown("""
| Parametro | Descrizione | Range consigliato |
|---|---|---|
| `n_estimators` | Numero di alberi XGBoost. Più è alto, più il modello è preciso ma lento. | 200–800 |
| `max_depth` | Profondità massima di ogni albero. Valori alti = rischio overfitting. | 4–7 |
| `learning_rate` | Velocità di apprendimento (eta). Bilanciare con n_estimators: più basso = più alberi necessari. | 0.005–0.1 |
| `subsample` | Frazione dei campioni usata per ogni albero. Riduce overfitting. | 0.7–0.9 |
| `colsample_bytree` | Frazione delle feature usata per ogni albero. Riduce la correlazione tra alberi. | 0.6–0.9 |
| `min_child_weight` | Peso minimo dei campioni in una foglia. Aumentare se il dataset è piccolo. | 2–6 |
| `reg_alpha (L1)` | Regolarizzazione L1: annulla feature poco rilevanti. | 0.0–0.5 |
| `reg_lambda (L2)` | Regolarizzazione L2: riduce smoothly i pesi. | 0.5–2.0 |
| `early_stopping_rounds` | Ferma il training se la val-loss non migliora per N rounds consecutivi. | 20–100 |
| `Validation split` | Frazione dei dati usata per la validazione (non per il training). | 0.10–0.20 |
| `CV folds` | K-fold cross-validation per valutare la stabilità del modello. 0 = disabilitata. | 0, 3, 5, 10 |
| `Random seed` | Seme per la riproducibilità dei risultati. | Qualsiasi intero |

Il modello usa **{n_features} feature** tra cui:
- Ruoli DB: `ruolo_combinato`, `ruolo_offensivo`, `ruolo_difensivo` (3 dimensioni separate)
- RAPTOR offensivo/difensivo, LEBRON off/def, OWS/DWS, FIC
- Feature di interazione: `OBPM × USG%`, `DBPM × REB%`, `two_way_score`
- Trend di efficienza (TS%), durabilità, segnali di carriera
            """.format(n_features=len(__import__('basketball_ai.models.performance_model', fromlist=['FEATURE_COLS']).FEATURE_COLS)))

        # -----------------------------------------------------------------------
        # Preset buttons
        # -----------------------------------------------------------------------
        _PRESETS = {
            "🏆 Preciso": dict(
                n_estimators=1000, max_depth=6, learning_rate=0.01,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
                reg_alpha=0.1, reg_lambda=1.5, test_size=0.15, seed=42,
                early_stopping=50, cv_folds=5,
                label="Alta precisione – addestramento lento (~5–15 min), ottimi risultati",
            ),
            "⚖️ Bilanciato": dict(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, test_size=0.15, seed=42,
                early_stopping=30, cv_folds=0,
                label="Bilanciato – addestramento medio (~2–4 min), buoni risultati (default)",
            ),
            "⚡ Veloce": dict(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                subsample=0.70, colsample_bytree=0.70, min_child_weight=4,
                reg_alpha=0.0, reg_lambda=0.5, test_size=0.20, seed=42,
                early_stopping=20, cv_folds=0,
                label="Veloce – addestramento rapido (<30 s), precisione ridotta",
            ),
        }
        st.markdown("**Preset rapidi**")
        _pcols = st.columns(3)
        for _pi, (_pname, _pvals) in enumerate(_PRESETS.items()):
            with _pcols[_pi]:
                if st.button(_pname, key=f"preset_{_pi}", width="stretch",
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
                    st.session_state["t_early_stop"] = _pvals["early_stopping"]
                    st.session_state["t_cv_folds"]   = _pvals["cv_folds"]
                    st.rerun()

        # Initialise slider keys only on the very first render; presets and
        # manual moves update session_state directly, so setdefault is a no-op
        # on subsequent reruns — avoiding the "default value + session_state"
        # conflict warning.
        _slider_defaults = {
            "t_nest": 300, "t_depth": 5, "t_lr": 0.05,
            "t_sub": 0.8, "t_colsample": 0.8, "t_mcw": 3,
            "t_alpha": 0.1, "t_lambda": 1.0, "t_early_stop": 30,
            "t_cv_folds": 0, "t_seed": 42, "t_split": 0.15,
        }
        for _sk, _sv in _slider_defaults.items():
            st.session_state.setdefault(_sk, _sv)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**XGBoost**")
            xgb_n_estimators = st.slider(
                "n_estimators", 50, 1000, step=50, key="t_nest",
                help="Numero di alberi. Più è alto, più il modello è preciso ma lento. Range consigliato: 200–500.",
            )
            xgb_max_depth = st.slider(
                "max_depth", 2, 12, key="t_depth",
                help="Profondità massima di ogni albero. Valori alti aumentano il rischio di overfitting. Range consigliato: 4–7.",
            )
            xgb_lr = st.slider(
                "learning_rate", 0.005, 0.3, step=0.005, format="%.3f", key="t_lr",
                help="Velocità di apprendimento (eta). Valori bassi richiedono più alberi. Range consigliato: 0.01–0.1.",
            )
            xgb_subsample = st.slider(
                "subsample", 0.4, 1.0, step=0.05, key="t_sub",
                help="Frazione dei campioni usata per addestrare ogni albero. Riduce overfitting. Range consigliato: 0.7–0.9.",
            )
            xgb_colsample = st.slider(
                "colsample_bytree", 0.3, 1.0, step=0.05, key="t_colsample",
                help="Frazione delle feature usata per ogni albero. Riduce la correlazione tra alberi. Range consigliato: 0.6–0.9.",
            )
            xgb_min_child_weight = st.slider(
                "min_child_weight", 1, 10, key="t_mcw",
                help="Peso minimo dei campioni in una foglia. Aumentare se il dataset è piccolo. Range consigliato: 2–6.",
            )
        with col2:
            st.markdown("**Regolarizzazione**")
            xgb_reg_alpha = st.slider(
                "reg_alpha (L1)", 0.0, 1.0, step=0.05, key="t_alpha",
                help="Regolarizzazione L1: annulla feature poco rilevanti. Range consigliato: 0.0–0.5.",
            )
            xgb_reg_lambda = st.slider(
                "reg_lambda (L2)", 0.0, 3.0, step=0.1, key="t_lambda",
                help="Regolarizzazione L2: riduce smoothly i pesi. Range consigliato: 0.5–2.0.",
            )
            st.markdown("**Early Stopping & Cross-Validation**")
            early_stopping_rounds = st.slider(
                "early_stopping_rounds", 0, 200, step=10, key="t_early_stop",
                help="Ferma il training se la val-loss non migliora per N rounds. 0 = disabilitato.",
            )
            cv_folds_gui = st.select_slider(
                "CV folds (k-fold)", options=[0, 3, 5, 10], key="t_cv_folds",
                help="K-fold cross-validation per stimare la stabilità del modello. 0 = disabilitata.",
            )
            st.markdown("**Generale**")
            seed = st.number_input(
                "Random seed", min_value=0, max_value=99999, key="t_seed",
                help="Seme per la riproducibilità dei risultati. Qualsiasi intero va bene.",
            )
            test_size = st.slider(
                "Validation split", 0.05, 0.40, step=0.05, key="t_split",
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
                        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds > 0 else None,
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

                    metrics_dict = {
                        "train_rmse": train_rmse,
                        "val_rmse":   val_rmse,
                        "val_mae":    val_mae,
                        "val_r2":     val_r2,
                    }

                    # Optional k-fold cross-validation
                    if cv_folds_gui > 1:
                        log(f"Avvio {cv_folds_gui}-fold cross-validation …")
                        progress.progress(55, text=f"{cv_folds_gui}-fold cross-validation …")
                        from sklearn.model_selection import KFold as _KFold
                        kf = _KFold(n_splits=cv_folds_gui, shuffle=True, random_state=int(seed))
                        X_np = X.values.astype(float)
                        cv_rmses_gui = []
                        cv_r2s_gui   = []
                        for fold_idx, (tr_i, va_i) in enumerate(_kf_iter := kf.split(X_np), 1):
                            _Xtr, _Xva = X_np[tr_i], X_np[va_i]
                            _ytr, _yva = y_full[tr_i], y_full[va_i]
                            _sc2 = StandardScaler()
                            _Xtr_sc = _sc2.fit_transform(_Xtr)
                            _Xva_sc = _sc2.transform(_Xva)
                            _m2 = XGBRegressor(
                                n_estimators=xgb_n_estimators,
                                max_depth=xgb_max_depth, learning_rate=xgb_lr,
                                subsample=xgb_subsample, colsample_bytree=xgb_colsample,
                                min_child_weight=xgb_min_child_weight,
                                reg_alpha=xgb_reg_alpha, reg_lambda=xgb_reg_lambda,
                                random_state=int(seed), verbosity=0,
                            )
                            _m2.fit(_Xtr_sc, _ytr, verbose=False)
                            _ypred = _m2.predict(_Xva_sc)
                            fold_rmse = float(np.sqrt(np.mean((_ypred - _yva) ** 2)))
                            fold_r2   = float(skm.r2_score(_yva, _ypred))
                            cv_rmses_gui.append(fold_rmse)
                            cv_r2s_gui.append(fold_r2)
                            log(f"  Fold {fold_idx}/{cv_folds_gui}  RMSE={fold_rmse:.4f}  R²={fold_r2:.4f}")
                        cv_mean = float(np.mean(cv_rmses_gui))
                        cv_std  = float(np.std(cv_rmses_gui))
                        cv_r2_mean = float(np.mean(cv_r2s_gui))
                        log(f"CV  RMSE={cv_mean:.4f} ±{cv_std:.4f}   R²={cv_r2_mean:.4f}")
                        metrics_dict.update({
                            "cv_mean_rmse": cv_mean,
                            "cv_std_rmse":  cv_std,
                            "cv_mean_r2":   cv_r2_mean,
                        })

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
            _n_cv_cols = 6 if "cv_mean_rmse" in m else 4
            c1, c2, c3, c4, *_rest = st.columns(_n_cv_cols)
            c1.metric("Train RMSE", f"{m['train_rmse']:.4f}")
            c2.metric("Val RMSE",   f"{m['val_rmse']:.4f}")
            c3.metric("Val MAE",    f"{m['val_mae']:.4f}")
            c4.metric("Val R²",     f"{m['val_r2']:.4f}")
            if "cv_mean_rmse" in m and _rest:
                _rest[0].metric("CV RMSE",    f"{m['cv_mean_rmse']:.4f} ±{m['cv_std_rmse']:.4f}")
                _rest[1].metric("CV R²",      f"{m['cv_mean_r2']:.4f}")

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

with _safe_tab(tab_pred, "Predizioni"):
    st.header("🎯 Predizioni")
    if not _can_access("predictions"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
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
                _team_league_p   = _get_team_league_map(data_p)
                team_opts_p = {
                    _to_int(r["id"]): _make_team_display_name(r, _team_country_p, _display_map_p,
                                                               _team_league_p)
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
                        # U12: trajectory and style-fit charts available in gui.components.charts

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
                                    width="stretch",
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

with _safe_tab(tab_scen, "Scenari"):
    st.header("🔀 Scenari")
    if not _can_access("scenarios"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
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
                _team_league_s  = _get_team_league_map(data_s)
                team_opts_s = {
                    _to_int(r["id"]): _make_team_display_name(r, _team_country_s, _display_map_s,
                                                               _team_league_s)
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

with _safe_tab(tab_chat, "Chat"):
    st.header("💬 Assistente AI")
    if not _can_access("chat"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
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


    # ===========================================================================
# TAB 6 – SCOUTING AI
# ===========================================================================

with _safe_tab(tab_scout, "Scouting AI"):
    st.header("🔬 Scouting AI — Intelligence Platform")
    if not _can_access("scouting"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
        st.caption(
            "Genera report di scouting automatici, trova i giocatori più simili nel database, "
            "analizza il DNA prestativo e individua il miglior contesto di squadra."
        )

        data_sc = _require_data()
        if data_sc is not None:
            engine_sc = _require_engine(data_sc)
            if engine_sc is not None:

                # ------------------------------------------------------------------
                # Player selector
                # ------------------------------------------------------------------
                _sc_players = {
                    _to_int(r["id"]): str(r["name"])
                    for _, r in data_sc["players"].iterrows()
                }
                sc_col1, sc_col2 = st.columns([2, 1])
                with sc_col1:
                    _sc_srch = st.text_input(
                        "🔍 Cerca giocatore",
                        key="_s_scout_player",
                        placeholder="Nome giocatore…",
                        label_visibility="collapsed",
                    )
                _sc_filtered = (
                    {k: v for k, v in _sc_players.items()
                     if _sc_srch.strip().lower() in v.lower()}
                    if _sc_srch.strip() else _sc_players
                )
                if not _sc_filtered:
                    _sc_filtered = _sc_players
                scout_player_id = sc_col1.selectbox(
                    "Giocatore da analizzare",
                    list(_sc_filtered.keys()),
                    format_func=lambda x: _sc_filtered.get(x, str(x)),
                    key="scout_player",
                )
                with sc_col2:
                    _sc_season = st.number_input(
                        "Stagione", min_value=2000, max_value=2040,
                        value=2024, key="scout_season",
                    )

                if st.button("🔬 Genera Report Scouting", type="primary", key="scout_run"):
                    with st.spinner("Analisi AI in corso …"):
                        try:
                            import numpy as np

                            # --- Fetch player data --------------------------------
                            _sc_player_row = data_sc["player_dict"].get(scout_player_id, {})
                            _sc_name       = _sc_filtered.get(scout_player_id, f"Giocatore {scout_player_id}")
                            _sc_age        = _to_int(_sc_player_row.get("age", 25))
                            _sc_pos        = str(_sc_player_row.get("position", "—"))
                            _sc_nat        = str(_sc_player_row.get("nationality", "—"))
                            _sc_height     = _to_int(_sc_player_row.get("height_cm", 0))
                            _sc_weight     = _to_int(_sc_player_row.get("weight_kg", 0))

                            _sc_stats_df   = data_sc["player_stats"]
                            # Normalize both sides to a clean integer string before
                            # comparing: player_id may be str "123", int 123, or
                            # float 123.0 depending on the load path and DB column type.
                            _sc_pid_str    = str(int(scout_player_id))
                            _sc_mask       = (
                                _sc_stats_df["player_id"]
                                .astype(str)
                                .str.replace(r"\.0$", "", regex=True)
                                == _sc_pid_str
                            )
                            _sc_p_stats    = _sc_stats_df[_sc_mask].sort_values("season")
                            _sc_latest     = _sc_p_stats.iloc[-1].to_dict() if not _sc_p_stats.empty else {}
                            # Prefer the latest Regular Season row for DNA and strengths
                            # (PO/CUP rows may have fewer games and outlier stats).
                            if (
                                not _sc_p_stats.empty
                                and "competition" in _sc_p_stats.columns
                            ):
                                _rs_rows = _sc_p_stats[_sc_p_stats["competition"] == "RS"]
                                _sc_latest_rs = _rs_rows.iloc[-1].to_dict() if not _rs_rows.empty else _sc_latest
                            else:
                                _sc_latest_rs = _sc_latest

                            # --- Predict current performance ----------------------
                            _sc_cur_team = _sc_player_row.get("current_team_id")
                            if _sc_cur_team is None and not _sc_p_stats.empty:
                                _sc_cur_team = _sc_p_stats.iloc[-1].get("team_id")
                            if _sc_cur_team is None:
                                _sc_cur_team = data_sc["teams"].iloc[0]["id"]

                            _sc_pred = engine_sc.predict_in_team(
                                scout_player_id, _to_int(_sc_cur_team), season=int(_sc_season)
                            )

                            # --- Career trajectory --------------------------------
                            _sc_traj = engine_sc.predict_age_trajectory(
                                scout_player_id,
                                age_range=(max(18, _sc_age - 2), min(40, _sc_age + 8)),
                                team_id=_to_int(_sc_cur_team),
                                season_base=int(_sc_season),
                            )
                            _sc_peak = max(_sc_traj, key=lambda p: p.predicted_rating)

                            # --- Best team fits -----------------------------------
                            _sc_best_teams = engine_sc.best_team_fit(
                                scout_player_id, top_n=5, season=int(_sc_season)
                            )

                            # --- Player comparables (cosine similarity) -----------
                            _COMP_COLS = [
                                "points", "rebounds", "assists", "steals", "blocks",
                                "fg_pct", "three_point_pct", "usg_pct", "bpm", "vorp",
                                "per", "ts_pct", "plus_minus",
                            ]
                            _avail_cols = [c for c in _COMP_COLS if c in _sc_stats_df.columns]
                            _sc_comparables = []
                            if _avail_cols and not _sc_p_stats.empty:
                                _sc_vec = np.array([
                                    float(_sc_latest.get(c, 0) or 0) for c in _avail_cols
                                ])
                                _sc_vec_norm = np.linalg.norm(_sc_vec)

                                # Build per-player average stat vector
                                _sc_grouped = (
                                    _sc_stats_df.groupby("player_id")[_avail_cols]
                                    .mean()
                                    .fillna(0)
                                )
                                _comp_rows = []
                                for _cpid, _crow in _sc_grouped.iterrows():
                                    if _cpid == scout_player_id:
                                        continue
                                    _cv = np.array([float(_crow.get(c, 0) or 0)
                                                    for c in _avail_cols])
                                    _cn = np.linalg.norm(_cv)
                                    if _sc_vec_norm < 1e-9 or _cn < 1e-9:
                                        continue
                                    _sim = float(np.dot(_sc_vec, _cv) / (_sc_vec_norm * _cn))
                                    _comp_rows.append((_cpid, _sim))

                                _comp_rows.sort(key=lambda x: x[1], reverse=True)
                                _player_names = {
                                    _to_int(r["id"]): str(r["name"])
                                    for _, r in data_sc["players"].iterrows()
                                }
                                _sc_comparables = [
                                    {
                                        "Giocatore comparabile": _player_names.get(_to_int(_cpid), f"#{_cpid}"),
                                        "Similarità": round(_sim * 100, 1),
                                    }
                                    for _cpid, _sim in _comp_rows[:5]
                                ]

                            # Build per-competition stats dict for the detailed view
                            _sc_stats_by_comp: dict = {}
                            if not _sc_p_stats.empty:
                                if "competition" in _sc_p_stats.columns:
                                    for _comp_key in _sc_p_stats["competition"].dropna().unique():
                                        _comp_rows_df = _sc_p_stats[_sc_p_stats["competition"] == _comp_key]
                                        if not _comp_rows_df.empty:
                                            _sc_stats_by_comp[str(_comp_key)] = _comp_rows_df.iloc[-1].to_dict()
                                else:
                                    _sc_stats_by_comp["RS"] = _sc_latest

                            # Store results in session state for display
                            st.session_state["scout_result"] = {
                                "name":            _sc_name,
                                "age":             _sc_age,
                                "pos":             _sc_pos,
                                "nat":             _sc_nat,
                                "height":          _sc_height,
                                "weight":          _sc_weight,
                                "pred":            _sc_pred,
                                "traj":            _sc_traj,
                                "peak":            _sc_peak,
                                "best_teams":      _sc_best_teams,
                                "comparables":     _sc_comparables,
                                "latest_stats":    _sc_latest,
                                "latest_stats_rs": _sc_latest_rs,
                                "stats_by_comp":   _sc_stats_by_comp,
                                "avail_cols":      _avail_cols,
                            }
                        except Exception as exc:
                            st.error(f"Errore durante l'analisi: {exc}")

                # ------------------------------------------------------------------
                # Display results
                # ------------------------------------------------------------------
                if "scout_result" in st.session_state:
                    _sr = st.session_state["scout_result"]
                    _pred = _sr["pred"]

                    st.divider()

                    # Header card
                    st.subheader(f"🏀 {_sr['name']}")
                    _h1, _h2, _h3, _h4, _h5 = st.columns(5)
                    _h1.metric("Età",       _sr["age"])
                    _h2.metric("Posizione", _sr["pos"])
                    _h3.metric("Nazionalità", _sr["nat"])
                    _h4.metric("Altezza",   f"{_sr['height']} cm" if _sr["height"] else "—")
                    _h5.metric("Peso",      f"{_sr['weight']} kg" if _sr["weight"] else "—")

                    st.divider()

                    # --- AI Scouting Narrative ---
                    st.subheader("📋 Report di Scouting AI")

                    _rating     = _pred.predicted_rating
                    _peak_r     = _sr["peak"].predicted_rating
                    _peak_age   = _sr["peak"].age
                    _seasons_to = max(0, _peak_age - _sr["age"])

                    # Use RS (Regular Season) stats for strengths/DNA when available
                    _ls_rs_available = bool(_sr.get("latest_stats_rs"))
                    _ls = _sr["latest_stats_rs"] if _ls_rs_available else _sr["latest_stats"]

                    # Minutes context for tier label
                    _ls_mpg = float(_ls.get("minutes_per_game", 0) or 0)
                    _ls_gp  = float(_ls.get("games_played", 0) or 0)
                    _mpg_ctx = (
                        f" ({_ls_mpg:.0f} min/g, {_ls_gp:.0f} partite)"
                        if _ls_mpg > 0 and _ls_gp > 0 else ""
                    )

                    # Rating tier label – adjusted thresholds reflect post-mpg-factor ratings
                    if _rating >= 8.0:
                        _tier = "élite assoluta (top mondiale)"
                    elif _rating >= 7.0:
                        _tier = "giocatore di alto livello"
                    elif _rating >= 6.0:
                        _tier = "giocatore competitivo di buon livello"
                    elif _rating >= 5.0:
                        _tier = "role player / titolare di buon livello"
                    elif _rating >= 4.0:
                        _tier = "giocatore di rotazione"
                    else:
                        _tier = "riserva / potenziale di sviluppo"

                    # Trend from trajectory
                    _traj = _sr["traj"]
                    if len(_traj) >= 3:
                        _recent = [p.predicted_rating for p in _traj[:3]]
                        _trend_str = (
                            "in crescita 📈" if _recent[-1] > _recent[0] + 0.1
                            else "in calo 📉" if _recent[-1] < _recent[0] - 0.1
                            else "stabile ➡️"
                        )
                    else:
                        _trend_str = "stabile ➡️"

                    # Best team info
                    _best_team_name = (
                        _sr["best_teams"][0].team_name if _sr["best_teams"] else "N/D"
                    )
                    _best_team_rating = (
                        _sr["best_teams"][0].predicted_rating if _sr["best_teams"] else 0.0
                    )

                    # Strengths from RS stats (proportional to minutes to avoid
                    # inflating bench-player strengths via per-game stats).
                    _strengths = []
                    if float(_ls.get("points", 0) or 0) >= 12:
                        _strengths.append("scorer prolifico")
                    if float(_ls.get("rebounds", 0) or 0) >= 6:
                        _strengths.append("rimbalzista dominante")
                    if float(_ls.get("assists", 0) or 0) >= 5:
                        _strengths.append("playmaker di qualità")
                    if float(_ls.get("steals", 0) or 0) >= 1.2:
                        _strengths.append("difensore aggressivo")
                    if float(_ls.get("blocks", 0) or 0) >= 1.2:
                        _strengths.append("shot-blocker")
                    if float(_ls.get("three_point_pct", 0) or 0) >= 0.36:
                        _strengths.append("tiratore da tre efficiente")
                    if float(_ls.get("ts_pct", 0) or 0) >= 0.56:
                        _strengths.append("finisher efficiente")
                    if float(_ls.get("usg_pct", 0) or 0) >= 25:
                        _strengths.append("prima opzione offensiva")
                    if float(_ls.get("bpm", 0) or 0) >= 2.0:
                        _strengths.append("impatto positivo sul campo")
                    if not _strengths:
                        # If no stats are populated, say so rather than "versatile e bilanciato"
                        _has_any_stat = any(
                            float(_ls.get(c, 0) or 0) > 0
                            for c in ("points", "rebounds", "assists", "steals")
                        )
                        _strengths = (
                            ["dati statistici insufficienti per analisi specifica"]
                            if not _has_any_stat
                            else ["giocatore di rotazione / ruolo specifico"]
                        )

                    # Peak description: distinguish future vs already-past peak
                    if _seasons_to > 0:
                        _peak_desc = (
                            f"Il picco prestativo è atteso a **{_peak_age} anni** "
                            f"(rating stimato al picco: **{_peak_r:.2f}**), "
                            f"tra **{_seasons_to} {'stagione' if _seasons_to == 1 else 'stagioni'}**."
                        )
                    else:
                        _peak_desc = (
                            f"Il modello stima che il picco prestativo era intorno ai "
                            f"**{_peak_age} anni** (rating stimato: **{_peak_r:.2f}**) — "
                            f"il giocatore ha già superato o è vicino al suo apice."
                        )

                    st.markdown(f"""
    **{_sr['name']}** è un giocatore di **{_sr['age']} anni** ({_sr['pos']}{_mpg_ctx}) classificato come
    **{_tier}**, con un rating AI attuale di **{_rating:.2f}/10** e una traiettoria di rendimento
    **{_trend_str}**.

    **Punti di forza identificati dall'AI:** {', '.join(_strengths)}.

    {_peak_desc}

    Il contesto ottimale per esprimere il suo massimo potenziale è **{_best_team_name}**,
    dove il modello stima un rating di **{_best_team_rating:.2f}** — la miglior opportunità
    nel database corrente.

    > *Report generato automaticamente dall'AI Basketball Intelligence Engine.
    > Basato su modello ensemble (XGBoost + Random Forest) con {len(_traj)} punti di traiettoria.*
    """)

                    # --- Performance DNA Chart ---
                    st.subheader("🧬 DNA Prestativo")
                    # BPM normalisation: shift [-10, +10] → [0%, 100%] so that
                    # negative values still render (0% = BPM −10, 50% = neutral, 100% = +10).
                    _BPM_MIN, _BPM_MAX = -10.0, 10.0
                    _dna_cols = [
                        ("Punti",         "points",           20.0),
                        ("Rimbalzi",      "rebounds",         12.0),
                        ("Assist",        "assists",          10.0),
                        ("Rubate",        "steals",            3.0),
                        ("Stoppate",      "blocks",            3.0),
                        ("% tiro",        "fg_pct",            1.0),
                        ("% 3pt",         "three_point_pct",   1.0),
                        ("Efficienza TS", "ts_pct",            1.0),
                        ("BPM",           "bpm",              10.0),
                        ("USG%",          "usg_pct",           1.0),
                    ]
                    # Use RS stats for DNA chart; BPM can be negative → floor at 0
                    _dna_data = {}
                    for _label, _col, _max_val in _dna_cols:
                        _raw_val = _ls.get(_col)
                        if _raw_val is None:
                            # Column absent from stats dict → show 0% (not 50% for BPM)
                            _pct = 0.0
                        else:
                            _raw = float(_raw_val or 0)
                            if _col == "bpm":
                                _pct = min(100.0, max(0.0, (_raw - _BPM_MIN) / (_BPM_MAX - _BPM_MIN) * 100))
                            else:
                                _pct = min(100.0, max(0.0, _raw / _max_val * 100))
                        _dna_data[_label] = round(_pct, 1)

                    if all(v == 0.0 for v in _dna_data.values()):
                        st.info(
                            "📭 Statistiche non disponibili per questo giocatore. "
                            "Il DNA verrà visualizzato una volta che i dati saranno caricati nel database."
                        )
                    else:
                        _dna_df = pd.DataFrame(
                            list(_dna_data.items()), columns=["Metrica", "Percentuale (0-100)"]
                        ).set_index("Metrica")
                        st.bar_chart(_dna_df, height=350)
                        _dna_src = "RS (Regular Season)" if _ls_rs_available else "ultima stagione disponibile"
                        st.caption(
                            f"Valori normalizzati rispetto ai massimi di riferimento. "
                            f"Fonte dati: {_dna_src}. "
                            f"BPM riscalato: 50% = neutro (0.0), 100% = +{_BPM_MAX:.0f}, 0% = {_BPM_MIN:.0f}."
                        )

                    # --- Career Trajectory Chart ---
                    st.subheader("📈 Traiettoria di Carriera")
                    _traj_df = pd.DataFrame([
                        {
                            "Età":      pt.age,
                            "Rating":   pt.predicted_rating,
                            "CI basso": pt.confidence_low,
                            "CI alto":  pt.confidence_high,
                        }
                        for pt in _sr["traj"]
                    ]).set_index("Età")
                    st.line_chart(_traj_df[["Rating", "CI basso", "CI alto"]])

                    _c1, _c2 = st.columns(2)

                    # --- Player Comparables ---
                    with _c1:
                        st.subheader("👥 Giocatori Comparabili")
                        if _sr["comparables"]:
                            _comp_df = pd.DataFrame(_sr["comparables"])
                            st.dataframe(_comp_df, hide_index=True, width="stretch")
                            st.caption(
                                "Similarità basata su coseno dei vettori statistici "
                                f"({len(_sr['avail_cols'])} metriche)."
                            )
                        else:
                            st.info("Dati insufficienti per calcolare comparabili.")

                    # --- Best Team Fits ---
                    with _c2:
                        st.subheader("🏆 Top 5 Squadre Ideali")
                        if _sr["best_teams"]:
                            _teams_df = pd.DataFrame([
                                {
                                    "#": t.rank,
                                    "Squadra":       t.team_name,
                                    "Lega":          t.league_name,
                                    "Rating previsto": round(t.predicted_rating, 2),
                                    "Compatibilità": round(t.compatibility_score * 100, 1),
                                }
                                for t in _sr["best_teams"]
                            ])
                            st.dataframe(_teams_df, hide_index=True, width="stretch")
                        else:
                            st.info("Nessuna squadra disponibile.")

                    # --- Stats by competition ---
                    with st.expander("📊 Statistiche dettagliate per competizione", expanded=False):
                        _stats_by_comp = _sr.get("stats_by_comp", {})
                        if _stats_by_comp:
                            _comp_labels = {"RS": "🏆 Regular Season", "PO": "🔥 Playoff",
                                            "CUP": "🥇 Coppa Nazionale", "SUPERCUP": "⭐ Supercoppa"}
                            for _comp_key in ["RS", "PO", "CUP", "SUPERCUP"]:
                                if _comp_key not in _stats_by_comp:
                                    continue
                                _comp_row = _stats_by_comp[_comp_key]
                                _comp_clean = {
                                    k: v for k, v in _comp_row.items()
                                    if v is not None and str(v) not in ("", "nan")
                                    and k not in ("player_id", "competition")
                                }
                                if not _comp_clean:
                                    continue
                                st.markdown(f"**{_comp_labels.get(_comp_key, _comp_key)}**")
                                st.dataframe(
                                    pd.DataFrame(
                                        list(_comp_clean.items()),
                                        columns=["Metrica", "Valore"],
                                    ),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                        else:
                            # Fallback: show all latest stats flat
                            _raw_stats = {k: v for k, v in _sr["latest_stats"].items()
                                          if v is not None and str(v) not in ("", "nan")}
                            if _raw_stats:
                                st.dataframe(
                                    pd.DataFrame(list(_raw_stats.items()), columns=["Metrica", "Valore"]),
                                    hide_index=True, use_container_width=True,
                                )
                            else:
                                st.info("Nessuna statistica disponibile nel database per questo giocatore.")


with _safe_tab(tab_mapping, "Mapping"):
    if not _can_access("mapping"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
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
        # Sub-tab 2 – Team display-name mapping (league-aware)
        # -----------------------------------------------------------------------
        with _map_display_tab:
            st.markdown(
                "Scegli quale campo usare come **nome visualizzato** delle squadre per ogni lega.  \n"
                "- **TeamName** — nome completo della squadra (es. *Olimpia Milano*)  \n"
                "- **ShortName** — abbreviazione/acronimo dal DB (es. *MIL*, *OLI*)  \n\n"
                "La chiave è il **codice lega** (es. `ITA1`, `GRC1`), che identifica univocamente "
                "la lega anche quando nella stessa nazione esistono più campionati (es. `ITA1` vs `ITA2`).  \n"
                "Come fallback puoi usare anche il prefisso nazione (es. `ITA`, `GRC`).  \n"
                "La configurazione viene salvata nel file `.env` e applicata immediatamente."
            )

            # ----- Auto-populate league codes from loaded data ------------------
            _loaded_league_codes: list[str] = []
            if "data" in st.session_state:
                _lg_df = st.session_state["data"].get("leagues")
                if _lg_df is not None and "id" in _lg_df.columns:
                    _loaded_league_codes = sorted(
                        {
                            str(r["id"]).strip().upper()
                            for _, r in _lg_df.iterrows()
                            if str(r.get("id", "")).strip()
                        }
                    )

            # ----- Load persisted mapping from env-var --------------------------
            if "team_display_map_rows" not in st.session_state:
                _initial_map = parse_team_display_map(os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, ""))
                _seed_keys = _loaded_league_codes or sorted(_initial_map.keys())
                st.session_state["team_display_map_rows"] = [
                    {
                        "Lega (es. ITA1, GRC1)": key,
                        "Campo da visualizzare": _initial_map.get(key, "short_name"),
                    }
                    for key in _seed_keys
                ] or [{"Lega (es. ITA1, GRC1)": "", "Campo da visualizzare": "short_name"}]
            else:
                # If new league codes were loaded from DB, add any missing ones
                _existing_keys = {
                    r["Lega (es. ITA1, GRC1)"].strip().upper()
                    for r in st.session_state["team_display_map_rows"]
                    if r.get("Lega (es. ITA1, GRC1)", "").strip()
                }
                _current_saved = parse_team_display_map(os.environ.get(TEAM_DISPLAY_FIELD_MAP_ENV, ""))
                for _lk in _loaded_league_codes:
                    if _lk and _lk not in _existing_keys:
                        st.session_state["team_display_map_rows"].append({
                            "Lega (es. ITA1, GRC1)": _lk,
                            "Campo da visualizzare": _current_saved.get(_lk, "short_name"),
                        })

            if not _loaded_league_codes:
                st.info(
                    "ℹ️ Nessun dato SQL caricato. Puoi aggiungere manualmente i codici lega oppure "
                    "carica prima i dati dal tab **📂 Dati** per auto-popolare la lista."
                )

            _display_df = pd.DataFrame(st.session_state["team_display_map_rows"])

            _edited_display = st.data_editor(
                _display_df,
                key="team_display_map_editor",
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "Lega (es. ITA1, GRC1)": st.column_config.TextColumn(
                        "Lega (es. ITA1, GRC1)",
                        width="medium",
                        help=(
                            "Codice lega come appare nel DB, es. **ITA1**, **GRC1**, **ITA2**.  \n"
                            "Puoi anche usare un prefisso nazione (es. **ITA**, **GRC**) come fallback "
                            "per tutte le leghe di quella nazione non mappate esplicitamente."
                        ),
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
                    str(r["Lega (es. ITA1, GRC1)"]).strip().upper(): r["Campo da visualizzare"]
                    for _, r in _edited_display.iterrows()
                    if str(r.get("Lega (es. ITA1, GRC1)", "")).strip()
                       and r.get("Campo da visualizzare") in ("name", "short_name")
                }
                _prev_country_map = _get_team_country_map(st.session_state["data"])
                _prev_league_map  = _get_team_league_map(st.session_state["data"])
                _teams_preview    = st.session_state["data"].get("teams")
                if _teams_preview is not None and not _teams_preview.empty:
                    with st.expander("👁️ Anteprima nomi squadre", expanded=True):
                        _prev_rows = []
                        for _, _tr in _teams_preview.iterrows():
                            _tid = _to_int(_tr.get("id", 0))
                            _lc  = _prev_league_map.get(_tid, "")
                            _cc  = _prev_country_map.get(_tid, "")
                            _prev_rows.append({
                                "Squadra (nome DB)": str(_tr.get("name", "")),
                                "ShortName DB":      str(_tr.get("short_name", "") or ""),
                                "Lega":              _lc,
                                "Nazione":           _cc,
                                "Visualizzato come": _make_team_display_name(
                                    _tr, _prev_country_map, _prev_map, _prev_league_map
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(_prev_rows),
                            hide_index=True,
                            width="stretch",
                        )

            st.divider()
            _dc1, _dc2, _dc3 = st.columns([2, 2, 6])

            with _dc1:
                if st.button("💾 Salva display mapping", type="primary", key="display_map_save"):
                    _new_map_str = ",".join(
                        f"{str(r['Lega (es. ITA1, GRC1)']).strip().upper()}:{r['Campo da visualizzare']}"
                        for _, r in _edited_display.iterrows()
                        if str(r.get("Lega (es. ITA1, GRC1)", "")).strip()
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

# ===========================================================================
# TAB 7 – GESTIONE UTENTI (solo admin)
# ===========================================================================

with _safe_tab(tab_admin, "Utenti"):
    st.header("👥 Gestione Utenti")
    _is_admin = st.session_state.get("current_role") == "admin"

    if not _is_admin:
        st.warning("⛔ Accesso riservato agli amministratori.")
    else:
        # Load all users and roles once at the start of the admin block
        _all_users     = load_users()
        _current_roles = load_roles()

        # -----------------------------------------------------------------------
        # User list
        # -----------------------------------------------------------------------
        st.subheader(f"Utenti registrati ({len(_all_users)})")
        if _all_users:
            _users_df = pd.DataFrame([
                {
                    "Username": uname,
                    "Ruolo":    info.get("role", "—"),
                    "Creato da": info.get("created_by", "—"),
                }
                for uname, info in _all_users.items()
            ])
            st.dataframe(_users_df, width="stretch", hide_index=True)
        else:
            st.caption("Nessun utente trovato.")

        st.divider()

        # -----------------------------------------------------------------------
        # Create user
        # -----------------------------------------------------------------------
        with st.expander("➕ Crea nuovo utente", expanded=False):
            with st.form("admin_create_user"):
                _new_uname = st.text_input("Username", key="au_new_uname")
                _new_pw    = st.text_input("Password", type="password", key="au_new_pw")
                _new_pw2   = st.text_input("Conferma password", type="password", key="au_new_pw2")
                _new_role  = st.selectbox(
                    "Ruolo", list(_current_roles.keys()), key="au_new_role"
                )
                _create_btn = st.form_submit_button("Crea utente", type="primary")

            if _create_btn:
                if not _new_uname.strip():
                    st.error("Inserisci un username.")
                elif not _new_pw:
                    st.error("Inserisci una password.")
                elif _new_pw != _new_pw2:
                    st.error("Le password non coincidono.")
                else:
                    _ok = create_user(
                        _new_uname, _new_pw, _new_role,
                        created_by=st.session_state.get("current_user", "admin"),
                    )
                    if _ok:
                        st.success(f"✅ Utente **{_new_uname.strip().lower()}** creato con ruolo **{_new_role}**.")
                        st.rerun()
                    else:
                        st.error(f"Username **{_new_uname.strip().lower()}** già esistente.")

        # -----------------------------------------------------------------------
        # Reset password
        # -----------------------------------------------------------------------
        with st.expander("🔑 Reset password utente", expanded=False):
            _all_usernames = list(_all_users.keys())
            if _all_usernames:
                with st.form("admin_reset_pw"):
                    _reset_uname = st.selectbox(
                        "Utente da modificare", _all_usernames, key="au_reset_uname"
                    )
                    _reset_pw  = st.text_input("Nuova password", type="password", key="au_reset_pw")
                    _reset_pw2 = st.text_input(
                        "Conferma nuova password", type="password", key="au_reset_pw2"
                    )
                    _reset_btn = st.form_submit_button("Reimposta password", type="primary")

                if _reset_btn:
                    if not _reset_pw:
                        st.error("Inserisci una nuova password.")
                    elif _reset_pw != _reset_pw2:
                        st.error("Le password non coincidono.")
                    else:
                        _ok = reset_password(_reset_uname, _reset_pw)
                        if _ok:
                            revoke_user_sessions(_reset_uname)
                            st.success(f"✅ Password di **{_reset_uname}** aggiornata.")
                        else:
                            st.error("Errore durante il reset della password.")
            else:
                st.caption("Nessun utente disponibile.")

        # -----------------------------------------------------------------------
        # Delete user
        # -----------------------------------------------------------------------
        with st.expander("🗑️ Elimina utente", expanded=False):
            _deletable = [u for u in _all_users if u != st.session_state.get("current_user")]
            if _deletable:
                with st.form("admin_delete_user"):
                    _del_uname = st.selectbox(
                        "Utente da eliminare", _deletable, key="au_del_uname"
                    )
                    _del_btn = st.form_submit_button("❌ Elimina", type="primary")

                if _del_btn:
                    _ok = delete_user(_del_uname)
                    if _ok:
                        revoke_user_sessions(_del_uname)
                        st.success(f"✅ Utente **{_del_uname}** eliminato.")
                        st.rerun()
                    else:
                        st.error("Errore durante l'eliminazione.")
            else:
                st.caption("Non è possibile eliminare il proprio account o altri utenti.")

        # -----------------------------------------------------------------------
        # Change own password
        # -----------------------------------------------------------------------
        st.divider()
        st.subheader("🔒 Cambia la tua password")
        with st.form("admin_change_own_pw"):
            _own_old  = st.text_input("Password attuale",   type="password", key="au_own_old")
            _own_new  = st.text_input("Nuova password",     type="password", key="au_own_new")
            _own_new2 = st.text_input("Conferma password",  type="password", key="au_own_new2")
            _own_btn  = st.form_submit_button("Aggiorna password", type="primary")

        if _own_btn:
            if not _own_new:
                st.error("Inserisci una nuova password.")
            elif _own_new != _own_new2:
                st.error("Le password non coincidono.")
            else:
                _ok = change_password(
                    st.session_state.get("current_user", ""),
                    _own_old, _own_new,
                )
                if _ok:
                    revoke_user_sessions(st.session_state.get("current_user", ""))
                    st.success("✅ Password aggiornata con successo.")
                else:
                    st.error("Password attuale errata.")

        # -----------------------------------------------------------------------
        # Role management – define roles and their section-level privileges
        # -----------------------------------------------------------------------
        st.divider()
        st.subheader("🛡️ Gestione Ruoli e Permessi")
        st.caption(
            "Definisci quali **sezioni** (tab) ogni ruolo può visualizzare. "
            "Le modifiche vengono applicate immediatamente agli utenti con quel ruolo."
        )

        _current_roles = load_roles()
        _role_tab_names = list(_current_roles.keys())

        # Tabs – one per role
        if _role_tab_names:
            _role_tabs = st.tabs([f"👤 {r}" for r in _role_tab_names])
            for _ri, (_rt, _rname) in enumerate(zip(_role_tabs, _role_tab_names)):
                with _rt:
                    _rdata = _current_roles[_rname]
                    st.markdown(
                        f"**Descrizione:** {_rdata.get('description', '—')}  \n"
                        f"**Sezioni attive:** "
                        + ", ".join(
                            f"`{SECTION_LABELS.get(s, s)}`"
                            for s in _rdata.get("sections", [])
                        ) or "nessuna"
                    )
                    # Checkbox per ogni sezione
                    with st.form(f"role_edit_{_rname}"):
                        _new_desc = st.text_input(
                            "Descrizione ruolo",
                            value=_rdata.get("description", ""),
                            key=f"rdesc_{_rname}",
                        )
                        st.markdown("**Sezioni accessibili:**")
                        _cols = st.columns(2)
                        _new_sections: list[str] = []
                        for _si, _sec in enumerate(ALL_SECTIONS):
                            _checked = _sec in _rdata.get("sections", [])
                            _label = SECTION_LABELS.get(_sec, _sec)
                            _col = _cols[_si % 2]
                            if _col.checkbox(_label, value=_checked, key=f"rs_{_rname}_{_sec}"):
                                _new_sections.append(_sec)
                        _save_role_btn = st.form_submit_button(
                            f"💾 Salva permessi '{_rname}'", type="primary"
                        )

                    if _save_role_btn:
                        _current_roles[_rname]["description"] = _new_desc
                        _current_roles[_rname]["sections"]    = _new_sections
                        save_roles(_current_roles)
                        st.success(f"✅ Permessi del ruolo **{_rname}** aggiornati.")
                        st.rerun()

        st.divider()
        _rc1, _rc2 = st.columns(2)

        # Add custom role
        with _rc1:
            st.markdown("**➕ Crea nuovo ruolo**")
            with st.form("admin_new_role"):
                _nr_name = st.text_input("Nome ruolo (es. scout, coach)", key="nr_name")
                _nr_desc = st.text_input("Descrizione", key="nr_desc")
                _nr_sections: list[str] = []
                st.markdown("Sezioni:")
                _nrc = st.columns(2)
                for _si, _sec in enumerate(ALL_SECTIONS):
                    if _nrc[_si % 2].checkbox(
                        SECTION_LABELS.get(_sec, _sec), key=f"nrs_{_sec}"
                    ):
                        _nr_sections.append(_sec)
                _nr_btn = st.form_submit_button("Crea ruolo", type="primary")

            if _nr_btn:
                _nr_key = _nr_name.strip().lower().replace(" ", "_")
                if not _nr_key:
                    st.error("Inserisci un nome per il ruolo.")
                elif _nr_key in _current_roles:
                    st.error(f"Il ruolo **{_nr_key}** esiste già.")
                else:
                    _current_roles[_nr_key] = {
                        "description": _nr_desc.strip(),
                        "sections":    _nr_sections,
                    }
                    save_roles(_current_roles)
                    st.success(f"✅ Ruolo **{_nr_key}** creato.")
                    st.rerun()

        # Delete custom role
        with _rc2:
            st.markdown("**🗑️ Elimina ruolo**")
            _deletable_roles = [
                r for r in _current_roles
                if r not in ("admin", "analyst", "viewer")
            ]
            if _deletable_roles:
                with st.form("admin_del_role"):
                    _dr_name = st.selectbox(
                        "Ruolo da eliminare", _deletable_roles, key="dr_name"
                    )
                    _dr_btn = st.form_submit_button("❌ Elimina ruolo", type="primary")
                if _dr_btn:
                    del _current_roles[_dr_name]
                    save_roles(_current_roles)
                    st.success(f"✅ Ruolo **{_dr_name}** eliminato.")
                    st.rerun()
            else:
                st.caption(
                    "Nessun ruolo personalizzato da eliminare. "
                    "I ruoli predefiniti (admin, analyst, viewer) non possono essere eliminati."
                )


# ===========================================================================
# TAB – AUDIT LOG
# ===========================================================================

with _safe_tab(tab_audit, "Audit"):
    if not _can_access("admin"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
        st.header("📋 Audit Log")

        col1, col2, col3 = st.columns(3)
        with col1:
            audit_user   = st.text_input("Filter by user", key="audit_user")
        with col2:
            audit_tenant = st.text_input("Filter by tenant", key="audit_tenant")
        with col3:
            audit_limit  = st.number_input("Limit", min_value=10, max_value=500, value=100, step=10, key="audit_limit")

        if st.button("Load audit log", key="btn_audit"):
            try:
                from basketball_ai.api.audit import AuditDB
                db = AuditDB()
                rows = db.query(
                    user=audit_user or None,
                    tenant=audit_tenant or None,
                    limit=int(audit_limit),
                )
                if rows:
                    df = pd.DataFrame(rows)
                    if "ts" in df.columns:
                        df["datetime"] = df["ts"].apply(
                            lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else ""
                        )
                        df = df.drop(columns=["ts", "id"], errors="ignore")
                        df = df[["datetime"] + [c for c in df.columns if c != "datetime"]]
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"Showing {len(rows)} records")
                else:
                    st.info("No audit records found.")
            except Exception as exc:
                st.error(f"Could not load audit log: {exc}")

# ===========================================================================
# TAB – HEALTH DASHBOARD
# ===========================================================================

with _safe_tab(tab_health, "Health"):
    if not _can_access("admin"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
        st.header("🏥 System Health")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Service Status")
            try:
                import requests
                api_port = os.environ.get("API_PORT", "8000")
                resp = requests.get(f"http://localhost:{api_port}/health/ready", timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("✅ API Ready")
                    st.json(data)
                else:
                    st.error(f"❌ API Not Ready (HTTP {resp.status_code})")
            except Exception as exc:
                st.warning(f"⚠️ Could not reach API: {exc}")

        with col2:
            st.subheader("Ingestion State")
            try:
                from basketball_ai.data.ingestion import IngestionTracker
                tracker = IngestionTracker()
                stats = tracker.stats()
                failed_list = tracker.list_failed()
                st.metric("DONE", stats.get("DONE", 0))
                st.metric("FAILED", stats.get("FAILED", 0))
                st.metric("DEAD", stats.get("DEAD", 0))
                if failed_list:
                    st.warning(f"{len(failed_list)} failed/dead items")
                    if st.checkbox("Show failed items", key="show_failed"):
                        st.dataframe(pd.DataFrame(failed_list), use_container_width=True)
            except Exception as exc:
                st.info(f"Ingestion tracker: {exc}")

        st.subheader("In-process Metrics")
        if st.button("Load metrics", key="btn_metrics"):
            try:
                from basketball_ai.api.metrics import get_snapshot
                snap = get_snapshot()
                st.caption(f"Uptime: {snap.get('uptime_seconds', 0):.0f}s")
                routes = snap.get("routes", [])
                if routes:
                    df = pd.DataFrame(routes)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No route metrics yet.")
            except Exception as exc:
                st.info(f"Metrics: {exc}")

# ===========================================================================
# TAB – DRIFT MONITOR
# ===========================================================================

with _safe_tab(tab_drift, "Drift"):
    if not _can_access("data"):
        st.warning("⛔ Non hai i permessi per accedere a questa sezione.")
    else:
        st.header("📊 Feature Drift Monitor")

        st.info(
            "Compares the current data distribution to the reference distribution "
            "captured during the last training run. PSI > 0.25 indicates significant drift."
        )

        if "data" not in st.session_state or st.session_state.data is None:
            st.warning("Load data first (use the 📂 Dati tab).")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Compute PSI drift", key="btn_psi"):
                    try:
                        from basketball_ai.monitoring.drift import capture_reference, compute_psi_report
                        data = st.session_state.data
                        ref  = capture_reference(data)
                        report = compute_psi_report(data, ref)
                        if report:
                            df = pd.DataFrame(
                                [{"feature": k, "psi": v, "status": "🔴 DRIFT" if v > 0.25 else ("🟡 MODERATE" if v > 0.10 else "🟢 OK")}
                                 for k, v in sorted(report.items(), key=lambda x: -x[1])]
                            )
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No drift metrics available.")
                    except Exception as exc:
                        st.error(f"PSI drift failed: {exc}")

            with col2:
                if st.button("Compute KS drift", key="btn_ks"):
                    try:
                        from basketball_ai.monitoring.drift import compute_ks_drift
                        data = st.session_state.data
                        ps   = data.get("player_stats")
                        if ps is not None and not ps.empty:
                            mid = len(ps) // 2
                            ref_df = ps.iloc[:mid]
                            cur_df = ps.iloc[mid:]
                            result = compute_ks_drift(ref_df, cur_df)
                            df = pd.DataFrame(
                                [{"feature": k, "ks_pvalue": v,
                                  "status": "🟢 OK" if v > 0.05 else "🔴 DRIFT"}
                                 for k, v in sorted(result.items(), key=lambda x: x[1])]
                            )
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No player_stats data loaded.")
                    except Exception as exc:
                        st.error(f"KS drift failed: {exc}")

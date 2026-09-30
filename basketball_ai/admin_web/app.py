"""FastAPI backend for the custom AI_Model operations console.

The admin service intentionally stays separate from the public inference API.
It serves a same-origin static UI plus authenticated JSON endpoints on port 8501.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from basketball_ai.auth.auth import (
    ADMIN_CREDENTIALS_FILE,
    AUDIT_LOG_FILE,
    change_password,
    check_credentials,
    create_session_token,
    ensure_default_admin,
    revoke_session_token,
    revoke_user_sessions,
    validate_session_token,
)
from basketball_ai.data.connection_profiles import load_profiles, profile_url, save_profile
from basketball_ai.data.postgres_loader import get_engine, load_all_data
from basketball_ai.data.training_snapshots import (
    create_training_snapshot,
    list_training_snapshots,
    load_training_snapshot,
)
from basketball_ai.models.promote import (
    MAX_SEGMENT_REGRESSION_PCT,
    MIN_BACKTEST_SAMPLES,
    MIN_COMPETITION_SAMPLES,
    PROMOTION_THRESHOLD_PCT,
    TARGET_INTERVAL_COVERAGE,
    get_promotion_status,
    promote_if_better,
    rollback_to_previous,
)

MODEL_ROOT = Path(os.getenv("MODEL_DIR", "models_saved"))
STATIC_ROOT = Path(__file__).with_name("static")
API_BASE_URL = os.getenv("ADMIN_API_BASE_URL", "http://api:8000").rstrip("/")
SESSION_COOKIE = "hm_ai_admin_session"
CSRF_COOKIE = "hm_ai_admin_csrf"
COOKIE_SECURE = os.getenv("ADMIN_COOKIE_SECURE", "true").strip().lower() not in {"0", "false", "no"}
COOKIE_TTL_DAYS = int(os.getenv("SESSION_COOKIE_TTL_DAYS", "7"))
SEASON_LIFECYCLE_FILE = Path(os.getenv("SEASON_LIFECYCLE_FILE", "/app/config/season_lifecycle.json"))


@dataclass
class AdminState:
    data: dict[str, Any] | None = None
    active_profile: str | None = field(default_factory=lambda: os.getenv("DATABASE_PROFILE") or None)
    summary: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] | None = None
    training: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "idle",
            "stage": "Ready",
            "progress": 0,
            "message": "No training run in progress.",
            "run_id": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
    )
    lock: threading.RLock = field(default_factory=threading.RLock)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-admin-training"))


STATE = AdminState()
app = FastAPI(title="AI Model Control Center", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/assets", StaticFiles(directory=str(STATIC_ROOT)), name="assets")


class LoginPayload(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class ProfilePayload(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(min_length=1, max_length=128)
    user: str = Field(min_length=1, max_length=128)
    password: str = Field(default="", max_length=1024)
    source_schema: str = Field(default="AI_Source", min_length=1, max_length=128)
    ai_schema: str = Field(default="AI", min_length=1, max_length=128)


class ConfirmPayload(BaseModel):
    confirm: bool = False


class PasswordPayload(BaseModel):
    old_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=1, max_length=512)


class SeasonLifecyclePayload(BaseModel):
    season: int = Field(ge=2000, le=2100)
    status: str = Field(pattern="^(in_progress|complete|locked)$")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _season_lifecycle() -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(SEASON_LIFECYCLE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_season_lifecycle(value: dict[str, dict[str, str]]) -> None:
    SEASON_LIFECYCLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SEASON_LIFECYCLE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    if os.name != "nt":
        temporary.chmod(0o600)
    temporary.replace(SEASON_LIFECYCLE_FILE)
    if os.name != "nt":
        SEASON_LIFECYCLE_FILE.chmod(0o600)


def _lifecycle_rows(seasons: list[int]) -> list[dict[str, Any]]:
    values = _season_lifecycle()
    return [
        {"season": season, "status": values.get(str(season), {}).get("status", "unclassified")}
        for season in sorted(seasons, reverse=True)
    ]


def _seasons(data: dict[str, Any] | None) -> list[int]:
    if not data or "player_stats" not in data or data["player_stats"].empty:
        return []
    values: set[int] = set()
    for value in data["player_stats"]["season"].dropna():
        try:
            values.add(int(str(value).split("-")[0]))
        except (TypeError, ValueError):
            continue
    return sorted(values)


def _competitions(data: dict[str, Any] | None) -> list[str]:
    if not data or "player_stats" not in data or "competition" not in data["player_stats"].columns:
        return []
    return sorted(
        {
            str(value).strip().upper()
            for value in data["player_stats"]["competition"].dropna()
            if str(value).strip()
        }
    )


def _summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    seasons = _seasons(data)
    competitions = _competitions(data)
    return {
        "players": len(data.get("players", pd.DataFrame())),
        "player_rows": len(data.get("player_stats", pd.DataFrame())),
        "team_rows": len(data.get("team_season_stats", pd.DataFrame())),
        "competitions": competitions,
        "seasons": seasons,
        "season_min": min(seasons) if seasons else None,
        "season_max": max(seasons) if seasons else None,
        "source_contract": data.get("source_contract", "competition-v2:canonical"),
    }


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    try:
        profiles = load_profiles().values()
    except Exception:
        profiles = []
    for profile in profiles:
        password = str(profile.get("password", ""))
        if password:
            message = message.replace(password, "********")
    return message[:1200]


def _registry() -> dict[str, Any]:
    try:
        return get_promotion_status(str(MODEL_ROOT))
    except Exception:
        return {"production": None, "candidate": None, "previous": None, "history": []}


def _candidate_quality(candidate: dict[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    report = candidate.get("backtest") or {}
    overall = report.get("overall") or {}
    folds = [fold for fold in report.get("folds", []) if isinstance(fold, dict) and fold.get("valid")]
    return {
        "rmse": overall.get("rmse"),
        "mae": overall.get("mae"),
        "r2": overall.get("r2"),
        "coverage": overall.get("interval_coverage"),
        "samples": overall.get("n"),
        "seasons": len(folds),
        "competitions": len(report.get("by_competition") or {}),
        "folds": [
            {"target_season": fold.get("target_season"), "rmse": fold.get("rmse")}
            for fold in folds
            if fold.get("rmse") is not None
        ],
    }


def _user_from_request(request: Request) -> dict[str, str]:
    token = request.cookies.get(SESSION_COOKIE, "")
    valid = validate_session_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Authentication required")
    username, role = valid
    return {"username": username, "role": role, "token": token}


def _operator(user: dict[str, str] = Depends(_user_from_request)) -> dict[str, str]:
    if user["role"] not in {"admin", "analyst"}:
        raise HTTPException(status_code=403, detail="Operator role required")
    return user


def _admin(user: dict[str, str] = Depends(_user_from_request)) -> dict[str, str]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required")
    return user


def _csrf(
    request: Request,
    x_csrf_token: str = Header(default=""),
    user: dict[str, str] = Depends(_operator),
) -> dict[str, str]:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not cookie or not x_csrf_token or not secrets.compare_digest(cookie, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return user


def _audit_rows(limit: int = 200) -> list[dict[str, Any]]:
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            continue
    return rows


def _set_training(**updates: Any) -> None:
    with STATE.lock:
        STATE.training.update(updates)


def _training_worker(
    data: dict[str, Any],
    active_profile: str | None,
    snapshot: dict[str, Any] | None = None,
) -> None:
    from basketball_ai.models.backtest import run_backtest
    from basketball_ai.models.promote import register_candidate
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    try:
        if snapshot and snapshot.get("snapshot_id"):
            snapshot_data = load_training_snapshot(str(snapshot["snapshot_id"]))
            manifest = snapshot_data.get("training_snapshot_manifest", {})
            if manifest.get("sha256") != snapshot.get("sha256"):
                raise RuntimeError("Training snapshot checksum/manifest mismatch")
            data = snapshot_data
            seasons = _seasons(data)
            _set_training(
                stage="Training snapshot activated",
                progress=8,
                message=f"Training from immutable snapshot {snapshot['snapshot_id']}…",
            )
        seasons = _seasons(data)
        _set_training(status="running", stage="Training ensemble", progress=12, message="Training the strict competition-aware ensemble…")
        model = StrictProductionEnsembleModel()
        metrics = model.train(data)

        _set_training(stage="Walk-forward backtest", progress=58, message="Running leakage-safe out-of-time folds…")
        # Use every eligible OOT target season by default. An explicit
        # MODEL_BACKTEST_FOLDS value can cap runtime for large histories.
        default_folds = max(1, len(seasons) - 4)
        try:
            configured_folds = int(os.getenv("MODEL_BACKTEST_FOLDS", str(default_folds)))
        except ValueError:
            configured_folds = default_folds
        n_backtest_folds = min(max(1, configured_folds), default_folds)
        report = run_backtest(
            data,
            n_folds=n_backtest_folds,
            output_path=str(MODEL_ROOT / "backtest_report.json"),
        )
        if not report.get("valid") or not report.get("folds"):
            raise RuntimeError("Backtest invalid; candidate cannot enter the registry")

        _set_training(stage="Saving immutable run", progress=88, message="Saving candidate artifacts and registry metadata…")
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        run_dir = MODEL_ROOT / "runs" / run_id
        metadata = {
            **(metrics or {}),
            "model_run_id": run_id,
            "model_version": "2.6.0",
            "feature_version": "forecast-t-plus-1-persistence-delta-v1",
            "data_cutoff": datetime.now(timezone.utc).date().isoformat(),
            "latest_observed_season": max(seasons),
            "database_profile": active_profile,
            "training_snapshot_id": (snapshot or {}).get("snapshot_id"),
            "training_snapshot_sha256": (snapshot or {}).get("sha256"),
            "source_contract": data.get("source_contract", "competition-v2:canonical"),
            "backtest": report,
        }
        model.save(str(run_dir), metadata)
        register_candidate(str(MODEL_ROOT), run_id, run_dir, metadata)
        _set_training(
            status="complete",
            stage="Candidate ready",
            progress=100,
            message="Training and walk-forward validation completed. Production is unchanged until promotion.",
            run_id=run_id,
            error=None,
            finished_at=_utcnow(),
        )
    except Exception as exc:  # pragma: no cover - exercised by production failures
        _set_training(
            status="failed",
            stage="Failed",
            progress=100,
            message="Training failed.",
            error=_safe_error(exc),
            finished_at=_utcnow(),
        )


def _probe(path: str) -> dict[str, Any]:
    try:
        req = URLRequest(f"{API_BASE_URL}{path}", headers={"Accept": "application/json"})
        with urlopen(req, timeout=2.0) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body[:300]}
            return {"ok": 200 <= response.status < 300, "status": response.status, "body": parsed}
    except Exception as exc:
        return {"ok": False, "status": None, "body": None, "error": str(exc)[:300]}


def _scenario_entity_rows(entity: str, query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search canonical source entities directly in PostgreSQL."""
    with STATE.lock:
        profile_name = STATE.active_profile
    if not profile_name:
        return []
    profile = load_profiles().get(profile_name)
    if not profile:
        return []
    schema = str(profile.get("source_schema", "AI_Source"))
    if not _SCHEMA_RE.fullmatch(schema):
        raise RuntimeError("Invalid PostgreSQL source schema")
    table = "Players" if entity == "player" else "Teams"
    columns = (
        "id, global_id, name, position, current_league_key"
        if entity == "player"
        else "id, global_id, name, short_name, league_id"
    )
    search = str(query or "").strip()
    max_rows = max(1, min(int(limit), 20))
    if search:
        statement = text(
            f'SELECT {columns} FROM "{schema}"."{table}" '
            "WHERE lower(coalesce(name, '')) LIKE lower(:query) "
            "OR lower(coalesce(short_name, '')) LIKE lower(:query) "
            "ORDER BY CASE WHEN lower(coalesce(name, '')) = lower(:exact) THEN 0 "
            "WHEN lower(coalesce(name, '')) LIKE lower(:prefix) THEN 1 ELSE 2 END, "
            "name NULLS LAST LIMIT :limit"
        )
        params = {"query": f"%{search}%", "exact": search, "prefix": f"{search}%", "limit": max_rows}
    else:
        statement = text(
            f'SELECT {columns} FROM "{schema}"."{table}" '
            "ORDER BY name NULLS LAST LIMIT :limit"
        )
        params = {"limit": max_rows}
    with get_engine(profile_url(profile_name)).connect() as connection:
        rows = connection.execute(statement, params).mappings().all()
    records = []
    for row in rows:
        internal_id = row.get("id")
        if internal_id in (None, ""):
            continue
        name = str(row.get("name") or internal_id).strip()
        if entity == "player":
            values = (row.get("position"), row.get("current_league_key"))
        else:
            values = (row.get("short_name"),)
        subtitle = " · ".join(
            str(value).strip() for value in values
            if value not in (None, "") and str(value).strip() != name
        )
        records.append({
            "selection_id": str(internal_id),
            "name": name,
            "subtitle": subtitle,
            "identity_status": "canonical" if row.get("global_id") not in (None, "") else "unreconciled",
        })
    return records


def _resolve_scenario_selections(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate opaque admin selections to canonical global IDs server-side."""
    resolved = dict(payload)
    with STATE.lock:
        profile_name = STATE.active_profile
    if not profile_name:
        raise RuntimeError("Seleziona e carica prima un profilo database")
    profile = load_profiles().get(profile_name)
    if not profile:
        raise RuntimeError("Profilo database attivo non disponibile")
    schema = str(profile.get("source_schema", "AI_Source"))
    if not _SCHEMA_RE.fullmatch(schema):
        raise RuntimeError("Invalid PostgreSQL source schema")
    with get_engine(profile_url(profile_name)).connect() as connection:
        for entity, selection_key, global_key, table in (
            ("player", "player_selection_ids", "player_global_ids", "Players"),
            ("team", "team_selection_ids", "team_global_ids", "Teams"),
        ):
            selections = [str(value).strip() for value in resolved.pop(selection_key, []) if str(value).strip()]
            if not selections:
                continue
            placeholders, params = [], {}
            for index, value in enumerate(selections):
                key = f"id_{index}"
                placeholders.append(f":{key}")
                params[key] = value
            rows = connection.execute(
                text(
                    f'SELECT id, global_id FROM "{schema}"."{table}" '
                    f'WHERE id::text IN ({", ".join(placeholders)})'
                ),
                params,
            ).mappings().all()
            by_id = {
                str(row["id"]): str(row["global_id"])
                for row in rows
                if row.get("global_id") not in (None, "")
            }
            missing = [value for value in selections if value not in by_id]
            if missing:
                raise RuntimeError(
                    f"{entity} selezionato non ha ancora un GlobalId riconciliato: {missing[0]}"
                )
            resolved[global_key] = [by_id[value] for value in selections]
    return resolved

def _evaluate_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    """Forward an authenticated admin request to the API scenario engine."""
    api_key = os.environ.get("API_KEY", "").strip()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = URLRequest(
        f"{API_BASE_URL}/api/v2/scenarios/evaluate",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=45.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"Scenario API: {detail}") from exc
    except URLError as exc:
        raise RuntimeError("Scenario API non raggiungibile") from exc


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if (
        request.url.path == "/"
        or request.url.path.startswith("/admin-api/")
        or request.url.path.startswith("/assets/")
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
def startup() -> None:
    ensure_default_admin()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/admin-api/login")
def login(payload: LoginPayload) -> Response:
    ok, user = check_credentials(payload.username, payload.password)
    if not ok:
        raise HTTPException(status_code=401, detail="Credenziali non valide")
    username = payload.username.strip().lower()
    role = str(user.get("role", "viewer"))
    token = create_session_token(username, role, ttl_days=COOKIE_TTL_DAYS)
    csrf = secrets.token_urlsafe(32)
    response = JSONResponse({"authenticated": True, "user": {"username": username, "role": role}})
    max_age = COOKIE_TTL_DAYS * 86400
    response.set_cookie(SESSION_COOKIE, token, max_age=max_age, httponly=True, secure=COOKIE_SECURE, samesite="strict", path="/")
    response.set_cookie(CSRF_COOKIE, csrf, max_age=max_age, httponly=False, secure=COOKIE_SECURE, samesite="strict", path="/")
    return response


@app.get("/admin-api/session")
def session(user: dict[str, str] = Depends(_user_from_request)) -> dict[str, Any]:
    return {"authenticated": True, "user": {"username": user["username"], "role": user["role"]}}


@app.post("/admin-api/logout")
def logout(request: Request, user: dict[str, str] = Depends(_user_from_request)) -> Response:
    revoke_session_token(user["token"])
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    return response


@app.get("/admin-api/overview")
async def overview(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    registry = _registry()
    with STATE.lock:
        data_loaded = STATE.data is not None
        summary = dict(STATE.summary)
        active_profile = STATE.active_profile
    seasons = summary.get("seasons", [])
    production = registry.get("production") or {}
    candidate = registry.get("candidate") or {}
    api_live = await asyncio.to_thread(_probe, "/health/live")
    readiness = [
        {
            "component": "PostgreSQL source",
            "status": "ok" if data_loaded else "warning",
            "label": "OK" if data_loaded else "Da verificare",
            "detail": active_profile or "Seleziona e carica un profilo",
            "action": "data",
        },
        {
            "component": "Competition context",
            "status": "ok" if data_loaded and summary.get("team_rows", 0) > 0 else "warning",
            "label": "OK" if data_loaded and summary.get("team_rows", 0) > 0 else "Da verificare",
            "detail": '"AI_Source"."PlayerCompetitionStats" + "TeamCompetitionStats"',
            "action": "data",
        },
        {
            "component": "Training dataset",
            "status": "ok" if len(seasons) >= 5 else "warning",
            "label": "OK" if len(seasons) >= 5 else "Warning",
            "detail": f"{len(seasons)} stagioni disponibili (minimo richiesto: 5)",
            "action": "training",
        },
        {
            "component": "Production model",
            "status": "ok" if production else "danger",
            "label": "Production" if production else "Not promoted",
            "detail": production.get("run_id") or "Nessun modello in produzione",
            "action": "registry",
        },
    ]
    return {
        "utc": _utcnow(),
        "database": active_profile or "Not selected",
        "data_loaded": data_loaded,
        "summary": summary,
        "production": production,
        "candidate": candidate,
        "candidate_quality": _candidate_quality(candidate),
        "readiness": readiness,
        "api": {"online": bool(api_live.get("ok")), "version": "V2", "detail": api_live},
    }


@app.get("/admin-api/profiles")
def profiles(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    values = load_profiles()
    masked = {
        name: {key: ("••••••••" if key == "password" else value) for key, value in profile.items()}
        for name, profile in values.items()
    }
    with STATE.lock:
        active = STATE.active_profile
    return {"profiles": masked, "active": active}


@app.post("/admin-api/profiles")
def save_database_profile(payload: ProfilePayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    existing = load_profiles().get(payload.name, {})
    password = payload.password or str(existing.get("password", ""))
    if not password:
        raise HTTPException(status_code=400, detail="Password PostgreSQL richiesta")
    save_profile(
        payload.name,
        {
            "host": payload.host,
            "port": payload.port,
            "database": payload.database,
            "user": payload.user,
            "password": password,
            "source_schema": payload.source_schema,
            "ai_schema": payload.ai_schema,
        },
    )
    return {"ok": True, "profile": payload.name, "actor": user["username"]}


@app.post("/admin-api/profiles/{name}/test")
async def test_database_profile(name: str, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    del user
    if name not in load_profiles():
        raise HTTPException(status_code=404, detail="Profilo non trovato")

    def test() -> None:
        with get_engine(profile_url(name)).connect() as connection:
            connection.exec_driver_sql("SELECT 1")

    try:
        await asyncio.to_thread(test)
        return {"ok": True, "message": "Connessione PostgreSQL riuscita"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc


@app.post("/admin-api/profiles/{name}/load")
async def load_database_profile(name: str, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    del user
    profiles_map = load_profiles()
    profile = profiles_map.get(name)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profilo non trovato")
    try:
        # The admin load step prepares the core forecasting dataset.  Simulation
        # feeds contain possession-level data and can be orders of magnitude
        # larger; they are deliberately loaded only by consumers that need them.
        loaded = await asyncio.to_thread(
            load_all_data,
            profile_url(name),
            profile.get("source_schema", "AI_Source"),
            include_optional=False,
            purpose="training",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc
    summary = _summary(loaded)
    snapshot = await asyncio.to_thread(
        create_training_snapshot,
        loaded,
        profile=name,
    )
    with STATE.lock:
        STATE.data = loaded
        STATE.active_profile = name
        STATE.summary = summary
        STATE.snapshot = snapshot
    return {"ok": True, "profile": name, "summary": summary, "snapshot": snapshot}


@app.get("/admin-api/snapshots")
def snapshots(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    with STATE.lock:
        active = dict(STATE.snapshot) if STATE.snapshot else None
    return {"active": active, "snapshots": list_training_snapshots()}


@app.get("/admin-api/season-lifecycle")
def season_lifecycle(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    with STATE.lock:
        seasons = list(STATE.summary.get("seasons", []))
    return {"seasons": _lifecycle_rows(seasons)}


@app.post("/admin-api/season-lifecycle")
def update_season_lifecycle(
    payload: SeasonLifecyclePayload, user: dict[str, str] = Depends(_csrf)
) -> dict[str, Any]:
    values = _season_lifecycle()
    values[str(payload.season)] = {"status": payload.status, "updated_at": _utcnow(), "updated_by": user["username"]}
    _save_season_lifecycle(values)
    return {"ok": True, "season": payload.season, "status": payload.status}


@app.post("/admin-api/snapshots/{snapshot_id}/activate")
async def activate_snapshot(
    snapshot_id: str, user: dict[str, str] = Depends(_csrf)
) -> dict[str, Any]:
    del user
    try:
        loaded = await asyncio.to_thread(load_training_snapshot, snapshot_id)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc
    summary = _summary(loaded)
    manifest = next(
        (item for item in list_training_snapshots() if item.get("snapshot_id") == snapshot_id),
        {"snapshot_id": snapshot_id},
    )
    with STATE.lock:
        STATE.data = loaded
        STATE.active_profile = str(manifest.get("profile") or STATE.active_profile or "snapshot")
        STATE.summary = summary
        STATE.snapshot = manifest
    return {"ok": True, "profile": STATE.active_profile, "summary": summary, "snapshot": manifest}


@app.get("/admin-api/training")
def training(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    with STATE.lock:
        summary = dict(STATE.summary)
        job = dict(STATE.training)
        active = STATE.active_profile
        snapshot = dict(STATE.snapshot) if STATE.snapshot else None
    seasons = summary.get("seasons", [])
    competitions = summary.get("competitions", [])
    lifecycle = _lifecycle_rows(seasons)
    can_train_snapshot = bool(lifecycle) and all(item["status"] == "complete" for item in lifecycle)
    return {
        "profile": active,
        "snapshot": snapshot,
        "summary": summary,
        "job": job,
        "can_start": bool(STATE.data is not None and len(seasons) >= 5 and can_train_snapshot and job.get("status") != "running"),
        "lifecycle": lifecycle,
        "contract": [
            {"stage": "Fit", "seasons": f"through {seasons[-2]}" if len(seasons) >= 2 else "—", "purpose": "Same player + league + competition, exact t → t+1"},
            {"stage": "Calibration", "seasons": str(seasons[-1]) if seasons else "—", "purpose": "Final ensemble + per-competition intervals"},
            {"stage": "Backtest", "seasons": "expanding OOT folds", "purpose": "Exact production ensemble, including by_competition"},
            {"stage": "Promotion", "seasons": "all OOT folds", "purpose": "Global + league + competition regression gates"},
        ],
        "configuration": {
            "forecast_horizon": "exactly t+1 season",
            "feature_version": "forecast-t-plus-1-persistence-delta-v1",
            "pairing": "same player + league + competition",
            "split": "whole target seasons",
            "calibration": "global + competition split-conformal",
            "observed_competitions": competitions,
            "random_seed": 42,
        },
    }


@app.post("/admin-api/training/start")
def start_training(user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    with STATE.lock:
        if STATE.data is None:
            raise HTTPException(status_code=400, detail="Carica prima un profilo database")
        seasons = _seasons(STATE.data)
        if len(seasons) < 5:
            raise HTTPException(status_code=400, detail="Servono almeno cinque stagioni")
        blocked = [row["season"] for row in _lifecycle_rows(seasons) if row["status"] != "complete"]
        if blocked:
            raise HTTPException(
                status_code=400,
                detail="Classifica come Complete le stagioni presenti nello snapshot prima del training: "
                + ", ".join(map(str, blocked)),
            )
        if STATE.training.get("status") == "running":
            raise HTTPException(status_code=409, detail="Un training è già in corso")
        data = STATE.data
        active = STATE.active_profile
        snapshot = dict(STATE.snapshot) if STATE.snapshot else None
        STATE.training = {
            "status": "running",
            "stage": "Queued",
            "progress": 2,
            "message": "Training queued…",
            "run_id": None,
            "error": None,
            "started_at": _utcnow(),
            "finished_at": None,
            "actor": user["username"],
        }
    STATE.executor.submit(_training_worker, data, active, snapshot)
    return {"ok": True, "job": dict(STATE.training)}


@app.get("/admin-api/backtests")
def backtests(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    candidate = _registry().get("candidate") or {}
    report = candidate.get("backtest") or {}
    return {"candidate": candidate, "report": report, "quality": _candidate_quality(candidate)}


@app.get("/admin-api/scenario-entities")
def scenario_entities(
    entity: str = "player",
    q: str = "",
    limit: int = 12,
    user: dict[str, str] = Depends(_operator),
) -> dict[str, Any]:
    del user
    entity = entity.strip().lower()
    if entity not in {"player", "team"}:
        raise HTTPException(status_code=400, detail="entity must be player or team")
    with STATE.lock:
        loaded = STATE.data is not None
    if not loaded:
        raise HTTPException(status_code=409, detail="Carica prima il database nella pagina Dati e snapshot")
    return {"entity": entity, "items": _scenario_entity_rows(entity, q, limit)}


@app.post("/admin-api/scenarios/evaluate")
async def evaluate_scenario(payload: dict[str, Any], user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    """Expose the production scenario engine in the authenticated admin UI."""
    del user
    try:
        resolved_payload = _resolve_scenario_selections(payload)
        return await asyncio.to_thread(_evaluate_scenario, resolved_payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from exc


@app.get("/admin-api/registry")
def registry(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    return _registry()


@app.post("/admin-api/registry/promote")
def promote(payload: ConfirmPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    del user
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Conferma richiesta")
    result = promote_if_better(str(MODEL_ROOT))
    if not result.get("promoted"):
        raise HTTPException(status_code=409, detail=result.get("reason", "Promotion rejected"))
    return result


@app.post("/admin-api/registry/rollback")
def rollback(payload: ConfirmPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    del user
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Conferma richiesta")
    result = rollback_to_previous(str(MODEL_ROOT))
    if not result.get("rolled_back"):
        raise HTTPException(status_code=409, detail=result.get("reason", "Rollback unavailable"))
    return result


@app.get("/admin-api/api-health")
async def api_health(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    live, ready = await asyncio.gather(
        asyncio.to_thread(_probe, "/health/live"),
        asyncio.to_thread(_probe, "/health/ready"),
    )
    return {
        "base_url": API_BASE_URL,
        "live": live,
        "ready": ready,
        "endpoints": [
            "GET /health/live",
            "GET /health/ready",
            "POST /api/v2/predictions/player-team",
            "POST /api/v2/scenarios/evaluate",
        ],
    }


@app.get("/admin-api/audit")
def audit(limit: int = 100, user: dict[str, str] = Depends(_admin)) -> dict[str, Any]:
    del user
    return {"entries": _audit_rows(max(1, min(limit, 500)))}


@app.get("/admin-api/settings")
def settings(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    return {
        "user": {"username": user["username"], "role": user["role"]},
        "runtime": {
            "model_root": str(MODEL_ROOT),
            "api_base_url": API_BASE_URL,
            "database_profile": STATE.active_profile,
            "cookie_secure": COOKIE_SECURE,
            "admin_credentials_file": str(ADMIN_CREDENTIALS_FILE),
        },
        "promotion_gates": {
            "min_improvement_pct": PROMOTION_THRESHOLD_PCT,
            "max_segment_regression_pct": MAX_SEGMENT_REGRESSION_PCT,
            "min_backtest_samples": MIN_BACKTEST_SAMPLES,
            "min_competition_samples": MIN_COMPETITION_SAMPLES,
            "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
        },
    }


@app.post("/admin-api/settings/password")
def update_password(payload: PasswordPayload, request: Request, user: dict[str, str] = Depends(_csrf)) -> Response:
    try:
        changed = change_password(user["username"], payload.old_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not changed:
        raise HTTPException(status_code=400, detail="Password attuale non valida")
    revoke_user_sessions(user["username"])
    response = JSONResponse({"ok": True, "reauthenticate": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    return response

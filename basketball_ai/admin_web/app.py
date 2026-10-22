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
import re
import shutil
from pathlib import Path
import secrets
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from basketball_ai.auth.auth import (
    ADMIN_CREDENTIALS_FILE,
    AUDIT_LOG_FILE,
    _audit,
    change_password,
    check_credentials,
    create_session_token,
    ensure_default_admin,
    revoke_session_token,
    revoke_user_sessions,
    validate_session_token,
)
from basketball_ai.data.connection_profiles import load_profiles, profile_url, save_profile
from basketball_ai.data.loader import _to_int
from basketball_ai.data.postgres_loader import _source_object_diagnostics
from basketball_ai.data.postgres_loader import get_engine, load_all_data
from basketball_ai.data.training_snapshots import (
    create_training_snapshot,
    list_training_snapshots,
    load_training_snapshot,
)
from basketball_ai.models.production_training import _numeric_seasons
from basketball_ai.admin_web.model_lifecycle import (
    cleanup_incomplete_candidates,
    inspect_model,
    promote_model,
    rollback_model,
)
from basketball_ai.bb_rating.engine import BBRatingEngine
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine
from basketball_ai.models.strict_production import StrictProductionEnsembleModel
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

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    bb_rating_prediction_job: dict[str, Any] = field(default_factory=lambda: {"status": "idle", "stage": "Ready", "progress": 0, "message": "No BB Prediction Model run in progress.", "error": None, "started_at": None, "finished_at": None, "output": None})
    bb_rating_prediction_job_executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-admin-bb-rating-prediction"))
    bb_rating: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "idle",
            "stage": "Ready",
            "progress": 0,
            "message": "No BB-Rating calibration run in progress.",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "calibration_version": None,
            "bb_rating_version": None,
            "output": None,
        }
    )
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancel_events: dict[str, threading.Event] = field(default_factory=dict)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-admin-training"))
    bb_rating_executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-admin-bb-rating"))
    future_performance: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "idle",
            "stage": "Ready",
            "progress": 0,
            "message": "No Future Performance training run in progress.",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "future_performance_version": None,
            "feature_version": None,
            "output": None,
        }
    )
    future_performance_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ai-admin-future-performance",
        )
    )


STATE = AdminState()

CAUSAL_REFRESH_LOCK = threading.RLock()
CAUSAL_REFRESH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-admin-causal-refresh")
CAUSAL_REFRESH_JOB: dict[str, Any] = {
    "status": "idle",
    "kind": None,
    "stage": "Ready",
    "progress": 0,
    "message": "Nessun refresh causale eseguito.",
    "results": [],
    "summary": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}

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
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)


class PasswordPayload(BaseModel):
    old_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=1, max_length=512)


class SeasonLifecyclePayload(BaseModel):
    season: int = Field(ge=2000, le=2100)
    status: str = Field(pattern="^(in_progress|complete|locked)$")

class DiagnosticPayload(BaseModel):
    model: str = Field(pattern="^(bb_rating_prediction|bb_rating|future_performance|metric_rating)$")
    player_global_id: str = Field(min_length=1, max_length=128)
    team_global_id: str = Field(default="", max_length=128)
    league: str = Field(default="ITA1", min_length=1, max_length=64)
    season: int = Field(default=2025, ge=2000, le=2100)
    competition: str = Field(default="RS", min_length=1, max_length=32)
    metrics: list[str] = Field(default_factory=list, max_length=30)


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


def _training_seasons(seasons: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return complete seasons for training, current seasons to exclude, and blocking seasons."""
    lifecycle = {row["season"]: row["status"] for row in _lifecycle_rows(seasons)}
    training_seasons = sorted(
        season for season in seasons if lifecycle.get(season) == "complete"
    )
    excluded_in_progress = sorted(
        season for season in seasons if lifecycle.get(season) == "in_progress"
    )
    blocking = sorted(
        season
        for season in seasons
        if lifecycle.get(season) not in {"complete", "in_progress"}
    )
    return training_seasons, excluded_in_progress, blocking


def _future_performance_training_seasons(
    seasons: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Select eligible Future Performance target years from observed season data.

    Future Performance learns only consecutive t -> t+1 pairs within each player,
    league and competition. At the dataset level, a year can be a target only
    when the immediately preceding year is also observed. This excludes the
    earliest season, which can supply source features but cannot be a target.

    Older observed seasons labelled in_progress by a stale lifecycle file, or
    left unclassified because the lifecycle registry was introduced later, are
    historical once a later season is present and remain eligible for pair
    construction. The latest observed season is excluded as a target when it is
    in_progress or unclassified. Explicit locked seasons and invalid lifecycle
    values remain blocking; those are deliberate controls and must not be
    bypassed.
    """
    observed = sorted({int(season) for season in seasons})
    if not observed:
        return [], [], []

    observed_set = set(observed)
    candidate_targets = {season for season in observed if season - 1 in observed_set}
    values = _season_lifecycle()
    latest_observed = observed[-1]
    lifecycle = {
        season: str(values.get(str(season), {}).get("status", "unclassified")).strip().lower()
        for season in observed
    }

    training_target_seasons: list[int] = []
    excluded_latest: list[int] = []
    blocking: list[int] = []

    for season in observed:
        status = lifecycle[season]
        if status == "locked":
            blocking.append(season)
        elif status == "complete":
            if season in candidate_targets:
                training_target_seasons.append(season)
        elif status in {"in_progress", "unclassified"}:
            if season < latest_observed:
                if season in candidate_targets:
                    # A later observed year proves this is no longer the active
                    # frontier; exact training pairs are still checked within
                    # each player/league/competition by _build_pairs.
                    training_target_seasons.append(season)
            elif season in candidate_targets:
                excluded_latest.append(season)
        else:
            blocking.append(season)

    return training_target_seasons, excluded_latest, blocking

def _filter_training_data(data: dict[str, Any]) -> dict[str, Any]:
    """Exclude in-progress seasons from model fitting/backtesting while preserving them in snapshots."""
    seasons = _seasons(data)
    training_seasons, _, _ = _training_seasons(seasons)
    allowed = set(training_seasons)
    filtered = dict(data)
    for key, frame in data.items():
        if not isinstance(frame, pd.DataFrame) or "season" not in frame.columns:
            continue
        years = pd.to_numeric(
            frame["season"].astype(str).str.split("-").str[0],
            errors="coerce",
        )
        filtered[key] = frame.loc[years.isin(allowed)].copy()
    return filtered


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


def _set_bb_rating(**updates: Any) -> None:
    with STATE.lock:
        STATE.bb_rating.update(updates)


def _set_future_performance(**updates: Any) -> None:
    with STATE.lock:
        STATE.future_performance.update(updates)


class TrainingCancelled(RuntimeError):
    """Raised when an operator requests cancellation at a safe checkpoint."""


_JOB_STATE_ATTRIBUTES = {
    "training": "training",
    "bb-rating": "bb_rating",
    "future-performance": "future_performance",
    "bb-rating-prediction": "bb_rating_prediction_job",
}


def _new_cancel_event(job_name: str) -> threading.Event:
    if job_name not in _JOB_STATE_ATTRIBUTES:
        raise ValueError(f"Unsupported cancellable job: {job_name}")
    event = threading.Event()
    with STATE.lock:
        STATE.cancel_events[job_name] = event
    return event


def _check_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise TrainingCancelled("Cancellation requested by operator")


def _mark_job_cancelled(job_name: str, event: threading.Event) -> None:
    with STATE.lock:
        job = getattr(STATE, _JOB_STATE_ATTRIBUTES[job_name])
        job.update(
            status="cancelled",
            stage="Cancelled",
            progress=100,
            message="Training interrupted by operator; production was not changed.",
            error=None,
            cancel_requested=False,
            finished_at=_utcnow(),
        )
    event.clear()


def _future_performance_path() -> Path:
    return MODEL_ROOT / "future_performance"


def _bb_rating_paths() -> dict[str, Path]:
    root = MODEL_ROOT / "bb_rating_calibration"
    uncertainty = root / "bb_rating_uncertainty.json"
    if not uncertainty.exists():
        uncertainty = MODEL_ROOT / "bb_rating_uncertainty.json"
    return {
        "root": root,
        "report": root / "bb_rating_calibration.json",
        "markdown": root / "bb_rating_calibration.md",
        "uncertainty": uncertainty,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _file_modified_at(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _bb_rating_status() -> dict[str, Any]:
    paths = _bb_rating_paths()
    report_path = paths["report"]
    uncertainty_path = paths["uncertainty"]
    report = _read_json(report_path) if report_path.exists() else None
    uncertainty = _read_json(uncertainty_path) if uncertainty_path.exists() else None
    # A missing or malformed optional uncertainty artifact must not crash
    # health/status endpoints; report the release as incomplete instead.
    report = report if isinstance(report, dict) else {}
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    fitted = report.get("uncertainty_calibration") or {}
    dataset = report.get("dataset") or {}
    validation = report.get("validation_signals") or {}
    support = fitted.get("support") or {}
    ready = bool(
        bool(report)
        and bool(uncertainty)
        and report.get("calibration_version")
        and report.get("bb_rating_version")
        and uncertainty.get("status")
    )
    with STATE.lock:
        job = dict(STATE.bb_rating)
        active_profile = STATE.active_profile
    return {
        "ready": ready,
        "status": "ready" if ready else ("artifact_missing" if not report and not uncertainty else "incomplete"),
        "calibration_version": report.get("calibration_version") if report else None,
        "bb_rating_version": report.get("bb_rating_version") if report else None,
        "dataset": {
            "rows": dataset.get("rows"),
            "players": dataset.get("players"),
            "leagues": dataset.get("leagues") or [],
            "seasons": dataset.get("seasons") or [],
            "season_min": dataset.get("season_min"),
            "season_max": dataset.get("season_max"),
            "competitions": dataset.get("competitions") or [],
            "source_contract": dataset.get("source_contract"),
        },
        "configuration": (report or {}).get("configuration") or {},
        "validation": {
            "score_available_rate": validation.get("score_available_rate"),
            "primary_context_share": validation.get("primary_context_share"),
            "limited_context_share": validation.get("limited_context_share"),
            "explainable_metric_count": validation.get("explainable_metric_count"),
        },
        "uncertainty": {
            "status": fitted.get("status") or uncertainty.get("status"),
            "training_target_seasons": fitted.get("training_target_seasons") or uncertainty.get("training_target_seasons") or [],
            "excluded_target_seasons": fitted.get("excluded_target_seasons") or uncertainty.get("excluded_target_seasons") or [],
            "training_rows": fitted.get("training_rows") or uncertainty.get("training_rows"),
            "support": support,
            "selected_structure": ((report or {}).get("uncertainty_validation") or {}).get("selected_structure") or {},
            "oos_models": ((report or {}).get("uncertainty_validation") or {}).get("models") or [],
            "fallback_order": ["league+exposure", "league", "exposure", "global"],
        },
        "files": {
            "report": str(report_path) if report_path.exists() else None,
            "markdown": str(paths["markdown"]) if paths["markdown"].exists() else None,
            "uncertainty": str(uncertainty_path) if uncertainty_path.exists() else None,
            "report_modified_at": _file_modified_at(report_path) if report_path.exists() else None,
            "uncertainty_modified_at": _file_modified_at(uncertainty_path) if uncertainty_path.exists() else None,
        },
        "active_profile": active_profile,
        "job": job,
    }


def _future_performance_status() -> dict[str, Any]:
    root = _future_performance_path()
    metadata_path = root / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    models = metadata.get("model_files") if isinstance(metadata, dict) else {}
    artifacts_ok = bool(
        isinstance(metadata, dict)
        and metadata.get("status") == "fitted"
        and models
        and all((root / str(relative)).is_file() for relative in models.values())
    )
    backtest = (metadata or {}).get("backtest") or {}
    target_metrics = (metadata or {}).get("target_metrics") or {}
    uncertainty = (metadata or {}).get("uncertainty_by_target") or {}
    with STATE.lock:
        job = dict(STATE.future_performance)
        active_profile = STATE.active_profile
    return {
        "ready": artifacts_ok,
        "status": "ready" if artifacts_ok else ("artifact_missing" if not metadata else "incomplete"),
        "future_performance_version": (metadata or {}).get("future_performance_version"),
        "feature_version": (metadata or {}).get("feature_version"),
        "dataset": {
            "n_pairs": (metadata or {}).get("n_pairs"),
            "n_players": (metadata or {}).get("n_players"),
            "leagues": (metadata or {}).get("leagues") or [],
            "competitions": (metadata or {}).get("competitions") or [],
            "training_target_seasons": (metadata or {}).get("training_target_seasons") or [],
        },
        "targets": {
            key: {
                **(target_metrics.get(key) or {}),
                "uncertainty": uncertainty.get(key) or {},
            }
            for key in target_metrics
        },
        "validation": {
            "status": backtest.get("status"),
            "method": backtest.get("method"),
            "target_seasons": backtest.get("target_seasons") or [],
            "folds": backtest.get("folds") or [],
            "summary": backtest.get("summary") or {},
        },
        "files": {
            "root": str(root),
            "metadata": str(metadata_path) if metadata_path.exists() else None,
            "metadata_modified_at": _file_modified_at(metadata_path) if metadata_path.exists() else None,
        },
        "active_profile": active_profile,
        "job": job,
    }


def _future_performance_worker(active_profile: str | None, actor: str, cancel_event: threading.Event | None = None) -> None:
    from basketball_ai.future_performance import PlayerFuturePerformanceModel

    candidate_dir: Path | None = None
    try:
        _check_cancelled(cancel_event)
        if not active_profile:
            raise RuntimeError("Seleziona prima un profilo PostgreSQL")
        profile = load_profiles().get(active_profile)
        if not profile:
            raise RuntimeError(f"Profilo PostgreSQL '{active_profile}' non disponibile")

        _set_future_performance(
            status="running",
            stage="Loading training data",
            progress=8,
            message=f"Caricamento dati Future Performance dal profilo {active_profile}…",
            error=None,
            started_at=_utcnow(),
            finished_at=None,
        )
        _audit("future_performance_training_started", actor, active_profile)

        data = load_all_data(
            profile_url(active_profile),
            str(profile.get("source_schema", "AI_Source")),
            purpose="training",
        )
        _check_cancelled(cancel_event)
        stats = data.get("player_stats")
        if not isinstance(stats, pd.DataFrame) or stats.empty:
            raise RuntimeError("Nessun dato player_stats disponibile per Future Performance")

        observed_seasons = _seasons(data)
        training_target_seasons, excluded_latest, blocking = (
            _future_performance_training_seasons(observed_seasons)
        )
        if blocking:
            raise RuntimeError(
                "Dataset contiene stagioni bloccate o con stato lifecycle non valido: "
                + ", ".join(map(str, blocking))
            )
        if len(training_target_seasons) < 5:
            raise RuntimeError(
                "Servono almeno cinque stagioni target idonee per Future Performance. "
                f"Stagioni osservate: {observed_seasons}; "
                f"stagioni target idonee: {training_target_seasons}; "
                f"ultima stagione osservata esclusa: {excluded_latest}."
            )

        _set_future_performance(
            stage="Walk-forward + fitting",
            progress=25,
            message=(
                "Costruzione delle coppie t→t+1, validazione expanding OOS e fit multivariato…"
                + (
                    f" (stagione/i più recente/i {excluded_latest} esclusa/e come target)"
                    if excluded_latest
                    else ""
                )
            ),
        )
        _check_cancelled(cancel_event)
        model = PlayerFuturePerformanceModel()
        metadata = model.fit(
            data,
            target_seasons=training_target_seasons,
            backtest=True,
            cancel_check=lambda: _check_cancelled(cancel_event),
        )
        _check_cancelled(cancel_event)

        _set_future_performance(
            stage="Saving candidate artifact",
            progress=88,
            message="Scrittura candidate Future Performance; production resta invariato…",
            future_performance_version=metadata.get("future_performance_version"),
            feature_version=metadata.get("feature_version"),
        )
        candidate_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        root = MODEL_ROOT / "candidates" / "future-performance" / candidate_id
        candidate_dir = root
        root.mkdir(parents=True, exist_ok=False)
        _check_cancelled(cancel_event)
        paths = model.save(root)
        _check_cancelled(cancel_event)
        details = {
            "future_performance_version": metadata.get("future_performance_version"),
            "feature_version": metadata.get("feature_version"),
            "n_pairs": metadata.get("n_pairs"),
            "n_players": metadata.get("n_players"),
            "target_seasons": training_target_seasons,
            "excluded_latest_observed_seasons": excluded_latest,
        }
        _audit(
            "future_performance_training_completed",
            actor,
            active_profile,
            json.dumps(details, ensure_ascii=False),
        )
        _set_future_performance(
            status="complete",
            stage="Model ready",
            progress=100,
            message="Future Performance candidate pronto e validabile. Production non è stata modificata.",
            error=None,
            finished_at=_utcnow(),
            future_performance_version=metadata.get("future_performance_version"),
            feature_version=metadata.get("feature_version"),
            output=paths,
        )
    except TrainingCancelled:
        if candidate_dir is not None:
            shutil.rmtree(candidate_dir, ignore_errors=True)
        _audit("future_performance_training_cancelled", actor, active_profile or "", "Cancelled by operator")
        if cancel_event is not None:
            _mark_job_cancelled("future-performance", cancel_event)
        else:
            _set_future_performance(status="cancelled", stage="Cancelled", progress=100, message="Training cancelled; production unchanged.", finished_at=_utcnow())
    except Exception as exc:  # pragma: no cover
        error = _safe_error(exc)
        _audit("future_performance_training_failed", actor, active_profile or "", error)
        _set_future_performance(
            status="failed",
            stage="Failed",
            progress=100,
            message="Future Performance training failed.",
            error=error,
            finished_at=_utcnow(),
        )


def _bb_rating_worker(active_profile: str | None, actor: str, cancel_event: threading.Event | None = None) -> None:
    from basketball_ai.bb_rating.calibration import (
        BBRatingCalibrationConfig,
        build_calibration_report,
        write_calibration_report,
    )

    candidate_dir: Path | None = None
    try:
        _check_cancelled(cancel_event)
        if not active_profile:
            raise RuntimeError("Seleziona prima un profilo PostgreSQL")
        profile = load_profiles().get(active_profile)
        if not profile:
            raise RuntimeError(f"Profilo PostgreSQL '{active_profile}' non disponibile")
        _set_bb_rating(
            status="running",
            stage="Loading analysis data",
            progress=10,
            message=f"Caricamento dati BB-Rating dal profilo {active_profile}…",
            error=None,
            started_at=_utcnow(),
            finished_at=None,
        )
        _audit("bb_rating_calibration_started", actor, active_profile)
        data = load_all_data(
            profile_url(active_profile),
            str(profile.get("source_schema", "AI_Source")),
            purpose="analysis",
        )
        _check_cancelled(cancel_event)
        stats = data.get("player_stats")
        if not isinstance(stats, pd.DataFrame) or stats.empty:
            raise RuntimeError("Nessun dato player_stats disponibile per la calibrazione BB-Rating")
        _set_bb_rating(
            stage="Building calibration report",
            progress=42,
            message="Calcolo percentili, validazione OOS e struttura uncertainty…",
        )
        _check_cancelled(cancel_event)
        report = build_calibration_report(data, config=BBRatingCalibrationConfig())
        _check_cancelled(cancel_event)
        _set_bb_rating(
            stage="Writing serving artifact",
            progress=82,
            message="Scrittura report e artifact di serving…",
            calibration_version=report.get("calibration_version"),
            bb_rating_version=report.get("bb_rating_version"),
        )
        candidate_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        output_dir = MODEL_ROOT / "candidates" / "bb-rating" / candidate_id
        candidate_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        _check_cancelled(cancel_event)
        paths = write_calibration_report(report, str(output_dir))
        _check_cancelled(cancel_event)
        details = {
            "calibration_version": report.get("calibration_version"),
            "bb_rating_version": report.get("bb_rating_version"),
            "rows": (report.get("dataset") or {}).get("rows"),
            "uncertainty_status": (report.get("uncertainty_calibration") or {}).get("status"),
        }
        _audit("bb_rating_calibration_completed", actor, active_profile, json.dumps(details, ensure_ascii=False))
        _set_bb_rating(
            status="complete",
            stage="Calibration ready",
            progress=100,
            message="BB-Rating calibration candidate pronto; production resta invariata fino alla promozione.",
            error=None,
            finished_at=_utcnow(),
            calibration_version=report.get("calibration_version"),
            bb_rating_version=report.get("bb_rating_version"),
            output=paths,
        )
    except TrainingCancelled:
        if candidate_dir is not None:
            shutil.rmtree(candidate_dir, ignore_errors=True)
        _audit("bb_rating_calibration_cancelled", actor, active_profile or "", "Cancelled by operator")
        if cancel_event is not None:
            _mark_job_cancelled("bb-rating", cancel_event)
        else:
            _set_bb_rating(status="cancelled", stage="Cancelled", progress=100, message="Calibration cancelled; production unchanged.", finished_at=_utcnow())
    except Exception as exc:  # pragma: no cover
        error = _safe_error(exc)
        _audit("bb_rating_calibration_failed", actor, active_profile or "", error)
        _set_bb_rating(
            status="failed",
            stage="Failed",
            progress=100,
            message="BB-Rating calibration failed.",
            error=error,
            finished_at=_utcnow(),
        )


def _bb_rating_prediction_job_worker(active_profile: str | None, actor: str, cancel_event: threading.Event | None = None) -> None:
    from basketball_ai.admin_web.model_lifecycle import validate_artifact
    from basketball_ai.bb_rating.future_prediction import BBRatingFuturePredictionModel
    from basketball_ai.data.postgres_loader import load_all_data

    candidate_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output_dir = MODEL_ROOT / "candidates" / "bb-rating-prediction" / candidate_id
    try:
        _check_cancelled(cancel_event)
        if not active_profile:
            raise RuntimeError("Seleziona prima un profilo PostgreSQL")
        os.environ["DATABASE_PROFILE"] = active_profile
        with STATE.lock:
            STATE.bb_rating_prediction_job.update(status="running", stage="Loading canonical history", progress=8, message="Caricamento dello storico canonico completo per il BB Prediction Model…", error=None, started_at=_utcnow(), finished_at=None, output=None)
        _audit("bb_rating_prediction_training_started", actor, active_profile)
        data = load_all_data(purpose="analysis")
        _check_cancelled(cancel_event)
        with STATE.lock:
            STATE.bb_rating_prediction_job.update(stage="Walk-forward BB-Rating prediction", progress=25, message="Calcolo dei target BB-Rating ufficiali, degli scenari e della validazione walk-forward temporale…")
        model = BBRatingFuturePredictionModel(min_oof_samples=500)
        metadata = model.fit(data, include_latest_target_season=False)
        metadata["database_profile"] = active_profile
        snapshot_manifest = data.get("training_snapshot_manifest")
        metadata["training_snapshot_id"] = snapshot_manifest.get("snapshot_id") if isinstance(snapshot_manifest, dict) else None
        model.metadata.update(metadata)
        _check_cancelled(cancel_event)
        if not metadata.get("promotion_gate_passed", False):
            # The release gate evaluates the final serving policy, including safe
            # contextual fallbacks. Calibrated-only MAE is reported separately.
            mae = metadata.get(
                "oof_mae_after_scenario_fallback",
                metadata.get("oof_mae_after_calibration"),
            )
            mae_calibrated_model = metadata.get("oof_mae_after_calibration")
            baseline_mae = metadata.get("oof_baseline_mae")
            target_seasons = metadata.get("walk_forward_target_seasons") or []
            fold_count = int(metadata.get("walk_forward_fold_count") or len(target_seasons))
            regressions = metadata.get("segment_regressions") or []
            oof_support = metadata.get("oof_support") or {}
            unique_oof_players = oof_support.get("unique_players")
            minimum_oof_players = int(metadata.get("minimum_oof_unique_players") or 40)
            gate_failures: list[str] = []
            if fold_count < int(metadata.get("minimum_walk_forward_target_seasons") or 3):
                gate_failures.append(
                    f"Solo {fold_count} stagioni OOF; ne servono almeno "
                    f"{int(metadata.get('minimum_walk_forward_target_seasons') or 3)}"
                )
            if unique_oof_players is None:
                gate_failures.append("Conteggio dei giocatori OOF distinti assente")
            elif int(unique_oof_players) < minimum_oof_players:
                gate_failures.append(
                    f"Solo {int(unique_oof_players)} giocatori OOF distinti; ne servono almeno {minimum_oof_players}"
                )
            if int(metadata.get("duplicate_target_context_pairs") or 0) != 0:
                gate_failures.append("Sono presenti coppie duplicate per contesto target")
            if mae is None or baseline_mae is None:
                gate_failures.append("Metriche MAE modello/baseline non disponibili")
            elif float(mae) >= float(baseline_mae):
                gate_failures.append(
                    f"MAE modello {float(mae):.3f} non migliore del baseline "
                    f"{float(baseline_mae):.3f}"
                )
            if regressions:
                gate_failures.append(
                    f"{len(regressions)} gruppi supportati oltre la soglia di regressione"
                )
            if not gate_failures:
                gate_failures.append("Gate di promozione non superato; verificare il report completo")

            top_regressions = sorted(
                regressions,
                key=lambda row: float(row.get("mae") or 0) - float(row.get("baseline_mae") or 0),
                reverse=True,
            )[:10]
            failure_message = "Candidate non salvato. " + "; ".join(gate_failures) + "."
            diagnostics = {
                "candidate_id": candidate_id,
                "promotion_gate_passed": False,
                "gate_failures": gate_failures,
                "oof_samples": metadata.get("oof_samples"),
                "training_support": metadata.get("training_support") or {},
                "oof_support": oof_support,
                "calibration_support": metadata.get("calibration_support") or {},
                "duplicate_target_context_pairs": metadata.get("duplicate_target_context_pairs"),
                "unresolved_team_identity_rows": metadata.get("unresolved_team_identity_rows"),
                "mae": mae,
                "mae_calibrated_model": mae_calibrated_model,
                "mae_after_scenario_fallback": mae,
                "baseline_mae": baseline_mae,
                "rmse": metadata.get("oof_rmse_after_calibration"),
                "target_seasons": metadata.get("target_seasons") or [],
                "walk_forward_target_seasons": target_seasons,
                "walk_forward_fold_count": fold_count,
                "segment_regression_count": len(regressions),
                "segment_regressions": top_regressions,
                "under_supported_segment_count": len(metadata.get("under_supported_segments") or []),
                "scenario_metrics": metadata.get("scenario_metrics") or {},
            }
            _audit(
                "bb_rating_prediction_gate_failed",
                actor,
                active_profile,
                json.dumps(diagnostics, ensure_ascii=False, default=str),
            )
            with STATE.lock:
                STATE.bb_rating_prediction_job.update(
                    status="failed",
                    stage="Validation gates failed",
                    progress=100,
                    message="Walk-forward completato, ma i gate di qualità non sono stati superati.",
                    error=failure_message,
                    candidate_id=None,
                    output=diagnostics,
                    cancel_requested=False,
                    finished_at=_utcnow(),
                )
            return
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        artifact = model.save(output_dir)
        _check_cancelled(cancel_event)
        validation = validate_artifact("bb-rating-prediction", output_dir)
        _check_cancelled(cancel_event)
        if not validation.get("valid"):
            raise RuntimeError(validation.get("reason", "BB Prediction Model artifact validation failed"))
        metrics = metadata.get("overall_metrics") or metadata.get("metrics") or {}
        details = {
            "candidate_id": candidate_id,
            "oof_samples": metadata.get("oof_samples"),
            "training_support": metadata.get("training_support") or {},
            "oof_support": metadata.get("oof_support") or {},
            "calibration_support": metadata.get("calibration_support") or {},
            "duplicate_target_context_pairs": metadata.get("duplicate_target_context_pairs"),
            "unresolved_team_identity_rows": metadata.get("unresolved_team_identity_rows"),
            "mae": metadata.get(
                "oof_mae_after_scenario_fallback",
                metrics.get("mae", metadata.get("oof_mae_after_calibration")),
            ),
            "mae_calibrated_model": metadata.get("oof_mae_after_calibration"),
            "mae_after_scenario_fallback": metadata.get("oof_mae_after_scenario_fallback"),
            "rmse": metadata.get("oof_rmse_after_calibration", metrics.get("rmse")),
            "baseline_mae": metadata.get("oof_baseline_mae", metrics.get("baseline_mae")),
            "scenario_metrics": metadata.get("scenario_metrics") or {},
            "target_seasons": metadata.get("target_seasons") or [],
        }
        _audit("bb_rating_prediction_candidate_ready", actor, active_profile, json.dumps(details, ensure_ascii=False, default=str))
        with STATE.lock:
            STATE.bb_rating_prediction_job.update(status="complete", stage="Candidate validated", progress=100, message="BB Prediction Model validato come candidate. La produzione non è stata modificata.", error=None, finished_at=_utcnow(), candidate_id=candidate_id, output={**details, "path": str(output_dir), "validation": validation, "artifact": artifact})
    except TrainingCancelled:
        shutil.rmtree(output_dir, ignore_errors=True)
        _audit("bb_rating_prediction_training_cancelled", actor, active_profile or "", "Cancelled by operator")
        if cancel_event is not None:
            _mark_job_cancelled("bb-rating-prediction", cancel_event)
        else:
            with STATE.lock:
                STATE.bb_rating_prediction_job.update(status="cancelled", stage="Cancelled", progress=100, message="Training annullato; produzione invariata.", error=None, finished_at=_utcnow())
    except Exception as exc:
        error = _safe_error(exc)
        expected_artifact = output_dir / "bb_rating_future_model.joblib"
        if not expected_artifact.is_file():
            shutil.rmtree(output_dir, ignore_errors=True)
        _audit("bb_rating_prediction_training_failed", actor, active_profile or "", json.dumps({"candidate_id": candidate_id, "error": error}, ensure_ascii=False))
        with STATE.lock:
            STATE.bb_rating_prediction_job.update(status="failed", stage="Failed", progress=100, message="Training BB Prediction Model fallito; produzione invariata.", error=error, candidate_id=None, output=None, cancel_requested=False, finished_at=_utcnow())



def _training_worker(
    data: dict[str, Any],
    active_profile: str | None,
    snapshot: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    from basketball_ai.models.backtest import run_backtest
    from basketball_ai.models.promote import register_candidate
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    run_dir: Path | None = None
    try:
        _check_cancelled(cancel_event)
        if snapshot and snapshot.get("snapshot_id"):
            snapshot_data = load_training_snapshot(str(snapshot["snapshot_id"]))
            _check_cancelled(cancel_event)
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
        observed_seasons = _seasons(data)
        training_seasons, excluded_in_progress, blocking = _training_seasons(observed_seasons)
        if blocking:
            raise RuntimeError(
                "Snapshot contiene stagioni non idonee al training: "
                + ", ".join(map(str, blocking))
            )
        if len(training_seasons) < 5:
            raise RuntimeError("Servono almeno cinque stagioni Complete per il training")
        data = _filter_training_data(data)
        seasons = training_seasons
        _set_training(
            status="running",
            stage="Training ensemble",
            progress=12,
            message=(
                "Training the strict competition-aware ensemble…"
                + (
                    f" ({len(excluded_in_progress)} stagione/i In corso escluse)"
                    if excluded_in_progress
                    else ""
                )
            ),
        )
        model = StrictProductionEnsembleModel()
        metrics = model.train(data)
        _check_cancelled(cancel_event)

        _set_training(stage="Walk-forward backtest", progress=58, message="Running leakage-safe out-of-time folds…")
        # Use every eligible OOT target season by default. An explicit
        # MODEL_BACKTEST_FOLDS value can cap runtime for large histories.
        default_folds = max(1, len(seasons) - 4)
        try:
            configured_folds = int(os.getenv("MODEL_BACKTEST_FOLDS", str(default_folds)))
        except ValueError:
            configured_folds = default_folds
        n_backtest_folds = min(max(1, configured_folds), default_folds)
        _check_cancelled(cancel_event)
        report = run_backtest(
            data,
            n_folds=n_backtest_folds,
            output_path=str(MODEL_ROOT / "backtest_report.json"),
        )
        _check_cancelled(cancel_event)
        if not report.get("valid") or not report.get("folds"):
            raise RuntimeError("Backtest invalid; candidate cannot enter the registry")

        _set_training(stage="Saving immutable run", progress=88, message="Saving candidate artifacts and registry metadata…")
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        run_dir = MODEL_ROOT / "runs" / run_id
        _check_cancelled(cancel_event)
        metadata = {
            **(metrics or {}),
            "model_run_id": run_id,
            "model_version": "2.6.0",
            "feature_version": "forecast-t-plus-1-persistence-delta-v1",
            "data_cutoff": datetime.now(timezone.utc).date().isoformat(),
            "latest_observed_season": max(observed_seasons),
            "training_seasons": seasons,
            "excluded_in_progress_seasons": excluded_in_progress,
            "database_profile": active_profile,
            "training_snapshot_id": (snapshot or {}).get("snapshot_id"),
            "training_snapshot_sha256": (snapshot or {}).get("sha256"),
            "source_contract": data.get("source_contract", "competition-v2:canonical"),
            "backtest": report,
        }
        model.save(str(run_dir), metadata)
        _check_cancelled(cancel_event)
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
    except TrainingCancelled:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)
        if cancel_event is not None:
            _audit("prediction_training_cancelled", "system", active_profile or "", "Cancelled by operator")
            _mark_job_cancelled("training", cancel_event)
        else:
            _set_training(status="cancelled", stage="Cancelled", progress=100, message="Training cancelled; production unchanged.", error=None, finished_at=_utcnow())
    except Exception as exc:  # pragma: no cover - exercised by production failures
        _set_training(
            status="failed",
            stage="Failed",
            progress=100,
            message="Training failed.",
            error=_safe_error(exc),
            finished_at=_utcnow(),
        )



def _api_request(path: str, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    api_key = os.environ.get("API_KEY", "").strip()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = URLRequest(f"{API_BASE_URL}{path}", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started = datetime.now(timezone.utc)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"raw": raw[:5000]}
            elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
            return {"ok": 200 <= response.status < 300, "status": response.status, "latency_ms": round(elapsed_ms, 1), "body": body}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw[:5000]}
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        return {"ok": False, "status": exc.code, "latency_ms": round(elapsed_ms, 1), "body": body}
    except (URLError, TimeoutError, OSError) as exc:
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        return {"ok": False, "status": None, "latency_ms": round(elapsed_ms, 1), "body": None, "error": str(exc)[:500]}

def _diagnostic_request(payload: DiagnosticPayload) -> tuple[str, dict[str, Any]]:
    base = {"player_global_id": payload.player_global_id.strip(), "league": payload.league.strip().upper(), "season": payload.season}
    if payload.model == "bb_rating_prediction":
        return "/api/v2/bb-rating-prediction/player", {
            "player_global_id": payload.player_global_id.strip(),
            "source_season": payload.season,
            "source_league": payload.league.strip().upper(),
            "source_competition": payload.competition.strip().upper(),
            "target_league": payload.league.strip().upper(),
            "target_competition": payload.competition.strip().upper(),
        }
    if payload.model == "bb_rating":
        return "/api/v2/bb-rating/player", {**base, "phase": payload.competition.strip().upper(), "include_history": True, "history_limit": 8}
    if payload.model == "future_performance":
        return "/api/v2/future-performance/player", {**base, "competition": payload.competition.strip().upper()}
    metrics = [str(value).strip().upper() for value in payload.metrics if str(value).strip()]
    if not metrics:
        metrics = ["RAPTOR", "LEBRON", "VORP"]
    return "/api/v2/metric-rating/player-snapshot", {**base, "phase": payload.competition.strip().upper(), "metrics": metrics}

def _diagnostic_assertions(model: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    body = result.get("body") if isinstance(result.get("body"), dict) else {}
    checks = [{"label": "HTTP 2xx", "ok": bool(result.get("ok"))}]
    if model == "bb_rating":
        checks += [
            {"label": "BB-Rating 1–100", "ok": isinstance(body.get("bb_rating"), (int, float)) and 1 <= float(body["bb_rating"]) <= 100},
            {"label": "History presente", "ok": isinstance(body.get("history"), list)},
            {"label": "Uncertainty presente", "ok": isinstance(body.get("uncertainty"), dict)},
        ]
    elif model == "future_performance":
        checks += [
            {"label": "Targets presenti", "ok": isinstance(body.get("targets"), dict) and bool(body.get("targets"))},
            {"label": "Versione modello presente", "ok": bool(body.get("future_performance_version"))},
        ]
    elif model == "bb_rating_prediction":
        prediction = body.get("prediction") if isinstance(body.get("prediction"), dict) else {}
        score = prediction.get("predicted_bb_rating")
        low = prediction.get("confidence_low_100")
        high = prediction.get("confidence_high_100")
        interval_ok = isinstance(low, (int, float)) and isinstance(high, (int, float)) and 1 <= float(low) <= float(high) <= 100
        checks += [
            {"label": "BB-Rating previsto 1–100", "ok": isinstance(score, (int, float)) and 1 <= float(score) <= 100},
            {"label": "Intervallo di incertezza 1–100", "ok": interval_ok},
            {"label": "Scenario dichiarato", "ok": bool(prediction.get("scenario"))},
        ]
    else:
        checks.append({"label": "Snapshot metriche presente", "ok": isinstance(body.get("metrics"), (dict, list)) or isinstance(body.get("ratings"), (dict, list))})
    return checks

def _run_diagnostic(payload: DiagnosticPayload, actor: str) -> dict[str, Any]:
    path, request_payload = _diagnostic_request(payload)
    result = _api_request(path, request_payload)
    checks = _diagnostic_assertions(payload.model, result)
    overall = bool(result.get("ok")) and all(bool(check["ok"]) for check in checks)
    _audit("api_diagnostic_test", actor, payload.model)
    return {"model": payload.model, "endpoint": path, "request": request_payload, "result": result, "checks": checks, "overall_ok": overall}


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


def _restart_inference_api(timeout_seconds: float = 90.0) -> dict[str, Any]:
    """Request a controlled API-container restart and wait for readiness."""
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "API_KEY is not configured; automatic API restart is unavailable"}
    request = URLRequest(
        f"{API_BASE_URL}/internal/restart",
        data=b"{}",
        headers={"Accept": "application/json", "Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8.0) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
            if response.status != 202 or not body.get("accepted"):
                return {"ok": False, "error": "Inference API did not accept the restart request", "status": response.status}
    except Exception as exc:
        return {"ok": False, "error": f"Could not request API restart: {str(exc)[:300]}"}

    # Avoid mistaking the still-running old process for a successful restart.
    time.sleep(2.0)
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    last_probe: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_probe = _probe("/health/ready")
        if last_probe.get("ok"):
            return {"ok": True, "status": "ready", "health": last_probe.get("body")}
        time.sleep(1.0)
    return {
        "ok": False,
        "error": "Inference API did not become ready after the restart request",
        "last_health": last_probe,
    }


def _scenario_entity_rows(entity: str, query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search canonical entities with DB lookup and optional loaded-data enrichment.

    PostgreSQL/AI_Source is authoritative for the base player/team identity.
    Loaded control-center data is strictly additive enrichment and must never be
    required for the lookup to succeed.
    """
    with STATE.lock:
        profile_name = STATE.active_profile
        loaded = STATE.data

    max_rows = max(1, min(int(limit), 20))
    search = str(query or "").strip()
    rows: list[dict[str, Any]] = []

    if profile_name:
        profile = load_profiles().get(profile_name)
        if profile:
            schema = str(profile.get("source_schema", "AI_Source"))
            if not _SCHEMA_RE.fullmatch(schema):
                raise RuntimeError("Invalid PostgreSQL source schema")
            table = "Players" if entity == "player" else "Teams"

            # Keep the authoritative lookup limited to columns guaranteed by the
            # canonical AI_Source contract. Fields such as current_league_key,
            # current_team_id, short_name and league_key are enrichment-only.
            columns = (
                "id, global_id, name, position"
                if entity == "player"
                else "id, global_id, name, league_id"
            )
            if search:
                if entity == "player":
                    search_where = (
                        "lower(coalesce(name, '')) LIKE lower(:query) "
                        "OR lower(coalesce(CAST(global_id AS text), '')) LIKE lower(:query)"
                    )
                else:
                    search_where = (
                        "lower(coalesce(name, '')) LIKE lower(:query) "
                        "OR lower(coalesce(CAST(global_id AS text), '')) LIKE lower(:query)"
                    )
                statement = text(
                    f'SELECT {columns} FROM "{schema}"."{table}" '
                    f"WHERE {search_where} "
                    "ORDER BY CASE WHEN lower(coalesce(name, '')) = lower(:exact) THEN 0 "
                    "WHEN lower(coalesce(name, '')) LIKE lower(:prefix) THEN 1 ELSE 2 END, "
                    f"name NULLS LAST LIMIT {max_rows}"
                )
                params = {
                    "query": f"%{search}%",
                    "exact": search,
                    "prefix": f"{search}%",
                }
            else:
                statement = text(
                    f'SELECT {columns} FROM "{schema}"."{table}" '
                    "ORDER BY name NULLS LAST LIMIT :limit"
                )
                params = {"limit": max_rows}

            try:
                with get_engine(profile_url(profile_name)).connect() as connection:
                    rows = [dict(row) for row in connection.execute(statement, params).mappings().all()]
            except Exception as exc:
                engine = get_engine(profile_url(profile_name))
                try:
                    diagnostics = _source_object_diagnostics(engine, schema, [table])
                finally:
                    engine.dispose()
                current_user = diagnostics.get("current_user", "unknown")
                object_info = diagnostics.get("objects", {}).get(table, {})
                detail = (
                    f'Scenario entity lookup failed for "{schema}"."{table}" '
                    f'as PostgreSQL user "{current_user}": {str(exc).strip() or exc.__class__.__name__}; '
                    f'object_exists={bool(object_info.get("exists"))}; '
                    f'select_allowed={bool(object_info.get("select"))}'
                )
                raise RuntimeError(detail) from exc

    if not rows and isinstance(loaded, dict):
        table_key = "players" if entity == "player" else "teams"
        frame = loaded.get(table_key)
        if isinstance(frame, pd.DataFrame) and not frame.empty and "id" in frame.columns:
            work = frame.copy()

            def contains(column: str) -> pd.Series:
                if column not in work.columns:
                    return pd.Series(False, index=work.index)
                return work[column].fillna("").astype(str).str.contains(
                    search, case=False, regex=False, na=False
                )

            if search:
                match = contains("name") | contains("global_id")
                if entity == "team":
                    match = match | contains("short_name")
                work = work.loc[match].copy()

                def rank(value: Any) -> int:
                    value = str(value or "").strip().lower()
                    needle = search.lower()
                    if value == needle:
                        return 0
                    if value.startswith(needle):
                        return 1
                    return 2

                work["_search_rank"] = (
                    work["name"].map(rank) if "name" in work.columns else 2
                )
                work = work.sort_values(
                    ["_search_rank", "name"],
                    ascending=[True, True],
                    na_position="last",
                    kind="stable",
                )
            elif "name" in work.columns:
                work = work.sort_values("name", na_position="last", kind="stable")
            rows = work.head(max_rows).to_dict("records")

    if not rows:
        return []

    def value_text(source_row: Any, key: str) -> str:
        value = source_row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    team_global_by_id: dict[str, str] = {}
    team_name_by_id: dict[str, str] = {}
    if entity == "player" and isinstance(loaded, dict):
        teams_frame = loaded.get("teams")
        if isinstance(teams_frame, pd.DataFrame) and {"id", "global_id"}.issubset(teams_frame.columns):
            for team_row in teams_frame.to_dict("records"):
                raw_team_id = value_text(team_row, "id")
                numeric_team_id = _to_int(team_row.get("id")) if raw_team_id else None
                team_id = str(numeric_team_id) if numeric_team_id is not None else raw_team_id
                global_value = value_text(team_row, "global_id")
                name_value = value_text(team_row, "name")
                if team_id and global_value:
                    team_global_by_id[team_id] = global_value
                if team_id and name_value:
                    team_name_by_id[team_id] = name_value

    records: list[dict[str, Any]] = []
    loaded_entity_by_id: dict[str, dict[str, Any]] = {}
    loaded_entity_by_global_id: dict[str, dict[str, Any]] = {}
    loaded_entities_by_name: dict[str, list[dict[str, Any]]] = {}
    if isinstance(loaded, dict):
        entity_frame = loaded.get("players" if entity == "player" else "teams")
        if isinstance(entity_frame, pd.DataFrame) and "id" in entity_frame.columns:
            loaded_records = entity_frame.to_dict("records")
            loaded_entity_by_id = {
                value_text(item, "id"): item
                for item in loaded_records
                if value_text(item, "id")
            }
            loaded_entity_by_global_id = {
                value_text(item, "global_id"): item
                for item in loaded_records
                if value_text(item, "global_id")
            }
            for item in loaded_records:
                exact_name = " ".join(value_text(item, "name").casefold().split())
                if exact_name:
                    loaded_entities_by_name.setdefault(exact_name, []).append(item)
    # Name-only enrichment is permitted only for a unique exact-name match;
    # ambiguous names never borrow current-team metadata from another player.
    loaded_entity_by_unique_name = {
        name: matches[0]
        for name, matches in loaded_entities_by_name.items()
        if len(matches) == 1
    }
    for row in rows:
        # Canonical GlobalId is strongest, internal ID is a compatibility fallback,
        # and unique exact-name matching enriches metadata only (never identity).
        by_global_id = loaded_entity_by_global_id.get(value_text(row, "global_id"), {})
        by_internal_id = loaded_entity_by_id.get(value_text(row, "id"), {})
        exact_name = " ".join(value_text(row, "name").casefold().split())
        by_unique_name = loaded_entity_by_unique_name.get(exact_name, {})
        loaded_entity = by_global_id or by_internal_id or by_unique_name
        row = {**loaded_entity, **row}
        # SQL source contracts may expose these enrichment-only fields as NULL;
        # preserve trustworthy in-memory values instead of overwriting them.
        for metadata_key in ("current_team_id", "current_league_key", "short_name"):
            if not value_text(row, metadata_key) and value_text(loaded_entity, metadata_key):
                row[metadata_key] = loaded_entity[metadata_key]
        internal_id = value_text(row, "id")
        if not internal_id:
            continue
        global_id = value_text(row, "global_id")
        name = value_text(row, "name") or global_id or internal_id
        if entity == "player":
            subtitle_values = (value_text(row, "position"),)
        else:
            # Team autocomplete keeps the short-name label from the loaded
            # canonical snapshot while the full name remains the display name.
            subtitle_values = (value_text(row, "short_name"),)
        subtitle = " · ".join(value for value in subtitle_values if value and value != name)

        league_keys: list[str] = []
        contexts: list[dict[str, Any]] = []
        try:
            if isinstance(loaded, dict):
                if entity == "player":
                    stats_frame = loaded.get("player_stats")
                    required = {"player_id", "league_key", "season", "competition"}
                    if isinstance(stats_frame, pd.DataFrame) and required.issubset(stats_frame.columns):
                        target_id = _to_int(internal_id)
                        context_rows = stats_frame[
                            stats_frame["player_id"].map(_to_int) == target_id
                        ].copy()
                        if not context_rows.empty:
                            context_rows["_season_year"] = _numeric_seasons(context_rows)
                            context_rows = context_rows.dropna(subset=["_season_year"]).sort_values(
                                ["league_key", "_season_year", "competition"],
                                ascending=[True, False, True],
                                kind="stable",
                            )
                            for league_key, group in context_rows.groupby(
                                context_rows["league_key"].astype(str).str.strip().str.upper(),
                                sort=True,
                            ):
                                if not league_key:
                                    continue
                                latest = group.iloc[0]
                                team_id_text = value_text(latest, "team_id")
                                contexts.append(
                                    {
                                        "league_key": league_key,
                                        "season": int(latest["_season_year"]),
                                        "competition": value_text(latest, "competition").upper(),
                                        "team_global_id": team_global_by_id.get(team_id_text, ""),
                                        "team_name": team_name_by_id.get(team_id_text, ""),
                                    }
                                )
                            league_keys = [item["league_key"] for item in contexts]
                else:
                    team_history = loaded.get("team_season_stats")
                    required = {"global_id", "league_key", "season", "competition"}
                    if isinstance(team_history, pd.DataFrame) and required.issubset(team_history.columns):
                        context_rows = team_history[
                            team_history["global_id"].astype(str).str.strip() == global_id
                        ].copy()
                        if not context_rows.empty:
                            context_rows["_season_year"] = _numeric_seasons(context_rows)
                            context_rows = context_rows.dropna(subset=["_season_year"]).sort_values(
                                ["league_key", "_season_year"],
                                ascending=[True, False],
                                kind="stable",
                            )
                            for league_key, group in context_rows.groupby(
                                context_rows["league_key"].astype(str).str.strip().str.upper(),
                                sort=True,
                            ):
                                if not league_key:
                                    continue
                                latest = group.iloc[0]
                                contexts.append(
                                    {
                                        "league_key": league_key,
                                        "season": int(latest["_season_year"]),
                                        "competition": value_text(latest, "competition").upper(),
                                    }
                                )
                            league_keys = [item["league_key"] for item in contexts]
        except (KeyError, TypeError, ValueError, OverflowError):
            # Autocomplete must never fail because optional context enrichment is malformed.
            league_keys = []
            contexts = []

        current_league_key = ""
        current_team_global_id = ""
        current_team_name = ""
        if entity == "player":
            # Loaded player metadata is an authoritative current-context hint;
            # historical competition stats may not include the current season's
            # team relation (or may not carry team_id at all).
            metadata_league_key = value_text(row, "current_league_key").upper()
            raw_metadata_team_id = value_text(row, "current_team_id")
            numeric_metadata_team_id = _to_int(row.get("current_team_id")) if raw_metadata_team_id else None
            metadata_team_id = (
                str(numeric_metadata_team_id)
                if numeric_metadata_team_id is not None
                else raw_metadata_team_id
            )
            if contexts:
                latest_context = max(
                    contexts,
                    key=lambda item: (item.get("season", 0), item.get("league_key", "")),
                )
                current_league_key = metadata_league_key or str(latest_context.get("league_key") or "")
                current_team_global_id = str(latest_context.get("team_global_id") or "")
                current_team_name = str(latest_context.get("team_name") or "")
            else:
                current_league_key = metadata_league_key
            current_team_global_id = current_team_global_id or team_global_by_id.get(metadata_team_id, "")
            current_team_name = current_team_name or team_name_by_id.get(metadata_team_id, "")
            if current_league_key and current_league_key not in league_keys:
                league_keys.append(current_league_key)

        records.append(
            {
                "selection_id": internal_id,
                "global_id": global_id,
                "name": name,
                "subtitle": subtitle,
                "league_key": current_league_key if entity == "player" else "",
                "current_team_global_id": current_team_global_id,
                "current_team_name": current_team_name,
                "league_keys": league_keys,
                "contexts": contexts,
                "identity_status": "canonical" if global_id else "unreconciled",
            }
        )
    return records

def _resolve_scenario_selections(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate opaque admin selections to canonical global IDs server-side."""
    resolved = dict(payload)
    with STATE.lock:
        profile_name = STATE.active_profile
        loaded = STATE.data
    if not profile_name:
        if not isinstance(loaded, dict):
            raise RuntimeError("Seleziona e carica prima un profilo database")
        # Resolve selections against the already-loaded canonical snapshot.
        # This is a controlled fallback, not fuzzy matching: only exact internal
        # IDs with non-empty GlobalIds are accepted.
        for entity, selection_key, global_key, table_key in (
            ("player", "player_selection_ids", "player_global_ids", "players"),
            ("team", "team_selection_ids", "team_global_ids", "teams"),
        ):
            selections = [
                str(value).strip()
                for value in resolved.pop(selection_key, [])
                if str(value).strip()
            ]
            if not selections:
                continue
            frame = loaded.get(table_key)
            if not isinstance(frame, pd.DataFrame) or not {"id", "global_id"}.issubset(frame.columns):
                raise RuntimeError(f"Dataset {entity} non disponibile per risolvere la selezione")
            lookup = {
                str(row["id"]).strip(): str(row["global_id"]).strip()
                for row in frame.to_dict("records")
                if row.get("id") is not None and str(row.get("global_id") or "").strip()
            }
            missing = [value for value in selections if value not in lookup]
            if missing:
                raise RuntimeError(
                    f"{entity} selezionato non ha un GlobalId riconciliato: {missing[0]}"
                )
            resolved[global_key] = [lookup[value] for value in selections]
        return resolved
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
def index() -> HTMLResponse:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    marker = '<script src="/assets/future_performance_ui.js?v=2.5.1" defer></script>'
    if marker not in html:
        html = html.replace("</body>", marker + "\n</body>")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store"},
    )


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


def _api_bb_prediction_ready() -> bool:
    """Check only the direct BB Prediction release, never the retired 0-10 model."""
    response = _probe("/health")
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    return bool(response.get("ok") and body.get("bb_rating_future_prediction_loaded"))


@app.get("/admin-api/overview")
async def overview(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    with STATE.lock:
        data_loaded = STATE.data is not None
        summary = dict(STATE.summary)
        active_profile = STATE.active_profile
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
            "component": "BB Prediction Model",
            "status": "ok" if _api_bb_prediction_ready() else "warning",
            "label": "Ready" if _api_bb_prediction_ready() else "Candidate da validare/promuovere",
            "detail": "Previsione diretta del BB-Rating ufficiale 1–100",
            "action": "model-releases",
        },
        {
            "component": "Player Future Performance",
            "status": "ok" if _future_performance_status().get("ready") else "warning",
            "label": "Ready" if _future_performance_status().get("ready") else "Model da generare",
            "detail": (
                f"v{_future_performance_status().get('future_performance_version') or '—'} · "
                f"{(_future_performance_status().get('dataset') or {}).get('n_pairs') or 0} training pairs"
            ),
            "action": "future-performance",
        },
        {
            "component": "BB-Rating",
            "status": "ok" if _bb_rating_status().get("ready") else "warning",
            "label": "Ready" if _bb_rating_status().get("ready") else "Artifact da verificare",
            "detail": (
                f"v{_bb_rating_status().get('bb_rating_version') or '—'} · "
                f"calibration {_bb_rating_status().get('calibration_version') or '—'}"
            ),
            "action": "bb-rating",
        },
    ]
    return {
        "utc": _utcnow(),
        "database": active_profile or "Not selected",
        "data_loaded": data_loaded,
        "summary": summary,
        "bb_rating": _bb_rating_status(),
        "future_performance": _future_performance_status(),
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


@app.post("/admin-api/auto-load")
async def auto_load_database(user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    """Refresh core data and prepare a reproducible snapshot when the BO is opened."""
    profiles_map = load_profiles()
    with STATE.lock:
        active = STATE.active_profile

    configured = os.getenv("DATABASE_PROFILE", "").strip()
    if active not in profiles_map:
        active = configured if configured in profiles_map else None
    if active is None and len(profiles_map) == 1:
        active = next(iter(profiles_map))

    if active is None:
        if not profiles_map:
            message = "Nessun profilo PostgreSQL configurato. Crea un profilo in Dati e snapshot."
        else:
            message = (
                "Sono presenti più profili PostgreSQL e nessuno è selezionato. "
                "Imposta DATABASE_PROFILE oppure seleziona un profilo univoco."
            )
        return {"ok": False, "loaded": False, "message": message, "profiles_available": sorted(profiles_map)}

    result = await load_database_profile(active, user)
    return {
        **result,
        "loaded": True,
        "message": f"Dati aggiornati automaticamente dal database PostgreSQL '{active}'.",
    }


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
    training_seasons, excluded_in_progress, blocking = _training_seasons(seasons)
    training_data = _filter_training_data(STATE.data) if STATE.data is not None else None
    training_summary = _summary(training_data)
    can_train_snapshot = bool(lifecycle) and len(training_seasons) >= 5 and not blocking
    return {
        "profile": active,
        "snapshot": snapshot,
        "summary": summary,
        "training_summary": training_summary,
        "training_seasons": training_seasons,
        "excluded_in_progress_seasons": excluded_in_progress,
        "blocking_seasons": blocking,
        "job": job,
        "can_start": bool(STATE.data is not None and can_train_snapshot and job.get("status") != "running"),
        "lifecycle": lifecycle,
        "contract": [
            {"stage": "Fit", "seasons": f"through {training_seasons[-2]}" if len(training_seasons) >= 2 else "—", "purpose": "Same player + league + competition, exact t → t+1"},
            {"stage": "Calibration", "seasons": str(training_seasons[-1]) if training_seasons else "—", "purpose": "Final ensemble + per-competition intervals"},
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
        training_seasons, excluded_in_progress, blocked = _training_seasons(seasons)
        if len(training_seasons) < 5:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Servono almeno cinque stagioni Complete per il training"
                    + (
                        f"; le stagioni In corso escluse: {', '.join(map(str, excluded_in_progress))}"
                        if excluded_in_progress
                        else ""
                    )
                ),
            )
        if blocked:
            raise HTTPException(
                status_code=400,
                detail="Classifica come Complete le stagioni non idonee presenti nello snapshot prima del training: "
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
            "cancel_requested": False,
        }
    cancel_event = _new_cancel_event("training")
    STATE.executor.submit(_training_worker, data, active, snapshot, cancel_event)
    return {"ok": True, "job": dict(STATE.training)}


@app.post("/admin-api/jobs/{job_name}/cancel")
def cancel_training_job(
    job_name: str,
    user: dict[str, str] = Depends(_csrf),
) -> dict[str, Any]:
    del user
    attribute = _JOB_STATE_ATTRIBUTES.get(job_name)
    if attribute is None:
        raise HTTPException(status_code=404, detail="Unknown training job")
    with STATE.lock:
        job = getattr(STATE, attribute)
        if job.get("status") not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="No active training job to cancel")
        event = STATE.cancel_events.get(job_name)
        if event is None:
            raise HTTPException(status_code=409, detail="Cancellation is not available for this job instance")
        event.set()
        job.update(
            cancel_requested=True,
            message="Cancellation requested; waiting for the current computation to reach a safe checkpoint.",
        )
    return {
        "ok": True,
        "job": job_name,
        "status": "cancellation_requested",
        "message": "Interruzione richiesta; il job si fermerà al prossimo checkpoint sicuro.",
    }


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
    try:
        items = _scenario_entity_rows(entity, q, limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from exc
    return {"entity": entity, "items": items}


@app.post("/admin-api/scenarios/evaluate")
async def evaluate_scenario(payload: dict[str, Any], user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    """Expose the production scenario engine in the authenticated admin UI."""
    del user
    try:
        resolved_payload = _resolve_scenario_selections(payload)
        return await asyncio.to_thread(_evaluate_scenario, resolved_payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from exc


def _cleanup_idle_incomplete_model_candidates() -> dict[str, list[str]]:
    with STATE.lock:
        jobs = {
            "bb-rating-prediction": STATE.bb_rating_prediction_job.get("status"),
            "bb-rating": STATE.bb_rating.get("status"),
            "future-performance": STATE.future_performance.get("status"),
        }
    removed: dict[str, list[str]] = {}
    for model, status in jobs.items():
        if status in {"queued", "running"}:
            continue
        stale = cleanup_incomplete_candidates(MODEL_ROOT, model)
        if stale:
            removed[model] = stale
            _audit(
                "incomplete_model_candidates_removed",
                "system",
                model,
                json.dumps({"candidate_ids": stale}, ensure_ascii=False),
            )
    return removed


@app.get("/admin-api/bb-rating-prediction")
def bb_rating_prediction_status(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    _cleanup_idle_incomplete_model_candidates()
    with STATE.lock:
        job = dict(STATE.bb_rating_prediction_job)
    return {"job": job, "release": inspect_model(MODEL_ROOT, "bb-rating-prediction"), "minimum_oof_samples": 500, "target_scale": "official_bb_rating_1_100"}


@app.post("/admin-api/bb-rating-prediction/train")
def start_bb_rating_prediction(user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    actor = user["username"]
    with STATE.lock:
        job, active_profile = dict(STATE.bb_rating_prediction_job), STATE.active_profile
    if job.get("status") in {"running", "queued"}:
        raise HTTPException(status_code=409, detail="BB Prediction Model training already running")
    if not active_profile:
        raise HTTPException(status_code=400, detail="Seleziona prima un profilo PostgreSQL")
    with STATE.lock:
        STATE.bb_rating_prediction_job.update(
            status="queued",
            stage="Queued",
            progress=0,
            message="BB Prediction Model training queued.",
            error=None,
            cancel_requested=False,
            candidate_id=None,
            output=None,
            started_at=_utcnow(),
            finished_at=None,
        )
    cancel_event = _new_cancel_event("bb-rating-prediction")
    STATE.bb_rating_prediction_job_executor.submit(_bb_rating_prediction_job_worker, active_profile, actor, cancel_event)
    with STATE.lock:
        current = dict(STATE.bb_rating_prediction_job)
    return {"ok": True, "profile": active_profile, "job": current}


class CausalPanelRefreshPayload(BaseModel):
    kind: str = Field(pattern="^(team_switch|teammate_exposure)$")
    league: str = Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    competition: str = Field(default="RS", min_length=1, max_length=16, pattern=r"^[A-Za-z0-9_]+$")
    through_season: int = Field(ge=2000, le=2100)
    teammate_global_id: str | None = Field(default=None, min_length=1, max_length=128)


@app.post("/admin-api/causal-panels/refresh")
def refresh_causal_panel(payload: CausalPanelRefreshPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    """Explicitly refresh a causal evidence panel; never triggered by chat."""
    actor = user["username"]
    with STATE.lock:
        data = STATE.data
        active_profile = STATE.active_profile
    if not active_profile:
        raise HTTPException(status_code=400, detail="Seleziona prima un profilo PostgreSQL")
    if not isinstance(data, dict) or data.get("player_stats") is None:
        raise HTTPException(status_code=409, detail="Carica prima i dati canonici da PostgreSQL")
    if payload.kind == "teammate_exposure" and not payload.teammate_global_id:
        raise HTTPException(status_code=400, detail="teammate_global_id è obbligatorio per teammate exposure")
    league_dict = data.get("league_dict", {}) or {}
    league_id = next(
        (int(key) for key, value in league_dict.items()
         if str(value.get("league_key", value.get("code", ""))).strip().upper() == payload.league.upper()),
        None,
    )
    if league_id is None:
        raise HTTPException(status_code=400, detail=f"Lega non presente nel dataset caricato: {payload.league}")
    try:
        bb_engine = BBRatingEngine(data["player_stats"], data.get("players"), data.get("leagues"))
        scenario_engine = ChatScenarioEngine(StrictProductionEnsembleModel(), data, bb_rating_engine=bb_engine)
        database_engine = get_engine(profile_url(active_profile))
        if payload.kind == "team_switch":
            result = scenario_engine.refresh_team_switch_causal_panel(
                database_engine, source_league=league_id, competition=payload.competition,
                through_season=payload.through_season,
            )
        else:
            result = scenario_engine.refresh_teammate_exposure_causal_panel(
                database_engine, source_league=league_id, competition=payload.competition,
                through_season=payload.through_season, teammate_global_id=payload.teammate_global_id,
            )
        status = str(result.get("status", "unavailable"))
        _audit(actor, "causal_panel_refresh", payload.kind, json.dumps({
            "league": payload.league, "competition": payload.competition,
            "through_season": payload.through_season, "result": result,
        }, default=str))
        return {"ok": status in {"complete", "written", "refreshed"}, "kind": payload.kind,
                "league": payload.league.upper(), "competition": payload.competition.upper(),
                "through_season": payload.through_season, **result}
    except HTTPException:
        raise
    except Exception as exc:
        _audit(actor, "causal_panel_refresh_failed", payload.kind, str(exc))
        raise HTTPException(status_code=500, detail=f"Refresh pannello causale fallito: {exc}") from exc



class CausalPanelBulkRefreshPayload(BaseModel):
    kind: str = Field(pattern="^(team_switch|teammate_exposure)$")


def _causal_refresh_contexts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate currently available league/competition pairs at their latest completed season."""
    stats = data.get("player_stats")
    required = {"league_id", "league_key", "competition", "season"}
    if not isinstance(stats, pd.DataFrame) or stats.empty or not required.issubset(stats.columns):
        raise RuntimeError("Il dataset canonico non contiene lega, stagione e competizione.")
    frame = stats[list(required)].copy()
    frame["league_id"] = frame["league_id"].map(_to_int)
    frame["league_key"] = frame["league_key"].fillna("").astype(str).str.strip().str.upper()
    frame["competition"] = frame["competition"].fillna("").astype(str).str.strip().str.upper()
    frame["_year"] = pd.to_numeric(
        frame["season"].astype(str).str.split("-").str[0], errors="coerce"
    )
    frame = frame[
        frame["league_id"].notna()
        & frame["league_key"].ne("")
        & frame["competition"].ne("")
        & frame["_year"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError("Nessun contesto lega/competizione disponibile nel dataset.")

    global_seasons = _seasons(data)
    latest_observed = max(global_seasons) if global_seasons else None
    lifecycle = _season_lifecycle()
    contexts: list[dict[str, Any]] = []
    for (league_id, league_key, competition), rows in frame.groupby(
        ["league_id", "league_key", "competition"], sort=True
    ):
        seasons = sorted({int(value) for value in rows["_year"].tolist()})
        if len(seasons) < 2:
            continue
        completed = []
        for season in seasons:
            status = str(lifecycle.get(str(season), {}).get("status", "unclassified")).strip().lower()
            if status == "complete":
                completed.append(season)
            elif status in {"in_progress", "unclassified"} and latest_observed is not None and season < latest_observed:
                # A newer observed season proves this context year is historical,
                # even when a legacy lifecycle registry has not been backfilled.
                completed.append(season)
        if not completed:
            continue
        contexts.append({
            "league_id": int(league_id),
            "league": league_key,
            "competition": competition,
            "through_season": max(completed),
        })
    return contexts


def _run_causal_panel_refresh_all(kind: str, actor: str, active_profile: str, data: dict[str, Any]) -> None:
    started = _utcnow()
    with CAUSAL_REFRESH_LOCK:
        CAUSAL_REFRESH_JOB.update(
            status="running", kind=kind, stage="Enumerazione contesti", progress=1,
            message="Individuazione automatica di tutte le leghe e competizioni con stagioni completate…",
            results=[], summary=None, error=None, started_at=started, finished_at=None,
        )
    database_engine = None
    try:
        contexts = _causal_refresh_contexts(data)
        if not contexts:
            raise RuntimeError("Non ci sono contesti con almeno due stagioni e una stagione completata.")
        bb_engine = BBRatingEngine(data["player_stats"], data.get("players"), data.get("leagues"))
        scenario_engine = ChatScenarioEngine(StrictProductionEnsembleModel(), data, bb_rating_engine=bb_engine)
        database_engine = get_engine(profile_url(active_profile))
        results: list[dict[str, Any]] = []
        exceptions = 0
        refreshed = 0
        skipped = 0
        total_rows = 0
        with CAUSAL_REFRESH_LOCK:
            CAUSAL_REFRESH_JOB.update(
                stage="Refresh in corso", progress=2,
                message=f"Contesti individuati: {len(contexts)}. Avvio refresh automatico.",
            )
        _audit(actor, "causal_panel_refresh_all_started", kind, json.dumps({"contexts": len(contexts)}))

        for index, context in enumerate(contexts, start=1):
            try:
                if kind == "team_switch":
                    result = scenario_engine.refresh_team_switch_causal_panel(
                        database_engine,
                        source_league=context["league_id"],
                        competition=context["competition"],
                        through_season=context["through_season"],
                    )
                else:
                    result = scenario_engine.refresh_teammate_exposure_causal_panel(
                        database_engine,
                        source_league=context["league_id"],
                        competition=context["competition"],
                        through_season=context["through_season"],
                        teammate_global_id=None,
                    )
                status = str(result.get("status", "unavailable"))
                rows_written = int(result.get("rows", 0) or 0)
                total_rows += rows_written
                if status in {"complete", "written", "refreshed"}:
                    refreshed += 1
                else:
                    skipped += 1
                entry = {
                    "league": context["league"],
                    "competition": context["competition"],
                    "through_season": context["through_season"],
                    "status": status,
                    "rows": rows_written,
                    "reason": result.get("reason"),
                }
            except Exception as exc:
                exceptions += 1
                entry = {
                    "league": context["league"],
                    "competition": context["competition"],
                    "through_season": context["through_season"],
                    "status": "failed",
                    "rows": 0,
                    "reason": _safe_error(exc),
                }
            results.append(entry)
            with CAUSAL_REFRESH_LOCK:
                CAUSAL_REFRESH_JOB.update(
                    progress=min(99, 2 + int(index / len(contexts) * 97)),
                    stage=f"Contesto {index}/{len(contexts)}",
                    message=f"{index}/{len(contexts)} contesti elaborati · {refreshed} aggiornati · {skipped} senza coorte utilizzabile · {exceptions} errori.",
                    results=results[-250:],
                    summary={
                        "contexts_total": len(contexts),
                        "contexts_refreshed": refreshed,
                        "contexts_skipped": skipped,
                        "contexts_failed": exceptions,
                        "rows_written": total_rows,
                    },
                )

        summary = {
            "contexts_total": len(contexts),
            "contexts_refreshed": refreshed,
            "contexts_skipped": skipped,
            "contexts_failed": exceptions,
            "rows_written": total_rows,
        }
        final_status = "failed" if exceptions == len(contexts) else ("partial" if exceptions else "complete")
        with CAUSAL_REFRESH_LOCK:
            CAUSAL_REFRESH_JOB.update(
                status=final_status, stage="Completato" if final_status == "complete" else "Terminato con anomalie",
                progress=100, message=(
                    f"Refresh concluso: {refreshed} contesti aggiornati, {skipped} senza coorte utilizzabile, "
                    f"{exceptions} errori; {total_rows} righe scritte."
                ), summary=summary, finished_at=_utcnow(),
            )
        _audit(actor, "causal_panel_refresh_all_finished", kind, json.dumps(summary))
    except Exception as exc:
        error = _safe_error(exc)
        with CAUSAL_REFRESH_LOCK:
            CAUSAL_REFRESH_JOB.update(
                status="failed", stage="Errore", message="Refresh causale non completato.",
                error=error, finished_at=_utcnow(),
            )
        _audit(actor, "causal_panel_refresh_all_failed", kind, error)
    finally:
        if database_engine is not None:
            try:
                database_engine.dispose()
            except Exception:
                pass


@app.post("/admin-api/causal-panels/refresh-all")
def start_causal_panel_refresh_all(
    payload: CausalPanelBulkRefreshPayload, user: dict[str, str] = Depends(_csrf)
) -> dict[str, Any]:
    with STATE.lock:
        data = STATE.data
        active_profile = STATE.active_profile
    if not active_profile:
        raise HTTPException(status_code=400, detail="Seleziona prima un profilo PostgreSQL")
    if not isinstance(data, dict) or not isinstance(data.get("player_stats"), pd.DataFrame) or data["player_stats"].empty:
        raise HTTPException(status_code=409, detail="Carica prima i dati canonici da PostgreSQL")
    with CAUSAL_REFRESH_LOCK:
        if CAUSAL_REFRESH_JOB.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Un refresh causale globale è già in esecuzione.")
        CAUSAL_REFRESH_JOB.update(
            status="queued", kind=payload.kind, stage="In coda", progress=0,
            message="Refresh globale accodato.", results=[], summary=None, error=None,
            started_at=_utcnow(), finished_at=None,
        )
        CAUSAL_REFRESH_EXECUTOR.submit(
            _run_causal_panel_refresh_all, payload.kind, user["username"], active_profile, data
        )
        snapshot = json.loads(json.dumps(CAUSAL_REFRESH_JOB, default=str))
    return {"ok": True, "job": snapshot}


@app.get("/admin-api/causal-panels/refresh-all")
def get_causal_panel_refresh_all(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    with CAUSAL_REFRESH_LOCK:
        return json.loads(json.dumps(CAUSAL_REFRESH_JOB, default=str))


@app.get("/admin-api/model-releases")
def model_releases(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    _cleanup_idle_incomplete_model_candidates()
    return {"models": {model: inspect_model(MODEL_ROOT, model) for model in ("bb-rating", "future-performance", "bb-rating-prediction")}, "automatic_api_restart_configured": bool(os.environ.get("API_KEY", "").strip()), "note": "Promotions and rollbacks request an API restart automatically and wait for readiness; candidate training does not restart production."}


@app.post("/admin-api/model-releases/{model}/promote")
def promote_auxiliary_model(model: str, payload: ConfirmPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Conferma richiesta")
    try:
        result = promote_model(MODEL_ROOT, model, payload.candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    restart = _restart_inference_api()
    result["api_restart"] = restart
    result["restart_required"] = not restart.get("ok")
    _audit("model_release_promoted", user["username"], model, json.dumps(result, ensure_ascii=False))
    return result


@app.post("/admin-api/model-releases/{model}/rollback")
def rollback_auxiliary_model(model: str, payload: ConfirmPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Conferma richiesta")
    try:
        result = rollback_model(MODEL_ROOT, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    restart = _restart_inference_api()
    result["api_restart"] = restart
    result["restart_required"] = not restart.get("ok")
    _audit("model_release_rollback", user["username"], model, json.dumps(result, ensure_ascii=False))
    return result


@app.get("/admin-api/bb-rating")
def bb_rating_status(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    return _bb_rating_status()


@app.post("/admin-api/bb-rating/calibrate")
def start_bb_rating_calibration(user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    actor = user["username"]
    with STATE.lock:
        job = dict(STATE.bb_rating)
        active_profile = STATE.active_profile
    if job.get("status") in {"running", "queued"}:
        raise HTTPException(status_code=409, detail="BB-Rating calibration already running")
    if not active_profile:
        raise HTTPException(status_code=400, detail="Seleziona prima un profilo PostgreSQL")
    with STATE.lock:
        STATE.bb_rating.update(
            status="queued",
            stage="Queued",
            progress=0,
            message="BB-Rating calibration queued.",
            error=None,
            cancel_requested=False,
            started_at=_utcnow(),
            finished_at=None,
        )
    cancel_event = _new_cancel_event("bb-rating")
    STATE.bb_rating_executor.submit(_bb_rating_worker, active_profile, actor, cancel_event)
    with STATE.lock:
        current = dict(STATE.bb_rating)
    return {"ok": True, "profile": active_profile, "job": current}


@app.get("/admin-api/future-performance")
def future_performance_status(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    return _future_performance_status()


@app.post("/admin-api/future-performance/train")
def start_future_performance_training(
    user: dict[str, str] = Depends(_csrf),
) -> dict[str, Any]:
    actor = user["username"]
    with STATE.lock:
        job = dict(STATE.future_performance)
        active_profile = STATE.active_profile
    if job.get("status") in {"running", "queued"}:
        raise HTTPException(status_code=409, detail="Future Performance training already running")
    if not active_profile:
        raise HTTPException(status_code=400, detail="Seleziona prima un profilo PostgreSQL")
    _set_future_performance(
        status="queued",
        stage="Queued",
        progress=0,
        message="Future Performance training queued.",
        error=None,
        started_at=_utcnow(),
        finished_at=None,
        cancel_requested=False,
        actor=actor,
    )
    cancel_event = _new_cancel_event("future-performance")
    STATE.future_performance_executor.submit(
        _future_performance_worker,
        active_profile,
        actor,
        cancel_event,
    )
    with STATE.lock:
        current = dict(STATE.future_performance)
    return {"ok": True, "profile": active_profile, "job": current}


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
    restart = _restart_inference_api()
    result["api_restart"] = restart
    result["restart_required"] = not restart.get("ok")
    return result


@app.post("/admin-api/registry/rollback")
def rollback(payload: ConfirmPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    del user
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Conferma richiesta")
    result = rollback_to_previous(str(MODEL_ROOT))
    if not result.get("rolled_back"):
        raise HTTPException(status_code=409, detail=result.get("reason", "Rollback unavailable"))
    restart = _restart_inference_api()
    result["api_restart"] = restart
    result["restart_required"] = not restart.get("ok")
    return result



@app.post("/admin-api/diagnostics/test")
def diagnostics_test(payload: DiagnosticPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    return _run_diagnostic(payload, user["username"])

@app.post("/admin-api/diagnostics/health-check")
async def diagnostics_health_check(user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    actor = user["username"]
    results = await asyncio.gather(
        asyncio.to_thread(_probe, "/health/live"),
        asyncio.to_thread(_probe, "/health/ready"),
        asyncio.to_thread(_probe, "/health"),
    )
    checks = [{"label": label, "endpoint": endpoint, "result": result, "overall_ok": bool(result.get("ok"))}
              for label, endpoint, result in (
                  ("Inference live", "GET /health/live", results[0]),
                  ("Inference ready", "GET /health/ready", results[1]),
                  ("Inference health", "GET /health", results[2]),
              )]
    _audit("api_health_check", actor, "health")
    return {"overall_ok": all(item["overall_ok"] for item in checks), "checks": checks}

@app.get("/admin-api/api-health")
async def api_health(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    live, ready, bb = await asyncio.gather(
        asyncio.to_thread(_probe, "/health/live"),
        asyncio.to_thread(_probe, "/health/ready"),
        asyncio.to_thread(_probe, "/health"),
    )
    bb_body = bb.get("body") if isinstance(bb.get("body"), dict) else {}
    status = _bb_rating_status() or {}
    future_status = _future_performance_status() or {}
    return {
        "base_url": API_BASE_URL,
        "live": live,
        "ready": ready,
        "bb_rating": {
            "api": bb,
            "loaded": bool(bb_body.get("bb_rating_loaded")),
            "uncertainty_loaded": bool(bb_body.get("bb_rating_uncertainty_loaded")),
            "calibration_version": status.get("calibration_version"),
            "bb_rating_version": status.get("bb_rating_version"),
            "runtime_calibration_version": bb_body.get("bb_rating_calibration_version"),
            "runtime_bb_rating_version": bb_body.get("bb_rating_uncertainty_version") or bb_body.get("bb_rating_version"),
        },
        "future_performance": {
            "api": bb,
            "loaded": bool(bb_body.get("future_performance_loaded")),
            "version": bb_body.get("future_performance_version"),
            "feature_version": bb_body.get("future_performance_feature_version"),
            "artifact_version": future_status.get("future_performance_version"),
        },
        "bb_rating_prediction": {
            "loaded": bool(bb_body.get("bb_rating_future_prediction_loaded")),
            "version": bb_body.get("bb_rating_future_prediction_version"),
            "target_scale": "1-100",
        },
        "endpoints": [
            "GET /health/live",
            "GET /health/ready",
            "GET /health",
            "POST /api/v2/bb-rating/player",
            "POST /api/v2/bb-rating-prediction/player",
            "POST /api/v2/future-performance/player",
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



class ScheduleJobPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    task_key: str = Field(pattern="^(refresh_context_seasonal|refresh_simulation_sources|refresh_scenario_serving|refresh_training_serving|bb_prediction_training|bb_rating_calibration|future_performance_training)$")
    interval_minutes: int = Field(ge=60, le=525600)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


SCHEDULED_TASKS = {
    "refresh_context_seasonal": {"required": {"season"}, "allowed": {"season"}},
    "refresh_simulation_sources": {"required": {"season"}, "allowed": {"season", "league", "leagues"}},
    # Competition is accepted only for older persisted jobs. New jobs refresh
    # every competition found in the selected leagues.
    "refresh_scenario_serving": {"required": {"season"}, "allowed": {"season", "league", "leagues", "competition"}},
    "refresh_training_serving": {"required": {"season"}, "allowed": {"season", "league", "leagues", "competition"}},
    "bb_prediction_training": {"required": set(), "allowed": set()},
    "bb_rating_calibration": {"required": set(), "allowed": set()},
    "future_performance_training": {"required": set(), "allowed": set()},
}


def _validate_schedule_parameters(task_key: str, parameters: dict[str, Any]) -> dict[str, Any]:
    spec = SCHEDULED_TASKS[task_key]
    if set(parameters) - spec["allowed"]:
        raise HTTPException(status_code=400, detail="Parametri non previsti per il job selezionato")
    missing = spec["required"] - set(parameters)
    if missing:
        raise HTTPException(status_code=400, detail="Parametri mancanti: " + ", ".join(sorted(missing)))
    if task_key in {"refresh_simulation_sources", "refresh_scenario_serving", "refresh_training_serving"}:
        if not parameters.get("league") and not parameters.get("leagues"):
            raise HTTPException(status_code=400, detail="Seleziona Tutte le leghe oppure almeno una lega.")
        if parameters.get("league") and parameters.get("leagues"):
            raise HTTPException(status_code=400, detail="Usa la selezione legacy 'league' oppure 'leagues', non entrambe.")
    if "season" in parameters:
        try:
            season = int(parameters["season"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Stagione non valida")
        if not 2000 <= season <= 2100:
            raise HTTPException(status_code=400, detail="Stagione non valida")
        parameters["season"] = season
    if "league" in parameters:
        parameters["league"] = str(parameters["league"]).strip().upper()
        if not re.fullmatch(r"[A-Za-z0-9_]{2,32}", parameters["league"]):
            raise HTTPException(status_code=400, detail="Codice lega non valido")
    if "leagues" in parameters:
        raw_leagues = parameters["leagues"]
        if not isinstance(raw_leagues, list) or not raw_leagues or len(raw_leagues) > 100:
            raise HTTPException(status_code=400, detail="La selezione delle leghe non è valida.")
        leagues = list(dict.fromkeys(str(value).strip().upper() for value in raw_leagues if str(value).strip()))
        if not leagues:
            raise HTTPException(status_code=400, detail="Seleziona Tutte le leghe oppure almeno una lega.")
        if "ALL" in leagues and leagues != ["ALL"]:
            raise HTTPException(status_code=400, detail="'ALL' non può essere combinato con singole leghe.")
        if any(not re.fullmatch(r"[A-Za-z0-9_]{2,32}", value) and value != "ALL" for value in leagues):
            raise HTTPException(status_code=400, detail="Codice lega non valido.")
        parameters["leagues"] = leagues
    if "competition" in parameters:
        parameters["competition"] = str(parameters["competition"]).strip().upper()
        if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", parameters["competition"]):
            raise HTTPException(status_code=400, detail="Competizione non valida")
    return parameters


def _available_source_contexts(engine) -> list[tuple[str, str]]:
    """Current league/competition catalogue from canonical rows, not a static list."""
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT DISTINCT upper(btrim(league_key)) AS league_key,
                            upper(btrim(competition)) AS competition
              FROM "AI_Source"."PlayerCompetitionStats"
             WHERE league_key IS NOT NULL AND btrim(league_key) <> ''
               AND competition IS NOT NULL AND btrim(competition) <> ''
             ORDER BY 1, 2
        """)).mappings().all()
    return [(str(row["league_key"]), str(row["competition"])) for row in rows]


def _scheduled_league_codes(parameters: dict[str, Any], engine) -> list[str]:
    contexts = _available_source_contexts(engine)
    available = sorted({league for league, _ in contexts})
    raw = parameters.get("leagues")
    if isinstance(raw, list):
        selected = available if "ALL" in [str(value).upper() for value in raw] else [
            str(value).strip().upper() for value in raw
        ]
    elif parameters.get("league"):
        selected = [str(parameters["league"]).strip().upper()]
    else:
        selected = available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise RuntimeError("Leghe non presenti nei dati canonici: " + ", ".join(missing))
    return selected


def _scheduler_engine():
    with STATE.lock:
        profile = STATE.active_profile
    if not profile:
        raise RuntimeError("Nessun profilo PostgreSQL selezionato")
    return get_engine(profile_url(profile))


def _execute_scheduled_task(task_key: str, parameters: dict[str, Any], actor: str) -> tuple[str, dict[str, Any]]:
    if task_key in {"bb_prediction_training", "bb_rating_calibration", "future_performance_training"}:
        endpoint = {
            "bb_prediction_training": start_bb_rating_prediction,
            "bb_rating_calibration": start_bb_rating_calibration,
            "future_performance_training": start_future_performance_training,
        }[task_key]
        response = endpoint({"username": actor, "role": "admin", "token": "scheduler"})
        return "submitted", {"message": response.get("message", "Job di training inviato"), "job": response.get("job", {})}

    engine = _scheduler_engine()
    if task_key == "refresh_context_seasonal":
        with engine.begin() as connection:
            connection.execute(
                text('CALL "AI_Source"."RefreshContextSeasonal"(:season)'),
                {"season": int(parameters["season"])},
            )
        return "complete", {"message": "Procedura SQL completata", "task_key": task_key, "parameters": parameters}

    leagues = _scheduled_league_codes(parameters, engine)
    if not leagues:
        raise RuntimeError("Nessuna lega disponibile nei dati canonici.")
    season = int(parameters["season"])
    operations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    if task_key == "refresh_simulation_sources":
        contexts = [(league, None) for league in leagues]
    elif task_key in {"refresh_scenario_serving", "refresh_training_serving"}:
        if parameters.get("competition"):
            requested_competition = str(parameters["competition"]).upper()
            available = set(_available_source_contexts(engine))
            contexts = [
                (league, requested_competition)
                for league in leagues
                if (league, requested_competition) in available
            ]
        else:
            available = _available_source_contexts(engine)
            selected_leagues = set(leagues)
            contexts = [(league, competition) for league, competition in available if league in selected_leagues]
    else:
        raise RuntimeError(f"Operazione schedulata sconosciuta: {task_key}")

    if not contexts:
        raise RuntimeError("Nessuna combinazione lega/competizione disponibile per la pianificazione.")

    for league, competition in contexts:
        try:
            with engine.begin() as connection:
                if task_key == "refresh_simulation_sources":
                    connection.execute(
                        text('CALL "AI_Source"."RefreshSimulationSources"(:league, :season)'),
                        {"league": league, "season": season},
                    )
                elif task_key == "refresh_scenario_serving":
                    connection.execute(
                        text('CALL "AI_Source"."RefreshScenarioServing"(:league, :season, :competition)'),
                        {"league": league, "season": season, "competition": str(competition).upper()},
                    )
                else:
                    connection.execute(
                        text('CALL "AI_Source"."RefreshTrainingServing"(:league, :season, :competition)'),
                        {"league": league, "season": season, "competition": str(competition).upper()},
                    )
            operations.append({"league": league, "competition": competition, "status": "complete"})
        except Exception as exc:
            failures.append({"league": league, "competition": str(competition or ""), "error": _safe_error(exc)})

    result = {
        "message": f"Elaborate {len(operations)} combinazioni; errori: {len(failures)}.",
        "task_key": task_key,
        "season": season,
        "leagues": leagues,
        "competitions": sorted({competition for _, competition in contexts if competition}),
        "contexts_total": len(contexts),
        "contexts_completed": len(operations),
        "contexts_failed": len(failures),
        "operations": operations[-250:],
        "failures": failures[-100:],
    }
    if failures and not operations:
        return "failed", result
    if failures:
        return "partial", result
    return "complete", result


def _run_due_scheduled_job() -> bool:
    try:
        engine = _scheduler_engine()
        with engine.begin() as connection:
            row = connection.execute(text('''
                SELECT job_id, name, task_key, parameters, interval_minutes, created_by
                  FROM "AI_Source"."AIModelScheduledJob"
                 WHERE enabled AND next_run_at <= now()
                 ORDER BY next_run_at, job_id
                 FOR UPDATE SKIP LOCKED LIMIT 1
            ''')).mappings().first()
            if not row:
                return False
            params = row["parameters"] if isinstance(row["parameters"], dict) else json.loads(row["parameters"] or "{}")
            run_id = connection.execute(text('''
                INSERT INTO "AI_Source"."AIModelJobRun"(job_id, task_key, parameters, status)
                VALUES (:job_id, :task_key, CAST(:parameters AS jsonb), 'running')
                RETURNING run_id
            '''), {"job_id": row["job_id"], "task_key": row["task_key"], "parameters": json.dumps(params)}).scalar_one()
            connection.execute(text('''
                UPDATE "AI_Source"."AIModelScheduledJob"
                   SET next_run_at = now() + make_interval(mins => interval_minutes),
                       last_run_at = now(), last_status = 'running', last_error = NULL, updated_at = now()
                 WHERE job_id = :job_id
            '''), {"job_id": row["job_id"]})
        try:
            status, result = _execute_scheduled_task(str(row["task_key"]), params, str(row["created_by"] or "scheduler"))
            error = None
        except Exception as exc:
            status, result, error = "failed", {}, _safe_error(exc)
        with engine.begin() as connection:
            connection.execute(text('''
                UPDATE "AI_Source"."AIModelJobRun"
                   SET status=:status, finished_at=now(), result=CAST(:result AS jsonb), error=:error
                 WHERE run_id=:run_id
            '''), {"status": status, "result": json.dumps(result, default=str), "error": error, "run_id": run_id})
            connection.execute(text('''
                UPDATE "AI_Source"."AIModelScheduledJob"
                   SET last_status=:status, last_error=:error, updated_at=now()
                 WHERE job_id=:job_id
            '''), {"status": status, "error": error, "job_id": row["job_id"]})
        return True
    except Exception:
        # Missing migration, unavailable DB, or transient connection failure:
        # retry on the next poll without breaking admin API startup.
        return False


def _scheduler_loop() -> None:
    while True:
        try:
            _run_due_scheduled_job()
        except Exception:
            pass
        time.sleep(15)


@app.on_event("startup")
def _start_scheduler_worker() -> None:
    if not getattr(STATE, "_scheduler_thread_started", False):
        STATE._scheduler_thread_started = True
        threading.Thread(target=_scheduler_loop, name="ai-model-scheduler", daemon=True).start()


@app.get("/admin-api/scheduler/options")
def scheduler_options(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    engine = _scheduler_engine()
    contexts = _available_source_contexts(engine)
    by_league: dict[str, list[str]] = {}
    for league, competition in contexts:
        by_league.setdefault(league, []).append(competition)
    return {
        "leagues": sorted(by_league),
        "competitions": {league: sorted(set(values)) for league, values in by_league.items()},
    }


@app.get("/admin-api/scheduler/jobs")
def list_scheduled_jobs(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    engine = _scheduler_engine()
    with engine.connect() as connection:
        jobs = connection.execute(text('SELECT * FROM "AI_Source"."AIModelScheduledJob" ORDER BY job_id')).mappings().all()
        runs = connection.execute(text('SELECT * FROM "AI_Source"."AIModelJobRun" ORDER BY started_at DESC LIMIT 100')).mappings().all()

    def serialise(row):
        return {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in dict(row).items()}
    return {"jobs": [serialise(row) for row in jobs], "runs": [serialise(row) for row in runs], "tasks": sorted(SCHEDULED_TASKS)}


@app.post("/admin-api/scheduler/jobs")
def create_scheduled_job(payload: ScheduleJobPayload, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    parameters = _validate_schedule_parameters(payload.task_key, payload.parameters)
    engine = _scheduler_engine()
    with engine.begin() as connection:
        row = connection.execute(text('''
            INSERT INTO "AI_Source"."AIModelScheduledJob"
                (name, task_key, parameters, interval_minutes, enabled, next_run_at, created_by)
            VALUES (:name, :task_key, CAST(:parameters AS jsonb), :interval_minutes, :enabled,
                    now() + make_interval(mins => :interval_minutes), :created_by)
            RETURNING job_id, name, task_key, parameters, interval_minutes, enabled, next_run_at
        '''), {"name": payload.name, "task_key": payload.task_key, "parameters": json.dumps(parameters),
               "interval_minutes": payload.interval_minutes, "enabled": payload.enabled, "created_by": user["username"]}).mappings().one()
    _audit(user["username"], "schedule_created", str(row["job_id"]), json.dumps(dict(row), default=str))
    return {"ok": True, "job": {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in dict(row).items()}}


@app.post("/admin-api/scheduler/jobs/{job_id}/toggle")
def toggle_scheduled_job(job_id: int, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    engine = _scheduler_engine()
    with engine.begin() as connection:
        row = connection.execute(text('''
            UPDATE "AI_Source"."AIModelScheduledJob"
               SET enabled = NOT enabled, updated_at=now(),
                   next_run_at = CASE WHEN NOT enabled THEN now() + make_interval(mins => interval_minutes) ELSE next_run_at END
             WHERE job_id=:job_id
             RETURNING job_id, enabled
        '''), {"job_id": job_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Pianificazione non trovata")
    _audit(user["username"], "schedule_toggled", str(job_id), str(row["enabled"]))
    return {"ok": True, "job_id": job_id, "enabled": row["enabled"]}


@app.delete("/admin-api/scheduler/jobs/{job_id}")
def delete_scheduled_job(job_id: int, user: dict[str, str] = Depends(_csrf)) -> dict[str, Any]:
    engine = _scheduler_engine()
    with engine.begin() as connection:
        result = connection.execute(text('DELETE FROM "AI_Source"."AIModelScheduledJob" WHERE job_id=:job_id'), {"job_id": job_id})
    if not result.rowcount:
        raise HTTPException(status_code=404, detail="Pianificazione non trovata")
    _audit(user["username"], "schedule_deleted", str(job_id), "")
    return {"ok": True, "job_id": job_id}

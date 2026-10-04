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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

class DiagnosticPayload(BaseModel):
    model: str = Field(pattern="^(prediction|bb_rating|future_performance|metric_rating)$")
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
    fitted = (report or {}).get("uncertainty_calibration") or {}
    dataset = (report or {}).get("dataset") or {}
    validation = (report or {}).get("validation_signals") or {}
    support = fitted.get("support") or {}
    ready = bool(
        report
        and uncertainty
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


def _future_performance_worker(active_profile: str | None, actor: str) -> None:
    from basketball_ai.future_performance import PlayerFuturePerformanceModel

    try:
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
        stats = data.get("player_stats")
        if not isinstance(stats, pd.DataFrame) or stats.empty:
            raise RuntimeError("Nessun dato player_stats disponibile per Future Performance")

        observed_seasons = _seasons(data)
        training_seasons, excluded_in_progress, blocking = _training_seasons(observed_seasons)
        if blocking:
            raise RuntimeError(
                "Snapshot contiene stagioni non idonee al training: "
                + ", ".join(map(str, blocking))
            )
        if len(training_seasons) < 5:
            raise RuntimeError("Servono almeno cinque stagioni Complete per Future Performance")

        _set_future_performance(
            stage="Walk-forward + fitting",
            progress=25,
            message=(
                "Costruzione delle coppie t→t+1, validazione expanding OOS e fit multivariato…"
                + (
                    f" ({len(excluded_in_progress)} stagione/i In corso escluse come target)"
                    if excluded_in_progress
                    else ""
                )
            ),
        )
        model = PlayerFuturePerformanceModel()
        metadata = model.fit(
            data,
            target_seasons=training_seasons,
            backtest=True,
        )

        _set_future_performance(
            stage="Saving production artifact",
            progress=88,
            message="Scrittura artifact Future Performance separato dal Prediction Model…",
            future_performance_version=metadata.get("future_performance_version"),
            feature_version=metadata.get("feature_version"),
        )
        root = _future_performance_path()
        root.mkdir(parents=True, exist_ok=True)
        paths = model.save(root)
        details = {
            "future_performance_version": metadata.get("future_performance_version"),
            "feature_version": metadata.get("feature_version"),
            "n_pairs": metadata.get("n_pairs"),
            "n_players": metadata.get("n_players"),
            "target_seasons": training_seasons,
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
            message="Future Performance model pronto. Prediction Model e BB-Rating non sono stati modificati.",
            error=None,
            finished_at=_utcnow(),
            future_performance_version=metadata.get("future_performance_version"),
            feature_version=metadata.get("feature_version"),
            output=paths,
        )
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


def _bb_rating_worker(active_profile: str | None, actor: str) -> None:
    from basketball_ai.bb_rating.calibration import (
        BBRatingCalibrationConfig,
        build_calibration_report,
        write_calibration_report,
    )

    try:
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
        stats = data.get("player_stats")
        if not isinstance(stats, pd.DataFrame) or stats.empty:
            raise RuntimeError("Nessun dato player_stats disponibile per la calibrazione BB-Rating")
        _set_bb_rating(
            stage="Building calibration report",
            progress=42,
            message="Calcolo percentili, validazione OOS e struttura uncertainty…",
        )
        report = build_calibration_report(data, config=BBRatingCalibrationConfig())
        _set_bb_rating(
            stage="Writing serving artifact",
            progress=82,
            message="Scrittura report e artifact di serving…",
            calibration_version=report.get("calibration_version"),
            bb_rating_version=report.get("bb_rating_version"),
        )
        output_dir = MODEL_ROOT / "bb_rating_calibration"
        paths = write_calibration_report(report, str(output_dir))
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
            message="BB-Rating calibration and uncertainty artifact ready. Public BB-Rating is unchanged.",
            error=None,
            finished_at=_utcnow(),
            calibration_version=report.get("calibration_version"),
            bb_rating_version=report.get("bb_rating_version"),
            output=paths,
        )
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
    if payload.model == "prediction":
        if not payload.team_global_id.strip():
            raise HTTPException(status_code=400, detail="Prediction richiede anche Team Global ID")
        return "/api/v2/predictions/player-team", {**base, "team_global_id": payload.team_global_id.strip(), "competition": payload.competition.strip().upper()}
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
    elif model == "prediction":
        checks += [
            {"label": "Prediction presente", "ok": any(k in body for k in ("prediction", "predicted_rating", "predicted_rating_100", "final_prediction"))},
            {"label": "Intervallo presente", "ok": any(k in body for k in ("confidence_low", "confidence_high", "confidence_low_100", "confidence_high_100"))},
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


def _scenario_entity_rows(entity: str, query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search entities from the dataset already loaded by the Control Center.

    The loaded dataset is the same canonical AI_Source contract used by the
    admin workflow. Keeping entity lookup in-memory avoids a second, fragile
    SQL query solely for autocomplete and never exposes global IDs to the UI.
    """
    with STATE.lock:
        loaded = STATE.data
    if not isinstance(loaded, dict):
        return []

    table = "players" if entity == "player" else "teams"
    frame = loaded.get(table)
    if not isinstance(frame, pd.DataFrame) or frame.empty or "id" not in frame.columns:
        return []

    search = str(query or "").strip()
    max_rows = max(1, min(int(limit), 20))
    work = frame.copy()

    def contains(column: str) -> pd.Series:
        if column not in work.columns:
            return pd.Series(False, index=work.index)
        return work[column].fillna("").astype(str).str.contains(
            search,
            case=False,
            regex=False,
            na=False,
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

        work["_search_rank"] = work["name"].map(rank) if "name" in work.columns else 2
        work = work.sort_values(
            ["_search_rank", "name"],
            ascending=[True, True],
            na_position="last",
            kind="stable",
        )
    elif "name" in work.columns:
        work = work.sort_values("name", na_position="last", kind="stable")

    records: list[dict[str, Any]] = []
    team_global_by_id: dict[str, str] = {}
    team_name_by_id: dict[str, str] = {}
    if entity == "player":
        teams_frame = loaded.get("teams")
        if isinstance(teams_frame, pd.DataFrame) and {"id", "global_id"}.issubset(teams_frame.columns):
            for row in teams_frame.to_dict("records"):
                if row.get("id") is None or pd.isna(row.get("id")):
                    continue
                key = str(row["id"])
                global_value = row.get("global_id")
                if global_value not in (None, ""):
                    team_global_by_id[key] = str(global_value)
                name_value = row.get("name")
                if name_value not in (None, ""):
                    team_name_by_id[key] = str(name_value).strip()

    for row in work.head(max_rows).to_dict("records"):
        internal_id = row.get("id")
        if internal_id is None or pd.isna(internal_id):
            continue

        def value_text(key: str) -> str:
            value = row.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            return str(value).strip()

        def value_text_from_row(source_row: Any, key: str) -> str:
            value = source_row.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            return str(value).strip()

        name = value_text("name") or value_text("global_id") or str(internal_id)
        global_id = value_text("global_id")
        if entity == "player":
            subtitle_values = (value_text("position"), value_text("current_league_key"))
        else:
            subtitle_values = (value_text("short_name"),)

        subtitle = " · ".join(
            value for value in subtitle_values if value and value != name
        )
        league_keys: list[str] = []
        contexts: list[dict[str, Any]] = []
        if entity == "player":
            stats_frame = loaded.get("player_stats")
            if isinstance(stats_frame, pd.DataFrame) and "player_global_id" in stats_frame.columns:
                context_rows = stats_frame[
                    stats_frame["player_global_id"].astype(str).str.strip() == global_id
                ].copy()
                if not context_rows.empty and "league_key" in context_rows.columns:
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
                        team_id_text = value_text_from_row(latest, "team_id")
                        contexts.append(
                            {
                                "league_key": league_key,
                                "season": int(latest["_season_year"]),
                                "competition": value_text_from_row(latest, "competition").upper(),
                                "team_global_id": team_global_by_id.get(team_id_text, ""),
                                "team_name": team_name_by_id.get(team_id_text, ""),
                            }
                        )
                    league_keys = [item["league_key"] for item in contexts]
        else:
            team_history = loaded.get("team_season_stats")
            if isinstance(team_history, pd.DataFrame) and "global_id" in team_history.columns:
                context_rows = team_history[
                    team_history["global_id"].astype(str).str.strip() == global_id
                ].copy()
                if not context_rows.empty and "league_key" in context_rows.columns:
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
                                "competition": value_text_from_row(latest, "competition").upper(),
                            }
                        )
                    league_keys = [item["league_key"] for item in contexts]

        records.append(
            {
                "selection_id": str(internal_id),
                "global_id": global_id,
                "name": name,
                "subtitle": subtitle,
                "league_key": value_text("current_league_key") if entity == "player" else value_text("league_key"),
                "current_team_global_id": (
                    team_global_by_id.get(value_text("current_team_id"), "")
                    if entity == "player"
                    else ""
                ),
                "current_team_name": (
                    team_name_by_id.get(value_text("current_team_id"), "")
                    if entity == "player"
                    else ""
                ),
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
def index() -> HTMLResponse:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    marker = '<script src="/assets/future_performance_ui.js" defer></script>'
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


@app.get("/admin-api/overview")
async def overview(user: dict[str, str] = Depends(_operator)) -> dict[str, Any]:
    del user
    registry = _registry()
    with STATE.lock:
        data_loaded = STATE.data is not None
        summary = dict(STATE.summary)
        active_profile = STATE.active_profile
    seasons = summary.get("seasons", [])
    training_seasons, excluded_in_progress, blocking = _training_seasons(seasons)
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
            "status": "ok" if len(training_seasons) >= 5 and not blocking else "warning",
            "label": "OK" if len(training_seasons) >= 5 and not blocking else "Warning",
            "detail": (
                f"{len(training_seasons)} stagioni Complete per il training"
                + (f"; {len(excluded_in_progress)} In corso escluse" if excluded_in_progress else "")
                + (f"; bloccanti: {', '.join(map(str, blocking))}" if blocking else "")
            ),
            "action": "training",
        },
        {
            "component": "Production model",
            "status": "ok" if production else "danger",
            "label": "Production" if production else "Not promoted",
            "detail": production.get("run_id") or "Nessun modello in produzione",
            "action": "registry",
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
        "production": production,
        "candidate": candidate,
        "candidate_quality": _candidate_quality(candidate),
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
    if job.get("status") == "running":
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
            started_at=_utcnow(),
            finished_at=None,
        )
    STATE.bb_rating_executor.submit(_bb_rating_worker, active_profile, actor)
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
        actor=actor,
    )
    STATE.future_performance_executor.submit(
        _future_performance_worker,
        active_profile,
        actor,
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
    status = _bb_rating_status()
    future_status = _future_performance_status()
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
        "endpoints": [
            "GET /health/live",
            "GET /health/ready",
            "GET /health",
            "POST /api/v2/predictions/player-team",
            "POST /api/v2/bb-rating/player",
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
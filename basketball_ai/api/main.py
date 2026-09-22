"""FastAPI application for the production AI_Model v2 inference surface.

The public service intentionally exposes only typed v2 prediction/scenario
contracts plus health/observability endpoints. Legacy v1 routers and
unversioned prediction endpoints are not mounted.
"""
from __future__ import annotations

import hmac
import json as _json
import logging
import os
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from basketball_ai.api.limiter import SLOWAPI_AVAILABLE, limiter

logger = logging.getLogger(__name__)
app_state: Dict[str, Any] = {}

try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class _JsonFormatter(logging.Formatter):
    """Emit one structured JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _json.dumps(payload, ensure_ascii=False)


def _parse_allowed_origins() -> List[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not raw:
        if os.environ.get("API_ENV", "production").lower() == "development":
            logger.warning("[API] ALLOWED_ORIGINS not set; allowing all origins in development")
            return ["*"]
        return []
    if raw == "*":
        logger.warning("[API] wildcard CORS enabled")
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _empty_data() -> Dict[str, Any]:
    import pandas as pd

    return {
        "leagues": pd.DataFrame(),
        "teams": pd.DataFrame(),
        "players": pd.DataFrame(),
        "player_stats": pd.DataFrame(),
        "team_player_relations": pd.DataFrame(),
        "team_season_stats": pd.DataFrame(),
        "league_dict": {},
        "team_dict": {},
        "player_dict": {},
        "league_teams": {},
    }


def _load_app_state() -> None:
    """Load PostgreSQL data and the exact production model implementation.

    A missing/invalid model deliberately leaves ``engine`` as ``None`` so
    ``/health/ready`` returns 503 and v2 inference cannot silently fall back to
    an untrained estimator.
    """
    from basketball_ai.models.strict_production import (
        StrictProductionEnsembleModel,
        StrictWhatIfEngine,
    )

    data_source = os.environ.get("DATA_SOURCE", "postgres")
    data_dir = os.environ.get("DATA_DIR", "data/sample")
    model_dir = os.environ.get("MODEL_DIR", "models_saved")

    try:
        if data_source == "postgres":
            from basketball_ai.data.postgres_loader import load_all_data

            logger.info("[API] Loading canonical PostgreSQL data")
            data = load_all_data()
        else:
            from basketball_ai.data.loader import data_exists, load_all_data

            if not data_exists(data_dir):
                raise RuntimeError("development CSV data not found")
            logger.info("[API] Loading development CSV data")
            data = load_all_data(data_dir)
    except Exception as exc:
        logger.error("[API] Data load failed: %s", exc, exc_info=True)
        app_state["data"] = _empty_data()
        app_state["engine"] = None
        return

    app_state["data"] = data
    app_state["engine"] = None

    # FASE I — precomputed Metric Rating distributions (AI.MetricDistribution).
    # Missing/empty only disables the rating endpoints; forecasts keep working.
    metric_distributions: list = []
    if data_source == "postgres":
        try:
            from basketball_ai.metric_rating.store import load_metric_distributions

            metric_distributions = load_metric_distributions()
            logger.info(
                "[API] Loaded %d metric distributions", len(metric_distributions)
            )
        except Exception as exc:
            logger.warning(
                "[API] Metric distributions unavailable (rating endpoints disabled): %s",
                exc,
            )
    app_state["metric_distributions"] = metric_distributions

    perf_path = Path(model_dir) / "performance_model.joblib"
    compat_path = Path(model_dir) / "compatibility_model.joblib"
    if not perf_path.exists() or not compat_path.exists():
        logger.error("[API] Production model files are missing from %s", model_dir)
        return

    try:
        ensemble = StrictProductionEnsembleModel()
        ensemble.load(model_dir)
        if not ensemble.is_trained:
            raise RuntimeError("loaded ensemble is not trained")
        app_state["engine"] = StrictWhatIfEngine(ensemble, data)
        logger.info("[API] Data and strict production model loaded")
    except Exception as exc:
        logger.error("[API] Model load failed: %s", exc, exc_info=True)
        app_state["engine"] = None


def _model_metadata(model_dir: str) -> dict[str, str]:
    path = Path(model_dir) / "metadata.json"
    if not path.exists():
        return {
            "model_run_id": "unversioned",
            "model_version": "v2",
            "feature_version": "unknown",
            "data_cutoff": "1970-01-01",
        }
    try:
        metadata = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("[API] Invalid model metadata")
        metadata = {}
    return {
        "model_run_id": str(metadata.get("model_run_id", "unversioned")),
        "model_version": str(metadata.get("model_version", "v2")),
        "feature_version": str(metadata.get("feature_version", "unknown")),
        "data_cutoff": str(metadata.get("data_cutoff", "1970-01-01")),
    }


def _validate_production_security() -> None:
    if os.environ.get("API_ENV", "development").lower() != "production":
        return
    if not _parse_allowed_origins():
        raise RuntimeError("ALLOWED_ORIGINS is required when API_ENV=production")
    if not SLOWAPI_AVAILABLE:
        raise RuntimeError("slowapi is required when API_ENV=production")
    if not os.environ.get("JWT_SECRET", "").strip() and not os.environ.get("API_KEY", "").strip():
        raise RuntimeError("JWT_SECRET or API_KEY is required when API_ENV=production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("LOG_FORMAT", "text").lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    else:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    _validate_production_security()
    _load_app_state()
    app.state.data = app_state.get("data", {})
    app.state.engine = app_state.get("engine")
    app.state.metric_distributions = app_state.get("metric_distributions", [])
    app.state.model_metadata = _model_metadata(os.environ.get("MODEL_DIR", "models_saved"))

    yield

    logger.info("[API] Shutting down")
    try:
        from basketball_ai.api.audit import AuditDB

        AuditDB()
    except Exception:
        logger.debug("[API] Audit store unavailable during shutdown", exc_info=True)


def _decode_jwt(token: str, primary_secret: str) -> dict[str, Any]:
    import jwt

    algorithm = os.environ.get("JWT_ALGORITHM", "HS256").strip() or "HS256"
    secrets_raw = os.environ.get("JWT_SECRETS", "").strip()
    secrets: list[str] = [primary_secret]
    if secrets_raw:
        try:
            entries = _json.loads(secrets_raw)
            parsed = [str(item.get("secret", "")) for item in entries if isinstance(item, dict) and item.get("secret")]
            if parsed:
                secrets = parsed
        except Exception:
            logger.warning("[API] JWT_SECRETS is invalid JSON; using JWT_SECRET only")

    for secret in secrets:
        try:
            return jwt.decode(token, secret, algorithms=[algorithm], options={"verify_exp": True})
        except jwt.ExpiredSignatureError:
            raise
        except jwt.InvalidTokenError:
            continue
    raise jwt.InvalidTokenError("token verification failed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="BBallstat AI_Model",
        description="Typed basketball prediction and scenario service for Chat V3 and server-side clients.",
        version="2.4.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def body_size_middleware(request: Request, call_next):
        max_bytes = int(os.environ.get("MAX_REQUEST_BODY_MB", "10")) * 1_000_000
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                media_type="application/problem+json",
                content={
                    "type": "https://httpstatuses.com/413",
                    "title": "Request Entity Too Large",
                    "status": 413,
                    "detail": f"Request body exceeds {max_bytes // 1_000_000}MB limit",
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "0")
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    @app.middleware("http")
    async def request_audit_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        request.state.request_id = request_id
        started = _time.monotonic()
        response = await call_next(request)
        latency_ms = (_time.monotonic() - started) * 1000
        response.headers["X-Request-ID"] = request_id

        try:
            from basketball_ai.api.metrics import record_request

            record_request(
                path=request.url.path,
                method=request.method,
                status=response.status_code,
                latency_ms=latency_ms,
            )
        except Exception:
            pass

        try:
            from basketball_ai.tenancy import TenantManager

            tenant_id = getattr(request.state, "tenant_id", "default") or "default"
            TenantManager().record_usage(tenant_id, request.url.path)
        except Exception:
            pass

        try:
            from basketball_ai.api.audit import AuditDB

            AuditDB().log(
                request_id=request_id,
                user=getattr(request.state, "current_user", ""),
                tenant=getattr(request.state, "tenant_id", ""),
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=latency_ms,
            )
        except Exception:
            pass
        return response

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        open_paths = {"/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"}
        if request.url.path in open_paths:
            return await call_next(request)

        jwt_secret = os.environ.get("JWT_SECRET", "").strip()
        api_key = os.environ.get("API_KEY", "").strip()

        if jwt_secret:
            if request.headers.get("X-API-Key"):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Use Bearer authentication"})
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Missing Bearer token"})
            try:
                payload = _decode_jwt(auth_header[7:], jwt_secret)
            except Exception:
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid or expired token"})
            request.state.tenant_id = payload.get("tenant_id", "default")
            request.state.current_user = payload.get("sub", "")
            request.state.current_role = payload.get("role", "")
        elif api_key:
            supplied = request.headers.get("X-API-Key", "")
            if not supplied or not hmac.compare_digest(supplied, api_key):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid or missing X-API-Key header"})
            request.state.tenant_id = "default"
            request.state.current_user = "service"
            request.state.current_role = "service"
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from fastapi import HTTPException as _HTTPException
    from fastapi.exceptions import RequestValidationError as _RequestValidationError

    @app.exception_handler(_HTTPException)
    async def http_exception_handler(request: Request, exc: _HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            media_type="application/problem+json",
            content={
                "type": f"https://httpstatuses.com/{exc.status_code}",
                "title": exc.detail if isinstance(exc.detail, str) else "Error",
                "status": exc.status_code,
                "detail": exc.detail,
                "instance": str(request.url),
            },
        )

    @app.exception_handler(_RequestValidationError)
    async def validation_exception_handler(request: Request, exc: _RequestValidationError):
        return JSONResponse(
            status_code=422,
            media_type="application/problem+json",
            content={
                "type": "https://httpstatuses.com/422",
                "title": "Validation Error",
                "status": 422,
                "detail": "Request body/parameters failed validation",
                "errors": exc.errors(),
                "instance": str(request.url),
            },
        )

    if SLOWAPI_AVAILABLE:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    if _PROMETHEUS_AVAILABLE:
        _Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    from basketball_ai.api.routes.predictions_v2 import router as predictions_v2_router
    from basketball_ai.api.routes.scenarios_v2 import router as scenarios_v2_router
    from basketball_ai.api.routes.metric_rating_v2 import router as metric_rating_v2_router

    app.include_router(predictions_v2_router)
    app.include_router(scenarios_v2_router)
    app.include_router(metric_rating_v2_router)

    @app.get("/health")
    def health(request: Request):
        data = getattr(request.app.state, "data", {})
        return {"status": "ok", "data_loaded": bool(data.get("player_dict"))}

    @app.get("/health/live")
    def health_live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def health_ready(request: Request):
        data = getattr(request.app.state, "data", {})
        engine = getattr(request.app.state, "engine", None)
        data_loaded = bool(data.get("player_dict"))
        model_loaded = engine is not None
        if data_loaded and model_loaded:
            return {"status": "ready", "data": True, "model": True}
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "data": data_loaded,
                "model": model_loaded,
                "detail": "data or trained model not loaded",
            },
        )

    return app


app = create_app()

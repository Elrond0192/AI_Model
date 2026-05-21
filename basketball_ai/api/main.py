"""FastAPI application factory and startup lifecycle."""
from __future__ import annotations
import json as _json
import logging
import os
import threading as _threading
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as _BaseModel

from basketball_ai.api.limiter import limiter, SLOWAPI_AVAILABLE

logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# AP8 – In-flight idempotency deduplication
# ---------------------------------------------------------------------------
_in_flight: set = set()
_in_flight_lock = _threading.Lock()


# ---------------------------------------------------------------------------
# O1 – Structured JSON logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for structured log aggregators."""
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return _json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# O2 – Prometheus metrics (optional)
# ---------------------------------------------------------------------------

try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def _parse_allowed_origins() -> List[str]:
    """Read ALLOWED_ORIGINS env var (comma-separated).

    Falls back to ``["*"]`` **only** when the env var is explicitly set to ``"*"``
    or when ``API_ENV`` is ``"development"``.  In production the caller must set
    the whitelist explicitly; an empty / unset var causes the server to start with
    no CORS origins permitted (safe default).
    """
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not raw:
        api_env = os.environ.get("API_ENV", "production").lower()
        if api_env == "development":
            logger.warning("[API] ALLOWED_ORIGINS not set – allowing all origins in development mode")
            return ["*"]
        return []
    if raw == "*":
        logger.warning("[API] ALLOWED_ORIGINS='*' – wildcard CORS allowed (insecure in production)")
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    logger.info("[API] CORS allowed origins: %s", origins)
    return origins


def _load_app_state() -> None:
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine

    data_dir   = os.environ.get("DATA_DIR",   "data/sample")
    model_dir  = os.environ.get("MODEL_DIR",  "models_saved")
    data_source = os.environ.get("DATA_SOURCE", "file")

    # Load data from Azure SQL or local CSV files
    if data_source == "sql":
        logger.info("[API] Loading data from Azure SQL Server …")
        try:
            from basketball_ai.data.sql_loader import load_all_data
            data = load_all_data()
        except Exception as exc:
            logger.error("[API] ERROR loading SQL data: %s", exc)
            app_state["data"]        = _empty_data()
            app_state["engine"]      = None
            app_state["chat_engine"] = None
            return
    else:
        from basketball_ai.data.loader import load_all_data, data_exists
        if not data_exists(data_dir):
            logger.warning("[API] data not found – run --mode generate-data first")
            app_state["data"]        = _empty_data()
            app_state["engine"]      = None
            app_state["chat_engine"] = None
            return
        logger.info("[API] Loading data from CSV files …")
        data = load_all_data(data_dir)

    app_state["data"] = data

    ensemble    = EnsembleModel()
    perf_path   = Path(model_dir) / "performance_model.joblib"
    compat_path = Path(model_dir) / "compatibility_model.joblib"

    if perf_path.exists() and compat_path.exists():
        logger.info("[API] Loading pre-trained models …")
        ensemble.load(model_dir)
    else:
        logger.warning("[API] No pre-trained models found – starting without trained models.")

    engine = WhatIfEngine(ensemble, data)
    app_state["engine"]      = engine
    app_state["chat_engine"] = None   # instantiated lazily on first /chat request
    logger.info("[API] Ready.")


def _empty_data() -> Dict[str, Any]:
    import pandas as pd
    return {
        "leagues": pd.DataFrame(), "teams": pd.DataFrame(),
        "players": pd.DataFrame(), "player_stats": pd.DataFrame(),
        "team_player_relations": pd.DataFrame(),
        "league_dict": {}, "team_dict": {}, "player_dict": {}, "league_teams": {},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_format = os.environ.get("LOG_FORMAT", "text").lower()
    if log_format == "json":
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
    # AP7: CORS fail-fast – refuse to start in production if ALLOWED_ORIGINS is unset.
    _cors_origins = _parse_allowed_origins()
    if not _cors_origins and os.environ.get("API_ENV", "development").lower() == "production":
        raise RuntimeError(
            "[API] ALLOWED_ORIGINS env var is required in production mode "
            "(API_ENV=production). Set it to a comma-separated list of allowed "
            "origins or set API_ENV=development to allow all origins during "
            "local development."
        )
    # AP6: rate limiting must be available in production.
    if not SLOWAPI_AVAILABLE and os.environ.get("API_ENV", "development").lower() == "production":
        raise RuntimeError(
            "[API] slowapi is required for rate limiting in production mode "
            "(API_ENV=production). Install it with: pip install slowapi"
        )
    _load_app_state()
    # Sync app.state for DI-based access (P5 – gradual migration)
    app.state.data = app_state.get("data", {})
    app.state.engine = app_state.get("engine")
    app.state.chat_engine = app_state.get("chat_engine")

    # --- Graceful shutdown ---------------------------------------------------
    import asyncio
    import signal as _signal
    _shutdown_event = asyncio.Event()

    def _handle_sigterm(*_):
        logger.info("[API] SIGTERM received – initiating graceful shutdown")
        _shutdown_event.set()

    try:
        _signal.signal(_signal.SIGTERM, _handle_sigterm)
    except (OSError, ValueError):
        pass  # Not available on all platforms (e.g. Windows, non-main thread)

    yield

    # Flush audit log on shutdown
    logger.info("[API] Shutting down – flushing resources")
    try:
        from basketball_ai.api.audit import AuditDB
        AuditDB()  # init (no-op if already initialized)
        logger.info("[API] Audit DB flushed")
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Basketball Player Performance AI",
        description=(
            "Professional AI system for estimating basketball player performance "
            "across leagues and teams. Supports What-If scenario analysis, "
            "age trajectory, transfer impact, team/player fit ranking, "
            "and a natural-language chat interface."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # AP1 – JWT-primary auth middleware.
    # JWT_SECRET: JWT mode (primary). API_KEY: legacy mode with warning.
    # Leave both empty to disable (dev mode).
    # -----------------------------------------------------------------------

    @app.middleware("http")
    async def body_size_middleware(request: Request, call_next):
        """AP9 – Reject requests exceeding MAX_REQUEST_BODY_MB (default 10MB)."""
        max_bytes = int(os.environ.get("MAX_REQUEST_BODY_MB", "10")) * 1_000_000
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                media_type="application/problem+json",
                content={
                    "type":   "https://httpstatuses.com/413",
                    "title":  "Request Entity Too Large",
                    "status": 413,
                    "detail": f"Request body exceeds {max_bytes // 1_000_000}MB limit",
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """Add OWASP-recommended security headers to every response (S8)."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "0")  # modern browsers: use CSP instead
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    @app.middleware("http")
    async def request_id_audit_middleware(request: Request, call_next):
        """Inject X-Request-ID, idempotency dedup, and write to SQLite audit log."""
        req_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        request.state.request_id = req_id

        # AP8 – Idempotency-Key: reject concurrent duplicate POSTs
        idempotency_key = request.headers.get("Idempotency-Key", "").strip()
        if request.method == "POST" and idempotency_key:
            with _in_flight_lock:
                if idempotency_key in _in_flight:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "type":   "https://httpstatuses.com/409",
                            "title":  "Duplicate Request",
                            "status": 409,
                            "detail": "A request with this Idempotency-Key is already in flight",
                        },
                    )
                _in_flight.add(idempotency_key)
        t0 = _time.monotonic()
        try:
            response = await call_next(request)
        finally:
            if request.method == "POST" and idempotency_key:
                with _in_flight_lock:
                    _in_flight.discard(idempotency_key)
        latency_ms = (_time.monotonic() - t0) * 1000
        # Record tenant usage (non-blocking best-effort)
        try:
            from basketball_ai.tenancy import TenantManager
            tenant_id = getattr(request.state, "tenant_id", "default") or "default"
            TenantManager().record_usage(tenant_id, request.url.path)
        except Exception:
            pass
        response.headers["X-Request-ID"] = req_id
        # Record metrics
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
        # Write audit record (non-blocking best-effort)
        try:
            from basketball_ai.api.audit import AuditDB
            AuditDB().log(
                request_id=req_id,
                user=getattr(request.state, "current_user", ""),
                tenant=getattr(request.state, "current_tenant", ""),
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
        open_paths = {
            "/health", "/health/live", "/health/ready",
            "/docs", "/redoc", "/openapi.json",
            "/api/v1/auth/token", "/api/v1/auth/refresh",
        }
        if request.url.path in open_paths:
            return await call_next(request)

        _jwt_secret = os.environ.get("JWT_SECRET", "").strip()
        _api_key_val = os.environ.get("API_KEY", "").strip()

        if _jwt_secret:
            # JWT primary mode: reject legacy X-API-Key
            if request.headers.get("X-API-Key"):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "X-API-Key is deprecated. Use Authorization: Bearer <token>"},
                )
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing or invalid Authorization header"},
                )
            token = auth_header[7:]
            try:
                import jwt as _jwt
                import json as _json
                # Support multi-secret rotation via JWT_SECRETS env var.
                _jwt_secrets_raw = os.environ.get("JWT_SECRETS", "").strip()
                if _jwt_secrets_raw:
                    try:
                        _entries = _json.loads(_jwt_secrets_raw)
                        _secret_list = [(e["kid"], e["secret"]) for e in _entries if e.get("secret")]
                    except Exception:
                        _secret_list = [("default", _jwt_secret)]
                else:
                    _secret_list = [("default", _jwt_secret)]
                payload = None
                for _kid, _sec in _secret_list:
                    try:
                        payload = _jwt.decode(
                            token,
                            _sec,
                            algorithms=["HS256"],
                            options={"verify_exp": True},
                        )
                        break
                    except _jwt.ExpiredSignatureError:
                        return JSONResponse(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            content={"detail": "Invalid or expired token"},
                        )
                    except _jwt.InvalidTokenError:
                        pass
                if payload is None:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or expired token"},
                    )
                # Inject tenant context so downstream routes can scope queries.
                request.state.tenant_id = payload.get("tenant_id", "default")
                request.state.current_user = payload.get("sub", "")
                request.state.current_role = payload.get("role", "")
            except Exception:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token"},
                )
        elif _api_key_val:
            # Legacy API-key mode
            logger.warning("[AP1] JWT_SECRET not set – using legacy API key mode")
            key = request.headers.get("X-API-Key", "")
            if key != _api_key_val:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing X-API-Key header"},
                )
        return await call_next(request)

    # CORS – whitelist explicit origins from env var (safe default: none)
    allowed_origins = _parse_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RFC 7807 problem+json error responses
    from fastapi import HTTPException as _HTTPException
    from fastapi.exceptions import RequestValidationError as _RVError
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(_HTTPException)
    async def http_exception_handler(request: Request, exc: _HTTPException):
        return _JSONResponse(
            status_code=exc.status_code,
            media_type="application/problem+json",
            content={
                "type":     f"https://httpstatuses.com/{exc.status_code}",
                "title":    exc.detail if isinstance(exc.detail, str) else "Error",
                "status":   exc.status_code,
                "detail":   exc.detail,
                "instance": str(request.url),
            },
        )

    @app.exception_handler(_RVError)
    async def validation_exception_handler(request: Request, exc: _RVError):
        return _JSONResponse(
            status_code=422,
            media_type="application/problem+json",
            content={
                "type":     "https://httpstatuses.com/422",
                "title":    "Validation Error",
                "status":   422,
                "detail":   "Request body/parameters failed validation",
                "errors":   exc.errors(),
                "instance": str(request.url),
            },
        )

    # Rate limiting via slowapi (no-op when slowapi not installed)
    if SLOWAPI_AVAILABLE:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("[API] Rate limiting enabled")

    if _PROMETHEUS_AVAILABLE:
        _Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        logger.info("[O2] Prometheus metrics exposed at /metrics")

    from basketball_ai.api.routes.players     import router as players_router
    from basketball_ai.api.routes.teams       import router as teams_router
    from basketball_ai.api.routes.predictions import router as predictions_router
    from basketball_ai.api.routes.scenarios   import router as scenarios_router
    from basketball_ai.api.routes.chat        import router as chat_router
    from basketball_ai.api.routes.wordpress   import router as wordpress_router
    from basketball_ai.api.routes.auth        import router as auth_router

    prefix = "/api/v1"
    app.include_router(auth_router,        prefix=prefix)
    app.include_router(players_router,     prefix=prefix)
    app.include_router(teams_router,       prefix=prefix)
    app.include_router(predictions_router, prefix=prefix)
    app.include_router(scenarios_router,   prefix=prefix)
    app.include_router(chat_router,        prefix=prefix)
    app.include_router(wordpress_router,   prefix=prefix)

    @app.get("/health")
    def health(request: Request):
        from basketball_ai.api import _deps
        data = _deps.get_data(request)
        return {
            "status": "ok",
            "data_loaded": bool(data.get("player_dict")),
        }

    @app.get("/health/live")
    def health_live():
        """Kubernetes / Docker liveness probe – always 200 when the process is up."""
        return {"status": "alive"}

    @app.get("/health/ready")
    def health_ready(request: Request):
        """Readiness probe – 200 when data and models are loaded, 503 otherwise."""
        from basketball_ai.api import _deps
        data  = _deps.get_data(request)
        engine = _deps.get_engine(request)
        data_loaded  = bool(data.get("player_dict"))
        model_loaded = engine is not None
        if data_loaded and model_loaded:
            return {"status": "ready", "data": True, "model": True}
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse(
            status_code=503,
            content={
                "status":  "not_ready",
                "data":    data_loaded,
                "model":   model_loaded,
                "detail":  "data or model not yet loaded",
            },
        )

    @app.get("/predictions/{player_id}/explain")
    def explain_prediction(
        player_id: int,
        team_id: int = 1,
        season: Optional[str] = None,
        request: Request = None,
    ):
        """Return top-5 SHAP feature contributions for this prediction."""
        from basketball_ai.features.player_features import compute_player_features
        from basketball_ai.api import _deps
        engine = _deps.get_engine(request)
        data = _deps.get_data(request)
        if engine is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
        try:
            shap_vals = engine.ensemble.perf_model.get_shap_values(
                compute_player_features(player_id, data, season=season)
            )
            if not shap_vals:
                return JSONResponse(status_code=503, content={"detail": "SHAP not available"})
            top5 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            return {
                "player_id":   player_id,
                "top_factors": [{"feature": k, "shap_value": v} for k, v in top5],
            }
        except Exception as exc:
            logger.error("[explain_prediction] player_id=%s: %s", player_id, exc, exc_info=True)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/api/v1/internal/metrics")
    def internal_metrics(request: Request):
        """In-process request metrics – admin only (AP5)."""
        role = getattr(request.state, "current_role", "")
        if role not in ("admin", "") and os.environ.get("JWT_SECRET"):
            from fastapi.responses import JSONResponse as _J
            return _J(status_code=403, content={"detail": "Admin only"})
        from basketball_ai.api.metrics import get_snapshot
        return get_snapshot()

    @app.get("/api/v1/internal/audit")
    def internal_audit(
        request: Request,
        user: Optional[str] = None,
        tenant: Optional[str] = None,
        limit: int = 200,
    ):
        """Audit log query – admin only."""
        role = getattr(request.state, "current_role", "")
        if role not in ("admin", "") and os.environ.get("JWT_SECRET"):
            from fastapi.responses import JSONResponse as _J
            return _J(status_code=403, content={"detail": "Admin only"})
        from basketball_ai.api.audit import AuditDB
        return {"records": AuditDB().query(user=user, tenant=tenant, limit=limit)}

    @app.post("/api/v1/internal/models/promote")
    def models_promote(request: Request, min_improvement_pct: float = 2.0):
        """Promote candidate model to production (admin only)."""
        role = getattr(request.state, "current_role", "")
        if role not in ("admin", "") and os.environ.get("JWT_SECRET"):
            from fastapi.responses import JSONResponse as _J
            return _J(status_code=403, content={"detail": "Admin only"})
        from basketball_ai.models.promote import promote_if_better
        model_dir = os.environ.get("MODEL_DIR", "models_saved")
        return promote_if_better(model_dir, min_improvement_pct=min_improvement_pct)

    @app.post("/api/v1/internal/models/rollback")
    def models_rollback(request: Request):
        """Roll back production model to previous (admin only)."""
        role = getattr(request.state, "current_role", "")
        if role not in ("admin", "") and os.environ.get("JWT_SECRET"):
            from fastapi.responses import JSONResponse as _J
            return _J(status_code=403, content={"detail": "Admin only"})
        from basketball_ai.models.promote import rollback_to_previous
        model_dir = os.environ.get("MODEL_DIR", "models_saved")
        return rollback_to_previous(model_dir)

    @app.get("/api/v1/internal/models/status")
    def models_status(request: Request):
        """Return model promotion status."""
        from basketball_ai.models.promote import get_promotion_status
        model_dir = os.environ.get("MODEL_DIR", "models_saved")
        return get_promotion_status(model_dir)

    return app


# ---------------------------------------------------------------------------
# AP8 – Batch predictions
# ---------------------------------------------------------------------------

class BatchPredictionRequest(_BaseModel):
    pairs: List[dict]  # list of {"player_id": int, "team_id": int}
    season: Optional[str] = None
    competition: str = "RS"


app = create_app()


@app.post("/predictions/batch", tags=["predictions"])
def predict_batch(req: BatchPredictionRequest, request: Request):
    """Predict ratings for multiple (player_id, team_id) pairs in one call."""
    from basketball_ai.api import _deps
    engine = _deps.get_engine(request)
    data = _deps.get_data(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    results = []
    for pair in req.pairs[:50]:  # limit to 50 pairs
        try:
            player_id = int(pair.get("player_id", 0))
            team_id = int(pair.get("team_id", 0))
            result = engine.predict(player_id, team_id, data,
                                    season=req.season,
                                    competition=req.competition)
            results.append({
                "player_id": player_id,
                "team_id": team_id,
                "rating": result.rating,
                "confidence_interval": list(result.confidence_interval),
            })
        except Exception as exc:
            results.append({
                "player_id": pair.get("player_id"),
                "team_id": pair.get("team_id"),
                "error": str(exc),
            })
    return {"count": len(results), "results": results}

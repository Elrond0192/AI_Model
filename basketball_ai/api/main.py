"""FastAPI application factory and startup lifecycle."""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from basketball_ai.api.limiter import limiter, SLOWAPI_AVAILABLE

logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}


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
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_app_state()
    yield


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
    # JWT or static API-key authentication.
    # Set API_KEY env var for static key, or JWT_SECRET for JWT mode.
    # Leave both empty to disable (dev mode).
    # -----------------------------------------------------------------------
    _api_key = os.environ.get("API_KEY", "").strip()

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if _api_key:
            # Allow unauthenticated access to health check, docs and auth endpoints
            open_paths = {"/health", "/docs", "/redoc", "/openapi.json",
                          "/api/v1/auth/token", "/api/v1/auth/refresh"}
            if request.url.path not in open_paths:
                key = request.headers.get("X-API-Key", "")
                if key != _api_key:
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

    # Rate limiting via slowapi (no-op when slowapi not installed)
    if SLOWAPI_AVAILABLE:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("[API] Rate limiting enabled")

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
    def health():
        return {
            "status": "ok",
            "data_loaded": bool(app_state.get("data", {}).get("player_dict")),
        }

    return app


app = create_app()

"""FastAPI application factory and startup lifecycle."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app_state: Dict[str, Any] = {}


def _load_app_state() -> None:
    from src.models.ensemble import EnsembleModel
    from src.scenarios.engine import WhatIfEngine

    data_dir   = os.environ.get("DATA_DIR",   "data/sample")
    model_dir  = os.environ.get("MODEL_DIR",  "models_saved")
    data_source = os.environ.get("DATA_SOURCE", "file")

    # Load data from Azure SQL or local CSV files
    if data_source == "sql":
        print("[API] Loading data from Azure SQL Server …")
        try:
            from src.data.sql_loader import load_all_data
            data = load_all_data()
        except Exception as exc:
            print(f"[API] ERROR loading SQL data: {exc}")
            app_state["data"]        = _empty_data()
            app_state["engine"]      = None
            app_state["chat_engine"] = None
            return
    else:
        from src.data.loader import load_all_data, data_exists
        if not data_exists(data_dir):
            print("[API] WARNING: data not found – run --mode generate-data first")
            app_state["data"]        = _empty_data()
            app_state["engine"]      = None
            app_state["chat_engine"] = None
            return
        print("[API] Loading data from CSV files …")
        data = load_all_data(data_dir)

    app_state["data"] = data

    ensemble    = EnsembleModel()
    perf_path   = Path(model_dir) / "performance_model.joblib"
    compat_path = Path(model_dir) / "compatibility_model.joblib"

    if perf_path.exists() and compat_path.exists():
        print("[API] Loading pre-trained models …")
        ensemble.load(model_dir)
    else:
        print("[API] No pre-trained models found – starting without trained models.")

    engine = WhatIfEngine(ensemble, data)
    app_state["engine"]      = engine
    app_state["chat_engine"] = None   # instantiated lazily on first /chat request
    print("[API] Ready.")


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
    # Optional API-key authentication
    # Set API_KEY env var to enable; leave empty to disable (dev mode).
    # -----------------------------------------------------------------------
    _api_key = os.environ.get("API_KEY", "").strip()

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if _api_key:
            # Allow unauthenticated access to health check and docs
            open_paths = {"/health", "/docs", "/redoc", "/openapi.json"}
            if request.url.path not in open_paths:
                key = request.headers.get("X-API-Key", "")
                if key != _api_key:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing X-API-Key header"},
                    )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    from src.api.routes.players     import router as players_router
    from src.api.routes.teams       import router as teams_router
    from src.api.routes.predictions import router as predictions_router
    from src.api.routes.scenarios   import router as scenarios_router
    from src.api.routes.chat        import router as chat_router
    from src.api.routes.wordpress   import router as wordpress_router

    prefix = "/api/v1"
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

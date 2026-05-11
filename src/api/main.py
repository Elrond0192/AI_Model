"""FastAPI application factory and startup lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Shared mutable state (populated on startup)
app_state: Dict[str, Any] = {}


def _load_app_state() -> None:
    """Load data and models into app_state (called at startup)."""
    from src.data.loader import load_all_data, data_exists
    from src.models.ensemble import EnsembleModel
    from src.scenarios.engine import WhatIfEngine

    data_dir = "data/sample"
    model_dir = "models_saved"

    if not data_exists(data_dir):
        print("[API] WARNING: data not found – generate data first with --mode generate-data")
        app_state["data"] = _empty_data()
        app_state["engine"] = None
        return

    print("[API] Loading data …")
    data = load_all_data(data_dir)
    app_state["data"] = data

    ensemble = EnsembleModel()
    perf_path = Path(model_dir) / "performance_model.joblib"
    compat_path = Path(model_dir) / "compatibility_model.joblib"

    if perf_path.exists() and compat_path.exists():
        print("[API] Loading pre-trained models …")
        ensemble.load(model_dir)
    else:
        print("[API] No pre-trained models found – running inference without model (fallback).")

    engine = WhatIfEngine(ensemble, data)
    app_state["engine"] = engine
    print("[API] Ready.")


def _empty_data() -> Dict[str, Any]:
    import pandas as pd
    return {
        "leagues": pd.DataFrame(),
        "teams": pd.DataFrame(),
        "players": pd.DataFrame(),
        "player_stats": pd.DataFrame(),
        "team_player_relations": pd.DataFrame(),
        "league_dict": {},
        "team_dict": {},
        "player_dict": {},
        "league_teams": {},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_app_state()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Football Performance AI",
        description="Estimate and compare football player performance across leagues and teams.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.routes.players import router as players_router
    from src.api.routes.teams import router as teams_router
    from src.api.routes.predictions import router as predictions_router
    from src.api.routes.scenarios import router as scenarios_router

    prefix = "/api/v1"
    app.include_router(players_router, prefix=prefix)
    app.include_router(teams_router, prefix=prefix)
    app.include_router(predictions_router, prefix=prefix)
    app.include_router(scenarios_router, prefix=prefix)

    @app.get("/health")
    def health():
        return {"status": "ok", "data_loaded": bool(app_state.get("data", {}).get("player_dict"))}

    return app


app = create_app()

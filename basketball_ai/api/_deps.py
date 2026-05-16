"""FastAPI dependency providers (P5) – centralises access to app.state."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from fastapi import Request
logger = logging.getLogger(__name__)

def get_data(request: Request) -> Dict[str, Any]:
    """Priority: app_state global (for TestClient compat) → app.state → {}"""
    try:
        from basketball_ai.api.main import app_state
        data = app_state.get("data", {})
        if data and data.get("player_dict"):
            return data
    except Exception as exc:
        logger.debug("[_deps] get_data app_state access failed: %s", exc)
    return getattr(request.app.state, "data", {})

def get_engine(request: Request) -> Optional[Any]:
    try:
        from basketball_ai.api.main import app_state
        engine = app_state.get("engine")
        if engine is not None:
            return engine
    except Exception as exc:
        logger.debug("[_deps] get_engine app_state access failed: %s", exc)
    return getattr(request.app.state, "engine", None)

def get_chat_engine(request: Request) -> Optional[Any]:
    ce = None
    try:
        from basketball_ai.api.main import app_state
        ce = app_state.get("chat_engine")
    except Exception as exc:
        logger.debug("[_deps] get_chat_engine app_state access failed: %s", exc)
    if ce is None:
        ce = getattr(request.app.state, "chat_engine", None)
    if ce is not None:
        return ce
    engine = get_engine(request)
    data = get_data(request)
    if engine is None or not data or not data.get("player_dict"):
        return None
    from basketball_ai.chat.engine import ChatEngine
    ce = ChatEngine(engine, data)
    try:
        from basketball_ai.api.main import app_state
        app_state["chat_engine"] = ce
    except Exception:
        pass
    try:
        request.app.state.chat_engine = ce
    except Exception:
        pass
    return ce

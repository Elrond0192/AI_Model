"""Chat API route – natural-language interface over the trained basketball AI.

Endpoint
--------
POST /api/v1/chat

Request body
------------
{
  "message": "How good is LeBron at the Lakers?",
  "session_id": "optional-uuid-for-conversation-continuity"
}

Response
--------
{
  "reply":       "**LeBron James** at **Los Angeles Lakers** ...",
  "session_id":  "550e8400-e29b-41d4-a716-446655440000",
  "intent":      "predict",
  "data":        { ... structured prediction data ... },
  "suggestions": ["When will LeBron peak?", ...]
}
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter

from basketball_ai.api.schemas import ChatRequest, ChatMessageResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_chat_engine():
    """Lazily instantiate ChatEngine on first request and cache in app_state."""
    from basketball_ai.api.main import app_state

    ce = app_state.get("chat_engine")
    if ce is not None:
        return ce

    engine = app_state.get("engine")
    data   = app_state.get("data")
    if engine is None or not data or not data.get("player_dict"):
        return None

    from basketball_ai.chat.engine import ChatEngine
    ce = ChatEngine(engine, data)
    app_state["chat_engine"] = ce
    return ce


@router.post("", response_model=ChatMessageResponse)
def chat(req: ChatRequest):
    """Send a natural-language basketball question and get an AI-powered reply.

    Maintains conversation context across calls using ``session_id``.
    Omit ``session_id`` to start a new session; the server will generate one
    and return it – include it in subsequent requests to continue the thread.
    """
    ce = _get_chat_engine()

    if ce is None:
        fallback_id = req.session_id or str(uuid.uuid4())
        return ChatMessageResponse(
            reply=(
                "The AI model is not yet loaded. "
                "Please train or load a model first, then try again."
            ),
            session_id=fallback_id,
            intent="error",
            data={},
            suggestions=["Try again after the model is loaded."],
        )

    session_id = req.session_id or str(uuid.uuid4())
    result = ce.process(req.message, session_id)

    return ChatMessageResponse(
        reply=result.reply,
        session_id=result.session_id,
        intent=result.intent,
        data=result.data,
        suggestions=result.suggestions,
    )

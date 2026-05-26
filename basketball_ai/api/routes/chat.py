"""Chat API route – natural-language interface over the trained basketball AI.

Endpoint
--------
POST /api/v1/chat

Headers
-------
X-User-Hash (optional):
    SHA-256 hex digest of the WordPress user ID.  When present, sessions are
    linked to this user so conversation context persists across browser
    sessions and the compound IP+user_hash rate-limit key is used (§6.2).

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
from typing import Optional

from fastapi import APIRouter, Header, Request

from basketball_ai.api.schemas import ChatRequest, ChatMessageResponse
from basketball_ai.api.limiter import chat_limiter, RATE_LIMIT_CHAT
from basketball_ai.api._deps import get_chat_engine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatMessageResponse)
@chat_limiter.limit(RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    req: ChatRequest,
    x_user_hash: Optional[str] = Header(default=None, alias="X-User-Hash"),
):
    """Send a natural-language basketball question and get an AI-powered reply.

    Maintains conversation context across calls using ``session_id``.
    Omit ``session_id`` to start a new session; the server will generate one
    and return it – include it in subsequent requests to continue the thread.

    Pass ``X-User-Hash`` (SHA-256 of the WordPress user ID) to bind sessions
    to an authenticated user and enable per-user compound rate limiting.
    """
    ce = get_chat_engine(request)

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
    result = ce.process(req.message, session_id, user_hash=x_user_hash)

    return ChatMessageResponse(
        reply=result.reply,
        session_id=result.session_id,
        intent=result.intent,
        data=result.data,
        suggestions=result.suggestions,
    )


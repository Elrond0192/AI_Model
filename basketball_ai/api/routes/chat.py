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

Request headers (optional)
---------------------------
X-User-Hash: <sha256-hex>   WordPress user identifier sent by hoopmetrics-chat
                             plugin.  Used to scope sessions per WP user and
                             apply per-user rate limiting.

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

from fastapi import APIRouter, Header, Request
from typing import Optional

from basketball_ai.api.schemas import ChatRequest, ChatMessageResponse
from basketball_ai.api.limiter import limiter, RATE_LIMIT_CHAT, get_chat_key
from basketball_ai.api._deps import get_chat_engine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatMessageResponse)
@limiter.limit(RATE_LIMIT_CHAT, key_func=get_chat_key)
async def chat(
    request: Request,
    req: ChatRequest,
    user_hash: Optional[str] = Header(default=None, alias="X-User-Hash"),
):
    """Send a natural-language basketball question and get an AI-powered reply.

    Maintains conversation context across calls using ``session_id``.
    Omit ``session_id`` to start a new session; the server will generate one
    and return it – include it in subsequent requests to continue the thread.

    The optional ``X-User-Hash`` header (sent by the WordPress
    ``hoopmetrics-chat`` plugin) links the session to a specific WordPress
    user and enables per-user rate limiting.
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

    # Attach the user hash to the session so the persistent store can link it
    # to the WordPress user for future reference (§6.2).
    result = ce.process(req.message, session_id, user_hash=user_hash)

    return ChatMessageResponse(
        reply=result.reply,
        session_id=result.session_id,
        intent=result.intent,
        data=result.data,
        suggestions=result.suggestions,
    )

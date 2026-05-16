"""Intent handler registry.

Usage::

    from basketball_ai.chat.handlers import register_handler, get_handler

    @register_handler(Intent.PREDICT)
    def handle_predict(context: dict) -> str:
        ...

The ``context`` dict contains: ``intent``, ``message``, ``entities``,
``session``, ``engine``, ``data``.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
from basketball_ai.chat.intent import Intent

_REGISTRY: Dict[Intent, Callable] = {}


def register_handler(intent: Intent):
    """Decorator: register a function as the handler for *intent*."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[intent] = fn
        return fn
    return decorator


def get_handler(intent: Intent) -> Optional[Callable]:
    """Return the registered handler for *intent*, or None."""
    return _REGISTRY.get(intent)

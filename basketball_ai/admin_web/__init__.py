"""Custom operations console for AI_Model.

This package also registers small admin extensions that must be available as
soon as the FastAPI application is imported.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from basketball_ai.data.connection_profiles import delete_profile, load_profiles

# Importing the application here is intentional. When Uvicorn imports
# ``basketball_ai.admin_web.app:app`` Python initialises this package first; the
# submodule is loaded once, then these routes/middleware are registered on the
# same FastAPI instance before the server starts accepting requests.
from . import app as _admin_app


@_admin_app.app.delete("/admin-api/profiles/{name}")
def delete_database_profile(
    name: str,
    user: dict[str, str] = Depends(_admin_app._csrf),
) -> dict[str, object]:
    """Delete one persisted PostgreSQL profile and clear stale active state."""
    del user
    if name not in load_profiles():
        raise HTTPException(status_code=404, detail="Profilo non trovato")
    try:
        deleted = delete_profile(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Profilo non trovato")

    with _admin_app.STATE.lock:
        if _admin_app.STATE.active_profile == name:
            _admin_app.STATE.active_profile = None
            _admin_app.STATE.data = None
            _admin_app.STATE.summary = {}

    return {"ok": True, "profile": name, "active": _admin_app.STATE.active_profile}


@_admin_app.app.middleware("http")
async def inject_profile_delete_ui(request: Request, call_next):
    """Load the profile-delete control without changing the public admin URL."""
    response = await call_next(request)
    if request.url.path != "/" or response.status_code != 200:
        return response
    if "text/html" not in response.headers.get("content-type", ""):
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    marker = b"</body>"
    script = b'<script src="/assets/profile-delete.js" defer></script>'
    if script not in body and marker in body:
        body = body.replace(marker, script + marker, 1)

    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(content=body, status_code=response.status_code, headers=headers)

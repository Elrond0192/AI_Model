"""Runtime PostgreSQL connection profiles for the authenticated admin UI.

Secrets are stored in a dedicated runtime file (0600 on POSIX), never in git.
In production the same profiles may be injected read-only with
``DATABASE_PROFILES_JSON``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
PROFILE_FILE = Path(os.getenv("DATABASE_PROFILES_FILE", "/app/config/database_profiles.json"))


def load_profiles() -> dict[str, dict[str, Any]]:
    injected = os.getenv("DATABASE_PROFILES_JSON", "").strip()
    raw = json.loads(injected) if injected else (json.loads(PROFILE_FILE.read_text(encoding="utf-8")) if PROFILE_FILE.exists() else {})
    if not isinstance(raw, dict):
        raise ValueError("Database profiles must be an object")
    profiles = {
        name: dict(profile)
        for name, profile in raw.items()
        if _NAME.fullmatch(name) and isinstance(profile, dict)
    }
    # Transparently migrate pre-PascalCase runtime profiles. The persisted
    # secret file is updated only on the next explicit save.
    for profile in profiles.values():
        if profile.get("source_schema") == "ai_source":
            profile["source_schema"] = "AI_Source"
        if profile.get("ai_schema") == "ai":
            profile["ai_schema"] = "AI"
    return profiles


def _write_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = PROFILE_FILE.with_name(f".{PROFILE_FILE.name}.tmp")
    temp_file.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    if os.name != "nt":
        temp_file.chmod(0o600)
    temp_file.replace(PROFILE_FILE)
    if os.name != "nt":
        PROFILE_FILE.chmod(0o600)


def save_profile(name: str, profile: dict[str, Any]) -> None:
    if os.getenv("DATABASE_PROFILES_JSON"):
        raise RuntimeError("Injected database profiles are read-only")
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid profile name")
    profiles = load_profiles()
    profiles[name] = profile
    _write_profiles(profiles)


def delete_profile(name: str) -> bool:
    """Delete a persisted database profile.

    Returns ``True`` when the profile existed and was removed, ``False`` when
    it was already absent. Injected profiles stay read-only, just like saves.
    """
    if os.getenv("DATABASE_PROFILES_JSON"):
        raise RuntimeError("Injected database profiles are read-only")
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid profile name")
    profiles = load_profiles()
    if name not in profiles:
        return False
    del profiles[name]
    _write_profiles(profiles)
    return True


def profile_url(name: str) -> str:
    profile = load_profiles().get(name)
    if not profile:
        raise KeyError(f"Unknown database profile: {name}")
    required = ("host", "database", "user", "password")
    if any(not str(profile.get(k, "")).strip() for k in required):
        raise ValueError(f"Incomplete database profile: {name}")
    host = str(profile["host"]).strip()
    database = str(profile["database"]).strip()
    user = quote_plus(str(profile["user"]))
    password = quote_plus(str(profile["password"]))
    port = int(profile.get("port", 5432))
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def active_profile_name() -> str:
    name = os.getenv("DATABASE_PROFILE", "").strip()
    if name:
        return name
    profiles = load_profiles()
    if len(profiles) == 1:
        return next(iter(profiles))
    raise RuntimeError("Set DATABASE_PROFILE to one configured profile")

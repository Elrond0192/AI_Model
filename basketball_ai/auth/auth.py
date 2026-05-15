"""User authentication and management for the Basketball Performance AI platform.

Users are stored in a JSON file (``users.json``) at the project root.  The file
is **not** committed to git (see ``.gitignore``); on first run a default admin
account is created automatically and the temporary password is displayed once.

Roles and permissions are stored in ``roles.json`` at the project root.  Each
role defines the list of **sections** (GUI tabs) that it can access.  The three
built-in roles (``admin``, ``analyst``, ``viewer``) are seeded automatically;
admins can create additional custom roles and assign them fine-grained
section-level privileges through the Gestione Utenti tab.

Sections
--------
- ``data``        – 📂 Dati (data source)
- ``training``    – 🏋️ Training
- ``predictions`` – 🎯 Predizioni
- ``scenarios``   – 🔀 Scenari
- ``chat``        – 💬 Chat
- ``scouting``    – 🔬 Scouting AI
- ``mapping``     – 🗺️ Mapping colonne DB
- ``admin``       – 👥 Gestione Utenti

Password security
-----------------
Passwords are hashed with PBKDF2-HMAC-SHA-256 (260 000 iterations) with a
per-user random 32-byte salt.  No plain-text password is ever stored on disk.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import secrets
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

#: Path of the users file.  Can be overridden via BASKETBALL_AI_USERS_FILE env var.
USERS_FILE = Path(
    os.environ.get(
        "BASKETBALL_AI_USERS_FILE",
        str(Path(__file__).parent.parent.parent / "users.json"),
    )
)

#: Path of the roles file.  Can be overridden via BASKETBALL_AI_ROLES_FILE env var.
ROLES_FILE = Path(
    os.environ.get(
        "BASKETBALL_AI_ROLES_FILE",
        str(Path(__file__).parent.parent.parent / "roles.json"),
    )
)

#: Path of the active sessions file.  Can be overridden via BASKETBALL_AI_SESSIONS_FILE.
SESSIONS_FILE = Path(
    os.environ.get(
        "BASKETBALL_AI_SESSIONS_FILE",
        str(Path(__file__).parent.parent.parent / "sessions.json"),
    )
)

#: Default session TTL in days (overridden by SESSION_COOKIE_TTL_DAYS env var).
_SESSION_TTL_DAYS_DEFAULT = 7

#: Ordered list of all section identifiers that map to GUI tabs.
ALL_SECTIONS: List[str] = [
    "data", "training", "predictions", "scenarios",
    "chat", "scouting", "mapping", "admin",
]

#: Human-readable labels for each section (used in the admin UI).
SECTION_LABELS: Dict[str, str] = {
    "data":        "📂 Dati",
    "training":    "🏋️ Training",
    "predictions": "🎯 Predizioni",
    "scenarios":   "🔀 Scenari",
    "chat":        "💬 Chat",
    "scouting":    "🔬 Scouting AI",
    "mapping":     "🗺️ Mapping",
    "admin":       "👥 Gestione Utenti",
}

#: Default role definitions.  These are used when ``roles.json`` does not exist
#: and as the baseline when a role is missing from the file.
DEFAULT_ROLES: Dict[str, Dict] = {
    "admin": {
        "description": "Accesso completo a tutte le sezioni",
        "sections": list(ALL_SECTIONS),
    },
    "analyst": {
        "description": "Accesso alle funzionalità analitiche (no gestione utenti/mapping)",
        "sections": ["data", "training", "predictions", "scenarios", "chat", "scouting"],
    },
    "viewer": {
        "description": "Accesso in sola lettura a predizioni e chat",
        "sections": ["predictions", "chat", "scouting"],
    },
}

_PBKDF2_ITERATIONS = 260_000
_PBKDF2_DIGEST     = "sha256"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str = "") -> Tuple[str, str]:
    """Hash *password* with PBKDF2-HMAC-SHA-256.

    Returns ``(hex_hash, hex_salt)``.  If *salt* is not supplied a fresh
    random 32-byte salt is generated.
    """
    if not salt:
        salt = secrets.token_hex(32)
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_DIGEST, password.encode("utf-8"), salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    )
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    candidate_hash, _ = _hash_password(password, salt)
    return hmac.compare_digest(candidate_hash, stored_hash)


# ---------------------------------------------------------------------------
# Users – public API
# ---------------------------------------------------------------------------

def load_users() -> Dict[str, Dict]:
    """Return the full users dict from disk.

    Returns an empty dict when the file does not exist or is corrupt.
    """
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: Dict[str, Dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def check_credentials(username: str, password: str) -> Tuple[bool, Dict]:
    """Verify *username* / *password* against the stored users file.

    Returns ``(True, user_dict)`` on success, ``(False, {})`` on failure.
    """
    users = load_users()
    key   = username.strip().lower()
    user  = users.get(key)
    if user is None:
        return False, {}
    if _verify_password(password, user["hash"], user["salt"]):
        return True, user
    return False, {}


def create_user(
    username: str,
    password: str,
    role: str = "viewer",
    created_by: str = "system",
) -> bool:
    """Create a new user.  Returns ``False`` when the username already exists."""
    key   = username.strip().lower()
    users = load_users()
    if key in users:
        return False
    hashed, salt = _hash_password(password)
    users[key] = {
        "hash":       hashed,
        "salt":       salt,
        "role":       role,
        "created_by": created_by,
    }
    _save_users(users)
    return True


def delete_user(username: str) -> bool:
    """Delete *username*.  Returns ``False`` when the user does not exist."""
    key   = username.strip().lower()
    users = load_users()
    if key not in users:
        return False
    del users[key]
    _save_users(users)
    return True


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change password after verifying the old one.  Returns ``False`` on failure."""
    key   = username.strip().lower()
    users = load_users()
    user  = users.get(key)
    if user is None:
        return False
    if not _verify_password(old_password, user["hash"], user["salt"]):
        return False
    hashed, salt = _hash_password(new_password)
    users[key].update({"hash": hashed, "salt": salt})
    _save_users(users)
    return True


def reset_password(username: str, new_password: str) -> bool:
    """Admin password reset – does not require the old password."""
    key   = username.strip().lower()
    users = load_users()
    if key not in users:
        return False
    hashed, salt = _hash_password(new_password)
    users[key].update({"hash": hashed, "salt": salt})
    _save_users(users)
    return True


def ensure_default_admin() -> Optional[str]:
    """Create a default admin account when no users exist.

    Returns the generated one-time password (to be shown once in the UI),
    or ``None`` when users already exist.
    """
    if load_users():
        return None
    password = secrets.token_urlsafe(12)
    create_user("admin", password, role="admin", created_by="system")
    return password


# ---------------------------------------------------------------------------
# Roles – public API
# ---------------------------------------------------------------------------

def load_roles() -> Dict[str, Dict]:
    """Return the full roles dict from disk, falling back to :data:`DEFAULT_ROLES`."""
    if not ROLES_FILE.exists():
        return dict(DEFAULT_ROLES)
    try:
        return json.loads(ROLES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_ROLES)


def save_roles(roles: Dict[str, Dict]) -> None:
    """Persist *roles* to :data:`ROLES_FILE`."""
    ROLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROLES_FILE.write_text(json.dumps(roles, indent=2), encoding="utf-8")


def ensure_default_roles() -> None:
    """Create the default roles file when it does not yet exist."""
    if not ROLES_FILE.exists():
        save_roles(DEFAULT_ROLES)


def get_role_sections(role: str) -> List[str]:
    """Return the list of section identifiers accessible to *role*.

    Falls back to :data:`DEFAULT_ROLES` when the role is missing from disk,
    and ultimately to the ``viewer`` defaults.
    """
    roles = load_roles()
    entry = roles.get(role) or DEFAULT_ROLES.get(role) or DEFAULT_ROLES["viewer"]
    return list(entry.get("sections", []))


def can_access(role: str, section: str) -> bool:
    """Return ``True`` when *role* is permitted to access *section*."""
    return section in get_role_sections(role)


# ---------------------------------------------------------------------------
# Session tokens – persistent browser login
# ---------------------------------------------------------------------------

def _get_session_secret() -> bytes:
    """Return the HMAC/signing secret for session tokens.

    Reads ``SESSION_SECRET_KEY`` from the environment.  When the variable is
    absent, a random key is generated, persisted to the project ``.env`` file
    for convenience, and a warning is emitted.  In production **always** set
    ``SESSION_SECRET_KEY`` to a fixed, high-entropy value so that sessions
    survive server restarts and are consistent across replicas.
    """
    key = os.environ.get("SESSION_SECRET_KEY", "")
    if key:
        return key.encode("utf-8")

    # Dev-mode fallback: generate a key and persist it so subsequent runs reuse it.
    warnings.warn(
        "SESSION_SECRET_KEY is not set.  A random key has been generated and "
        "saved to .env – existing sessions will be invalidated on restart.  "
        "Set SESSION_SECRET_KEY to a fixed 64-char hex value in production.",
        stacklevel=2,
    )
    new_key = secrets.token_hex(32)
    env_path = Path(__file__).parent.parent.parent / ".env"
    try:
        text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        if "SESSION_SECRET_KEY=" not in text:
            with env_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\nSESSION_SECRET_KEY={new_key}\n")
    except OSError:
        pass
    os.environ["SESSION_SECRET_KEY"] = new_key
    return new_key.encode("utf-8")


def _hash_token(raw_token: str) -> str:
    """Return the SHA-256 hex digest of *raw_token* for safe server-side storage."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _load_sessions() -> Dict[str, Dict]:
    """Load the active sessions from disk.  Returns ``{}`` on error."""
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sessions(sessions: Dict[str, Dict]) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def create_session_token(
    username: str,
    role: str,
    ttl_days: int = _SESSION_TTL_DAYS_DEFAULT,
) -> str:
    """Create a persistent session token for *username*.

    Returns the raw (unhashed) token that must be stored in the browser cookie.
    Only the SHA-256 hash is persisted server-side in ``sessions.json``.
    Expired entries are pruned automatically on each call.
    """
    raw_token = secrets.token_hex(32)
    token_hash = _hash_token(raw_token)
    now = datetime.datetime.utcnow()
    expires_at = (now + datetime.timedelta(days=ttl_days)).isoformat()
    sessions = _load_sessions()
    # Prune expired entries while we're here
    now_iso = now.isoformat()
    sessions = {h: e for h, e in sessions.items() if e.get("expires_at", "") > now_iso}
    sessions[token_hash] = {
        "username":   username.strip().lower(),
        "role":       role,
        "expires_at": expires_at,
    }
    _save_sessions(sessions)
    return raw_token


def validate_session_token(raw_token: str) -> Optional[Tuple[str, str]]:
    """Validate *raw_token* and return ``(username, role)`` or ``None``.

    Returns ``None`` when the token is unknown, malformed, or expired.
    """
    if not raw_token or not isinstance(raw_token, str):
        return None
    token_hash = _hash_token(raw_token.strip())
    sessions = _load_sessions()
    entry = sessions.get(token_hash)
    if entry is None:
        return None
    if datetime.datetime.utcnow().isoformat() > entry.get("expires_at", ""):
        # Expired – clean up
        sessions.pop(token_hash, None)
        _save_sessions(sessions)
        return None
    return entry["username"], entry["role"]


def revoke_session_token(raw_token: str) -> None:
    """Revoke a single session token (call on explicit logout)."""
    if not raw_token:
        return
    token_hash = _hash_token(raw_token.strip())
    sessions = _load_sessions()
    if token_hash in sessions:
        sessions.pop(token_hash)
        _save_sessions(sessions)


def revoke_user_sessions(username: str) -> None:
    """Revoke **all** active sessions for *username*.

    Called when the user's password is changed or the account is deleted so
    that any previously issued tokens (e.g. on other devices / browsers) are
    immediately invalidated.
    """
    key = username.strip().lower()
    sessions = _load_sessions()
    pruned = {h: e for h, e in sessions.items() if e.get("username", "") != key}
    if len(pruned) != len(sessions):
        _save_sessions(pruned)


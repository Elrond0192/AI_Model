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

import hashlib
import hmac
import json
import os
import secrets
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
# Storage
# ---------------------------------------------------------------------------

#: Path of the users file.  Can be overridden via BASKETBALL_AI_USERS_FILE env var.
USERS_FILE = Path(
    os.environ.get(
        "BASKETBALL_AI_USERS_FILE",
        str(Path(__file__).parent.parent.parent / "users.json"),
    )
)

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
# Public API
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

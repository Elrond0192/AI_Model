"""User authentication and management for the Basketball Performance AI platform.

Users are stored in a JSON file (``users.json``) at the project root.  The file
is **not** committed to git (see ``.gitignore``); on first run a default admin
account is created automatically and the temporary password is displayed once.

Roles
-----
- ``"admin"`` – full access including user management.
- ``"analyst"`` – access to all analytic features, read-only user list.
- ``"viewer"`` – read-only access to predictions and scenarios.

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
from typing import Dict, Optional, Tuple

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

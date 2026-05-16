"""Tests for authentication helpers: lockout, session tokens, password changes."""
from __future__ import annotations
import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _clean_lockout_registry():
    """Clear the in-process lockout registry before each test."""
    from basketball_ai.auth import auth as auth_mod
    auth_mod._lockout_registry.clear()
    yield
    auth_mod._lockout_registry.clear()


@pytest.fixture()
def tmp_users_file(tmp_path, monkeypatch):
    """Redirect USERS_FILE to a fresh temp file for each test."""
    from basketball_ai.auth import auth as auth_mod
    users_file = tmp_path / "users.json"
    monkeypatch.setattr(auth_mod, "USERS_FILE", users_file)
    return users_file


class TestAccountLockout:
    def test_no_lockout_on_first_failure(self, tmp_users_file):
        from basketball_ai.auth.auth import check_credentials, create_user
        create_user("alice", "secret123", role="viewer", created_by="test")
        ok, info = check_credentials("alice", "wrong")
        assert not ok
        assert "locked_until" not in info

    def test_lockout_after_max_attempts(self, tmp_users_file):
        from basketball_ai.auth import auth as auth_mod
        from basketball_ai.auth.auth import check_credentials, create_user
        create_user("bob", "securepass", role="viewer", created_by="test")
        for _ in range(auth_mod._MAX_FAILED_ATTEMPTS):
            ok, _ = check_credentials("bob", "wrong")
            assert not ok
        # Now should be locked
        ok, info = check_credentials("bob", "securepass")  # even with correct pw
        assert not ok
        assert "locked_until" in info

    def test_lockout_resets_on_success(self, tmp_users_file):
        from basketball_ai.auth import auth as auth_mod
        from basketball_ai.auth.auth import check_credentials, create_user
        create_user("carol", "mypassword", role="viewer", created_by="test")
        # Fail 2 times (not enough to lock)
        for _ in range(auth_mod._MAX_FAILED_ATTEMPTS - 2):
            check_credentials("carol", "wrong")
        # Succeed – should clear counter
        ok, _ = check_credentials("carol", "mypassword")
        assert ok
        # Now fail again – counter should start fresh
        for _ in range(auth_mod._MAX_FAILED_ATTEMPTS - 1):
            ok, info = check_credentials("carol", "wrong")
            assert not ok
            assert "locked_until" not in info

    def test_unknown_user_increments_counter(self, tmp_users_file):
        from basketball_ai.auth import auth as auth_mod
        from basketball_ai.auth.auth import check_credentials
        for _ in range(auth_mod._MAX_FAILED_ATTEMPTS):
            ok, _ = check_credentials("ghost_user", "anything")
            assert not ok
        key = "ghost_user"
        fail_count, locked_until = auth_mod._lockout_registry.get(key, (0, ""))
        assert fail_count >= auth_mod._MAX_FAILED_ATTEMPTS


class TestAdminCredentialsFile:
    def test_ensure_admin_writes_file(self, tmp_path, monkeypatch):
        from basketball_ai.auth import auth as auth_mod
        users_file = tmp_path / "users.json"
        creds_file = tmp_path / ".admin_credentials"
        monkeypatch.setattr(auth_mod, "USERS_FILE", users_file)
        monkeypatch.setattr(auth_mod, "ADMIN_CREDENTIALS_FILE", creds_file)
        result = auth_mod.ensure_default_admin()
        assert result == str(creds_file)
        assert creds_file.exists()
        content = creds_file.read_text(encoding="utf-8")
        assert "username: admin" in content
        assert "password:" in content

    def test_ensure_admin_noop_when_users_exist(self, tmp_path, monkeypatch):
        from basketball_ai.auth import auth as auth_mod
        users_file = tmp_path / "users.json"
        creds_file = tmp_path / ".admin_credentials"
        monkeypatch.setattr(auth_mod, "USERS_FILE", users_file)
        monkeypatch.setattr(auth_mod, "ADMIN_CREDENTIALS_FILE", creds_file)
        # Create a user first
        auth_mod.create_user("admin", "pass", role="admin", created_by="test")
        result = auth_mod.ensure_default_admin()
        assert result is None

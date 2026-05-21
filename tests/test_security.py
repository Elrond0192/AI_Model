"""Tests for enterprise security improvements."""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Password policy tests
# ---------------------------------------------------------------------------

def test_password_policy_too_short():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("Short1!")
    assert any("12" in e or "characters" in e for e in errors)


def test_password_policy_valid():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("SecureP@ssword99")
    assert errors == []


def test_password_policy_no_uppercase():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("lowercase1special!")
    assert any("uppercase" in e for e in errors)


def test_password_policy_no_digit():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("NoDigitAtAll!!")
    assert any("digit" in e for e in errors)


def test_password_policy_no_special():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("NoSpecialChar123")
    assert any("special" in e for e in errors)


def test_password_policy_contains_username():
    from basketball_ai.auth.auth import _validate_password_policy
    errors = _validate_password_policy("adminPassw0rd!", username="admin")
    assert any("username" in e for e in errors)


# ---------------------------------------------------------------------------
# PII masking tests
# ---------------------------------------------------------------------------

def test_pii_masking_viewer():
    from basketball_ai.api.middleware.pii import mask_player_pii
    player = {"id": 1, "name": "John", "birth_date": "1990-01-01", "weight_kg": 85, "height_cm": 195}
    masked = mask_player_pii(player, role="viewer")
    assert masked["birth_date"] is None
    assert masked["weight_kg"] is None
    assert masked["height_cm"] is None
    assert masked["name"] == "John"  # name not masked


def test_pii_masking_admin():
    from basketball_ai.api.middleware.pii import mask_player_pii
    player = {"id": 1, "name": "John", "birth_date": "1990-01-01", "weight_kg": 85}
    masked = mask_player_pii(player, role="admin")
    assert masked["birth_date"] == "1990-01-01"
    assert masked["weight_kg"] == 85


def test_pii_masking_analyst():
    from basketball_ai.api.middleware.pii import mask_player_pii
    player = {"id": 1, "name": "John", "height_cm": 195}
    masked = mask_player_pii(player, role="analyst")
    assert masked["height_cm"] == 195


def test_pii_masking_list():
    from basketball_ai.api.middleware.pii import mask_players_pii
    players = [
        {"id": 1, "birth_date": "1990-01-01"},
        {"id": 2, "birth_date": "1988-06-15"},
    ]
    masked = mask_players_pii(players, role="viewer")
    assert all(p["birth_date"] is None for p in masked)


# ---------------------------------------------------------------------------
# Tenant isolation tests
# ---------------------------------------------------------------------------

def test_filter_by_tenant_no_column():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant
    df = pd.DataFrame({"id": [1, 2, 3]})
    result = filter_by_tenant(df, "t1")
    assert len(result) == 3  # no tenant_id column → unchanged


def test_filter_by_tenant_with_column():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant
    df = pd.DataFrame({"id": [1, 2], "tenant_id": ["t1", "t2"]})
    result = filter_by_tenant(df, "t1")
    assert list(result["id"]) == [1]


def test_filter_by_tenant_default_passthrough():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant
    df = pd.DataFrame({"id": [1, 2], "tenant_id": ["t1", "t2"]})
    result = filter_by_tenant(df, "default")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# JWT multi-secret tests
# ---------------------------------------------------------------------------

def test_get_jwt_secrets_single(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "mysecret")
    monkeypatch.delenv("JWT_SECRETS", raising=False)
    from basketball_ai.api.routes import auth as auth_mod
    import importlib
    importlib.reload(auth_mod)
    secrets = auth_mod._get_jwt_secrets()
    assert len(secrets) == 1
    assert secrets[0] == ("default", "mysecret")


def test_get_jwt_secrets_multi(monkeypatch):
    import json
    monkeypatch.setenv("JWT_SECRETS", json.dumps([
        {"kid": "k1", "secret": "s1"},
        {"kid": "k2", "secret": "s2"},
    ]))
    monkeypatch.delenv("JWT_SECRET", raising=False)
    from basketball_ai.api.routes import auth as auth_mod
    import importlib
    importlib.reload(auth_mod)
    secrets = auth_mod._get_jwt_secrets()
    assert len(secrets) == 2
    assert secrets[0] == ("k1", "s1")

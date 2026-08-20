"""Tests for enterprise security improvements."""
from __future__ import annotations

import json
import time


def test_password_policy_too_short():
    from basketball_ai.auth.auth import _validate_password_policy

    errors = _validate_password_policy("Short1!")
    assert any("12" in error or "characters" in error for error in errors)


def test_password_policy_valid():
    from basketball_ai.auth.auth import _validate_password_policy

    assert _validate_password_policy("SecureP@ssword99") == []


def test_password_policy_no_uppercase():
    from basketball_ai.auth.auth import _validate_password_policy

    assert any("uppercase" in error for error in _validate_password_policy("lowercase1special!"))


def test_password_policy_no_digit():
    from basketball_ai.auth.auth import _validate_password_policy

    assert any("digit" in error for error in _validate_password_policy("NoDigitAtAll!!"))


def test_password_policy_no_special():
    from basketball_ai.auth.auth import _validate_password_policy

    assert any("special" in error for error in _validate_password_policy("NoSpecialChar123"))


def test_password_policy_contains_username():
    from basketball_ai.auth.auth import _validate_password_policy

    assert any("username" in error for error in _validate_password_policy("adminPassw0rd!", username="admin"))


def test_pii_masking_viewer():
    from basketball_ai.api.middleware.pii import mask_player_pii

    player = {"id": 1, "name": "John", "birth_date": "1990-01-01", "weight_kg": 85, "height_cm": 195}
    masked = mask_player_pii(player, role="viewer")
    assert masked["birth_date"] is None
    assert masked["weight_kg"] is None
    assert masked["height_cm"] is None
    assert masked["name"] == "John"


def test_pii_masking_admin():
    from basketball_ai.api.middleware.pii import mask_player_pii

    player = {"id": 1, "name": "John", "birth_date": "1990-01-01", "weight_kg": 85}
    masked = mask_player_pii(player, role="admin")
    assert masked["birth_date"] == "1990-01-01"
    assert masked["weight_kg"] == 85


def test_pii_masking_analyst():
    from basketball_ai.api.middleware.pii import mask_player_pii

    player = {"id": 1, "name": "John", "height_cm": 195}
    assert mask_player_pii(player, role="analyst")["height_cm"] == 195


def test_pii_masking_list():
    from basketball_ai.api.middleware.pii import mask_players_pii

    players = [
        {"id": 1, "birth_date": "1990-01-01"},
        {"id": 2, "birth_date": "1988-06-15"},
    ]
    assert all(player["birth_date"] is None for player in mask_players_pii(players, role="viewer"))


def test_filter_by_tenant_no_column():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant

    frame = pd.DataFrame({"id": [1, 2, 3]})
    assert len(filter_by_tenant(frame, "t1")) == 3


def test_filter_by_tenant_with_column():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant

    frame = pd.DataFrame({"id": [1, 2], "tenant_id": ["t1", "t2"]})
    assert list(filter_by_tenant(frame, "t1")["id"]) == [1]


def test_filter_by_tenant_default_passthrough():
    import pandas as pd
    from basketball_ai.api.middleware.tenant import filter_by_tenant

    frame = pd.DataFrame({"id": [1, 2], "tenant_id": ["t1", "t2"]})
    assert len(filter_by_tenant(frame, "default")) == 2


def test_decode_jwt_single_secret(monkeypatch):
    import jwt
    from basketball_ai.api.main import _decode_jwt

    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.delenv("JWT_SECRETS", raising=False)
    token = jwt.encode({"sub": "service", "exp": int(time.time()) + 60}, "primary", algorithm="HS256")
    assert _decode_jwt(token, "primary")["sub"] == "service"


def test_decode_jwt_rotated_secret(monkeypatch):
    import jwt
    from basketball_ai.api.main import _decode_jwt

    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv(
        "JWT_SECRETS",
        json.dumps([
            {"kid": "old", "secret": "old-secret"},
            {"kid": "new", "secret": "new-secret"},
        ]),
    )
    token = jwt.encode({"sub": "service", "exp": int(time.time()) + 60}, "new-secret", algorithm="HS256")
    assert _decode_jwt(token, "primary")["sub"] == "service"

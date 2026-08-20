import json
import pytest

def test_profiles_require_explicit_selection(tmp_path, monkeypatch):
    from basketball_ai.data import connection_profiles as profiles
    monkeypatch.setattr(profiles, "PROFILE_FILE", tmp_path / "profiles.json")
    profiles.PROFILE_FILE.write_text(json.dumps({"one":{"host":"db1"},"two":{"host":"db2"}}))
    monkeypatch.delenv("DATABASE_PROFILE", raising=False)
    with pytest.raises(RuntimeError): profiles.active_profile_name()

def test_profile_url_encodes_credentials(tmp_path, monkeypatch):
    from basketball_ai.data import connection_profiles as profiles
    monkeypatch.setattr(profiles, "PROFILE_FILE", tmp_path / "profiles.json")
    profiles.save_profile("prod", {"host":"host.docker.internal","port":5432,"database":"hoopmetrics","user":"ai user","password":"p@ss"})
    assert profiles.profile_url("prod") == "postgresql+psycopg://ai+user:p%40ss@host.docker.internal:5432/hoopmetrics"

def test_loader_rejects_unsafe_schema(monkeypatch):
    import basketball_ai.data.postgres_loader as loader
    monkeypatch.setattr(loader, "get_engine", lambda _: object())
    with pytest.raises(ValueError): loader.load_all_data("postgresql+psycopg://u:p@h/d", 'ai_source;DROP')

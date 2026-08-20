"""Tests for enterprise data-layer components."""
from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# IngestionTracker tests
# ---------------------------------------------------------------------------

def _make_tracker(tmp_path):
    from basketball_ai.data.ingestion import IngestionTracker
    return IngestionTracker(db_path=tmp_path / "ingestion.db")


def test_ingestion_tracker_new_game_should_process(tmp_path):
    t = _make_tracker(tmp_path)
    assert t.should_process("game_001") is True


def test_ingestion_tracker_done_not_reprocessed(tmp_path):
    t = _make_tracker(tmp_path)
    assert t.should_process("game_002") is True
    t.mark_done("game_002")
    assert t.should_process("game_002") is False


def test_ingestion_tracker_failed_retries(tmp_path):
    from basketball_ai.data.ingestion import _MAX_RETRIES, STATUS_DEAD
    t = _make_tracker(tmp_path)
    t.should_process("game_003")
    for _ in range(_MAX_RETRIES):
        t.mark_failed("game_003", "boom")
    assert t.get_status("game_003") == STATUS_DEAD
    assert t.should_process("game_003") is False


def test_ingestion_tracker_stats(tmp_path):
    t = _make_tracker(tmp_path)
    t.should_process("g1")
    t.mark_done("g1")
    t.should_process("g2")
    t.mark_failed("g2", "err")
    s = t.stats()
    assert s.get("DONE", 0) >= 1


def test_ingestion_tracker_list_failed(tmp_path):
    t = _make_tracker(tmp_path)
    t.should_process("gx")
    t.mark_failed("gx", "some error")
    failed = t.list_failed()
    assert any(r["game_id"] == "gx" for r in failed)


# ---------------------------------------------------------------------------
# NameResolutionCache tests
# ---------------------------------------------------------------------------

def _make_cache(tmp_path):
    from basketball_ai.data.name_cache import NameResolutionCache
    return NameResolutionCache(path=tmp_path / "name_cache.json", ttl=3600)


def test_name_cache_miss_returns_none(tmp_path):
    c = _make_cache(tmp_path)
    assert c.resolve_player("Unknown Player") is None


def test_name_cache_set_and_resolve(tmp_path):
    c = _make_cache(tmp_path)
    c.set_player("LeBron James", 42)
    assert c.resolve_player("LeBron James") == 42
    # case insensitive
    assert c.resolve_player("lebron james") == 42


def test_name_cache_team_set_and_resolve(tmp_path):
    c = _make_cache(tmp_path)
    c.set_team("Lakers", 10)
    assert c.resolve_team("Lakers") == 10
    assert c.resolve_team("lakers") == 10


def test_name_cache_persist_reload(tmp_path):
    c = _make_cache(tmp_path)
    c.set_player("Steph Curry", 99)
    c2 = _make_cache(tmp_path)
    assert c2.resolve_player("Steph Curry") == 99


def test_name_cache_ttl_expiry(tmp_path):
    from basketball_ai.data.name_cache import NameResolutionCache
    c = NameResolutionCache(path=tmp_path / "nc.json", ttl=0)  # TTL=0 → instant expiry
    c.set_player("Expired Player", 1)
    time.sleep(0.01)
    assert c.resolve_player("Expired Player") is None


def test_name_cache_stats(tmp_path):
    c = _make_cache(tmp_path)
    c.set_player("A", 1)
    c.set_team("B", 2)
    s = c.stats()
    assert s["players"]["valid"] == 1
    assert s["teams"]["valid"] == 1


def test_name_cache_invalidate(tmp_path):
    c = _make_cache(tmp_path)
    c.set_player("X", 7)
    c.invalidate_player("X")
    assert c.resolve_player("X") is None


# ---------------------------------------------------------------------------
# constants.py new additions
# ---------------------------------------------------------------------------

def test_constants_supported_leagues():
    from basketball_ai.constants import SUPPORTED_LEAGUES
    assert "ITA1" in SUPPORTED_LEAGUES
    assert "GRC1" in SUPPORTED_LEAGUES


def test_constants_supported_seasons():
    from basketball_ai.constants import SUPPORTED_SEASONS
    assert SUPPORTED_SEASONS and all(isinstance(season, int) for season in SUPPORTED_SEASONS)


def test_is_safe_identifier():
    from basketball_ai.constants import is_safe_identifier
    assert is_safe_identifier("ITA1_2024_25") is True
    assert is_safe_identifier("") is False
    assert is_safe_identifier("table; DROP TABLE x") is False
    assert is_safe_identifier("1invalid") is False

from __future__ import annotations

import pandas as pd

from basketball_ai.data import postgres_loader as loader


class _Engine:
    def dispose(self):
        pass


def test_predictive_scenario_does_not_open_database(monkeypatch):
    monkeypatch.setattr(loader, "get_engine", lambda url=None: (_ for _ in ()).throw(AssertionError("database opened")))
    result = loader.load_scenario_feeds(
        "player_team",
        player_ids=[1],
        team_ids=[2],
        league_keys=["ITA1"],
        season=2025,
        competition="RS",
    )
    assert all(frame.empty for frame in result.values())


def test_advanced_scenario_query_is_explicit_and_bounded(monkeypatch):
    statements = []
    monkeypatch.setattr(loader, "get_engine", lambda url=None: _Engine())
    monkeypatch.setattr(loader, "_scenario_cache", {})

    def fake_read_sql(statement, engine, params):
        statements.append((str(statement), params))
        return pd.DataFrame()

    monkeypatch.setattr(loader.pd, "read_sql", fake_read_sql)
    result = loader.load_scenario_feeds(
        "defensive_matchup",
        player_ids=[11, 22],
        team_ids=[33],
        league_keys=["EL"],
        season=2025,
        competition="PO",
    )

    assert len(statements) == 2
    for sql, params in statements:
        assert "SELECT *" not in sql.upper()
        assert "season BETWEEN" in sql
        assert "upper(competition)" in sql
        assert "league_key IN" in sql
        assert params["season_min"] == 2022
        assert params["season_max"] == 2025
        assert params["competition"] == "PO"
    assert result["play_type_stats"].empty


def test_scenario_feed_cache_avoids_second_query(monkeypatch):
    calls = 0
    monkeypatch.setattr(loader, "get_engine", lambda url=None: _Engine())
    monkeypatch.setattr(loader, "_scenario_cache", {})

    def fake_read_sql(statement, engine, params):
        nonlocal calls
        calls += 1
        return pd.DataFrame()

    monkeypatch.setattr(loader.pd, "read_sql", fake_read_sql)
    kwargs = {
        "player_ids": [11],
        "team_ids": [],
        "league_keys": ["ITA1"],
        "season": 2025,
        "competition": "RS",
    }
    loader.load_scenario_feeds("shot_profile_counterfactual", **kwargs)
    loader.load_scenario_feeds("shot_profile_counterfactual", **kwargs)
    assert calls == 1

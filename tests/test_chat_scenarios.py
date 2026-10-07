"""Regression tests for the composable Chat V3 scenario surface."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from basketball_ai.api.scenario_contracts_v2 import ScenarioRequestV2
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine


def _data() -> dict:
    player_rows = [
        # Subject: distinct RS/PO values in the same league/season.
        {"player_id": 1, "team_id": 101, "league_id": 10, "season": 2024, "competition": "RS", "rating": 7.0, "games_played": 30, "minutes_per_game": 28, "points": 15, "assists": 5, "ts_pct": .58, "usg_pct": .24, "bpm": 3.0, "dbpm": 0.5, "three_point_pct": .37, "three_par": .40},
        {"player_id": 1, "team_id": 201, "league_id": 20, "season": 2024, "competition": "RS", "rating": 6.2, "games_played": 18, "minutes_per_game": 23, "points": 12, "assists": 4, "ts_pct": .55, "usg_pct": .21, "bpm": 1.2, "dbpm": 0.3, "three_point_pct": .35, "three_par": .46},
        {"player_id": 1, "team_id": 101, "league_id": 10, "season": 2024, "competition": "PO", "rating": 7.5, "games_played": 8, "minutes_per_game": 31, "points": 17, "assists": 5.5, "ts_pct": .60, "usg_pct": .27, "bpm": 4.0, "dbpm": 0.8, "three_point_pct": .39, "three_par": .42, "ruolo_combinato": "PRIMARY CREATOR"},
        {"player_id": 2, "team_id": 101, "league_id": 10, "season": 2024, "competition": "PO", "rating": 6.8, "games_played": 8, "minutes_per_game": 26, "points": 12, "assists": 2, "ts_pct": .61, "usg_pct": .20, "bpm": 1.5, "dbpm": 1.1, "three_point_pct": .41, "three_par": .52},
    ]
    # Real consecutive cross-league transitions used by the empirical transfer layer.
    for offset in range(6):
        pid = 20 + offset
        player_rows.extend([
            {"player_id": pid, "team_id": 100 + offset, "league_id": 10, "season": 2023, "competition": "RS", "rating": 7.0 + offset * .05, "games_played": 25},
            {"player_id": pid, "team_id": 200 + offset, "league_id": 20, "season": 2024, "competition": "RS", "rating": 6.6 + offset * .05, "games_played": 24},
        ])
    players = pd.DataFrame([
        {"id": 1, "name": "Alpha", "position": "PG", "current_team_id": 101, "current_league_id": 10},
        {"id": 2, "name": "Beta", "position": "SG", "current_team_id": 101, "current_league_id": 10},
        *[{"id": 20 + i, "name": f"Mover {i}", "position": "SF", "current_team_id": 200 + i, "current_league_id": 20} for i in range(6)],
    ])
    teams = pd.DataFrame([
        {"id": 101, "name": "Source Club", "league_id": 10, "global_id": "T-SOURCE", "pace": 72.0, "offensive_rating": 112.0, "defensive_rating": 108.0},
        {"id": 201, "name": "Target Club", "league_id": 20, "global_id": "T-TARGET", "pace": 76.0, "offensive_rating": 116.0, "defensive_rating": 110.0},
    ])
    leagues = pd.DataFrame([
        {"id": 10, "name": "League A", "league_key": "A", "competitiveness_score": .75},
        {"id": 20, "name": "EuroLeague", "league_key": "EL", "competitiveness_score": 1.0},
    ])
    team_history = pd.DataFrame([
        {"team_id": 101, "league_id": 10, "season": 2024, "competition": "RS", "pace": 72.0, "offensive_rating": 112.0, "defensive_rating": 108.0, "net_rtg": 4.0, "three_point_attempt_rate": .36, "assists_per_game": 21.0, "star_player_usage": .28},
        {"team_id": 101, "league_id": 10, "season": 2024, "competition": "PO", "pace": 70.0, "offensive_rating": 110.0, "defensive_rating": 106.0, "net_rtg": 4.0, "three_point_attempt_rate": .34, "assists_per_game": 20.0, "star_player_usage": .30},
        {"team_id": 201, "league_id": 20, "season": 2024, "competition": "RS", "pace": 76.0, "offensive_rating": 116.0, "defensive_rating": 110.0, "net_rtg": 6.0, "three_point_attempt_rate": .42, "assists_per_game": 24.0, "star_player_usage": .27},
    ])
    player_dict = {int(row["id"]): {**row.to_dict(), "global_id": f"P-{int(row['id'])}"} for _, row in players.iterrows()}
    team_dict = {int(row["id"]): row.to_dict() for _, row in teams.iterrows()}
    league_dict = {int(row["id"]): row.to_dict() for _, row in leagues.iterrows()}
    return {
        "player_stats": pd.DataFrame(player_rows),
        "team_season_stats": team_history,
        "players": players,
        "teams": teams,
        "leagues": leagues,
        "team_player_relations": pd.DataFrame(columns=["player_id", "team_id", "season", "role"]),
        "player_dict": player_dict,
        "team_dict": team_dict,
        "league_dict": league_dict,
        "league_teams": {10: [101], 20: [201]},
    }


def test_contract_is_open_to_normalised_future_competitions():
    request = ScenarioRequestV2(
        scenario="player_competition",
        season=2025,
        competition="Final Four",
        style_overrides={"pace": "higher"},
    )
    assert request.competition == "FINAL_FOUR"
    assert request.style_overrides["pace"] == "higher"


def test_player_intelligence_discovers_all_current_league_contexts_when_unspecified():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS", "parameters": {"question_key": "current_level"}},
        [1], [], None, None,
    )
    contexts = result["result"]["contexts"]
    assert [item["league_key"] for item in contexts] == ["A", "EL"]
    assert all("player_competition" in item["analyses"] for item in contexts)
    assert all("model_evidence" in item["analyses"] for item in contexts)
    assert all(item["analyses"]["player_competition"]["result"]["latest"]["rating"] in {7.0, 6.2} for item in contexts)

def test_player_intelligence_filters_current_contexts_by_explicit_team():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS", "parameters": {"question_key": "current_level"}},
        [1], [101], None, None,
    )
    contexts = result["result"]["contexts"]
    assert [item["league_key"] for item in contexts] == ["A"]
    assert all("player_competition" in item["analyses"] for item in contexts)


def test_player_competition_keeps_playoffs_isolated():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "player_competition", "season": 2024, "competition": "PO", "comparison_competition": "RS"},
        [1], [], 10, None,
    )
    assert result["result"]["latest"]["rating"] == 7.5
    assert result["result"]["comparison"]["rating"] == 7.0
    assert result["result"]["deltas"]["rating"] == 0.5
    assert result["evidence"][0]["games"] == 8


def test_playoff_role_uses_po_evidence_not_rs_alias():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "playoff_role", "season": 2024, "competition": "PO"},
        [1], [], 10, None,
    )
    assert result["result"]["playoff_rating"] == 7.5
    assert result["result"]["rs_to_po_rating_delta"] == 0.5
    assert result["result"]["role_signal"] in {"PRIMARY_OPTION", "SECONDARY_OPTION"}


def test_cross_league_transfer_is_learned_from_real_transitions():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "league_transfer", "season": 2024, "competition": "RS", "target_competition": "RS"},
        [1], [], 10, 20,
    )
    assert result["support"]["method"] == "empirical_cross_league_transition"
    assert result["support"]["samples"] == 6
    assert result["result"]["projected_rating"] < result["result"]["source_rating"]


def test_pair_fit_penalises_usage_overlap_and_reports_components():
    engine = ChatScenarioEngine(SimpleNamespace(), _data())
    result = engine.evaluate(
        {"scenario": "player_pair", "season": 2024, "competition": "PO"},
        [1, 2], [], 10, None,
    )
    pair = result["result"]["pair_fit"]
    assert 0.0 <= pair["score"] <= 1.0
    assert "usage_overlap" in pair
    assert "spacing" in pair
    assert result["support"]["method"] == "analytical_pair_complementarity"


def test_scenario_api_resolves_global_ids_and_returns_structured_payload(monkeypatch):
    import basketball_ai.api.routes.scenarios_v2 as route

    seen = {}

    class FakeEngine:
        def __init__(self, ensemble, data):
            seen["ensemble"] = ensemble

        def evaluate(self, spec, player_ids, team_ids, source_league, target_league):
            seen.update({"spec": spec, "players": player_ids, "teams": team_ids, "source": source_league, "target": target_league})
            return {"result": {"ok": True}, "support": {"method": "test"}, "evidence": [], "limitations": []}

    monkeypatch.setattr(route, "ChatScenarioEngine", FakeEngine)
    monkeypatch.setattr(route, "resolve_league_id", lambda data, value: 10 if value == "ITA1" else 20)
    monkeypatch.setattr(
        route,
        "load_scenario_feeds",
        lambda *args, **kwargs: {"pbp_events": pd.DataFrame()},
    )
    app = FastAPI()
    app.include_router(route.router)
    data = _data()
    data["player_dict"][1]["global_id"] = "PLAYER-GLOBAL"
    data["team_dict"][201]["global_id"] = "TEAM-GLOBAL"
    app.state.data = data
    app.state.engine = SimpleNamespace(ensemble=object())
    app.state.model_metadata = {"model_run_id": "r", "model_version": "2.3.0", "feature_version": "scenario-v1", "data_cutoff": "2026-06-30"}
    response = TestClient(app).post(
        "/api/v2/scenarios/evaluate",
        json={"scenario": "player_team", "player_global_ids": ["PLAYER-GLOBAL"], "team_global_ids": ["TEAM-GLOBAL"], "source_league": "ITA1", "target_league": "EL", "season": 2025, "competition": "RS"},
    )
    assert response.status_code == 200
    assert response.json()["result"] == {"ok": True}
    assert seen["players"] == [1]
    assert seen["teams"] == [201]
    assert (seen["source"], seen["target"]) == (10, 20)

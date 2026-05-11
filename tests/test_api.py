"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Build a minimal app with pre-loaded test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with minimal test data pre-loaded."""
    # Build tiny dataset
    leagues = pd.DataFrame([{
        "id": 1, "name": "Premier League", "country": "England",
        "tier": 1, "competitiveness_score": 0.98,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i+1}", "league_id": 1,
        "playing_style": "possession", "formation": "4-3-3",
        "avg_possession": 55.0, "pressing_intensity": 7.0,
        "defensive_line": 6.5, "passing_tempo": 7.5, "league_tier": 1,
    } for i in range(10)])
    positions = ["GK", "CB", "FB", "CM", "AM", "W", "ST"]
    players = pd.DataFrame([{
        "id": i + 1, "name": f"Player {i+1}",
        "age": 22 + (i % 14), "position": positions[i % len(positions)],
        "nationality": "English", "foot": "right",
        "height": 180.0, "weight": 76.0,
        "current_team_id": (i % 10) + 1, "current_league_id": 1,
    } for i in range(50)])
    np.random.seed(42)
    stat_rows = []
    for _, p in players.iterrows():
        for season in [2022, 2023, 2024]:
            m90 = np.random.uniform(10, 30)
            stat_rows.append({
                "player_id": int(p["id"]), "season": season,
                "team_id": int(p["current_team_id"]), "league_id": 1,
                "goals": round(np.random.uniform(0, 10), 2),
                "assists": round(np.random.uniform(0, 8), 2),
                "matches_played": int(np.random.randint(15, 35)),
                "minutes": round(m90 * 90, 1),
                "pass_accuracy": round(np.random.uniform(65, 90), 1),
                "dribbles": round(np.random.uniform(0.5, 3.5), 2),
                "tackles": round(np.random.uniform(0.3, 3.0), 2),
                "interceptions": round(np.random.uniform(0.2, 2.0), 2),
                "aerial_duels_won": round(np.random.uniform(0.3, 4.0), 2),
                "rating": round(np.random.uniform(5.5, 8.5), 2),
                "xG": round(np.random.uniform(0, 8), 3),
                "xA": round(np.random.uniform(0, 6), 3),
                "progressive_passes": round(np.random.uniform(1, 7), 2),
                "key_passes": round(np.random.uniform(0.3, 3.0), 2),
            })
    player_stats = pd.DataFrame(stat_rows)
    rel = pd.DataFrame([{
        "team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
        "season": 2024, "role": "starter",
        "jersey_number": (int(p["id"]) % 99) + 1,
    } for _, p in players.iterrows()])

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}
    league_teams = {}
    for _, t in teams.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    data = {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": player_stats, "team_player_relations": rel,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": league_teams,
    }

    # Train a tiny ensemble
    from src.models.ensemble import EnsembleModel
    from src.scenarios.engine import WhatIfEngine

    ensemble = EnsembleModel()
    ensemble.train(data)
    engine = WhatIfEngine(ensemble, data)

    # Patch app_state directly
    from src.api import main as api_main
    api_main.app_state["data"] = data
    api_main.app_state["engine"] = engine

    from src.api.main import app
    with TestClient(app, raise_server_exceptions=True) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class TestPlayersAPI:
    def test_list_players(self, client):
        resp = client.get("/api/v1/players?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_list_players_filter_position(self, client):
        resp = client.get("/api/v1/players?position=ST&limit=20")
        assert resp.status_code == 200
        for p in resp.json():
            assert p["position"] == "ST"

    def test_get_player(self, client):
        resp = client.get("/api/v1/players/1")
        assert resp.status_code == 200
        assert resp.json()["id"] == 1

    def test_get_player_not_found(self, client):
        resp = client.get("/api/v1/players/99999")
        assert resp.status_code == 404

    def test_get_player_stats(self, client):
        resp = client.get("/api/v1/players/1/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert isinstance(stats, list)
        assert all("rating" in s for s in stats)

    def test_get_player_profile(self, client):
        resp = client.get("/api/v1/players/1/profile")
        assert resp.status_code == 200
        profile = resp.json()
        assert "form_score" in profile
        assert "peak_rating" in profile
        assert "player" in profile


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

class TestTeamsAPI:
    def test_list_teams(self, client):
        resp = client.get("/api/v1/teams?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    def test_get_team(self, client):
        resp = client.get("/api/v1/teams/1")
        assert resp.status_code == 200
        assert resp.json()["id"] == 1

    def test_get_team_not_found(self, client):
        resp = client.get("/api/v1/teams/99999")
        assert resp.status_code == 404

    def test_team_analysis(self, client):
        resp = client.get("/api/v1/teams/1/analysis")
        assert resp.status_code == 200
        analysis = resp.json()
        assert "avg_squad_rating" in analysis
        assert "style_strengths" in analysis


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestPredictionsAPI:
    def test_predict_player_in_team(self, client):
        resp = client.post("/api/v1/predictions/player/1/team/1")
        assert resp.status_code == 200
        pred = resp.json()
        assert "predicted_rating" in pred
        assert 4.0 <= pred["predicted_rating"] <= 10.0

    def test_predict_trajectory(self, client):
        resp = client.get("/api/v1/predictions/player/1/trajectory?age_from=20&age_to=30")
        assert resp.status_code == 200
        traj = resp.json()
        assert isinstance(traj, list)
        assert len(traj) == 11

    def test_predict_peak(self, client):
        resp = client.get("/api/v1/predictions/player/1/peak")
        assert resp.status_code == 200
        peak = resp.json()
        assert "peak_rating" in peak
        assert "peak_age" in peak

    def test_predict_player_not_found(self, client):
        resp = client.post("/api/v1/predictions/player/99999/team/1")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class TestScenariosAPI:
    def test_what_if(self, client):
        resp = client.post("/api/v1/scenarios/what-if", json={
            "player_id": 1, "team_id": 2, "season": 2024,
        })
        assert resp.status_code == 200
        assert "predicted_rating" in resp.json()

    def test_best_teams(self, client):
        resp = client.get("/api/v1/scenarios/best-teams/1?top_n=5")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert all("predicted_rating" in r for r in results)

    def test_best_players(self, client):
        resp = client.get("/api/v1/scenarios/best-players/1?position=ST&top_n=3")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)

    def test_compare_scenarios(self, client):
        resp = client.post("/api/v1/scenarios/compare", json={
            "player_id": 1, "team_ids": [1, 2, 3], "season": 2024,
        })
        assert resp.status_code == 200
        result = resp.json()
        assert "scenarios" in result
        assert "best_scenario" in result

    def test_transfer_impact(self, client):
        resp = client.post("/api/v1/scenarios/transfer-impact", json={
            "player_id": 1, "from_team_id": 1, "to_team_id": 2, "season": 2024,
        })
        assert resp.status_code == 200
        result = resp.json()
        assert "rating_delta" in result
        assert "recommendation" in result

    def test_what_if_teammates(self, client):
        resp = client.post("/api/v1/scenarios/what-if-teammates", json={
            "player_id": 1, "team_id": 1,
            "hypothetical_avg_rating": 8.0, "season": 2024,
        })
        assert resp.status_code == 200
        assert "predicted_rating" in resp.json()

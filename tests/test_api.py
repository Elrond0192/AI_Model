"""Tests for the Basketball Performance AI FastAPI endpoints."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]


@pytest.fixture(scope="module")
def client():
    """TestClient backed by minimal basketball test data."""
    leagues = pd.DataFrame([{
        "id": 1, "name": "NBA", "country": "USA", "tier": 1,
        "competitiveness_score": 1.0, "avg_pace": 100.0, "avg_offensive_rating": 113.0,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i+1}", "league_id": 1,
        "playing_style": ["pace_and_space","pick_and_roll","isolation","defensive","motion_offense","post_up"][i % 6],
        "formation": "small_ball",
        "pace": 97.0 + i,
        "offensive_rating": 110.0, "defensive_rating": 108.0,
        "three_point_attempt_rate": 0.38, "assists_per_game": 25.0,
        "star_player_usage": 0.28, "league_tier": 1,
    } for i in range(10)])

    np.random.seed(42)
    players = pd.DataFrame([{
        "id": i + 1, "name": f"Player {i+1}",
        "age": 22 + (i % 14),
        "position": POSITIONS[i % len(POSITIONS)],
        "nationality": "American",
        "height_cm": 193.0, "weight_kg": 92.0,
        "dominant_hand": "right",
        "current_team_id": (i % 10) + 1, "current_league_id": 1,
        "draft_year": None, "draft_pick": None,
    } for i in range(50)])

    stat_rows = []
    for _, p in players.iterrows():
        for season in ["2021-22", "2022-23", "2023-24"]:
            mpg = np.random.uniform(15, 35)
            stat_rows.append({
                "player_id": int(p["id"]), "season": season,
                "team_id": int(p["current_team_id"]), "league_id": 1,
                "games_played": int(np.random.randint(30, 80)),
                "minutes_per_game": round(mpg, 1),
                "points": round(np.random.uniform(6, 25), 1),
                "rebounds": round(np.random.uniform(2, 12), 1),
                "offensive_rebounds": round(np.random.uniform(0.5, 3), 1),
                "defensive_rebounds": round(np.random.uniform(1.5, 9), 1),
                "assists": round(np.random.uniform(1, 8), 1),
                "steals": round(np.random.uniform(0.3, 2.0), 2),
                "blocks": round(np.random.uniform(0.1, 2.5), 2),
                "turnovers": round(np.random.uniform(0.5, 4.0), 1),
                "personal_fouls": round(np.random.uniform(1, 4), 1),
                "fg_pct": round(np.random.uniform(0.38, 0.58), 3),
                "three_point_pct": round(np.random.uniform(0.28, 0.45), 3),
                "ft_pct": round(np.random.uniform(0.65, 0.90), 3),
                "plus_minus": round(np.random.uniform(-8, 8), 1),
                "per": round(np.random.uniform(10, 25), 2),
                "ts_pct": round(np.random.uniform(0.50, 0.65), 3),
                "usg_pct": round(np.random.uniform(14, 30), 2),
                "bpm": round(np.random.uniform(-3, 6), 2),
                "vorp": round(np.random.uniform(-0.5, 4), 2),
                "win_shares": round(np.random.uniform(0, 12), 2),
                "ast_ratio": round(np.random.uniform(5, 30), 2),
                "reb_pct": round(np.random.uniform(3, 20), 2),
                "rating": round(np.random.uniform(5.0, 8.5), 3),
            })
    stats = pd.DataFrame(stat_rows)

    rels = pd.DataFrame([{
        "team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
        "season": "2023-24",
        "role": "starter" if i % 3 == 0 else "rotation",
        "jersey_number": (i % 99) + 1,
    } for i, (_, p) in enumerate(players.iterrows())])

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict   = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}

    data = {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": {1: list(range(1, 11))},
    }

    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine
    ensemble = EnsembleModel()
    ensemble.train(data)
    engine = WhatIfEngine(ensemble, data)

    from basketball_ai.api.main import app
    with TestClient(app, raise_server_exceptions=True) as tc:
        # Overwrite app_state AFTER lifespan startup so our basketball test data wins
        from basketball_ai.api import main as api_main
        api_main.app_state["data"]   = data
        api_main.app_state["engine"] = engine
        yield tc


# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_live(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_ready_with_data_and_model(self, client):
        resp = client.get("/health/ready")
        # The test client loads data and model, so should be ready
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestPlayersAPI:
    def test_list_players(self, client):
        resp = client.get("/api/v1/players?limit=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) <= 10

    def test_list_players_filter_position(self, client):
        resp = client.get("/api/v1/players?position=PG&limit=20")
        assert resp.status_code == 200
        for p in resp.json():
            assert p["position"] == "PG"

    def test_get_player(self, client):
        resp = client.get("/api/v1/players/1")
        assert resp.status_code == 200
        assert resp.json()["id"] == 1

    def test_get_player_not_found(self, client):
        assert client.get("/api/v1/players/99999").status_code == 404

    def test_get_player_stats(self, client):
        resp = client.get("/api/v1/players/1/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert isinstance(stats, list)
        assert all("rating" in s for s in stats)
        assert all("points" in s for s in stats)

    def test_get_player_profile(self, client):
        resp = client.get("/api/v1/players/1/profile")
        assert resp.status_code == 200
        profile = resp.json()
        assert "form_score" in profile
        assert "peak_rating" in profile
        assert "position_group" in profile
        assert profile["position_group"] in ("guard", "wing", "big")


class TestTeamsAPI:
    def test_list_teams(self, client):
        resp = client.get("/api/v1/teams?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    def test_get_team(self, client):
        resp = client.get("/api/v1/teams/1")
        assert resp.status_code == 200
        t = resp.json()
        assert t["id"] == 1
        assert "pace" in t
        assert "playing_style" in t

    def test_get_team_not_found(self, client):
        assert client.get("/api/v1/teams/99999").status_code == 404

    def test_team_analysis(self, client):
        resp = client.get("/api/v1/teams/1/analysis")
        assert resp.status_code == 200
        a = resp.json()
        assert "avg_squad_rating" in a
        assert "style_strengths" in a
        assert isinstance(a["style_strengths"], list)


class TestPredictionsAPI:
    def test_predict_player_in_team(self, client):
        resp = client.post("/api/v1/predictions/player/1/team/1")
        assert resp.status_code == 200
        pred = resp.json()
        assert "predicted_rating" in pred
        assert 3.5 <= pred["predicted_rating"] <= 10.0

    def test_predict_trajectory(self, client):
        resp = client.get("/api/v1/predictions/player/1/trajectory?age_from=22&age_to=30")
        assert resp.status_code == 200
        traj = resp.json()
        assert isinstance(traj, list)
        assert len(traj) == 9

    def test_predict_peak(self, client):
        resp = client.get("/api/v1/predictions/player/1/peak")
        assert resp.status_code == 200
        peak = resp.json()
        assert "peak_rating" in peak
        assert "peak_age" in peak

    def test_predict_not_found(self, client):
        assert client.post("/api/v1/predictions/player/99999/team/1").status_code == 404


class TestScenariosAPI:
    def test_what_if(self, client):
        resp = client.post("/api/v1/scenarios/what-if", json={"player_id": 1, "team_id": 2, "season": 2024})
        assert resp.status_code == 200
        assert "predicted_rating" in resp.json()

    def test_best_teams(self, client):
        resp = client.get("/api/v1/scenarios/best-teams/1?top_n=5")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert all("predicted_rating" in r for r in results)

    def test_best_players(self, client):
        resp = client.get("/api/v1/scenarios/best-players/1?position=PG&top_n=3")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_compare_scenarios(self, client):
        resp = client.post("/api/v1/scenarios/compare", json={"player_id": 1, "team_ids": [1, 2, 3], "season": 2024})
        assert resp.status_code == 200
        result = resp.json()
        assert "scenarios" in result
        assert "best_scenario" in result

    def test_transfer_impact(self, client):
        resp = client.post("/api/v1/scenarios/transfer-impact", json={"player_id": 1, "from_team_id": 1, "to_team_id": 2, "season": 2024})
        assert resp.status_code == 200
        result = resp.json()
        assert "rating_delta" in result
        assert "recommendation" in result

    def test_what_if_teammates(self, client):
        resp = client.post("/api/v1/scenarios/what-if-teammates", json={"player_id": 1, "team_id": 1, "hypothetical_avg_rating": 8.0, "season": 2024})
        assert resp.status_code == 200
        assert "predicted_rating" in resp.json()


class TestExplainEndpoint:
    def test_explain_returns_top_factors(self, client):
        resp = client.get("/predictions/1/explain?team_id=1")
        # May be 200 or 503 depending on shap availability
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "top_factors" in data


class TestBatchPrediction:
    def test_batch_predict(self, client):
        payload = {"pairs": [{"player_id": 1, "team_id": 1}, {"player_id": 2, "team_id": 2}]}
        resp = client.post("/predictions/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["count"] == 2

    def test_batch_predict_pagination_headers(self, client):
        resp = client.get("/api/v1/players?limit=10&offset=0")
        assert resp.status_code == 200
        assert "x-total-count" in resp.headers

    def test_teams_pagination_headers(self, client):
        resp = client.get("/api/v1/teams?limit=5&offset=0")
        assert resp.status_code == 200
        assert "x-total-count" in resp.headers

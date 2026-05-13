"""Tests for the WordPress-optimised API endpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]


@pytest.fixture(scope="module")
def wp_client():
    leagues = pd.DataFrame([{
        "id": 1, "name": "NBA", "country": "USA", "tier": 1,
        "competitiveness_score": 1.0, "avg_pace": 100.0, "avg_offensive_rating": 113.0,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i + 1}", "league_id": 1,
        "playing_style": ["pace_and_space", "pick_and_roll", "isolation",
                          "defensive", "motion_offense", "post_up"][i % 6],
        "formation": "small_ball",
        "pace": 97.0 + i, "offensive_rating": 110.0, "defensive_rating": 108.0,
        "three_point_attempt_rate": 0.38, "assists_per_game": 25.0,
        "star_player_usage": 0.28, "league_tier": 1,
    } for i in range(6)])

    np.random.seed(1)
    players = pd.DataFrame([{
        "id": i + 1, "name": f"Player {i + 1}",
        "age": 22 + (i % 12),
        "position": POSITIONS[i % len(POSITIONS)],
        "nationality": "American",
        "height_cm": 193.0, "weight_kg": 92.0,
        "dominant_hand": "right",
        "current_team_id": (i % 6) + 1, "current_league_id": 1,
        "draft_year": None, "draft_pick": None,
    } for i in range(20)])

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
    rels  = pd.DataFrame([{
        "team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
        "season": "2023-24", "role": "starter", "jersey_number": (i % 99) + 1,
    } for i, (_, p) in enumerate(players.iterrows())])

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict   = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}

    data = {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": {1: list(range(1, 7))},
    }

    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine
    ensemble = EnsembleModel()
    ensemble.train(data)
    engine = WhatIfEngine(ensemble, data)

    from basketball_ai.api.main import app
    with TestClient(app, raise_server_exceptions=True) as tc:
        from basketball_ai.api import main as api_main
        api_main.app_state["data"]        = data
        api_main.app_state["engine"]      = engine
        api_main.app_state["chat_engine"] = None
        yield tc


class TestWordPressPlayerCard:
    def test_player_card_structure(self, wp_client):
        resp = wp_client.get("/api/v1/wordpress/player-card/1")
        assert resp.status_code == 200
        card = resp.json()
        assert card["player_id"] == 1
        assert "name" in card
        assert "position" in card
        assert "age" in card
        assert "current_team" in card
        assert "generated_at" in card

    def test_player_card_has_rating(self, wp_client):
        resp = wp_client.get("/api/v1/wordpress/player-card/1")
        assert resp.status_code == 200
        card = resp.json()
        assert card["current_rating"] is not None
        assert 3.5 <= card["current_rating"] <= 10.0

    def test_player_card_has_peak(self, wp_client):
        resp = wp_client.get("/api/v1/wordpress/player-card/1")
        card = resp.json()
        assert card["peak_rating"] is not None
        assert card["peak_age"] is not None

    def test_player_card_has_top_teams(self, wp_client):
        resp = wp_client.get("/api/v1/wordpress/player-card/1")
        card = resp.json()
        assert isinstance(card["top_teams"], list)
        assert len(card["top_teams"]) > 0
        first = card["top_teams"][0]
        assert "rank" in first
        assert "team_name" in first
        assert "predicted_rating" in first

    def test_player_card_confidence_interval(self, wp_client):
        resp = wp_client.get("/api/v1/wordpress/player-card/1")
        card = resp.json()
        assert card["confidence_low"] is not None
        assert card["confidence_high"] is not None
        assert card["confidence_low"] <= card["current_rating"] <= card["confidence_high"]

    def test_player_card_not_found(self, wp_client):
        assert wp_client.get("/api/v1/wordpress/player-card/99999").status_code == 404

    def test_multiple_players(self, wp_client):
        for pid in range(1, 6):
            resp = wp_client.get(f"/api/v1/wordpress/player-card/{pid}")
            assert resp.status_code == 200
            assert resp.json()["player_id"] == pid

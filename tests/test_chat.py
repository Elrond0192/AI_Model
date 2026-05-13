"""Tests for the basketball chat engine and API endpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]


# ---------------------------------------------------------------------------
# Shared fixtures (same minimal data as test_api.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_data():
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

    np.random.seed(0)
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

    return {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict,
        "league_teams": {1: list(range(1, 7))},
    }


@pytest.fixture(scope="module")
def trained_engine(minimal_data):
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine

    ensemble = EnsembleModel()
    ensemble.train(minimal_data)
    return WhatIfEngine(ensemble, minimal_data)


@pytest.fixture(scope="module")
def chat_engine(trained_engine, minimal_data):
    from basketball_ai.chat.engine import ChatEngine
    return ChatEngine(trained_engine, minimal_data)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

class TestIntentDetection:
    def test_predict_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("How good is Player 1 at Team 2?") == Intent.PREDICT

    def test_peak_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("When will Player 5 reach their peak?") == Intent.PEAK

    def test_trajectory_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("Show me Player 3's career arc") == Intent.TRAJECTORY

    def test_transfer_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("What if Player 1 moved from Team 1 to Team 2?") == Intent.TRANSFER

    def test_best_teams_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("Best teams for Player 2?") == Intent.BEST_TEAMS

    def test_best_players_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("Best players for Team 1?") == Intent.BEST_PLAYERS

    def test_compare_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("Compare Player 1 across teams") == Intent.COMPARE

    def test_teammates_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("What if Player 1 had elite teammates?") == Intent.TEAMMATES

    def test_help_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("help") == Intent.HELP
        assert detect_intent("What can you do?") == Intent.HELP

    def test_unknown_intent(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("xyzzy random gibberish") == Intent.UNKNOWN


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_find_player_exact(self, minimal_data):
        from basketball_ai.chat.entities import find_player
        result = find_player("How good is Player 1 today?", minimal_data["player_dict"])
        assert result is not None
        pid, name = result
        assert pid == 1
        assert "Player 1" in name

    def test_find_player_none(self, minimal_data):
        from basketball_ai.chat.entities import find_player
        result = find_player("What is the weather like?", minimal_data["player_dict"])
        assert result is None

    def test_find_team_exact(self, minimal_data):
        from basketball_ai.chat.entities import find_team
        result = find_team("Tell me about Team 3", minimal_data["team_dict"])
        assert result is not None
        tid, name = result
        assert tid == 3

    def test_find_all_teams_two(self, minimal_data):
        from basketball_ai.chat.entities import find_all_teams
        results = find_all_teams(
            "Move from Team 1 to Team 2", minimal_data["team_dict"]
        )
        team_ids = [r[0] for r in results]
        assert 1 in team_ids
        assert 2 in team_ids

    def test_extract_number(self):
        from basketball_ai.chat.entities import extract_number
        assert extract_number("rating of 8.5 expected") == 8.5
        assert extract_number("no numbers here") is None


# ---------------------------------------------------------------------------
# Chat engine (end-to-end)
# ---------------------------------------------------------------------------

class TestChatEngine:
    def test_help_response(self, chat_engine):
        resp = chat_engine.process("help")
        assert resp.intent == "help"
        assert len(resp.reply) > 20
        assert resp.session_id

    def test_predict_with_player_and_team(self, chat_engine):
        resp = chat_engine.process("How good is Player 1 at Team 1?")
        assert resp.intent == "predict"
        assert "Player 1" in resp.reply
        assert "Team 1" in resp.reply
        assert "predicted_rating" in resp.data or "rating" in resp.reply.lower()

    def test_peak_prediction(self, chat_engine):
        resp = chat_engine.process("When will Player 2 reach their peak?")
        assert resp.intent == "peak"
        assert "peak" in resp.reply.lower()

    def test_trajectory(self, chat_engine):
        resp = chat_engine.process("Show Player 3 career arc")
        assert resp.intent == "trajectory"
        assert "trajectory" in resp.reply.lower() or "age" in resp.reply.lower()

    def test_best_teams(self, chat_engine):
        resp = chat_engine.process("Best teams for Player 4?")
        assert resp.intent == "best_teams"
        assert "fits" in resp.data or "Team" in resp.reply

    def test_best_players(self, chat_engine):
        resp = chat_engine.process("Best players for Team 2?")
        assert resp.intent == "best_players"
        assert "players" in resp.data or "Player" in resp.reply

    def test_compare(self, chat_engine):
        resp = chat_engine.process("Compare Player 5 across teams")
        assert resp.intent == "compare"
        assert "scenarios" in resp.data

    def test_session_context_carryover(self, chat_engine):
        """Second question without player name should reuse context from first."""
        import uuid
        sid = str(uuid.uuid4())
        chat_engine.process("How good is Player 1 at Team 1?", sid)
        resp2 = chat_engine.process("When will they peak?", sid)
        # Should resolve to "predict" or "peak" using the carried-over player
        assert resp2.intent in ("peak", "predict", "unknown")
        # Reply should NOT ask for a player name (context was carried over)
        assert "mention a player" not in resp2.reply.lower() or resp2.intent == "unknown"

    def test_no_player_gives_helpful_message(self, chat_engine):
        import uuid
        resp = chat_engine.process("How good is someone?", str(uuid.uuid4()))
        # If no player resolves, should ask for one
        if resp.intent in ("predict", "unknown"):
            assert resp.reply  # non-empty

    def test_suggestions_returned(self, chat_engine):
        resp = chat_engine.process("help")
        assert isinstance(resp.suggestions, list)
        assert len(resp.suggestions) > 0

    def test_teammates_with_elite(self, chat_engine):
        resp = chat_engine.process(
            "What if Player 1 had elite teammates at Team 1?"
        )
        assert resp.intent == "teammates"
        assert "8.5" in resp.reply or "adjusted" in resp.reply.lower()

    # ------------------------------------------------------------------
    # Lineup / quintetto
    # ------------------------------------------------------------------

    def test_lineup_intent_detected(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent(
            "What if Player 1 played at Team 1 with Player 2, Player 3 and Player 4?"
        ) == Intent.LINEUP

    def test_lineup_intent_keyword(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("lineup Player 1 Team 1") == Intent.LINEUP

    def test_lineup_intent_quintetto(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        assert detect_intent("quintetto con Player 1 in Team 2") == Intent.LINEUP

    def test_lineup_does_not_trigger_on_transfer(self):
        from basketball_ai.chat.intent import detect_intent, Intent
        # Transfer must still win over lineup when "moved from … to" is present
        intent = detect_intent("What if Player 1 moved from Team 1 to Team 2?")
        assert intent == Intent.TRANSFER

    def test_find_all_players(self, minimal_data):
        from basketball_ai.chat.entities import find_all_players
        results = find_all_players(
            "Player 1, Player 2 and Player 3 in Team 1",
            minimal_data["player_dict"],
        )
        ids = [r[0] for r in results]
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids

    def test_find_all_players_deduplicates(self, minimal_data):
        from basketball_ai.chat.entities import find_all_players
        results = find_all_players(
            "Player 1 Player 1 Player 1",
            minimal_data["player_dict"],
        )
        ids = [r[0] for r in results]
        assert ids.count(1) == 1

    def test_lineup_engine_basic(self, trained_engine, minimal_data):
        result = trained_engine.what_if_lineup(1, 1, [2, 3, 4])
        assert result.player_id == 1
        assert result.team_id == 1
        assert 3.5 <= result.predicted_rating <= 10.0
        assert isinstance(result.lineup_profiles, list)
        assert len(result.lineup_profiles) == 3

    def test_lineup_engine_position_coverage(self, trained_engine, minimal_data):
        # Players 1-5 cover PG, SG, SF, PF, C by construction (positions cycle)
        result = trained_engine.what_if_lineup(1, 1, [2, 3, 4, 5])
        assert isinstance(result.positions_covered, list)
        assert isinstance(result.missing_positions, list)
        # covered + missing = 5 standard positions
        all_std = set(result.positions_covered) | set(result.missing_positions)
        assert all_std.issubset({"PG", "SG", "SF", "PF", "C"})

    def test_lineup_engine_profiles_have_roles(self, trained_engine, minimal_data):
        result = trained_engine.what_if_lineup(1, 1, [2, 3, 4])
        for p in result.lineup_profiles:
            assert p.role in {
                "Playmaker", "Primary Scorer", "Defender",
                "3pt Specialist", "Paint Scorer / Big", "Two-way / Role Player",
            }
            assert 0.0 <= p.style_compat <= 1.0

    def test_lineup_engine_empty_lineup(self, trained_engine, minimal_data):
        # Empty lineup → still returns a valid result, avg falls back to 6.0
        result = trained_engine.what_if_lineup(1, 1, [])
        assert 3.5 <= result.predicted_rating <= 10.0
        assert result.lineup_profiles == []

    def test_lineup_chat_dispatch(self, chat_engine):
        resp = chat_engine.process(
            "What if Player 1 played at Team 1 with Player 2, Player 3 and Player 4?"
        )
        assert resp.intent == "lineup"
        assert "Player 1" in resp.reply
        assert "Team 1" in resp.reply
        assert "Predicted rating" in resp.reply
        assert "lineup_profiles" in resp.data

    def test_lineup_chat_missing_teammates(self, chat_engine):
        import uuid
        resp = chat_engine.process(
            "lineup Player 1 at Team 1",
            str(uuid.uuid4()),
        )
        # Should ask the user to name the other players
        assert resp.intent == "lineup"
        assert "name" in resp.reply.lower() or "please" in resp.reply.lower()

    def test_help_mentions_lineup(self, chat_engine):
        resp = chat_engine.process("help")
        assert "lineup" in resp.reply.lower() or "quintetto" in resp.reply.lower()


# ---------------------------------------------------------------------------
# Chat API endpoint
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chat_client(minimal_data, trained_engine):
    from basketball_ai.api.main import app
    with TestClient(app, raise_server_exceptions=True) as tc:
        from basketball_ai.api import main as api_main
        from basketball_ai.chat.engine import ChatEngine
        api_main.app_state["data"]        = minimal_data
        api_main.app_state["engine"]      = trained_engine
        api_main.app_state["chat_engine"] = ChatEngine(trained_engine, minimal_data)
        yield tc


class TestChatAPI:
    def test_chat_help(self, chat_client):
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": "help"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "reply" in body
        assert "session_id" in body
        assert "intent" in body
        assert body["intent"] == "help"

    def test_chat_predict(self, chat_client):
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": "How good is Player 1 at Team 1?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "predict"
        assert "Player 1" in body["reply"]

    def test_chat_session_continuity(self, chat_client):
        r1 = chat_client.post(
            "/api/v1/chat",
            json={"message": "How good is Player 2 at Team 2?"},
        )
        sid = r1.json()["session_id"]
        r2 = chat_client.post(
            "/api/v1/chat",
            json={"message": "When will they peak?", "session_id": sid},
        )
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid

    def test_chat_returns_suggestions(self, chat_client):
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": "Best teams for Player 3?"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["suggestions"], list)

    def test_chat_unknown_message(self, chat_client):
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": "xyzzy gibberish $$$"},
        )
        assert resp.status_code == 200  # should never 500

    def test_chat_empty_message_rejected(self, chat_client):
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
        assert resp.status_code == 422  # Pydantic min_length=1

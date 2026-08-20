"""Tests for basketball ML models."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from basketball_ai.models.age_curve import age_performance_factor, PEAK_AGES, peak_age_window, age_trajectory
from basketball_ai.models.performance_model import PerformanceModel
from basketball_ai.models.compatibility_model import CompatibilityModel
from basketball_ai.models.ensemble import EnsembleModel, PredictionResult


# ---------------------------------------------------------------------------
# Shared tiny dataset
# ---------------------------------------------------------------------------

POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]

@pytest.fixture(scope="module")
def tiny_data():
    leagues = pd.DataFrame([{
        "id": 1, "name": "NBA", "country": "USA", "tier": 1,
        "competitiveness_score": 1.0, "avg_pace": 100.0, "avg_offensive_rating": 113.0,
    }, {
        "id": 2, "name": "EuroLeague", "country": "Europe", "tier": 2,
        "competitiveness_score": 0.85, "avg_pace": 93.0, "avg_offensive_rating": 108.0,
    }])
    teams = pd.DataFrame([{
        "id": i + 1, "name": f"Team {i+1}", "league_id": (i % 2) + 1,
        "playing_style": ["pace_and_space","pick_and_roll","isolation","defensive","motion_offense","post_up"][i % 6],
        "formation": "small_ball",
        "pace": 95.0 + i * 2,
        "offensive_rating": 108.0 + i,
        "defensive_rating": 107.0 + i,
        "three_point_attempt_rate": 0.35 + i * 0.01,
        "assists_per_game": 24.0 + i,
        "star_player_usage": 0.28,
        "league_tier": (i % 2) + 1,
    } for i in range(10)])

    np.random.seed(42)
    players = pd.DataFrame([{
        "id": i + 1, "name": f"Player {i+1}",
        "age": 21 + (i % 15),
        "position": POSITIONS[i % len(POSITIONS)],
        "nationality": "American",
        "height_cm": 190 + i % 20,
        "weight_kg": 90 + i % 30,
        "dominant_hand": "right",
        "current_team_id": (i % 10) + 1,
        "current_league_id": (i % 2) + 1,
        "draft_year": None, "draft_pick": None,
    } for i in range(60)])

    stat_rows = []
    _ROLES_COMBO = ["playmaker", "scorer", "forward", "big", "wing"]
    _ROLES_OFF   = ["scorer", "facilitator", "spot_up", "post", "cutter"]
    _ROLES_DEF   = ["lockdown", "stopper", "help_side", "rim_protector", "versatile"]
    for _, p in players.iterrows():
        for season in ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]:
            mpg = np.random.uniform(15, 35)
            stat_rows.append({
                "player_id": int(p["id"]), "season": season,
                "team_id": int(p["current_team_id"]), "league_id": int(p["current_league_id"]),
                "games_played": int(np.random.randint(30, 80)),
                "minutes_per_game": round(mpg, 1),
                "points": round(np.random.uniform(6, 25), 1),
                "rebounds": round(np.random.uniform(2, 12), 1),
                "offensive_rebounds": round(np.random.uniform(0.5, 3), 1),
                "defensive_rebounds": round(np.random.uniform(2, 9), 1),
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
                "obpm": round(np.random.uniform(-2, 4), 2),
                "dbpm": round(np.random.uniform(-2, 3), 2),
                "vorp": round(np.random.uniform(-0.5, 4), 2),
                "win_shares": round(np.random.uniform(0, 12), 2),
                "ast_ratio": round(np.random.uniform(5, 30), 2),
                "reb_pct": round(np.random.uniform(3, 20), 2),
                "tov_pct": round(np.random.uniform(8, 20), 1),
                "ast_pct": round(np.random.uniform(5, 30), 1),
                "orb_pct": round(np.random.uniform(1, 8), 1),
                "drb_pct": round(np.random.uniform(5, 25), 1),
                # DB role columns
                "ruolo_combinato": _ROLES_COMBO[int(p["id"]) % len(_ROLES_COMBO)],
                "ruolo_offensivo": _ROLES_OFF[int(p["id"]) % len(_ROLES_OFF)],
                "ruolo_difensivo": _ROLES_DEF[int(p["id"]) % len(_ROLES_DEF)],
                "rating": round(np.random.uniform(5.0, 8.5), 3),
                # Starter status (SF field)
                "games_started": int(np.random.randint(0, 82)),
                "starter_pct": round(np.random.uniform(0.0, 1.0), 3),
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
    league_teams = {}
    for _, t in teams.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    return {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": stats, "team_player_relations": rels,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": league_teams,
    }


# ---------------------------------------------------------------------------
# Age curve
# ---------------------------------------------------------------------------

class TestAgeCurve:
    def test_factor_at_peak_is_one(self):
        for pos in PEAK_AGES:
            assert age_performance_factor(PEAK_AGES[pos], pos) == pytest.approx(1.0, abs=0.02)

    def test_hybrid_positions(self):
        for pos in ["PG/SG", "SG/SF", "SF/PF", "PF/C", "SG/PF"]:
            assert 0.4 <= age_performance_factor(PEAK_AGES[pos], pos) <= 1.0

    def test_trajectory_peaks_at_correct_age(self):
        traj = age_trajectory("PG", 18, 40)
        best_age = max(traj, key=lambda t: t[1])[0]
        assert abs(best_age - PEAK_AGES["PG"]) <= 2

    def test_peak_window_contains_peak(self):
        for pos in PEAK_AGES:
            start, end = peak_age_window(pos, threshold=0.90)
            assert start <= PEAK_AGES[pos] <= end


# ---------------------------------------------------------------------------
# Performance model
# ---------------------------------------------------------------------------

class TestPerformanceModel:
    def test_train_and_predict(self, tiny_data):
        model = PerformanceModel()
        metrics = model.train(tiny_data)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] < 5.0

    def test_predict_from_features_range(self, tiny_data):
        model = PerformanceModel()
        model.train(tiny_data)
        feats = {c: np.random.uniform(0, 1) for c in model.feature_names}
        feats["age"] = 25.0
        r = model.predict_from_features(feats)
        assert 3.5 <= r <= 10.0

    def test_untrained_fallback(self):
        model = PerformanceModel()
        r = model.predict_from_features({"form_score": 7.0})
        assert 4.0 <= r <= 10.0

    def test_feature_importances(self, tiny_data):
        model = PerformanceModel()
        model.train(tiny_data)
        imp = model.feature_importances()
        assert len(imp) == len(model.feature_names)

    def test_save_load(self, tiny_data, tmp_path):
        model = PerformanceModel()
        model.train(tiny_data)
        path = str(tmp_path / "perf.joblib")
        model.save(path)
        model2 = PerformanceModel()
        model2.load(path)
        feats = {c: 0.5 for c in model.feature_names}
        assert model.predict_from_features(feats) == pytest.approx(
            model2.predict_from_features(feats), abs=0.01
        )


# ---------------------------------------------------------------------------
# Compatibility model
# ---------------------------------------------------------------------------

class TestCompatibilityModel:
    def test_train(self, tiny_data):
        model = CompatibilityModel()
        model.train(tiny_data)
        assert model.is_trained

    def test_score_in_range(self, tiny_data):
        model = CompatibilityModel()
        model.train(tiny_data)
        player_id = list(tiny_data["player_dict"].keys())[0]
        team_id   = list(tiny_data["team_dict"].keys())[0]
        s = model.score(player_id, team_id, tiny_data)
        assert 0.50 <= s <= 1.0

    def test_untrained_fallback(self, tiny_data):
        model = CompatibilityModel()
        player_id = list(tiny_data["player_dict"].keys())[0]
        team_id   = list(tiny_data["team_dict"].keys())[0]
        assert model.score(player_id, team_id, tiny_data) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensemble(tiny_data):
    e = EnsembleModel()
    e.train(tiny_data)
    return e


class TestEnsemble:
    def test_predict_returns_result(self, ensemble, tiny_data):
        r = ensemble.predict(1, 1, tiny_data)
        assert isinstance(r, PredictionResult)

    def test_predicted_rating_in_range(self, ensemble, tiny_data):
        r = ensemble.predict(1, 1, tiny_data)
        assert 3.5 <= r.predicted_rating <= 10.0

    def test_confidence_interval_valid(self, ensemble, tiny_data):
        r = ensemble.predict(1, 1, tiny_data)
        assert r.confidence_low <= r.predicted_rating <= r.confidence_high

    def test_trajectory_older_than_peak_drops(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        traj   = engine.predict_age_trajectory(1, age_range=(22, 38))
        ratings = [p.predicted_rating for p in traj]
        peak_idx = ratings.index(max(ratings))
        # Ratings after the peak should not all be higher than peak
        assert all(r <= max(ratings) + 0.5 for r in ratings)
        assert peak_idx < len(ratings) - 1  # peak is not at the last age

    def test_compare_scenarios_sorted(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        result = engine.compare_scenarios(1, [1, 2, 3])
        ratings = [s["rating"] for s in result.scenarios]
        assert ratings == sorted(ratings, reverse=True)

    def test_simulate_transfer(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        result = engine.simulate_transfer(1, 1, 2)
        assert isinstance(result.rating_delta, float)
        assert result.recommendation != ""

    def test_predict_peak_age_reasonable(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        peak   = engine.predict_peak(1)
        assert 18 <= peak.peak_age <= 40

    def test_what_if_better_teammates(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        better = engine.what_if_teammates(1, 1, 9.0)
        worse  = engine.what_if_teammates(1, 1, 4.0)
        assert better.predicted_rating >= worse.predicted_rating

    def test_best_team_fit(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine = WhatIfEngine(ensemble, tiny_data)
        fits = engine.best_team_fit(1, top_n=5)
        assert len(fits) <= 5
        assert all(f.rank >= 1 for f in fits)

    def test_best_player_for_team_position_filter(self, ensemble, tiny_data):
        from basketball_ai.scenarios.engine import WhatIfEngine
        engine  = WhatIfEngine(ensemble, tiny_data)
        players = engine.best_player_for_team(1, position="PG", top_n=3)
        assert len(players) >= 1
        for p in players:
            assert "PG" in p.position

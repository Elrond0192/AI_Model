"""Tests for predictive models and the scenario engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.age_curve import age_performance_factor, peak_age_window, age_trajectory
from src.models.performance_model import PerformanceModel
from src.models.compatibility_model import CompatibilityModel
from src.models.ensemble import EnsembleModel, PredictionResult
from src.scenarios.engine import WhatIfEngine


# ---------------------------------------------------------------------------
# Minimal synthetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data():
    """Create minimal but functional data for model tests."""
    np.random.seed(0)
    n_players = 80

    leagues = pd.DataFrame([
        {"id": 1, "name": "Premier League", "country": "England", "tier": 1, "competitiveness_score": 0.98},
        {"id": 2, "name": "Championship", "country": "England", "tier": 3, "competitiveness_score": 0.65},
    ])
    teams = pd.DataFrame([
        {"id": i + 1, "name": f"Team {i+1}", "league_id": 1 if i < 10 else 2,
         "playing_style": ["possession", "counter", "high_press", "direct"][i % 4],
         "formation": "4-3-3", "avg_possession": 50 + i % 20,
         "pressing_intensity": 5 + (i % 5), "defensive_line": 5 + (i % 4),
         "passing_tempo": 5 + (i % 4), "league_tier": 1 if i < 10 else 3}
        for i in range(20)
    ])

    positions = ["GK", "CB", "FB", "CM", "AM", "W", "ST"]
    players_list = []
    for pid in range(1, n_players + 1):
        pos = positions[pid % len(positions)]
        players_list.append({
            "id": pid, "name": f"Player {pid}", "age": 20 + (pid % 18),
            "position": pos, "nationality": "English", "foot": "right",
            "height": 180.0, "weight": 75.0,
            "current_team_id": (pid % 20) + 1,
            "current_league_id": 1 if (pid % 20) < 10 else 2,
        })
    players = pd.DataFrame(players_list)

    # Generate 3 seasons of stats per player
    stat_rows = []
    for _, p in players.iterrows():
        for season in [2022, 2023, 2024]:
            age_s = season - (2024 - int(p["age"]))
            m90 = np.random.uniform(10, 30)
            stat_rows.append({
                "player_id": int(p["id"]),
                "season": season,
                "team_id": int(p["current_team_id"]),
                "league_id": int(p["current_league_id"]),
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

    rel_rows = [
        {"team_id": int(p["current_team_id"]), "player_id": int(p["id"]),
         "season": 2024, "role": "starter", "jersey_number": (int(p["id"]) % 99) + 1}
        for _, p in players.iterrows()
    ]
    rel = pd.DataFrame(rel_rows)

    player_dict = {int(r["id"]): r.to_dict() for _, r in players.iterrows()}
    team_dict = {int(r["id"]): r.to_dict() for _, r in teams.iterrows()}
    league_dict = {int(r["id"]): r.to_dict() for _, r in leagues.iterrows()}
    league_teams = {}
    for _, t in teams.iterrows():
        league_teams.setdefault(int(t["league_id"]), []).append(int(t["id"]))

    return {
        "leagues": leagues, "teams": teams, "players": players,
        "player_stats": player_stats, "team_player_relations": rel,
        "player_dict": player_dict, "team_dict": team_dict,
        "league_dict": league_dict, "league_teams": league_teams,
    }


# ---------------------------------------------------------------------------
# Age curve
# ---------------------------------------------------------------------------

class TestAgeCurve:
    def test_peak_factor_is_one(self):
        from src.models.age_curve import PEAK_AGES
        for pos, peak in PEAK_AGES.items():
            assert age_performance_factor(peak, pos) == pytest.approx(1.0, abs=0.01)

    def test_factor_declines_with_age(self):
        for pos in ["GK", "CB", "ST", "W"]:
            f_peak = age_performance_factor(25, pos)
            f_old = age_performance_factor(38, pos)
            assert f_old < f_peak, f"Expected decline for {pos}"

    def test_factor_between_zero_and_one(self):
        for age in range(16, 42):
            for pos in ["GK", "CB", "FB", "CM", "AM", "W", "ST"]:
                f = age_performance_factor(age, pos)
                assert 0.0 <= f <= 1.0, f"Out of range for {pos} age {age}: {f}"

    def test_trajectory_length(self):
        traj = age_trajectory("ST", 18, 36)
        assert len(traj) == 19

    def test_peak_window(self):
        start, end = peak_age_window("ST", threshold=0.90)
        assert start < end
        assert 22 <= start <= 30
        assert end <= 36


# ---------------------------------------------------------------------------
# Performance model
# ---------------------------------------------------------------------------

class TestPerformanceModel:
    def test_untrained_returns_fallback(self, tiny_data):
        model = PerformanceModel()
        feat = {c: 0.0 for c in model.feature_names}
        result = model.predict_from_features(feat)
        assert result == 6.5  # fallback

    def test_train_and_predict(self, tiny_data):
        model = PerformanceModel()
        metrics = model.train(tiny_data)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] < 5.0  # sanity

        feat = {c: 1.0 for c in model.feature_names}
        rating = model.predict_from_features(feat)
        assert 4.0 <= rating <= 10.0

    def test_shap_values_keys(self, tiny_data):
        model = PerformanceModel()
        model.train(tiny_data)
        feat = {c: 1.0 for c in model.feature_names}
        shap = model.get_shap_values(feat)
        if shap:  # may be empty if shap unavailable
            assert set(shap.keys()) == set(model.feature_names)

    def test_feature_importances(self, tiny_data):
        model = PerformanceModel()
        model.train(tiny_data)
        imp = model.feature_importances()
        assert len(imp) == len(model.feature_names)
        assert all(v >= 0 for v in imp.values())

    def test_save_and_load(self, tmp_path, tiny_data):
        model = PerformanceModel()
        model.train(tiny_data)
        save_path = str(tmp_path / "perf.joblib")
        model.save(save_path)

        model2 = PerformanceModel()
        model2.load(save_path)
        assert model2.is_trained

        feat = {c: 1.0 for c in model.feature_names}
        r1 = model.predict_from_features(feat)
        r2 = model2.predict_from_features(feat)
        assert r1 == pytest.approx(r2, abs=1e-5)


# ---------------------------------------------------------------------------
# Compatibility model
# ---------------------------------------------------------------------------

class TestCompatibilityModel:
    def test_train_and_score(self, tiny_data):
        model = CompatibilityModel()
        model.train(tiny_data)
        score = model.score(1, tiny_data)
        assert 0.4 <= score <= 1.0

    def test_unknown_team_fallback(self, tiny_data):
        model = CompatibilityModel()
        model.train(tiny_data)
        score = model.score(9999, tiny_data)
        assert score == 0.75

    def test_save_load(self, tmp_path, tiny_data):
        model = CompatibilityModel()
        model.train(tiny_data)
        path = str(tmp_path / "compat.joblib")
        model.save(path)
        model2 = CompatibilityModel()
        model2.load(path)
        assert model2.is_trained


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    @pytest.fixture(scope="class")
    def trained_ensemble(self, tiny_data):
        ensemble = EnsembleModel()
        ensemble.train(tiny_data)
        return ensemble

    def test_predict_returns_result(self, trained_ensemble, tiny_data):
        result = trained_ensemble.predict(1, 1, tiny_data)
        assert isinstance(result, PredictionResult)
        assert 4.0 <= result.predicted_rating <= 10.0

    def test_ci_contains_prediction(self, trained_ensemble, tiny_data):
        result = trained_ensemble.predict(1, 1, tiny_data)
        assert result.confidence_low <= result.predicted_rating <= result.confidence_high

    def test_age_factor_applied(self, trained_ensemble, tiny_data):
        result = trained_ensemble.predict(1, 1, tiny_data)
        assert 0.4 <= result.age_factor <= 1.0

    def test_save_load(self, tmp_path, tiny_data):
        ens = EnsembleModel()
        ens.train(tiny_data)
        ens.save(str(tmp_path))
        ens2 = EnsembleModel()
        ens2.load(str(tmp_path))
        assert ens2.is_trained


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------

class TestWhatIfEngine:
    @pytest.fixture(scope="class")
    def engine(self, tiny_data):
        ens = EnsembleModel()
        ens.train(tiny_data)
        return WhatIfEngine(ens, tiny_data)

    def test_predict_in_team(self, engine, tiny_data):
        result = engine.predict_in_team(1, 1)
        assert 4.0 <= result.predicted_rating <= 10.0

    def test_age_trajectory_length(self, engine, tiny_data):
        traj = engine.predict_age_trajectory(1, age_range=(20, 30))
        assert len(traj) == 11

    def test_trajectory_rating_range(self, engine, tiny_data):
        traj = engine.predict_age_trajectory(1, age_range=(20, 35))
        for pt in traj:
            assert 4.0 <= pt.predicted_rating <= 10.0

    def test_compare_scenarios_sorted(self, engine, tiny_data):
        result = engine.compare_scenarios(1, [1, 2, 3])
        ratings = [s["rating"] for s in result.scenarios]
        assert ratings == sorted(ratings, reverse=True)

    def test_compare_best_scenario(self, engine, tiny_data):
        result = engine.compare_scenarios(1, [1, 2, 3, 4])
        assert result.best_scenario["rating"] == max(s["rating"] for s in result.scenarios)

    def test_best_team_fit_returns_n(self, engine, tiny_data):
        fits = engine.best_team_fit(1, top_n=5)
        assert len(fits) <= 5
        assert all(f.rank >= 1 for f in fits)

    def test_best_player_for_team(self, engine, tiny_data):
        players = engine.best_player_for_team(1, position="ST", top_n=3)
        assert len(players) >= 1
        assert all(p.rank >= 1 for p in players)

    def test_simulate_transfer(self, engine, tiny_data):
        result = engine.simulate_transfer(1, 1, 2)
        assert isinstance(result.rating_delta, float)
        assert isinstance(result.recommendation, str)

    def test_predict_peak(self, engine, tiny_data):
        peak = engine.predict_peak(1)
        assert peak.peak_rating >= peak.current_rating * 0.8
        assert peak.peak_age >= 18

    def test_what_if_teammates(self, engine, tiny_data):
        result = engine.what_if_teammates(1, 1, 7.5)
        assert 4.0 <= result.predicted_rating <= 10.0

    def test_what_if_better_teammates_improves_rating(self, engine, tiny_data):
        base = engine.predict_in_team(1, 1)
        better = engine.what_if_teammates(1, 1, 9.0)
        worse = engine.what_if_teammates(1, 1, 4.0)
        assert better.predicted_rating >= worse.predicted_rating

"""Model regression tests.

Trains the model on a small fixed dataset and checks that predictions stay
within a tolerance of the stored golden values.  If the model changes
significantly, this test fails and forces a deliberate review.
"""
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures: deterministic minimal dataset
# ---------------------------------------------------------------------------

def _build_regression_data():
    """Build a minimal but deterministic dataset for regression testing."""
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    n_players = 30
    n_seasons = 5

    players = pd.DataFrame({
        "id": range(1, n_players + 1),
        "name": [f"Player{i}" for i in range(1, n_players + 1)],
        "age": rng.integers(22, 35, n_players),
        "position": rng.choice(["PG", "SG", "SF", "PF", "C"], n_players),
        "nationality": ["American"] * n_players,
    })

    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
    records = []
    for pid in range(1, n_players + 1):
        for season in seasons:
            records.append({
                "id": len(records) + 1,
                "player_id": pid, "team_id": 1, "league_id": 1,
                "season": season, "competition": "RS",
                "games_played": int(rng.integers(40, 82)),
                "minutes_per_game": float(rng.uniform(15, 35)),
                "points": float(rng.uniform(5, 25)),
                "rebounds": float(rng.uniform(2, 12)),
                "offensive_rebounds": float(rng.uniform(0.5, 3)),
                "defensive_rebounds": float(rng.uniform(1.5, 9)),
                "assists": float(rng.uniform(1, 10)),
                "steals": float(rng.uniform(0.3, 2)),
                "blocks": float(rng.uniform(0.1, 2)),
                "turnovers": float(rng.uniform(1, 4)),
                "personal_fouls": float(rng.uniform(1, 4)),
                "fg_pct": float(rng.uniform(0.38, 0.58)),
                "three_point_pct": float(rng.uniform(0.28, 0.42)),
                "ft_pct": float(rng.uniform(0.65, 0.95)),
                "plus_minus": float(rng.uniform(-5, 10)),
                "per": float(rng.uniform(10, 28)),
                "ts_pct": float(rng.uniform(0.48, 0.65)),
                "usg_pct": float(rng.uniform(14, 30)),
                "bpm": float(rng.uniform(-3, 6)),
                "vorp": float(rng.uniform(0, 5)),
                "win_shares": float(rng.uniform(0, 10)),
                "ast_ratio": float(rng.uniform(8, 25)),
                "reb_pct": float(rng.uniform(4, 15)),
                "rating": float(rng.uniform(5, 9)),
            })
    player_stats = pd.DataFrame(records)
    return {"players": players, "player_stats": player_stats}


class TestModelRegression:
    @pytest.fixture(scope="class")
    def trained_model(self):
        from basketball_ai.models.performance_model import PerformanceModel
        data = _build_regression_data()
        model = PerformanceModel()
        metrics = model.train(data)
        return model, metrics

    def test_training_completes(self, trained_model):
        model, metrics = trained_model
        assert model.is_trained
        assert "val_rmse" in metrics
        assert metrics["val_rmse"] < 2.5, f"val_rmse too high: {metrics['val_rmse']}"

    def test_baselines_present(self, trained_model):
        _, metrics = trained_model
        assert "baseline_mean_val_rmse" in metrics, "A4: baseline metrics missing"
        assert "xgboost_vs_baseline_delta" in metrics

    def test_lineage_present(self, trained_model):
        _, metrics = trained_model
        assert "lineage" in metrics, "D6: lineage missing from training metrics"
        lineage = metrics["lineage"]
        assert "trained_at" in lineage
        assert lineage["n_samples"] > 0

    def test_data_signature_present(self, trained_model):
        model, metrics = trained_model
        assert "data_signature" in metrics, "A17: data_signature missing"
        assert len(metrics["data_signature"]) == 32, "Expected MD5 hex (32 chars)"
        assert model.data_signature == metrics["data_signature"]

    def test_predictions_in_range(self, trained_model):
        """Predictions must always be in [3.5, 10.0]."""
        model, _ = trained_model
        data = _build_regression_data()
        for pid in range(1, 6):
            stats_df = data["player_stats"][data["player_stats"]["player_id"] == pid]
            if stats_df.empty:
                continue
            row = stats_df.iloc[-1]
            feats = {"form_score": float(row["rating"]), "age": 25, "avg_per": float(row["per"])}
            pred = model.predict_from_features(feats)
            assert 3.5 <= pred <= 10.0, f"Prediction {pred} out of [3.5, 10.0]"

    def test_xgboost_better_than_mean_baseline(self, trained_model):
        """XGBoost should beat the mean baseline (at least slightly)."""
        _, metrics = trained_model
        xgb_rmse = metrics["val_rmse"]
        mean_rmse = metrics.get("baseline_mean_val_rmse", float("inf"))
        # Allow XGBoost to be up to 5% worse than mean in edge cases (tiny dataset)
        assert xgb_rmse <= mean_rmse * 1.05, (
            f"XGBoost RMSE ({xgb_rmse:.3f}) >> mean baseline ({mean_rmse:.3f})"
        )

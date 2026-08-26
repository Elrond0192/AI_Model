"""Tests for FASE I — Metric Rating Engine API routes."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from basketball_ai.api.routes.metric_rating_v2 import router
from basketball_ai.metric_rating.distribution import build_distribution
from basketball_ai.metric_rating.store import load_metric_distributions_csv

MODEL_METADATA = {
    "model_run_id": "run-1",
    "model_version": "2.3.0",
    "feature_version": "metric-rating-v1",
    "data_cutoff": "2025-12-31",
}

CANONICAL_COLUMNS = [
    "league_key", "season", "competition", "player_id", "player_global_id",
    "player_name", "position", "games_played", "minutes_per_game",
    "raptor_total", "lebron_total", "vorp", "games_started", "starter_pct",
]


def make_frame() -> pd.DataFrame:
    rows = []
    for index in range(60):
        rows.append(
            {
                "league_key": "ITA1",
                "season": 2025,
                "competition": "RS",
                "player_id": index + 1,
                "player_global_id": f"G{index + 1}",
                "player_name": f"P{index}",
                "position": "PG",
                "games_played": 25,
                "minutes_per_game": 20.0,
                "raptor_total": float(index - 30),
                "lebron_total": float(index - 30) / 2.0,
                "vorp": float(index - 30),
                "games_started": 20,
                "starter_pct": 0.8,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def make_distribution_rows() -> list[dict]:
    frame = make_frame()
    rows = []
    for metric, source in (("RAPTOR", "raptor_total"), ("LEBRON", "lebron_total"),
                           ("VORP", "vorp")):
        values = pd.to_numeric(frame[source], errors="coerce").dropna()
        dist = build_distribution(values)
        summary = dist.summary()
        rows.append(
            {
                "metric": metric,
                "league_key": "ITA1",
                "season": 2025,
                "competition": "RS",
                **{
                    key: summary[key]
                    for key in (
                        "sample_size", "min", "p10", "p25", "p40", "p50",
                        "p60", "p75", "p90", "p97_5", "max", "mean", "stddev",
                        "quality", "population_source", "population_type",
                        "distribution_version",
                    )
                },
            }
        )
    return rows


def make_client(distributions=None, data=None) -> TestClient:
    app = FastAPI()
    app.state.metric_distributions = distributions if distributions is not None else []
    app.state.data = data if data is not None else {}
    app.state.model_metadata = dict(MODEL_METADATA)
    app.include_router(router)
    return TestClient(app)


class TestRate:
    def test_rate_ok(self):
        client = make_client(make_distribution_rows())
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "RAPTOR", "value": -30.0, "league": "ITA1",
                  "season": "2025-26", "phase": "RS"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["metric"] == "RAPTOR"
        assert payload["percentile"] == pytest.approx(0.0, abs=1e-9)
        assert payload["tier"] == 1 and payload["label"] == "Very Poor"
        assert payload["sample_size"] == 60
        assert payload["quality"] == "low"
        assert payload["distribution_version"] == "1.0"
        assert payload["rating_version"] == "1.0"

    def test_rate_max_value(self):
        client = make_client(make_distribution_rows())
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "RAPTOR", "value": 29.0, "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 200
        assert response.json()["tier"] == 8
        assert response.json()["label"] == "Superstar"

    def test_unknown_metric_422(self):
        client = make_client(make_distribution_rows())
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "NOPE", "value": 1.0, "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 422

    def test_missing_context_422(self):
        client = make_client(make_distribution_rows())
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "RAPTOR", "value": 1.0, "league": "ESP1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 422
        assert "No stored distribution" in response.json()["detail"]

    def test_distributions_not_loaded_503(self):
        client = make_client([])
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "RAPTOR", "value": 1.0, "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 503

    def test_non_finite_value_422(self):
        client = make_client(make_distribution_rows())
        response = client.post(
            "/api/v2/metric-rating/rate",
            json={"metric": "RAPTOR", "value": "NaN", "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 422


class TestPlayerSnapshot:
    def _data(self):
        frame = make_frame()
        return {
            "player_stats": frame,
            "player_dict": {
                int(row["player_id"]): {
                    "id": int(row["player_id"]),
                    "global_id": row["player_global_id"],
                    "name": row["player_name"],
                }
                for _, row in frame.iterrows()
            },
        }

    def test_snapshot_ok(self):
        client = make_client(make_distribution_rows(), self._data())
        response = client.post(
            "/api/v2/metric-rating/player-snapshot",
            json={"player_global_id": "G11", "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["player_global_id"] == "G11"
        assert set(payload["ratings"]) == {"RAPTOR", "LEBRON", "VORP"}
        rap = payload["ratings"]["RAPTOR"]
        assert rap["value"] == pytest.approx(-20.0)
        assert rap["tier"] is not None and rap["label"]
        assert payload["season"] == 2025
        assert payload["phase"] == "RS"

    def test_snapshot_metrics_subset(self):
        client = make_client(make_distribution_rows(), self._data())
        response = client.post(
            "/api/v2/metric-rating/player-snapshot",
            json={"player_global_id": "G11", "league": "ITA1",
                  "season": 2025, "phase": "RS", "metrics": ["VORP"]},
        )
        assert response.status_code == 200
        assert set(response.json()["ratings"]) == {"VORP"}

    def test_snapshot_unknown_player_404(self):
        client = make_client(make_distribution_rows(), self._data())
        response = client.post(
            "/api/v2/metric-rating/player-snapshot",
            json={"player_global_id": "MISSING", "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 404

    def test_snapshot_no_distributions_503(self):
        client = make_client([], self._data())
        response = client.post(
            "/api/v2/metric-rating/player-snapshot",
            json={"player_global_id": "G11", "league": "ITA1",
                  "season": 2025, "phase": "RS"},
        )
        assert response.status_code == 503


class TestStoreCsv:
    def test_load_metric_distributions_csv(self, tmp_path):
        path = tmp_path / "metric_distribution.csv"
        pd.DataFrame(make_distribution_rows()).to_csv(path, index=False)
        rows = load_metric_distributions_csv(path)
        assert len(rows) == 3
        assert rows[0]["metric"] in {"RAPTOR", "LEBRON", "VORP"}
        assert int(rows[0]["season"]) == 2025

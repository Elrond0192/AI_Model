from fastapi import FastAPI
from fastapi.testclient import TestClient

from basketball_ai.api.routes.bb_rating_prediction_v2 import router


class _FakeModel:
    def predict_from_data(self, data, **kwargs):
        return {
            "source_bb_rating": 63,
            "predicted_bb_rating": 68.5,
            "confidence_low_100": 60.0,
            "confidence_high_100": 77.0,
            "scenario": "team_change_same_league",
            "calibration_level": "destination_league+scenario",
            "calibration_samples": 100,
            "interval_samples": 100,
            "source_league": kwargs.get("source_league") or "EL",
            "source_competition": kwargs.get("source_competition"),
            "target_league": kwargs.get("target_league"),
            "target_competition": kwargs.get("target_competition"),
            "target_season": kwargs.get("source_season") + 1,
            "model_version": "1.0.0",
            "target_scale": "BB-Rating 1-100",
        }


def _app(model=None):
    app = FastAPI()
    app.include_router(router)
    app.state.bb_rating_future_prediction_model = model
    app.state.data = {"player_stats": "canonical"}
    return app


def test_direct_bb_rating_endpoint_returns_same_scale_source_and_future():
    client = TestClient(_app(_FakeModel()))
    response = client.post("/api/v2/bb-rating-prediction/player", json={
        "player_global_id": "174019",
        "source_season": 2026,
        "source_league": "EL",
        "source_competition": "RS",
        "target_league": "ITA1",
        "target_competition": "RS",
        "target_team_global_id": "TEAM-1",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["source_bb_rating"] == 63
    assert body["prediction"]["predicted_bb_rating"] == 68.5
    assert body["target_season"] == 2027
    assert body["method"] == "direct_bb_rating_t_plus_1"


def test_direct_bb_rating_endpoint_fails_closed_without_approved_model():
    client = TestClient(_app())
    response = client.post("/api/v2/bb-rating-prediction/player", json={
        "player_global_id": "174019",
        "source_season": 2026,
        "target_league": "ITA1",
    })
    assert response.status_code == 503

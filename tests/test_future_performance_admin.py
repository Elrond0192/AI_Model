import json
from pathlib import Path

from basketball_ai.admin_web import app as admin


def test_future_performance_status_reads_fitted_artifact(monkeypatch, tmp_path):
    root = tmp_path / "models_saved"
    models = root / "future_performance" / "models"
    models.mkdir(parents=True)
    model_file = models / "pts_per_36.joblib"
    model_file.write_bytes(b"placeholder")

    metadata = {
        "status": "fitted",
        "future_performance_version": "1.0",
        "feature_version": "player-future-performance-v1",
        "n_pairs": 120,
        "n_players": 40,
        "leagues": ["ITA1"],
        "competitions": ["RS"],
        "training_target_seasons": [2019, 2020, 2021, 2022],
        "model_files": {"pts_per_36": "models/pts_per_36.joblib"},
        "target_metrics": {
            "pts_per_36": {
                "n_training": 120,
                "oos": {
                    "folds": 2,
                    "n_oos": 50,
                    "rmse_mean": 3.1,
                    "mae_mean": 2.2,
                },
            }
        },
        "uncertainty_by_target": {
            "pts_per_36": {"p50": 2.0, "p75": 3.0, "p90": 4.0, "n_oos": 50}
        },
        "backtest": {
            "status": "validated_oos",
            "method": "expanding_walk_forward_player_season",
            "target_seasons": [2021, 2022],
            "folds": [],
            "summary": {},
        },
    }
    (root / "future_performance" / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    monkeypatch.setattr(admin, "MODEL_ROOT", Path(root))
    result = admin._future_performance_status()

    assert result["ready"] is True
    assert result["future_performance_version"] == "1.0"
    assert result["dataset"]["n_pairs"] == 120
    assert result["targets"]["pts_per_36"]["uncertainty"]["p90"] == 4.0


def test_future_performance_start_requires_active_profile(monkeypatch):
    with admin.STATE.lock:
        admin.STATE.active_profile = None
        admin.STATE.future_performance["status"] = "idle"

    try:
        admin.start_future_performance_training({
            "username": "admin",
            "role": "admin",
            "token": "x",
        })
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    else:
        raise AssertionError("expected active profile validation")

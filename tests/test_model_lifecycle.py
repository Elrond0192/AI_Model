from __future__ import annotations

import json
from pathlib import Path

import pytest

from basketball_ai.admin_web.model_lifecycle import inspect_model, promote_model, rollback_model, validate_artifact


def _prediction_artifact(path: Path, version: str = "1.0", rows: int = 600) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "prediction_rating_calibration.json").write_text(
        json.dumps({
            "calibration_version": version,
            "prediction_model_version": "2.6.0",
            "dataset": {"rows": rows, "target_seasons": [2022, 2023, 2024]},
            "fit": {"x_thresholds": [0.0, 10.0], "y_thresholds_native": [0.0, 10.0]},
            "validation": {"status": "validated_oos", "n_folds": 2},
        }),
        encoding="utf-8",
    )


def test_candidate_must_validate_before_promotion(tmp_path: Path):
    candidate = tmp_path / "candidates" / "prediction-calibration" / "bad"
    _prediction_artifact(candidate, rows=12)

    with pytest.raises(RuntimeError, match="500 OOS rows"):
        promote_model(tmp_path, "prediction-calibration")

    assert not (tmp_path / "prediction_calibration").exists()


def test_promotion_and_rollback_are_atomic_and_keep_previous(tmp_path: Path):
    production = tmp_path / "prediction_calibration"
    _prediction_artifact(production, version="0.9")
    candidate = tmp_path / "candidates" / "prediction-calibration" / "candidate-1"
    _prediction_artifact(candidate, version="1.0")

    result = promote_model(tmp_path, "prediction-calibration")
    assert result["promoted"] is True
    assert result["restart_required"] is True
    assert validate_artifact("prediction-calibration", production)["version"] == "1.0"
    assert result["previous_path"] is not None
    assert Path(result["previous_path"]).is_dir()

    rollback = rollback_model(tmp_path, "prediction-calibration")
    assert rollback["rolled_back"] is True
    assert validate_artifact("prediction-calibration", production)["version"] == "0.9"


def test_inspection_reports_candidate_and_production_validity(tmp_path: Path):
    production = tmp_path / "prediction_calibration"
    _prediction_artifact(production, version="1.0")
    candidate = tmp_path / "candidates" / "prediction-calibration" / "candidate-2"
    _prediction_artifact(candidate, version="1.1")

    result = inspect_model(tmp_path, "prediction-calibration")
    assert result["production"]["valid"] is True
    assert result["candidate"]["valid"] is True
    assert result["candidate"]["version"] == "1.1"




def test_bb_rating_candidate_requires_matching_uncertainty_artifact(tmp_path: Path):
    candidate = tmp_path / "candidates" / "bb-rating" / "candidate-1"
    candidate.mkdir(parents=True)
    (candidate / "bb_rating_calibration.json").write_text(
        json.dumps({
            "calibration_version": "1.17",
            "bb_rating_version": "1.9",
            "dataset": {"rows": 1200},
        }),
        encoding="utf-8",
    )
    (candidate / "bb_rating_uncertainty.json").write_text(
        json.dumps({"status": "fitted", "bb_rating_version": "1.8"}),
        encoding="utf-8",
    )
    result = validate_artifact("bb-rating", candidate)
    assert result["valid"] is False
    assert "versions do not match" in result["reason"]


def test_future_performance_candidate_requires_all_declared_model_files(tmp_path: Path):
    candidate = tmp_path / "candidates" / "future-performance" / "candidate-1"
    candidate.mkdir(parents=True)
    (candidate / "metadata.json").write_text(
        json.dumps({
            "status": "fitted",
            "future_performance_version": "1.0",
            "model_files": {"target": "missing.joblib"},
        }),
        encoding="utf-8",
    )
    result = validate_artifact("future-performance", candidate)
    assert result["valid"] is False
    assert "missing or unsafe" in result["reason"]


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        inspect_model(".", "../prediction-calibration")


def test_bb_rating_candidate_can_be_promoted_with_matching_uncertainty(tmp_path: Path):
    candidate = tmp_path / "candidates" / "bb-rating" / "candidate-bb"
    candidate.mkdir(parents=True)
    (candidate / "bb_rating_calibration.json").write_text(
        json.dumps({
            "calibration_version": "1.18",
            "bb_rating_version": "1.9",
            "dataset": {"rows": 1200, "seasons": [2022, 2023, 2024, 2025], "season_max": 2025},
        }),
        encoding="utf-8",
    )
    (candidate / "bb_rating_uncertainty.json").write_text(
        json.dumps({"status": "fitted", "bb_rating_version": "1.9"}),
        encoding="utf-8",
    )

    result = promote_model(tmp_path, "bb-rating", "candidate-bb")

    production = tmp_path / "bb_rating_calibration"
    assert result["promoted"] is True
    assert validate_artifact("bb-rating", production)["valid"] is True
    inspected = inspect_model(tmp_path, "bb-rating")
    assert inspected["production"]["version"] == "1.18"
    assert inspected["production"]["seasons"] == [2022, 2023, 2024, 2025]
    assert not candidate.exists()


def test_future_performance_candidate_can_be_promoted_with_model_files(tmp_path: Path):
    candidate = tmp_path / "candidates" / "future-performance" / "candidate-fp"
    (candidate / "models").mkdir(parents=True)
    (candidate / "models" / "target.joblib").write_bytes(b"test-artifact")
    (candidate / "metadata.json").write_text(
        json.dumps({
            "status": "fitted",
            "future_performance_version": "1.1",
            "feature_version": "player-future-performance-v1",
            "training_target_seasons": [2021, 2022, 2023, 2024, 2025],
            "n_pairs": 1500,
            "model_files": {"target": "models/target.joblib"},
        }),
        encoding="utf-8",
    )

    result = promote_model(tmp_path, "future-performance", "candidate-fp")

    production = tmp_path / "future_performance"
    assert result["promoted"] is True
    assert validate_artifact("future-performance", production)["valid"] is True
    inspected = inspect_model(tmp_path, "future-performance")
    assert inspected["production"]["version"] == "1.1"
    assert inspected["production"]["seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert not candidate.exists()

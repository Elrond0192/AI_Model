from __future__ import annotations

import json
from pathlib import Path

import pytest

from basketball_ai.admin_web.model_lifecycle import (
    cleanup_incomplete_candidates,
    inspect_model,
    promote_model,
    rollback_model,
    validate_artifact,
)


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
        inspect_model(".", "../unsupported-model")


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



def _bb_rating_prediction_artifact(path: Path, version: str = "1.0.0") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps({
            "model": "BBRatingFuturePredictionModel",
            "model_version": version,
            "feature_version": "bb-rating-direct-target-v2",
            "target": "official_bb_rating_t_plus_1",
            "target_scale": [1, 100],
            "bb_rating_version": "1.9",
            "oof_samples": 600,
            "training_samples": 2000,
            "promotion_gate_passed": True,
            "scenario_metrics": {"same_team_same_league": {"n": 300, "mae": 5.0}},
        }),
        encoding="utf-8",
    )
    (path / "bb_rating_future_model.joblib").write_bytes(b"test-artifact")
    (path / "calibration.json").write_text(json.dumps({"method": "test"}), encoding="utf-8")


def test_bb_rating_prediction_candidate_requires_oof_gate_and_can_promote(tmp_path: Path):
    candidate = tmp_path / "candidates" / "bb-rating-prediction" / "candidate-bbp"
    _bb_rating_prediction_artifact(candidate)
    result = validate_artifact("bb-rating-prediction", candidate)
    assert result["valid"] is True
    assert result["version"] == "1.0.0"

    promoted = promote_model(tmp_path, "bb-rating-prediction", "candidate-bbp")
    production = tmp_path / "bb_rating_future_prediction"
    assert promoted["promoted"] is True
    assert validate_artifact("bb-rating-prediction", production)["valid"] is True
    assert not candidate.exists()


def test_bb_rating_prediction_candidate_fails_without_baseline_gate(tmp_path: Path):
    candidate = tmp_path / "candidates" / "bb-rating-prediction" / "candidate-bad"
    _bb_rating_prediction_artifact(candidate)
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    metadata["promotion_gate_passed"] = False
    (candidate / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    result = validate_artifact("bb-rating-prediction", candidate)
    assert result["valid"] is False
    assert "baseline" in result["reason"]

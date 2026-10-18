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


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        inspect_model(".", "../prediction-calibration")

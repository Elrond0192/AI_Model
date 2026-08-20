"""Tests for production model lifecycle and monitoring."""
from __future__ import annotations

from pathlib import Path


def _valid_backtest(rmse: float = 0.5, league_rmse: float | None = None) -> dict:
    segment_rmse = rmse if league_rmse is None else league_rmse
    return {
        "valid": True,
        "overall": {
            "n": 100,
            "rmse": rmse,
            "mae": rmse * 0.8,
            "bias": 0.0,
            "interval_coverage": 0.90,
        },
        "overall_rmse": rmse,
        "base_rmse": rmse + 0.10,
        "persistence_rmse": rmse + 0.20,
        "ensemble_vs_base_delta": 0.10,
        "ensemble_vs_persistence_delta": 0.20,
        "by_league": {
            "ITA1": {"n": 50, "rmse": segment_rmse},
        },
        "folds": [{"valid": True, "target_season": 2025, "rmse": rmse}],
    }


def _run(tmp_path: Path, run_id: str, rmse: float = 0.5) -> tuple[Path, dict]:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    for name in (
        "performance_model.joblib",
        "compatibility_model.joblib",
        "conformal.joblib",
        "production_state.joblib",
    ):
        (run_dir / name).write_bytes(b"artifact")
    metadata = {
        "model_run_id": run_id,
        "model_version": "2.1.0",
        "feature_version": "forecast-t-plus-1-v2",
        "data_cutoff": "2026-08-20",
        "backtest": _valid_backtest(rmse),
    }
    import json

    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir, metadata


def _register(tmp_path: Path, run_id: str, rmse: float = 0.5) -> None:
    from basketball_ai.models.promote import register_candidate

    run_dir, metadata = _run(tmp_path, run_id, rmse)
    register_candidate(str(tmp_path), run_id, run_dir, metadata)


def test_get_git_sha():
    from basketball_ai.models.performance_model import _get_git_sha

    assert isinstance(_get_git_sha(), str)


def test_promote_no_candidate(tmp_path):
    from basketball_ai.models.promote import promote_if_better

    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is False


def test_promote_activates_immutable_candidate(tmp_path):
    from basketball_ai.models.promote import get_promotion_status, promote_if_better

    _register(tmp_path, "run-v2", 0.50)
    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is True
    assert (tmp_path / "production" / "performance_model.joblib").is_file()
    assert (tmp_path / "production" / "production_state.joblib").is_file()
    assert (tmp_path / "production" / "production_manifest.json").is_file()
    status = get_promotion_status(str(tmp_path))
    assert status["production"]["run_id"] == "run-v2"
    assert status["candidate"] is None


def test_promote_requires_valid_full_ensemble_gates(tmp_path):
    from basketball_ai.models.promote import promote_if_better, register_candidate

    run_dir, metadata = _run(tmp_path, "bad", 0.50)
    metadata["backtest"]["ensemble_vs_base_delta"] = -0.01
    register_candidate(str(tmp_path), "bad", run_dir, metadata)
    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is False
    assert "does not beat the base" in result["reason"]
    assert not (tmp_path / "production").exists()


def test_promote_rejects_missing_interval_coverage(tmp_path):
    from basketball_ai.models.promote import promote_if_better, register_candidate

    run_dir, metadata = _run(tmp_path, "no-coverage", 0.50)
    metadata["backtest"]["overall"].pop("interval_coverage")
    register_candidate(str(tmp_path), "no-coverage", run_dir, metadata)
    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is False
    assert "interval coverage is missing" in result["reason"]


def test_candidate_requires_versioned_runtime_state(tmp_path):
    import pytest

    from basketball_ai.models.promote import register_candidate

    run_dir, metadata = _run(tmp_path, "missing-state", 0.50)
    (run_dir / "production_state.joblib").unlink()
    with pytest.raises(RuntimeError, match="missing required model artifacts"):
        register_candidate(str(tmp_path), "missing-state", run_dir, metadata)


def test_promote_rejects_insufficient_improvement(tmp_path):
    from basketball_ai.models.promote import promote_if_better

    _register(tmp_path, "run-v1", 0.50)
    assert promote_if_better(str(tmp_path))["promoted"] is True

    _register(tmp_path, "run-v2", 0.495)
    result = promote_if_better(str(tmp_path), min_improvement_pct=2.0)
    assert result["promoted"] is False
    assert "improvement" in result["reason"]


def test_promote_rejects_large_segment_regression(tmp_path):
    from basketball_ai.models.promote import promote_if_better, register_candidate

    _register(tmp_path, "run-v1", 0.60)
    assert promote_if_better(str(tmp_path))["promoted"] is True

    run_dir, metadata = _run(tmp_path, "run-v2", 0.50)
    metadata["backtest"]["by_league"]["ITA1"]["rmse"] = 0.80
    register_candidate(str(tmp_path), "run-v2", run_dir, metadata)
    result = promote_if_better(str(tmp_path), min_improvement_pct=0.0)
    assert result["promoted"] is False
    assert "league ITA1" in result["reason"]


def test_rollback_restores_previous_artifacts(tmp_path):
    from basketball_ai.models.promote import (
        get_promotion_status,
        promote_if_better,
        rollback_to_previous,
    )

    _register(tmp_path, "run-v1", 0.60)
    assert promote_if_better(str(tmp_path))["promoted"] is True
    _register(tmp_path, "run-v2", 0.50)
    assert promote_if_better(str(tmp_path), min_improvement_pct=0.0)["promoted"] is True

    result = rollback_to_previous(str(tmp_path))
    assert result["rolled_back"] is True
    status = get_promotion_status(str(tmp_path))
    assert status["production"]["run_id"] == "run-v1"
    assert status["previous"]["run_id"] == "run-v2"


def test_production_model_dir_fails_closed(tmp_path):
    import pytest

    from basketball_ai.models.promote import production_model_dir

    with pytest.raises(RuntimeError, match="No promoted production model"):
        production_model_dir(str(tmp_path))


def test_drift_ks():
    import numpy as np
    import pandas as pd

    from basketball_ai.monitoring.drift import compute_ks_drift

    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {"points": rng.normal(10, 2, 100), "assists": rng.normal(5, 1, 100)}
    )
    cur = pd.DataFrame(
        {"points": rng.normal(10, 2, 100), "assists": rng.normal(5, 1, 100)}
    )
    result = compute_ks_drift(ref, cur, cols=["points", "assists"])
    assert "points" in result
    assert 0.0 <= result["points"] <= 1.0


def test_drift_segment():
    import numpy as np
    import pandas as pd

    from basketball_ai.monitoring.drift import capture_reference, compute_segment_drift

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "points": rng.normal(10, 2, n),
            "assists": rng.normal(5, 1, n),
            "league_id": rng.integers(1, 4, n),
        }
    )
    data = {"player_stats": df}
    ref = capture_reference(data)
    assert isinstance(compute_segment_drift(data, ref), dict)

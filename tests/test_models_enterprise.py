"""Tests for enterprise model improvements."""
from __future__ import annotations

import json
from pathlib import Path


def test_get_git_sha():
    from basketball_ai.models.performance_model import _get_git_sha
    sha = _get_git_sha()
    # Should be a string (may be empty in CI without git)
    assert isinstance(sha, str)


def test_promote_no_candidate(tmp_path):
    from basketball_ai.models.promote import promote_if_better
    (tmp_path / "registry.json").write_text("{}")
    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is False


def test_promote_with_candidate(tmp_path):
    from basketball_ai.models.promote import promote_if_better
    reg = {"candidate": {"val_rmse": 0.5, "version": "v2"}}
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    result = promote_if_better(str(tmp_path))
    assert result["promoted"] is True
    # Now production should be set
    reg2 = json.loads((tmp_path / "registry.json").read_text())
    assert "production" in reg2
    assert "candidate" not in reg2


def test_promote_no_improvement(tmp_path):
    from basketball_ai.models.promote import promote_if_better
    reg = {
        "candidate":  {"val_rmse": 0.8, "version": "v2"},
        "production": {"val_rmse": 0.5, "version": "v1"},
    }
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    result = promote_if_better(str(tmp_path), min_improvement_pct=2.0)
    assert result["promoted"] is False


def test_rollback(tmp_path):
    from basketball_ai.models.promote import promote_if_better
    # Set up: promote a candidate first
    reg = {
        "candidate":  {"val_rmse": 0.4, "version": "v2"},
        "production": {"val_rmse": 0.5, "version": "v1"},
    }
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    promote_if_better(str(tmp_path), min_improvement_pct=0.0)

    from basketball_ai.models.promote import rollback_to_previous
    result = rollback_to_previous(str(tmp_path))
    assert result["rolled_back"] is True
    reg2 = json.loads((tmp_path / "registry.json").read_text())
    # previous v1 should be back as production
    assert reg2["production"]["version"] == "v1"


def test_get_promotion_status(tmp_path):
    from basketball_ai.models.promote import get_promotion_status
    reg = {"production": {"val_rmse": 0.5}, "candidate": {"val_rmse": 0.3}}
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    status = get_promotion_status(str(tmp_path))
    assert "production" in status
    assert "candidate" in status


def test_drift_ks(tmp_path):
    """KS drift test returns pvalues."""
    import pandas as pd
    import numpy as np
    from basketball_ai.monitoring.drift import compute_ks_drift
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"points": rng.normal(10, 2, 100), "assists": rng.normal(5, 1, 100)})
    cur = pd.DataFrame({"points": rng.normal(10, 2, 100), "assists": rng.normal(5, 1, 100)})
    result = compute_ks_drift(ref, cur, cols=["points", "assists"])
    assert "points" in result
    assert 0.0 <= result["points"] <= 1.0


def test_drift_segment(tmp_path):
    """Segment drift returns per-league dict."""
    import pandas as pd
    import numpy as np
    from basketball_ai.monitoring.drift import capture_reference, compute_segment_drift
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "points":    rng.normal(10, 2, n),
        "assists":   rng.normal(5, 1, n),
        "league_id": rng.integers(1, 4, n),
    })
    data = {"player_stats": df}
    ref = capture_reference(data)
    result = compute_segment_drift(data, ref)
    assert isinstance(result, dict)

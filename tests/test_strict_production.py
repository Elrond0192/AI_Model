"""Tests for the exact production model contract."""
from __future__ import annotations

import math

import pandas as pd


def test_position_as_of_uses_roster_position_and_ignores_future_state():
    from basketball_ai.models.strict_production import position_as_of

    relations = pd.DataFrame(
        [
            {"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"},
            {"player_id": 7, "team_id": 10, "season": 2024, "role": "SF"},
        ]
    )
    assert position_as_of(7, relations, 2022, "PG") == "SG"
    assert position_as_of(7, relations, 2023, "PG") == "SG"
    assert position_as_of(7, relations, 2024, "PG") == "SF"


def test_position_as_of_rejects_non_position_legacy_role():
    from basketball_ai.models.strict_production import position_as_of

    relations = pd.DataFrame(
        [{"player_id": 7, "team_id": 10, "season": 2022, "role": "starter"}]
    )
    assert position_as_of(7, relations, 2022, "PF") == "PF"


def test_runtime_snapshot_patches_player_position_without_mutating_input():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    data = {
        "player_dict": {7: {"id": 7, "position": "SF"}},
        "team_player_relations": pd.DataFrame(
            [
                {"player_id": 7, "team_id": 10, "season": 2022, "role": "SG"},
                {"player_id": 7, "team_id": 10, "season": 2024, "role": "SF"},
            ]
        ),
    }
    model = StrictProductionEnsembleModel()
    patched = model._data_with_asof_position(7, data, 2022)
    assert patched["player_dict"][7]["position"] == "SG"
    assert data["player_dict"][7]["position"] == "SF"


def test_finite_sample_conformal_uses_single_direct_quantile(monkeypatch):
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    monkeypatch.setenv("MODEL_TARGET_INTERVAL_COVERAGE", "0.90")
    residuals = [float(value) / 10.0 for value in range(1, 21)]
    model = StrictProductionEnsembleModel()
    model._calibrate_conformal(residuals)

    rank = min(len(residuals), math.ceil((len(residuals) + 1) * 0.90))
    expected = sorted(residuals)[rank - 1]
    assert model._conformal_q_lo == expected
    assert model._conformal_q_hi == expected
    assert model._conformal_nominal_coverage == 0.90


def test_finite_sample_conformal_refuses_too_small_holdout():
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel

    model = StrictProductionEnsembleModel()
    model._calibrate_conformal([0.1] * 9)
    assert model._conformal_q_lo is None
    assert model._conformal_q_hi is None

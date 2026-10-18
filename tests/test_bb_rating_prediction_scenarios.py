import numpy as np
import pandas as pd
import pytest

from basketball_ai.bb_rating.prediction_scenarios import (
    BBRatingScenarioCalibrator,
    SCENARIO_LEAGUE_CHANGE,
    SCENARIO_SAME_TEAM,
    SCENARIO_TEAM_AND_LEAGUE_CHANGE,
    SCENARIO_TEAM_CHANGE,
    SCENARIO_UNKNOWN,
    classify_transition,
)


@pytest.mark.parametrize(
    ("source_team", "target_team", "source_league", "target_league", "expected"),
    [
        (10, 10, 1, 1, SCENARIO_SAME_TEAM),
        (10, 20, 1, 1, SCENARIO_TEAM_CHANGE),
        (10, 10, 1, 2, SCENARIO_LEAGUE_CHANGE),
        (10, 20, 1, 2, SCENARIO_TEAM_AND_LEAGUE_CHANGE),
        (None, 20, 1, 2, SCENARIO_UNKNOWN),
    ],
)
def test_classify_transition(source_team, target_team, source_league, target_league, expected):
    assert classify_transition(source_team, target_team, source_league, target_league) == expected


def test_scenario_calibration_prefers_destination_league_and_roundtrips():
    rows = []
    for i in range(80):
        prediction = 20 + (i % 40) * 1.5
        rows.append({
            "predicted_bb_rating": prediction,
            "observed_bb_rating": min(100, prediction + 8),
            "scenario": SCENARIO_TEAM_CHANGE,
            "target_league_id": 7,
        })
    # Other scenario trains a different mapping to verify scenario separation.
    for i in range(80):
        prediction = 20 + (i % 40) * 1.5
        rows.append({
            "predicted_bb_rating": prediction,
            "observed_bb_rating": max(1, prediction - 7),
            "scenario": SCENARIO_SAME_TEAM,
            "target_league_id": 7,
        })

    calibrator = BBRatingScenarioCalibrator(min_samples=20)
    report = calibrator.fit(pd.DataFrame(rows))
    assert report["n_samples"] == 160
    result = calibrator.predict(50, scenario=SCENARIO_TEAM_CHANGE, target_league_id=7)
    assert result["calibration_level"] == "destination_league+scenario"
    assert result["predicted_bb_rating"] > 50
    restored = BBRatingScenarioCalibrator.from_dict(calibrator.to_dict())
    restored_result = restored.predict(50, scenario=SCENARIO_TEAM_CHANGE, target_league_id=7)
    assert restored_result["predicted_bb_rating"] == pytest.approx(
        result["predicted_bb_rating"], abs=0.2
    )


def test_sparse_scenario_falls_back_to_global_and_clips_output():
    frame = pd.DataFrame({
        "predicted_bb_rating": np.linspace(10, 90, 50),
        "observed_bb_rating": np.linspace(15, 95, 50),
        "scenario": ["common"] * 50,
        "target_league_id": [1] * 50,
    })
    calibrator = BBRatingScenarioCalibrator(min_samples=10)
    calibrator.fit(frame)
    result = calibrator.predict(120, scenario="rare", target_league_id=99)
    assert result["calibration_level"] == "global"
    assert 1 <= result["predicted_bb_rating"] <= 100


def test_calibration_rejects_missing_columns_and_empty_valid_rows():
    calibrator = BBRatingScenarioCalibrator()
    with pytest.raises(ValueError, match="Missing calibration columns"):
        calibrator.fit(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="No valid BB-Rating"):
        calibrator.fit(pd.DataFrame({
            "predicted_bb_rating": [0],
            "observed_bb_rating": [101],
            "scenario": ["common"],
        }))

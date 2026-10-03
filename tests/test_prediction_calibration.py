from pathlib import Path

import numpy as np

from basketball_ai.prediction_calibration import (
    build_prediction_calibration,
    native_to_100,
    write_prediction_calibration,
)


def _records():
    rows = []
    rng = np.random.default_rng(7)
    for season in (2022, 2023, 2024, 2025):
        for _ in range(300):
            prediction = float(rng.uniform(2.0, 9.0))
            actual = float(np.clip(1.0 + 0.88 * prediction + rng.normal(0, 0.35), 0, 10))
            rows.append({
                "target_season": season,
                "prediction": prediction,
                "actual": actual,
            })
    return rows


def test_prediction_calibration_is_monotonic_and_1_100():
    report = build_prediction_calibration(_records())
    mapping = report["fit"]

    values = [native_to_100(v, mapping) for v in np.linspace(0, 10, 101)]
    assert all(1.0 <= value <= 100.0 for value in values)
    assert all(left <= right + 1e-9 for left, right in zip(values, values[1:]))
    assert native_to_100(0.0, mapping) >= 1.0
    assert native_to_100(10.0, mapping) <= 100.0
    assert report["prediction_model_version"] == "2.6.0"


def test_prediction_calibration_writes_json_and_markdown(tmp_path: Path):
    report = build_prediction_calibration(_records())
    paths = write_prediction_calibration(report, str(tmp_path))
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()

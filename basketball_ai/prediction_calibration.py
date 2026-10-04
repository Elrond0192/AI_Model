"""Calibration layer that maps Prediction Model 0-10 output to a user-facing 1-100 scale.

This module never changes the Prediction Model estimator or its native 0-10 output.
It learns a monotonic OOS mapping from the model's predicted next-season rating
to the realized next-season rating and persists the mapping as JSON.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

PREDICTION_CALIBRATION_VERSION = "1.0"
PREDICTION_MODEL_VERSION = "2.6.0"
NATIVE_MIN = 0.0
NATIVE_MAX = 10.0
DISPLAY_MIN = 1.0
DISPLAY_MAX = 100.0
MIN_FINAL_SAMPLES = 500


def _finite_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        try:
            prediction = float(row["prediction"])
            actual = float(row["actual"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(prediction) and math.isfinite(actual)):
            continue
        rows.append({
            "prediction": float(np.clip(prediction, NATIVE_MIN, NATIVE_MAX)),
            "actual": float(np.clip(actual, NATIVE_MIN, NATIVE_MAX)),
            "target_season": int(row["target_season"]) if row.get("target_season") is not None else None,
        })
    return rows


def _rmse(prediction: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def _mae(prediction: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual)))


def _fit_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([r["prediction"] for r in rows], dtype=float)
    y = np.asarray([r["actual"] for r in rows], dtype=float)
    if len(rows) < MIN_FINAL_SAMPLES:
        raise ValueError(f"at least {MIN_FINAL_SAMPLES} OOS rows are required")

    calibrator = IsotonicRegression(
        y_min=NATIVE_MIN,
        y_max=NATIVE_MAX,
        increasing=True,
        out_of_bounds="clip",
    ).fit(x, y)

    thresholds = np.asarray(calibrator.X_thresholds_, dtype=float)
    values = np.asarray(calibrator.y_thresholds_, dtype=float)
    if thresholds.size < 2:
        raise ValueError("calibration mapping has fewer than two thresholds")

    return {
        "method": "isotonic_oos",
        "x_thresholds": thresholds.tolist(),
        "y_thresholds_native": values.tolist(),
        "fit_rows": int(len(rows)),
        "fit_prediction_min": float(x.min()),
        "fit_prediction_max": float(x.max()),
        "fit_actual_mean": float(y.mean()),
        "fit_actual_std": float(y.std()),
    }


def native_to_100(native_value: float, mapping: dict[str, Any]) -> float:
    x = float(np.clip(native_value, NATIVE_MIN, NATIVE_MAX))
    thresholds = np.asarray(mapping["x_thresholds"], dtype=float)
    values = np.asarray(mapping["y_thresholds_native"], dtype=float)
    calibrated = float(np.interp(x, thresholds, values))
    calibrated = float(np.clip(calibrated, NATIVE_MIN, NATIVE_MAX))
    # Preserve both endpoints: 0 -> 1, 10 -> 100.
    return float(np.clip(DISPLAY_MIN + (DISPLAY_MAX - DISPLAY_MIN) * calibrated / NATIVE_MAX,
                          DISPLAY_MIN, DISPLAY_MAX))


def _temporal_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seasons = sorted({r["target_season"] for r in rows if r["target_season"] is not None})
    folds = []
    for target in seasons[1:]:
        train = [r for r in rows if r["target_season"] is not None and r["target_season"] < target]
        test = [r for r in rows if r["target_season"] == target]
        if len(train) < MIN_FINAL_SAMPLES or not test:
            continue
        mapping = _fit_mapping(train)
        raw = np.asarray([r["prediction"] for r in test], dtype=float)
        actual = np.asarray([r["actual"] for r in test], dtype=float)
        calibrated = np.asarray([native_to_100(v, mapping) for v in raw], dtype=float)
        calibrated_native = (calibrated - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN) * NATIVE_MAX
        folds.append({
            "target_season": int(target),
            "calibration_train_seasons": sorted({r["target_season"] for r in train if r["target_season"] is not None}),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "raw_rmse_native": _rmse(raw, actual),
            "calibrated_rmse_native": _rmse(calibrated_native, actual),
            "raw_mae_native": _mae(raw, actual),
            "calibrated_mae_native": _mae(calibrated_native, actual),
            "mean_calibration_delta_native": float(np.mean(calibrated_native - raw)),
        })
    return {
        "method": "expanding_walk_forward_isotonic",
        "status": "validated_oos" if folds else "insufficient_data",
        "folds": folds,
        "n_folds": len(folds),
    }


def build_prediction_calibration(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = _finite_rows(records)
    if len(rows) < MIN_FINAL_SAMPLES:
        raise ValueError(f"only {len(rows)} valid OOS rows; {MIN_FINAL_SAMPLES} required")

    mapping = _fit_mapping(rows)
    validation = _temporal_validation(rows)

    x = np.asarray([r["prediction"] for r in rows], dtype=float)
    y = np.asarray([r["actual"] for r in rows], dtype=float)
    calibrated_native = np.asarray(
        [native_to_100(v, mapping) for v in x], dtype=float
    )
    calibrated_native = (calibrated_native - 1.0) / 9.9 * 10.0

    return {
        "calibration_version": PREDICTION_CALIBRATION_VERSION,
        "prediction_model_version": PREDICTION_MODEL_VERSION,
        "display_contract": {
            "native_scale": "0-10",
            "display_scale": "1-100",
            "mapping": "1 + 9.9 * calibrated_native / 10",
            "purpose": "user-facing presentation only; native model output is unchanged",
        },
        "dataset": {
            "rows": int(len(rows)),
            "target_seasons": sorted({r["target_season"] for r in rows if r["target_season"] is not None}),
        },
        "validation": validation,
        "fit": {
            **mapping,
            "raw_rmse_native": _rmse(x, y),
            "calibrated_rmse_native": _rmse(calibrated_native, y),
            "raw_mae_native": _mae(x, y),
            "calibrated_mae_native": _mae(calibrated_native, y),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_prediction_calibration(report: dict[str, Any], output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "prediction_rating_calibration.json"
    md_path = root / "prediction_rating_calibration.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    validation = report["validation"]
    fit = report["fit"]
    md_path.write_text(
        "\n".join([
            "# Prediction Rating Calibration",
            "",
            f"- Calibration version: **{report['calibration_version']}**",
            f"- Prediction Model: **{report['prediction_model_version']}**",
            f"- OOS rows: **{report['dataset']['rows']}**",
            f"- Target seasons: **{report['dataset']['target_seasons']}**",
            "- Native model output: **0–10**",
            "- User-facing calibrated output: **1–100**",
            "- Method: **expanding walk-forward isotonic calibration**",
            "",
            "The calibration is a serving/presentation layer. It does not retrain, "
            "modify or post-process the native Prediction Model prediction.",
            "",
            "## OOS validation",
            "",
            f"- Status: **{validation['status']}**",
            f"- Folds: **{validation['n_folds']}**",
            "",
            "## Final fit",
            "",
            f"- Raw RMSE (native): **{fit['raw_rmse_native']:.6f}**",
            f"- Calibrated RMSE (native): **{fit['calibrated_rmse_native']:.6f}**",
            f"- Raw MAE (native): **{fit['raw_mae_native']:.6f}**",
            f"- Calibrated MAE (native): **{fit['calibrated_mae_native']:.6f}**",
        ]),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(md_path)}


def load_prediction_calibration(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("calibration_version") != PREDICTION_CALIBRATION_VERSION:
        return None
    if payload.get("prediction_model_version") != PREDICTION_MODEL_VERSION:
        return None
    fit = payload.get("fit") or {}
    if not fit.get("x_thresholds") or not fit.get("y_thresholds_native"):
        return None
    return payload

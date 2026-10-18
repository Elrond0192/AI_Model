"""Atomic candidate/production artifact lifecycle for auxiliary model releases."""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_MODEL_PATHS = {
    "prediction-calibration": ("prediction_calibration", "prediction_rating_calibration.json"),
    "bb-rating": ("bb_rating_calibration", "bb_rating_calibration.json"),
    "future-performance": ("future_performance", "metadata.json"),
    "bb-rating-prediction": ("bb_rating_future_prediction", "metadata.json"),
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _model_name(model: str) -> str:
    if model not in _MODEL_PATHS:
        raise ValueError(f"Unsupported model release: {model}")
    return model


def validate_artifact(model: str, directory: str | Path) -> dict[str, Any]:
    model = _model_name(model)
    root = Path(directory)
    filename = _MODEL_PATHS[model][1]
    artifact_path = root / filename
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "reason": f"Missing or invalid {filename}: {exc}"}
    if not isinstance(artifact, dict):
        return {"valid": False, "reason": f"{filename} must contain a JSON object"}

    if model == "prediction-calibration":
        dataset = artifact.get("dataset") or {}
        fit = artifact.get("fit") or {}
        mapping = fit
        if int(dataset.get("rows") or 0) < 500:
            return {"valid": False, "reason": "Prediction calibration requires at least 500 OOS rows"}
        x_values = mapping.get("x_thresholds") or []
        y_values = mapping.get("y_thresholds_native") or []
        if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
            return {"valid": False, "reason": "Prediction calibration mapping requires equally sized threshold arrays with at least two values"}
        try:
            x_numbers = [float(value) for value in x_values]
            y_numbers = [float(value) for value in y_values]
        except (TypeError, ValueError):
            return {"valid": False, "reason": "Prediction calibration thresholds must be numeric"}
        import math
        if not all(math.isfinite(value) for value in x_numbers + y_numbers):
            return {"valid": False, "reason": "Prediction calibration thresholds must be finite"}
        if any(left >= right for left, right in zip(x_numbers, x_numbers[1:])):
            return {"valid": False, "reason": "Prediction calibration x thresholds must be strictly increasing"}
        if any(left > right for left, right in zip(y_numbers, y_numbers[1:])):
            return {"valid": False, "reason": "Prediction calibration mapping must be monotonic"}
        validation = artifact.get("validation") or {}
        if validation.get("status") != "validated_oos" or int(validation.get("n_folds") or 0) < 1:
            return {"valid": False, "reason": "Prediction calibration requires at least one valid temporal OOS fold"}
        if not artifact.get("prediction_model_version"):
            return {"valid": False, "reason": "Prediction model version is missing"}
    elif model == "bb-rating":
        uncertainty_path = root / "bb_rating_uncertainty.json"
        if not uncertainty_path.exists() and root.name == "bb_rating_calibration":
            uncertainty_path = root.parent / "bb_rating_uncertainty.json"
        try:
            uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"valid": False, "reason": f"Missing or invalid bb_rating_uncertainty.json: {exc}"}
        if not artifact.get("calibration_version") or not artifact.get("bb_rating_version"):
            return {"valid": False, "reason": "BB-Rating version metadata is missing"}
        if int((artifact.get("dataset") or {}).get("rows") or 0) < 500:
            return {"valid": False, "reason": "BB-Rating calibration requires at least 500 source rows"}
        if not isinstance(uncertainty, dict) or uncertainty.get("status") not in {"fitted", "ready", "complete"}:
            return {"valid": False, "reason": "BB-Rating uncertainty artifact is incomplete or not fitted"}
        if uncertainty.get("bb_rating_version") and uncertainty.get("bb_rating_version") != artifact.get("bb_rating_version"):
            return {"valid": False, "reason": "BB-Rating and uncertainty artifact versions do not match"}
    elif model == "bb-rating-prediction":
        if artifact.get("model") != "BBRatingFuturePredictionModel":
            return {"valid": False, "reason": "Direct BB-Rating model metadata identifier is invalid"}
        if artifact.get("target") != "official_bb_rating_t_plus_1" or artifact.get("target_scale") != [1, 100]:
            return {"valid": False, "reason": "Direct BB-Rating model target contract must be official 1-100 BB-Rating"}
        if artifact.get("bb_rating_version") is None or not artifact.get("feature_version"):
            return {"valid": False, "reason": "BB-Rating engine or feature version metadata is missing"}
        if int(artifact.get("oof_samples") or 0) < 500:
            return {"valid": False, "reason": "Direct BB-Rating model requires at least 500 walk-forward OOF rows"}
        if not artifact.get("promotion_gate_passed"):
            return {"valid": False, "reason": "Walk-forward MAE did not beat the context-aware baseline"}
        for filename in ("bb_rating_future_model.joblib", "calibration.json"):
            if not (root / filename).is_file():
                return {"valid": False, "reason": f"Direct BB-Rating artifact missing: {filename}"}
        metrics = artifact.get("scenario_metrics") or {}
        if not metrics:
            return {"valid": False, "reason": "Per-scenario walk-forward metrics are missing"}
    else:
        if artifact.get("status") != "fitted":
            return {"valid": False, "reason": "Future Performance metadata is not fitted"}
        files = artifact.get("model_files") or {}
        if not files:
            return {"valid": False, "reason": "Future Performance model_files is empty"}
        for relative in files.values():
            candidate = (root / str(relative)).resolve()
            if root.resolve() not in candidate.parents or not candidate.is_file():
                return {"valid": False, "reason": f"Future Performance artifact missing or unsafe: {relative}"}
    return {
        "valid": True,
        "model": model,
        "artifact": filename,
        "version": artifact.get("model_version") or artifact.get("calibration_version") or artifact.get("future_performance_version") or artifact.get("bb_rating_version"),
        "prediction_model_version": artifact.get("prediction_model_version"),
        "feature_version": artifact.get("feature_version"),
        "data_cutoff": artifact.get("data_cutoff") or artifact.get("data_cutoff_season") or (artifact.get("dataset") or {}).get("data_cutoff") or (artifact.get("dataset") or {}).get("season_max"),
        "rows": (artifact.get("dataset") or {}).get("rows") or artifact.get("n_pairs") or artifact.get("oof_samples") or artifact.get("training_samples"),
        "seasons": (artifact.get("dataset") or {}).get("target_seasons") or (artifact.get("dataset") or {}).get("seasons") or artifact.get("training_target_seasons") or artifact.get("target_seasons") or artifact.get("seasons") or [],
        "validation": artifact.get("validation") or artifact.get("backtest") or ({"oof_mae_after_calibration": artifact.get("oof_mae_after_calibration"), "oof_baseline_mae": artifact.get("oof_baseline_mae"), "scenario_metrics": artifact.get("scenario_metrics")} if model == "bb-rating-prediction" else {}),
        "generated_at": artifact.get("generated_at"),
    }


def _candidate_root(root: Path, model: str) -> Path:
    return root / "candidates" / model


def cleanup_incomplete_candidates(root: str | Path, model: str) -> list[str]:
    """Remove candidate directories that never produced their required primary artifact.

    Callers must ensure that the model's training job is not running before
    invoking this cleanup. Candidates with a present but invalid artifact are
    retained for diagnosis.
    """
    model = _model_name(model)
    candidates_root = _candidate_root(Path(root), model)
    filename = _MODEL_PATHS[model][1]
    removed: list[str] = []
    if not candidates_root.is_dir():
        return removed
    for path in candidates_root.iterdir():
        if not path.is_dir() or (path / filename).is_file():
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            # Failure to clean up must not turn a read-only release inspection
            # into a 500; the invalid candidate will remain unselectable.
            continue
        if not path.exists():
            removed.append(path.name)
    return removed


def _release_root(root: Path, model: str) -> Path:
    return root / "releases" / model


def inspect_model(root: str | Path, model: str) -> dict[str, Any]:
    model = _model_name(model)
    root = Path(root)
    production = root / _MODEL_PATHS[model][0]
    candidates_root = _candidate_root(root, model)
    candidates = []
    if candidates_root.is_dir():
        for path in sorted((p for p in candidates_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            validation = validate_artifact(model, path)
            candidates.append({
                "id": path.name,
                "path": str(path),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                **validation,
            })
    production_validation = validate_artifact(model, production) if production.exists() else {"valid": False, "reason": "No production artifact"}
    selected_candidate = next((candidate for candidate in candidates if candidate.get("valid")), None)
    latest_invalid_candidate = next((candidate for candidate in candidates if not candidate.get("valid")), None)
    pointer = _release_root(root, model) / "previous.json"
    previous = None
    try:
        previous_info = json.loads(pointer.read_text(encoding="utf-8"))
        previous_path = Path(previous_info["path"])
        if previous_path.is_dir():
            previous = {"path": str(previous_path), "id": previous_path.name, **validate_artifact(model, previous_path)}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return {
        "model": model,
        "production_path": str(production),
        "production": production_validation,
        "candidate": selected_candidate,
        "latest_invalid_candidate": latest_invalid_candidate,
        "candidates": candidates,
        "previous": previous,
    }


def promote_model(root: str | Path, model: str, candidate_id: str | None = None) -> dict[str, Any]:
    model = _model_name(model)
    root = Path(root)
    with _LOCK:
        candidates_root = _candidate_root(root, model)
        if candidate_id:
            if candidate_id in {".", ".."} or Path(candidate_id).name != candidate_id or "/" in candidate_id or "\\" in candidate_id:
                raise ValueError("Invalid candidate id")
            candidate = candidates_root / candidate_id
        else:
            available = sorted(
                (p for p in candidates_root.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ) if candidates_root.is_dir() else []
            candidate = next(
                (path for path in available if validate_artifact(model, path).get("valid")),
                available[0] if available else None,
            )
        if candidate is None or not candidate.is_dir():
            raise RuntimeError("No candidate available to promote")
        validation = validate_artifact(model, candidate)
        if not validation.get("valid"):
            raise RuntimeError(validation.get("reason", "Candidate validation failed"))
        target = root / _MODEL_PATHS[model][0]
        releases = _release_root(root, model)
        releases.mkdir(parents=True, exist_ok=True)
        previous_pointer = releases / "previous.json"
        old_production = None
        if target.exists():
            # Preserve legacy deployments whose uncertainty JSON was stored
            # beside, rather than inside, the calibration directory.
            if model == "bb-rating":
                legacy_uncertainty = root / "bb_rating_uncertainty.json"
                nested_uncertainty = target / "bb_rating_uncertainty.json"
                if legacy_uncertainty.is_file() and not nested_uncertainty.exists():
                    shutil.copy2(legacy_uncertainty, nested_uncertainty)
            old_production = releases / f"production-{_utc_stamp()}"
            os.replace(target, old_production)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate, target)
            if old_production:
                previous_pointer.write_text(json.dumps({"path": str(old_production)}, indent=2), encoding="utf-8")
        except Exception:
            if old_production and old_production.exists() and not target.exists():
                os.replace(old_production, target)
            raise
        return {
            "promoted": True,
            "model": model,
            "version": validation.get("version"),
            "production_path": str(target),
            "previous_path": str(old_production) if old_production else None,
            "restart_required": True,
            "message": "Candidate validated and promoted. The Model Center will restart the inference API and verify readiness.",
        }


def rollback_model(root: str | Path, model: str) -> dict[str, Any]:
    model = _model_name(model)
    root = Path(root)
    with _LOCK:
        releases = _release_root(root, model)
        pointer = releases / "previous.json"
        try:
            info = json.loads(pointer.read_text(encoding="utf-8"))
            previous = Path(info["path"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("No previous release is available") from exc
        target = root / _MODEL_PATHS[model][0]
        if previous.resolve().parent != releases.resolve():
            raise RuntimeError("Previous release pointer is outside the model release directory")
        if not previous.is_dir():
            raise RuntimeError("Previous release directory is missing")
        validation = validate_artifact(model, previous)
        if not validation.get("valid"):
            raise RuntimeError(f"Previous release failed validation: {validation.get('reason')}")
        current = None
        if target.exists():
            current = releases / f"rolled-back-{_utc_stamp()}"
            os.replace(target, current)
        try:
            os.replace(previous, target)
            if current:
                pointer.write_text(json.dumps({"path": str(current)}, indent=2), encoding="utf-8")
            else:
                pointer.unlink(missing_ok=True)
        except Exception:
            if current and current.exists() and not target.exists():
                os.replace(current, target)
            raise
        return {
            "rolled_back": True,
            "model": model,
            "version": validation.get("version"),
            "production_path": str(target),
            "restart_required": True,
            "message": "Previous release restored. The Model Center will restart the inference API and verify readiness.",
        }

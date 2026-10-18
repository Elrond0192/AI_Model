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
        if len(mapping.get("x_thresholds") or []) < 2 or len(mapping.get("y_thresholds_native") or []) < 2:
            return {"valid": False, "reason": "Prediction calibration mapping has fewer than two thresholds"}
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
        if not isinstance(uncertainty, dict) or not uncertainty.get("status"):
            return {"valid": False, "reason": "BB-Rating uncertainty artifact is incomplete"}
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
        "version": artifact.get("calibration_version") or artifact.get("future_performance_version") or artifact.get("bb_rating_version"),
        "prediction_model_version": artifact.get("prediction_model_version"),
        "feature_version": artifact.get("feature_version"),
        "data_cutoff": artifact.get("data_cutoff"),
        "rows": (artifact.get("dataset") or {}).get("rows") or artifact.get("n_pairs"),
        "seasons": (artifact.get("dataset") or {}).get("target_seasons") or artifact.get("training_target_seasons") or [],
        "validation": artifact.get("validation") or artifact.get("backtest") or {},
        "generated_at": artifact.get("generated_at"),
    }


def _candidate_root(root: Path, model: str) -> Path:
    return root / "candidates" / model


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
        "candidate": candidates[0] if candidates else None,
        "candidates": candidates,
        "previous": previous,
    }


def promote_model(root: str | Path, model: str, candidate_id: str | None = None) -> dict[str, Any]:
    model = _model_name(model)
    root = Path(root)
    with _LOCK:
        candidates_root = _candidate_root(root, model)
        if candidate_id:
            candidate = candidates_root / candidate_id
            if candidate.parent != candidates_root:
                raise ValueError("Invalid candidate id")
        else:
            available = sorted((p for p in candidates_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) if candidates_root.is_dir() else []
            candidate = available[0] if available else None
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
            "message": "Candidate validated and promoted. Restart the inference API to load the new artifact.",
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
            "message": "Previous release restored. Restart the inference API to load it.",
        }

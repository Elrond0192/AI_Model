"""Immutable model-run registry, promotion gates and rollback."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable
import uuid

_logger = logging.getLogger(__name__)

PROMOTION_THRESHOLD_PCT = float(os.getenv("MODEL_PROMOTION_THRESHOLD_PCT", "2.0"))
MAX_SEGMENT_REGRESSION_PCT = float(os.getenv("MODEL_MAX_SEGMENT_REGRESSION_PCT", "15.0"))
MIN_BACKTEST_SAMPLES = int(os.getenv("MODEL_MIN_BACKTEST_SAMPLES", "1"))
TARGET_INTERVAL_COVERAGE = float(os.getenv("MODEL_TARGET_INTERVAL_COVERAGE", "0.90"))
INTERVAL_COVERAGE_TOLERANCE = float(os.getenv("MODEL_INTERVAL_COVERAGE_TOLERANCE", "0.10"))
_REQUIRED_ARTIFACTS = (
    "performance_model.joblib",
    "compatibility_model.joblib",
    "conformal.joblib",
    "metadata.json",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_registry(model_root: Path) -> dict:
    path = model_root / "registry.json"
    if not path.exists():
        return {"history": []}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, list):
        return {"history": parsed}
    if not isinstance(parsed, dict):
        raise RuntimeError("registry.json must contain an object")
    parsed.setdefault("history", [])
    return parsed


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _save_registry(model_root: Path, registry: dict) -> None:
    _atomic_json(model_root / "registry.json", registry)


def _run_path(model_root: Path, entry: dict) -> Path:
    raw = str(entry.get("path", "")).strip()
    if not raw:
        raise RuntimeError("Registry entry has no immutable run path")
    path = (model_root / raw).resolve()
    root = model_root.resolve()
    if root not in path.parents:
        raise RuntimeError("Registry run path escapes model root")
    return path


def _artifacts_exist(run_dir: Path) -> bool:
    return run_dir.is_dir() and all((run_dir / name).is_file() for name in _REQUIRED_ARTIFACTS)


def register_candidate(
    model_dir: str,
    run_id: str,
    run_path: str | Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Register an immutable trained run without changing production."""
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_path).resolve()
    root = model_root.resolve()
    if root not in run_dir.parents:
        raise ValueError("run_path must live below model_dir")
    if not _artifacts_exist(run_dir):
        raise RuntimeError("Candidate run is missing required model artifacts")
    report = metadata.get("backtest") or {}
    overall = report.get("overall") or {}
    entry = {
        "run_id": str(run_id),
        "path": str(run_dir.relative_to(root)),
        "status": "candidate",
        "registered_at": _utcnow(),
        "model_version": metadata.get("model_version"),
        "feature_version": metadata.get("feature_version"),
        "data_cutoff": metadata.get("data_cutoff"),
        "overall_rmse": overall.get("rmse", report.get("overall_rmse")),
        "backtest": report,
        "metadata": metadata,
    }
    registry = _load_registry(model_root)
    registry["candidate"] = entry
    registry.setdefault("history", []).append(
        {"event": "candidate_registered", "at": _utcnow(), "run_id": str(run_id)}
    )
    _save_registry(model_root, registry)
    return entry


def _promotion_gate_errors(candidate: dict) -> list[str]:
    errors: list[str] = []
    report = candidate.get("backtest") or {}
    overall = report.get("overall") or {}
    if not report.get("valid"):
        errors.append("backtest is not valid")
    rmse = overall.get("rmse", report.get("overall_rmse"))
    try:
        if not math.isfinite(float(rmse)):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("overall RMSE is missing or non-finite")
    n = int(overall.get("n", 0) or 0)
    if n < MIN_BACKTEST_SAMPLES:
        errors.append(f"backtest has {n} samples; minimum is {MIN_BACKTEST_SAMPLES}")

    # The contextual ensemble must justify its complexity on untouched seasons.
    ensemble_vs_base = report.get("ensemble_vs_base_delta")
    if ensemble_vs_base is None or float(ensemble_vs_base) < 0:
        errors.append("full ensemble does not beat the base XGBoost forecast")
    ensemble_vs_persistence = report.get("ensemble_vs_persistence_delta")
    if ensemble_vs_persistence is None or float(ensemble_vs_persistence) < 0:
        errors.append("full ensemble does not beat previous-season persistence")

    coverage = overall.get("interval_coverage")
    if coverage is not None:
        lo = TARGET_INTERVAL_COVERAGE - INTERVAL_COVERAGE_TOLERANCE
        hi = min(1.0, TARGET_INTERVAL_COVERAGE + INTERVAL_COVERAGE_TOLERANCE)
        if not lo <= float(coverage) <= hi:
            errors.append(
                f"interval coverage {float(coverage):.3f} outside [{lo:.3f}, {hi:.3f}]"
            )
    return errors


def _segment_regressions(candidate: dict, production: dict) -> list[str]:
    cand_segments = (candidate.get("backtest") or {}).get("by_league") or {}
    prod_segments = (production.get("backtest") or {}).get("by_league") or {}
    errors: list[str] = []
    for segment, cand in cand_segments.items():
        prod = prod_segments.get(segment)
        if not isinstance(cand, dict) or not isinstance(prod, dict):
            continue
        if min(int(cand.get("n", 0) or 0), int(prod.get("n", 0) or 0)) < 5:
            continue
        cand_rmse = cand.get("rmse")
        prod_rmse = prod.get("rmse")
        if cand_rmse is None or prod_rmse in (None, 0):
            continue
        regression_pct = (float(cand_rmse) - float(prod_rmse)) / float(prod_rmse) * 100.0
        if regression_pct > MAX_SEGMENT_REGRESSION_PCT:
            errors.append(
                f"league {segment} RMSE regresses {regression_pct:.2f}% "
                f"(limit {MAX_SEGMENT_REGRESSION_PCT:.2f}%)"
            )
    return errors


def _activate_run(model_root: Path, entry: dict) -> Path:
    source = _run_path(model_root, entry)
    if not _artifacts_exist(source):
        raise RuntimeError("Run artifacts are incomplete")
    production = model_root / "production"
    stage = model_root / f".production-{uuid.uuid4().hex}"
    backup = model_root / f".production-backup-{uuid.uuid4().hex}"
    shutil.copytree(source, stage)
    _atomic_json(
        stage / "production_manifest.json",
        {
            "run_id": entry.get("run_id"),
            "activated_at": _utcnow(),
            "source_path": entry.get("path"),
        },
    )
    try:
        if production.exists():
            production.rename(backup)
        stage.rename(production)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if production.exists() and backup.exists():
            shutil.rmtree(production, ignore_errors=True)
        if backup.exists() and not production.exists():
            backup.rename(production)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return production


def promote_if_better(
    model_dir: str = "models_saved",
    min_improvement_pct: float = PROMOTION_THRESHOLD_PCT,
) -> Dict[str, Any]:
    """Promote a candidate only after OOT full-ensemble and segment gates pass."""
    model_root = Path(model_dir)
    registry = _load_registry(model_root)
    candidate = registry.get("candidate")
    production = registry.get("production")
    if not candidate:
        return {"promoted": False, "reason": "No candidate model in registry"}

    errors = _promotion_gate_errors(candidate)
    if production:
        errors.extend(_segment_regressions(candidate, production))
        cand_rmse = candidate.get("overall_rmse")
        prod_rmse = production.get("overall_rmse")
        if cand_rmse is None or prod_rmse in (None, 0):
            errors.append("candidate/production OOT RMSE is unavailable")
        else:
            improvement = (float(prod_rmse) - float(cand_rmse)) / float(prod_rmse) * 100
            if improvement < float(min_improvement_pct):
                errors.append(
                    f"OOT RMSE improvement {improvement:.2f}% is below "
                    f"required {float(min_improvement_pct):.2f}%"
                )
    if errors:
        reason = "; ".join(errors)
        _logger.info("[Promotion] rejected: %s", reason)
        return {"promoted": False, "reason": reason, "gates": errors}

    _activate_run(model_root, candidate)
    if production:
        registry["previous"] = production
    registry["production"] = {
        **candidate,
        "status": "production",
        "promoted_at": _utcnow(),
    }
    registry.pop("candidate", None)
    registry.setdefault("history", []).append(
        {"event": "promoted", "at": _utcnow(), "run_id": candidate.get("run_id")}
    )
    _save_registry(model_root, registry)
    return {
        "promoted": True,
        "reason": f"Promoted run {candidate.get('run_id')} to production",
        "run_id": candidate.get("run_id"),
        "production_dir": str(model_root / "production"),
    }


def rollback_to_previous(model_dir: str = "models_saved") -> Dict[str, Any]:
    model_root = Path(model_dir)
    registry = _load_registry(model_root)
    previous = registry.get("previous")
    production = registry.get("production")
    if not previous:
        return {"rolled_back": False, "reason": "No previous model to roll back to"}
    _activate_run(model_root, previous)
    registry["production"] = {
        **previous,
        "status": "production",
        "promoted_at": _utcnow(),
    }
    if production:
        registry["previous"] = production
    registry.setdefault("history", []).append(
        {"event": "rollback", "at": _utcnow(), "run_id": previous.get("run_id")}
    )
    _save_registry(model_root, registry)
    return {
        "rolled_back": True,
        "reason": f"Rolled back to run {previous.get('run_id')}",
        "run_id": previous.get("run_id"),
    }


def get_promotion_status(model_dir: str = "models_saved") -> Dict[str, Any]:
    registry = _load_registry(Path(model_dir))
    return {
        "production": registry.get("production"),
        "candidate": registry.get("candidate"),
        "previous": registry.get("previous"),
        "history": registry.get("history", [])[-20:],
    }


def production_model_dir(model_dir: str = "models_saved") -> Path:
    """Return the active production artifact directory or fail closed."""
    path = Path(model_dir) / "production"
    if not _artifacts_exist(path):
        raise RuntimeError("No promoted production model is available")
    return path

"""Model promotion and rollback logic.

Models are promoted from 'candidate' to 'production' status in registry.json
only when the new model's val_rmse improves over the current production model
by at least PROMOTION_THRESHOLD_PCT percent.

Usage::

    from basketball_ai.models.promote import promote_if_better, rollback_to_previous

    result = promote_if_better("models_saved", min_improvement_pct=2.0)
    # result: {"promoted": True, "reason": "..."}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

_logger = logging.getLogger(__name__)

PROMOTION_THRESHOLD_PCT = float(
    __import__("os").environ.get("MODEL_PROMOTION_THRESHOLD_PCT", "2.0")
)


def _load_registry(model_dir: Path) -> dict:
    reg_path = model_dir / "registry.json"
    if not reg_path.exists():
        return {}
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    return registry if isinstance(registry, dict) else {"history": registry}


def _save_registry(model_dir: Path, reg: dict) -> None:
    (model_dir / "registry.json").write_text(
        json.dumps(reg, indent=2, default=str), encoding="utf-8"
    )


def promote_if_better(
    model_dir: str = "models_saved",
    min_improvement_pct: float = PROMOTION_THRESHOLD_PCT,
) -> Dict[str, Any]:
    """Promote 'candidate' model to 'production' if it beats the current production model.

    Args:
        model_dir:            Directory containing registry.json and model files.
        min_improvement_pct:  Minimum percentage improvement in val_rmse required.

    Returns:
        Dict with keys 'promoted' (bool) and 'reason' (str).
    """
    model_path = Path(model_dir)
    reg = _load_registry(model_path)

    candidate  = reg.get("candidate")
    production = reg.get("production")

    if not candidate:
        return {"promoted": False, "reason": "No candidate model in registry"}

    cand_rmse = candidate.get("val_rmse")
    if cand_rmse is None:
        return {"promoted": False, "reason": "Candidate has no val_rmse metric"}

    if production:
        prod_rmse = production.get("val_rmse")
        if prod_rmse is not None:
            improvement = (prod_rmse - cand_rmse) / prod_rmse * 100
            if improvement < min_improvement_pct:
                msg = (
                    f"Candidate val_rmse={cand_rmse:.4f} does not improve over "
                    f"production val_rmse={prod_rmse:.4f} by {min_improvement_pct}% "
                    f"(actual improvement: {improvement:.2f}%)"
                )
                _logger.info("[Promotion] Rejected: %s", msg)
                return {"promoted": False, "reason": msg}

    # Archive current production as 'previous'
    if production:
        reg["previous"] = production
        _logger.info("[Promotion] Archived production as previous")

    # Promote candidate
    reg["production"] = {**candidate, "status": "production"}
    reg.pop("candidate", None)
    _save_registry(model_path, reg)

    prod_rmse_str = (
        f"{production.get('val_rmse'):.4f}"
        if production and production.get("val_rmse") is not None
        else "?"
    )
    msg = (
        f"Promoted candidate (val_rmse={cand_rmse:.4f}) to production"
        + (f", replaced production (val_rmse={prod_rmse_str})" if production else "")
    )
    _logger.info("[Promotion] %s", msg)
    return {"promoted": True, "reason": msg}


def rollback_to_previous(model_dir: str = "models_saved") -> Dict[str, Any]:
    """Roll back production to the previously saved model.

    Args:
        model_dir: Directory containing registry.json.

    Returns:
        Dict with 'rolled_back' (bool) and 'reason' (str).
    """
    model_path = Path(model_dir)
    reg = _load_registry(model_path)

    previous   = reg.get("previous")
    production = reg.get("production")

    if not previous:
        return {"rolled_back": False, "reason": "No previous model to roll back to"}

    reg["candidate"] = production  # demote current to candidate
    reg["production"] = {**previous, "status": "production"}
    reg.pop("previous", None)
    _save_registry(model_path, reg)

    _logger.info("[Rollback] Rolled back to previous model")
    return {"rolled_back": True, "reason": "Rolled back to previous model"}


def get_promotion_status(model_dir: str = "models_saved") -> Dict[str, Any]:
    """Return current promotion state from registry.json."""
    reg = _load_registry(Path(model_dir))
    return {
        "production": reg.get("production"),
        "candidate":  reg.get("candidate"),
        "previous":   reg.get("previous"),
    }

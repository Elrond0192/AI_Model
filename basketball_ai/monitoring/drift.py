"""Population Stability Index (PSI) – data drift detection.

Computes PSI between a reference distribution (captured during training)
and the current distribution of numeric feature columns.

PSI formula (per feature, per bin):
  PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

PSI interpretation (common thresholds):
  < 0.10  – no significant change, model is stable
  0.10–0.25 – moderate change, monitor
  > 0.25  – significant drift, consider retraining

Usage
-----
After training::

    from basketball_ai.monitoring.drift import capture_reference, compute_psi_report

    ref = capture_reference(train_data)  # called once after training
    ...

At reload / inference time::

    report = compute_psi_report(current_data, ref)
    # report is a dict {feature: psi_value, ...}
    # A WARNING is logged for any feature with PSI > 0.25

The reference dict can be persisted (JSON/joblib) so the comparison
survives service restarts.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Drift thresholds
_PSI_WARN  = 0.10
_PSI_ALERT = 0.25

# Features to monitor (subset of numeric columns typical in player_stats)
_MONITOR_COLS: List[str] = [
    "points", "assists", "rebounds", "steals", "blocks",
    "ts_pct", "usg_pct", "bpm", "obpm", "dbpm",
    "per", "raptor_off", "raptor_def", "spm",
    "gm_sc", "ows", "dws",
    "games_played", "minutes_per_game",
    "rating",
]

_N_BINS = 10  # number of equal-width bins used for PSI


def capture_reference(data: Dict) -> Dict[str, Dict]:
    """Capture per-column distribution statistics from a training dataset.

    Args:
        data: Full data dict from the PostgreSQL loader.

    Returns:
        Reference dict mapping column → {"edges": ..., "expected_pct": ...}.
    """
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    ref: Dict[str, Dict] = {}

    for col in _MONITOR_COLS:
        if col not in player_stats.columns:
            continue
        vals = player_stats[col].dropna().values.astype(float)
        if len(vals) < 10:
            continue
        counts, edges = np.histogram(vals, bins=_N_BINS)
        total = counts.sum()
        expected_pct = (counts / total).clip(min=1e-6).tolist()
        ref[col] = {
            "edges": edges.tolist(),
            "expected_pct": expected_pct,
            "n_samples": int(total),
        }

    logger.info(
        "[DriftDetection] Reference distribution captured for %d features.",
        len(ref),
    )
    return ref


def _psi_one(expected_pct: List[float], actual_pct: List[float]) -> float:
    """Compute PSI for a single feature given its bin percentages."""
    e = np.array(expected_pct, dtype=float).clip(min=1e-6)
    a = np.array(actual_pct, dtype=float).clip(min=1e-6)
    # Normalise in case actual totals differ slightly
    a = a / a.sum()
    e = e / e.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi_report(
    current_data: Dict,
    reference: Optional[Dict[str, Dict]],
    warn_threshold: float = _PSI_WARN,
    alert_threshold: float = _PSI_ALERT,
) -> Dict[str, float]:
    """Compare the current dataset against the reference distribution.

    Args:
        current_data:    Full data dict with current player_stats.
        reference:       Reference dict from ``capture_reference``.
        warn_threshold:  PSI above this value triggers a WARNING log.
        alert_threshold: PSI above this value triggers an ERROR log and
                         marks the feature as potentially stale.

    Returns:
        Dict mapping feature name → PSI value.
        An empty dict is returned if reference is None or empty.
    """
    if not reference:
        return {}

    player_stats: pd.DataFrame = current_data.get("player_stats", pd.DataFrame())
    if player_stats.empty:
        return {}

    report: Dict[str, float] = {}
    stale_features: List[str] = []

    for col, ref_info in reference.items():
        if col not in player_stats.columns:
            continue
        vals = player_stats[col].dropna().values.astype(float)
        if len(vals) < 5:
            continue
        edges = np.array(ref_info["edges"])
        # Compute actual histogram using same bin edges as reference
        counts, _ = np.histogram(vals, bins=edges)
        total = counts.sum()
        if total == 0:
            continue
        actual_pct = (counts / total).clip(min=1e-6).tolist()
        psi = _psi_one(ref_info["expected_pct"], actual_pct)
        report[col] = round(psi, 4)

        if psi >= alert_threshold:
            stale_features.append(col)
            logger.warning(
                "[DriftDetection] SIGNIFICANT DRIFT detected for '%s': "
                "PSI=%.4f (threshold=%.2f). "
                "Model may be stale relative to current data. Consider retraining.",
                col, psi, alert_threshold,
            )
        elif psi >= warn_threshold:
            logger.info(
                "[DriftDetection] Moderate drift for '%s': PSI=%.4f", col, psi
            )

    if stale_features:
        logger.warning(
            "[DriftDetection] %d feature(s) show significant drift: %s",
            len(stale_features), stale_features,
        )
    else:
        logger.info(
            "[DriftDetection] No significant data drift detected (%d features checked).",
            len(report),
        )

    return report


# ---------------------------------------------------------------------------
# Per-segment drift (by league or role) – KS test
# ---------------------------------------------------------------------------

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def compute_ks_drift(
    reference_df: "pd.DataFrame",
    current_df: "pd.DataFrame",
    cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run two-sample KS test between reference and current distributions.

    Returns dict mapping feature → ks_pvalue.
    p-value < 0.05 indicates significant drift.

    Falls back to a simple mean-shift heuristic when scipy is not available.
    """
    if cols is None:
        cols = _MONITOR_COLS

    result: Dict[str, float] = {}
    for col in cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].dropna().values.astype(float)
        cur_vals = current_df[col].dropna().values.astype(float)
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue
        if _SCIPY_AVAILABLE:
            _, pval = _scipy_stats.ks_2samp(ref_vals, cur_vals)
            result[col] = round(float(pval), 6)
        else:
            # Fallback: mean-shift ratio as proxy
            ref_mean = float(np.mean(ref_vals))
            cur_mean = float(np.mean(cur_vals))
            if ref_mean != 0:
                shift = abs(cur_mean - ref_mean) / abs(ref_mean)
            else:
                shift = abs(cur_mean - ref_mean)
            # Convert to a pseudo-pvalue: shift > 0.2 → significant
            result[col] = round(float(max(0.0, 1.0 - shift * 5)), 6)
    return result


def compute_segment_drift(
    data: Dict,
    reference: Optional[Dict[str, Dict]],
    segment_col: str = "league_id",
) -> Dict[str, Dict[str, float]]:
    """Compute PSI drift broken down by a segment column.

    Args:
        data:         Full data dict (from loader).
        reference:    Reference dict from ``capture_reference``.
        segment_col:  Column to segment by (e.g. ``"league_id"``).

    Returns:
        Dict mapping segment_value → {feature: psi_value}.
    """
    if not reference:
        return {}

    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    if player_stats.empty or segment_col not in player_stats.columns:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for seg_val, group in player_stats.groupby(segment_col):
        seg_data = {"player_stats": group}
        seg_report = compute_psi_report(seg_data, reference)
        if seg_report:
            result[str(seg_val)] = seg_report

    return result

"""Guarded observational causal estimators for player/team attribution.

This module deliberately returns no effect when design, support or diagnostics
are inadequate. Its estimate is a cohort-level team-switch ATT, not a
player-specific causal effect.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


_REQUIRED = {
    "global_id", "treatment", "outcome", "baseline_bb_rating",
    "minutes_pre", "usage_pre", "age_pre", "team_strength_pre", "transition_season", "role_pre",
}


def _unavailable(reason: str, *, rows: int = 0, treated: int = 0, control: int = 0) -> dict[str, Any]:
    return {
        "available": False,
        "estimand": "average_treatment_effect_on_treated_team_switchers",
        "effect_bb_rating_points": None,
        "confidence_interval_95": None,
        "identification": "unavailable",
        "reason": reason,
        "diagnostics": {
            "rows": int(rows), "treated": int(treated), "control": int(control),
        },
    }


def _smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    if not len(x_t) or not len(x_c):
        return math.inf
    var_t = float(np.var(x_t, ddof=1)) if len(x_t) > 1 else 0.0
    var_c = float(np.var(x_c, ddof=1)) if len(x_c) > 1 else 0.0
    pooled = math.sqrt((var_t + var_c) / 2.0)
    delta = float(np.mean(x_t) - np.mean(x_c))
    return abs(delta / pooled) if pooled > 1e-9 else (0.0 if abs(delta) < 1e-9 else math.inf)


def _weighted_smd(x_t: np.ndarray, x_c: np.ndarray, w_t: np.ndarray, w_c: np.ndarray) -> float:
    if not len(x_t) or not len(x_c):
        return math.inf
    w_t = np.asarray(w_t, dtype=float)
    w_c = np.asarray(w_c, dtype=float)
    w_t = w_t / max(float(w_t.sum()), 1e-12)
    w_c = w_c / max(float(w_c.sum()), 1e-12)
    mean_t = float(np.sum(w_t * x_t))
    mean_c = float(np.sum(w_c * x_c))
    var_t = float(np.sum(w_t * (x_t - mean_t) ** 2))
    var_c = float(np.sum(w_c * (x_c - mean_c) ** 2))
    pooled = math.sqrt((var_t + var_c) / 2.0)
    delta = abs(mean_t - mean_c)
    return delta / pooled if pooled > 1e-9 else (0.0 if delta < 1e-9 else math.inf)


def estimate_team_switch_att(panel: pd.DataFrame) -> dict[str, Any]:
    """Estimate a cohort-level team-switch ATT using propensity and outcome
    nuisance models on a single transition per canonical player. Strict support gates
    prevent a numeric estimate when overlap or balance is inadequate.
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return _unavailable("causal_panel_empty")
    missing = sorted(_REQUIRED - set(panel.columns))
    if missing:
        return _unavailable("causal_panel_missing_columns:" + ",".join(missing), rows=len(panel))

    frame = panel.copy()
    frame["treatment"] = pd.to_numeric(frame["treatment"], errors="coerce")
    frame["outcome"] = pd.to_numeric(frame["outcome"], errors="coerce")
    numeric_covariates = ["baseline_bb_rating", "minutes_pre", "usage_pre", "age_pre", "team_strength_pre", "transition_season"]
    for col in numeric_covariates:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["role_pre"] = frame["role_pre"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["global_id", "treatment", "outcome", *numeric_covariates]
    )
    role_dummies = pd.get_dummies(frame["role_pre"], prefix="role", drop_first=True, dtype=float)
    covariate_frame = pd.concat(
        [frame[numeric_covariates].reset_index(drop=True), role_dummies.reset_index(drop=True)],
        axis=1,
    )
    covariates = list(covariate_frame.columns)
    frame = frame[frame["treatment"].isin([0, 1])].drop_duplicates("global_id", keep="last")
    n = len(frame)
    t = frame["treatment"].to_numpy(dtype=int) if n else np.array([], dtype=int)
    nt = int(t.sum())
    nc = int(n - nt)
    if n < 60 or nt < 20 or nc < 20:
        return _unavailable("insufficient_treated_control_support", rows=n, treated=nt, control=nc)

    x = covariate_frame.to_numpy(dtype=float)
    y = frame["outcome"].to_numpy(dtype=float)
    x_original = x.copy()
    t_original = t.copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    try:
        propensity_model = LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")
        propensity_model.fit(x_scaled, t)
        propensity = propensity_model.predict_proba(x_scaled)[:, 1]
        # Do not clip propensity and pretend that a lack of overlap is solved.
        in_overlap = (propensity >= 0.05) & (propensity <= 0.95)
        overlap_share = float(in_overlap.mean())
        if overlap_share < 0.80 or int((in_overlap & (t == 1)).sum()) < 20 or int((in_overlap & (t == 0)).sum()) < 20:
            return _unavailable("propensity_overlap_failed", rows=n, treated=nt, control=nc)

        frame = frame.loc[in_overlap].reset_index(drop=True)
        x_scaled = x_scaled[in_overlap]
        y = y[in_overlap]
        t = t[in_overlap]
        propensity = propensity[in_overlap]
        nt, nc = int(t.sum()), int((1 - t).sum())
        if min(nt, nc) < 20:
            return _unavailable("insufficient_overlap_after_trim", rows=len(frame), treated=nt, control=nc)

        m0 = LinearRegression().fit(x_scaled[t == 0], y[t == 0])
        mu0 = m0.predict(x_scaled)
        p = np.clip(propensity, 0.05, 0.95)
        # ATT doubly-robust score: treated outcome residual minus the propensity-weighted control residual.
        control_weights = p[t == 0] / (1.0 - p[t == 0])
        control_weights = control_weights / max(float(control_weights.mean()), 1e-12)
        control_residual = float(np.mean(control_weights * (y[t == 0] - mu0[t == 0])))
        effect = float(np.mean(y[t == 1] - mu0[t == 1]) - control_residual)
        # Influence-function interval; production claims remain conditional on
        # the stated no-unmeasured-confounding and positivity assumptions.
        treated_residual = y[t == 1] - mu0[t == 1]
        weighted_control_residual = control_weights * (y[t == 0] - mu0[t == 0])
        se = math.sqrt(
            (float(np.var(treated_residual, ddof=1)) / max(nt, 1) if nt > 1 else 0.0)
            + (float(np.var(weighted_control_residual, ddof=1)) / max(nc, 1) if nc > 1 else 0.0)
        )
        ci = [round(effect - 1.96 * se, 3), round(effect + 1.96 * se, 3)]

        balance_before = {}
        balance_after = {}
        for idx, name in enumerate(covariates):
            balance_before[name] = round(_smd(x_original[t_original == 1, idx],
                                               x_original[t_original == 0, idx]), 3)
            balance_after[name] = round(_weighted_smd(x_scaled[t == 1, idx], x_scaled[t == 0, idx], np.ones(nt), control_weights), 3)
        max_smd = max(balance_after.values()) if balance_after else math.inf
        if max_smd > 0.25:
            return _unavailable("covariate_balance_failed", rows=len(frame), treated=nt, control=nc) | {
                "diagnostics": {
                    "rows": len(frame), "treated": nt, "control": nc,
                    "propensity_overlap_share": round(overlap_share, 3),
                    "standardized_mean_difference_after": balance_after,
                    "max_abs_smd_after": round(max_smd, 3),
                }
            }

        return {
            "available": True,
            "estimand": "average_treatment_effect_on_treated_team_switchers",
            "treatment": "team_switch_between_consecutive_completed_seasons",
            "outcome": "next_observed_season_canonical_bb_rating_1_100",
            "effect_bb_rating_points": round(effect, 3),
            "confidence_interval_95": ci,
            "identification": "observational_assumptions_required",
            "diagnostics": {
                "rows": len(frame), "treated": nt, "control": nc,
                "propensity_overlap_share": round(overlap_share, 3),
                "standardized_mean_difference_before": balance_before,
                "standardized_mean_difference_after": balance_after,
                "max_abs_smd_after": round(max_smd, 3),
            },
            "assumptions": [
                "No unmeasured confounding after conditioning on recorded pre-treatment covariates.",
                "Consistency of the team-switch treatment definition.",
                "Positivity/common support within the analyzed league and competition.",
                "BB-Rating calibration is comparable across the included seasons.",
            ],
            "limitations": [
                "Cohort-level ATT for observed team switchers, not an individual player's causal effect.",
                "Observational estimate; it is not proof of causality and remains conditional on untestable assumptions.",
                "The estimate must not be used if calibration releases or league/competition contexts are not comparable.",
            ],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _unavailable("estimator_failed:" + str(exc)[:160], rows=n, treated=nt, control=nc)

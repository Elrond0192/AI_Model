"""Guarded observational causal estimators for player/team attribution.

This module deliberately returns no effect when design, support or diagnostics
are inadequate. Its estimate is a cohort-level team-switch ATT, not a
player-specific causal effect.
"""
from __future__ import annotations

import math
import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
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
    return delta / pooled if pooled > 1e-9 else (0.0 if delta < 1e-9 else 999.0)


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
    frame = frame[frame["treatment"].isin([0, 1])].drop_duplicates("global_id", keep="last")
    role_dummies = pd.get_dummies(frame["role_pre"], prefix="role", drop_first=True, dtype=float)
    covariate_frame = pd.concat(
        [frame[numeric_covariates].reset_index(drop=True), role_dummies.reset_index(drop=True)],
        axis=1,
    )
    covariates = list(covariate_frame.columns)
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


def estimate_teammate_exposure_att(
    panel: pd.DataFrame,
    *,
    teammate_global_id: str | None = None,
) -> dict[str, Any]:
    """Estimate an observational cohort ATT for a named teammate exposure.

    Each row represents one focal player's latest adjacent completed-season
    transition. Treatment is a newly observed shared-team relationship: the focal
    player and named teammate were on different teams in the baseline season and
    are on the same team in the outcome season. Continuing teammate relationships
    and pairs without an observed teammate baseline team are excluded by the
    builder. Outcome is canonical next-season BB-Rating (1-100).
    """
    required = _REQUIRED | {"teammate_global_id", "league_key", "competition"}
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return _unavailable("teammate_panel_empty")
    missing = sorted(required - set(panel.columns))
    if missing:
        return _unavailable("teammate_panel_missing_columns:" + ",".join(missing), rows=len(panel))
    frame = panel.copy()
    if teammate_global_id:
        frame = frame[
            frame["teammate_global_id"].astype(str).str.strip().str.casefold()
            == str(teammate_global_id).strip().casefold()
        ].copy()
    if frame.empty:
        return _unavailable("named_teammate_not_in_panel")
    # Do not pool league/competition scales or treatment definitions.
    contexts = frame[["league_key", "competition"]].dropna().drop_duplicates()
    if len(contexts) != 1:
        return _unavailable("teammate_panel_mixes_league_or_competition", rows=len(frame))
    # Reuse the guarded estimator, while keeping teammate identity in the
    # treatment definition and the result separate from team-switch ATT.
    result = estimate_team_switch_att(frame)
    result["estimand"] = "observational_teammate_exposure_ATT"
    result["treatment"] = "newly_observed_shared_team_relationship_between_completed_seasons"
    result["teammate_global_id"] = str(teammate_global_id) if teammate_global_id else None
    result["outcome"] = "next_observed_season_canonical_bb_rating_1_100"
    result["limitations"] = list(result.get("limitations", [])) + [
        "This is a cohort-level observational teammate-exposure estimate, not the named focal player's individual causal effect.",
        "Treatment definition, temporal ordering and within-context positivity must be verified before interpretation.",
        "Shared minutes, lineup Net Rating and team outcomes are not substitutes for the focal player's canonical BB-Rating outcome.",
    ]
    return result


def persist_causal_panel(
    panel: pd.DataFrame,
    engine: Any,
    *,
    source_type: str,
    league_key: str,
    competition: str,
) -> dict[str, Any]:
    """Atomically replace one auditable source/context slice of the causal panel.

    The caller must supply outcomes already produced by the canonical BB-Rating
    engine and a pre-registered treatment/control definition. No native source
    rating fallback is accepted. This helper persists evidence; it does not
    validate causal identification or automatically schedule a refresh.
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return {"status": "not_written", "rows": 0, "reason": "panel_empty"}
    if not source_type.strip() or not league_key.strip() or not competition.strip():
        return {"status": "not_written", "rows": 0, "reason": "context_required"}
    required = _REQUIRED | {"league_key", "competition"}
    missing = sorted(required - set(panel.columns))
    if missing:
        return {
            "status": "not_written", "rows": 0,
            "reason": "panel_missing_columns:" + ",".join(missing),
        }
    frame = panel.copy()
    if "rating" in frame.columns or "native_rating" in frame.columns:
        return {
            "status": "not_written", "rows": 0,
            "reason": "native_rating_columns_forbidden_use_canonical_bb_rating",
        }
    frame["global_id"] = frame["global_id"].astype(str).str.strip()
    frame["league_key"] = frame["league_key"].astype(str).str.strip().str.upper()
    frame["competition"] = frame["competition"].astype(str).str.strip().str.upper()
    frame = frame[
        (frame["league_key"] == league_key.strip().upper())
        & (frame["competition"] == competition.strip().upper())
    ].copy()
    if frame.empty:
        return {"status": "not_written", "rows": 0, "reason": "context_has_no_rows"}
    frame["treatment"] = pd.to_numeric(frame["treatment"], errors="coerce")
    frame["outcome"] = pd.to_numeric(frame["outcome"], errors="coerce")
    if not frame["treatment"].isin([0, 1]).all():
        return {"status": "not_written", "rows": 0, "reason": "treatment_must_be_binary"}
    if frame["global_id"].eq("").any() or frame["global_id"].isna().any():
        return {"status": "not_written", "rows": 0, "reason": "canonical_global_id_required"}
    if frame["outcome"].isna().any() or not frame["outcome"].between(1, 100).all():
        return {"status": "not_written", "rows": 0, "reason": "canonical_bb_rating_outcome_required"}
    identity_columns = ["global_id", "transition_season"]
    if source_type.strip() == "teammate_exposure":
        identity_columns.append("teammate_global_id")
    if frame.duplicated(identity_columns).any():
        return {"status": "not_written", "rows": 0, "reason": "duplicate_player_transition"}
    frame["source_type"] = source_type.strip()
    frame["next_outcome"] = frame["outcome"].astype(float)
    treatment_definition = (
        "newly_observed_shared_team_relationship_between_completed_seasons"
        if source_type.strip() == "teammate_exposure"
        else "team_switch_between_adjacent_completed_seasons"
    )
    def build_panel_details(row):
        details = {
            "estimand": source_type.strip(),
            "treatment_definition": treatment_definition,
            "identification": "observational_assumptions_required",
        }
        # Keep the reason attached to the persisted cohort row so the operator
        # can distinguish identity/context gaps from missing source metrics.
        # This uses the existing JSONB details column; no schema migration needed.
        diagnostic = row.get("_team_strength_diagnostic")
        if (
            source_type.strip() == "teammate_exposure"
            and isinstance(diagnostic, dict)
            and pd.isna(row.get("team_strength_pre"))
        ):
            # Only persist the extra payload for missing values to limit JSONB
            # overhead on the much larger set of complete rows.
            details["team_strength_pre_diagnostic"] = diagnostic
        return json.dumps(details)

    frame["details"] = frame.apply(build_panel_details, axis=1)
    frame["refreshed_at"] = pd.Timestamp.now(tz="UTC")
    if "teammate_global_id" not in frame.columns:
        frame["teammate_global_id"] = None
    columns = [
        "treatment", "next_outcome", "source_type", "details", "global_id",
        "teammate_global_id", "league_key", "competition", "baseline_bb_rating",
        "minutes_pre", "usage_pre", "age_pre", "team_strength_pre",
        "transition_season", "role_pre", "refreshed_at",
    ]
    persisted = frame[columns].copy()
    persisted["treatment"] = persisted["treatment"].astype(int).astype(str)
    persisted["details"] = persisted["details"].map(json.loads)
    teammate_ids = (
        frame["teammate_global_id"].dropna().astype(str).str.strip().unique().tolist()
        if "teammate_global_id" in frame.columns else []
    )
    teammate_scope = teammate_ids[0] if len(teammate_ids) == 1 else None
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    'DELETE FROM "AI_Source"."SimulationCausalPanel" '
                    'WHERE source_type = :source_type AND league_key = :league_key '
                    'AND competition = :competition '
                    'AND (CAST(:teammate_global_id AS text) IS NULL '
                    'OR teammate_global_id = CAST(:teammate_global_id AS text))'
                ),
                {
                    "source_type": source_type.strip(),
                    "league_key": league_key.strip().upper(),
                    "competition": competition.strip().upper(),
                    "teammate_global_id": teammate_scope,
                },
            )
            persisted.to_sql(
                "SimulationCausalPanel",
                connection,
                schema="AI_Source",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
                dtype={"details": JSONB},
            )
    except Exception as exc:
        return {
            "status": "failed", "rows": 0,
            "reason": "causal_panel_persist_failed:" + str(exc)[:240],
        }
    return {
        "status": "refreshed",
        "rows": int(len(persisted)),
        "source_type": source_type.strip(),
        "league_key": league_key.strip().upper(),
        "competition": competition.strip().upper(),
    }

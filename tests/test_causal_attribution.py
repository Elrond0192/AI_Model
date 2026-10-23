import numpy as np
import pandas as pd

from basketball_ai.scenarios.causal_attribution import (
    estimate_team_switch_att,
    estimate_teammate_exposure_att,
    persist_causal_panel,
)


def _panel(n=240, seed=19):
    rng = np.random.default_rng(seed)
    baseline = rng.normal(60, 8, n)
    minutes = np.clip(rng.normal(22, 5, n), 5, 38)
    usage = np.clip(rng.normal(.20, .05, n), .05, .4)
    age = np.clip(rng.normal(27, 4, n), 18, 38)
    strength = rng.normal(0, 1, n)
    treatment = rng.binomial(1, .45, n)
    outcome = 0.65 * baseline + 0.12 * minutes + 8 * treatment + 2 * strength + rng.normal(0, 3, n)
    return pd.DataFrame({
        "global_id": [f"p{i}" for i in range(n)],
        "treatment": treatment,
        "outcome": outcome,
        "baseline_bb_rating": baseline,
        "minutes_pre": minutes,
        "usage_pre": usage,
        "age_pre": age,
        "team_strength_pre": strength,
        "transition_season": rng.integers(2018, 2026, n),
        "role_pre": rng.choice(["G", "F", "C"], n),
    })


def test_team_switch_att_requires_adequate_support():
    result = estimate_team_switch_att(_panel(25))
    assert result["available"] is False
    assert result["effect_bb_rating_points"] is None
    assert result["identification"] == "unavailable"


def test_team_switch_att_returns_cohort_effect_only_with_support_and_balance():
    result = estimate_team_switch_att(_panel())
    assert result["estimand"] == "average_treatment_effect_on_treated_team_switchers"
    assert result["identification"] in {"observational_assumptions_required", "unavailable"}
    if result["available"]:
        assert result["effect_bb_rating_points"] is not None
        assert result["confidence_interval_95"][0] <= result["effect_bb_rating_points"] <= result["confidence_interval_95"][1]
        assert "No unmeasured confounding" in result["assumptions"][0]
        assert "not an individual player's causal effect" in result["limitations"][0]


def test_team_switch_att_rejects_missing_covariates():
    result = estimate_team_switch_att(pd.DataFrame({"treatment": [0, 1], "outcome": [50, 60]}))
    assert result["available"] is False
    assert result["reason"].startswith("causal_panel_missing_columns")


def test_teammate_exposure_att_requires_explicit_panel_contract():
    panel = _panel(80)
    panel["teammate_global_id"] = ["TEAMMATE-X"] * len(panel)
    panel["league_key"] = "EL"
    panel["competition"] = "RS"
    result = estimate_teammate_exposure_att(panel, teammate_global_id="TEAMMATE-X")
    assert result["estimand"] == "observational_teammate_exposure_ATT"
    assert result["teammate_global_id"] == "TEAMMATE-X"
    assert result["identification"] in {"observational_assumptions_required", "unavailable"}
    assert any("not the named focal player's individual causal effect" in item for item in result["limitations"])


def test_teammate_exposure_att_rejects_mixed_context_and_missing_teammate():
    panel = _panel(80)
    panel["teammate_global_id"] = ["TEAMMATE-X"] * len(panel)
    panel["league_key"] = "EL"
    panel["competition"] = "RS"
    mixed = panel.copy()
    mixed.loc[mixed.index[-1], "competition"] = "PO"
    result = estimate_teammate_exposure_att(mixed, teammate_global_id="TEAMMATE-X")
    assert result["available"] is False
    assert result["reason"] == "teammate_panel_mixes_league_or_competition"
    missing = estimate_teammate_exposure_att(panel, teammate_global_id="NO-SUCH-TEAMMATE")
    assert missing["available"] is False
    assert missing["reason"] == "named_teammate_not_in_panel"


def test_persist_causal_panel_rejects_native_rating_and_noncanonical_outcomes():
    panel = _panel(40)
    panel["league_key"] = "EL"
    panel["competition"] = "RS"
    panel["rating"] = 7.2
    rejected = persist_causal_panel(
        panel, None, source_type="team_switch", league_key="EL", competition="RS"
    )
    assert rejected["status"] == "not_written"
    assert rejected["reason"] == "native_rating_columns_forbidden_use_canonical_bb_rating"

    panel = _panel(40)
    panel["league_key"] = "EL"
    panel["competition"] = "RS"
    panel["outcome"] = 0.72
    rejected = persist_causal_panel(
        panel, None, source_type="team_switch", league_key="EL", competition="RS"
    )
    assert rejected["status"] == "not_written"
    assert rejected["reason"] == "canonical_bb_rating_outcome_required"



def test_team_switch_overlap_failure_reports_arm_specific_propensity_diagnostics():
    rng = np.random.default_rng(20261023)
    n_per_arm = 80
    panel = _panel(n_per_arm * 2, seed=2026)
    panel["treatment"] = np.r_[
        np.ones(n_per_arm, dtype=int),
        np.zeros(n_per_arm, dtype=int),
    ]
    # Make treated and control cohorts strongly separated on several pre-treatment
    # covariates so the fitted propensity model has insufficient common support.
    panel["baseline_bb_rating"] = np.r_[
        rng.normal(90, 0.8, n_per_arm), rng.normal(30, 0.8, n_per_arm)
    ]
    panel["minutes_pre"] = np.r_[
        rng.normal(36, 0.6, n_per_arm), rng.normal(7, 0.6, n_per_arm)
    ]
    panel["usage_pre"] = np.r_[
        rng.normal(0.38, 0.008, n_per_arm), rng.normal(0.07, 0.008, n_per_arm)
    ]
    panel["age_pre"] = np.r_[
        rng.normal(35, 0.5, n_per_arm), rng.normal(19, 0.5, n_per_arm)
    ]
    panel["team_strength_pre"] = np.r_[
        rng.normal(4, 0.12, n_per_arm), rng.normal(-4, 0.12, n_per_arm)
    ]
    panel["role_pre"] = ["G"] * n_per_arm + ["C"] * n_per_arm
    panel["outcome"] = rng.normal(60, 5, n_per_arm * 2)

    result = estimate_team_switch_att(panel)

    assert result["available"] is False
    assert result["reason"] == "propensity_overlap_failed"
    overlap = result["diagnostics"]["propensity_overlap"]
    assert overlap["required_overlap_share"] == 0.80
    assert overlap["lower_bound"] == 0.05
    assert overlap["upper_bound"] == 0.95
    assert overlap["rows_in_overlap"] < 0.80 * len(panel)
    assert overlap["treated"]["rows"] == n_per_arm
    assert overlap["control"]["rows"] == n_per_arm
    assert {"p05", "p50", "p95"}.issubset(overlap["treated"]["quantiles"])
    assert {"p05", "p50", "p95"}.issubset(overlap["control"]["quantiles"])
    assert "overlap_share_below_0.80" in overlap["failure_reasons"]
    assert "treated_overlap_below_20" in overlap["failure_reasons"]
    assert "control_overlap_below_20" in overlap["failure_reasons"]

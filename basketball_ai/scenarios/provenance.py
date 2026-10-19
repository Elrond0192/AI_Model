"""Stable provenance contract for scenario answers.

This metadata describes what kind of claim a result supports. It deliberately
does not promote a heuristic or a small counterfactual into a validated
prediction simply because a numeric result exists.
"""
from __future__ import annotations

from typing import Any

_OBSERVED = {
    "player_competition", "team_competition", "player_trend", "team_trend",
    "player_compare", "team_compare", "player_pair", "metric_explanation",
    "role_analysis", "performance_decomposition", "performance_stability",
    "team_usage_analysis", "shooting_decomposition", "defensive_decomposition",
    "model_evidence",
}
_HYPOTHETICAL = {
    "player_team", "league_transfer", "team_add_player", "team_replace_player",
    "player_role_change", "role_minutes_projection", "style_change", "lineup_fit",
    "lineup_synergy", "lineup_optimizer", "roster_optimizer", "composite_scenario",
    "shot_profile_counterfactual", "causal_effect", "causal_team_effect",
    "opponent_matchup", "defensive_matchup", "play_type_matchup",
}
_FORWARD_QUESTION_KEYS = {
    "potential", "prediction", "bb_rating_prediction", "future_performance",
}
_HISTORICAL_QUESTION_KEYS = {
    "history", "trend", "change_vs_last_season", "season_comparison",
}


def build_scenario_provenance(
    scenario: str,
    spec: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a conservative, machine-readable answer provenance summary."""
    scenario = str(scenario or "").strip().lower()
    result = payload.get("result", {})
    result = result if isinstance(result, dict) else {}
    support = payload.get("support", {})
    support = support if isinstance(support, dict) else {}
    evidence = payload.get("evidence", [])
    evidence = evidence if isinstance(evidence, list) else []
    parameters = spec.get("parameters", {})
    parameters = parameters if isinstance(parameters, dict) else {}

    question_keys = result.get("question_keys", [])
    if not isinstance(question_keys, list):
        question_keys = [question_keys]
    keys = {str(key).strip().lower() for key in question_keys if str(key).strip()}

    if scenario in _HYPOTHETICAL:
        answer_type = "hypothetical"
        temporal_scope = "hypothetical"
    elif scenario == "player_intelligence" and keys & _FORWARD_QUESTION_KEYS:
        answer_type = "mixed" if keys & (_HISTORICAL_QUESTION_KEYS | {"current_level", "performance"}) else "forecast"
        temporal_scope = "mixed" if answer_type == "mixed" else "future"
    elif scenario in {"age_trajectory", "probabilistic_boxscore", "regression_risk"} or keys & _FORWARD_QUESTION_KEYS:
        answer_type = "forecast"
        temporal_scope = "future"
    elif scenario in _OBSERVED or scenario == "player_intelligence":
        answer_type = "observed_analysis"
        temporal_scope = "historical" if keys & _HISTORICAL_QUESTION_KEYS else "observed"
    else:
        # Unknown/composite scenarios are not presumed to be predictions.
        answer_type = "analysis"
        temporal_scope = "scenario"

    source_season = result.get("source_season", parameters.get("player_season"))
    target_season = result.get("target_season", result.get("scenario_season", parameters.get("target_season")))
    context_season = result.get("latest_season", spec.get("season"))
    if source_season is not None:
        try:
            source_season = int(source_season)
        except (TypeError, ValueError):
            source_season = None
    if target_season is not None:
        try:
            target_season = int(target_season)
        except (TypeError, ValueError):
            target_season = None
    if context_season is not None:
        try:
            context_season = int(context_season)
        except (TypeError, ValueError):
            context_season = None

    impact_status = str(result.get("acquisition_impact_status", "") or "").strip().lower()
    if impact_status:
        validation_status = impact_status
    elif "walk_forward_validated" in str(support.get("method", "")).lower() or "validated" in str(support.get("method", "")).lower():
        validation_status = "walk_forward_validated"
    elif str(support.get("method", "")).lower() == "observed_data":
        validation_status = "not_applicable"
    elif support.get("method"):
        validation_status = "not_reported"
    else:
        validation_status = "not_assessed"

    sample_count = support.get("samples")
    try:
        sample_count = max(0, int(sample_count)) if sample_count is not None else None
    except (TypeError, ValueError):
        sample_count = None

    confidence = str(support.get("confidence", "") or "").strip().lower() or "not_assessed"
    method = str(support.get("method", "") or "").strip() or "not_reported"

    explicit_inputs: dict[str, Any] = {}
    for field in ("player_overrides", "style_overrides"):
        raw = spec.get(field)
        if isinstance(raw, dict) and raw:
            # Inputs are included only when supplied explicitly by the caller.
            explicit_inputs[field] = {
                str(key): value
                for key, value in raw.items()
                if str(key) in {
                    "minutes_per_game", "usg_pct", "role", "pace",
                    "three_point_attempt_rate", "assists_per_game",
                    "star_player_usage", "offensive_rating", "defensive_rating",
                }
                and isinstance(value, (str, int, float, bool))
            }

    impact_model = result.get("impact_model", {})
    impact_model = impact_model if isinstance(impact_model, dict) else {}
    validation = impact_model.get("validation", {})
    validation = validation if isinstance(validation, dict) else {}
    validation_scope = (
        impact_model.get("validation_scope")
        or validation.get("scope")
        or support.get("validation_scope")
        or {}
    )
    if not isinstance(validation_scope, dict):
        validation_scope = {}

    # A qualitative tier is not a calibrated probability. Transfer impact is
    # observational even when its walk-forward score passes validation.
    not_modeled: list[str] = []
    if scenario in {"team_add_player", "team_replace_player"}:
        not_modeled = [
            "Player availability and injuries are not inferred without reliable input data.",
            "Contracts, salary, locker-room fit and guaranteed rotation decisions are not modeled.",
            "Acquisition impact uses a historical roster-quality/turnover proxy; explicit role, minutes and usage assumptions affect the player projection but are not individually causal effects in the team outcome model.",
        ]

    return {
        "scenario": scenario,
        "answer_type": answer_type,
        "temporal_scope": temporal_scope,
        "context_season": context_season,
        "source_season": source_season,
        "target_season": target_season,
        "source_league": spec.get("source_league"),
        "target_league": spec.get("target_league"),
        "competition": spec.get("competition"),
        "target_competition": spec.get("target_competition"),
        "method": method,
        "sample_count": sample_count,
        "evidence_count": len(evidence),
        "confidence": confidence,
        "confidence_is_probability": False,
        "validation_status": validation_status,
        "validation_scope": validation_scope,
        "explicit_inputs": explicit_inputs,
        "not_modeled": not_modeled,
    }

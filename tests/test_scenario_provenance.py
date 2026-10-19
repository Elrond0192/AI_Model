from basketball_ai.scenarios.provenance import build_scenario_provenance


def test_transfer_is_marked_hypothetical_and_keeps_source_target_seasons():
    provenance = build_scenario_provenance(
        "team_add_player",
        {
            "scenario": "team_add_player",
            "season": 2025,
            "source_league": "ITA1",
            "target_league": "EL",
            "competition": "RS",
            "target_competition": "RS",
            "parameters": {"player_season": 2025, "target_season": 2026},
        },
        {
            "result": {
                "source_season": 2025,
                "target_season": 2026,
                "acquisition_impact_status": "uncalibrated",
            },
            "evidence": [{"type": "historical_roster_transitions", "count": 12}],
            "support": {"method": "acquisition_impact_not_validated", "samples": 12, "confidence": "low"},
            "limitations": ["Not causally identified"],
        },
    )
    assert provenance["answer_type"] == "hypothetical"
    assert provenance["temporal_scope"] == "hypothetical"
    assert provenance["source_season"] == 2025
    assert provenance["target_season"] == 2026
    assert provenance["validation_status"] == "uncalibrated"
    assert provenance["confidence"] == "low"


def test_player_intelligence_forecast_is_marked_future_not_observed_rating():
    provenance = build_scenario_provenance(
        "player_intelligence",
        {"scenario": "player_intelligence", "season": 2025, "parameters": {}},
        {
            "result": {"question_keys": ["bb_rating_prediction"]},
            "evidence": [{"type": "prediction"}],
            "support": {"method": "direct_bb_rating_prediction", "samples": 120, "confidence": "medium"},
            "limitations": [],
        },
    )
    assert provenance["answer_type"] == "forecast"
    assert provenance["temporal_scope"] == "future"
    assert provenance["validation_status"] == "not_reported"


def test_observed_analysis_does_not_claim_validation_for_sample_count_alone():
    provenance = build_scenario_provenance(
        "player_competition",
        {"scenario": "player_competition", "season": 2024, "source_league": "EL"},
        {
            "result": {"latest_season": 2024},
            "evidence": [{"type": "historical_seasons", "count": 3}],
            "support": {"method": "observed_data", "samples": 3, "confidence": "low"},
            "limitations": [],
        },
    )
    assert provenance["answer_type"] == "observed_analysis"
    assert provenance["context_season"] == 2024
    assert provenance["validation_status"] == "not_applicable"
    assert provenance["sample_count"] == 3


def test_unvalidated_method_name_never_becomes_a_validation_pass():
    provenance = build_scenario_provenance(
        "team_add_player",
        {
            "scenario": "team_add_player",
            "season": 2025,
            "parameters": {"player_season": 2024, "team_context_season": 2025, "target_season": 2026},
        },
        {
            "result": {
                "source_season": 2024,
                "team_context_season": 2025,
                "target_season": 2026,
                "acquisition_impact_status": "uncalibrated",
                "applied_player_context": {
                    "minutes_per_game": 28.0,
                    "usg_pct": 0.24,
                    "role": "starter",
                    "unapproved_internal": "must not pass",
                },
            },
            "evidence": [{"type": "historical_roster_transitions", "count": 12}],
            "support": {
                "method": "acquisition_impact_not_validated",
                "samples": 12,
                "confidence": "low",
            },
            "limitations": [],
        },
    )

    assert provenance["validation_status"] == "uncalibrated"
    assert provenance["confidence_is_probability"] is False
    assert provenance["applied_inputs"] == {
        "minutes_per_game": 28.0,
        "usg_pct": 0.24,
        "role": "starter",
    }
    assert "unapproved_internal" not in provenance["applied_inputs"]


def test_validation_status_is_not_inferred_from_ambiguous_method_names():
    provenance = build_scenario_provenance(
        "player_team",
        {"scenario": "player_team", "season": 2024, "parameters": {}},
        {
            "result": {"source_season": 2023, "target_season": 2025},
            "evidence": [{"type": "historical_counterfactual"}],
            "support": {"method": "strict_supervised_historical_counterfactual", "samples": 6, "confidence": "medium"},
            "limitations": [],
        },
    )

    assert provenance["validation_status"] == "not_reported"
    assert provenance["confidence_is_probability"] is False

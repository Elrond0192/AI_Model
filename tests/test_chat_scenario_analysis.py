"""Coverage and behavior tests for descriptive/comparative Chat V3 scenarios."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from basketball_ai.api.scenario_contracts_v2 import ScenarioRequestV2
from basketball_ai.data.nationality_codes import nationality_matches, normalize_nationality
from basketball_ai.scenarios.scouting_archetypes import ARCHETYPE_METRICS
from basketball_ai.scenarios.advanced_role_archetypes import map_advanced_roles
from basketball_ai.scenarios.chat_scenario_engine import (
    ChatScenarioEngine,
    _bb_rating_contribution_decomposition,
)


def _analysis_data() -> dict:
    players = pd.DataFrame([
        {"id": 1, "name": "Alpha", "position": "PG", "current_team_id": 101, "current_league_id": 10},
        {"id": 2, "name": "Beta", "position": "SG", "current_team_id": 101, "current_league_id": 10},
        {"id": 3, "name": "Gamma", "position": "SF", "current_team_id": 102, "current_league_id": 10},
        {"id": 4, "name": "Delta", "position": "C", "current_team_id": 102, "current_league_id": 10},
    ])
    teams = pd.DataFrame([
        {"id": 101, "name": "Red", "league_id": 10, "pace": 72.0, "offensive_rating": 112.0, "defensive_rating": 108.0},
        {"id": 102, "name": "Blue", "league_id": 10, "pace": 75.0, "offensive_rating": 114.0, "defensive_rating": 109.0},
    ])
    leagues = pd.DataFrame([
        {"id": 10, "name": "League A", "league_key": "A", "competitiveness_score": .75},
    ])
    rows = []
    for year in (2023, 2024):
        for pid, team_id, rating, usg, ts, ast, three, dbpm, bpm in (
            (1, 101, 7.0 + .2 * (year - 2023), .24, .58, 5.0, .37, .5, 3.0),
            (2, 101, 6.8 + .1 * (year - 2023), .20, .61, 2.0, .41, 1.1, 1.5),
            (3, 102, 7.2 + .1 * (year - 2023), .22, .59, 3.0, .36, .8, 2.2),
            (4, 102, 6.9 + .1 * (year - 2023), .18, .63, 1.5, .18, 1.6, 2.0),
        ):
            rows.append({
                "player_id": pid, "team_id": team_id, "league_id": 10,
                "season": year, "competition": "RS", "rating": rating,
                "games_played": 28, "minutes_per_game": 26 + pid,
                "points": 10 + pid + (year - 2023), "rebounds": 3 + pid, "assists": ast,
                "steals": 1.0, "blocks": .4, "ts_pct": ts, "usg_pct": usg,
                "bpm": bpm, "obpm": bpm - .5, "dbpm": dbpm, "net_rtg": 2 + pid,
                "ortg": 110 + pid, "drtg": 108 - pid * .2,
                "three_point_pct": three, "three_par": .30 + pid * .03,
                "ast_pct": .15 + pid * .02, "tov_pct": .12,
                "raptor_total": bpm * .8, "lebron_total": bpm * .7,
            })
    # Separate playoff context for observed competition comparisons.
    rows.extend([
        {**rows[-8], "season": 2024, "competition": "PO", "rating": 7.6, "games_played": 9, "points": 17, "usg_pct": .28},
        {**rows[-7], "season": 2024, "competition": "PO", "rating": 6.7, "games_played": 9, "points": 12, "usg_pct": .20},
    ])
    team_history = pd.DataFrame([
        {"team_id": 101, "league_id": 10, "season": 2023, "competition": "RS", "pace": 71.0, "offensive_rating": 110.0, "defensive_rating": 109.0, "net_rtg": 1.0, "three_point_attempt_rate": .34, "assists_per_game": 20.0, "star_player_usage": .27},
        {"team_id": 101, "league_id": 10, "season": 2024, "competition": "RS", "pace": 72.0, "offensive_rating": 112.0, "defensive_rating": 108.0, "net_rtg": 4.0, "three_point_attempt_rate": .36, "assists_per_game": 21.0, "star_player_usage": .28},
        {"team_id": 101, "league_id": 10, "season": 2024, "competition": "PO", "pace": 69.0, "offensive_rating": 109.0, "defensive_rating": 106.0, "net_rtg": 3.0, "three_point_attempt_rate": .33, "assists_per_game": 19.0, "star_player_usage": .30},
        {"team_id": 102, "league_id": 10, "season": 2023, "competition": "RS", "pace": 74.0, "offensive_rating": 113.0, "defensive_rating": 110.0, "net_rtg": 3.0, "three_point_attempt_rate": .39, "assists_per_game": 22.0, "star_player_usage": .26},
        {"team_id": 102, "league_id": 10, "season": 2024, "competition": "RS", "pace": 75.0, "offensive_rating": 114.0, "defensive_rating": 109.0, "net_rtg": 5.0, "three_point_attempt_rate": .40, "assists_per_game": 23.0, "star_player_usage": .27},
    ])
    return {
        "player_stats": pd.DataFrame(rows),
        "team_season_stats": team_history,
        "players": players,
        "teams": teams,
        "leagues": leagues,
        "team_player_relations": pd.DataFrame(columns=["player_id", "team_id", "season", "role"]),
        "player_dict": {int(row.id): row.to_dict() for _, row in players.iterrows()},
        "team_dict": {int(row.id): row.to_dict() for _, row in teams.iterrows()},
        "league_dict": {10: leagues.iloc[0].to_dict()},
        "league_teams": {10: [101, 102]},
    }


def _engine() -> ChatScenarioEngine:
    data = _analysis_data()
    for player_id, player in data["player_dict"].items():
        player["global_id"] = f"PLAYER-{player_id}"

    class _FixtureBBRating:
        def rate_player(self, global_id, *, league, season, phase):
            player_id = int(str(global_id).rsplit("-", 1)[1])
            league_id = next(
                key for key, value in data["league_dict"].items()
                if str(value.get("league_key", "")).upper() == league
            )
            rows = data["player_stats"]
            rows = rows[
                (rows["player_id"] == player_id)
                & (rows["league_id"] == league_id)
                & (rows["season"].astype(int) == int(season))
                & (rows["competition"].astype(str).str.upper() == phase)
            ]
            if rows.empty:
                raise ValueError("fixture BB-Rating context unavailable")
            score = int(round(float(rows.iloc[-1]["rating"]) * 10.0))
            return SimpleNamespace(score=score, quality="good", metric_coverage=1.0)

    return ChatScenarioEngine(
        SimpleNamespace(), data, bb_rating_engine=_FixtureBBRating()
    )


def test_bb_prediction_model_uses_direct_official_rating_and_hides_native_rating():
    data = _analysis_data()
    data["player_dict"][1]["global_id"] = "PLAYER-ALPHA"

    class HistoricalBBRating:
        def rate_player(self, global_id, *, league, season, phase):
            assert global_id == "PLAYER-ALPHA"
            assert league == "A"
            return SimpleNamespace(score={2023: 61, 2024: 65}[season])

    class DirectBBPrediction:
        def predict_from_data(self, source_data, **kwargs):
            assert kwargs["player_global_id"] == "PLAYER-ALPHA"
            assert kwargs["source_season"] == 2024
            assert kwargs["target_league"] == "A"
            assert kwargs["target_competition"] == "RS"
            return {
                "source_bb_rating": 65.0,
                "predicted_bb_rating": 72.8,
                "confidence_low_100": 68.3,
                "confidence_high_100": 77.2,
                "target_season": 2025,
                "scenario": "same_context",
                "calibration_level": "scenario",
                "calibration_samples": 120,
            }

    engine = ChatScenarioEngine(
        SimpleNamespace(),
        data,
        bb_rating_engine=HistoricalBBRating(),
        bb_rating_prediction_model=DirectBBPrediction(),
    )
    result = engine.evaluate(
        {
            "scenario": "model_evidence",
            "season": 2024,
            "competition": "RS",
            "parameters": {"question_key": "potential"},
        },
        [1],
        [],
        10,
        None,
    )

    prediction = result["bb_rating_prediction"]
    assert prediction["available"] is True
    assert prediction["target_season"] == 2025
    assert prediction["predicted_bb_rating"] == pytest.approx(72.8)
    assert prediction["source_bb_rating"] == pytest.approx(65.0)
    assert prediction["confidence_low_100"] == pytest.approx(68.3)
    assert prediction["confidence_high_100"] == pytest.approx(77.2)
    assert "BB-Rating contestuale" in prediction["interpretation"]
    assert "current_bb_rating" not in prediction
    assert "delta_vs_current_bb_rating" not in prediction
    assert result["bb_rating"]["score"] == 65
    assert "predicted_rating" not in prediction
    assert "confidence_low" not in prediction
    assert "confidence_high" not in prediction
    assert "base_rating" not in prediction



def test_public_scenario_payload_suppresses_raw_rating_and_preserves_canonical_scores():
    from basketball_ai.api.routes.scenarios_v2 import _sanitize_public_scenario_payload

    payload = {
        "result": {
            "bb_rating": {"available": True, "score": 65},
            "what_if": {
                "projected_rating": 6.9,
                "empirical_transfer_delta": -0.4,
                "league_prior_delta": -0.2,
                "team_fit_delta": 0.1,
                "team_fit": {"compatibility": 0.92},
            },
            "season_comparison": {
                "previous_rating": 7.2,
                "current_rating": 6.8,
                "rating_delta": -0.4,
                "previous_bb_rating": 71,
                "current_bb_rating": 65,
                "bb_rating_delta": -6,
            },
            "metric_explanation": {
                "metrics": [
                    {"metric": "rating", "value": 6.8},
                    {"metric": "bb_rating", "value": 65, "scale": "1-100"},
                ],
            },
            "bb_rating_prediction": {
                "target_season": 2027,
                "predicted_rating_delta": 0.8,
                "outcome": {"label": "meaningful_positive_change"},
                "predicted_rating": 7.25,
                "confidence_low": 6.8,
                "confidence_high": 7.7,
                "predicted_bb_rating": 72.8,
                "confidence_low_100": 68.3,
                "confidence_high_100": 77.2,
            },
        },
    }

    clean = _sanitize_public_scenario_payload(payload)
    encoded = str(clean)
    assert clean["result"]["bb_rating"]["score"] == 65
    for forbidden_key in ("projected_rating", "empirical_transfer_delta", "league_prior_delta", "team_fit_delta"):
        assert forbidden_key not in clean["result"]["what_if"]
    comparison = clean["result"]["season_comparison"]
    assert comparison["previous_bb_rating"] == 71
    assert comparison["current_bb_rating"] == 65
    assert comparison["bb_rating_delta"] == -6
    assert [row["metric"] for row in clean["result"]["metric_explanation"]["metrics"]] == ["bb_rating"]
    forecast = clean["result"]["bb_rating_prediction"]
    assert "predicted_rating_delta" not in forecast
    assert "outcome" not in forecast
    assert forecast["predicted_bb_rating"] == 72.8
    assert forecast["confidence_low_100"] == 68.3
    assert forecast["confidence_high_100"] == 77.2
    assert "target_season" in forecast
    for forbidden_key in (
        "previous_rating", "current_rating", "rating_delta", "predicted_rating",
        "confidence_low", "confidence_high",
    ):
        assert f"'{forbidden_key}':" not in encoded
    assert "'metric': 'rating'" not in encoded


def test_team_competition_compares_playoffs_to_regular_season():
    result = _engine().evaluate(
        {"scenario": "team_competition", "season": 2024, "competition": "PO", "comparison_competition": "RS"},
        [], [101], 10, None,
    )
    assert result["result"]["latest"]["pace"] == 69.0
    assert result["result"]["comparison"]["pace"] == 72.0
    assert result["result"]["deltas"]["pace"] == -3.0
    assert result["support"]["method"] == "observed_data"


def test_player_and_team_trends_return_ordered_series():
    engine = _engine()
    player = engine.evaluate(
        {"scenario": "player_trend", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    team = engine.evaluate(
        {"scenario": "team_trend", "season": 2024, "competition": "RS"},
        [], [101], 10, None,
    )
    assert [row["season"] for row in player["result"]["series"]] == [2023, 2024]
    assert player["result"]["rating_change"] == pytest.approx(.2)
    assert player["result"]["peak_rating"] == pytest.approx(7.2)
    assert [row["season"] for row in team["result"]["series"]] == [2023, 2024]


def test_rich_player_and_team_compare_stays_in_context():
    engine = _engine()
    players = engine.evaluate(
        {"scenario": "player_compare", "season": 2024, "competition": "RS"},
        [1, 2, 3], [], 10, None,
    )
    teams = engine.evaluate(
        {"scenario": "team_compare", "season": 2024, "competition": "RS"},
        [], [101, 102], 10, None,
    )
    assert [row["player"] for row in players["result"]["players"]] == ["Alpha", "Beta", "Gamma"]
    assert players["result"]["players"][0]["metrics"]["rating"] == pytest.approx(7.2)
    assert [row["team"] for row in teams["result"]["teams"]] == ["Red", "Blue"]
    assert teams["result"]["teams"][1]["metrics"]["net_rtg"] == 5.0


def test_lineup_fit_builds_all_pairwise_combinations():
    result = _engine().evaluate(
        {"scenario": "lineup_fit", "season": 2024, "competition": "RS"},
        [1, 2, 3, 4], [101], 10, None,
    )
    assert len(result["result"]["pairwise"]) == 6
    assert 0.0 <= result["result"]["overall_fit"] <= 1.0
    assert result["evidence"][0]["count"] == 6
    assert result["support"]["method"] == "analytical_lineup_complementarity"


def test_player_similarity_uses_standardised_same_context_pool():
    result = _engine().evaluate(
        {"scenario": "player_similarity", "season": 2024, "competition": "RS", "top_n": 2},
        [1], [], 10, None,
    )
    similar = result["result"]["similar_players"]
    assert len(similar) == 2
    assert all(row["player"] != "Alpha" for row in similar)
    assert similar[0]["distance"] <= similar[1]["distance"]
    assert result["support"]["samples"] == 3


def test_distribution_step_uses_same_context_and_fallback():
    engine = _engine()
    pace_step = engine._distribution_step("pace", 10, "RS", 2024)
    missing_step = engine._distribution_step("star_player_usage", 99, "RS", 2024)
    assert pace_step > 0
    assert missing_step == pytest.approx(.04)


def test_contract_rejects_unknown_scenario_and_bad_nested_keys():
    with pytest.raises(ValidationError):
        ScenarioRequestV2(scenario="invented", season=2025)
    with pytest.raises(ValidationError):
        ScenarioRequestV2(
            scenario="style_change",
            season=2025,
            style_overrides={"secret_metric": "higher"},
        )
    with pytest.raises(ValidationError):
        ScenarioRequestV2(
            scenario="style_change",
            season=2025,
            style_overrides={"pace": "extreme"},
        )
    with pytest.raises(ValidationError):
        ScenarioRequestV2(
            scenario="player_role_change",
            season=2025,
            player_overrides={"salary": 10},
        )


def test_scenarios_fail_explicitly_when_required_entities_are_missing():
    engine = _engine()
    with pytest.raises(ValueError, match="player is required"):
        engine.evaluate(
            {"scenario": "player_trend", "season": 2024, "competition": "RS"},
            [], [], 10, None,
        )
    with pytest.raises(ValueError, match="team is required"):
        engine.evaluate(
            {"scenario": "team_trend", "season": 2024, "competition": "RS"},
            [], [], 10, None,
        )



def test_player_intelligence_analysis_layers():
    engine = _engine()
    for scenario in (
        "metric_explanation",
        "role_analysis",
        "performance_stability",
        "team_usage_analysis",
        "regression_risk",
        "potential_synthesis",
        "shooting_decomposition",
        "defensive_decomposition",
    ):
        players = [1]
        teams = [101] if scenario == "team_usage_analysis" else []
        result = engine.evaluate(
            {"scenario": scenario, "season": 2024, "competition": "RS"},
            players, teams, 10, None,
        )
        assert result["result"]["player"] == "Alpha"
        assert result["support"]["method"]


def test_metric_explanation_uses_same_context_percentiles():
    data = _analysis_data()
    data["player_dict"][1]["global_id"] = "PLAYER-1"
    extra_row = data["player_stats"][
        (data["player_stats"]["player_id"] == 4)
        & (data["player_stats"]["season"] == 2024)
        & (data["player_stats"]["competition"] == "RS")
    ].iloc[0].to_dict()
    extra_row.update({"player_id": 5, "rating": 7.4, "ts_pct": .66, "usg_pct": .19})
    data["player_stats"] = pd.concat(
        [data["player_stats"], pd.DataFrame([extra_row])],
        ignore_index=True,
    )
    engine = _engine()
    engine.data = data
    engine.advanced = engine.advanced.__class__(data)
    result = engine.evaluate(
        {"scenario": "metric_explanation", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    metrics = {row["metric"]: row for row in result["result"]["metrics"]}
    assert "rating" not in metrics
    assert metrics["bb_rating"]["value"] == 72
    assert metrics["ts_pct"]["context_percentile"] is not None
    assert metrics["usg_pct"]["meaning"] == "Share of team possessions used"


def test_role_analysis_tracks_current_role_and_history():
    result = _engine().evaluate(
        {"scenario": "role_analysis", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    assert result["result"]["current_role"] == "scoring_role"
    assert len(result["result"]["history"]) == 2


def test_regression_risk_is_descriptive_not_probability():
    result = _engine().evaluate(
        {"scenario": "regression_risk", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    assert 0.0 <= result["result"]["risk_index"] <= 1.0
    assert "probabilità calibrata" in result["limitations"][0]


def test_player_intelligence_orchestrates_question_layers():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_key": "why_performing"}},
        [1], [], 10, None,
    )
    assert result["result"]["question_key"] == "why_performing"
    assert set(result["result"]["analyses"]) == {"performance_decomposition", "metric_explanation", "model_evidence"}
    assert result["result"]["answer_mode"] == "contextual_synthesis"


def test_player_intelligence_composes_performance_attribution_bundle():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_key": "performance_attribution"}},
        [1], [], 10, None,
    )
    analyses = result["result"]["analyses"]
    assert set(analyses) == {
        "performance_decomposition",
        "role_analysis",
        "team_usage_analysis",
        "causal_team_effect",
        "metric_explanation",
        "model_evidence",
    }
    assert analyses["performance_decomposition"]["result"]["previous_season"] == 2023
    assert analyses["team_usage_analysis"]["result"]["team"] == "Red"
    assert analyses["causal_team_effect"]["result"]["causal_identification"] == "unavailable"
    assert analyses["causal_team_effect"]["result"]["causal_effect"] is None


def test_player_intelligence_composes_multiple_question_keys():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_keys": ["real_improvement", "current_role", "stability"]}},
        [1], [], 10, None,
    )
    assert result["result"]["question_key"] is None
    assert result["result"]["question_keys"] == ["real_improvement", "current_role", "stability"]
    assert set(result["result"]["analyses"]) == {
        "performance_decomposition",
        "metric_explanation",
        "role_analysis",
        "performance_stability",
        "model_evidence",
    }


def test_causal_team_effect_uses_canonical_bb_rating_and_controls_role_minutes_usage():
    data = _analysis_data()
    for player_id, player in data["player_dict"].items():
        player["global_id"] = f"PLAYER-{player_id}"
    # A team change and falling BB-Rating are deliberate, while native source
    # rating follows a different trend. The analysis must use only canonical
    # BB-Rating returned by the injected engine.
    data["player_stats"] = pd.concat([
        data["player_stats"],
        pd.DataFrame([
            {
                "player_id": 1, "team_id": 101, "league_id": 10, "season": 2020,
                "competition": "RS", "rating": 9.2, "games_played": 25,
                "minutes_per_game": 28, "points": 12, "assists": 4,
                "ts_pct": .57, "usg_pct": .20, "bpm": 2.0, "dbpm": .3,
            },
            {
                "player_id": 1, "team_id": 101, "league_id": 10, "season": 2021,
                "competition": "RS", "rating": 9.4, "games_played": 26,
                "minutes_per_game": 29, "points": 13, "assists": 4.2,
                "ts_pct": .58, "usg_pct": .21, "bpm": 2.1, "dbpm": .3,
            },
            {
                "player_id": 1, "team_id": 101, "league_id": 10, "season": 2022,
                "competition": "RS", "rating": 8.9, "games_played": 25,
                "minutes_per_game": 25, "points": 11, "assists": 4,
                "ts_pct": .56, "usg_pct": .22, "bpm": 2.0, "dbpm": .3,
            },
        ]),
    ], ignore_index=True)
    target_mask = (
        (data["player_stats"]["player_id"] == 1)
        & (data["player_stats"]["season"].astype(int) == 2024)
        & (data["player_stats"]["competition"].astype(str).str.upper() == "RS")
    )
    data["player_stats"].loc[target_mask, "team_id"] = 102
    data["team_season_stats"] = pd.concat([
        data["team_season_stats"],
        pd.DataFrame([
            {"team_id": 101, "league_id": 10, "season": 2020, "competition": "RS", "pace": 70.0, "offensive_rating": 108.0, "defensive_rating": 111.0, "net_rtg": -1.0, "three_point_attempt_rate": .32, "assists_per_game": 19.0},
            {"team_id": 102, "league_id": 10, "season": 2020, "competition": "RS", "pace": 73.0, "offensive_rating": 112.0, "defensive_rating": 110.0, "net_rtg": 2.0, "three_point_attempt_rate": .38, "assists_per_game": 21.0},
            {"team_id": 101, "league_id": 10, "season": 2021, "competition": "RS", "pace": 70.5, "offensive_rating": 109.0, "defensive_rating": 110.5, "net_rtg": -0.5, "three_point_attempt_rate": .33, "assists_per_game": 19.5},
            {"team_id": 102, "league_id": 10, "season": 2021, "competition": "RS", "pace": 73.5, "offensive_rating": 112.5, "defensive_rating": 109.5, "net_rtg": 2.5, "three_point_attempt_rate": .385, "assists_per_game": 21.5},
            {"team_id": 101, "league_id": 10, "season": 2022, "competition": "RS", "pace": 70.0, "offensive_rating": 108.0, "defensive_rating": 111.0, "net_rtg": -1.0, "three_point_attempt_rate": .32, "assists_per_game": 19.0},
            {"team_id": 102, "league_id": 10, "season": 2022, "competition": "RS", "pace": 73.0, "offensive_rating": 112.0, "defensive_rating": 110.0, "net_rtg": 2.0, "three_point_attempt_rate": .38, "assists_per_game": 21.0},
        ]),
    ], ignore_index=True)
    data["lineup_stints"] = pd.DataFrame([
        {
            "league_key": "A", "season": 2023, "competition": "RS",
            "team_id": 101, "opponent_team_id": 102,
            "player_ids": "1,2,3,4,5", "possessions": 150,
            "net_rtg": 6.0, "ortg": 110.0, "drtg": 104.0,
        },
        {
            "league_key": "A", "season": 2023, "competition": "RS",
            "team_id": 101, "opponent_team_id": 102,
            "player_ids": "2,3,4,5,6", "possessions": 150,
            "net_rtg": 1.0, "ortg": 106.0, "drtg": 105.0,
        },
        {
            "league_key": "A", "season": 2023, "competition": "RS",
            "team_id": 101, "opponent_team_id": 102,
            "player_ids": "1,3,4,5,6", "possessions": 150,
            "net_rtg": 6.0, "ortg": 110.0, "drtg": 104.0,
        },
        {
            "league_key": "A", "season": 2024, "competition": "RS",
            "team_id": 102, "opponent_team_id": 101,
            "player_ids": "1,2,3,4,5", "possessions": 150,
            "net_rtg": -4.0, "ortg": 103.0, "drtg": 107.0,
        },
        {
            "league_key": "A", "season": 2024, "competition": "RS",
            "team_id": 102, "opponent_team_id": 101,
            "player_ids": "2,3,4,5,6", "possessions": 150,
            "net_rtg": -1.0, "ortg": 105.0, "drtg": 106.0,
        },
        {
            "league_key": "A", "season": 2024, "competition": "RS",
            "team_id": 102, "opponent_team_id": 101,
            "player_ids": "1,3,4,5,6", "possessions": 150,
            "net_rtg": -4.0, "ortg": 103.0, "drtg": 107.0,
        },
    ])

    class CanonicalBBRating:
        def rate_player(self, global_id, *, league, season, phase):
            assert global_id == "PLAYER-1"
            assert league == "A"
            assert phase == "RS"
            return SimpleNamespace(score={2020: 80, 2021: 78, 2022: 76, 2023: 70, 2024: 63}[int(season)])

    result = ChatScenarioEngine(
        SimpleNamespace(), data, bb_rating_engine=CanonicalBBRating()
    ).evaluate(
        {"scenario": "causal_team_effect", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    payload = result["result"]
    assert payload["causal_identification"] == "unavailable"
    assert payload["causal_effect"] is None
    assert payload["association"]["bb_rating_vs_team_context_correlation"] is not None
    assert payload["association"]["adjusted"]["status"] == "insufficient_history"
    assert payload["association"]["adjusted"]["controls"] == [
        "season trend", "minutes per game", "usage", "observed role"
    ]
    assert payload["current_context"]["bb_rating"] == 63
    assert "rating" not in payload["current_context"]
    lineup = payload["lineup_context_association"]
    assert lineup["status"] == "exploratory_observational_association"
    assert lineup["outcome"] == "observed_team_net_rating"
    assert lineup["is_causal"] is False
    assert lineup["seasons_compared"] == 2
    assert lineup["player_lineup_possessions"] == 300
    assert lineup["other_lineup_possessions"] == 300
    assert lineup["net_rating_with_player"] == 1.0
    assert lineup["net_rating_without_player"] == 0.0
    assert lineup["difference"] == 1.0
    beta_association = next(
        item for item in lineup["teammate_associations"]
        if item["teammate"] == "Beta" and item["season"] == 2023
    )
    assert beta_association["status"] == "exploratory_observational_association"
    assert beta_association["possessions_with_both"] == 150
    assert beta_association["possessions_with_player_without_teammate"] == 150
    assert beta_association["difference"] == 0.0
    assert payload["team_switches"]
    switch = payload["team_switches"][-1]
    assert switch["bb_rating_change"] == -7
    assert switch["minutes_per_game_change"] is not None
    assert "rating_change" not in switch
    assert result["support"]["method"] == "within_player_observational_bb_rating_team_context"
    assert any("effetto causale" in item for item in result["limitations"])


def test_player_intelligence_composes_causal_team_effect_as_observational():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_key": "causal_team_effect"}},
        [1], [], 10, None,
    )
    assert set(result["result"]["analyses"]) == {"causal_team_effect", "model_evidence"}
    analysis = result["result"]["analyses"]["causal_team_effect"]
    assert analysis["result"]["causal_identification"] == "unavailable"
    assert analysis["result"]["causal_effect"] is None


def test_player_intelligence_rejects_unknown_question_key():
    with pytest.raises(ValueError, match="unsupported Player Intelligence question_key"):
        _engine().evaluate(
            {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
             "parameters": {"question_key": "unknown"}},
            [1], [], 10, None,
        )


def test_decomposition_uses_canonical_bb_rating_and_rate_adjusted_production():
    result = _engine().evaluate(
        {"scenario": "performance_decomposition", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    payload = result["result"]
    assert payload["previous_season"] == 2023
    assert payload["current_season"] == 2024
    assert payload["current_season_status"] == "complete"
    assert payload["is_provisional"] is False
    assert set(payload["deltas"]) == {"volume", "efficiency", "role", "impact"}
    assert payload["deltas"]["volume"]["points"] == pytest.approx(1.0)
    # Native statistical rating must not leak through the decomposition payload.
    assert "rating" not in payload["deltas"]["impact"]
    assert "rating" not in payload["previous"]
    assert "rating" not in payload["current"]
    assert payload["previous_bb_rating"] == 70
    assert payload["current_bb_rating"] == 72
    assert payload["bb_rating_delta"] == 2
    assert payload["bb_rating_contribution_decomposition"]["available"] is False
    assert payload["rate_adjusted_production"]["unit"] == "per_36_minutes"
    assert payload["rate_adjusted_production"]["previous_per36"]["points_per36"] == pytest.approx(14.667)
    assert payload["rate_adjusted_production"]["current_per36"]["points_per36"] == pytest.approx(16.0)
    assert payload["interpretation"]["causal_attribution"] == "not_identified"


def test_bb_rating_component_decomposition_reconciles_without_causal_claim():
    previous = {
        "score": 60,
        "rating_composition": {
            "unrounded_bb_rating": 60.2,
            "metrics": [
                {"metric": "A", "display_name": "Efficienza al tiro", "dimension": "offense",
                 "contextual_percentile_pct": 55.0, "effective_weight": 0.5,
                 "score_deviation_from_midpoint_points": 6.0},
                {"metric": "B", "display_name": "Impatto difensivo", "dimension": "defense",
                 "contextual_percentile_pct": 52.0, "effective_weight": 0.5,
                 "score_deviation_from_midpoint_points": 3.7},
            ],
        },
    }
    current = {
        "score": 63,
        "rating_composition": {
            "unrounded_bb_rating": 63.1,
            "metrics": [
                {"metric": "A", "display_name": "Efficienza al tiro", "dimension": "offense",
                 "contextual_percentile_pct": 60.0, "effective_weight": 0.5,
                 "score_deviation_from_midpoint_points": 8.2},
                {"metric": "B", "display_name": "Impatto difensivo", "dimension": "defense",
                 "contextual_percentile_pct": 55.0, "effective_weight": 0.5,
                 "score_deviation_from_midpoint_points": 4.4},
            ],
        },
    }

    result = _bb_rating_contribution_decomposition(previous, current, 2023, 2024)

    assert result is not None
    assert result["available"] is True
    assert result["attribution_type"] == "mathematical_not_causal"
    assert result["bb_rating_delta"] == 3
    assert result["unrounded_bb_rating_delta_points"] == pytest.approx(2.9)
    assert result["sum_component_delta_points"] == pytest.approx(2.9)
    assert result["rounding_residual_points"] == pytest.approx(0.1)
    by_metric = {row["metric"]: row for row in result["components"]}
    assert by_metric["A"]["contribution_delta_points"] == pytest.approx(2.2)
    assert by_metric["B"]["contribution_delta_points"] == pytest.approx(0.7)
    assert result["dimensions"][0]["contribution_delta_points"] == pytest.approx(2.2)


def test_decomposition_marks_current_calendar_season_provisional():
    engine = _engine()
    data = engine.data
    current_year = int(pd.Timestamp.now().year)
    player_rows = data["player_stats"]
    target_mask = (
        (player_rows["player_id"] == 1)
        & (player_rows["league_id"] == 10)
        & (player_rows["competition"].astype(str).str.upper() == "RS")
    )
    base_rows = player_rows[target_mask].copy()
    data["player_stats"] = player_rows.loc[~target_mask].copy()
    previous_row = base_rows[base_rows["season"].astype(int) == 2023].iloc[0].to_dict()
    current_row = base_rows[base_rows["season"].astype(int) == 2024].iloc[0].to_dict()
    previous_row["season"] = current_year - 1
    current_row["season"] = current_year
    current_row["minutes_per_game"] = 35.0
    current_row["points"] = 14.0
    data["player_stats"] = pd.concat(
        [data["player_stats"], pd.DataFrame([previous_row, current_row])],
        ignore_index=True,
    )
    result = engine.evaluate(
        {"scenario": "performance_decomposition", "season": current_year, "competition": "RS"},
        [1], [], 10, None,
    )
    payload = result["result"]
    assert payload["current_season"] == current_year
    assert payload["current_season_status"] == "in_progress"
    assert payload["is_provisional"] is True



def test_player_scouting_filters_identity_and_ranks_playmakers():
    engine = _engine()
    data = engine.data
    data["player_dict"][1].update({"position": "PG", "nationality": "ITA", "birth_date": "2000-01-01"})
    data["player_dict"][2].update({"position": "SG", "nationality": "USA", "birth_date": "1998-01-01"})
    data["player_dict"][3].update({"position": "PG", "nationality": "ITA", "birth_date": "2002-01-01"})
    data["player_dict"][4].update({"position": "C", "nationality": "ITA", "birth_date": "1997-01-01"})
    result = engine.evaluate(
        {"scenario": "player_scouting", "season": 2024, "competition": "RS", "top_n": 5,
         "parameters": {"archetype": "playmaker", "role": "PG", "nationality": "ITA",
                        "age_range": [20, 25], "min_games": 10}},
        [], [], 10, None,
    )
    candidates = result["result"]["candidates"]
    assert candidates
    assert all(row["position"] == "PG" for row in candidates)
    assert all(row["nationality"] == "ITA" for row in candidates)
    assert all(20 <= row["age"] <= 25 for row in candidates)
    assert all(0 <= row["fit_score"] <= 100 for row in candidates)
    assert result["support"]["method"] == "transparent_weighted_percentile_scouting"



def test_player_scouting_scores_roster_complementarity_when_team_is_provided():
    engine = _engine()
    engine.data["team_player_relations"] = pd.DataFrame([
        {"player_id": 1, "team_id": 101, "season": 2024, "role": "PG"},
        {"player_id": 2, "team_id": 101, "season": 2024, "role": "SG"},
    ])
    result = engine.evaluate(
        {
            "scenario": "player_scouting",
            "season": 2024,
            "competition": "RS",
            "top_n": 10,
            "parameters": {"archetype": "balanced", "min_games": 10},
        },
        [],
        [101],
        10,
        None,
    )

    team_fit = result["result"]["team_fit"]
    candidates = {row["player"]: row for row in result["result"]["candidates"]}
    assert team_fit["status"] == "available"
    assert team_fit["completeness"] == "full"
    assert team_fit["team"] == "Red"
    assert team_fit["roster_source"] == "season_roster_relations_plus_observed_stats"
    assert team_fit["roster_player_count"] == 2
    assert team_fit["roster_players_with_stats"] == 2
    assert team_fit["roster_stats_coverage"]["statistics_coverage_pct"] == 100.0
    assert team_fit["roster_stats_coverage"]["complete"] is True
    assert team_fit["roster_needs"]
    assert result["result"]["ranking_basis"] == "team_fit"
    assert result["support"]["method"] == "roster_aware_scouting_fit"

    # Candidates outside the roster are compared with both current roster players;
    # existing roster members are compared with the other member, not themselves.
    assert candidates["Gamma"]["complementarity_pairs"] == 2
    assert candidates["Delta"]["complementarity_pairs"] == 2
    assert candidates["Alpha"]["complementarity_pairs"] == 1
    assert candidates["Beta"]["complementarity_pairs"] == 1
    assert all(
        row["team_fit_score"] is not None
        and 0 <= row["fit_score"] <= 100
        and row["complementarity_score"] is not None
        for row in candidates.values()
    )
    assert team_fit["complementarity_pairs_evaluated"] == 6
    assert result["evidence"][-1] == {"type": "roster_complementarity_pairs", "count": 6}
    assert result["result"]["acquisition_impact_model"]["status"] == "roster_history_unavailable"
    assert all(row["acquisition_impact"]["estimated_incremental_net_rating"] is None for row in candidates.values())



def test_player_scouting_marks_observed_stats_fallback_as_partial_fit():
    engine = _engine()
    result = engine.evaluate(
        {
            "scenario": "player_scouting",
            "season": 2024,
            "competition": "RS",
            "top_n": 10,
            "parameters": {"archetype": "balanced", "min_games": 10},
        },
        [],
        [101],
        10,
        None,
    )

    team_fit = result["result"]["team_fit"]
    assert team_fit["status"] == "partial"
    assert team_fit["completeness"] == "partial"
    assert team_fit["roster_source"] == "observed_player_competition_stats"
    assert team_fit["roster_stats_coverage"]["membership_source_is_authoritative"] is False
    assert team_fit["roster_stats_coverage"]["complete"] is False
    assert result["result"]["ranking_basis"] == "team_fit_partial"
    assert result["support"]["method"] == "partial_roster_aware_scouting_fit"
    assert all(row["team_fit_score"] is not None for row in result["result"]["candidates"])
    assert any("non va interpretata come valutazione completa" in item for item in result["limitations"])


def test_player_scouting_marks_missing_roster_statistics_as_partial_fit():
    engine = _engine()
    engine.data["team_player_relations"] = pd.DataFrame([
        {"player_id": 1, "team_id": 101, "season": 2024, "role": "PG"},
        {"player_id": 2, "team_id": 101, "season": 2024, "role": "SG"},
        {"player_id": 99, "team_id": 101, "season": 2024, "role": "C"},
    ])
    result = engine.evaluate(
        {
            "scenario": "player_scouting",
            "season": 2024,
            "competition": "RS",
            "top_n": 10,
            "parameters": {"archetype": "balanced", "min_games": 10},
        },
        [],
        [101],
        10,
        None,
    )

    team_fit = result["result"]["team_fit"]
    coverage = team_fit["roster_stats_coverage"]
    assert team_fit["status"] == "partial"
    assert team_fit["roster_player_count"] == 3
    assert team_fit["roster_players_with_stats"] == 2
    assert coverage["members_without_usable_stats"] == 1
    assert coverage["statistics_coverage_pct"] == pytest.approx(66.7)
    assert coverage["complete"] is False
    assert result["result"]["ranking_basis"] == "team_fit_partial"


def test_player_scouting_does_not_claim_team_fit_without_roster_context():
    result = _engine().evaluate(
        {
            "scenario": "player_scouting",
            "season": 2024,
            "competition": "RS",
            "top_n": 3,
            "parameters": {"archetype": "balanced", "min_games": 10},
        },
        [],
        [999],
        10,
        None,
    )

    assert result["result"]["team_fit"]["status"] == "unavailable"
    assert result["result"]["ranking_basis"] == "individual_scouting"
    assert all(row["team_fit_score"] is None for row in result["result"]["candidates"])
    assert any("Team Fit completo non è stato calcolato" in item for item in result["limitations"])

def test_player_scouting_rejects_unknown_archetype():
    with pytest.raises(ValueError, match="unsupported scouting archetype"):
        _engine().evaluate(
            {"scenario": "player_scouting", "season": 2024, "competition": "RS",
             "source_league": "10", "parameters": {"archetype": "magic"}},
            [], [], 10, None,
        )



def test_player_scouting_is_registered_in_request_contract():
    request = ScenarioRequestV2(
        scenario="player_scouting",
        source_league="ITA1",
        season=2026,
        competition="RS",
        top_n=10,
        parameters={
            "archetype": "playmaker",
            "nationality": "ITA",
            "min_age": 22,
            "max_age": 28,
            "role": "PG",
        },
    )
    assert request.scenario == "player_scouting"
    assert request.parameters["archetype"] == "playmaker"
    assert request.top_n == 10


def test_player_intelligence_finds_recent_contexts_across_adjacent_seasons():
    data = _analysis_data()
    players = data["players"].copy()
    players["global_id"] = players["id"].map({
        1: "174019", 2: "global-2", 3: "global-3", 4: "global-4"
    })
    players = pd.concat([
        players,
        pd.DataFrame([{
            "id": 5, "name": "Leandro Nicolas Bolmaro", "position": "SG",
            "current_team_id": 201, "current_league_id": 20, "global_id": "174019",
        }]),
    ], ignore_index=True)
    data["players"] = players
    data["player_dict"] = {
        int(row["id"]): row
        for row in players.to_dict("records")
    }

    base_stats = data["player_stats"]
    data["player_stats"] = pd.concat([
        base_stats,
        pd.DataFrame([
            {
                "player_id": 1, "team_id": 101, "league_id": 10,
                "season": 2026, "competition": "RS", "rating": 7.1,
                "games_played": 1, "minutes_per_game": 25.0,
                "points": 11.0, "rebounds": 5.0, "assists": 2.0,
                "global_id": "174019",
            },
            {
                "player_id": 5, "team_id": 201, "league_id": 20,
                "season": 2025, "competition": "RS", "rating": 6.8,
                "games_played": 24, "minutes_per_game": 19.0,
                "points": 6.0, "rebounds": 3.0, "assists": 2.5,
                "global_id": "174019",
            },
            {
                "player_id": 5, "team_id": 201, "league_id": 20,
                "season": 2024, "competition": "RS", "rating": 6.6,
                "games_played": 28, "minutes_per_game": 17.0,
                "points": 5.0, "rebounds": 2.5, "assists": 2.0,
                "global_id": "174019",
            },
        ]),
    ], ignore_index=True)
    engine = ChatScenarioEngine(SimpleNamespace(), data)

    contexts = engine._player_intelligence_context_leagues(
        spec={"season": 2026},
        player_id=1,
        season=2026,
        competition="RS",
        source_league=None,
    )

    # League 10 has a 2026 row; league 20's latest row is 2025. The canonical
    # identity must expose both recent contexts while excluding league 20's 2024
    # row from discovery.
    assert contexts == [10, 20]



def test_nationality_codes_are_canonicalized_and_matched_exactly():
    assert normalize_nationality("DEU") == "DE"
    assert normalize_nationality("GER") == "DE"
    assert normalize_nationality("ITA") == "ITA"
    assert nationality_matches("DE", "DEU")
    assert nationality_matches("DEU", "DE")
    assert nationality_matches("ITA", "ITA")
    assert not nationality_matches("IT", "ITA")
    assert nationality_matches("Germany", "DEU")
    assert not nationality_matches("DE", "DEN")


def test_player_scouting_exposes_extensible_multi_label_archetype_profiles():
    engine = _engine()
    result = engine.evaluate(
        {
            "scenario": "player_scouting",
            "season": 2024,
            "competition": "RS",
            "top_n": 5,
            "parameters": {"archetype": "balanced", "min_games": 0},
        },
        [], [], 10, None,
    )
    candidates = result["result"]["candidates"]
    assert len(ARCHETYPE_METRICS) >= 50
    assert {"playmaker", "scorer", "perimeter_defender", "rim_protector",
            "three_and_d", "high_impact_low_usage", "big"}.issubset(ARCHETYPE_METRICS)
    assert candidates
    for candidate in candidates:
        profiles = candidate["archetype_profiles"]
        assert profiles
        assert len(profiles) <= 8
        assert all(0 <= item["score"] <= 100 for item in profiles)
        assert all(item["confidence"] in {"low", "medium", "high"} for item in profiles)
        assert candidate["archetype_evidence"] == "league_season_competition_percentiles"
        coverage = candidate["archetype_profile_coverage"]
        assert coverage["catalog_size"] == len(ARCHETYPE_METRICS)
        assert coverage["scored_profiles"] + coverage["unscored_profile_count"] == coverage["catalog_size"]
        assert isinstance(candidate["primary_archetypes"], list)


def test_hoopmetrics_advanced_roles_map_to_primary_scouting_archetypes():
    mapped = map_advanced_roles(
        "PRIMARY BALL HANDLER",
        "PERIMETER DEFENDER",
        "ELITE 2-WAY PG",
    )
    assert mapped["offensive"] == "PRIMARY BALL HANDLER"
    assert mapped["defensive"] == "PERIMETER DEFENDER"
    assert mapped["combined"] == "ELITE 2-WAY PG"
    assert {"playmaker", "primary_creator", "perimeter_defender", "two_way"}.issubset(
        set(mapped["archetypes"])
    )
    assert mapped["source"] == "hoopmetrics_engine_advanced_roles"
    assert mapped["mapping_coverage"] == 1.0


def test_hoopmetrics_unknown_advanced_role_is_preserved_and_reported():
    mapped = map_advanced_roles("NEW FUTURE ROLE", None, None)
    assert mapped["offensive"] == "NEW FUTURE ROLE"
    assert mapped["archetypes"] == []
    assert mapped["unmapped_roles"] == [
        {"dimension": "offensive", "role": "NEW FUTURE ROLE"}
    ]

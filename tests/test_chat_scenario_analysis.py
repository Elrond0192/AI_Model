"""Coverage and behavior tests for descriptive/comparative Chat V3 scenarios."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from basketball_ai.api.scenario_contracts_v2 import ScenarioRequestV2
from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine


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
                "points": 10 + pid, "rebounds": 3 + pid, "assists": ast,
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
    return ChatScenarioEngine(SimpleNamespace(), _analysis_data())


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
    result = _engine().evaluate(
        {"scenario": "metric_explanation", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    metrics = {row["metric"]: row for row in result["result"]["metrics"]}
    assert metrics["rating"]["context_percentile"] is not None
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
    assert "not a calibrated probability" in result["limitations"][0]


def test_player_intelligence_orchestrates_question_layers():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_key": "why_performing"}},
        [1], [], 10, None,
    )
    assert result["result"]["question_key"] == "why_performing"
    assert set(result["result"]["analyses"]) == {"performance_decomposition", "metric_explanation"}
    assert result["result"]["answer_mode"] == "evidence_composition"


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
    }


def test_causal_team_effect_returns_observational_association_without_causal_claim():
    result = _engine().evaluate(
        {"scenario": "causal_team_effect", "season": 2024, "competition": "RS"},
        [1], [], 10, None,
    )
    assert result["result"]["causal_identification"] == "unavailable"
    assert result["result"]["causal_effect"] is None
    assert result["result"]["association"]["rating_vs_team_context_correlation"] is not None
    assert result["support"]["method"] == "within_player_observational_team_context"
    assert any("non un effet causale" in item for item in result["limitations"])


def test_player_intelligence_composes_causal_team_effect_as_observational():
    result = _engine().evaluate(
        {"scenario": "player_intelligence", "season": 2024, "competition": "RS",
         "parameters": {"question_key": "causal_team_effect"}},
        [1], [], 10, None,
    )
    assert set(result["result"]["analyses"]) == {"causal_team_effect"}
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


def test_decomposition():
    result = _engine().evaluate({"scenario": "performance_decomposition", "season": 2024, "competition": "RS"}, [1], [], 10, None)
    assert result["result"]["previous_season"] == 2023
    assert result["result"]["current_season"] == 2024
    assert set(result["result"]["deltas"]) == {"volume", "efficiency", "role", "impact"}
    assert result["result"]["deltas"]["volume"]["points"] == pytest.approx(1.0)
    assert result["result"]["deltas"]["impact"]["rating"] == pytest.approx(0.2)



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
                        "min_age": 20, "max_age": 25, "min_games": 10}},
        [], [], 10, None,
    )
    candidates = result["result"]["candidates"]
    assert candidates
    assert all(row["position"] == "PG" for row in candidates)
    assert all(row["nationality"] == "ITA" for row in candidates)
    assert all(20 <= row["age"] <= 25 for row in candidates)
    assert all(0 <= row["fit_score"] <= 100 for row in candidates)
    assert result["support"]["method"] == "transparent_weighted_percentile_scouting"


def test_player_scouting_rejects_unknown_archetype():
    with pytest.raises(ValueError, match="unsupported scouting archetype"):
        _engine().evaluate(
            {"scenario": "player_scouting", "season": 2024, "competition": "RS",
             "parameters": {"archetype": "magic"}},
            [], [], 10, None,
        )

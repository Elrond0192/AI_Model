"""Regression tests for simulation, defensive inference and optimization."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from basketball_ai.scenarios.chat_scenario_engine import ChatScenarioEngine
from basketball_ai.scenarios.advanced_simulation_engine import AdvancedSimulationEngine


def _data() -> dict:
    players = pd.DataFrame([
        {"id": i, "name": f"Player {i}", "position": ["PG", "SG", "SF", "PF", "C", "G"][i - 1]}
        for i in range(1, 7)
    ])
    stats = []
    for season in (2022, 2023, 2024):
        for pid in range(1, 7):
            stats.append({
                "player_id": pid, "team_id": 10 if pid <= 5 else 20,
                "league_id": 100, "season": season, "competition": "RS",
                "games_played": 28, "minutes_per_game": 22 + pid,
                "points": 8 + pid * 1.7, "rebounds": 2 + pid,
                "offensive_rebounds": pid * .3, "defensive_rebounds": 2 + pid * .7,
                "assists": 1 + pid * .8, "steals": .5 + pid * .1,
                "blocks": pid * .12, "turnovers": 1 + pid * .2,
                "personal_fouls": 2.0, "ts_pct": .54 + pid * .01,
                "usg_pct": .16 + pid * .015, "bpm": -1 + pid * .7,
                "dbpm": -.5 + pid * .25, "ortg": 105 + pid,
                "drtg": 112 - pid, "net_rtg": -7 + pid * 2,
                "rating": 6 + pid * .2, "three_point_pct": .31 + pid * .01,
                "three_par": .25 + pid * .03, "ft_pct": .75,
            })
    teams = pd.DataFrame([
        {"id": 10, "name": "Home", "league_id": 100},
        {"id": 20, "name": "Opponent", "league_id": 100},
    ])
    team_stats = pd.DataFrame([
        {"team_id": 10, "league_id": 100, "season": 2024, "competition": "RS", "pace": 72, "offensive_rating": 112, "defensive_rating": 109},
        {"team_id": 20, "league_id": 100, "season": 2024, "competition": "RS", "pace": 68, "offensive_rating": 108, "defensive_rating": 103},
    ])
    lineups = pd.DataFrame([
        {"player_ids": [1, 2, 3, 4, 5], "possessions": 180, "net_rtg": 9.0},
        {"offense_player_id": 1, "defense_player_id": 6, "possessions": 80, "assignment_probability": .72, "points_allowed": 63, "turnovers_forced": 9},
    ])
    play_types = pd.DataFrame([
        {"player_id": 1, "team_id": 10, "competition": "RS", "season": 2024, "play_type": "pick_and_roll_ball_handler", "possessions": 140, "ppp": 1.06},
        {"player_id": None, "team_id": 20, "competition": "RS", "season": 2024, "play_type": "pick_and_roll_ball_handler", "possessions": 160, "ppp_allowed": .88},
    ])
    shots = pd.DataFrame([
        {"player_id": 1, "zone": "rim", "attempts": 80, "fg_pct": .65},
        {"player_id": 1, "zone": "long_mid", "attempts": 50, "fg_pct": .39},
        {"player_id": 1, "zone": "above_break_three", "attempts": 70, "fg_pct": .36},
    ])
    return {
        "players": players, "teams": teams, "player_stats": pd.DataFrame(stats),
        "team_season_stats": team_stats, "lineup_stints": lineups,
        "play_type_stats": play_types, "shot_profiles": shots,
        "pbp_events": pd.DataFrame(), "team_player_relations": pd.DataFrame(),
        "player_dict": {int(r.id): r.to_dict() for _, r in players.iterrows()},
        "team_dict": {int(r.id): r.to_dict() for _, r in teams.iterrows()},
        "league_dict": {100: {"id": 100, "name": "League"}},
        "league_teams": {100: [10, 20]},
    }


def _evaluate(scenario: str, players: list[int], teams: list[int] | None = None, **extra):
    spec = {"scenario": scenario, "season": 2024, "competition": "RS", **extra}
    return ChatScenarioEngine(SimpleNamespace(), _data()).evaluate(
        spec, players, teams or [], 100, 100
    )


def test_probabilistic_boxscore_has_joint_quantiles_and_seeded_reproducibility():
    first = _evaluate("probabilistic_boxscore", [1], parameters={"simulations": 1000, "seed": 42, "thresholds": {"points": [10, 20]}})
    second = _evaluate("probabilistic_boxscore", [1], parameters={"simulations": 1000, "seed": 42, "thresholds": {"points": [10, 20]}})
    assert first["result"]["distribution"] == second["result"]["distribution"]
    points = first["result"]["distribution"]["points"]
    assert points["quantiles"]["p10"] < points["quantiles"]["p90"]
    assert set(points["probabilities"]) == {"gte_10", "gte_20"}


def test_opponent_and_play_type_condition_one_distribution():
    result = _evaluate("opponent_matchup", [1], [20], parameters={"simulations": 750, "seed": 7})
    assert result["result"]["opponent"] == "Opponent"
    assert result["result"]["play_type_matchup"][0]["possessions"] == 300
    assert result["support"]["method"] == "opponent_conditioned_boxscore_distribution"


def test_defensive_matchup_is_explicitly_inferred_from_weighted_exposure():
    result = _evaluate("defensive_matchup", [1, 6])
    assert result["result"]["assignment_status"] == "inferred_not_observed"
    assert result["result"]["estimated_matchup_possessions"] == 57.6
    assert result["support"]["method"] == "inferred_assignment_from_pbp_lineup_exposure"


def test_shot_counterfactual_shrinks_new_zone_accuracy():
    result = _evaluate("shot_profile_counterfactual", [1], parameters={"shot_transfers": [{"from": "long_mid", "to": "above_break_three", "fraction": .5}]})
    assert result["result"]["applied_transfers"][0]["attempts"] == 25
    assert result["result"]["scenario_profile"]["above_break_three"]["frequency"] > .35


def test_lineup_synergy_and_optimizer_use_partial_pooling():
    synergy = _evaluate("lineup_synergy", [1, 2, 3, 4, 5])
    optimized = _evaluate("lineup_optimizer", [1, 2, 3, 4, 5, 6], top_n=2)
    assert synergy["result"]["possessions"] == 180
    assert synergy["support"]["method"] == "partial_pooling_lineup_synergy"
    assert optimized["result"]["evaluated_lineups"] == 6
    assert len(optimized["result"]["lineups"]) == 2


def test_composite_scenario_returns_single_conditioned_distribution():
    result = _evaluate("composite_scenario", [1], [20], parameters={"simulations": 600, "seed": 9, "shot_transfers": [{"from": "long_mid", "to": "rim", "fraction": .2}]})
    assert "distribution" in result["result"]
    assert "opponent_matchup" in result["result"]["components"]
    assert "shot_counterfactual" in result["result"]["components"]
    assert result["support"]["method"] == "single_conditioned_composite_simulation"


def test_roster_optimizer_respects_requested_size():
    result = _evaluate("roster_optimizer", [1, 2, 3, 4, 5, 6], [10], parameters={"roster_size": 5})
    assert len(result["result"]["roster"]) == 5
    assert result["support"]["method"] == "constrained_local_search_over_lineup_synergy"


def test_causal_layer_returns_aipw_only_when_overlap_is_supported():
    data = _data()
    rows = []
    for index in range(80):
        treatment = float(index % 2)
        baseline = 8.0 + (index % 7) * .3
        rows.append({
            "treatment": treatment,
            "baseline": baseline,
            "next_outcome": baseline + treatment * 1.5 + ((index % 3) - 1) * .05,
        })
    data["causal_panel"] = pd.DataFrame(rows)
    result = AdvancedSimulationEngine(data).causal_effect(
        {"season": 2024, "parameters": {"treatment": "pace_increase", "outcome": "points", "covariates": ["baseline"]}},
        [], [], 100, 100,
    )
    assert result["result"]["identification"] == "supported"
    assert 1.3 < result["result"]["causal_effect"] < 1.7
    assert result["support"]["method"] == "doubly_robust_aipw"

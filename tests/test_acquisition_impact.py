"""Regression tests for the walk-forward acquisition impact estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from basketball_ai.scenarios.acquisition_impact import (
    build_team_transition_samples,
    fit_acquisition_impact_model,
    predict_acquisition_impact,
)


def _synthetic_roster_history() -> dict:
    """Create a deterministic panel whose future net-rating deltas use known features."""
    rng = np.random.default_rng(712)
    years = list(range(2010, 2023))
    team_ids = list(range(1, 13))
    roster_size = 8
    player_ids = list(range(1001, 1001 + len(team_ids) * roster_size))
    ability = {pid: float(rng.normal(6.4, 1.0)) for pid in player_ids}

    rosters_by_year: dict[int, dict[int, list[int]]] = {}
    rosters = {
        team_id: player_ids[(team_id - 1) * roster_size:team_id * roster_size].copy()
        for team_id in team_ids
    }
    for year in years:
        if year != years[0]:
            # Swap unequal profiles between teams to create realistic changes
            # in roster quality and turnover from one season to the next.
            pairs = [(1, 7), (2, 9), (3, 11), (4, 8), (5, 10), (6, 12)]
            rng.shuffle(pairs)
            for first_team, second_team in pairs[:4]:
                first_index = int(rng.integers(0, roster_size))
                second_index = int(rng.integers(0, roster_size))
                rosters[first_team][first_index], rosters[second_team][second_index] = (
                    rosters[second_team][second_index],
                    rosters[first_team][first_index],
                )
        rosters_by_year[year] = {team: members.copy() for team, members in rosters.items()}

    stats_rows = []
    relation_rows = []
    rating_lookup: dict[tuple[int, int], float] = {}
    player_team_lookup: dict[tuple[int, int], int] = {}
    for year in years:
        for team_id, members in rosters_by_year[year].items():
            for pid in members:
                rating = ability[pid] + float(rng.normal(0.0, 0.04))
                rating_lookup[(pid, year)] = rating
                player_team_lookup[(pid, year)] = team_id
                stats_rows.append({
                    "player_id": pid,
                    "team_id": team_id,
                    "league_id": 10,
                    "season": year,
                    "competition": "RS",
                    "rating": rating,
                    "games_played": 30,
                    "minutes_per_game": 24.0,
                })
                relation_rows.append({
                    "player_id": pid, "team_id": team_id, "season": year,
                    "league_id": 10, "role": "rotation",
                })

    net_by_team_year: dict[tuple[int, int], float] = {}
    team_rows = []
    for team_id in team_ids:
        first_net = float(rng.normal(0.0, 2.0))
        net_by_team_year[(team_id, years[0])] = first_net
        team_rows.append({
            "team_id": team_id, "league_id": 10, "season": years[0],
            "competition": "RS", "net_rtg": first_net,
        })

    for year in years[1:]:
        prior_year = year - 1
        prior_prior_year = year - 2
        for team_id in team_ids:
            new_members = set(rosters_by_year[year][team_id])
            old_members = set(rosters_by_year[prior_year][team_id])
            new_quality = np.mean([rating_lookup[(pid, prior_year)] for pid in new_members])
            old_quality = np.mean([rating_lookup[(pid, prior_year)] for pid in old_members])
            quality_delta = float(new_quality - old_quality)
            prior_net = net_by_team_year[(team_id, prior_year)]
            prior_change = (
                prior_net - net_by_team_year[(team_id, prior_prior_year)]
                if (team_id, prior_prior_year) in net_by_team_year else 0.0
            )
            # Synthetic outcomes are intentionally generated from features known
            # before the target season; the test checks validation plumbing, not
            # the real-world magnitude of basketball effects.
            delta = (
                2.4 * quality_delta
                - 0.25 * prior_net
                + 0.35 * prior_change
                + float(rng.normal(0.0, 0.05))
            )
            value = prior_net + delta
            net_by_team_year[(team_id, year)] = value
            team_rows.append({
                "team_id": team_id, "league_id": 10, "season": year,
                "competition": "RS", "net_rtg": value,
            })

    players = pd.DataFrame([
        {"id": pid, "global_id": str(900000 + pid), "name": f"Player {pid}"}
        for pid in player_ids
    ])
    return {
        "players": players,
        "player_dict": {
            int(row["id"]): {"id": int(row["id"]), "global_id": str(row["global_id"])}
            for row in players.to_dict("records")
        },
        "player_stats": pd.DataFrame(stats_rows),
        "team_player_relations": pd.DataFrame(relation_rows),
        "team_season_stats": pd.DataFrame(team_rows),
    }


def test_acquisition_model_builds_prior_season_only_transition_features():
    data = _synthetic_roster_history()
    rows = build_team_transition_samples(data, league_id=10, competition="RS", through_season=2022)

    assert rows
    assert all(row["season"] > 2010 for row in rows)
    assert all(row["season"] <= 2022 for row in rows)
    assert all(0.0 <= row["roster_turnover"] <= 1.0 for row in rows)
    assert all(np.isfinite(row["roster_quality_delta"]) for row in rows)
    assert all(np.isfinite(row["target_net_rtg_change"]) for row in rows)


def test_acquisition_model_must_pass_walk_forward_validation_before_scoring():
    model = fit_acquisition_impact_model(
        _synthetic_roster_history(), league_id=10, competition="RS", through_season=2022
    )

    assert model["status"] == "validated"
    validation = model["validation"]
    assert validation["samples"] >= 20
    assert validation["seasons"] >= 3
    assert validation["rmse"] < validation["baseline_rmse"]
    assert validation["beats_zero_change_baseline"] is True

    impact = predict_acquisition_impact(
        model,
        roster_quality_delta=0.5,
        transaction_turnover_delta=0.15,
        prior_net_rtg=2.0,
        prior_net_rtg_change=0.25,
    )
    assert impact["available"] is True
    assert impact["status"] == "validated"
    assert impact["estimated_incremental_net_rating"] is not None
    assert impact["incremental_net_rating_interval_90"]["lower"] <= impact[
        "incremental_net_rating_interval_90"
    ]["upper"]
    assert impact["predicted_team_net_rating_change_interval_80"]["lower"] <= impact[
        "predicted_team_net_rating_change_interval_80"
    ]["upper"]
    assert "non identifica l'effetto causale" in impact["interpretation"]


def test_acquisition_model_fails_closed_without_roster_history():
    model = fit_acquisition_impact_model(
        {
            "player_stats": pd.DataFrame(),
            "team_player_relations": pd.DataFrame(),
            "team_season_stats": pd.DataFrame(),
        },
        league_id=10,
        competition="RS",
        through_season=2026,
    )

    impact = predict_acquisition_impact(
        model,
        roster_quality_delta=1.0,
        transaction_turnover_delta=0.1,
        prior_net_rtg=3.0,
        prior_net_rtg_change=0.2,
    )
    assert model["status"] == "roster_history_unavailable"
    assert impact["available"] is False
    assert impact.get("estimated_incremental_net_rating") is None


def test_acquisition_model_returns_no_numeric_impact_when_validation_is_not_good_enough():
    model = fit_acquisition_impact_model(
        {
            "player_stats": pd.DataFrame(),
            "team_player_relations": pd.DataFrame(),
            "team_season_stats": pd.DataFrame(),
        },
        league_id=10,
        competition="RS",
        through_season=2026,
    )
    impact = predict_acquisition_impact(
        model,
        roster_quality_delta=0.4,
        transaction_turnover_delta=0.15,
        prior_net_rtg=0.0,
        prior_net_rtg_change=0.0,
    )
    assert impact["status"] != "validated"
    assert impact["available"] is False

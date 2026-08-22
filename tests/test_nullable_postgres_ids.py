from __future__ import annotations

import pandas as pd


def test_normalise_ids_uses_nullable_int64_and_preserves_bigints():
    from basketball_ai.data.postgres_loader import _normalise_ids

    big_player = 9_007_199_254_740_993
    big_team = 9_007_199_254_740_995
    big_league = 9_007_199_254_740_997

    data = {
        "leagues": pd.DataFrame({"id": pd.Series([big_league], dtype="object")}),
        "teams": pd.DataFrame(
            {
                "id": pd.Series([big_team], dtype="object"),
                "league_id": pd.Series([big_league], dtype="object"),
            }
        ),
        "players": pd.DataFrame(
            {
                "id": pd.Series([big_player], dtype="object"),
                "current_team_id": pd.Series([None], dtype="object"),
                "current_league_id": pd.Series([None], dtype="object"),
            }
        ),
        "player_stats": pd.DataFrame(
            {
                "player_id": pd.Series([big_player], dtype="object"),
                "team_id": pd.Series([big_team], dtype="object"),
                "league_id": pd.Series([big_league], dtype="object"),
                "season": [2025],
                "competition": ["RS"],
            }
        ),
        "team_player_relations": pd.DataFrame(
            {
                "player_id": pd.Series([big_player], dtype="object"),
                "team_id": pd.Series([big_team], dtype="object"),
                "season": [2025],
            }
        ),
        "team_season_stats": pd.DataFrame(
            {
                "team_id": pd.Series([big_team], dtype="object"),
                "league_id": pd.Series([big_league], dtype="object"),
                "season": [2025],
                "competition": ["RS"],
            }
        ),
    }

    _normalise_ids(data)

    assert str(data["players"]["current_team_id"].dtype) == "Int64"
    assert str(data["players"]["current_league_id"].dtype) == "Int64"
    assert int(data["player_stats"].iloc[0]["player_id"]) == big_player
    assert int(data["player_stats"].iloc[0]["team_id"]) == big_team
    assert int(data["player_stats"].iloc[0]["league_id"]) == big_league


def test_fill_current_team_league_accepts_nullable_int64_targets():
    from basketball_ai.data.loader import _fill_current_team_league
    from basketball_ai.data.postgres_loader import _normalise_ids

    player_id = 101
    team_id = 202
    league_id = 303

    players = pd.DataFrame(
        {
            "id": [player_id],
            "current_team_id": [float("nan")],
            "current_league_id": [float("nan")],
        }
    )
    stats = pd.DataFrame(
        {
            "player_id": [player_id],
            "team_id": [team_id],
            "league_id": [league_id],
            "season": [2025],
            "competition": ["RS"],
        }
    )

    data = {
        "players": players,
        "player_stats": stats,
        "leagues": pd.DataFrame(),
        "teams": pd.DataFrame(),
        "team_player_relations": pd.DataFrame(),
        "team_season_stats": pd.DataFrame(),
    }
    _normalise_ids(data)
    _fill_current_team_league(players, stats)

    assert str(players["current_team_id"].dtype) == "Int64"
    assert str(players["current_league_id"].dtype) == "Int64"
    assert int(players.loc[0, "current_team_id"]) == team_id
    assert int(players.loc[0, "current_league_id"]) == league_id

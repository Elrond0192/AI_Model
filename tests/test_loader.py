from __future__ import annotations

import pandas as pd

from basketball_ai.data.loader import _to_int, load_all_data, load_players


def test_to_int_supports_decimal_and_hex_strings():
    assert _to_int("42") == 42
    assert _to_int("0000009B") == 155
    assert _to_int("0x9B") == 155
    assert _to_int(10.0) == 10


def test_load_players_converts_hex_nullable_id_fields(tmp_path):
    pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Test Player",
                "age": 24,
                "position": "PG",
                "nationality": "IT",
                "height_cm": 190,
                "weight_kg": 85,
                "dominant_hand": "R",
                "current_team_id": "0000009B",
                "current_league_id": "0000000A",
                "draft_year": "000007D0",
                "draft_pick": "0000001E",
            }
        ]
    ).to_csv(tmp_path / "players.csv", index=False)

    players = load_players(str(tmp_path))

    assert players[0].current_team_id == 155
    assert players[0].current_league_id == 10
    assert players[0].draft_year == 2000
    assert players[0].draft_pick == 30


def test_load_all_data_converts_hex_ids_for_lookup_dicts(tmp_path):
    pd.DataFrame([{"id": "0000009B", "name": "League"}]).to_csv(tmp_path / "leagues.csv", index=False)
    pd.DataFrame([{"id": "000000A0", "league_id": "0000009B", "name": "Team"}]).to_csv(
        tmp_path / "teams.csv", index=False
    )
    pd.DataFrame([{"id": "00000001", "name": "Player"}]).to_csv(tmp_path / "players.csv", index=False)
    pd.DataFrame([{"player_id": 1}]).to_csv(tmp_path / "player_stats.csv", index=False)
    pd.DataFrame([{"team_id": 1, "player_id": 1}]).to_csv(tmp_path / "team_player_relations.csv", index=False)

    data = load_all_data(str(tmp_path))

    assert 155 in data["league_dict"]
    assert 160 in data["team_dict"]
    assert 1 in data["player_dict"]
    assert data["league_teams"][155] == [160]

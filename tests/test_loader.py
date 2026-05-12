from __future__ import annotations

import pandas as pd

from basketball_ai.data.loader import (
    _to_int,
    _derive_playing_style,
    _compute_star_player_usage,
    _fill_current_team_league,
    load_all_data,
    load_players,
)


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


# ---------------------------------------------------------------------------
# Tests for _derive_playing_style
# ---------------------------------------------------------------------------

def _make_teams_df(rows):
    return pd.DataFrame(rows)


def test_derive_playing_style_pace_and_space():
    df = _make_teams_df([
        {"id": 1, "pace": 90, "three_point_attempt_rate": 0.55, "assists_per_game": 20, "defensive_rating": 112},
        {"id": 2, "pace": 60, "three_point_attempt_rate": 0.20, "assists_per_game": 15, "defensive_rating": 100},
        {"id": 3, "pace": 65, "three_point_attempt_rate": 0.22, "assists_per_game": 16, "defensive_rating": 101},
        {"id": 4, "pace": 63, "three_point_attempt_rate": 0.25, "assists_per_game": 17, "defensive_rating": 102},
        {"id": 5, "pace": 62, "three_point_attempt_rate": 0.23, "assists_per_game": 18, "defensive_rating": 103},
    ])
    _derive_playing_style(df)
    assert df.loc[df["id"] == 1, "playing_style"].iloc[0] == "pace_and_space"


def test_derive_playing_style_defensive():
    df = _make_teams_df([
        {"id": 1, "pace": 58, "three_point_attempt_rate": 0.35, "assists_per_game": 20, "defensive_rating": 95},
        {"id": 2, "pace": 80, "three_point_attempt_rate": 0.42, "assists_per_game": 28, "defensive_rating": 115},
        {"id": 3, "pace": 82, "three_point_attempt_rate": 0.44, "assists_per_game": 29, "defensive_rating": 116},
        {"id": 4, "pace": 83, "three_point_attempt_rate": 0.45, "assists_per_game": 30, "defensive_rating": 117},
        {"id": 5, "pace": 84, "three_point_attempt_rate": 0.46, "assists_per_game": 31, "defensive_rating": 118},
    ])
    _derive_playing_style(df)
    assert df.loc[df["id"] == 1, "playing_style"].iloc[0] == "defensive"


def test_derive_playing_style_no_op_when_too_few_teams():
    df = _make_teams_df([
        {"id": 1, "pace": 80, "three_point_attempt_rate": 0.5, "assists_per_game": 25, "defensive_rating": 108},
        {"id": 2, "pace": 70, "three_point_attempt_rate": 0.3, "assists_per_game": 18, "defensive_rating": 112},
    ])
    # No playing_style column yet; with < 4 teams, function should leave df unchanged
    _derive_playing_style(df)
    assert "playing_style" not in df.columns


def test_derive_playing_style_no_op_when_missing_columns():
    df = pd.DataFrame([{"id": 1, "pace": 75}])
    _derive_playing_style(df)
    assert "playing_style" not in df.columns


# ---------------------------------------------------------------------------
# Tests for _compute_star_player_usage
# ---------------------------------------------------------------------------

def test_compute_star_player_usage_sets_max_usg_pct():
    teams_df = pd.DataFrame([{"id": 10}])
    stats_df = pd.DataFrame([
        {"team_id": 10, "player_id": 1, "usg_pct": 32.0, "games_played": 30, "minutes_per_game": 28.0},
        {"team_id": 10, "player_id": 2, "usg_pct": 18.0, "games_played": 30, "minutes_per_game": 22.0},
    ])
    _compute_star_player_usage(teams_df, stats_df)
    assert abs(teams_df.loc[0, "star_player_usage"] - 0.32) < 1e-6


def test_compute_star_player_usage_clips_to_bounds():
    teams_df = pd.DataFrame([{"id": 1}])
    stats_df = pd.DataFrame([
        {"team_id": 1, "player_id": 1, "usg_pct": 70.0, "games_played": 30, "minutes_per_game": 35.0},
    ])
    _compute_star_player_usage(teams_df, stats_df)
    assert teams_df.loc[0, "star_player_usage"] <= 0.60


def test_compute_star_player_usage_hex_team_id():
    teams_df = pd.DataFrame([{"id": "0000009B"}])   # hex → 155
    stats_df = pd.DataFrame([
        {"team_id": "0000009B", "player_id": 1, "usg_pct": 28.0, "games_played": 20, "minutes_per_game": 25.0},
    ])
    _compute_star_player_usage(teams_df, stats_df)
    assert abs(teams_df.loc[0, "star_player_usage"] - 0.28) < 1e-6


def test_compute_star_player_usage_fallback_on_missing_data():
    teams_df = pd.DataFrame([{"id": 99}])
    stats_df = pd.DataFrame([{"team_id": 1, "player_id": 1, "usg_pct": 25.0, "games_played": 30, "minutes_per_game": 20.0}])
    _compute_star_player_usage(teams_df, stats_df)
    assert teams_df.loc[0, "star_player_usage"] == 0.25  # fallback default


# ---------------------------------------------------------------------------
# Tests for _fill_current_team_league
# ---------------------------------------------------------------------------

def test_fill_current_team_league_fills_null_values():
    players_df = pd.DataFrame([
        {"id": 1, "current_team_id": None, "current_league_id": None},
    ])
    stats_df = pd.DataFrame([
        {"player_id": 1, "team_id": 42, "league_id": 7, "season": "2023-24"},
        {"player_id": 1, "team_id": 55, "league_id": 8, "season": "2022-23"},
    ])
    _fill_current_team_league(players_df, stats_df)
    # Should use the most recent season (2023-24)
    assert players_df.loc[0, "current_team_id"] == 42
    assert players_df.loc[0, "current_league_id"] == 7


def test_fill_current_team_league_does_not_overwrite_existing():
    players_df = pd.DataFrame([
        {"id": 1, "current_team_id": 99, "current_league_id": 3},
    ])
    stats_df = pd.DataFrame([
        {"player_id": 1, "team_id": 42, "league_id": 7, "season": "2023-24"},
    ])
    _fill_current_team_league(players_df, stats_df)
    # Existing values must be preserved
    assert players_df.loc[0, "current_team_id"] == 99
    assert players_df.loc[0, "current_league_id"] == 3


def test_fill_current_team_league_hex_ids():
    players_df = pd.DataFrame([
        {"id": "0000009B", "current_team_id": None, "current_league_id": None},
    ])
    stats_df = pd.DataFrame([
        {"player_id": "0000009B", "team_id": "000000A0", "league_id": "0000009B", "season": "2023-24"},
    ])
    _fill_current_team_league(players_df, stats_df)
    assert players_df.loc[0, "current_team_id"] == "000000A0"   # value from stats, not yet _to_int-ed


def test_load_all_data_enriches_teams_and_fills_player_current_team(tmp_path):
    """load_all_data must derive playing_style, star_player_usage and fill current team ids."""
    pd.DataFrame([{"id": 1, "name": "League", "country": "IT", "tier": 1,
                   "competitiveness_score": 1.0, "avg_pace": 75.0, "avg_offensive_rating": 110.0}]).to_csv(
        tmp_path / "leagues.csv", index=False
    )
    teams = [
        {"id": i, "name": f"T{i}", "league_id": 1, "playing_style": "balanced",
         "formation": "", "pace": 60 + i * 5, "offensive_rating": 108.0, "defensive_rating": 110.0 - i,
         "three_point_attempt_rate": 0.20 + i * 0.05, "assists_per_game": 18.0 + i,
         "star_player_usage": 0.25, "league_tier": 1, "short_name": f"T{i}", "net_rtg": 0.0}
        for i in range(1, 6)
    ]
    pd.DataFrame(teams).to_csv(tmp_path / "teams.csv", index=False)
    pd.DataFrame([{"id": 1, "name": "P1", "age": 25, "position": "PG", "nationality": "IT",
                   "height_cm": 190, "weight_kg": 85, "dominant_hand": "R",
                   "current_team_id": None, "current_league_id": None,
                   "draft_year": None, "draft_pick": None}]).to_csv(tmp_path / "players.csv", index=False)
    pd.DataFrame([{"player_id": 1, "team_id": 3, "league_id": 1, "season": "2023-24",
                   "games_played": 30, "minutes_per_game": 28.0, "usg_pct": 28.0,
                   "points": 18.0, "rebounds": 4.0, "assists": 7.0,
                   "offensive_rebounds": 1.0, "defensive_rebounds": 3.0,
                   "steals": 1.5, "blocks": 0.3, "turnovers": 2.0, "personal_fouls": 2.5,
                   "fg_pct": 0.47, "three_point_pct": 0.38, "ft_pct": 0.82,
                   "plus_minus": 3.0, "per": 22.0, "ts_pct": 0.58, "bpm": 2.0,
                   "vorp": 2.5, "win_shares": 6.0, "ast_ratio": 25.0, "reb_pct": 8.0,
                   "rating": 7.5}]).to_csv(tmp_path / "player_stats.csv", index=False)
    pd.DataFrame([{"team_id": 3, "player_id": 1, "season": "2023-24",
                   "role": "starter", "jersey_number": 7}]).to_csv(
        tmp_path / "team_player_relations.csv", index=False
    )

    data = load_all_data(str(tmp_path))
    teams_out = data["teams"]
    players_out = data["players"]

    # playing_style must not be 'balanced' for all teams (at least one differs)
    assert not all(teams_out["playing_style"] == "balanced")
    # star_player_usage must be derived (team 3 has player with 28% usage)
    star_usg = teams_out.loc[teams_out["id"] == 3, "star_player_usage"]
    assert not star_usg.empty
    assert abs(star_usg.iloc[0] - 0.28) < 1e-6
    # current_team_id filled from stats
    assert players_out.loc[players_out["id"] == 1, "current_team_id"].iloc[0] == 3
    assert players_out.loc[players_out["id"] == 1, "current_league_id"].iloc[0] == 1

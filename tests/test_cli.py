"""Unit tests for the production operations CLI."""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest


def _data() -> dict:
    player_stats = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 2,
                "league_id": 3,
                "season": season,
                "games_played": 20,
                "rating": 6.5,
            }
            for season in range(2021, 2026)
        ]
    )
    return {
        "players": pd.DataFrame(
            [{"id": 1, "global_id": "P1", "name": "Player", "position": "PG"}]
        ),
        "teams": pd.DataFrame(
            [{"id": 2, "global_id": "T2", "name": "Team", "league_id": 3}]
        ),
        "player_stats": player_stats,
        "team_season_stats": pd.DataFrame(
            [
                {
                    "team_id": 2,
                    "global_id": "T2",
                    "name": "Team",
                    "league_id": 3,
                    "season": season,
                    "pace": 75.0,
                    "offensive_rating": 112.0,
                    "defensive_rating": 108.0,
                }
                for season in range(2021, 2026)
            ]
        ),
    }


def test_parse_args_supports_lifecycle_commands():
    from basketball_ai.cli import parse_args

    args = parse_args(
        [
            "--mode",
            "promote",
            "--database-profile",
            "production",
            "--model-dir",
            "/tmp/models",
        ]
    )
    assert args.mode == "promote"
    assert args.database_profile == "production"
    assert args.model_dir == "/tmp/models"


def test_validate_data_pass(monkeypatch, capsys):
    import basketball_ai.cli as cli

    monkeypatch.setattr(cli, "_load_data", lambda args: _data())
    args = argparse.Namespace(database_profile="production")
    cli.mode_validate_data(args)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS"
    assert report["players"] == 1
    assert report["teams"] == 1
    assert report["team_season_stats"] == 5
    assert report["seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert report["consecutive_pairs"] == 4


def test_validate_data_fails_for_bad_rating_and_short_history(monkeypatch, capsys):
    import basketball_ai.cli as cli

    data = _data()
    data["player_stats"] = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 2,
                "league_id": 3,
                "season": 2025,
                "games_played": 5,
                "rating": 11.0,
            }
        ]
    )
    monkeypatch.setattr(cli, "_load_data", lambda args: data)
    with pytest.raises(SystemExit) as exc:
        cli.mode_validate_data(argparse.Namespace(database_profile=None))
    assert exc.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert any("outside [0, 10]" in issue for issue in report["issues"])
    assert any("at least 5" in issue for issue in report["issues"])
    assert any("consecutive" in issue for issue in report["issues"])


def test_validate_data_reports_non_consecutive_gaps(monkeypatch, capsys):
    import basketball_ai.cli as cli

    data = _data()
    data["player_stats"] = data["player_stats"][
        data["player_stats"]["season"] != 2023
    ].reset_index(drop=True)
    monkeypatch.setattr(cli, "_load_data", lambda args: data)
    with pytest.raises(SystemExit):
        cli.mode_validate_data(argparse.Namespace(database_profile=None))
    report = json.loads(capsys.readouterr().out)
    assert report["players_with_non_consecutive_gaps"] == 1


def test_mode_api_uses_promoted_directory(monkeypatch, tmp_path):
    import basketball_ai.cli as cli
    import uvicorn

    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app
        called.update(kwargs)

    production = tmp_path / "production"
    production.mkdir()
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("DATABASE_PROFILE", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setenv("API_ENV", "production")
    args = argparse.Namespace(
        database_profile="prod",
        model_dir=str(tmp_path),
        host="127.0.0.1",
        port=8123,
    )
    cli.mode_api(args)
    assert called == {
        "app": "basketball_ai.api.main:app",
        "host": "127.0.0.1",
        "port": 8123,
        "reload": False,
    }
    assert cli.os.environ["DATABASE_PROFILE"] == "prod"
    assert cli.os.environ["DATA_SOURCE"] == "postgres"
    assert cli.os.environ["MODEL_DIR"] == str(production)


def test_mode_api_fails_closed_without_production(monkeypatch, tmp_path):
    import basketball_ai.cli as cli

    monkeypatch.setenv("API_ENV", "production")
    with pytest.raises(RuntimeError, match="promoted production model"):
        cli.mode_api(
            argparse.Namespace(
                database_profile=None,
                model_dir=str(tmp_path),
                host="127.0.0.1",
                port=8000,
            )
        )


def test_main_dispatches(monkeypatch):
    import basketball_ai.cli as cli

    seen = {}

    def fake_validate(args):
        seen["mode"] = args.mode

    monkeypatch.setattr(cli, "mode_validate_data", fake_validate)
    cli.main(["--mode", "validate-data"])
    assert seen["mode"] == "validate-data"

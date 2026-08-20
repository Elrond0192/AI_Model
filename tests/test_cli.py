"""Unit tests for the production operations CLI."""
from __future__ import annotations

import argparse
import json

import pandas as pd


def _data() -> dict:
    return {
        "players": pd.DataFrame(
            [{"id": 1, "global_id": "P1", "name": "Player", "position": "PG"}]
        ),
        "teams": pd.DataFrame(
            [{"id": 2, "global_id": "T2", "name": "Team", "league_id": 3}]
        ),
        "player_stats": pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": season,
                    "games_played": 20,
                    "rating": 6.5,
                }
                for season in range(2021, 2026)
            ]
        ),
    }


def test_parse_args_and_data_cutoff():
    from basketball_ai.cli import _data_cutoff, parse_args

    args = parse_args(
        [
            "--mode",
            "api",
            "--database-profile",
            "production",
            "--model-dir",
            "/tmp/models",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )
    assert args.mode == "api"
    assert args.database_profile == "production"
    assert args.port == 9000
    assert _data_cutoff(_data()) == "2025-12-31"


def test_validate_data_pass(monkeypatch, capsys):
    import basketball_ai.cli as cli

    monkeypatch.setattr(cli, "_load_data", lambda args: _data())
    args = argparse.Namespace(database_profile="production")
    cli.mode_validate_data(args)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS"
    assert report["players"] == 1
    assert report["teams"] == 1
    assert report["seasons"] == [2021, 2022, 2023, 2024, 2025]


def test_validate_data_warns_for_bad_rating_and_short_history(monkeypatch, capsys):
    import basketball_ai.cli as cli

    data = _data()
    data["player_stats"] = pd.DataFrame(
        [{"player_id": 1, "season": 2025, "games_played": 5, "rating": 11.0}]
    )
    monkeypatch.setattr(cli, "_load_data", lambda args: data)
    cli.mode_validate_data(argparse.Namespace(database_profile=None))
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "WARN"
    assert any("outside [0, 10]" in issue for issue in report["issues"])
    assert any("at least 5" in issue for issue in report["issues"])


def test_mode_api_sets_runtime_and_calls_uvicorn(monkeypatch):
    import basketball_ai.cli as cli
    import uvicorn

    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("DATABASE_PROFILE", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    args = argparse.Namespace(
        database_profile="prod",
        model_dir="/models",
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
    assert cli.os.environ["MODEL_DIR"] == "/models"


def test_main_dispatches(monkeypatch):
    import basketball_ai.cli as cli

    seen = {}

    def fake_validate(args):
        seen["mode"] = args.mode

    monkeypatch.setattr(cli, "mode_validate_data", fake_validate)
    cli.main(["--mode", "validate-data"])
    assert seen["mode"] == "validate-data"

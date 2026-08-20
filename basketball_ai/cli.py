"""Command-line operations for AI_Model.

The production path is PostgreSQL -> train/backtest -> promote -> FastAPI.
Synthetic CSV generation and the retired WordPress JSON exporter are not part
of the production CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBallstat AI_Model operations")
    parser.add_argument(
        "--mode",
        choices=("train", "backtest", "validate-data", "publish-batch", "api"),
        required=True,
    )
    parser.add_argument("--database-profile", default=None)
    parser.add_argument("--model-dir", default="models_saved")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def _load_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from basketball_ai.data.postgres_loader import load_all_data

    return load_all_data()


def _data_cutoff(data: dict[str, Any]) -> str:
    stats = data["player_stats"]
    latest = int(pd.to_numeric(stats["season"], errors="coerce").dropna().max())
    return date(latest, 12, 31).isoformat()


def mode_train(args: argparse.Namespace) -> None:
    from basketball_ai.models.backtest import run_backtest
    from basketball_ai.models.ensemble import EnsembleModel

    data = _load_data(args)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    ensemble = EnsembleModel()
    metrics = ensemble.train(data)
    report = run_backtest(
        data,
        output_path=str(model_dir / "backtest_report.json"),
    )
    if not report.get("valid") or not report.get("folds"):
        raise RuntimeError("Backtest invalid; candidate model was not saved")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    metadata = {
        **(metrics or {}),
        "model_run_id": run_id,
        "model_version": os.getenv("MODEL_VERSION", "2.0.0"),
        "feature_version": "forecast-t-plus-1-v1",
        "data_cutoff": _data_cutoff(data),
        "database_profile": args.database_profile or os.getenv("DATABASE_PROFILE", ""),
        "backtest": report,
    }
    ensemble.save(str(model_dir), metadata)
    print(json.dumps(metadata, indent=2, default=str))


def mode_backtest(args: argparse.Namespace) -> None:
    from basketball_ai.models.backtest import run_backtest

    report = run_backtest(
        _load_data(args),
        output_path=str(Path(args.model_dir) / "backtest_report.json"),
    )
    print(json.dumps(report, indent=2, default=str))
    if not report.get("valid"):
        raise SystemExit(2)


def mode_validate_data(args: argparse.Namespace) -> None:
    data = _load_data(args)
    stats = data["player_stats"]
    players = data["players"]
    teams = data["teams"]

    issues: list[str] = []
    for name, frame, required in (
        ("players", players, ("id", "global_id", "name", "position")),
        ("teams", teams, ("id", "global_id", "name", "league_id")),
        ("player_stats", stats, ("player_id", "season", "games_played", "rating")),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            issues.append(f"{name}: missing {missing}")

    if "rating" in stats.columns:
        bad = stats["rating"].dropna()
        if ((bad < 0) | (bad > 10)).any():
            issues.append("player_stats.rating contains values outside [0, 10]")

    seasons = sorted(
        pd.to_numeric(stats["season"], errors="coerce").dropna().astype(int).unique()
    )
    if len(seasons) < 5:
        issues.append(
            f"only {len(seasons)} season(s) available; at least 5 are required by the operations console"
        )

    report = {
        "status": "PASS" if not issues else "WARN",
        "players": len(players),
        "teams": len(teams),
        "player_stats": len(stats),
        "seasons": seasons,
        "issues": issues,
    }
    print(json.dumps(report, indent=2))


def mode_publish_batch(args: argparse.Namespace) -> None:
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from basketball_ai.export.postgres_export import publish_current_team_forecasts

    count = publish_current_team_forecasts(args.model_dir)
    print(f"Published {count} current-team forecasts")


def mode_api(args: argparse.Namespace) -> None:
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    os.environ.setdefault("DATA_SOURCE", "postgres")
    os.environ.setdefault("MODEL_DIR", args.model_dir)

    import uvicorn

    uvicorn.run(
        "basketball_ai.api.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    {
        "train": mode_train,
        "backtest": mode_backtest,
        "validate-data": mode_validate_data,
        "publish-batch": mode_publish_batch,
        "api": mode_api,
    }[args.mode](args)


if __name__ == "__main__":
    main()

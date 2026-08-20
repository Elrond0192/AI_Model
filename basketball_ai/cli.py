"""Production command-line operations for AI_Model."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBallstat AI_Model operations")
    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "backtest",
            "validate-data",
            "publish-batch",
            "promote",
            "rollback",
            "api",
        ),
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


def _latest_season(data: dict[str, Any]) -> int:
    return int(pd.to_numeric(data["player_stats"]["season"], errors="coerce").dropna().max())


def mode_train(args: argparse.Namespace) -> None:
    from basketball_ai.models.backtest import run_backtest
    from basketball_ai.models.strict_production import StrictProductionEnsembleModel
    from basketball_ai.models.promote import register_candidate

    data = _load_data(args)
    model_root = Path(args.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    ensemble = StrictProductionEnsembleModel()
    metrics = ensemble.train(data)
    report = run_backtest(
        data,
        output_path=str(model_root / "backtest_report.json"),
    )
    if not report.get("valid") or not report.get("folds"):
        raise RuntimeError("Backtest invalid; candidate model was not saved")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = model_root / "runs" / run_id
    metadata = {
        **(metrics or {}),
        "model_run_id": run_id,
        "model_version": os.getenv("MODEL_VERSION", "2.1.0"),
        "feature_version": "forecast-t-plus-1-v2",
        "data_cutoff": datetime.now(timezone.utc).date().isoformat(),
        "latest_observed_season": _latest_season(data),
        "database_profile": args.database_profile or os.getenv("DATABASE_PROFILE", ""),
        "backtest": report,
    }
    ensemble.save(str(run_dir), metadata)
    candidate = register_candidate(str(model_root), run_id, run_dir, metadata)
    print(
        json.dumps(
            {"run": metadata, "registry_candidate": candidate},
            indent=2,
            default=str,
        )
    )


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
    team_history = data["team_season_stats"]

    issues: list[str] = []
    warnings: list[str] = []
    for name, frame, required in (
        ("players", players, ("id", "global_id", "name", "position")),
        ("teams", teams, ("id", "global_id", "name", "league_id")),
        ("player_stats", stats, ("player_id", "season", "games_played", "rating")),
        ("team_season_stats", team_history, ("team_id", "season", "pace", "offensive_rating", "defensive_rating")),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            issues.append(f"{name}: missing {missing}")

    bad = pd.to_numeric(stats["rating"], errors="coerce").dropna()
    if ((bad < 0) | (bad > 10)).any():
        issues.append("player_stats.rating contains values outside [0, 10]")
    if stats.duplicated(["player_id", "season"], keep=False).any():
        issues.append("player_stats has duplicate player_id+season rows")
    if team_history.duplicated(["team_id", "season"], keep=False).any():
        issues.append("team_season_stats has duplicate team_id+season rows")

    seasons = sorted(
        int(value)
        for value in pd.to_numeric(stats["season"], errors="coerce").dropna().unique()
    )
    if len(seasons) < 5:
        issues.append(f"only {len(seasons)} season(s) available; at least 5 are required")

    consecutive_pairs = 0
    players_with_gaps = 0
    for _, group in stats.groupby("player_id"):
        years = sorted(
            int(value)
            for value in pd.to_numeric(group["season"], errors="coerce").dropna().unique()
        )
        pairs = sum(1 for year in years if year + 1 in years)
        consecutive_pairs += pairs
        if len(years) > 1 and pairs < len(years) - 1:
            players_with_gaps += 1
    if consecutive_pairs == 0:
        issues.append("no consecutive t -> t+1 player-season pairs are available")

    rating_by_league: dict[str, dict[str, float]] = {}
    if "league_id" in stats.columns:
        for league_id, group in stats.groupby("league_id"):
            values = pd.to_numeric(group["rating"], errors="coerce").dropna()
            if values.empty:
                continue
            rating_by_league[str(league_id)] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        means = [entry["mean"] for entry in rating_by_league.values() if entry["n"] >= 20]
        if len(means) >= 2 and max(means) - min(means) > 1.0:
            warnings.append(
                "league rating means differ by more than 1.0; verify ValLegaPerGame is globally comparable before trusting cross-league forecasts"
            )

    report = {
        "status": "FAIL" if issues else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "players": len(players),
        "teams": len(teams),
        "player_stats": len(stats),
        "team_season_stats": len(team_history),
        "seasons": seasons,
        "consecutive_pairs": consecutive_pairs,
        "players_with_non_consecutive_gaps": players_with_gaps,
        "rating_by_league": rating_by_league,
        "issues": issues,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(2)


def mode_publish_batch(args: argparse.Namespace) -> None:
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from basketball_ai.export.postgres_export import publish_current_team_forecasts
    from basketball_ai.models.promote import production_model_dir

    active = production_model_dir(args.model_dir)
    count = publish_current_team_forecasts(str(active))
    print(f"Published {count} production current-team forecasts")


def mode_promote(args: argparse.Namespace) -> None:
    from basketball_ai.models.promote import promote_if_better

    result = promote_if_better(args.model_dir)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("promoted"):
        raise SystemExit(2)


def mode_rollback(args: argparse.Namespace) -> None:
    from basketball_ai.models.promote import rollback_to_previous

    result = rollback_to_previous(args.model_dir)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("rolled_back"):
        raise SystemExit(2)


def mode_api(args: argparse.Namespace) -> None:
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    os.environ.setdefault("DATA_SOURCE", "postgres")

    root = Path(args.model_dir)
    production = root / "production"
    if production.is_dir():
        os.environ["MODEL_DIR"] = str(production)
    elif os.environ.get("API_ENV", "development").lower() == "development" and (root / "metadata.json").exists():
        os.environ["MODEL_DIR"] = str(root)
    else:
        raise RuntimeError("API requires a promoted production model")

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
        "promote": mode_promote,
        "rollback": mode_rollback,
        "api": mode_api,
    }[args.mode](args)


if __name__ == "__main__":
    main()

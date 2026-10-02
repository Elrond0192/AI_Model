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

from basketball_ai.models.competition_training import normalize_competition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBallstat AI_Model operations")
    parser.add_argument(
        "--mode",
        choices=(
            "train", "backtest", "validate-data", "bb-rating-calibrate", "publish-batch",
            "promote", "rollback", "prepare-snapshot", "refresh-serving", "api",
        ),
        required=True,
    )
    parser.add_argument("--database-profile", default=None)
    parser.add_argument("--model-dir", default="models_saved")
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--league-key", default=None)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--diagnostic-stages", action="store_true")
    parser.add_argument("--compare-target-modes", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="models_saved/bb_rating_calibration")
    parser.add_argument("--min-peer-samples", type=int, default=25)
    parser.add_argument("--min-context-samples", type=int, default=10)
    parser.add_argument("--uncertainty-min-samples", type=int, default=50)
    parser.add_argument(
        "--include-latest-validation-season",
        action="store_true",
        help="Include the latest observed target season in OOS uncertainty validation.",
    )
    parser.add_argument("--min-season", type=int, default=None)
    parser.add_argument("--max-season", type=int, default=None)
    return parser.parse_args(argv)


def _load_data(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_id = getattr(args, "snapshot_id", None)
    if snapshot_id:
        from basketball_ai.data.training_snapshots import load_training_snapshot

        return load_training_snapshot(snapshot_id, getattr(args, "snapshot_dir", None))
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from basketball_ai.data.postgres_loader import load_all_data
    return load_all_data(purpose="training")


def mode_prepare_snapshot(args: argparse.Namespace) -> None:
    from basketball_ai.data.training_snapshots import create_training_snapshot

    data = _load_data(argparse.Namespace(database_profile=args.database_profile, snapshot_id=None))
    manifest = create_training_snapshot(
        data,
        profile=args.database_profile or os.getenv("DATABASE_PROFILE", "production"),
        root=args.snapshot_dir,
    )
    print(json.dumps(manifest, indent=2))


def mode_refresh_serving(args: argparse.Namespace) -> None:
    if not args.league_key or args.season is None or not args.competition:
        raise SystemExit("--league-key, --season and --competition are required")
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from sqlalchemy import text

    from basketball_ai.data.postgres_loader import get_engine

    engine = get_engine()
    try:
        with engine.begin() as connection:
            connection.execute(
                text('CALL "AI_Source"."RefreshContext"(:league, :season, :competition)'),
                {
                    "league": args.league_key,
                    "season": args.season,
                    "competition": args.competition,
                },
            )
    finally:
        engine.dispose()
    print(json.dumps({"refreshed": True, "league_key": args.league_key, "season": args.season, "competition": args.competition.upper()}))


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
    report = run_backtest(data, output_path=str(model_root / "backtest_report.json"))
    if not report.get("valid") or not report.get("folds"):
        raise RuntimeError("Backtest invalid; candidate model was not saved")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = model_root / "runs" / run_id
    metadata = {
        **(metrics or {}),
        "model_run_id": run_id,
        "model_version": os.getenv("MODEL_VERSION", "2.6.0"),
        "feature_version": "forecast-t-plus-1-persistence-delta-v1",
        "data_cutoff": datetime.now(timezone.utc).date().isoformat(),
        "latest_observed_season": _latest_season(data),
        "database_profile": args.database_profile or os.getenv("DATABASE_PROFILE", ""),
        "training_snapshot_id": data.get("training_snapshot_manifest", {}).get("snapshot_id"),
        "training_snapshot_sha256": data.get("training_snapshot_manifest", {}).get("sha256"),
        "source_contract": data.get("source_contract", "competition-v2:canonical"),
        "backtest": report,
    }
    ensemble.save(str(run_dir), metadata)
    candidate = register_candidate(str(model_root), run_id, run_dir, metadata)
    print(json.dumps({"run": metadata, "registry_candidate": candidate}, indent=2, default=str))


def mode_backtest(args: argparse.Namespace) -> None:
    from basketball_ai.models.backtest import run_backtest
    report = run_backtest(
        _load_data(args),
        output_path=str(Path(args.model_dir) / "backtest_report.json"),
        include_stage_metrics=args.diagnostic_stages,
        compare_target_modes=args.compare_target_modes,
    )
    print(json.dumps(report, indent=2, default=str))
    if not report.get("valid"):
        raise SystemExit(2)


def mode_bb_rating_calibrate(args: argparse.Namespace) -> None:
    from basketball_ai.bb_rating.calibration import (
        BBRatingCalibrationConfig,
        build_calibration_report,
        write_calibration_report,
    )
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile

    if args.snapshot_id:
        from basketball_ai.data.training_snapshots import load_training_snapshot
        data = load_training_snapshot(args.snapshot_id, args.snapshot_dir)
    else:
        # BB-Rating calibration describes observed performance, so use the
        # canonical analysis contract rather than the training-serving view.
        from basketball_ai.data.postgres_loader import load_all_data
        data = load_all_data(purpose="analysis")

    stats = data["player_stats"].copy()
    if args.league_key:
        league_value = str(args.league_key).strip().upper()
        if "league_key" in stats.columns:
            stats = stats.loc[stats["league_key"].astype(str).str.upper() == league_value]
    if args.season is not None:
        stats = stats.loc[pd.to_numeric(stats["season"], errors="coerce") == int(args.season)]
    if args.min_season is not None:
        stats = stats.loc[pd.to_numeric(stats["season"], errors="coerce") >= int(args.min_season)]
    if args.max_season is not None:
        stats = stats.loc[pd.to_numeric(stats["season"], errors="coerce") <= int(args.max_season)]
    if args.competition:
        competition_value = str(args.competition).strip().upper()
        stats = stats.loc[stats["competition"].astype(str).str.upper() == competition_value]
    if stats.empty:
        raise RuntimeError("No player_stats rows remain after BB-Rating calibration filters")
    data = {**data, "player_stats": stats.reset_index(drop=True)}

    report = build_calibration_report(
        data,
        config=BBRatingCalibrationConfig(
            min_peer_samples=args.min_peer_samples,
            min_context_samples=args.min_context_samples,
            uncertainty_min_samples=args.uncertainty_min_samples,
            exclude_latest_target_season=not args.include_latest_validation_season,
        ),
    )
    paths = write_calibration_report(report, args.output_dir)
    payload = {
        "calibration_version": report["calibration_version"],
        "bb_rating_version": report["bb_rating_version"],
        "dataset": report["dataset"],
        "contexts": report["contexts"],
        "score_distribution": report["score_distribution"],
        "uncertainty_validation": report["uncertainty_validation"],
        "validation_signals": report["validation_signals"],
        "warnings": report["warnings"],
        "output": paths,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def mode_validate_data(args: argparse.Namespace) -> None:
    data = _load_data(args)
    stats = data["player_stats"].copy()
    players = data["players"]
    teams = data["teams"]
    team_history = data["team_season_stats"].copy()

    issues: list[str] = []
    warnings: list[str] = []
    for name, frame, required in (
        ("players", players, ("id", "global_id", "name", "position")),
        ("teams", teams, ("id", "global_id", "name", "league_id")),
        ("player_stats", stats, ("player_id", "league_id", "season", "competition", "games_played", "rating")),
        ("team_season_stats", team_history, ("team_id", "league_id", "season", "competition", "pace", "offensive_rating", "defensive_rating")),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            issues.append(f"{name}: missing {missing}")

    bad = pd.to_numeric(stats["rating"], errors="coerce").dropna()
    if ((bad < 0) | (bad > 10)).any():
        issues.append("player_stats.rating contains values outside [0, 10]")

    stats["competition"] = stats["competition"].map(normalize_competition)
    team_history["competition"] = team_history["competition"].map(normalize_competition)
    if stats.duplicated(["player_id", "league_id", "season", "competition"], keep=False).any():
        issues.append("player_stats has duplicate player+league+season+competition rows")
    if team_history.duplicated(["team_id", "league_id", "season", "competition"], keep=False).any():
        issues.append("team history has duplicate team+league+season+competition rows")

    seasons = sorted(int(value) for value in pd.to_numeric(stats["season"], errors="coerce").dropna().unique())
    if len(seasons) < 5:
        issues.append(f"only {len(seasons)} season(s) available; at least 5 are required")

    consecutive_pairs = 0
    pairs_by_competition: dict[str, int] = {}
    contexts_with_gaps = 0
    for (_, _, competition), group in stats.groupby(["player_id", "league_id", "competition"]):
        years = sorted(int(value) for value in pd.to_numeric(group["season"], errors="coerce").dropna().unique())
        pairs = sum(1 for year in years if year + 1 in years)
        consecutive_pairs += pairs
        pairs_by_competition[str(competition)] = pairs_by_competition.get(str(competition), 0) + pairs
        if len(years) > 1 and pairs < len(years) - 1:
            contexts_with_gaps += 1
    if consecutive_pairs == 0:
        issues.append("no consecutive t -> t+1 player/league/competition pairs are available")

    rating_by_league: dict[str, dict[str, float]] = {}
    for league_id, group in stats.groupby("league_id"):
        values = pd.to_numeric(group["rating"], errors="coerce").dropna()
        if not values.empty:
            rating_by_league[str(league_id)] = {
                "n": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=0))
            }
    means = [entry["mean"] for entry in rating_by_league.values() if entry["n"] >= 20]
    if len(means) >= 2 and max(means) - min(means) > 1.0:
        warnings.append("league rating means differ by more than 1.0; verify target comparability")

    rating_by_competition: dict[str, dict[str, float]] = {}
    for competition, group in stats.groupby("competition"):
        values = pd.to_numeric(group["rating"], errors="coerce").dropna()
        games = pd.to_numeric(group["games_played"], errors="coerce").fillna(0)
        rating_by_competition[str(competition)] = {
            "n": int(len(values)),
            "games": int(games.sum()),
            "mean": float(values.mean()) if not values.empty else 0.0,
            "std": float(values.std(ddof=0)) if not values.empty else 0.0,
            "consecutive_pairs": int(pairs_by_competition.get(str(competition), 0)),
        }
        if pairs_by_competition.get(str(competition), 0) == 0:
            warnings.append(
                f"competition {competition} has no consecutive training pairs and will not be served until more history exists"
            )

    report = {
        "status": "FAIL" if issues else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "source_contract": data.get("source_contract"),
        "players": len(players),
        "teams": len(teams),
        "player_stats": len(stats),
        "team_season_stats": len(team_history),
        "seasons": seasons,
        "consecutive_pairs": consecutive_pairs,
        "consecutive_pairs_by_competition": pairs_by_competition,
        "contexts_with_non_consecutive_gaps": contexts_with_gaps,
        "rating_by_league": rating_by_league,
        "rating_by_competition": rating_by_competition,
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
    print(f"Published {count} production current-context forecasts")


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
    uvicorn.run("basketball_ai.api.main:app", host=args.host, port=args.port, reload=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    {
        "train": mode_train,
        "backtest": mode_backtest,
        "validate-data": mode_validate_data,
        "bb-rating-calibrate": mode_bb_rating_calibrate,
        "publish-batch": mode_publish_batch,
        "promote": mode_promote,
        "rollback": mode_rollback,
        "prepare-snapshot": mode_prepare_snapshot,
        "refresh-serving": mode_refresh_serving,
        "api": mode_api,
    }[args.mode](args)


if __name__ == "__main__":
    main()

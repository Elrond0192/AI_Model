"""Root CLI entry-point for the Basketball Performance AI system.

Usage:
    python main.py --mode generate-data
    python main.py --mode train [--database-profile NAME]
    python main.py --mode api   [--host HOST] [--port PORT]
    python main.py --mode publish-batch [--database-profile NAME]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Basketball Performance AI – CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["generate-data", "train", "demo", "api", "publish-batch", "validate-data", "backup", "backtest"],
        required=True,
    )
    parser.add_argument("--data-dir",  default="data/sample",   help="Data directory (file source)")
    parser.add_argument("--model-dir", default="models_saved",  help="Model save directory")
    parser.add_argument("--host",      default="0.0.0.0",       help="API host")
    parser.add_argument("--port",      type=int, default=8000,  help="API port")
    parser.add_argument("--seed",      type=int, default=42,    help="RNG seed")
    parser.add_argument(
        "--database-profile",
        default=None,
        help="Named PostgreSQL profile; defaults to DATABASE_PROFILE",
    )
    parser.add_argument(
        "--out-dir",
        default="wp_export",
        help="Output directory for export-wordpress JSON files",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        default=False,
        help="D8: Replace player names with anonymous IDs in export output",
    )
    return parser.parse_args(argv)


def _load_data(args):
    """Load the selected PostgreSQL profile."""
    import os
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    from basketball_ai.data.postgres_loader import load_all_data
    print(f"[Data] Loading PostgreSQL profile {args.database_profile or os.getenv('DATABASE_PROFILE', '<active>')} …")
    return load_all_data()


# ---------------------------------------------------------------------------

def mode_generate_data(args) -> None:
    from basketball_ai.data.generator import generate_data
    print("=" * 60)
    print("  Generating synthetic basketball data …")
    print("=" * 60)
    generate_data(output_dir=args.data_dir, seed=args.seed)
    print("Done.")


def mode_train(args) -> None:
    import uuid
    from datetime import date, datetime, timezone
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.models.backtest import run_backtest
    print("=" * 60)
    print("  Training models …")
    print("=" * 60)
    data = _load_data(args)
    ensemble = EnsembleModel()
    metrics = ensemble.train(data)
    report = run_backtest(data, output_path=str(Path(args.model_dir) / "backtest_report.json"))
    if not report.get("valid"):
        raise RuntimeError("Backtest failed; candidate was not saved")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    ensemble.save(args.model_dir, {**metrics, "model_run_id":run_id, "model_version":"2.0.0", "feature_version":"forecast-t-plus-1-v1", "data_cutoff":str(date.today()), "backtest":report})
    print("\nAll models trained and saved.")


def mode_demo(args) -> None:
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine
    from basketball_ai.data.loader import _to_int

    print("=" * 60)
    print("  Basketball Performance AI – Demo")
    print("=" * 60)

    data     = _load_data(args)
    ensemble = EnsembleModel()

    if (Path(args.model_dir) / "performance_model.joblib").exists():
        ensemble.load(args.model_dir)
    else:
        print("[WARN] No trained models – training now (may take a minute) …")
        ensemble.train(data)
        ensemble.save(args.model_dir)

    engine      = WhatIfEngine(ensemble, data)
    players_df  = data["players"]
    teams_df    = data["teams"]
    team_dict   = data["team_dict"]

    # --- Demo 1: Performance comparison across teams --------------------
    sample_pos = ["PG", "SG/SF", "PF/C", "C"]
    chosen = []
    for pos in sample_pos:
        subset = players_df[
            players_df["position"].str.contains(pos.split("/")[0], na=False) &
            players_df["age"].between(22, 30)
        ]
        if not subset.empty:
            chosen.append(subset.sample(1, random_state=42).iloc[0])
    if not chosen:
        chosen = [players_df.sample(4, random_state=42).iloc[i] for i in range(4)]

    tier1_teams = teams_df[teams_df["league_tier"] == 1].head(20)

    print("\n─── Demo 1: Performance in different teams ──────────────────")
    for player_row in chosen[:2]:
        pid     = _to_int(player_row["id"])
        cur_tid = _to_int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
        compare_tids = [cur_tid] + tier1_teams.sample(4, random_state=42)["id"].tolist()
        compare_tids = list(dict.fromkeys(compare_tids))[:5]
        result  = engine.compare_scenarios(pid, compare_tids)
        print(f"\n  Player: {player_row['name']}  (Position: {player_row['position']}, Age: {player_row['age']})")
        print(f"  {'Team':<35} {'Rating':>8}  {'CI':>15}")
        print(f"  {'-'*62}")
        for s in result.scenarios:
            mark = " ◀ BEST" if s["team_id"] == result.best_scenario.get("team_id") else ""
            ci   = f"[{s['confidence_low']:.2f}–{s['confidence_high']:.2f}]"
            print(f"  {s['team_name']:<35} {s['rating']:>8.3f}  {ci:>15}{mark}")

    # --- Demo 2: Age trajectory ------------------------------------------
    print("\n─── Demo 2: Age trajectory ──────────────────────────────────")
    young = players_df[
        players_df["age"].between(19, 23) &
        players_df["position"].str.contains("G|SG|PG", na=False, regex=True)
    ]
    if young.empty:
        young = players_df[players_df["age"] <= 23]
    if not young.empty:
        yr  = young.sample(1, random_state=7).iloc[0]
        pid = _to_int(yr["id"])
        traj = engine.predict_age_trajectory(pid, age_range=(int(yr["age"]), min(int(yr["age"]) + 12, 40)))
        print(f"\n  Player: {yr['name']}  (age {yr['age']}, pos {yr['position']})")
        print(f"  {'Age':>4}  {'Season':>6}  {'Rating':>8}  {'CI':>15}")
        print(f"  {'-'*42}")
        for pt in traj[::2]:
            ci = f"[{pt.confidence_low:.2f}–{pt.confidence_high:.2f}]"
            print(f"  {pt.age:>4}  {pt.season:>6}  {pt.predicted_rating:>8.3f}  {ci:>15}")

    # --- Demo 3: Transfer impact -----------------------------------------
    print("\n─── Demo 3: Transfer impact simulation ─────────────────────")
    if chosen:
        player_row = chosen[0]
        pid      = _to_int(player_row["id"])
        from_tid = _to_int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
        to_t     = tier1_teams.sample(1, random_state=5).iloc[0]
        to_tid   = _to_int(to_t["id"])
        transfer = engine.simulate_transfer(pid, from_tid, to_tid)
        from_name = team_dict.get(from_tid, {}).get("name", f"Team {from_tid}")
        to_name   = team_dict.get(to_tid,   {}).get("name", f"Team {to_tid}")
        print(f"\n  Player  : {player_row['name']}  ({player_row['position']})")
        print(f"  From    : {from_name}")
        print(f"  To      : {to_name}")
        print(f"  Before  : {transfer.rating_before:.3f}")
        print(f"  After   : {transfer.rating_after:.3f}")
        print(f"  Delta   : {transfer.rating_delta:+.3f}")
        print(f"  Verdict : {transfer.recommendation}")

    # --- Demo 4: Peak prediction -----------------------------------------
    print("\n─── Demo 4: Career peak prediction ─────────────────────────")
    for player_row in chosen[:2]:
        pid  = _to_int(player_row["id"])
        peak = engine.predict_peak(pid)
        print(f"\n  {peak.player_name:<35}  age {peak.current_age}, pos {player_row['position']}")
        print(f"    Current rating  : {peak.current_rating:.3f}")
        print(f"    Predicted peak  : {peak.peak_rating:.3f}  at age {peak.peak_age}")
        print(f"    Peak window     : age {peak.peak_window[0]}–{peak.peak_window[1]}")
        print(f"    Seasons to peak : {peak.seasons_to_peak}")

    # --- Demo 5: Best team fit -------------------------------------------
    print("\n─── Demo 5: Best team fit ───────────────────────────────────")
    if chosen:
        player_row = chosen[0]
        pid  = _to_int(player_row["id"])
        fits = engine.best_team_fit(pid, top_n=5)
        print(f"\n  Best teams for {player_row['name']} ({player_row['position']}):")
        for f in fits:
            print(f"  #{f.rank}  {f.team_name:<35}  ({f.league_name})  →  {f.predicted_rating:.3f}")

    # --- Demo 6: What-if teammates ---------------------------------------
    print("\n─── Demo 6: What-If teammates ───────────────────────────────")
    if chosen:
        player_row = chosen[0]
        pid      = _to_int(player_row["id"])
        cur_tid  = _to_int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
        base     = engine.predict_in_team(pid, cur_tid)
        better   = engine.what_if_teammates(pid, cur_tid, 8.5)
        worse    = engine.what_if_teammates(pid, cur_tid, 5.5)
        print(f"\n  Player: {player_row['name']}")
        print(f"  With actual teammates  : {base.predicted_rating:.3f}")
        print(f"  With elite teammates   : {better.predicted_rating:.3f}  (avg=8.5)")
        print(f"  With weaker teammates  : {worse.predicted_rating:.3f}  (avg=5.5)")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


def mode_api(args) -> None:
    import os
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    # Pass source and paths to the API via environment variables
    os.environ.setdefault("DATA_SOURCE", "postgres")
    if args.database_profile:
        os.environ["DATABASE_PROFILE"] = args.database_profile
    os.environ.setdefault("DATA_DIR",    args.data_dir)
    os.environ.setdefault("MODEL_DIR",   args.model_dir)
    print(f"[API] Starting Basketball Performance AI on {args.host}:{args.port}")
    uvicorn.run("basketball_ai.api.main:app", host=args.host, port=args.port, reload=False)


def mode_export_wordpress(args) -> None:
    """Export per-player JSON files ready for WordPress import."""
    import json
    from datetime import datetime, timezone
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.scenarios.engine import WhatIfEngine
    from tqdm import tqdm

    print("=" * 60)
    print("  Exporting WordPress player cards …")
    print("=" * 60)

    data     = _load_data(args)
    ensemble = EnsembleModel()

    if (Path(args.model_dir) / "performance_model.joblib").exists():
        ensemble.load(args.model_dir)
    else:
        print("[WARN] No trained models – training now …")
        ensemble.train(data)
        ensemble.save(args.model_dir)

    engine      = WhatIfEngine(ensemble, data)
    players_df  = data["players"]
    team_dict   = data["team_dict"]
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    exported = 0
    for _, row in tqdm(players_df.iterrows(), total=len(players_df),
                       desc="Player cards", unit="player", dynamic_ncols=True):
        pid = int(row["id"])
        cur_tid = row.get("current_team_id")

        # D8 – anonymisation: replace real name with a token when requested
        display_name = f"Player_{pid}" if getattr(args, "anonymize", False) else str(row.get("name", ""))

        current_rating = confidence_low = confidence_high = None
        if cur_tid and int(cur_tid) in team_dict:
            try:
                pred            = engine.predict_in_team(pid, int(cur_tid))
                current_rating  = pred.predicted_rating
                confidence_low  = pred.confidence_low
                confidence_high = pred.confidence_high
            except Exception:
                pass

        peak_rating = peak_age = None
        try:
            peak        = engine.predict_peak(pid)
            peak_rating = peak.peak_rating
            peak_age    = peak.peak_age
        except Exception:
            pass

        top_teams = []
        try:
            for f in engine.best_team_fit(pid, top_n=3):
                top_teams.append({
                    "rank":             f.rank,
                    "team_name":        f.team_name,
                    "league_name":      f.league_name,
                    "predicted_rating": f.predicted_rating,
                })
        except Exception:
            pass

        card = {
            "player_id":       pid,
            "name":            display_name,
            "position":        str(row.get("position", "")),
            "age":             int(row.get("age", 0)),
            "current_team":    str(team_dict.get(int(cur_tid), {}).get("name", "—"))
                               if cur_tid else "—",
            "current_rating":  current_rating,
            "confidence_low":  confidence_low,
            "confidence_high": confidence_high,
            "peak_rating":     peak_rating,
            "peak_age":        peak_age,
            "top_teams":       top_teams,
            "generated_at":    generated_at,
        }
        out_file = out_dir / f"player_{pid}.json"
        out_file.write_text(json.dumps(card, indent=2, ensure_ascii=False))
        exported += 1

    # Also write a combined file for bulk import
    all_ids  = [int(r["id"]) for _, r in players_df.iterrows()]
    manifest = {"player_ids": all_ids, "count": exported, "generated_at": generated_at}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  Exported {exported} player cards to '{args.out_dir}/'")
    print(f"  Manifest: {args.out_dir}/manifest.json")


# ---------------------------------------------------------------------------


def mode_publish_batch(args) -> None:
    from basketball_ai.export.postgres_export import publish_current_team_forecasts
    print(f"[Batch] Published {publish_current_team_forecasts(args.model_dir)} forecasts")

def mode_backtest(args) -> None:
    from basketball_ai.models.backtest import run_backtest
    report = run_backtest(_load_data(args), output_path=str(Path(args.model_dir) / "backtest_report.json"))
    print(report)
    if not report.get("valid"):
        raise SystemExit(2)


# ---------------------------------------------------------------------------


def mode_validate_data(args) -> None:
    """Run data quality gates against loaded data."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path
    import pandas as pd

    data = _load_data(args)
    player_stats: pd.DataFrame = data.get("player_stats", pd.DataFrame())
    players: pd.DataFrame = data.get("players", pd.DataFrame())
    teams: pd.DataFrame = data.get("teams", pd.DataFrame())

    issues: list[dict] = []

    def _issue(table: str, col: str, check: str, pct: float, detail: str = "") -> None:
        issues.append({
            "table": table, "column": col,
            "check": check, "pct_affected": round(pct, 4),
            "detail": detail,
        })

    # --- NULL checks on critical columns ---------------------------------
    critical = {
        "player_stats": ["player_id", "season", "team_id", "points", "games_played", "rating"],
        "players":      ["id", "name", "position"],
        "teams":        ["id", "name", "league_id"],
    }
    for tname, cols in critical.items():
        df = data.get(tname, pd.DataFrame())
        for col in cols:
            if col not in df.columns:
                continue
            null_pct = df[col].isna().mean()
            if null_pct > 0.05:
                _issue(tname, col, "null_rate", null_pct, f"{null_pct:.1%} NULLs in critical column")

    # --- Consistency: Fg2m <= Fg2a -----------------------------------------
    for col_made, col_att in [("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        if col_made in player_stats.columns and col_att in player_stats.columns:
            mask = player_stats[col_made] > player_stats[col_att]
            bad_pct = mask.mean()
            if bad_pct > 0:
                _issue("player_stats", col_made, "made_gt_attempted", bad_pct)

    # --- Pts approximation check ----------------------------------------
    pt_cols = {"pts", "fg2m", "fg3m", "ftm"}
    if pt_cols.issubset(player_stats.columns):
        expected = 2 * player_stats["fg2m"] + 3 * player_stats["fg3m"] + player_stats["ftm"]
        diff = (player_stats["pts"] - expected).abs()
        bad_pct = (diff > 5).mean()
        if bad_pct > 0.1:
            _issue("player_stats", "pts", "pts_approx_mismatch", bad_pct)

    # --- Rating range check -----------------------------------------------
    if "rating" in player_stats.columns:
        out_range = ((player_stats["rating"] < 0) | (player_stats["rating"] > 10)).mean()
        if out_range > 0:
            _issue("player_stats", "rating", "rating_out_of_range", out_range)

    # --- Report -----------------------------------------------------------
    report_dir = _Path("data/quality_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S")
    report_path = report_dir / f"quality_{ts}.json"
    report = {
        "generated_at": _dt.now(_tz.utc).isoformat(),
        "rows_player_stats": len(player_stats),
        "rows_players": len(players),
        "rows_teams": len(teams),
        "issues": issues,
        "status": "PASS" if not issues else "WARN",
    }
    report_path.write_text(_json.dumps(report, indent=2))
    print(f"\n[validate-data] {report['status']}: {len(issues)} issue(s) found.")
    for iss in issues:
        print(f"  ⚠ {iss['table']}.{iss['column']} [{iss['check']}] {iss['detail']}")
    print(f"  Report saved to: {report_path}")


def mode_backup(args) -> None:
    """Create a timestamped ZIP backup of all critical local state."""
    import hashlib as _hashlib
    import zipfile as _zipfile
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = _Path("backups")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"backup_{ts}.zip"

    targets = [
        _Path("models_saved"),
        _Path("roles.json"),
        _Path("sessions.json"),
        _Path("data/ingestion.db"),
        _Path("data/name_resolution_cache.json"),
    ]

    with _zipfile.ZipFile(zip_path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
        for target in targets:
            if not target.exists():
                continue
            if target.is_dir():
                for f in target.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(_Path(".")))
            else:
                zf.write(target, target)

    sha256 = _hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sums_path = zip_path.with_suffix(".sha256")
    sums_path.write_text(f"{sha256}  {zip_path.name}\n")

    print(f"[backup] Created: {zip_path}  ({zip_path.stat().st_size:,} bytes)")
    print(f"[backup] SHA256:  {sha256}")


# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    dispatch = {
        "generate-data":    mode_generate_data,
        "train":            mode_train,
        "demo":             mode_demo,
        "api":              mode_api,
        "publish-batch":    mode_publish_batch,
        "validate-data":    mode_validate_data,
        "backup":           mode_backup,
        "backtest":         mode_backtest,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

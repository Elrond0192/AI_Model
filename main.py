"""Root CLI entry-point for the Basketball Performance AI system.

Usage:
    python main.py --mode generate-data
    python main.py --mode train
    python main.py --mode demo
    python main.py --mode api [--host HOST] [--port PORT]
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
        "--mode", choices=["generate-data", "train", "demo", "api"], required=True,
    )
    parser.add_argument("--data-dir",  default="data/sample",   help="Data directory")
    parser.add_argument("--model-dir", default="models_saved",  help="Model save directory")
    parser.add_argument("--host",      default="0.0.0.0",       help="API host")
    parser.add_argument("--port",      type=int, default=8000,  help="API port")
    parser.add_argument("--seed",      type=int, default=42,    help="RNG seed")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------

def mode_generate_data(args) -> None:
    from src.data.generator import generate_data
    print("=" * 60)
    print("  Generating synthetic basketball data …")
    print("=" * 60)
    generate_data(output_dir=args.data_dir, seed=args.seed)
    print("Done.")


def mode_train(args) -> None:
    from src.data.loader import load_all_data, data_exists
    from src.models.ensemble import EnsembleModel
    print("=" * 60)
    print("  Training models …")
    print("=" * 60)
    if not data_exists(args.data_dir):
        print(f"[ERROR] Data not found in '{args.data_dir}'. Run --mode generate-data first.")
        sys.exit(1)
    data = load_all_data(args.data_dir)
    ensemble = EnsembleModel()
    ensemble.train(data)
    ensemble.save(args.model_dir)
    print("\nAll models trained and saved.")


def mode_demo(args) -> None:
    from src.data.loader import load_all_data, data_exists
    from src.models.ensemble import EnsembleModel
    from src.scenarios.engine import WhatIfEngine
    from src.utils.helpers import format_prediction_output

    print("=" * 60)
    print("  Basketball Performance AI – Demo")
    print("=" * 60)

    if not data_exists(args.data_dir):
        print(f"[ERROR] Data not found in '{args.data_dir}'. Run --mode generate-data first.")
        sys.exit(1)

    data     = load_all_data(args.data_dir)
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
    league_dict = data["league_dict"]

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
        pid     = int(player_row["id"])
        cur_tid = int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
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
        pid = int(yr["id"])
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
        pid      = int(player_row["id"])
        from_tid = int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
        to_t     = tier1_teams.sample(1, random_state=5).iloc[0]
        to_tid   = int(to_t["id"])
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
        pid  = int(player_row["id"])
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
        pid  = int(player_row["id"])
        fits = engine.best_team_fit(pid, top_n=5)
        print(f"\n  Best teams for {player_row['name']} ({player_row['position']}):")
        for f in fits:
            print(f"  #{f.rank}  {f.team_name:<35}  ({f.league_name})  →  {f.predicted_rating:.3f}")

    # --- Demo 6: What-if teammates ---------------------------------------
    print("\n─── Demo 6: What-If teammates ───────────────────────────────")
    if chosen:
        player_row = chosen[0]
        pid      = int(player_row["id"])
        cur_tid  = int(player_row["current_team_id"]) if player_row.get("current_team_id") else 1
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
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    print(f"[API] Starting Basketball Performance AI on {args.host}:{args.port}")
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=False)


# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    dispatch = {
        "generate-data": mode_generate_data,
        "train":         mode_train,
        "demo":          mode_demo,
        "api":           mode_api,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

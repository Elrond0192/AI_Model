"""Root CLI entry-point for the Football Performance AI system.

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


# ─── CLI argument parsing ────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Football Performance AI – CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["generate-data", "train", "demo", "api"],
        required=True,
        help="Run mode",
    )
    parser.add_argument("--data-dir", default="data/sample", help="Data directory")
    parser.add_argument("--model-dir", default="models_saved", help="Model save directory")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for data generation")
    return parser.parse_args(argv)


# ─── Modes ───────────────────────────────────────────────────────────────────

def mode_generate_data(args) -> None:
    from src.data.generator import generate_data
    print("=" * 60)
    print("  Generating synthetic football data …")
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
    print("  Football Performance AI – Demo")
    print("=" * 60)

    if not data_exists(args.data_dir):
        print(f"[ERROR] Data not found in '{args.data_dir}'. Run --mode generate-data first.")
        sys.exit(1)

    data = load_all_data(args.data_dir)
    ensemble = EnsembleModel()

    model_perf = Path(args.model_dir) / "performance_model.joblib"
    if model_perf.exists():
        ensemble.load(args.model_dir)
    else:
        print("[WARN] No trained models found – training now (this may take a minute)…")
        ensemble.train(data)
        ensemble.save(args.model_dir)

    engine = WhatIfEngine(ensemble, data)

    players_df = data["players"]
    teams_df = data["teams"]
    league_dict = data["league_dict"]
    team_dict = data["team_dict"]

    # Pick a few interesting players from different positions
    sample_positions = ["ST", "CM", "CB", "W"]
    chosen_players = []
    for pos in sample_positions:
        pos_players = players_df[
            (players_df["position"] == pos) & (players_df["age"].between(22, 30))
        ]
        if not pos_players.empty:
            chosen_players.append(pos_players.sample(1, random_state=42).iloc[0])

    if not chosen_players:
        chosen_players = [players_df.sample(4, random_state=42).iloc[i] for i in range(4)]

    # Demo 1: Predict in current team + two alternative tier-1 teams
    tier1_teams = teams_df[teams_df["league_tier"] == 1].head(20)

    print("\n─── Demo 1: Performance in different teams ──────────────────")
    for player_row in chosen_players[:2]:
        pid = int(player_row["id"])
        pname = player_row["name"]
        pos = player_row["position"]
        cur_tid = int(player_row["current_team_id"])

        # Compare: current team + 4 tier-1 teams
        compare_tids = [cur_tid] + tier1_teams.sample(4, random_state=42)["id"].tolist()
        compare_tids = list(dict.fromkeys(compare_tids))[:5]

        result = engine.compare_scenarios(pid, compare_tids)
        print(f"\n  Player: {pname}  (Position: {pos}, Age: {player_row['age']})")
        print(f"  {'Team':<35} {'Rating':>8}  {'CI':>15}")
        print(f"  {'-'*60}")
        for s in result.scenarios:
            tname = s["team_name"]
            r = s["rating"]
            ci = f"[{s['confidence_low']:.2f}–{s['confidence_high']:.2f}]"
            mark = " ◀ BEST" if s["team_id"] == result.best_scenario["team_id"] else ""
            print(f"  {tname:<35} {r:>8.3f}  {ci:>15}{mark}")

    # Demo 2: Age trajectory
    print("\n─── Demo 2: Age trajectory ──────────────────────────────────")
    young_player = players_df[
        (players_df["age"] <= 22) & (players_df["position"].isin(["ST", "W", "AM"]))
    ]
    if not young_player.empty:
        yr = young_player.sample(1, random_state=7).iloc[0]
        pid = int(yr["id"])
        traj = engine.predict_age_trajectory(pid, age_range=(yr["age"], min(yr["age"] + 12, 40)))
        print(f"\n  Player: {yr['name']}  (current age {yr['age']}, position {yr['position']})")
        print(f"  {'Age':>4}  {'Season':>6}  {'Rating':>8}  {'CI':>15}")
        print(f"  {'-'*40}")
        for pt in traj[::2]:  # every 2 years for brevity
            ci = f"[{pt.confidence_low:.2f}–{pt.confidence_high:.2f}]"
            print(f"  {pt.age:>4}  {pt.season:>6}  {pt.predicted_rating:>8.3f}  {ci:>15}")

    # Demo 3: Transfer impact
    print("\n─── Demo 3: Transfer impact simulation ─────────────────────")
    if len(chosen_players) >= 1:
        player_row = chosen_players[0]
        pid = int(player_row["id"])
        from_tid = int(player_row["current_team_id"])
        to_t = tier1_teams.sample(1, random_state=5).iloc[0]
        to_tid = int(to_t["id"])

        transfer = engine.simulate_transfer(pid, from_tid, to_tid)
        from_name = team_dict.get(from_tid, {}).get("name", f"Team {from_tid}")
        to_name = team_dict.get(to_tid, {}).get("name", f"Team {to_tid}")
        print(f"\n  Player   : {player_row['name']}")
        print(f"  From     : {from_name}")
        print(f"  To       : {to_name}")
        print(f"  Before   : {transfer.rating_before:.3f}")
        print(f"  After    : {transfer.rating_after:.3f}")
        print(f"  Delta    : {transfer.rating_delta:+.3f}")
        print(f"  Verdict  : {transfer.recommendation}")

    # Demo 4: Peak prediction
    print("\n─── Demo 4: Career peak prediction ─────────────────────────")
    for player_row in chosen_players[:2]:
        pid = int(player_row["id"])
        peak = engine.predict_peak(pid)
        print(f"\n  {peak.player_name:<35}  age {peak.current_age}")
        print(f"    Current rating : {peak.current_rating:.3f}")
        print(f"    Predicted peak : {peak.peak_rating:.3f}  at age {peak.peak_age}")
        print(f"    Peak window    : age {peak.peak_window[0]}–{peak.peak_window[1]}")
        print(f"    Seasons to peak: {peak.seasons_to_peak}")

    # Demo 5: Best team fit
    print("\n─── Demo 5: Best team fit ───────────────────────────────────")
    if chosen_players:
        player_row = chosen_players[0]
        pid = int(player_row["id"])
        fits = engine.best_team_fit(pid, top_n=5)
        print(f"\n  Best teams for {player_row['name']}:")
        for f in fits:
            print(f"  #{f.rank}  {f.team_name:<35}  ({f.league_name})  →  {f.predicted_rating:.3f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


def mode_api(args) -> None:
    import uvicorn
    from src.api.main import app
    print(f"Starting API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ─── Entry-point ─────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    dispatch = {
        "generate-data": mode_generate_data,
        "train": mode_train,
        "demo": mode_demo,
        "api": mode_api,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

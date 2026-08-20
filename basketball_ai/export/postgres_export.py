"""Publish bounded current-team forecasts from the promoted production model."""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from basketball_ai.data.postgres_loader import get_engine, load_all_data


def publish_current_team_forecasts(model_dir: str) -> int:
    from basketball_ai.models.ensemble import EnsembleModel
    from basketball_ai.models.production_training import build_historical_snapshot
    from basketball_ai.scenarios.engine import WhatIfEngine

    model_path = Path(model_dir)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Promoted model metadata not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ("model_run_id", "model_version", "feature_version", "data_cutoff")
    if any(not metadata.get(key) for key in required):
        raise RuntimeError("Model metadata is not contract-v2 compatible")

    data = load_all_data()
    model = EnsembleModel()
    model.load(str(model_path))
    latest_season = int(
        max(int(str(value).split("-")[0]) for value in data["player_stats"]["season"].dropna())
    )
    snapshot = build_historical_snapshot(data, latest_season)
    inference = WhatIfEngine(model, snapshot)

    rows = []
    for player_id, player in snapshot["player_dict"].items():
        team_id = player.get("current_team_id")
        player_gid = player.get("global_id")
        team = snapshot["team_dict"].get(team_id)
        if not team_id or not player_gid or not team or not team.get("global_id"):
            continue
        result = inference.predict_in_team(
            int(player_id), int(team_id), latest_season, competition="RS"
        )
        rows.append(
            {
                "model_run_id": metadata["model_run_id"],
                "player_global_id": str(player_gid),
                "team_global_id": str(team["global_id"]),
                "league": str(team.get("league_key", team.get("league_id", ""))),
                "season": latest_season,
                "competition": "RS",
                "target_season": latest_season + 1,
                "predicted_rating": result.predicted_rating,
                "confidence_low": result.confidence_low,
                "confidence_high": result.confidence_high,
                "payload": json.dumps({"method": "forecast_t_plus_1", "status": "production"}),
            }
        )

    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO ai.model_runs(model_run_id,model_version,feature_version,data_cutoff,status,metrics) "
                "VALUES (:model_run_id,:model_version,:feature_version,:data_cutoff,'production',CAST(:metrics AS jsonb)) "
                "ON CONFLICT(model_run_id) DO UPDATE SET status=EXCLUDED.status, metrics=EXCLUDED.metrics"
            ),
            {**metadata, "metrics": json.dumps(metadata.get("backtest", {}))},
        )
        statement = text(
            "INSERT INTO ai.player_forecasts(model_run_id,player_global_id,team_global_id,league,season,competition,target_season,predicted_rating,confidence_low,confidence_high,payload) "
            "VALUES (:model_run_id,:player_global_id,:team_global_id,:league,:season,:competition,:target_season,:predicted_rating,:confidence_low,:confidence_high,CAST(:payload AS jsonb)) "
            "ON CONFLICT(model_run_id,player_global_id,team_global_id,league,season,competition) DO UPDATE SET "
            "predicted_rating=EXCLUDED.predicted_rating,confidence_low=EXCLUDED.confidence_low,confidence_high=EXCLUDED.confidence_high,payload=EXCLUDED.payload,created_at=now()"
        )
        if rows:
            connection.execute(statement, rows)
    return len(rows)

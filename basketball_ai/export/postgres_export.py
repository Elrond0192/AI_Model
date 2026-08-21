"""Publish bounded current-context forecasts from the promoted model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from basketball_ai.data.postgres_loader import get_engine, load_all_data
from basketball_ai.models.competition_training import normalize_competition


def publish_current_team_forecasts(model_dir: str) -> int:
    from basketball_ai.models.strict_production import (
        StrictProductionEnsembleModel,
        StrictWhatIfEngine,
        build_historical_snapshot,
    )

    model_path = Path(model_dir)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Promoted model metadata not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ("model_run_id", "model_version", "feature_version", "data_cutoff")
    if any(not metadata.get(key) for key in required):
        raise RuntimeError("Model metadata is not contract-v2 compatible")

    data = load_all_data()
    model = StrictProductionEnsembleModel()
    model.load(str(model_path))
    latest_season = int(
        max(int(str(value).split("-")[0]) for value in data["player_stats"]["season"].dropna())
    )
    snapshot = build_historical_snapshot(data, latest_season)
    inference = StrictWhatIfEngine(model, snapshot)

    source = snapshot["player_stats"].copy()
    source["_season_year"] = pd.to_numeric(source["season"], errors="coerce")
    source = source[source["_season_year"] == latest_season]
    source = source.dropna(subset=["player_id", "team_id", "league_id", "competition"])

    rows = []
    seen: set[tuple[int, int, int, str]] = set()
    for _, context in source.iterrows():
        player_id = int(context["player_id"])
        team_id = int(context["team_id"])
        league_id = int(context["league_id"])
        competition = normalize_competition(context["competition"])
        key = (player_id, team_id, league_id, competition)
        if key in seen:
            continue
        seen.add(key)

        player = snapshot["player_dict"].get(player_id)
        team = data["team_dict"].get(team_id)
        if not player or not team or not player.get("global_id") or not team.get("global_id"):
            continue
        if competition not in model._competition_encoding_state:
            continue
        try:
            result = inference.predict_in_team(
                player_id,
                team_id,
                latest_season,
                competition=competition,
                league_id=league_id,
            )
        except (ValueError, RuntimeError):
            continue

        rows.append(
            {
                "model_run_id": metadata["model_run_id"],
                "player_global_id": str(player["global_id"]),
                "team_global_id": str(team["global_id"]),
                "league": str(context.get("league_key", league_id)),
                "season": latest_season,
                "competition": competition,
                "target_season": latest_season + 1,
                "predicted_rating": result.predicted_rating,
                "confidence_low": result.confidence_low,
                "confidence_high": result.confidence_high,
                "payload": json.dumps(
                    {
                        "method": "forecast_t_plus_1",
                        "status": "production",
                        "context": "isolated_league_competition",
                    }
                ),
            }
        )

    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "AI"."ModelRuns"(model_run_id,model_version,feature_version,data_cutoff,status,metrics) '
                "VALUES (:model_run_id,:model_version,:feature_version,:data_cutoff,'production',CAST(:metrics AS jsonb)) "
                "ON CONFLICT(model_run_id) DO UPDATE SET status=EXCLUDED.status, metrics=EXCLUDED.metrics"
            ),
            {**metadata, "metrics": json.dumps(metadata.get("backtest", {}))},
        )
        statement = text(
            'INSERT INTO "AI"."PlayerForecasts"(model_run_id,player_global_id,team_global_id,league,season,competition,target_season,predicted_rating,confidence_low,confidence_high,payload) '
            "VALUES (:model_run_id,:player_global_id,:team_global_id,:league,:season,:competition,:target_season,:predicted_rating,:confidence_low,:confidence_high,CAST(:payload AS jsonb)) "
            "ON CONFLICT(model_run_id,player_global_id,team_global_id,league,season,competition) DO UPDATE SET "
            "predicted_rating=EXCLUDED.predicted_rating,confidence_low=EXCLUDED.confidence_low,confidence_high=EXCLUDED.confidence_high,payload=EXCLUDED.payload,created_at=now()"
        )
        if rows:
            connection.execute(statement, rows)
    return len(rows)

"""Tests for the deterministic contextual BB-Rating engine and API."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from basketball_ai.bb_rating.calibration import build_calibration_report
from basketball_ai.bb_rating.engine import BBRatingEngine
from basketball_ai.api.routes.bb_rating_v2 import router


def make_stats() -> pd.DataFrame:
    rows = []
    for index in range(60):
        x = float(index)
        rows.append(
            {
                "player_id": index + 1,
                "player_global_id": f"G{index + 1}",
                "league_key": "ITA1",
                "season": 2025,
                "competition": "RS",
                "position": "PG",
                "age": 25,
                "ruolo_combinato": "Creator",
                "points": 10.0 + x / 3.0,
                "assists": 2.0 + x / 20.0,
                "rebounds": 3.0 + x / 15.0,
                "fg_pct": 0.40 + x / 300.0,
                "three_point_pct": 0.25 + x / 400.0,
                "ft_pct": 0.65 + x / 500.0,
                "minutes_per_game": 15.0 + x / 6.0,
                "games_played": 20 + int(x % 10),
                "starter_pct": x / 59.0,
                "plus_minus": -5.0 + x / 6.0,
                "pts_per_40": 15.0 + x / 4.0,
                "ast_per_40": 4.0 + x / 10.0,
                "three_par": 0.25 + x / 500.0,
                "true_usg_pct": 0.18 + x / 350.0,
                "clutch_ts_pct": 0.45 + x / 800.0,
                "clutch_net_rtg": -8.0 + x / 4.0,
                "ortg_diff": -6.0 + x / 3.0,
                "net_rtg": -3.0 + x / 2.0,
                "raptor_total": x / 5.0,
                "lebron_total": x / 6.0,
                "vorp": x / 12.0,
                "per": 10.0 + x / 8.0,
                "ts_pct": 0.48 + x / 1200.0,
                "scoring_efficiency": 0.75 + x / 100.0,
                "ast_pct": 0.10 + x / 300.0,
                "tov_pct": 0.18 - x / 1000.0,
                "raptor_def": -3.0 + x / 10.0,
                "dbpm": -2.0 + x / 20.0,
                "net_rtg_diff": -10.0 + x / 2.0,
                "stl_pct": 0.01 + x / 3000.0,
                "blk_pct": 0.01 + x / 6000.0,
                "reb_pct": 0.08 + x / 500.0,
                "hustle_index": 30.0 + x,
                "foul_drawing_rate": 0.05 + x / 1000.0,
                "usg_pct": 0.18 + x / 300.0,
            }
        )
    return pd.DataFrame(rows)


def make_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 61)),
            "global_id": [f"G{i}" for i in range(1, 61)],
            "name": [f"Player {i}" for i in range(1, 61)],
            "age": [25] * 60,
            "position": ["PG"] * 60,
        }
    )


def test_bb_rating_is_deterministic_and_contextual():
    engine = BBRatingEngine(make_stats(), make_players())
    result = engine.rate_player("G60", league="ITA1", season=2025, phase="RS")

    assert 1 <= result.score <= 100
    assert result.score >= 90
    assert result.peer_group["definition"] == "position+age+role"
    assert result.peer_group["sample_size"] == 60
    assert result.quality == "high"
    assert 0.0 <= result.metric_coverage <= 1.0
    assert "RAPTOR" in result.metrics
    assert result.metrics["RAPTOR"].percentile is not None
    payload = result.to_dict()
    assert payload["metrics"]["RAPTOR"]["meaning"]
    assert payload["metrics"]["RAPTOR"]["interpretation"]
    assert result.metrics["TOV%"].direction == "lower_better"
    payload_metrics = result.to_dict()["metrics"]
    for key in ("POINTS", "FG%", "MINUTES", "CLUTCH_TS%", "NET_RTG"):
        assert key in payload_metrics
        assert payload_metrics[key]["interpretation"]


def test_position_role_fallback_precedes_position_age_when_role_group_is_large():
    stats = make_stats().copy()
    stats.loc[:14, "age"] = 25
    stats.loc[15:, "age"] = 30

    engine = BBRatingEngine(stats, make_players())
    result = engine.rate_player("G1", league="ITA1", season=2025, phase="RS")

    assert result.peer_group["definition"] == "position+role"
    assert result.peer_group["sample_size"] == 60


def test_usg_is_explanatory_but_does_not_change_score():
    stats = make_stats()
    engine_a = BBRatingEngine(stats, make_players())
    score_a = engine_a.rate_player("G60", league="ITA1", season=2025, phase="RS")

    modified = stats.copy()
    modified.loc[modified["player_global_id"] == "G60", "usg_pct"] = 0.60
    engine_b = BBRatingEngine(modified, make_players())
    score_b = engine_b.rate_player("G60", league="ITA1", season=2025, phase="RS")

    assert score_b.score == score_a.score
    assert score_b.metrics["USG%"].percentile is not None
    assert "USG%" in score_b.explanation
    assert score_b.to_dict()["metrics"]["USG%"]["interpretation"]


def test_lower_turnover_rate_increases_metric_percentile():
    engine = BBRatingEngine(make_stats(), make_players())
    result = engine.rate_player("G1", league="ITA1", season=2025, phase="RS")
    # G1 has the worst raw TOV%, but lower is better, so its contextual
    # percentile must be low and the best turnover player must be higher.
    best = engine.rate_player("G60", league="ITA1", season=2025, phase="RS")
    assert best.metrics["TOV%"].percentile > result.metrics["TOV%"].percentile


def test_missing_metrics_are_reported_and_weight_is_renormalised():
    stats = make_stats().drop(columns=["vorp", "raptor_def", "dbpm"])
    engine = BBRatingEngine(stats, make_players())
    result = engine.rate_player("G60", league="ITA1", season=2025, phase="RS")

    assert result.metrics["VORP"].percentile is None
    assert result.metrics["RAPTOR_DEF"].percentile is None
    assert result.metrics["DBPM"].percentile is None
    assert result.metric_coverage < 1.0
    assert result.score >= 1



def test_calibration_uses_position_role_fallback():
    stats = make_stats().copy()
    stats.loc[:14, "age"] = 25
    stats.loc[15:, "age"] = 30

    report = build_calibration_report(
        {"player_stats": stats, "players": make_players(), "source_contract": "test"}
    )

    peer_sources = {item["peer_source"] for item in report["peer_sources"]}
    assert peer_sources == {"position+role"}
    assert report["validation_signals"]["role_peer_share"] == pytest.approx(1.0)
    assert report["validation_signals"]["peer_source_shares"]["position+role"] == pytest.approx(1.0)


def test_calibration_warns_on_constant_scoring_metric():
    stats = make_stats()
    stats["net_rtg_diff"] = 0.0
    report = build_calibration_report(
        {"player_stats": stats, "players": make_players(), "source_contract": "test"}
    )

    assert "NET_RTG_DIFF" in report["validation_signals"]["constant_scoring_metrics"]
    assert any("NET_RTG_DIFF" in warning for warning in report["warnings"])


def test_missing_context_raises():
    engine = BBRatingEngine(make_stats(), make_players())
    with pytest.raises(ValueError, match="No BB-Rating context"):
        engine.rate_player("G60", league="ESP1", season=2025, phase="RS")


def test_api_player_endpoint():
    app = FastAPI()
    app.state.bb_rating_engine = BBRatingEngine(make_stats(), make_players())
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/api/v2/bb-rating/player",
        json={
            "player_global_id": "G60",
            "league": "ITA1",
            "season": "2025-26",
            "phase": "RS",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 1 <= payload["bb_rating"] <= 100
    assert payload["bb_rating_version"] == "1.2"
    assert payload["peer_group"]["definition"] == "position+age+role"
    assert payload["metrics"]["USG%"]["percentile"] is not None


def test_bb_rating_calibration_reuses_peer_and_percentile_contract():
    stats = pd.concat(
        [
            make_stats(),
            make_stats().assign(
                season=2024,
                player_global_id=lambda frame: frame["player_global_id"] + "-2024",
            ),
        ],
        ignore_index=True,
    )
    report = build_calibration_report(
        {"player_stats": stats, "players": make_players(), "source_contract": "test"}
    )

    assert report["calibration_version"] == "1.2"
    assert report["bb_rating_version"] == "1.2"
    assert report["dataset"]["rows"] == 120
    assert report["score_distribution"]["n"] == 120
    assert report["validation_signals"]["registry_columns_ok"] is True
    assert report["validation_signals"]["scoring_metrics_present"] is True
    assert report["validation_signals"]["explainable_metric_count"] == len(
        report["registry_audit"]
    )
    assert report["stability"]["n_pairs"] == 0

    metric = next(item for item in report["metrics"] if item["metric"] == "RAPTOR")
    assert metric["column_exists"] is True
    assert metric["semantic_source_matches"] is True
    assert metric["n_available"] == 120

    peer_sources = {item["peer_source"] for item in report["peer_sources"]}
    assert peer_sources == {"position+age+role"}

    role_diag = report["role_peer_population"]
    assert role_diag["rows_with_role_and_age"] == 120
    assert role_diag["thresholds"]["25"]["rows_eligible"] == 120
    assert report["validation_signals"]["role_peer_share"] == pytest.approx(1.0)
    assert report["validation_signals"]["peer_source_shares"]["position+age+role"] == pytest.approx(1.0)
    assert report["validation_signals"]["scoring_registry_ok"] is True
    assert report["validation_signals"]["explanation_catalog_ok"] is True

    # Explicitly verify that the lower-is-better turnover signal is preserved.
    tov = next(item for item in report["metrics"] if item["metric"] == "TOV%")
    assert tov["direction"] == "lower_better"

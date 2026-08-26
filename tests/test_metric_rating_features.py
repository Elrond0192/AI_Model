"""Tests for FASE H — rating feature provider + training wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from basketball_ai.metric_rating.distribution import build_distribution
from basketball_ai.metric_rating.features import (
    add_rating_features,
    build_distribution_lookup,
    player_rating_snapshot,
)
from basketball_ai.metric_rating.rating import MetricRatingEngine
from basketball_ai.models.production_training import SeasonAheadPerformanceModel

CANONICAL_COLUMNS = [
    "league_key", "season", "competition", "player_id", "player_global_id",
    "player_name", "position", "games_played", "minutes_per_game",
    "raptor_total", "lebron_total", "vorp", "games_started", "starter_pct",
]


def make_frame() -> pd.DataFrame:
    rows = []
    for league in ("ITA1", "ESP1"):
        for index in range(60):
            rows.append(
                {
                    "league_key": league,
                    "season": 2025,
                    "competition": "RS",
                    "player_id": f"{league}-{index}",
                    "player_global_id": f"G{league}{index}",
                    "player_name": f"P{league}{index}",
                    "position": "PG",
                    "games_played": 25,
                    "minutes_per_game": 20.0,
                    "raptor_total": float(index - 30),
                    "lebron_total": float(index - 30) / 2.0,
                    "vorp": 0.0 if index % 10 == 0 else float(index - 30),
                    "games_started": 20,
                    "starter_pct": 0.8,
                }
            )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def make_distribution_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for metric, source in (("RAPTOR", "raptor_total"), ("LEBRON", "lebron_total"),
                           ("VORP", "vorp")):
        for (league, season, comp), group in frame.groupby(
            ["league_key", "season", "competition"]
        ):
            values = pd.to_numeric(group[source], errors="coerce").dropna()
            if values.empty:
                continue
            dist = build_distribution(values, distribution_version="1.0")
            summary = dist.summary()
            rows.append(
                {
                    "metric": metric,
                    "league_key": league,
                    "season": int(season),
                    "competition": comp,
                    **{
                        key: summary[key]
                        for key in (
                            "sample_size", "min", "p10", "p25", "p40", "p50",
                            "p60", "p75", "p90", "p97_5", "max", "mean",
                            "stddev", "quality", "population_source",
                            "population_type", "distribution_version",
                        )
                    },
                }
            )
    return rows


class TestVectorizedEquivalence:
    def test_matches_scalar_engine(self):
        frame = make_frame()
        dist_rows = make_distribution_rows(frame)
        augmented, added = add_rating_features(frame, dist_rows)
        assert set(added) == {
            "RAPTOR_value", "RAPTOR_percentile", "RAPTOR_zscore", "RAPTOR_tier",
            "LEBRON_value", "LEBRON_percentile", "LEBRON_zscore", "LEBRON_tier",
            "VORP_value", "VORP_percentile", "VORP_zscore", "VORP_tier",
        }
        lookup = build_distribution_lookup(dist_rows)
        engine = MetricRatingEngine()
        for metric, source in (("RAPTOR", "raptor_total"),
                               ("LEBRON", "lebron_total"), ("VORP", "vorp")):
            for _, row in frame.iterrows():
                value = row[source]
                if pd.isna(value):
                    continue
                dist = lookup[(metric, row["league_key"], int(row["season"]),
                               row["competition"])]
                rating = engine.rate_from_distribution(metric, float(value), dist)
                pct = augmented.loc[row.name, f"{metric}_percentile"]
                z = augmented.loc[row.name, f"{metric}_zscore"]
                tier = augmented.loc[row.name, f"{metric}_tier"]
                assert pct == pytest.approx(rating.percentile, abs=1e-12)
                assert z == pytest.approx(rating.zscore, abs=1e-12)
                assert tier == pytest.approx(float(rating.tier), abs=1e-12)

    def test_value_keeps_raw_column(self):
        frame = make_frame()
        augmented, _ = add_rating_features(frame, make_distribution_rows(frame))
        assert (augmented["RAPTOR_value"] == frame["raptor_total"]).all()
        # original column untouched
        assert "raptor_total" in augmented.columns

    def test_no_distribution_yields_nan(self):
        frame = make_frame()
        # only VORP distributions for ITA1: RAPTOR/LEBRON and ESP1 have none
        dist_rows = [
            row for row in make_distribution_rows(frame)
            if row["metric"] == "VORP" and row["league_key"] == "ITA1"
        ]
        augmented, _ = add_rating_features(frame, dist_rows)
        ita = frame["league_key"] == "ITA1"
        esp = frame["league_key"] == "ESP1"
        assert augmented.loc[ita, "VORP_percentile"].notna().all()
        assert augmented.loc[esp, "VORP_percentile"].isna().all()
        assert augmented["RAPTOR_percentile"].isna().all()
        assert augmented.loc[esp, "VORP_tier"].isna().all()
        assert augmented.loc[ita, "VORP_tier"].notna().all()

    def test_stddev_zero_zscore_nan(self):
        frame = make_frame()
        # degenerate context: ALL ITA1 VORP values equal -> stddev zero
        frame.loc[frame["league_key"] == "ITA1", "vorp"] = 5.0
        dist_rows = make_distribution_rows(frame)
        augmented, _ = add_rating_features(frame, dist_rows)
        ita = frame["league_key"] == "ITA1"
        assert augmented.loc[ita, "VORP_zscore"].isna().all()
        assert augmented.loc[ita, "VORP_percentile"].notna().all()


class TestPlayerRatingSnapshot:
    def test_snapshot(self):
        frame = make_frame()
        dist_rows = make_distribution_rows(frame)
        snapshot = player_rating_snapshot(
            frame, dist_rows,
            player_global_id="GITA10", league="ITA1", season=2025, phase="RS",
        )
        assert "RAPTOR" in snapshot and "VORP" in snapshot
        rap = snapshot["RAPTOR"]
        # "GITA10" = league ITA1 + index 0 -> raptor_total = 0 - 30
        assert rap["value"] == pytest.approx(-30.0)
        assert 0.0 <= rap["percentile"] <= 1.0
        assert rap["tier"] >= 1 and rap["label"]
        assert rap["quality"] == "low"  # 60 qualified players -> 50-99 band
        assert rap["distribution_version"] == "1.0"

    def test_snapshot_missing_context_reports_raw(self):
        frame = make_frame()
        snapshot = player_rating_snapshot(
            frame, [], player_global_id="GITA10", league="ITA1",
            season=2025, phase="RS",
        )
        assert snapshot["RAPTOR"]["value"] == pytest.approx(-30.0)
        assert snapshot["RAPTOR"]["percentile"] is None


class TestTrainingWiring:
    def _augmented_tiny_data(self, tiny_data):
        data = {}
        for key, value in tiny_data.items():
            data[key] = value.copy() if isinstance(value, pd.DataFrame) else value
        ps = data["player_stats"].copy()
        ps["league_key"] = ps["league_id"].map({1: "ITA1", 2: "ESP1"}).fillna("ITA1")
        ps["competition"] = "RS"
        ps["raptor_total"] = np.random.RandomState(0).normal(0.0, 5.0, len(ps))
        ps["lebron_total"] = ps["raptor_total"] / 2.0
        data["player_stats"] = ps
        return data

    def _tiny_distribution_rows(self, data):
        rows = []
        ps = data["player_stats"]
        for metric, source in (("RAPTOR", "raptor_total"), ("VORP", "vorp")):
            for (league, season, comp), group in ps.groupby(
                ["league_key", "season", "competition"]
            ):
                values = pd.to_numeric(group[source], errors="coerce").dropna()
                if values.empty:
                    continue
                dist = build_distribution(values)
                summary = dist.summary()
                rows.append(
                    {
                        "metric": metric,
                        "league_key": league,
                        "season": int(str(season).split("-")[0]),
                        "competition": comp,
                        **{
                            key: summary[key]
                            for key in (
                                "sample_size", "min", "p10", "p25", "p40",
                                "p50", "p60", "p75", "p90", "p97_5", "max",
                                "mean", "stddev", "quality",
                            )
                        },
                    }
                )
        return rows

    def test_without_distributions_columns_unchanged(self, tiny_data):
        model = SeasonAheadPerformanceModel()
        X, _ = model.prepare_features(tiny_data)
        assert not X.columns.str.contains("_percentile").any()
        assert not X.columns.str.contains("RAPTOR_").any()

    def test_with_distributions_adds_rating_features(self, tiny_data):
        data = self._augmented_tiny_data(tiny_data)
        dist_rows = self._tiny_distribution_rows(data)
        model = SeasonAheadPerformanceModel()
        X, _ = model.prepare_features(data, rating_distributions=dist_rows)
        assert "RAPTOR_value" in X.columns
        assert "RAPTOR_percentile" in X.columns
        assert "RAPTOR_zscore" in X.columns
        assert "RAPTOR_tier" in X.columns
        assert "VORP_tier" in X.columns
        assert model.rating_feature_columns
        # feature_names persisted for the model
        assert set(model.rating_feature_columns).issubset(X.columns)

"""Tests for the FASE G backfill generator (basketball_ai.metric_rating.backfill)."""
from __future__ import annotations

import pandas as pd
import pytest

from basketball_ai.metric_rating.backfill import (
    DISTRIBUTION_COLUMNS,
    RATING_COLUMNS,
    generate_backfill,
)

CANONICAL_COLUMNS = [
    "league_key", "season", "competition", "player_id", "player_global_id",
    "player_name", "position", "games_played", "minutes_per_game",
    "raptor_total", "lebron_total", "vorp", "games_started", "starter_pct",
]


def make_canonical_csv(tmp_path) -> str:
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
                    "raptor_total": float(index + 1),
                    "lebron_total": float(index + 1) / 2.0,
                    "vorp": 0.0 if index % 10 == 0 else float(index - 30),
                    "games_started": 20,
                    "starter_pct": 0.8,
                }
            )
    path = tmp_path / "player_stats.csv"
    pd.DataFrame(rows, columns=CANONICAL_COLUMNS).to_csv(path, index=False)
    return str(path)


class TestGenerateBackfill:
    def test_definitions_file(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        frame = pd.read_csv(out / "metric_definition.csv")
        assert set(frame["metric"]) == {"RAPTOR", "LEBRON", "VORP"}
        assert frame.loc[frame["metric"] == "RAPTOR", "direction"].iloc[0] == "higher_better"
        assert frame.loc[frame["metric"] == "VORP", "replacement_reference"].iloc[0] == 0.0

    def test_distributions_file(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        frame = pd.read_csv(out / "metric_distribution.csv")
        assert len(frame) == 6  # 2 leagues x 3 metrics (RS)
        raptor_ita = frame[
            (frame["metric"] == "RAPTOR") & (frame["league_key"] == "ITA1")
        ].iloc[0]
        assert raptor_ita["sample_size"] == 60
        assert raptor_ita["mean"] == pytest.approx(30.5)
        assert raptor_ita["p50"] == pytest.approx(30.5)
        assert raptor_ita["quality"] == "low"
        assert raptor_ita["population_source"] == "exact"
        assert not bool(raptor_ita["fallback_used"])
        assert str(raptor_ita["distribution_version"]) == "1.0"

    def test_ratings_file(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        frame = pd.read_csv(out / "metric_rating.csv")
        assert len(frame) == 6 * 60  # 2 leagues x 3 metrics x 60 players (VORP: all have values)
        raptor_ita = frame[
            (frame["metric"] == "RAPTOR") & (frame["league_key"] == "ITA1")
        ]
        top = raptor_ita.sort_values("value").iloc[-1]
        assert top["tier"] == 8 and top["label"] == "Superstar"
        assert top["percentile"] == pytest.approx(1.0)
        bottom = raptor_ita.sort_values("value").iloc[0]
        assert bottom["tier"] == 1 and bottom["label"] == "Very Poor"
        # VORP carries the replacement reference flags
        vorp = frame[frame["metric"] == "VORP"]
        assert vorp["above_reference"].notna().all()
        assert vorp["below_reference"].notna().all()
        assert set(frame["distribution_version"].astype(str)) == {"1.0"}
        assert set(frame["rating_version"].astype(str)) == {"1.0"}

    def test_versions_custom(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(
            make_canonical_csv(tmp_path), out,
            distribution_version="2.0", rating_version="3.0",
        )
        dist = pd.read_csv(out / "metric_distribution.csv")
        rating = pd.read_csv(out / "metric_rating.csv")
        assert set(dist["distribution_version"].astype(str)) == {"2.0"}
        assert set(rating["rating_version"].astype(str)) == {"3.0"}

    def test_apply_sql_generated(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        sql = (out / "apply_backfill.sql").read_text(encoding="utf-8")
        assert "\\copy \"AI\".\"MetricDistribution\"" in sql
        assert "\\copy \"AI\".\"MetricRating\"" in sql
        assert "metric_definition.csv" in sql
        assert "ON CONFLICT (metric) DO NOTHING" in sql

    def test_apply_sql_plain_generated(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        sql = (out / "apply_backfill_plain.sql").read_text(encoding="utf-8")
        assert sql.startswith("-- FASE G")
        assert "BEGIN;" in sql and "COMMIT;" in sql
        assert 'INSERT INTO "AI"."MetricDefinition"' in sql
        assert 'INSERT INTO "AI"."MetricDistribution"' in sql
        assert 'INSERT INTO "AI"."MetricRating"' in sql
        # Pure SQL: no psql meta-commands (they live in apply_backfill.sql).
        assert "\\copy" not in sql
        assert "\\set" not in sql
        assert "ON CONFLICT (metric) DO NOTHING" in sql

    def test_plain_sql_escapes_apostrophes(self, tmp_path):
        rows = [
            {
                "league_key": "ITA1", "season": 2025, "competition": "RS",
                "player_id": "1", "player_global_id": "G1",
                "player_name": "O'Neal, Shaquille", "position": "C",
                "games_played": 25, "minutes_per_game": 20.0,
                "raptor_total": 5.0, "lebron_total": 2.5, "vorp": 1.5,
                "games_started": 20, "starter_pct": 0.8,
            }
        ]
        path = tmp_path / "player_stats.csv"
        pd.DataFrame(rows, columns=CANONICAL_COLUMNS).to_csv(path, index=False)
        out = tmp_path / "backfill"
        generate_backfill(path, out)
        sql = (out / "apply_backfill_plain.sql").read_text(encoding="utf-8")
        assert "O''Neal, Shaquille" in sql

    def test_csvs_are_plain_importable(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        expected = {
            "metric_definition.csv": 8,
            "metric_distribution.csv": 22,
            "metric_rating.csv": 18,
        }
        for name, ncols in expected.items():
            text = (out / name).read_text(encoding="utf-8")
            lines = text.splitlines()
            assert len(lines[0].split(",")) == ncols, name
            assert '"' not in text, f"{name}: quoting present, not importable"
            bad = [line for line in lines[1:] if line.count(",") + 1 != ncols]
            assert not bad, f"{name}: {len(bad)} rows with wrong field count"
            # No comma inside ANY field (would be read as a delimiter).
            import csv as _csv, io as _io
            for row in _csv.reader(_io.StringIO(text)):
                assert all("," not in field for field in row), f"{name}: comma in field"

    def test_headers_match_schema_columns(self, tmp_path):
        out = tmp_path / "backfill"
        generate_backfill(make_canonical_csv(tmp_path), out)
        dist = pd.read_csv(out / "metric_distribution.csv")
        rating = pd.read_csv(out / "metric_rating.csv")
        # CSV headers must be exactly the columns used by the apply SQL
        assert list(dist.columns) == list(DISTRIBUTION_COLUMNS)
        assert list(rating.columns) == list(RATING_COLUMNS)

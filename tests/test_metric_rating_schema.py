from pathlib import Path


DDL_PATH = Path(__file__).resolve().parents[1] / "basketball_ai" / "data" / "ai_metric_schema.sql"


def test_metric_rating_storage_ddl_exists_and_is_idempotent():
    assert DDL_PATH.is_file()
    sql = DDL_PATH.read_text(encoding="utf-8")
    assert 'CREATE SCHEMA IF NOT EXISTS "AI"' in sql
    for table in ("MetricDefinition", "MetricDistribution", "MetricRating"):
        assert f'CREATE TABLE IF NOT EXISTS "AI"."{table}"' in sql
    assert "CREATE INDEX IF NOT EXISTS" in sql
    assert sql.strip().startswith("-- Metric Rating Engine storage contract.")
    assert sql.strip().endswith("COMMIT;")


def test_metric_rating_storage_matches_backfill_columns():
    sql = DDL_PATH.read_text(encoding="utf-8")
    required_columns = {
        "MetricDefinition": (
            "metric", "source_column", "direction", "zero_reference",
            "rating_enabled", "replacement_reference", "description", "value_label",
        ),
        "MetricDistribution": (
            "metric", "league_key", "season", "competition", "population_type",
            "population_source", "fallback_used", "sample_size", "min", "p10",
            "p25", "p40", "p50", "p60", "p75", "p90", "p97_5", "max",
            "mean", "stddev", "quality", "distribution_version",
        ),
        "MetricRating": (
            "metric", "player_global_id", "player_name", "league_key", "season",
            "competition", "value", "percentile", "zscore", "tier", "label",
            "quality", "population_source", "fallback_used", "distribution_version",
            "rating_version", "above_reference", "below_reference",
        ),
    }
    for table, columns in required_columns.items():
        assert f'CREATE TABLE IF NOT EXISTS "AI"."{table}"' in sql
        for column in columns:
            assert column in sql, f'{table} is missing backfill column: {column}'


def test_metric_rating_ddl_does_not_mutate_source_contract():
    sql = DDL_PATH.read_text(encoding="utf-8").upper()
    assert '"AI_SOURCE"' not in sql
    assert "DROP TABLE" not in sql
    assert "TRUNCATE" not in sql

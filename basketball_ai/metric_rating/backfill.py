"""FASE G — backfill generator for the Metric Rating Engine storage.

Reads the canonical POC dump (``data/metric_poc/player_stats.csv``) and
produces the files to populate ``AI.MetricDefinition`` / ``AI.MetricDistribution``
/ ``AI.MetricRating`` (see ``basketball_ai/data/ai_metric_schema.sql``):

- ``metric_definition.csv``   — the registered metrics (RAPTOR, LEBRON, VORP);
- ``metric_distribution.csv`` — the production distribution per
  (metric, league, season, competition), with fallback/quality provenance;
- ``metric_rating.csv``       — the rating for every player in each context;
- ``apply_backfill.sql``      — psql script that loads the CSVs.

Run::

    python -m basketball_ai.metric_rating.backfill \
        --csv data/metric_poc/player_stats.csv \
        --out data/metric_poc/backfill

Apply (from the repository root, after running the DDL)::

    psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_schema.sql
    psql -d YOUR_DATABASE -f data/metric_poc/backfill/apply_backfill.sql
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from basketball_ai.metric_rating.definitions import METRIC_DEFINITIONS
from basketball_ai.metric_rating.distribution import (
    DistributionStats,
    build_distribution,
)
from basketball_ai.metric_rating.poc import load_dump
from basketball_ai.metric_rating.population import PopulationBuilder, PopulationConfig
from basketball_ai.metric_rating.rating import MetricRatingEngine

DEFINITION_COLUMNS = (
    "metric", "source_column", "direction", "zero_reference", "rating_enabled",
    "replacement_reference", "description", "value_label",
)
DISTRIBUTION_COLUMNS = (
    "metric", "league_key", "season", "competition", "population_type",
    "population_source", "fallback_used", "sample_size", "min", "p10", "p25",
    "p40", "p50", "p60", "p75", "p90", "p97_5", "max", "mean", "stddev",
    "quality", "distribution_version",
)
RATING_COLUMNS = (
    "metric", "player_global_id", "player_name", "league_key", "season",
    "competition", "value", "percentile", "zscore", "tier", "label", "quality",
    "population_source", "fallback_used", "distribution_version",
    "rating_version", "above_reference", "below_reference",
)


def _definition_rows() -> list[dict[str, Any]]:
    rows = []
    for definition in METRIC_DEFINITIONS.values():
        rows.append(
            {
                "metric": definition.metric,
                "source_column": definition.source_column,
                "direction": definition.direction,
                "zero_reference": definition.zero_reference,
                "rating_enabled": definition.rating_enabled,
                "replacement_reference": definition.replacement_reference,
                "description": definition.description,
                "value_label": definition.value_label,
            }
        )
    return rows


def _distribution_row(
    metric: str,
    league: str,
    season: str,
    phase: str,
    distribution: DistributionStats,
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "league_key": league,
        "season": int(season),
        "competition": phase,
        "population_type": distribution.population_type,
        "population_source": distribution.population_source,
        "fallback_used": fallback_used,
        "sample_size": distribution.sample_size,
        "min": distribution.min,
        "p10": distribution.p10,
        "p25": distribution.p25,
        "p40": distribution.p40,
        "p50": distribution.p50,
        "p60": distribution.p60,
        "p75": distribution.p75,
        "p90": distribution.p90,
        "p97_5": distribution.p97_5,
        "max": distribution.max,
        "mean": distribution.mean,
        "stddev": distribution.stddev,
        "quality": distribution.quality,
        "distribution_version": distribution.distribution_version,
    }


def generate_backfill(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    min_games: int = 10,
    min_minutes_per_game: float = 0.0,
    min_samples: int = 50,
    distribution_version: str = "1.0",
    rating_version: str = "1.0",
) -> dict[str, Any]:
    """Generate the storage CSVs + apply SQL from the canonical POC dump."""
    frame = load_dump(csv_path)
    config = PopulationConfig(
        min_games=min_games,
        min_minutes_per_game=min_minutes_per_game,
        min_samples=min_samples,
    )
    builder = PopulationBuilder(config)
    engine = MetricRatingEngine(
        population_config=config,
        distribution_version=distribution_version,
        rating_version=rating_version,
    )

    distributions: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []

    contexts = sorted(
        {
            (str(r.league_key), str(r.season), str(r.competition).upper())
            for r in frame[["league_key", "season", "competition"]].itertuples(index=False)
        }
    )

    for metric, definition in METRIC_DEFINITIONS.items():
        source_column = definition.source_column
        if source_column not in frame.columns:
            continue
        for league, season, phase in contexts:
            population = builder.build(
                frame,
                source_column=source_column,
                league=league,
                season=season,
                phase=phase,
                fallback=True,
            )
            if population.sample_size == 0:
                continue
            distribution = build_distribution(
                population.values,
                population_source=population.population_source,
                population_type=population.population_type,
                population_key=population.population_key,
                distribution_version=distribution_version,
                quality_thresholds=engine.quality_thresholds,
            )
            distributions.append(
                _distribution_row(
                    metric, league, season, phase, distribution,
                    population.fallback_used,
                )
            )

            # Rate every player of the exact context against the stored
            # production distribution (fallback/quality stay explicit).
            mask = (
                (frame["league_key"].str.upper() == league.upper())
                & (frame["season"] == season)
                & (frame["competition"].str.upper() == phase.upper())
                & (frame["games_played"].fillna(0) >= config.min_games)
                & (frame["minutes_per_game"].fillna(0) >= config.min_minutes_per_game)
            )
            values = pd.to_numeric(frame.loc[mask, source_column], errors="coerce")
            for _, row in frame.loc[mask].iterrows():
                value = values.loc[row.name]
                if pd.isna(value):
                    continue
                rating = engine.rate_from_distribution(metric, float(value), distribution)
                payload = rating.to_dict(rounded=False)
                ratings.append(
                    {
                        "metric": metric,
                        "player_global_id": str(
                            row.get("player_global_id", "") or row.get("player_id", "")
                        ),
                        "player_name": str(row.get("player_name", "") or ""),
                        "league_key": league,
                        "season": int(season),
                        "competition": phase,
                        "value": payload["value"],
                        "percentile": payload["percentile"],
                        "zscore": payload["zscore"],
                        "tier": payload["tier"],
                        "label": payload["label"],
                        "quality": payload["quality"],
                        "population_source": payload["population_source"],
                        "fallback_used": payload["fallback_used"],
                        "distribution_version": payload["distribution_version"],
                        "rating_version": payload["rating_version"],
                        "above_reference": payload["above_reference"],
                        "below_reference": payload["below_reference"],
                    }
                )

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    definitions = _definition_rows()

    _write_csv(
        definitions, DEFINITION_COLUMNS, target / "metric_definition.csv",
        text_columns=("metric", "source_column", "direction", "description",
                      "value_label"),
    )
    _write_csv(
        distributions, DISTRIBUTION_COLUMNS, target / "metric_distribution.csv",
        text_columns=("metric", "league_key", "competition", "population_type",
                      "population_source"),
        string_columns=("distribution_version",),
    )
    _write_csv(
        ratings, RATING_COLUMNS, target / "metric_rating.csv",
        text_columns=("metric", "player_global_id", "player_name", "league_key",
                      "competition", "label", "quality", "population_source"),
        string_columns=("distribution_version", "rating_version"),
    )
    (target / "apply_backfill.sql").write_text(
        _render_apply_sql(target), encoding="utf-8"
    )
    (target / "apply_backfill_plain.sql").write_text(
        _render_apply_sql_plain(definitions, distributions, ratings),
        encoding="utf-8",
    )

    return {
        "definitions": len(definitions),
        "distributions": len(distributions),
        "ratings": len(ratings),
        "out_dir": str(target),
    }


def _render_apply_sql(out_dir: Path) -> str:
    rel = out_dir.resolve().as_posix()
    return f"""-- FASE G — load the backfill CSVs into "AI" (one-shot).
-- Paths are absolute, so this works from anywhere. Apply the DDL first:
--     psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_schema.sql
\\set ON_ERROR_STOP on
BEGIN;

-- Definitions (idempotent via temp table + ON CONFLICT).
CREATE TEMP TABLE _mdef (
    metric text, source_column text, direction text,
    zero_reference double precision, rating_enabled boolean,
    replacement_reference double precision, description text, value_label text
);
\\copy _mdef FROM '{rel}/metric_definition.csv' WITH (FORMAT csv, HEADER)
INSERT INTO "AI"."MetricDefinition"(metric, source_column, direction,
    zero_reference, rating_enabled, replacement_reference, description,
    value_label)
SELECT metric, source_column, direction, zero_reference, rating_enabled,
       replacement_reference, description, value_label
FROM _mdef
ON CONFLICT (metric) DO NOTHING;
DROP TABLE _mdef;

\\copy "AI"."MetricDistribution"(metric, league_key, season, competition,
    population_type, population_source, fallback_used, sample_size,
    min, p10, p25, p40, p50, p60, p75, p90, p97_5, max, mean, stddev,
    quality, distribution_version)
FROM '{rel}/metric_distribution.csv' WITH (FORMAT csv, HEADER);

\\copy "AI"."MetricRating"(metric, player_global_id, player_name, league_key,
    season, competition, value, percentile, zscore, tier, label, quality,
    population_source, fallback_used, distribution_version, rating_version,
    above_reference, below_reference)
FROM '{rel}/metric_rating.csv' WITH (FORMAT csv, HEADER);

COMMIT;
"""


_INT_FLOAT_RE = re.compile(r"^-?\d+\.0$")


def _sanitize(value: Any) -> str:
    """Strip characters that break naive CSV parsers (quoted fields) and
    normalize the pandas float-inference artifact ``46238.0`` -> ``46238``."""
    if value is None:
        return ""
    text = str(value)
    for character in (",", ";", "\t", '"', "\r", "\n"):
        text = text.replace(character, " ")
    if _INT_FLOAT_RE.fullmatch(text):
        return text[:-2]
    return text


def _write_csv(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    path: Path,
    *,
    text_columns: tuple[str, ...] = (),
    string_columns: tuple[str, ...] = (),
) -> None:
    """Write a plain, importable CSV: no quoting, no embedded delimiters,
    versions kept as text (e.g. '1.0' not 1)."""
    frame = pd.DataFrame(rows, columns=columns)
    for column in text_columns:
        frame[column] = frame[column].map(_sanitize)
    for column in string_columns:
        frame[column] = frame[column].astype(str)
    frame.to_csv(path, index=False)


def _sql_literal(value: Any) -> str:
    """SQL literal for a Python value (None -> NULL, NaN -> NULL, quotes escaped)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if pd.isna(value):
            return "NULL"
        return repr(float(value))
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _inserts(table: str, columns: tuple[str, ...], rows: list[dict[str, Any]],
             *, chunk: int = 500, on_conflict: str = "") -> str:
    """Chunked multi-row INSERT statements (works in any SQL tool)."""
    column_sql = ", ".join(f'"{column}"' for column in columns)
    statements: list[str] = []
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk]
        values = ",\n".join(
            "(" + ", ".join(_sql_literal(row[column]) for column in columns) + ")"
            for row in block
        )
        suffix = f" ON CONFLICT {on_conflict}" if on_conflict else ""
        statements.append(
            f'INSERT INTO {table}({column_sql}) VALUES\n{values}{suffix};'
        )
    return "\n\n".join(statements)


def _render_apply_sql_plain(
    definitions: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
) -> str:
    """Pure-SQL apply script (INSERT statements, no psql meta-commands), usable
    from pgAdmin / DBeaver / any SQL client."""
    return f"""-- FASE G — load the backfill into "AI" (pure SQL, no psql meta-commands).
-- Works in pgAdmin, DBeaver and psql. Apply the DDL first:
--     psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_schema.sql
-- Notes:
--   * definitions are idempotent (ON CONFLICT DO NOTHING);
--   * distributions/ratings are one-shot inserts (delete rows before re-running).
--   * faster alternative with psql: apply_backfill.sql.
BEGIN;

{_inserts('"AI"."MetricDefinition"', DEFINITION_COLUMNS, definitions, on_conflict='(metric) DO NOTHING')}

{_inserts('"AI"."MetricDistribution"', DISTRIBUTION_COLUMNS, distributions)}

{_inserts('"AI"."MetricRating"', RATING_COLUMNS, ratings)}

COMMIT;
"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m basketball_ai.metric_rating.backfill",
        description="Generate the FASE G backfill files for the Metric Rating Engine.",
    )
    parser.add_argument("--csv", required=True, help="Canonical POC dump CSV")
    parser.add_argument("--out", default="data/metric_poc/backfill")
    parser.add_argument("--min-games", type=int, default=10)
    parser.add_argument("--min-minutes", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args(argv)

    summary = generate_backfill(
        args.csv,
        args.out,
        min_games=args.min_games,
        min_minutes_per_game=args.min_minutes,
        min_samples=args.min_samples,
    )
    print(
        f"OK: definitions={summary['definitions']} "
        f"distributions={summary['distributions']} "
        f"ratings={summary['ratings']} -> {summary['out_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

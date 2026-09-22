"""FASE D — POC report generator for the Metric Rating Engine.

Consumes a CSV dump of the ``"AI_Source"`` contract (see
``basketball_ai/data/ai_metric_poc_dump.sql``) and produces the validation
report required by FASE 12:

- distribution table: metric × league × season × phase with
  ``sample_size``, ``mean``, ``stddev``, ``p10``, ``p25``, ``p50``, ``p75``,
  ``p90``, ``p97.5`` and population ``quality``;
- representative players per context: top / median / bottom with ``value``,
  ``percentile``, ``zscore``, ``tier``, ``label``, ``quality`` and fallback
  provenance;
- data-quality signal: fraction of exact-0 values per context (NULL→0
  coalescing detector, open issue #1).

Run::

    python -m basketball_ai.metric_rating.poc \
        --csv data/metric_poc/player_stats.csv \
        --out data/metric_poc/report

The report is written as ``distributions.json``, ``players.json`` and
``report.md``; the function also returns the dict for direct use.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from basketball_ai.metric_rating import (
    METRIC_DEFINITIONS,
    DistributionStats,
    MetricRating,
    MetricRatingEngine,
    PopulationBuilder,
    PopulationConfig,
    build_distribution,
)

CONTEXT_COLUMNS = ("league_key", "season", "competition")
REQUIRED_DUMP_COLUMNS = CONTEXT_COLUMNS + ("games_played", "minutes_per_game")
PLAYER_COLUMNS = ("player_id", "player_global_id", "player_name")


def load_dump(csv_path: str | Path) -> pd.DataFrame:
    """Load the POC dump CSV and validate the canonical columns."""
    frame = pd.read_csv(csv_path)
    missing = [
        column
        for column in REQUIRED_DUMP_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Dump CSV is missing canonical columns: {sorted(missing)}"
        )
    for column in CONTEXT_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()
    return frame


def _qualified_mask(
    frame: pd.DataFrame,
    config: PopulationConfig,
) -> pd.Series:
    return (
        (frame["games_played"].fillna(0) >= config.min_games)
        & (frame["minutes_per_game"].fillna(0) >= config.min_minutes_per_game)
    )


def _contexts(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    values = sorted(
        {
            (str(row.league_key), str(row.season), str(row.competition).upper())
            for row in frame[list(CONTEXT_COLUMNS)].itertuples(index=False)
        }
    )
    return values


def _distribution_row(
    metric: str,
    league: str,
    season: str,
    phase: str,
    distribution: DistributionStats,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "league": league,
        "season": season,
        "phase": phase,
    }
    row.update(distribution.summary())
    return row


def _quality_signal(
    metric: str,
    source_column: str,
    league: str,
    season: str,
    phase: str,
    exact: pd.DataFrame,
) -> dict[str, Any]:
    values = pd.to_numeric(exact[source_column], errors="coerce").dropna()
    zeros = int((values == 0.0).sum())
    return {
        "metric": metric,
        "league": league,
        "season": season,
        "phase": phase,
        "rows_with_value": int(values.size),
        "exact_zero_values": zeros,
        "zero_fraction": round(zeros / values.size, 4) if values.size else 0.0,
    }


def _representative_players(
    metric: str,
    league: str,
    season: str,
    phase: str,
    rows: pd.DataFrame,
    distribution: DistributionStats,
    engine: MetricRatingEngine,
) -> list[dict[str, Any]]:
    """Rate every player in the exact context against the distribution used
    in production (exact or fallback) and keep top/median/bottom."""
    if rows.empty or distribution.sample_size == 0:
        return []
    ratings: list[tuple[MetricRating, str, str]] = []
    for _, row in rows.iterrows():
        try:
            rating = engine.rate_from_distribution(
                metric, float(row[metric_source(metric)]), distribution
            )
        except (TypeError, ValueError):
            continue
        name = str(row.get("player_name", row.get("player_global_id", "?")))
        global_id = str(row.get("player_global_id", ""))
        ratings.append((rating, name, global_id))
    if not ratings:
        return []

    def percentile_key(item: tuple[MetricRating, str, str]) -> float:
        value = item[0].percentile
        return 0.0 if value is None else float(value)

    by_percentile = sorted(ratings, key=percentile_key)
    selected = {
        "top": by_percentile[-1],
        "bottom": by_percentile[0],
        "median": min(by_percentile, key=lambda item: abs(percentile_key(item) - 0.5)),
    }
    out: list[dict[str, Any]] = []
    for role, (rating, name, global_id) in selected.items():
        payload = rating.to_dict()
        payload.pop("population_key", None)
        out.append(
            {
                "metric": metric,
                "league": league,
                "season": season,
                "phase": phase,
                "role": role,
                "player": name,
                "player_global_id": global_id,
                **payload,
            }
        )
    return out


def metric_source(metric: str) -> str:
    return METRIC_DEFINITIONS[metric.upper()].source_column


def generate_poc_report(
    csv_path: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
    min_games: int = 10,
    min_minutes_per_game: float = 0.0,
    min_samples: int = 50,
) -> dict[str, Any]:
    """Generate the FASE D validation report from a dump CSV."""
    frame = load_dump(csv_path)
    config = PopulationConfig(
        min_games=min_games,
        min_minutes_per_game=min_minutes_per_game,
        min_samples=min_samples,
    )
    builder = PopulationBuilder(config)
    engine = MetricRatingEngine(population_config=config)
    contexts = _contexts(frame)

    distributions: list[dict[str, Any]] = []
    players: list[dict[str, Any]] = []
    quality_signals: list[dict[str, Any]] = []

    for metric, definition in METRIC_DEFINITIONS.items():
        source_column = definition.source_column
        if source_column not in frame.columns:
            continue
        for league, season, phase in contexts:
            exact = builder.build(
                frame,
                source_column=source_column,
                league=league,
                season=season,
                phase=phase,
                fallback=False,
            )
            production = builder.build(
                frame,
                source_column=source_column,
                league=league,
                season=season,
                phase=phase,
                fallback=True,
            )
            if exact.sample_size > 0:
                distributions.append(
                    _distribution_row(
                        metric,
                        league,
                        season,
                        phase,
                        build_distribution(
                            exact.values,
                            population_source=exact.population_source,
                            population_type=exact.population_type,
                            population_key=exact.population_key,
                            distribution_version=engine.distribution_version,
                            quality_thresholds=engine.quality_thresholds,
                        ),
                    )
                )
                quality_signals.append(
                    _quality_signal(
                        metric,
                        source_column,
                        league,
                        season,
                        phase,
                        frame.loc[exact_filters(frame, league, season, phase)],
                    )
                )
            if production.sample_size == 0:
                continue
            production_distribution = build_distribution(
                production.values,
                population_source=production.population_source,
                population_type=production.population_type,
                population_key=production.population_key,
                distribution_version=engine.distribution_version,
                quality_thresholds=engine.quality_thresholds,
            )
            context_rows = frame.loc[exact_filters(frame, league, season, phase)]
            context_rows = context_rows[_qualified_mask(context_rows, config)]
            players.extend(
                _representative_players(
                    metric,
                    league,
                    season,
                    phase,
                    context_rows,
                    production_distribution,
                    engine,
                )
            )

    report = {
        "generated_at": _now_iso(),
        "source": str(csv_path),
        "population_config": asdict(config),
        "distribution_version": engine.distribution_version,
        "rating_version": engine.rating_version,
        "distributions": distributions,
        "representative_players": players,
        "data_quality": quality_signals,
    }

    if out_dir is not None:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "distributions.json").write_text(
            json.dumps(report["distributions"], indent=2), encoding="utf-8"
        )
        (target / "players.json").write_text(
            json.dumps(report["representative_players"], indent=2), encoding="utf-8"
        )
        (target / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def exact_filters(
    frame: pd.DataFrame, league: str, season: str, phase: str
) -> pd.Series:
    return (
        (frame["league_key"].str.upper() == str(league).upper())
        & (frame["season"] == str(season))
        & (frame["competition"].str.upper() == str(phase).upper())
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Metric Rating Engine — POC report (FASE D)", ""]
    lines.append(f"- source: `{report['source']}`")
    lines.append(f"- population config: `{json.dumps(report['population_config'])}`")
    lines.append(f"- distribution_version: {report['distribution_version']} · "
                 f"rating_version: {report['rating_version']}")
    lines.append("")

    lines.append("## Distributions")
    header = ("| metric | league | season | phase | n | mean | stddev | p10 | "
              "p25 | p50 | p75 | p90 | p97.5 | quality |")
    lines.append(header)
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in report["distributions"]:
        lines.append(
            f"| {row['metric']} | {row['league']} | {row['season']} | {row['phase']} "
            f"| {row['sample_size']} | {row['mean']:.3f} | {row['stddev']:.3f} "
            f"| {row['p10']:.3f} | {row['p25']:.3f} | {row['p50']:.3f} "
            f"| {row['p75']:.3f} | {row['p90']:.3f} | {row['p97_5']:.3f} "
            f"| {row['quality']} |"
        )
    lines.append("")

    lines.append("## Representative players (top / median / bottom)")
    header = ("| metric | league | season | phase | role | player | value | "
              "percentile | zscore | tier | label | quality | fallback |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|")
    for row in report["representative_players"]:
        lines.append(
            f"| {row['metric']} | {row['league']} | {row['season']} | {row['phase']} "
            f"| {row['role']} | {row['player']} | {row['value']:.3f} "
            f"| {row['percentile']:.4f} | {_fmt_opt(row['zscore'])} "
            f"| {_fmt_opt(row['tier'])} | {_fmt_opt(row['label'])} "
            f"| {row['quality']} | {row['fallback_used']} |"
        )
    lines.append("")

    lines.append("## Data quality (NULL→0 signal)")
    header = "| metric | league | season | phase | rows | exact-zero | zero_fraction |"
    lines.append(header)
    lines.append("|---|---|---|---|---:|---:|---:|")
    for row in report["data_quality"]:
        lines.append(
            f"| {row['metric']} | {row['league']} | {row['season']} | {row['phase']} "
            f"| {row['rows_with_value']} | {row['exact_zero_values']} "
            f"| {row['zero_fraction']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_opt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m basketball_ai.metric_rating.poc",
        description="Generate the FASE D POC report for the Metric Rating Engine.",
    )
    parser.add_argument("--csv", required=True, help="Path to the AI_Source dump CSV")
    parser.add_argument("--out", default=None, help="Output directory for the report")
    parser.add_argument("--min-games", type=int, default=10)
    parser.add_argument("--min-minutes", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args(argv)

    report = generate_poc_report(
        args.csv,
        out_dir=args.out,
        min_games=args.min_games,
        min_minutes_per_game=args.min_minutes,
        min_samples=args.min_samples,
    )
    summary = {
        "distributions": len(report["distributions"]),
        "representative_players": len(report["representative_players"]),
        "out_dir": args.out,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

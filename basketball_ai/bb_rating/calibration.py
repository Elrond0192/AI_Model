"""BB-Rating calibration and real-data validation.

Diagnostic only: this module never changes production weights, peer thresholds,
model artifacts, PostgreSQL source data, or the Prediction Model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from basketball_ai.bb_rating.engine import (
    AGE_BANDS,
    BAND_LABELS,
    BB_RATING_METRICS,
    BB_RATING_VERSION,
    BBRatingEngine,
    MIN_CONTEXT_SAMPLES,
    MIN_PEER_SAMPLES,
    RATE_COLUMNS,
    RatingMetricSpec,
)
from basketball_ai.bb_rating.semantics import METRIC_SEMANTICS

CALIBRATION_VERSION = "1.0"


@dataclass(frozen=True)
class BBRatingCalibrationConfig:
    min_peer_samples: int = MIN_PEER_SAMPLES
    min_context_samples: int = MIN_CONTEXT_SAMPLES


def _normalise_series(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if column in RATE_COLUMNS:
        values = values.mask(values.abs() > 1.0, values / 100.0)
    return values.astype(float)


def _score_from_percentile(percentile: pd.Series) -> pd.Series:
    return (1.0 + 99.0 * percentile).round().clip(1, 100).astype("Int64")


def _summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0, "mean": None, "stddev": None, "min": None,
            "p10": None, "p25": None, "p50": None, "p75": None,
            "p90": None, "max": None,
        }
    q = clean.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "stddev": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p10": float(q.loc[0.10]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "max": float(clean.max()),
    }


def _assign_peer_groups(
    frame: pd.DataFrame,
    *,
    min_peer_samples: int,
    min_context_samples: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["_peer_source"] = "limited_context"
    result["_peer_key"] = pd.NA
    result["_peer_sample_size"] = 0

    context_cols = ["league_key", "season", "competition"]
    grouped = result.groupby(context_cols, dropna=False, sort=False).groups

    for context_key, labels in grouped.items():
        ctx = result.loc[list(labels)]
        if ctx.empty:
            continue

        position_counts = ctx.groupby("position_family", dropna=False).size().to_dict()
        pa_counts = ctx.groupby(
            ["position_family", "age_band"], dropna=False
        ).size().to_dict()
        role_counts = ctx.groupby(
            ["position_family", "age_band", "ruolo_combinato"], dropna=False
        ).size().to_dict()

        league, season, competition = context_key
        source_values: list[str] = []
        peer_sizes: list[int] = []
        peer_keys: list[str] = []

        for row in ctx.itertuples(index=False):
            position = str(getattr(row, "position_family", "") or "OTHER")
            age_band = str(getattr(row, "age_band", "") or "")
            role = str(getattr(row, "ruolo_combinato", "") or "").strip()

            role_count = role_counts.get((position, age_band, role), 0) if role else 0
            pa_count = pa_counts.get((position, age_band), 0)
            position_count = position_counts.get(position, 0)
            context_count = len(ctx)

            if role and role_count >= min_peer_samples:
                source = "position+age+role"
                size = role_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|position+age+role|"
                    f"{position}|{age_band}|{role}"
                )
            elif pa_count >= min_peer_samples:
                source = "position+age"
                size = pa_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|position+age|"
                    f"{position}|{age_band}"
                )
            elif position_count >= min_peer_samples:
                source = "position"
                size = position_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|position|{position}"
                )
            elif context_count >= min_context_samples:
                source = "league+season+phase"
                size = context_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|league+season+phase"
                )
            else:
                source = "limited_context"
                size = context_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|limited_context"
                )

            source_values.append(source)
            peer_sizes.append(int(size))
            peer_keys.append(peer_key)

        result.loc[ctx.index, "_peer_source"] = source_values
        result.loc[ctx.index, "_peer_sample_size"] = peer_sizes
        result.loc[ctx.index, "_peer_key"] = peer_keys

    return result


def _metric_percentiles(
    frame: pd.DataFrame,
    spec: RatingMetricSpec,
) -> tuple[pd.Series, pd.Series]:
    values = _normalise_series(frame[spec.source_column], spec.source_column)
    valid = values.notna()
    valid_n = (
        values.where(valid)
        .groupby(frame["_peer_key"], dropna=False)
        .transform("count")
    )
    rank_average = values.groupby(
        frame["_peer_key"], dropna=False
    ).rank(method="average")

    # Exactly match BBRatingEngine._percentile:
    # (count below + count at-or-below) / (2*n).
    percentile = (rank_average - 0.5) / valid_n
    percentile = percentile.clip(0.0, 1.0).where(valid)
    if spec.direction == "lower_better":
        percentile = (1.0 - percentile).where(valid)
    return values, percentile


def _registry_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BB_RATING_METRICS:
        semantic = METRIC_SEMANTICS.get(spec.key)
        rows.append(
            {
                "metric": spec.key,
                "source_column": spec.source_column,
                "column_exists": spec.source_column in frame.columns,
                "dimension": spec.dimension,
                "weight": float(spec.weight),
                "direction": spec.direction,
                "semantic_exists": semantic is not None,
                "semantic_source_matches": bool(
                    semantic is not None
                    and semantic.source_column == spec.source_column
                ),
                "rating_enabled": bool(spec.weight > 0.0),
            }
        )
    return rows


def _metric_summary(
    frame: pd.DataFrame,
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in registry:
        key = row["metric"]
        result = dict(row)
        if not row["column_exists"]:
            result.update(
                {
                    "n_rows": int(len(frame)),
                    "n_available": 0,
                    "missing_pct": 1.0,
                    "zero_pct_all_rows": None,
                    "zero_pct_available": None,
                    "percentile_summary": _summary(pd.Series(dtype=float)),
                    "effective_weight_mean": None,
                }
            )
            output.append(result)
            continue

        values = frame[f"_value_{key}"]
        percentiles = frame[f"_pct_{key}"]
        available = values.notna()
        n_available = int(available.sum())
        effective = None
        if row["rating_enabled"]:
            effective_values = frame[f"_effective_weight_{key}"]
            if effective_values.notna().any():
                effective = float(effective_values.dropna().mean())

        result.update(
            {
                "n_rows": int(len(frame)),
                "n_available": n_available,
                "missing_pct": (
                    float(1.0 - n_available / len(frame)) if len(frame) else 1.0
                ),
                "zero_pct_all_rows": (
                    float(values.eq(0).mean()) if len(frame) else None
                ),
                "zero_pct_available": (
                    float(values.loc[available].eq(0).mean())
                    if n_available else None
                ),
                "percentile_summary": _summary(percentiles),
                "effective_weight_mean": effective,
            }
        )
        output.append(result)
    return output


def _score_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result = _summary(frame["_bb_rating"])
    if not result["n"]:
        result.update(
            {
                "share_1_20": None,
                "share_21_40": None,
                "share_41_60": None,
                "share_61_80": None,
                "share_81_100": None,
                "share_floor_1": None,
                "share_ceiling_100": None,
            }
        )
        return result

    values = pd.to_numeric(frame["_bb_rating"], errors="coerce").dropna()
    result.update(
        {
            "share_1_20": float((values <= 20).mean()),
            "share_21_40": float(((values >= 21) & (values <= 40)).mean()),
            "share_41_60": float(((values >= 41) & (values <= 60)).mean()),
            "share_61_80": float(((values >= 61) & (values <= 80)).mean()),
            "share_81_100": float((values >= 81).mean()),
            "share_floor_1": float((values == 1).mean()),
            "share_ceiling_100": float((values == 100).mean()),
        }
    )
    return result


def _group_score_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = {
            column: (int(value) if column == "season" and pd.notna(value) else value)
            for column, value in zip(group_cols, keys)
        }
        row.update(_score_summary(group))
        row["n"] = int(len(group))
        rows.append(row)
    return rows


def _stability(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["player_global_id", "league_key", "competition", "season"]
    ).copy()
    group_cols = ["player_global_id", "league_key", "competition"]
    ordered["_next_season"] = ordered.groupby(group_cols)["season"].shift(-1)
    ordered["_next_score"] = ordered.groupby(group_cols)["_bb_rating"].shift(-1)

    pairs = ordered.loc[
        ordered["_next_season"].eq(ordered["season"] + 1)
        & ordered["_bb_rating"].notna()
        & ordered["_next_score"].notna()
    ]
    if pairs.empty:
        return {
            "n_pairs": 0,
            "mean_abs_change": None,
            "median_abs_change": None,
            "share_abs_change_le_5": None,
            "share_abs_change_le_10": None,
            "score_correlation": None,
        }

    changes = (pairs["_next_score"] - pairs["_bb_rating"]).abs()
    correlation = None
    if (
        len(pairs) >= 3
        and pairs["_bb_rating"].nunique() >= 2
        and pairs["_next_score"].nunique() >= 2
    ):
        correlation = float(
            pairs[["_bb_rating", "_next_score"]].corr().iloc[0, 1]
        )

    return {
        "n_pairs": int(len(pairs)),
        "mean_abs_change": float(changes.mean()),
        "median_abs_change": float(changes.median()),
        "share_abs_change_le_5": float((changes <= 5).mean()),
        "share_abs_change_le_10": float((changes <= 10).mean()),
        "score_correlation": correlation,
    }


def _diagnostic_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    contexts = report["contexts"]
    if contexts["share_below_min_peer_samples"] > 0.50:
        warnings.append(
            "Oltre il 50% dei contesti ha meno del numero minimo di peer; "
            "il fallback può essere frequente."
        )

    for metric in report["metrics"]:
        if metric["rating_enabled"] and metric["missing_pct"] > 0.30:
            warnings.append(
                f"{metric['metric']}: oltre il 30% delle righe non ha un valore "
                "utilizzabile; la copertura va verificata prima di fissare il peso."
            )

    score = report["score_distribution"]
    if score["n"]:
        if score["stddev"] is not None and score["stddev"] < 8.0:
            warnings.append(
                "La deviazione standard del BB-Rating complessivo è inferiore "
                "a 8 punti: verificare una possibile compressione."
            )
        if score["share_ceiling_100"] > 0.05:
            warnings.append(
                "Più del 5% dei rating è esattamente 100: verificare una "
                "possibile saturazione sul soffitto."
            )
        if score["share_floor_1"] > 0.05:
            warnings.append(
                "Più del 5% dei rating è esattamente 1: verificare una "
                "possibile saturazione sul pavimento."
            )
    return warnings


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BB-Rating Calibration Report",
        "",
        f"- Calibration version: {report['calibration_version']}",
        f"- BB-Rating version tested: {report['bb_rating_version']}",
        f"- Rows: **{report['dataset']['rows']}**",
        f"- Players: **{report['dataset']['players']}**",
        f"- Seasons: **{report['dataset']['season_min']} -> {report['dataset']['season_max']}**",
        "",
        "## Scope",
        "",
        "Diagnostic only. No production weights, peer thresholds, model artifacts, "
        "PostgreSQL source data, or Prediction Model logic is changed.",
        "",
        "## Registry audit",
        "",
        "| Metric | Column | Exists | Semantic | Weight | Direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["registry_audit"]:
        semantic_ok = row["semantic_exists"] and row["semantic_source_matches"]
        lines.append(
            f"| {row['metric']} | {row['source_column']} | "
            f"{'yes' if row['column_exists'] else 'NO'} | "
            f"{'yes' if semantic_ok else 'NO'} | {row['weight']:.4f} | "
            f"{row['direction']} |"
        )

    lines += [
        "",
        "## Context and peer populations",
        "",
        f"- Contexts: **{report['contexts']['n_contexts']}**",
        f"- Median context size: **{report['contexts']['median_context_size']:.1f}**",
        f"- Contexts below minimum peer size: **{report['contexts']['n_below_min_peer_samples']}** "
        f"({report['contexts']['share_below_min_peer_samples']:.1%})",
        "",
        "| Peer source | Rows | Contexts | Median peer size |",
        "|---|---:|---:|---:|",
    ]
    for row in report["peer_sources"]:
        lines.append(
            f"| {row['peer_source']} | {row['rows']} | {row['contexts']} | "
            f"{row['median_peer_sample_size']:.1f} |"
        )

    lines += [
        "",
        "## Metric coverage and distributions",
        "",
        "| Metric | Weight | Available | Missing | Zero among available | P50 percentile | Effective weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        summary = row["percentile_summary"]
        p50 = summary["p50"]
        effective = row["effective_weight_mean"]
        lines.append(
            f"| {row['metric']} | {row['weight']:.4f} | "
            f"{row['n_available']}/{row['n_rows']} | {row['missing_pct']:.1%} | "
            f"{'—' if row['zero_pct_available'] is None else f'{row['zero_pct_available']:.1%}'} | "
            f"{'—' if p50 is None else f'{p50:.3f}'} | "
            f"{'—' if effective is None else f'{effective:.4f}'} |"
        )

    score = report["score_distribution"]
    lines += [
        "",
        "## BB-Rating distribution",
        "",
        f"- N: **{score['n']}**",
        f"- Mean: **{score['mean']:.2f}**",
        f"- Stddev: **{score['stddev']:.2f}**",
        f"- P10/P50/P90: **{score['p10']:.1f} / {score['p50']:.1f} / {score['p90']:.1f}**",
        f"- Share 81–100: **{score['share_81_100']:.1%}**",
        f"- Share 1–20: **{score['share_1_20']:.1%}**",
        "",
        "## Stability",
        "",
        f"- Consecutive player-season pairs: **{report['stability']['n_pairs']}**",
        f"- Mean absolute change: **{report['stability']['mean_abs_change'] if report['stability']['mean_abs_change'] is not None else '—'}**",
        f"- Median absolute change: **{report['stability']['median_abs_change'] if report['stability']['median_abs_change'] is not None else '—'}**",
        f"- Correlation: **{report['stability']['score_correlation'] if report['stability']['score_correlation'] is not None else '—'}**",
        "",
        "## Diagnostic warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No heuristic diagnostic warnings triggered.")
    return "\n".join(lines) + "\n"


def build_calibration_report(
    data: dict[str, Any],
    *,
    config: BBRatingCalibrationConfig | None = None,
) -> dict[str, Any]:
    config = config or BBRatingCalibrationConfig()
    min_peer_samples = max(5, int(config.min_peer_samples))
    min_context_samples = max(1, int(config.min_context_samples))

    engine = BBRatingEngine(
        data.get("player_stats", pd.DataFrame()),
        data.get("players", pd.DataFrame()),
        min_peer_samples=min_peer_samples,
    )
    frame = engine.frame.copy()
    if frame.empty:
        raise ValueError("BB-Rating calibration requires a non-empty player_stats frame")

    frame = _assign_peer_groups(
        frame,
        min_peer_samples=min_peer_samples,
        min_context_samples=min_context_samples,
    )

    registry_audit = _registry_audit(frame)

    for spec in BB_RATING_METRICS:
        if spec.source_column not in frame.columns:
            frame[f"_value_{spec.key}"] = np.nan
            frame[f"_pct_{spec.key}"] = np.nan
            if spec.weight > 0:
                frame[f"_effective_weight_{spec.key}"] = np.nan
            continue

        values, percentiles = _metric_percentiles(frame, spec)
        frame[f"_value_{spec.key}"] = values
        frame[f"_pct_{spec.key}"] = percentiles

    scoring_specs = [spec for spec in BB_RATING_METRICS if spec.weight > 0]
    used_weight = pd.Series(0.0, index=frame.index)
    numerator = pd.Series(0.0, index=frame.index)

    for spec in scoring_specs:
        percentiles = frame[f"_pct_{spec.key}"]
        available = percentiles.notna()
        weight = float(spec.weight)
        used_weight = used_weight + available.astype(float) * weight
        numerator = numerator + percentiles.fillna(0.0) * weight

    frame["_composite_pct"] = (
        numerator / used_weight.replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    frame["_bb_rating"] = _score_from_percentile(frame["_composite_pct"])

    for spec in scoring_specs:
        available = frame[f"_pct_{spec.key}"].notna() & used_weight.gt(0)
        effective = pd.Series(np.nan, index=frame.index, dtype=float)
        effective.loc[available] = float(spec.weight) / used_weight.loc[available]
        frame[f"_effective_weight_{spec.key}"] = effective

    metrics = _metric_summary(frame, registry_audit)

    context_sizes = (
        frame.groupby(
            ["league_key", "season", "competition"], dropna=False
        ).size().rename("context_sample_size").reset_index()
    )
    below = int(
        (context_sizes["context_sample_size"] < min_peer_samples).sum()
    ) if not context_sizes.empty else 0

    peer_sources: list[dict[str, Any]] = []
    for source, group in frame.groupby("_peer_source", dropna=False, sort=True):
        peer_sources.append(
            {
                "peer_source": str(source),
                "rows": int(len(group)),
                "contexts": int(
                    group[["league_key", "season", "competition"]]
                    .drop_duplicates().shape[0]
                ),
                "median_peer_sample_size": float(group["_peer_sample_size"].median()),
                "mean_peer_sample_size": float(group["_peer_sample_size"].mean()),
                "min_peer_sample_size": int(group["_peer_sample_size"].min()),
                "max_peer_sample_size": int(group["_peer_sample_size"].max()),
            }
        )

    seasons = pd.to_numeric(frame["season"], errors="coerce").dropna()
    valid_ids = frame["player_global_id"].replace({"nan": np.nan}).dropna()

    report: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "bb_rating_version": BB_RATING_VERSION,
        "dataset": {
            "rows": int(len(frame)),
            "players": int(valid_ids.nunique()),
            "leagues": sorted(str(v) for v in frame["league_key"].dropna().unique()),
            "seasons": sorted(int(v) for v in seasons.unique()),
            "season_min": int(seasons.min()),
            "season_max": int(seasons.max()),
            "competitions": sorted(str(v) for v in frame["competition"].dropna().unique()),
            "source_contract": data.get("source_contract"),
        },
        "configuration": {
            "min_peer_samples": min_peer_samples,
            "min_context_samples": min_context_samples,
            "age_bands": [list(v) for v in AGE_BANDS],
        },
        "registry_audit": registry_audit,
        "contexts": {
            "n_contexts": int(len(context_sizes)),
            "median_context_size": (
                float(context_sizes["context_sample_size"].median())
                if not context_sizes.empty else 0.0
            ),
            "mean_context_size": (
                float(context_sizes["context_sample_size"].mean())
                if not context_sizes.empty else 0.0
            ),
            "n_below_min_peer_samples": below,
            "share_below_min_peer_samples": (
                float(below / len(context_sizes)) if len(context_sizes) else 0.0
            ),
        },
        "peer_sources": peer_sources,
        "metrics": metrics,
        "score_distribution": _score_summary(frame),
        "by_league": _group_score_summary(frame, ["league_key"]),
        "by_season": _group_score_summary(frame, ["season"]),
        "by_competition": _group_score_summary(frame, ["competition"]),
        "by_league_season": _group_score_summary(frame, ["league_key", "season"]),
        "stability": _stability(frame),
    }

    report["warnings"] = _diagnostic_warnings(report)
    report["validation_signals"] = {
        "registry_columns_ok": all(
            row["column_exists"]
            and row["semantic_exists"]
            and row["semantic_source_matches"]
            for row in registry_audit
        ),
        "scoring_metrics_present": all(
            row["column_exists"]
            for row in registry_audit
            if row["rating_enabled"]
        ),
        "score_available_rate": (
            float(frame["_bb_rating"].notna().mean()) if len(frame) else 0.0
        ),
        "peer_fallback_share": (
            float((~frame["_peer_source"].eq("position+age+role")).mean())
            if len(frame) else 1.0
        ),
        "explainable_metric_count": int(
            sum(row["semantic_exists"] for row in registry_audit)
        ),
    }
    return report


def write_calibration_report(
    report: dict[str, Any],
    out_dir: str,
) -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bb_rating_calibration.json"
    md_path = root / "bb_rating_calibration.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

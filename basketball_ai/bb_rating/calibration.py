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

CALIBRATION_VERSION = "1.7"


@dataclass(frozen=True)
class BBRatingCalibrationConfig:
    min_peer_samples: int = MIN_PEER_SAMPLES
    min_context_samples: int = MIN_CONTEXT_SAMPLES


def _normalise_series(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Canonical AI_Source stores unavailable NET_RTG_DIFF as 0 to preserve
    # the Prediction Model's numeric contract. BB-Rating treats that value as
    # missing because the audited authoritative On/Off source has no genuine
    # zero differential in 2018-2025. OnNetRtg/OffNetRtg themselves may
    # legitimately be exactly zero and must not be masked.
    if column == "net_rtg_diff":
        values = values.mask(values.abs() < 1e-12)
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
        role_only_counts = ctx.loc[
            ctx["ruolo_combinato"].astype(str).str.strip().ne("")
        ].groupby(
            ["position_family", "ruolo_combinato"], dropna=False
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
            role_only_count = role_only_counts.get((position, role), 0) if role else 0
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
            elif role and role_only_count >= min_peer_samples:
                source = "position+role"
                size = role_only_count
                peer_key = (
                    f"{league}|{int(season)}|{competition}|position+role|"
                    f"{position}|{role}"
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
    if spec.key == "NET_RTG_DIFF":
        on = pd.to_numeric(frame.get("on_net_rtg"), errors="coerce")
        off = pd.to_numeric(frame.get("off_net_rtg"), errors="coerce")
        complete_onoff = on.notna() & off.notna()
        values = values.where(complete_onoff)
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



def _role_peer_population_diagnostics(
    frame: pd.DataFrame,
    *,
    min_peer_samples: int,
) -> dict[str, Any]:
    valid = frame.loc[
        frame["age_band"].astype(str).str.strip().ne("")
        & frame["ruolo_combinato"].astype(str).str.strip().ne("")
    ].copy()
    group_cols = [
        "league_key",
        "season",
        "competition",
        "position_family",
        "age_band",
        "ruolo_combinato",
    ]
    if valid.empty:
        return {
            "rows_with_role_and_age": 0,
            "role_present_share": 0.0,
            "distinct_role_groups": 0,
            "group_size": {
                "mean": None,
                "median": None,
                "p90": None,
                "max": None,
            },
            "thresholds": {},
            "role_only_groups": 0,
            "role_only_group_size": {
                "mean": None,
                "median": None,
                "p90": None,
                "max": None,
            },
            "role_only_thresholds": {},
            "configured_min_peer_samples": int(min_peer_samples),
        }

    counts = valid.groupby(group_cols, dropna=False).size().rename("n")
    role_only = valid.loc[
        valid["ruolo_combinato"].astype(str).str.strip().ne("")
    ].groupby(
        ["league_key", "season", "competition", "position_family", "ruolo_combinato"],
        dropna=False,
    ).size().rename("n")
    thresholds = {}
    role_only_thresholds = {}
    for threshold in (5, 10, 15, 20, 25):
        eligible = counts.ge(threshold)
        thresholds[str(threshold)] = {
            "groups": int(eligible.sum()),
            "share_groups": float(eligible.mean()),
            "rows_eligible": int(counts.loc[eligible].sum()) if eligible.any() else 0,
            "share_rows": (
                float(counts.loc[eligible].sum() / len(valid))
                if len(valid) else 0.0
            ),
        }
        role_eligible = role_only.ge(threshold)
        role_only_thresholds[str(threshold)] = {
            "groups": int(role_eligible.sum()),
            "share_groups": (
                float(role_eligible.mean()) if len(role_only) else 0.0
            ),
            "rows_eligible": (
                int(role_only.loc[role_eligible].sum())
                if role_eligible.any() else 0
            ),
            "share_rows": (
                float(role_only.loc[role_eligible].sum() / len(valid))
                if len(valid) and role_eligible.any() else 0.0
            ),
        }

    return {
        "rows_with_role_and_age": int(len(valid)),
        "role_present_share": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "distinct_role_groups": int(len(counts)),
        "group_size": {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "p90": float(counts.quantile(0.90)),
            "max": int(counts.max()),
        },
        "thresholds": thresholds,
        "role_only_groups": int(len(role_only)),
        "role_only_group_size": {
            "mean": float(role_only.mean()) if len(role_only) else None,
            "median": float(role_only.median()) if len(role_only) else None,
            "p90": float(role_only.quantile(0.90)) if len(role_only) else None,
            "max": int(role_only.max()) if len(role_only) else None,
        },
        "role_only_thresholds": role_only_thresholds,
        "configured_min_peer_samples": int(min_peer_samples),
    }


def _exposure_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    def numeric_summary(column: str) -> dict[str, Any]:
        if column not in frame.columns:
            return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
        return {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "p10": float(values.quantile(0.10)),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
        }

    return {
        "games_played": numeric_summary("games_played"),
        "minutes_per_game": numeric_summary("minutes_per_game"),
    }



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
        if (
            metric["rating_enabled"]
            and metric["n_available"]
            and metric["zero_pct_available"] is not None
            and metric["zero_pct_available"] >= 0.99
        ):
            warnings.append(
                f"{metric['metric']}: almeno il 99% dei valori disponibili è esattamente "
                "zero; verificare se il campo è un placeholder di copertura anziché "
                "un segnale discriminante."
            )
        percentile_summary = metric["percentile_summary"]
        if (
            metric["rating_enabled"]
            and percentile_summary["n"] >= 50
            and percentile_summary["stddev"] is not None
            and percentile_summary["stddev"] < 0.02
        ):
            warnings.append(
                f"{metric['metric']}: la distribuzione dei percentili è quasi costante; "
                "il contributo al BB-Rating è attualmente poco discriminante."
            )

    net_diff = next((m for m in report["metrics"] if m["metric"] == "NET_RTG_DIFF"), None)
    if net_diff and net_diff["rating_enabled"] and net_diff["zero_pct_available"] is not None:
        if net_diff["zero_pct_available"] >= 0.30:
            warnings.append(
                "NET_RTG_DIFF: oltre il 30% dei valori disponibili è esattamente zero; "
                "verificare la provenienza On/Off prima di consolidare il peso."
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
        "## BB-Rating data source",
        "",
        "BB-Rating reads the canonical observed contract. On/Off fields are "
        "resolved upstream by ai_source_full.sql from the Analisi On/Off tables "
        "when the primary observation is NULL/zero and an authoritative source "
        "value is available.",
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
            f"| {row['peer_source']} | {row['rows']} ({row['share_rows']:.1%}) | "
            f"{row['contexts']} ({row['share_contexts']:.1%}) | "
            f"{row['median_peer_sample_size']:.1f} |"
        )

    role_diag = report["role_peer_population"]
    lines += [
        "",
        "### Role peer population diagnostics",
        "",
        f"- Rows with role + age: **{role_diag['rows_with_role_and_age']}** "
        f"({role_diag['role_present_share']:.1%})",
        f"- Distinct position+age+role groups: **{role_diag['distinct_role_groups']}**",
        f"- Median role-group size: **{role_diag['group_size']['median']:.1f}**",
        f"- P90 role-group size: **{role_diag['group_size']['p90']:.1f}**",
        f"- Position+role groups (age ignored): **{role_diag['role_only_groups']}**",
        f"- Median position+role group size: **{role_diag['role_only_group_size']['median']:.1f}**"
        if role_diag["role_only_group_size"]["median"] is not None else
        "- Median position+role group size: **—**",
        "",
        "| Minimum age+role group size | Groups | Rows eligible |",
        "|---:|---:|---:|",
    ]
    for threshold, row in role_diag["thresholds"].items():
        lines.append(
            f"| {threshold} | {row['groups']} ({row['share_groups']:.1%}) | "
            f"{row['rows_eligible']} ({row['share_rows']:.1%}) |"
        )
    lines += [
        "",
        "| Minimum position+role group size | Groups | Rows eligible |",
        "|---:|---:|---:|",
    ]
    for threshold, row in role_diag["role_only_thresholds"].items():
        lines.append(
            f"| {threshold} | {row['groups']} ({row['share_groups']:.1%}) | "
            f"{row['rows_eligible']} ({row['share_rows']:.1%}) |"
        )

    lines += [
        "",
        "### Exposure diagnostics",
        "",
        f"- Games played P10/P50/P90: **{report['exposure']['games_played']['p10']} / "
        f"{report['exposure']['games_played']['p50']} / "
        f"{report['exposure']['games_played']['p90']}**",
        f"- Minutes per game P10/P50/P90: **{report['exposure']['minutes_per_game']['p10']} / "
        f"{report['exposure']['minutes_per_game']['p50']} / "
        f"{report['exposure']['minutes_per_game']['p90']}**",
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
    total_rows = len(frame)
    total_contexts = len(context_sizes)
    for source, group in frame.groupby("_peer_source", dropna=False, sort=True):
        source_contexts = int(
            group[["league_key", "season", "competition"]]
            .drop_duplicates().shape[0]
        )
        peer_sources.append(
            {
                "peer_source": str(source),
                "rows": int(len(group)),
                "share_rows": float(len(group) / total_rows) if total_rows else 0.0,
                "contexts": source_contexts,
                "share_contexts": (
                    float(source_contexts / total_contexts)
                    if total_contexts else 0.0
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
        "role_peer_population": _role_peer_population_diagnostics(
            frame,
            min_peer_samples=min_peer_samples,
        ),
        "exposure": _exposure_diagnostics(frame),
        "metrics": metrics,
        "score_distribution": _score_summary(frame),
        "by_league": _group_score_summary(frame, ["league_key"]),
        "by_season": _group_score_summary(frame, ["season"]),
        "by_competition": _group_score_summary(frame, ["competition"]),
        "by_league_season": _group_score_summary(frame, ["league_key", "season"]),
        "stability": _stability(frame),
    }

    report["warnings"] = _diagnostic_warnings(report)
    scoring_registry_ok = all(
        row["column_exists"]
        and row["semantic_exists"]
        and row["semantic_source_matches"]
        for row in registry_audit
        if row["rating_enabled"]
    )
    explanation_catalog_ok = all(
        row["semantic_exists"] and row["semantic_source_matches"]
        for row in registry_audit
    )
    source_by_name = {
        row["peer_source"]: row["share_rows"] for row in peer_sources
    }
    constant_scoring_metrics = [
        row["metric"]
        for row in metrics
        if row["rating_enabled"]
        and row["n_available"]
        and row["zero_pct_available"] is not None
        and row["zero_pct_available"] >= 0.99
    ]
    role_peer_share = (
        source_by_name.get("position+age+role", 0.0)
        + source_by_name.get("position+role", 0.0)
    ) if len(frame) else 0.0
    report["validation_signals"] = {
        "registry_columns_ok": scoring_registry_ok,
        "scoring_registry_ok": scoring_registry_ok,
        "explanation_catalog_ok": explanation_catalog_ok,
        "scoring_metrics_present": all(
            row["column_exists"]
            for row in registry_audit
            if row["rating_enabled"]
        ),
        "score_available_rate": (
            float(frame["_bb_rating"].notna().mean()) if len(frame) else 0.0
        ),
        "peer_fallback_share": (
            float(1.0 - role_peer_share)
            if len(frame) else 1.0
        ),
        "peer_source_shares": source_by_name,
        "role_peer_share": role_peer_share,
        "limited_context_share": source_by_name.get("limited_context", 0.0),
        "constant_scoring_metrics": constant_scoring_metrics,
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

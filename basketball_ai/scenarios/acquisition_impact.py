"""Walk-forward validated observational estimates of roster acquisition impact.

The model predicts next-season team Net Rating change from historical roster
transitions. It is deliberately fail-closed: it emits a numeric acquisition
estimate only when there is enough history and walk-forward predictions beat a
no-change baseline. These are calibrated observational associations, not causal
claims about what a signing alone will cause.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from basketball_ai.data.loader import _to_int
from basketball_ai.models.competition_training import normalize_competition
from basketball_ai.models.production_training import season_year


FEATURES = (
    "roster_quality_delta",
    "roster_turnover",
    "prior_net_rtg",
    "prior_net_rtg_change",
)
MIN_FINAL_SAMPLES = 40
MIN_WALK_FORWARD_SAMPLES = 20
MIN_TRAIN_SAMPLES_PER_FOLD = 20
MIN_WALK_FORWARD_SEASONS = 3
MIN_WALK_FORWARD_SAMPLES_PER_SEASON = 8
RIDGE_ALPHA = 0.5


def _year(value: Any) -> int | None:
    try:
        return int(season_year(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _id(value: Any) -> int | None:
    try:
        return _to_int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _identity_keys(data: dict[str, Any]) -> dict[int, str]:
    """Resolve alternate internal player IDs to a canonical source identity."""
    out: dict[int, str] = {}
    players = data.get("players", pd.DataFrame())
    if isinstance(players, pd.DataFrame) and {"id", "global_id"}.issubset(players.columns):
        for row in players[["id", "global_id"]].to_dict("records"):
            pid = _id(row.get("id"))
            raw_gid = row.get("global_id")
            gid = "" if raw_gid is None or pd.isna(raw_gid) else str(raw_gid).strip()
            if gid.lower() in {"nan", "none", "null"}:
                gid = ""
            if pid is not None:
                out[pid] = f"global:{gid}" if gid else f"player:{pid}"
    for raw_pid, meta in (data.get("player_dict", {}) or {}).items():
        pid = _id(raw_pid)
        if pid is None or not isinstance(meta, dict) or pid in out:
            continue
        raw_gid = meta.get("global_id")
        gid = "" if raw_gid is None or pd.isna(raw_gid) else str(raw_gid).strip()
        if gid.lower() in {"nan", "none", "null"}:
            gid = ""
        out[pid] = f"global:{gid}" if gid else f"player:{pid}"
    return out


def _mean_top_eight(roster: set[str], prior_ratings: dict[str, float]) -> float | None:
    ratings = sorted(
        (prior_ratings[key] for key in roster if key in prior_ratings and math.isfinite(prior_ratings[key])),
        reverse=True,
    )[:8]
    if len(ratings) < 5:
        return None
    return float(np.mean(ratings))


def build_team_transition_samples(
    data: dict[str, Any],
    league_id: int,
    competition: str,
    through_season: int,
) -> list[dict[str, float | int]]:
    """Build leakage-safe team-season outcomes from observed roster transitions.

    For target season t, every player-quality input comes from t-1; the outcome
    is Net Rating(t) - Net Rating(t-1). Only teams with explicit membership rows
    in both seasons and sufficient prior-season player profiles are retained.
    """
    relations = data.get("team_player_relations", pd.DataFrame())
    stats = data.get("player_stats", pd.DataFrame())
    teams = data.get("team_season_stats", pd.DataFrame())
    if not all(isinstance(frame, pd.DataFrame) and not frame.empty for frame in (relations, stats, teams)):
        return []
    if not {"player_id", "team_id", "season"}.issubset(relations.columns):
        return []
    if not {"player_id", "league_id", "season", "competition", "rating"}.issubset(stats.columns):
        return []
    if not {"team_id", "league_id", "season", "competition", "net_rtg"}.issubset(teams.columns):
        return []

    identity = _identity_keys(data)

    def player_key(value: Any) -> str:
        pid = _id(value)
        return identity.get(pid, f"player:{pid}") if pid is not None else ""

    comp = normalize_competition(competition)
    league = _id(league_id)
    if league is None:
        return []

    relation_rows = relations.copy()
    relation_rows["_year"] = relation_rows["season"].map(_year)
    relation_rows["_team"] = relation_rows["team_id"].map(_id)
    relation_rows["_player_key"] = relation_rows["player_id"].map(player_key)
    relation_rows = relation_rows[
        (relation_rows["_year"].notna())
        & (relation_rows["_year"] <= int(through_season))
        & (relation_rows["_team"].notna())
        & (relation_rows["_player_key"] != "")
    ]
    if "league_id" in relation_rows.columns:
        relation_rows = relation_rows[relation_rows["league_id"].map(_id) == league]
    rosters: dict[tuple[int, int], set[str]] = {}
    for (team_id, year), group in relation_rows.groupby(["_team", "_year"], sort=False):
        rosters[(int(team_id), int(year))] = set(group["_player_key"].tolist())

    player_stats = stats.copy()
    player_stats["_year"] = player_stats["season"].map(_year)
    player_stats["_league"] = player_stats["league_id"].map(_id)
    player_stats["_comp"] = player_stats["competition"].map(normalize_competition)
    player_stats["_player_key"] = player_stats["player_id"].map(player_key)
    player_stats["_rating"] = pd.to_numeric(player_stats["rating"], errors="coerce")
    player_stats["_games"] = pd.to_numeric(
        player_stats["games_played"] if "games_played" in player_stats.columns else 0,
        errors="coerce",
    )
    player_stats["_minutes"] = pd.to_numeric(
        player_stats["minutes_per_game"] if "minutes_per_game" in player_stats.columns else 0,
        errors="coerce",
    )
    player_stats = player_stats[
        (player_stats["_league"] == league)
        & (player_stats["_comp"] == comp)
        & (player_stats["_year"].notna())
        & (player_stats["_year"] <= int(through_season))
        & (player_stats["_player_key"] != "")
        & (player_stats["_rating"].notna())
    ].copy()
    player_stats = player_stats.sort_values(
        ["_games", "_minutes", "_rating"], ascending=[False, False, False], na_position="last"
    ).drop_duplicates(["_player_key", "_year"], keep="first")
    ratings_by_year = {
        int(year): dict(zip(group["_player_key"], group["_rating"].astype(float)))
        for year, group in player_stats.groupby("_year", sort=False)
    }

    team_stats = teams.copy()
    team_stats["_year"] = team_stats["season"].map(_year)
    team_stats["_team"] = team_stats["team_id"].map(_id)
    team_stats["_league"] = team_stats["league_id"].map(_id)
    team_stats["_comp"] = team_stats["competition"].map(normalize_competition)
    team_stats["_net_rtg"] = pd.to_numeric(team_stats["net_rtg"], errors="coerce")
    team_stats = team_stats[
        (team_stats["_league"] == league)
        & (team_stats["_comp"] == comp)
        & (team_stats["_year"].notna())
        & (team_stats["_year"] <= int(through_season))
        & (team_stats["_team"].notna())
        & (team_stats["_net_rtg"].notna())
    ].copy()
    team_stats = team_stats.sort_values("_year").drop_duplicates(["_team", "_year"], keep="last")
    nets = {
        (int(row["_team"]), int(row["_year"])): float(row["_net_rtg"])
        for row in team_stats.to_dict("records")
    }

    samples: list[dict[str, float | int]] = []
    for (team_id, target_year), current_net in sorted(nets.items(), key=lambda item: (item[0][1], item[0][0])):
        prior_year = target_year - 1
        if (team_id, prior_year) not in nets:
            continue
        current_roster = rosters.get((team_id, target_year), set())
        prior_roster = rosters.get((team_id, prior_year), set())
        if not current_roster or not prior_roster:
            continue

        prior_ratings = ratings_by_year.get(prior_year, {})
        new_quality = _mean_top_eight(current_roster, prior_ratings)
        old_quality = _mean_top_eight(prior_roster, prior_ratings)
        if new_quality is None or old_quality is None:
            continue

        union = current_roster | prior_roster
        turnover = len(current_roster ^ prior_roster) / max(1, len(union))
        prior_net = nets[(team_id, prior_year)]
        prior_prior_net = nets.get((team_id, prior_year - 1), prior_net)
        prior_change = prior_net - prior_prior_net if (team_id, prior_year - 1) in nets else 0.0
        values = (new_quality - old_quality, turnover, prior_net, prior_change)
        if not all(math.isfinite(float(value)) for value in values):
            continue
        samples.append({
            "season": int(target_year),
            "team_id": int(team_id),
            "roster_quality_delta": float(new_quality - old_quality),
            "roster_turnover": float(turnover),
            "prior_net_rtg": float(prior_net),
            "prior_net_rtg_change": float(prior_change),
            "target_net_rtg_change": float(current_net - prior_net),
        })
    return samples


def _fit_ridge(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 8:
        return None
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    y = np.asarray([float(row["target_net_rtg_change"]) for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1], dtype=float) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    try:
        inverse = np.linalg.inv(gram)
        beta = inverse @ design.T @ y
    except np.linalg.LinAlgError:
        return None
    fitted = design @ beta
    dof = max(1, len(y) - design.shape[1])
    sigma2 = float(np.sum((y - fitted) ** 2) / dof)
    covariance = inverse * sigma2
    return {
        "center": center,
        "scale": scale,
        "beta": beta,
        "covariance": covariance,
        "sigma": float(math.sqrt(max(0.0, sigma2))),
    }


def _predict_fit(fit: dict[str, Any], features: dict[str, float]) -> float:
    raw = np.asarray([float(features[name]) for name in FEATURES], dtype=float)
    z = (raw - fit["center"]) / fit["scale"]
    return float(fit["beta"][0] + z @ fit["beta"][1:])


def fit_acquisition_impact_model(
    data: dict[str, Any],
    league_id: int,
    competition: str,
    through_season: int,
) -> dict[str, Any]:
    """Fit ridge regression, then validate using expanding-window seasons.

    A validated status requires adequate history, >=20 out-of-time predictions
    across >=3 seasons, and lower walk-forward RMSE than a zero-change baseline.
    """
    rows = build_team_transition_samples(data, league_id, competition, through_season)
    if not rows:
        return {
            "status": "roster_history_unavailable",
            "reason": "Non sono disponibili transizioni stagionali complete di roster e Net Rating.",
            "training_samples": 0,
            "validation": {"samples": 0, "seasons": 0, "by_target_season": [], "scope": {
                "league_id": int(league_id),
                "competition": normalize_competition(competition),
                "through_season": int(through_season),
                "target_seasons": [],
                "design": "expanding_window_walk_forward",
                "causal_effect_established": False,
            }},
            "validation_scope": {
                "league_id": int(league_id),
                "competition": normalize_competition(competition),
                "through_season": int(through_season),
                "target_seasons": [],
                "design": "expanding_window_walk_forward",
                "causal_effect_established": False,
            },
        }

    years = sorted({int(row["season"]) for row in rows})
    out_of_time: list[dict[str, float | int]] = []
    for held_out_year in years:
        train = [row for row in rows if int(row["season"]) < held_out_year]
        test = [row for row in rows if int(row["season"]) == held_out_year]
        if len(train) < MIN_TRAIN_SAMPLES_PER_FOLD or len({int(row["season"]) for row in train}) < 3 or not test:
            continue
        fit = _fit_ridge(train)
        if fit is None:
            continue
        for row in test:
            features = {name: float(row[name]) for name in FEATURES}
            prediction = _predict_fit(fit, features)
            observed = float(row["target_net_rtg_change"])
            out_of_time.append({
                "season": int(held_out_year),
                "observed": observed,
                "prediction": prediction,
                "residual": observed - prediction,
            })

    all_oof_seasons = sorted({int(row["season"]) for row in out_of_time})
    oof_samples_by_season = {
        year: sum(int(row["season"]) == year for row in out_of_time)
        for year in all_oof_seasons
    }
    supported_oof_seasons = [
        year for year, count in oof_samples_by_season.items()
        if count >= MIN_WALK_FORWARD_SAMPLES_PER_SEASON
    ]
    under_supported_oof_seasons = [
        year for year, count in oof_samples_by_season.items()
        if count < MIN_WALK_FORWARD_SAMPLES_PER_SEASON
    ]
    if len(rows) < MIN_FINAL_SAMPLES:
        status = "insufficient_historical_transitions"
        reason = f"Servono almeno {MIN_FINAL_SAMPLES} transizioni storiche complete; disponibili {len(rows)}."
    elif (
        len(out_of_time) < MIN_WALK_FORWARD_SAMPLES
        or len(supported_oof_seasons) < MIN_WALK_FORWARD_SEASONS
    ):
        status = "insufficient_walk_forward_validation"
        reason = (
            "Il backtest walk-forward richiede almeno "
            f"{MIN_WALK_FORWARD_SEASONS} stagioni target con almeno "
            f"{MIN_WALK_FORWARD_SAMPLES_PER_SEASON} predizioni fuori campione ciascuna; "
            f"stagioni supportate: {len(supported_oof_seasons)}."
        )
    else:
        observed = np.asarray([float(row["observed"]) for row in out_of_time])
        predicted = np.asarray([float(row["prediction"]) for row in out_of_time])
        residuals = observed - predicted
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        baseline_rmse = float(np.sqrt(np.mean(np.square(observed))))
        mae = float(np.mean(np.abs(residuals)))
        baseline_mae = float(np.mean(np.abs(observed)))
        bias = float(np.mean(residuals))
        if rmse < baseline_rmse and abs(bias) < max(0.01, baseline_rmse * 0.5):
            status = "validated"
            reason = "Il backtest walk-forward supera il baseline a variazione zero."
        else:
            status = "baseline_not_beaten"
            reason = "Il backtest walk-forward non dimostra un miglioramento affidabile rispetto al baseline a variazione zero."

    validation: dict[str, Any] = {
        "samples": len(out_of_time),
        "seasons": len(all_oof_seasons),
        "supported_target_seasons": supported_oof_seasons,
        "under_supported_target_seasons": under_supported_oof_seasons,
        "minimum_samples_per_target_season": MIN_WALK_FORWARD_SAMPLES_PER_SEASON,
        "minimum_supported_target_seasons": MIN_WALK_FORWARD_SEASONS,
    }
    cohort_regressions: list[int] = []
    if out_of_time:
        observed = np.asarray([float(row["observed"]) for row in out_of_time])
        predicted = np.asarray([float(row["prediction"]) for row in out_of_time])
        residuals = observed - predicted
        validation.update({
            "rmse": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
            "baseline_rmse": round(float(np.sqrt(np.mean(np.square(observed)))), 4),
            "mae": round(float(np.mean(np.abs(residuals))), 4),
            "baseline_mae": round(float(np.mean(np.abs(observed))), 4),
            "bias": round(float(np.mean(residuals)), 4),
            "beats_zero_change_baseline": bool(
                float(np.sqrt(np.mean(np.square(residuals))))
                < float(np.sqrt(np.mean(np.square(observed))))
            ),
            "prediction_interval_80_residual_quantiles": {
                "lower": round(float(np.quantile(residuals, 0.10)), 4),
                "upper": round(float(np.quantile(residuals, 0.90)), 4),
            },
        })

        # A league/competition-level aggregate can hide a bad held-out season.
        # Publish season cohorts explicitly and fail closed on material regression
        # only when a cohort has enough out-of-time examples to be informative.
        by_target_season: list[dict[str, Any]] = []
        for held_out_year in sorted({int(row["season"]) for row in out_of_time}):
            cohort = [row for row in out_of_time if int(row["season"]) == held_out_year]
            cohort_observed = np.asarray([float(row["observed"]) for row in cohort])
            cohort_predicted = np.asarray([float(row["prediction"]) for row in cohort])
            cohort_residuals = cohort_observed - cohort_predicted
            cohort_rmse = float(np.sqrt(np.mean(np.square(cohort_residuals))))
            cohort_baseline_rmse = float(np.sqrt(np.mean(np.square(cohort_observed))))
            supported = len(cohort) >= 8
            cohort_status = "under_supported" if not supported else (
                "validated" if cohort_rmse < cohort_baseline_rmse else "baseline_not_beaten"
            )
            if supported and cohort_baseline_rmse > 1e-9 and cohort_rmse > cohort_baseline_rmse * 1.5:
                cohort_regressions.append(held_out_year)
                cohort_status = "material_regression"
            by_target_season.append({
                "target_season": held_out_year,
                "samples": len(cohort),
                "status": cohort_status,
                "rmse": round(cohort_rmse, 4),
                "baseline_rmse": round(cohort_baseline_rmse, 4),
                "mae": round(float(np.mean(np.abs(cohort_residuals))), 4),
                "bias": round(float(np.mean(cohort_residuals)), 4),
                "beats_zero_change_baseline": bool(cohort_rmse < cohort_baseline_rmse),
            })
        validation["by_target_season"] = by_target_season
        validation["cohort_min_samples"] = 8

        if status == "validated" and cohort_regressions:
            status = "cohort_regression"
            reason = (
                "Il risultato aggregato supera il baseline, ma almeno una stagione target "
                "con supporto sufficiente mostra una regressione rilevante: "
                + ", ".join(str(year) for year in cohort_regressions)
                + "."
            )

    validation_scope = {
        "league_id": int(league_id),
        "competition": normalize_competition(competition),
        "through_season": int(through_season),
        "target_seasons": sorted({int(row["season"]) for row in out_of_time}),
        "design": "expanding_window_walk_forward",
        "causal_effect_established": False,
    }
    validation["scope"] = validation_scope

    final_fit = _fit_ridge(rows)
    if final_fit is None:
        status = "model_fit_failed"
        reason = "Impossibile stimare il modello storico con i dati disponibili."

    result: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "method": "expanding_window_ridge_roster_transition",
        "training_samples": len(rows),
        "training_seasons": len(years),
        "validation": validation,
        "validation_scope": validation_scope,
        "feature_names": list(FEATURES),
        "reference_roster_turnover": float(np.median([float(row["roster_turnover"]) for row in rows])),
        "residual_interval_80": (
            validation.get("prediction_interval_80_residual_quantiles")
            if validation.get("prediction_interval_80_residual_quantiles")
            else None
        ),
    }
    if final_fit is not None:
        coefficients = {
            name: float(final_fit["beta"][index + 1] / final_fit["scale"][index])
            for index, name in enumerate(FEATURES)
        }
        result["_fit"] = final_fit
        result["coefficients"] = coefficients
        result["intercept_at_feature_means"] = float(final_fit["beta"][0])
    return result


def predict_acquisition_impact(
    model: dict[str, Any],
    *,
    roster_quality_delta: float,
    transaction_turnover_delta: float,
    prior_net_rtg: float,
    prior_net_rtg_change: float,
) -> dict[str, Any]:
    """Estimate modelled net-rating change with/without an addition.

    The difference is an observational association derived from the quality and
    turnover changes. It must not be communicated as a causal effect.
    """
    if model.get("status") != "validated" or not isinstance(model.get("_fit"), dict):
        return {
            "status": str(model.get("status") or "uncalibrated"),
            "available": False,
            "reason": str(model.get("reason") or "Il modello non ha superato la validazione fuori campione."),
            "validation": model.get("validation", {}),
            "validation_scope": model.get("validation_scope", {}),
            "training_samples": int(model.get("training_samples", 0)),
        }

    fit = model["_fit"]
    reference_turnover = float(model.get("reference_roster_turnover", 0.0))
    baseline_features = {
        "roster_quality_delta": 0.0,
        "roster_turnover": float(np.clip(reference_turnover, 0.0, 1.0)),
        "prior_net_rtg": float(prior_net_rtg),
        "prior_net_rtg_change": float(prior_net_rtg_change),
    }
    acquisition_features = {
        "roster_quality_delta": float(roster_quality_delta),
        "roster_turnover": float(np.clip(reference_turnover + transaction_turnover_delta, 0.0, 1.0)),
        "prior_net_rtg": float(prior_net_rtg),
        "prior_net_rtg_change": float(prior_net_rtg_change),
    }
    baseline_change = _predict_fit(fit, baseline_features)
    acquisition_change = _predict_fit(fit, acquisition_features)
    incremental = acquisition_change - baseline_change

    delta_raw = np.asarray([
        acquisition_features[name] - baseline_features[name] for name in FEATURES
    ], dtype=float)
    delta_z = delta_raw / fit["scale"]
    beta_covariance = fit["covariance"][1:, 1:]
    effect_variance = max(0.0, float(delta_z @ beta_covariance @ delta_z))
    effect_se = math.sqrt(effect_variance)
    critical_value_90 = 1.645
    effect_interval = {
        "lower": round(incremental - critical_value_90 * effect_se, 4),
        "upper": round(incremental + critical_value_90 * effect_se, 4),
        "level": 0.90,
        "type": "coefficient_uncertainty_interval_not_causal",
    }

    residual_interval = model.get("residual_interval_80") or {}
    outcome_interval = {
        "lower": round(acquisition_change + float(residual_interval.get("lower", 0.0)), 4),
        "upper": round(acquisition_change + float(residual_interval.get("upper", 0.0)), 4),
        "level": 0.80,
        "type": "walk_forward_residual_interval_for_team_change",
    }
    return {
        "status": "validated",
        "available": True,
        "method": str(model.get("method")),
        "estimated_incremental_net_rating": round(incremental, 4),
        "incremental_net_rating_interval_90": effect_interval,
        "predicted_team_net_rating_change_without_acquisition": round(baseline_change, 4),
        "predicted_team_net_rating_change_with_acquisition": round(acquisition_change, 4),
        "predicted_team_net_rating_change_interval_80": outcome_interval,
        "validation": model.get("validation", {}),
        "training_samples": int(model.get("training_samples", 0)),
        "training_seasons": int(model.get("training_seasons", 0)),
        "interpretation": (
            "Previsione fuori campione del cambiamento stagionale del Net Rating e "
            "associazione incrementale osservazionale; non identifica l'effetto causale del singolo acquisto."
        ),
    }

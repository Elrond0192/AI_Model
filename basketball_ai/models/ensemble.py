"""Basketball ensemble model.

Combines:
  1. XGBoost performance model (base rating)
  2. Age curve (trajectory adjustments)
  3. Compatibility model (style fit)
  4. Contextual adjustments (league, role, spacing, adaptation)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from basketball_ai.models.age_curve import age_performance_factor, maybe_fit_from_db


from basketball_ai.utils.helpers import normalize_id as _normalize_id
from basketball_ai.models.compatibility_model import CompatibilityModel
from basketball_ai.models.performance_model import (
    PerformanceModel, COMPETITION_ENCODING, compute_po_features,
)
from basketball_ai.features.player_features import compute_player_features
from basketball_ai.features.context_features import compute_context_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants – centralised so that calibration changes require edits in
# one place only.
# ---------------------------------------------------------------------------

# Age-curve ratio bounds: caps how much the base rating can be amplified (or
# shrunk) when projecting a player to a different age.  A ceiling of 1.5×
# prevents unrealistically high peak projections for post-peak players.
_AGE_RATIO_MIN: float = 0.50
_AGE_RATIO_MAX: float = 1.50

# Contextual fit multiplier: maps the weighted ctx_score onto a rating
# multiplier with a wider range than the old ±2% so that position/style fit
# creates meaningful differentiation between teams for different players.
_CTX_MULT_OFFSET: float = 0.78   # value at ctx_score == 0.50
_CTX_MULT_SLOPE:  float = 0.55   # sensitivity to ctx_score above the anchor
_CTX_MULT_LO:     float = 0.72   # hard lower bound
_CTX_MULT_HI:     float = 1.08   # hard upper bound

# Playing-time factor: scales down ratings for bench players with low minutes.
# The baseline is calibrated per-instance from real data during training
# (75th percentile of MPG).  The value below is only the initial fallback
# before any model is trained.
_MPG_FACTOR_BASE:  float = 0.60   # factor at 0 min/game (theoretical floor)
_MPG_FACTOR_RANGE: float = 0.40   # additive range: base + range = 1.0 at baseline
_MPG_BASELINE_DEFAULT: float = 30.0  # module-level default; overridden per-instance


@dataclass
class PredictionResult:
    """Full prediction output for one player-team scenario."""
    player_id: int
    team_id: int
    season: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    base_rating: float
    age_factor: float
    compatibility_factor: float
    league_factor: float
    context_adjustment: float
    shap_values: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    competition: str = "RS"


class EnsembleModel:
    """Combines all sub-models into a single basketball prediction pipeline."""

    def __init__(
        self,
        performance_model: Optional[PerformanceModel] = None,
        compatibility_model: Optional[CompatibilityModel] = None,
        cache_maxsize: int = 512,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.perf_model   = performance_model   or PerformanceModel()
        self.compat_model = compatibility_model or CompatibilityModel()
        # Split-conformal calibration quantiles [q_lo, q_hi] stored after training.
        # When None the heuristic fallback is used.
        self._conformal_q_lo: Optional[float] = None
        self._conformal_q_hi: Optional[float] = None
        # Reference distribution for PSI drift monitoring
        self._drift_reference: Optional[Dict] = None
        # Pre-calibrated per-league quality factors (from competitiveness_score)
        self._league_factors: Dict = {}
        # MPG baseline: 75th-percentile minutes/game from training data.
        # Stored as an instance attribute (not a global) so multiple EnsembleModel
        # instances don't interfere with each other.
        self._mpg_baseline: float = _MPG_BASELINE_DEFAULT
        # In-memory LRU prediction cache with TTL.
        # Uses collections.OrderedDict for O(1) move_to_end / popitem operations.
        self._cache_maxsize = int(os.environ.get("CACHE_MAXSIZE", cache_maxsize))
        self._cache_ttl     = int(os.environ.get("CACHE_TTL_SECONDS", cache_ttl_seconds))
        self._prediction_cache: OrderedDict = OrderedDict()  # key → (result, timestamp)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> None:
        """Train all sub-models and calibrate conformal prediction intervals."""
        logger.info("[Ensemble] Training performance model …")
        metrics = self.perf_model.train(data)
        # Extract the conformal holdout residuals produced by the 3-way split
        # inside PerformanceModel.train().  Removing them from metrics keeps the
        # returned dict clean (just numeric scores).
        conformal_residuals = metrics.pop("conformal_residuals", None)
        # Fit age-curve parameters from real data (Fix 7: replaces hardcoded sigmas)
        try:
            if maybe_fit_from_db(data):
                logger.info("[Ensemble] Age curve parameters fitted from data.")
        except Exception as exc:
            logger.warning("[Ensemble] Age curve fit from data failed: %s", exc)
        logger.info("[Ensemble] Training compatibility model …")
        self.compat_model.train(data)
        self._calibrate_conformal(conformal_residuals)
        # Calibrate data-derived constants (no hardcoded values)
        self._calibrate_mpg_baseline(data)
        self._calibrate_league_factors(data)
        try:
            from basketball_ai.features.team_features import calibrate_style_bounds
            bounds = calibrate_style_bounds(data)
            logger.info("[Ensemble] Style normalization bounds calibrated from data: %s", bounds)
        except Exception as exc:
            logger.warning("[Ensemble] Style bound calibration failed: %s", exc)
        # Capture reference distribution for drift monitoring
        try:
            from basketball_ai.monitoring.drift import capture_reference
            self._drift_reference = capture_reference(data)
        except Exception as exc:
            logger.warning("[Ensemble] Could not capture drift reference: %s", exc)
            # _drift_reference remains None (set in __init__)
        logger.info("[Ensemble] Training complete.")
        return metrics

    def _calibrate_mpg_baseline(self, data: Dict[str, Any]) -> None:
        """Set self._mpg_baseline to the 75th percentile of MPG in the training data."""
        try:
            stats_df = data.get("player_stats")
            if stats_df is not None and "minutes_per_game" in stats_df.columns:
                vals = stats_df["minutes_per_game"].dropna()
                if not vals.empty:
                    baseline = float(np.percentile(vals, 75))
                    self._mpg_baseline = max(baseline, 20.0)  # floor of 20 min
                    logger.info("[Ensemble] MPG baseline calibrated from data: %.1f min/game", self._mpg_baseline)
        except Exception as exc:
            logger.warning("[Ensemble] MPG baseline calibration failed: %s", exc)

    def _calibrate_league_factors(self, data: Dict[str, Any]) -> None:
        """Pre-compute per-league quality factors from competitiveness_score."""
        try:
            leagues_df = data.get("leagues")
            if leagues_df is None or leagues_df.empty:
                return
            if "competitiveness_score" not in leagues_df.columns:
                return
            max_cs = float(leagues_df["competitiveness_score"].max())
            if max_cs <= 0:
                return
            self._league_factors = {
                str(row.get("id", "")): float(
                    np.clip(float(row.get("competitiveness_score", 1.0) or 1.0) / max_cs, 0.5, 1.0)
                )
                for _, row in leagues_df.iterrows()
            }
            logger.info(
                "[Ensemble] League factors calibrated from data (%d leagues): %s",
                len(self._league_factors), self._league_factors,
            )
        except Exception as exc:
            logger.warning("[Ensemble] League factor calibration failed: %s", exc)
            # _league_factors remains {} (set in __init__)

    def _calibrate_conformal(self, conformal_residuals: Optional[List[float]]) -> None:
        """Calibrate split-conformal prediction intervals from held-out residuals.

        The residuals **must** come from a set that was never used for model
        fitting or early-stopping (i.e. the conformal holdout produced by
        ``PerformanceModel.train()``'s internal 3-way split).

        If *conformal_residuals* is ``None`` or contains fewer than 10 samples
        the heuristic fallback CI remains in effect.

        Args:
            conformal_residuals: List of ``|y_true - y_pred|`` values from the
                conformal holdout set, as returned in
                ``PerformanceModel.train()`` metrics.
        """
        if not conformal_residuals or len(conformal_residuals) < 10:
            logger.warning(
                "[Ensemble] Conformal calibration skipped: too few residuals (%d). "
                "Heuristic CI will be used.",
                len(conformal_residuals) if conformal_residuals else 0,
            )
            return
        try:
            residuals = np.array(conformal_residuals, dtype=float)
            self._conformal_q_lo = float(np.percentile(residuals, 5))
            self._conformal_q_hi = float(np.percentile(residuals, 95))
            logger.info(
                "[Ensemble] Conformal calibration: q_lo=%.4f  q_hi=%.4f  (n=%d held-out samples)",
                self._conformal_q_lo, self._conformal_q_hi, len(residuals),
            )
        except Exception as exc:
            logger.warning("[Ensemble] Conformal calibration failed, using heuristic CI: %s", exc)
            self._conformal_q_lo = None
            self._conformal_q_hi = None

    # ------------------------------------------------------------------
    # Prediction cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: tuple) -> Optional[Any]:
        """Return cached prediction or None if missing/expired."""
        if key not in self._prediction_cache:
            return None
        result, ts = self._prediction_cache[key]
        if self._cache_ttl > 0 and (time.monotonic() - ts) > self._cache_ttl:
            # Expired – remove and return miss
            del self._prediction_cache[key]
            return None
        # O(1) LRU update: move the key to the most-recently-used end.
        self._prediction_cache.move_to_end(key)
        return result

    def _cache_set(self, key: tuple, result: Any) -> None:
        """Store a prediction result in the LRU cache."""
        if self._cache_maxsize <= 0:
            return  # caching disabled
        # Overwrite the entry (new timestamp) and move it to the MRU end.
        self._prediction_cache[key] = (result, time.monotonic())
        self._prediction_cache.move_to_end(key)
        # Evict the least-recently-used entry when over capacity.
        while len(self._prediction_cache) > self._cache_maxsize:
            self._prediction_cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Invalidate the entire prediction cache (e.g. after retraining)."""
        self._prediction_cache.clear()
        logger.info("[Ensemble] Prediction cache cleared.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: Optional[int] = None,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ) -> PredictionResult:
        """Generate a full basketball prediction for a player in a given team.

        Results are cached in-memory (LRU) with a configurable TTL.  The cache
        key is ``(player_id, team_id, season, target_age, competition)``.
        Data changes are NOT automatically detected; call ``clear_cache()`` if
        the underlying data is reloaded.

        Args:
            player_id:   Target player.
            team_id:     Target team.
            data:        Flat data dict from loader.load_all_data().
            season:      Season reference year.
            target_age:  Override age (used for trajectory projections).
            competition: Competition context – "RS" (Regular Season), "PO" (Playoffs),
                         "CUP", or "SUPERCUP".  The model was trained on data that
                         includes competition as a feature, so this changes the
                         predicted rating accordingly.

        Returns:
            PredictionResult with predicted rating and full breakdown.
        """
        if season is None:
            seasons = data.get("player_stats", pd.DataFrame()).get("season", pd.Series(dtype=object)).dropna()
            if seasons.empty:
                raise ValueError("season is required when data has no season")
            season = max(int(str(value).split("-")[0]) for value in seasons) + 1
        cache_key = (_normalize_id(player_id), _normalize_id(team_id), season, target_age, competition)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        result = self._predict_uncached(player_id, team_id, data, season, target_age, competition)
        self._cache_set(cache_key, result)
        return result

    def _predict_uncached(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: int,
        target_age: Optional[int] = None,
        competition: str = "RS",
    ) -> PredictionResult:
        """Internal uncached implementation of predict()."""
        player_row   = data["player_dict"].get(_normalize_id(player_id), {})
        position     = str(player_row.get("position", "PG"))
        current_age  = int(player_row.get("age", 26))
        age          = target_age if target_age is not None else current_age

        # 1. Player feature vector for XGBoost
        player_feats = compute_player_features(player_id, data, season=season, target_age=age)

        # Inject competition-aware features that the model was trained on
        player_feats["competition_enc"] = float(COMPETITION_ENCODING.get(competition, 0))

        # Inject DB role encoding – all three dimensions
        p_stats  = data["player_stats"][data["player_stats"]["player_id"] == _normalize_id(player_id)]
        if not p_stats.empty:
            latest_row          = p_stats.sort_values("season").iloc[-1]
            latest_role         = str(latest_row.get("ruolo_combinato",  "") or "").strip()
            latest_role_off     = str(latest_row.get("ruolo_offensivo",  "") or "").strip()
            latest_role_def     = str(latest_row.get("ruolo_difensivo",  "") or "").strip()
            latest_games_played = float(latest_row.get("games_played", 0) or 0)
            latest_mpg          = float(latest_row.get("minutes_per_game", 20) or 20)
        else:
            latest_role = latest_role_off = latest_role_def = ""
            latest_games_played = 0.0
            latest_mpg = 20.0
        # Explicit persistence features must use only the player history
        # available in the scoped/as-of prediction context. This mirrors the
        # source-season features built by PerformanceModel._precompute_history_features.
        rating_history = (
            pd.to_numeric(
                p_stats.sort_values("season")["rating"],
                errors="coerce",
            )
            .ffill()
            .fillna(6.5)
            .to_numpy(dtype=float)
        )
        if len(rating_history):
            last_rating = float(rating_history[-1])
            rating_delta_1 = float(rating_history[-1] - rating_history[-2]) if len(rating_history) >= 2 else 0.0
            rating_delta_2 = float(rating_history[-1] - rating_history[-3]) if len(rating_history) >= 3 else 0.0
            recent = rating_history[-3:]
            recent_rating_mean = float(np.mean(recent))
            recent_rating_std = float(np.std(recent))
        else:
            last_rating = 6.5
            rating_delta_1 = 0.0
            rating_delta_2 = 0.0
            recent_rating_mean = 6.5
            recent_rating_std = 0.0
        player_feats["last_rating"] = last_rating
        player_feats["rating_delta_1"] = rating_delta_1
        player_feats["rating_delta_2"] = rating_delta_2
        player_feats["recent_rating_mean"] = recent_rating_mean
        player_feats["recent_rating_std"] = recent_rating_std

        player_feats["role_enc"]     = float(self.perf_model.role_encoding.get(latest_role, 0))
        player_feats["role_off_enc"] = float(self.perf_model.role_off_encoding.get(latest_role_off, 0))
        player_feats["role_def_enc"] = float(self.perf_model.role_def_encoding.get(latest_role_def, 0))

        # Competition-aware production models treat nominal labels as one-hot.
        role_values = {
            "ruolo_combinato": latest_role,
            "ruolo_offensivo": latest_role_off,
            "ruolo_difensivo": latest_role_def,
        }
        for column, mapping in (
            getattr(self.perf_model, "role_feature_encodings", {}) or {}
        ).items():
            current = role_values.get(column, "")
            for value, feature_name in mapping.items():
                player_feats[feature_name] = 1.0 if current == value else 0.0

        for value, feature_name in (
            getattr(self.perf_model, "competition_feature_encodings", {}) or {}
        ).items():
            player_feats[feature_name] = (
                1.0 if str(value).strip().upper() == competition else 0.0
            )

        po_feats = compute_po_features(p_stats)
        player_feats.update(po_feats)

        # 2. Base rating from XGBoost
        # Keep the raw model output separately for diagnostics. In delta mode
        # this is the predicted one-season change, not an absolute rating.
        raw_xgb_prediction = float(
            self.perf_model.predict_from_features(player_feats)
        )
        base_rating = self.perf_model.predict_target_rating(player_feats)
        base_before_age = float(base_rating)

        # 3. Age-curve ratio (only for trajectory projections)
        # Cap the ratio at ×1.5 to prevent unrealistic peak projections for
        # post-peak players being projected back to their prime years.
        current_af = age_performance_factor(current_age, position)
        target_af  = age_performance_factor(age, position)
        af = target_af

        if target_age is not None and target_age != current_age and current_af > 0.01:
            age_ratio   = float(np.clip(target_af / current_af, _AGE_RATIO_MIN, _AGE_RATIO_MAX))
            base_rating = float(np.clip(base_rating * age_ratio, 3.5, 10.0))
        base_after_age = float(base_rating)

        # 4. Compatibility: KNN score [0,1] → multiplier [0.90,1.10].
        # A centred score of 0.50 is neutral (1.00).
        cf          = self.compat_model.score(player_id, team_id, data)
        compat_mult = 0.90 + cf * 0.20
        after_compatibility = float(np.clip(base_after_age * compat_mult, 3.5, 10.0))

        # 5. League quality factor — derived from competitiveness_score in real data.
        # No hardcoded tier map: the factor is proportional to the league's
        # competitiveness_score relative to the best league in the loaded dataset.
        team_row   = data["team_dict"].get(_normalize_id(team_id), {})
        league_id  = str(team_row.get("league_id", ""))
        # Use pre-calibrated per-league factor when available
        lf         = self._league_factors.get(league_id, None)
        if lf is None:
            # Fallback: look up from league_dict directly (loaded at prediction time)
            from basketball_ai.features.team_features import _league_tier_factor as _ltf
            lf = _ltf(team_row, data)
        after_league = float(np.clip(after_compatibility * lf, 3.5, 10.0))

        # 6. Contextual adjustment (position fit, style, role, adaptation, spacing)
        # Using a wider mapping range so player-specific position/style fit creates
        # meaningful differentiation between teams (different players rank teams differently).
        ctx = compute_context_features(player_id, team_id, data)
        # The exact same-league production context should not receive a fixed
        # +2% adaptation uplift; adaptation matters when crossing contexts.
        league_adaptation = ctx["league_adaptation_factor"]
        player_league = player_row.get("current_league_id")
        team_league = team_row.get("league_id")
        try:
            same_league = (
                player_league is not None
                and team_league is not None
                and str(player_league) == str(team_league)
            )
        except Exception:
            same_league = False
        if same_league:
            league_adaptation = 1.0
        ctx_score = (
            ctx["position_team_fit"]        * 0.25
            + ctx["style_compatibility"]    * 0.30
            + ctx["role_opportunity"]       * 0.20
            + league_adaptation             * 0.15
            + ctx["spacing_fit"]            * 0.10
        )
        # Map ctx_score (range ~0.55–0.98) to multiplier; wider range than before
        # so that a PG in pace-and-space vs a C in the same team get noticeably
        # different ratings, preventing all players from ranking teams identically.
        ctx_mult = float(np.clip(
            _CTX_MULT_OFFSET + (ctx_score - 0.50) * _CTX_MULT_SLOPE,
            _CTX_MULT_LO,
            _CTX_MULT_HI,
        ))
        after_context = float(np.clip(after_league * ctx_mult, 3.5, 10.0))

        # 7. Playing-time adjustment: bench players (low minutes) are penalised.
        # A player averaging 10 min/game contributes much less proven impact than
        # a 30-min starter even when per-36 stats look similar.
        # Factor: ≥30 min → 1.00; 20 min → 0.93; 10 min → 0.73
        mpg_factor = float(np.clip(
            _MPG_FACTOR_BASE + (min(latest_mpg, self._mpg_baseline) / self._mpg_baseline) * _MPG_FACTOR_RANGE,
            _MPG_FACTOR_BASE,
            1.00,
        ))

        # Final rating (apply mpg_factor before CI so interval is always consistent)
        adjusted = float(np.clip(after_context * mpg_factor, 3.5, 10.0))

        # Confidence interval
        # Prefer split-conformal quantiles from calibration (empirically grounded).
        # Fall back to heuristic sigma when not calibrated (e.g. model not yet trained).
        consistency      = float(player_feats.get("consistency_score", 0.5))
        games_reliability = float(np.clip(latest_games_played / 25.0, 0.4, 1.0))

        if self._conformal_q_lo is not None and self._conformal_q_hi is not None:
            # Scale the conformal quantiles by consistency and games_reliability so that
            # the interval still widens for low-sample / inconsistent players, but the
            # base width is anchored to the empirical calibration residuals.
            reliability_factor = games_reliability * (0.5 + 0.5 * consistency)
            ci_half_lo = max(0.10, self._conformal_q_lo / max(reliability_factor, 0.1))
            ci_half_hi = max(0.10, self._conformal_q_hi / max(reliability_factor, 0.1))
            ci_lo = float(np.clip(adjusted - ci_half_lo, 1.0, 10.0))
            ci_hi = float(np.clip(adjusted + ci_half_hi, 1.0, 10.0))
        else:
            # Heuristic fallback (pre-calibration)
            sigma = max(0.15, 0.85 * (1.0 - consistency) / games_reliability)
            ci_lo = float(np.clip(adjusted - 1.96 * sigma, 1.0, 10.0))
            ci_hi = float(np.clip(adjusted + 1.96 * sigma, 1.0, 10.0))

        shap_vals = self.perf_model.get_shap_values(player_feats)

        explanation = (
            f"Rating 0–10 basato su: "
            f"XGBoost ({len(self.perf_model.feature_names)} features, base {base_rating:.2f}) "
            f"× compatibilità stile ({compat_mult:.2f}) "
            f"× qualità lega (league {league_id}, fattore {lf:.3f}) "
            f"× contesto (posizione+stile+adattamento, fattore {ctx_mult:.3f}) "
            f"× minuti ({latest_mpg:.0f} min/g, fattore {mpg_factor:.2f}) "
            f"| età: {age} (curva {af:.3f}) "
            f"| ruolo: {latest_role or '—'} / off: {latest_role_off or '—'} / def: {latest_role_def or '—'} "
            f"| competizione: {competition}"
        )

        result = PredictionResult(
            player_id=player_id,
            team_id=team_id,
            season=season,
            predicted_rating=round(adjusted, 3),
            confidence_low=round(ci_lo, 3),
            confidence_high=round(ci_hi, 3),
            base_rating=round(base_rating, 3),
            age_factor=round(af, 3),
            compatibility_factor=round(cf, 3),
            league_factor=round(lf, 3),
            context_adjustment=round(ctx_mult, 3),
            shap_values=shap_vals,
            explanation=explanation,
            competition=competition,
        )
        # Private diagnostics consumed only by the walk-forward diagnostic
        # runner. They are deliberately not dataclass fields, so the public
        # prediction/API contract is unchanged.
        result._diagnostic_stages = {
            "raw_xgb_prediction": raw_xgb_prediction,
            "base_before_age": base_before_age,
            "base_after_age": base_after_age,
            "after_compatibility": after_compatibility,
            "after_league": after_league,
            "after_context": after_context,
            "final_prediction": adjusted,
        }
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = "models_saved", metrics: Optional[Dict[str, float]] = None) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.perf_model.save(f"{directory}/performance_model.joblib")
        self.compat_model.save(f"{directory}/compatibility_model.joblib")

        # Persist conformal calibration quantiles and instance-level calibrations
        calibration = {
            "conformal_q_lo": self._conformal_q_lo,
            "conformal_q_hi": self._conformal_q_hi,
            "league_factors":  self._league_factors,
            "mpg_baseline":    self._mpg_baseline,
        }
        import joblib as _joblib
        _joblib.dump(calibration, f"{directory}/conformal.joblib")

        # Persist drift reference if available
        if self._drift_reference:
            _joblib.dump(self._drift_reference, f"{directory}/drift_reference.joblib")

        # Write human-readable metadata alongside the model files.
        meta: Dict[str, Any] = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features":   self.perf_model.feature_names,
            "conformal_q_lo": self._conformal_q_lo,
            "conformal_q_hi": self._conformal_q_hi,
        }
        if metrics:
            meta.update(metrics)
        Path(directory, "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("[Ensemble] Models saved to %s/", directory)

    def load(self, directory: str = "models_saved") -> None:
        self.perf_model.load(f"{directory}/performance_model.joblib")
        self.compat_model.load(f"{directory}/compatibility_model.joblib")
        # Load conformal quantiles if available
        import joblib as _joblib
        conformal_path = Path(directory) / "conformal.joblib"
        if conformal_path.exists():
            calibration = _joblib.load(str(conformal_path))
            self._conformal_q_lo = calibration.get("conformal_q_lo")
            self._conformal_q_hi = calibration.get("conformal_q_hi")
            self._league_factors = calibration.get("league_factors", {})
            self._mpg_baseline   = calibration.get("mpg_baseline", _MPG_BASELINE_DEFAULT)
        # Load drift reference if available
        drift_path = Path(directory) / "drift_reference.joblib"
        if drift_path.exists():
            self._drift_reference = _joblib.load(str(drift_path))
        logger.info("[Ensemble] Models loaded from %s/", directory)

    def check_data_drift(self, current_data: Dict[str, Any]) -> Dict[str, float]:
        """Run PSI-based drift detection against the training reference.

        Logs warnings when individual features show significant drift.

        Args:
            current_data: Full data dict (same format as loader output).

        Returns:
            Dict mapping feature → PSI value.  Empty if no reference stored.
        """
        try:
            from basketball_ai.monitoring.drift import compute_psi_report
            return compute_psi_report(current_data, self._drift_reference)
        except Exception as exc:
            logger.warning("[Ensemble] Drift check failed: %s", exc)
            return {}

    @property
    def is_trained(self) -> bool:
        return self.perf_model.is_trained and self.compat_model.is_trained

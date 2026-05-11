"""Ensemble model: combines XGBoost, age curve, and compatibility scores.

Final prediction:
    adjusted_rating = base_rating * age_factor * compatibility_factor * league_factor

Confidence intervals are estimated from player consistency and age uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.models.age_curve import age_performance_factor, PEAK_AGES
from src.models.compatibility_model import CompatibilityModel
from src.models.performance_model import PerformanceModel
from src.features.player_features import (
    compute_player_features,
    compute_form_score,
    compute_consistency_score,
)
from src.features.team_features import compute_team_features
from src.features.context_features import compute_context_features


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


class EnsembleModel:
    """Combines all sub-models into a single prediction pipeline."""

    def __init__(
        self,
        performance_model: Optional[PerformanceModel] = None,
        compatibility_model: Optional[CompatibilityModel] = None,
    ) -> None:
        self.perf_model = performance_model or PerformanceModel()
        self.compat_model = compatibility_model or CompatibilityModel()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: Dict[str, Any]) -> None:
        """Train all sub-models."""
        print("[Ensemble] Training performance model …")
        self.perf_model.train(data)
        print("[Ensemble] Training compatibility model …")
        self.compat_model.train(data)
        print("[Ensemble] Training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        player_id: int,
        team_id: int,
        data: Dict[str, Any],
        season: int = 2024,
        target_age: Optional[int] = None,
    ) -> PredictionResult:
        """Generate a full prediction for a player in a given team/season.

        The XGBoost model already captures age effects through the `age` feature,
        so the ensemble only applies *small* multiplicative adjustments for:
          - team style compatibility (mapped to ±10 %)
          - league quality tier  (−0 to −14 %)
          - tactical context     (±5 %)

        For trajectory projections (`target_age` differs from `current_age`),
        the XGBoost base is additionally scaled by the age-curve ratio.

        Args:
            player_id: Player to predict.
            team_id: Target team.
            data: Loaded data dict from loader.load_all_data().
            season: Season reference year.
            target_age: Override age (used in trajectory projections).

        Returns:
            PredictionResult instance.
        """
        player = data["player_dict"].get(int(player_id), {})
        position = str(player.get("position", "CM"))
        current_age = int(player.get("age", 27))
        age = target_age if target_age is not None else current_age

        # 1. Base rating from XGBoost (captures age, position, per-90 stats)
        player_feats = compute_player_features(player_id, data, season=season, target_age=age)
        base_rating = self.perf_model.predict_from_features(player_feats)

        # 2. Age-curve ratio (only adjusts for *trajectory* when age differs)
        current_af = age_performance_factor(current_age, position)
        target_af = age_performance_factor(age, position)
        af = target_af  # stored for output / CI calculation

        if target_age is not None and target_age != current_age and current_af > 0.01:
            age_ratio = target_af / current_af
            base_rating = float(np.clip(base_rating * age_ratio, 3.5, 10.0))

        # 3. Compatibility: KNN score in [0,1]; remap to [0.90, 1.10]
        cf = self.compat_model.score(team_id, data)
        compat_mult = 0.90 + cf * 0.20  # neutral ≈ 1.05

        # 4. League quality factor  (tier-1 → 1.00, tier-5 → 0.86)
        team = data["team_dict"].get(int(team_id), {})
        tier = int(team.get("league_tier", 1))
        tier_map = {1: 1.00, 2: 0.98, 3: 0.95, 4: 0.91, 5: 0.86}
        lf = tier_map.get(tier, 0.95)

        # 5. Tactical context: position fit, style, role, adaptation → [0.95, 1.05]
        ctx = compute_context_features(player_id, team_id, data)
        ctx_score = (
            ctx["position_team_fit"] * 0.30
            + ctx["style_compatibility"] * 0.30
            + ctx["role_opportunity"] * 0.20
            + ctx["league_adaptation_factor"] * 0.20
        )
        # ctx_score typically 0.7–0.9; map 0.75 → 1.00 as neutral
        ctx_mult = 0.95 + (ctx_score - 0.75) * 0.20

        # Final prediction
        adjusted = base_rating * compat_mult * lf * ctx_mult
        adjusted = float(np.clip(adjusted, 3.5, 10.0))

        # Confidence interval (wider = less consistent)
        consistency = float(player_feats.get("consistency_score", 0.5))
        sigma = max(0.20, 0.80 * (1.0 - consistency))
        ci_lo = float(np.clip(adjusted - 1.96 * sigma, 1.0, 10.0))
        ci_hi = float(np.clip(adjusted + 1.96 * sigma, 1.0, 10.0))

        # SHAP values (best-effort)
        shap_vals = self.perf_model.get_shap_values(player_feats)

        explanation = (
            f"XGB={base_rating:.2f} × Compat={compat_mult:.3f} "
            f"× League(tier{tier})={lf:.3f} × Context={ctx_mult:.3f}  "
            f"[age_curve_af={af:.3f}]"
        )

        return PredictionResult(
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
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = "models_saved") -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.perf_model.save(f"{directory}/performance_model.joblib")
        self.compat_model.save(f"{directory}/compatibility_model.joblib")
        print(f"[Ensemble] Models saved to {directory}/")

    def load(self, directory: str = "models_saved") -> None:
        self.perf_model.load(f"{directory}/performance_model.joblib")
        self.compat_model.load(f"{directory}/compatibility_model.joblib")
        print(f"[Ensemble] Models loaded from {directory}/")

    @property
    def is_trained(self) -> bool:
        return self.perf_model.is_trained and self.compat_model.is_trained

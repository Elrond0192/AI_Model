"""Basketball ensemble model.

Combines:
  1. XGBoost performance model (base rating)
  2. Age curve (trajectory adjustments)
  3. Compatibility model (style fit)
  4. Contextual adjustments (league, role, spacing, adaptation)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from basketball_ai.models.age_curve import age_performance_factor, PEAK_AGES


from basketball_ai.utils.helpers import normalize_id as _normalize_id
from basketball_ai.models.compatibility_model import CompatibilityModel
from basketball_ai.models.performance_model import (
    PerformanceModel, COMPETITION_ENCODING, compute_po_features,
)
from basketball_ai.features.player_features import compute_player_features
from basketball_ai.features.team_features import compute_team_features
from basketball_ai.features.context_features import compute_context_features


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
    ) -> None:
        self.perf_model   = performance_model   or PerformanceModel()
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
        competition: str = "RS",
    ) -> PredictionResult:
        """Generate a full basketball prediction for a player in a given team.

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
        else:
            latest_role = latest_role_off = latest_role_def = ""
            latest_games_played = 0.0
        player_feats["role_enc"]     = float(self.perf_model.role_encoding.get(latest_role, 0))
        player_feats["role_off_enc"] = float(self.perf_model.role_off_encoding.get(latest_role_off, 0))
        player_feats["role_def_enc"] = float(self.perf_model.role_def_encoding.get(latest_role_def, 0))

        po_feats = compute_po_features(p_stats)
        player_feats.update(po_feats)

        # 2. Base rating from XGBoost
        base_rating = self.perf_model.predict_from_features(player_feats)

        # 3. Age-curve ratio (only for trajectory projections)
        current_af = age_performance_factor(current_age, position)
        target_af  = age_performance_factor(age, position)
        af = target_af

        if target_age is not None and target_age != current_age and current_af > 0.01:
            age_ratio   = target_af / current_af
            base_rating = float(np.clip(base_rating * age_ratio, 3.5, 10.0))

        # 4. Compatibility: KNN score [0,1] → mapped to multiplier [0.90, 1.10]
        cf          = self.compat_model.score(team_id, data)
        compat_mult = 0.90 + cf * 0.20   # neutral ≈ 1.05

        # 5. League quality factor
        # Tier 1 = NBA/top EuroLeague (1.00); higher tiers are progressively discounted.
        team_row = data["team_dict"].get(_normalize_id(team_id), {})
        tier     = int(team_row.get("league_tier", 1))
        tier_map = {1: 1.000, 2: 0.980, 3: 0.955, 4: 0.920, 5: 0.875, 6: 0.820, 7: 0.760}
        lf       = tier_map.get(tier, 0.900)

        # 6. Contextual adjustment (position fit, style, role, adaptation, spacing)
        ctx = compute_context_features(player_id, team_id, data)
        ctx_score = (
            ctx["position_team_fit"]        * 0.25
            + ctx["style_compatibility"]    * 0.30
            + ctx["role_opportunity"]       * 0.20
            + ctx["league_adaptation_factor"] * 0.15
            + ctx["spacing_fit"]            * 0.10
        )
        # Map ctx_score (typically 0.7–0.9) to multiplier around 1.00
        ctx_mult = 0.95 + (ctx_score - 0.75) * 0.20

        # Final rating
        adjusted = float(np.clip(base_rating * compat_mult * lf * ctx_mult, 3.5, 10.0))

        # Confidence interval – wider for inconsistent players, wider for small samples
        consistency = float(player_feats.get("consistency_score", 0.5))
        # Scale reliability 0.5 → 1.0 based on games_played (25+ games = full reliability)
        games_reliability = float(np.clip(latest_games_played / 25.0, 0.4, 1.0))
        sigma = max(0.15, 0.85 * (1.0 - consistency) / games_reliability)
        ci_lo = float(np.clip(adjusted - 1.96 * sigma, 1.0, 10.0))
        ci_hi = float(np.clip(adjusted + 1.96 * sigma, 1.0, 10.0))

        shap_vals = self.perf_model.get_shap_values(player_feats)

        explanation = (
            f"Rating 0–10 basato su: "
            f"XGBoost ({len(self.perf_model.feature_names)} features, base {base_rating:.2f}) "
            f"× compatibilità stile ({compat_mult:.2f}) "
            f"× qualità lega (tier {tier}, fattore {lf:.2f}) "
            f"× contesto (posizione+stile+adattamento, fattore {ctx_mult:.3f}) "
            f"| età: {age} (curva {af:.3f}) "
            f"| ruolo: {latest_role or '—'} / off: {latest_role_off or '—'} / def: {latest_role_def or '—'} "
            f"| competizione: {competition}"
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
            competition=competition,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = "models_saved", metrics: Optional[Dict[str, float]] = None) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.perf_model.save(f"{directory}/performance_model.joblib")
        self.compat_model.save(f"{directory}/compatibility_model.joblib")

        # Write human-readable metadata alongside the model files.
        meta: Dict[str, Any] = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features":   self.perf_model.feature_names,
        }
        if metrics:
            meta.update(metrics)
        Path(directory, "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[Ensemble] Models saved to {directory}/")

    def load(self, directory: str = "models_saved") -> None:
        self.perf_model.load(f"{directory}/performance_model.joblib")
        self.compat_model.load(f"{directory}/compatibility_model.joblib")
        print(f"[Ensemble] Models loaded from {directory}/")

    @property
    def is_trained(self) -> bool:
        return self.perf_model.is_trained and self.compat_model.is_trained

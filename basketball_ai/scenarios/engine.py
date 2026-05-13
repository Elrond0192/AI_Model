"""Basketball What-If scenario engine."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from basketball_ai.models.ensemble import EnsembleModel, PredictionResult
from basketball_ai.utils.helpers import team_display_name as _team_display_name


def _normalize_id(value: Any) -> Any:
    """Normalize an ID to int when possible, keep as-is for non-numeric strings like 'GRC1'."""
    if value is None:
        return None
    if isinstance(value, float):
        return int(value)
    if isinstance(value, int):
        return value
    v = str(value).strip()
    try:
        return int(v, 10)
    except (ValueError, TypeError):
        try:
            return int(v, 16)
        except (ValueError, TypeError):
            return v


from basketball_ai.models.age_curve import age_performance_factor, PEAK_AGES, peak_age_window
from basketball_ai.features.context_features import compute_context_features
from basketball_ai.features.player_features import compute_player_features
from basketball_ai.features.team_features import compute_team_features, get_style_position_compat


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    age: int
    season: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    age_factor: float


@dataclass
class ComparisonResult:
    player_id: int
    scenarios: List[Dict[str, Any]]
    best_scenario: Dict[str, Any]


@dataclass
class TeamFitResult:
    team_id: int
    team_name: str
    league_name: str
    predicted_rating: float
    compatibility_score: float
    position_fit: float
    rank: int


@dataclass
class PlayerFitResult:
    player_id: int
    player_name: str
    position: str
    predicted_rating: float
    current_team: str
    rank: int


@dataclass
class TransferImpactResult:
    player_id: int
    from_team_id: int
    to_team_id: int
    rating_before: float
    rating_after: float
    rating_delta: float
    adaptation_factor: float
    recommendation: str


@dataclass
class PeakPrediction:
    player_id: int
    player_name: str
    current_age: int
    peak_age: int
    current_rating: float
    peak_rating: float
    seasons_to_peak: int
    peak_window: Tuple[int, int]


# ---------------------------------------------------------------------------
# Lineup analysis types
# ---------------------------------------------------------------------------

@dataclass
class LineupMemberProfile:
    """Individual profile of one lineup member within a specific team context."""
    player_id: int
    player_name: str
    position: str
    predicted_rating: float
    role: str            # e.g. "Primary Scorer", "Playmaker", "Defender", "3pt Specialist"
    style_compat: float  # style × position compatibility with the target team


@dataclass
class LineupAnalysisResult:
    """Result of a what-if lineup scenario."""
    player_id: int
    player_name: str
    team_id: int
    predicted_rating: float
    confidence_low: float
    confidence_high: float
    avg_lineup_rating: float
    positions_covered: List[str]
    missing_positions: List[str]
    position_overlaps: Dict[str, int]   # position → count (only when > 1)
    lineup_profiles: List[LineupMemberProfile]
    explanation: str


# ---------------------------------------------------------------------------
# Advanced role taxonomy
# ---------------------------------------------------------------------------

ADVANCED_ROLES: Dict[str, str] = {
    "🎯 Playmaker":         "Palleggiatore primario – controlla il possesso (AST% alto)",
    "⚡ Scorer primario":   "Prima opzione offensiva (USG% > 25, tanti punti)",
    "🏹 Specialista 3pt":  "Spacer / tiratore da tre – allunga il campo",
    "🛡️ Difensore":        "Difensore d'élite (STL+BLK elevati, aggressivo)",
    "💪 Pivot / Big":       "Presenza nella pittura – rimbalzista e finalizzatore",
    "🔄 Ala tuttofare":     "Ala versatile – contribuisce in attacco e difesa",
    "📊 Role Player":       "Giocatore di ruolo – efficiente, USG basso",
    "🎭 Stretch Big":       "Lungo che tira da tre – allarga il campo",
}


@dataclass
class LineupSynergyResult:
    """Comprehensive synergy analysis for a multi-player lineup."""
    overall_synergy: float            # 0–10 composite score
    role_diversity_score: float       # 0–1
    positional_balance: float         # 0–1
    offensive_balance: float          # 0–1
    defensive_score: float            # 0–10
    style_compat: float               # 0–1
    role_distribution: Dict[str, List[str]]
    missing_roles: List[str]
    synergy_bonus: float              # rating boost (0–0.4)
    pairwise_compat: Dict[str, float]
    summary: str


@dataclass
class RoleCandidateResult:
    """Top candidates for a specific role."""
    role: str
    description: str
    candidates: List[Dict[str, Any]]


@dataclass
class LineupByRolesResult:
    """Result of role-based lineup search."""
    target_player_id: int
    target_player_name: str
    team_id: int
    season: int
    role_candidates: List[RoleCandidateResult]
    optimal_lineup: List[Dict[str, Any]]
    estimated_avg_rating: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WhatIfEngine:
    """High-level basketball scenario engine wrapping EnsembleModel."""

    def __init__(self, ensemble: EnsembleModel, data: Dict[str, Any]) -> None:
        self.ensemble = ensemble
        self.data     = data

    # ------------------------------------------------------------------
    def predict_in_team(
        self, player_id: int, target_team_id: int, season: int = 2024,
        competition: str = "RS",
    ) -> PredictionResult:
        """Predict how a player would perform at a given team.

        Args:
            competition: Competition context – "RS" (Regular Season), "PO"
                         (Playoffs), "CUP", or "SUPERCUP".
        """
        return self.ensemble.predict(
            player_id, target_team_id, self.data, season, competition=competition
        )

    # ------------------------------------------------------------------
    def predict_age_trajectory(
        self,
        player_id: int,
        age_range: Optional[Tuple[int, int]] = None,
        team_id: Optional[int] = None,
        season_base: int = 2024,
    ) -> List[TrajectoryPoint]:
        """Return rating predictions across an age range."""
        player      = self.data["player_dict"].get(_normalize_id(player_id), {})
        current_age = int(player.get("age", 25))
        position    = str(player.get("position", "PG"))

        if team_id is None:
            ct = player.get("current_team_id")
            team_id = _normalize_id(ct) if ct else 1

        lo, hi = age_range if age_range else (
            max(18, current_age - 4), min(40, current_age + 8)
        )

        result: List[TrajectoryPoint] = []
        for age in range(lo, hi + 1):
            season = season_base + (age - current_age)
            pred   = self.ensemble.predict(
                player_id, team_id, self.data, season=season, target_age=age
            )
            result.append(TrajectoryPoint(
                age=age, season=season,
                predicted_rating=pred.predicted_rating,
                confidence_low=pred.confidence_low,
                confidence_high=pred.confidence_high,
                age_factor=pred.age_factor,
            ))
        return result

    # ------------------------------------------------------------------
    def compare_scenarios(
        self, player_id: int, team_ids: List[int], season: int = 2024,
        competition: str = "RS",
    ) -> ComparisonResult:
        """Compare player performance across multiple team scenarios."""
        team_dict   = self.data["team_dict"]
        league_dict = self.data["league_dict"]

        scenarios = []
        for tid in team_ids:
            pred   = self.ensemble.predict(player_id, tid, self.data, season,
                                           competition=competition)
            team   = team_dict.get(_normalize_id(tid), {})
            league = league_dict.get(_normalize_id(team.get("league_id")), {})
            scenarios.append({
                "team_id":        tid,
                "team_name":      _team_display_name(team, f"Team {tid}"),
                "league_name":    str(league.get("name", "Unknown")),
                "competition":    competition,
                "rating":         pred.predicted_rating,
                "confidence_low": pred.confidence_low,
                "confidence_high": pred.confidence_high,
                "age_factor":     pred.age_factor,
                "compatibility_factor": pred.compatibility_factor,
                "explanation":    pred.explanation,
            })

        scenarios.sort(key=lambda s: s["rating"], reverse=True)
        return ComparisonResult(
            player_id=player_id,
            scenarios=scenarios,
            best_scenario=scenarios[0] if scenarios else {},
        )

    # ------------------------------------------------------------------
    def best_team_fit(
        self, player_id: int, league_id: Optional[int] = None,
        top_n: int = 10, season: int = 2024,
    ) -> List[TeamFitResult]:
        """Find the top-N teams where the player would perform best."""
        teams_df    = self.data["teams"]
        team_dict   = self.data["team_dict"]
        league_dict = self.data["league_dict"]

        candidate = teams_df if league_id is None else teams_df[teams_df["league_id"] == league_id]

        results: List[TeamFitResult] = []
        for _, row in candidate.iterrows():
            tid = _normalize_id(row["id"])
            try:
                pred   = self.ensemble.predict(player_id, tid, self.data, season)
                ctx    = compute_context_features(player_id, tid, self.data)
                team   = team_dict.get(tid, {})
                league = league_dict.get(_normalize_id(row["league_id"]), {})
                results.append(TeamFitResult(
                    team_id=tid,
                    team_name=_team_display_name(team, f"Team {tid}"),
                    league_name=str(league.get("name", "Unknown")),
                    predicted_rating=pred.predicted_rating,
                    compatibility_score=pred.compatibility_factor,
                    position_fit=ctx["position_team_fit"],
                    rank=0,
                ))
            except Exception:
                continue

        results.sort(key=lambda r: r.predicted_rating, reverse=True)
        for i, r in enumerate(results[:top_n]):
            r.rank = i + 1
        return results[:top_n]

    # ------------------------------------------------------------------
    def best_player_for_team(
        self, team_id: int, position: Optional[str] = None,
        top_n: int = 10, season: int = 2024,
    ) -> List[PlayerFitResult]:
        """Find the top-N players who would fit best at a team."""
        players_df = self.data["players"]
        team_dict  = self.data["team_dict"]

        if position:
            # Match pure or hybrid positions containing the string
            candidates = players_df[
                players_df["position"].str.contains(position, regex=False, na=False)
            ]
        else:
            candidates = players_df

        sample = candidates.sample(min(500, len(candidates)), random_state=42)

        results: List[PlayerFitResult] = []
        for _, row in sample.iterrows():
            pid = _normalize_id(row["id"])
            try:
                pred     = self.ensemble.predict(pid, team_id, self.data, season)
                cur_team = team_dict.get(_normalize_id(row.get("current_team_id")), {})
                results.append(PlayerFitResult(
                    player_id=pid,
                    player_name=str(row["name"]),
                    position=str(row["position"]),
                    predicted_rating=pred.predicted_rating,
                    current_team=_team_display_name(cur_team),
                    rank=0,
                ))
            except Exception:
                continue

        results.sort(key=lambda r: r.predicted_rating, reverse=True)
        for i, r in enumerate(results[:top_n]):
            r.rank = i + 1
        return results[:top_n]

    # ------------------------------------------------------------------
    def simulate_transfer(
        self, player_id: int, from_team_id: int, to_team_id: int,
        season: int = 2024, competition: str = "RS",
    ) -> TransferImpactResult:
        """Simulate the performance impact of a transfer between two teams."""
        pred_before = self.ensemble.predict(player_id, from_team_id, self.data, season,
                                            competition=competition)
        pred_after  = self.ensemble.predict(player_id, to_team_id,   self.data, season,
                                            competition=competition)
        ctx   = compute_context_features(player_id, to_team_id, self.data)
        delta = pred_after.predicted_rating - pred_before.predicted_rating

        if delta >= 0.3:
            rec = "Strongly recommended – significant improvement expected."
        elif delta >= 0.0:
            rec = "Neutral – marginal improvement expected."
        elif delta >= -0.3:
            rec = "Slight drop expected – evaluate carefully."
        else:
            rec = "Not recommended – notable performance drop predicted."

        return TransferImpactResult(
            player_id=player_id,
            from_team_id=from_team_id,
            to_team_id=to_team_id,
            rating_before=pred_before.predicted_rating,
            rating_after=pred_after.predicted_rating,
            rating_delta=round(delta, 3),
            adaptation_factor=ctx["league_adaptation_factor"],
            recommendation=rec,
        )

    # ------------------------------------------------------------------
    def predict_stats_at_team(
        self,
        player_id: int,
        team_id: int,
        season: int = 2024,
        competition: str = "RS",
    ) -> Optional[Dict[str, Any]]:
        """Predict which per-game statistics a player would average at *team_id*.

        The method:
          1. Computes the predicted performance rating at the target team.
          2. Computes the player's historical rating average.
          3. Scales the most recent season's per-game stats by the ratio
             ``predicted / historical``, capped at ±40 % to avoid unrealistic
             projections caused by outlier model predictions.
          4. Returns ``None`` when insufficient historical data exists.
        """
        # Scaling constants
        _RATIO_MIN         = 0.60   # cap downside to −40 %
        _RATIO_MAX         = 1.40   # cap upside  to +40 %
        _PCT_SCALE_MUTE    = 0.5    # mute ratio for shooting %s (less elastic than volume)
        _DEFAULT_USG_PCT   = 18.0   # fallback when usg_pct column is absent

        import pandas as _pd
        p_stats = self.data["player_stats"]
        pid_norm = _normalize_id(player_id)
        mask = p_stats["player_id"] == pid_norm
        pdf  = p_stats[mask].sort_values("season")
        if pdf.empty:
            return None

        # Historical average rating
        hist_rating = float(pdf["rating"].mean()) if "rating" in pdf.columns else 6.5
        if hist_rating <= 0:
            hist_rating = 6.5

        # Predicted rating at target team
        pred = self.predict_in_team(player_id, team_id, season, competition)
        ratio = float(np.clip(pred.predicted_rating / hist_rating, _RATIO_MIN, _RATIO_MAX))

        latest = pdf.iloc[-1]

        def _scale(col: str, pct_field: bool = False) -> Optional[float]:
            if col not in latest or _pd.isna(latest[col]):
                return None
            val = float(latest[col])
            if pct_field:
                # Shooting percentages are less elastic than volume stats
                return round(val * (1.0 + (ratio - 1.0) * _PCT_SCALE_MUTE), 3)
            return round(val * ratio, 1)

        result: Dict[str, Any] = {
            "predicted_rating":  pred.predicted_rating,
            "historical_rating": round(hist_rating, 2),
            "scaling_ratio":     round(ratio, 3),
            "season_reference":  str(latest.get("season", season)),
        }
        # Volume stats — scaled by ratio
        for col in ("points", "rebounds", "assists", "steals", "blocks", "minutes_per_game"):
            val = _scale(col)
            if val is not None:
                result[col] = val
        # Shooting percentages — muted scaling
        for col in ("ts_pct", "fg_pct", "three_point_pct"):
            val = _scale(col, pct_field=True)
            if val is not None:
                result[col] = val
        # Usage rate — kept at historical level (role/responsibility unchanged)
        raw_usg = latest.get("usg_pct")
        if raw_usg is not None and not _pd.isna(raw_usg):
            result["usg_pct"] = round(float(raw_usg), 1)
        else:
            result["usg_pct"] = _DEFAULT_USG_PCT

        return result

    # ------------------------------------------------------------------
    def predict_peak(self, player_id: int, team_id: Optional[int] = None) -> PeakPrediction:
        """Predict a player's career peak rating."""
        player      = self.data["player_dict"].get(_normalize_id(player_id), {})
        current_age = int(player.get("age", 25))
        position    = str(player.get("position", "PG"))
        name        = str(player.get("name", f"Player {player_id}"))

        if team_id is None:
            ct = player.get("current_team_id")
            team_id = _normalize_id(ct) if ct else 1

        traj = self.predict_age_trajectory(player_id, age_range=(18, 40), team_id=team_id)
        if not traj:
            primary = position.split("/")[0]
            default_peak = PEAK_AGES.get(position, PEAK_AGES.get(primary, 26))
            return PeakPrediction(
                player_id=player_id, player_name=name,
                current_age=current_age, peak_age=default_peak,
                current_rating=6.5, peak_rating=6.5,
                seasons_to_peak=0, peak_window=(default_peak - 1, default_peak + 1),
            )

        best         = max(traj, key=lambda p: p.predicted_rating)
        current_pred = next((p for p in traj if p.age == current_age), traj[0])
        window       = peak_age_window(position, threshold=0.93)

        return PeakPrediction(
            player_id=player_id, player_name=name,
            current_age=current_age, peak_age=best.age,
            current_rating=current_pred.predicted_rating,
            peak_rating=best.predicted_rating,
            seasons_to_peak=max(0, best.age - current_age),
            peak_window=window,
        )

    # ------------------------------------------------------------------
    def what_if_teammates(
        self, player_id: int, team_id: int,
        hypothetical_avg_rating: float, season: int = 2024,
    ) -> PredictionResult:
        """What if the player's teammates had a different average rating?"""
        pred       = self.ensemble.predict(player_id, team_id, self.data, season)
        team_feats = compute_team_features(team_id, self.data, exclude_player_id=player_id)
        current_tm = team_feats["avg_teammate_rating"]
        tm_delta   = (hypothetical_avg_rating - current_tm) * 0.08
        adjusted   = float(np.clip(pred.predicted_rating + tm_delta, 3.5, 10.0))

        return PredictionResult(
            player_id=pred.player_id, team_id=pred.team_id, season=pred.season,
            predicted_rating=round(adjusted, 3),
            confidence_low=round(max(1.0, adjusted - 0.5), 3),
            confidence_high=round(min(10.0, adjusted + 0.5), 3),
            base_rating=pred.base_rating, age_factor=pred.age_factor,
            compatibility_factor=pred.compatibility_factor,
            league_factor=pred.league_factor, context_adjustment=pred.context_adjustment,
            shap_values=pred.shap_values,
            explanation=(
                f"{pred.explanation}  "
                f"| Teammate δ: {tm_delta:+.3f} "
                f"(hypothetical avg={hypothetical_avg_rating:.1f} vs actual={current_tm:.1f})"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers for lineup analysis
    # ------------------------------------------------------------------

    _ALL_POSITIONS: frozenset = frozenset({"PG", "SG", "SF", "PF", "C"})

    @staticmethod
    def _assign_role(player_feats: Dict[str, Any]) -> str:
        """Heuristic role label from computed player features (European basketball calibrated)."""
        pm   = float(player_feats.get("playmaking_score", 0.0))
        df   = float(player_feats.get("defensive_score", 0.0))
        usg  = float(player_feats.get("avg_usg_pct", 0.0))
        prof = str(player_feats.get("scoring_profile", "efficient_scorer"))
        pts  = float(player_feats.get("pts_per_36", 0.0))
        stl  = float(player_feats.get("stl_per_36", 0.0))
        blk  = float(player_feats.get("blk_per_36", 0.0))
        # Lower thresholds calibrated for European basketball data
        if pm > 0.28:
            return "Playmaker"
        if (stl + blk) > 2.0 or df > 3.0:
            return "Defender"
        if prof == "3pt_specialist":
            return "3pt Specialist"
        if usg > 20 or pts > 18:
            return "Primary Scorer"
        if prof == "paint_scorer":
            return "Paint Scorer / Big"
        return "Two-way / Role Player"

    def what_if_lineup(
        self,
        player_id: int,
        team_id: int,
        lineup_player_ids: List[int],
        season: int = 2024,
    ) -> LineupAnalysisResult:
        """Predict player X's performance in team Y with a specific named lineup.

        Args:
            player_id:         The target player (player X).
            team_id:           The target team (team Y).
            lineup_player_ids: The other 4 (or fewer) named lineup members.

        The method:
          1. Predicts each named lineup member's rating at team Y.
          2. Computes position coverage and role distribution for the full 5-man unit.
          3. Overrides the ``avg_teammate_rating`` feature with the computed lineup average.
          4. Re-runs the ensemble model and returns a rich ``LineupAnalysisResult``.
        """
        player_dict = self.data["player_dict"]
        team_dict   = self.data["team_dict"]
        target_name = str(player_dict.get(_normalize_id(player_id), {}).get("name", f"Player {player_id}"))
        team_row    = team_dict.get(_normalize_id(team_id), {})
        team_style  = str(team_row.get("playing_style", "motion_offense"))

        # --- 1. Profile each lineup member --------------------------------
        profiles: List[LineupMemberProfile] = []
        for pid in lineup_player_ids:
            p_row = player_dict.get(_normalize_id(pid), {})
            pos   = str(p_row.get("position", "PG"))
            name  = str(p_row.get("name", f"Player {pid}"))
            try:
                pred = self.ensemble.predict(pid, team_id, self.data, season)
                r    = pred.predicted_rating
            except Exception:
                r = 6.0
            p_feats    = compute_player_features(pid, self.data)
            role       = self._classify_role_advanced(p_feats, pos)
            style_compat = get_style_position_compat(team_style, pos)
            profiles.append(LineupMemberProfile(
                player_id=pid, player_name=name, position=pos,
                predicted_rating=round(r, 3), role=role,
                style_compat=round(style_compat, 3),
            ))

        # --- 2. Position coverage (all 5 slots incl. target player) ------
        target_pos = str(player_dict.get(_normalize_id(player_id), {}).get("position", "PG"))
        all_positions = [target_pos] + [p.position for p in profiles]
        pos_counter: Counter = Counter()
        for pos in all_positions:
            for component in pos.split("/"):
                pos_counter[component] += 1

        covered      = [p for p in self._ALL_POSITIONS if p in pos_counter]
        missing      = [p for p in sorted(self._ALL_POSITIONS) if p not in pos_counter]
        overlaps     = {p: cnt for p, cnt in pos_counter.items() if cnt > 1}

        # --- 3. Compute lineup avg rating (excluding target player) -------
        avg_lineup = (
            float(np.mean([p.predicted_rating for p in profiles]))
            if profiles else 6.0
        )

        # --- 4. Predict target player with overridden teammate avg --------
        base_pred  = self.ensemble.predict(player_id, team_id, self.data, season)
        team_feats = compute_team_features(team_id, self.data, exclude_player_id=player_id)
        current_tm = team_feats["avg_teammate_rating"]
        tm_delta   = (avg_lineup - current_tm) * 0.08
        adjusted   = float(np.clip(base_pred.predicted_rating + tm_delta, 3.5, 10.0))

        # --- 5. Build explanation -----------------------------------------
        role_parts = [f"{p.player_name} ({p.position}, {p.role})" for p in profiles]
        miss_str   = ", ".join(missing) if missing else "none"
        over_str   = (
            ", ".join(f"{p}×{c}" for p, c in overlaps.items())
            if overlaps else "none"
        )
        explanation = (
            f"Lineup avg rating: {avg_lineup:.2f}  "
            f"(actual roster avg: {current_tm:.2f}, δ={tm_delta:+.3f})  |  "
            f"Missing positions: {miss_str}  |  "
            f"Position overlaps: {over_str}  |  "
            f"Members: {'; '.join(role_parts) if role_parts else 'none specified'}"
        )

        return LineupAnalysisResult(
            player_id=player_id,
            player_name=target_name,
            team_id=team_id,
            predicted_rating=round(adjusted, 3),
            confidence_low=round(max(1.0, adjusted - 0.5), 3),
            confidence_high=round(min(10.0, adjusted + 0.5), 3),
            avg_lineup_rating=round(avg_lineup, 3),
            positions_covered=sorted(covered),
            missing_positions=sorted(missing),
            position_overlaps=overlaps,
            lineup_profiles=profiles,
            explanation=explanation,
        )

    @staticmethod
    def _classify_role_advanced(player_feats: Dict[str, Any], position: str) -> str:
        """More granular role classification using ADVANCED_ROLES taxonomy.

        Thresholds are calibrated for European basketball (EuroLeague / national
        leagues) where pace and individual stats are typically lower than NBA.
        """
        pm   = float(player_feats.get("playmaking_score", 0.0))
        df   = float(player_feats.get("defensive_score", 0.0))
        usg  = float(player_feats.get("avg_usg_pct", 0.0))
        prof = str(player_feats.get("scoring_profile", "efficient_scorer"))
        pts  = float(player_feats.get("pts_per_36", 0.0))
        ast  = float(player_feats.get("ast_per_36", 0.0))
        stl  = float(player_feats.get("stl_per_36", 0.0))
        blk  = float(player_feats.get("blk_per_36", 0.0))
        ver  = float(player_feats.get("versatility_score", 0.0))
        primary = position.split("/")[0]

        # Playmaker: high AST/USG ratio or raw assists (EU guard threshold lowered)
        if pm > 0.28 or ast > 5.5:
            return "🎯 Playmaker"
        # Stretch big: PF/C with 3pt-oriented profile
        if primary in ("PF", "C") and prof == "3pt_specialist":
            return "🎭 Stretch Big"
        # Classic big
        if primary in ("PF", "C") and (prof == "paint_scorer" or usg < 20):
            return "💪 Pivot / Big"
        # Defender: STL+BLK or defensive score (EU thresholds lowered)
        if (stl + blk) > 2.0 or df > 3.0:
            return "🛡️ Difensore"
        # Primary scorer (EU usg/pts thresholds)
        if usg > 20 or pts > 18:
            return "⚡ Scorer primario"
        # 3-point specialist
        if prof == "3pt_specialist":
            return "🏹 Specialista 3pt"
        # Two-way wing
        if primary in ("SF", "SG") and ver > 0.40 and df > 1.5:
            return "🔄 Ala tuttofare"
        return "📊 Role Player"

    # ------------------------------------------------------------------
    def compute_lineup_synergy(
        self,
        player_ids: List[int],
        team_id: int,
        season: int = 2024,
    ) -> LineupSynergyResult:
        """Compute a comprehensive synergy score for a set of players."""
        player_dict = self.data["player_dict"]
        team_dict   = self.data["team_dict"]
        team_row    = team_dict.get(_normalize_id(team_id), {})
        team_style  = str(team_row.get("playing_style", "motion_offense"))

        profiles: List[Dict[str, Any]] = []
        for pid in player_ids:
            p_row = player_dict.get(_normalize_id(pid), {})
            pos   = str(p_row.get("position", "PG"))
            name  = str(p_row.get("name", f"Player {pid}"))
            try:
                pred   = self.ensemble.predict(pid, team_id, self.data, season)
                rating = pred.predicted_rating
            except Exception:
                rating = 6.0
            p_feats  = compute_player_features(pid, self.data)
            role     = self._classify_role_advanced(p_feats, pos)
            style_c  = get_style_position_compat(team_style, pos)
            profiles.append({
                "pid": pid, "name": name, "pos": pos, "rating": rating,
                "role": role, "style_compat": style_c, "feats": p_feats,
            })

        # 1. Role diversity
        unique_roles = {p["role"] for p in profiles}
        role_diversity_score = float(np.clip(len(unique_roles) / max(1, min(len(profiles), 5)), 0, 1))
        role_dist: Dict[str, List[str]] = {}
        for p in profiles:
            role_dist.setdefault(p["role"], []).append(p["name"])
        key_roles_needed = {"🎯 Playmaker", "🛡️ Difensore"}
        missing_roles = sorted(key_roles_needed - unique_roles)

        # 2. Positional balance
        pos_set: set = set()
        for p in profiles:
            for comp in p["pos"].split("/"):
                pos_set.add(comp)
        all_5 = {"PG", "SG", "SF", "PF", "C"}
        positional_balance = float(len(pos_set & all_5) / 5.0)

        # 3. Offensive balance (USG distribution + spacing)
        usgs    = [float(p["feats"].get("avg_usg_pct", 18.0)) for p in profiles]
        total   = sum(usgs)
        mx_usg  = max(usgs) if usgs else 18.0
        spacers = sum(1 for p in profiles
                      if p["feats"].get("scoring_profile", "") == "3pt_specialist")
        usg_ok  = (50 <= total <= 135) and mx_usg <= 38
        offensive_balance = float(np.clip(
            (1.0 if usg_ok else 0.65) * (1.0 + min(spacers, 3) * 0.08), 0, 1
        ))

        # 4. Defensive score
        defensive_score = float(np.clip(
            sum(float(p["feats"].get("defensive_score", 2.0)) for p in profiles)
            / max(1, len(profiles)) * 2, 0, 10
        ))

        # 5. Style compatibility (avg)
        style_compat = float(np.mean([p["style_compat"] for p in profiles])) if profiles else 0.5

        # 6. Pairwise role compatibility
        _ROLE_COMPAT: Dict[tuple, float] = {
            # Classic two-man game pairs
            ("🎯 Playmaker", "🏹 Specialista 3pt"):   0.92,
            ("🎯 Playmaker", "💪 Pivot / Big"):        0.88,
            ("🎯 Playmaker", "🎭 Stretch Big"):        0.90,
            ("🎯 Playmaker", "⚡ Scorer primario"):    0.82,
            ("🎯 Playmaker", "🛡️ Difensore"):         0.78,
            ("🎯 Playmaker", "🔄 Ala tuttofare"):      0.80,
            ("🎯 Playmaker", "📊 Role Player"):        0.72,
            ("⚡ Scorer primario", "🛡️ Difensore"):   0.85,
            ("⚡ Scorer primario", "🏹 Specialista 3pt"): 0.80,
            ("⚡ Scorer primario", "💪 Pivot / Big"):  0.76,
            ("⚡ Scorer primario", "🎭 Stretch Big"):  0.78,
            ("⚡ Scorer primario", "📊 Role Player"):  0.70,
            ("🏹 Specialista 3pt", "💪 Pivot / Big"):  0.80,
            ("🏹 Specialista 3pt", "🛡️ Difensore"):   0.75,
            ("🏹 Specialista 3pt", "🔄 Ala tuttofare"): 0.77,
            ("🏹 Specialista 3pt", "📊 Role Player"):  0.68,
            ("🛡️ Difensore", "📊 Role Player"):       0.75,
            ("🛡️ Difensore", "🔄 Ala tuttofare"):     0.78,
            ("🛡️ Difensore", "💪 Pivot / Big"):       0.72,
            ("🔄 Ala tuttofare", "🎯 Playmaker"):      0.80,
            ("🔄 Ala tuttofare", "💪 Pivot / Big"):    0.74,
            ("🔄 Ala tuttofare", "📊 Role Player"):    0.70,
            ("🎭 Stretch Big", "🎯 Playmaker"):        0.88,
            ("🎭 Stretch Big", "⚡ Scorer primario"):  0.82,
            ("🎭 Stretch Big", "🛡️ Difensore"):       0.74,
            ("🎭 Stretch Big", "📊 Role Player"):      0.68,
            ("💪 Pivot / Big", "📊 Role Player"):      0.65,
            # Same-role pairs (generally lower synergy due to role overlap)
            ("🎯 Playmaker", "🎯 Playmaker"):           0.62,
            ("⚡ Scorer primario", "⚡ Scorer primario"): 0.60,
            ("🛡️ Difensore", "🛡️ Difensore"):         0.65,
            ("🏹 Specialista 3pt", "🏹 Specialista 3pt"): 0.68,
            ("💪 Pivot / Big", "💪 Pivot / Big"):       0.58,
            ("🎭 Stretch Big", "🎭 Stretch Big"):       0.62,
            ("🔄 Ala tuttofare", "🔄 Ala tuttofare"):   0.66,
            ("📊 Role Player", "📊 Role Player"):       0.62,
        }
        pairwise: Dict[str, float] = {}
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                a, b  = profiles[i], profiles[j]
                pair  = tuple(sorted([a["role"], b["role"]]))
                base  = _ROLE_COMPAT.get(pair, 0.60)
                pos_a = set(a["pos"].split("/"))
                pos_b = set(b["pos"].split("/"))
                pos_adj  = 0.06 if not (pos_a & pos_b) else -0.04
                # Bonus for high-quality partnerships (both high-rated)
                avg_rtg  = (a["rating"] + b["rating"]) / 2.0
                rtg_adj  = float(np.clip((avg_rtg - 6.5) * 0.015, -0.05, 0.08))
                pairwise[f"{a['name']} – {b['name']}"] = float(
                    np.clip(base + pos_adj + rtg_adj, 0.30, 1.0)
                )

        # 7. Overall synergy (0–10, weighted composite)
        overall = float(np.clip(
            role_diversity_score  * 3.0
            + positional_balance  * 2.5
            + offensive_balance   * 2.0
            + (defensive_score / 10.0) * 1.5
            + style_compat        * 1.0,
            0, 10,
        ))
        synergy_bonus = float(np.clip(overall / 25.0, 0.0, 0.4))

        parts: List[str] = []
        if missing_roles:
            parts.append(f"⚠️ Ruoli mancanti: {', '.join(missing_roles)}")
        if spacers >= 2:
            parts.append(f"✅ {spacers} tiratori da 3 (ottimo spacing)")
        if positional_balance >= 1.0:
            parts.append("✅ Quintetto completo")
        if overall >= 7.0:
            parts.append("⭐ Alta sinergia")

        return LineupSynergyResult(
            overall_synergy=round(overall, 2),
            role_diversity_score=round(role_diversity_score, 2),
            positional_balance=round(positional_balance, 2),
            offensive_balance=round(offensive_balance, 2),
            defensive_score=round(defensive_score, 2),
            style_compat=round(style_compat, 2),
            role_distribution=role_dist,
            missing_roles=missing_roles,
            synergy_bonus=round(synergy_bonus, 3),
            pairwise_compat={k: round(v, 2) for k, v in pairwise.items()},
            summary=" | ".join(parts) if parts else "Quintetto equilibrato",
        )

    # ------------------------------------------------------------------
    def best_lineup_by_roles(
        self,
        target_player_id: int,
        team_id: int,
        desired_roles: List[str],
        season: int = 2024,
        top_n_per_role: int = 5,
    ) -> LineupByRolesResult:
        """Find the best players for each desired role around a target player.

        Args:
            target_player_id: The focal player.
            team_id:          Context team for predictions.
            desired_roles:    List of role keys from ADVANCED_ROLES.
            season:           Season for predictions.
            top_n_per_role:   How many candidates per role to return.
        """
        players_df  = self.data["players"]
        player_dict = self.data["player_dict"]
        team_dict   = self.data["team_dict"]
        target_name = str(player_dict.get(_normalize_id(target_player_id), {}).get(
            "name", f"Player {target_player_id}"
        ))

        # Sample players (cap for performance)
        sample = players_df.sample(min(700, len(players_df)), random_state=42)
        all_scored: List[Dict[str, Any]] = []
        for _, row in sample.iterrows():
            pid = _normalize_id(row["id"])
            if pid == _normalize_id(target_player_id):
                continue
            pos  = str(row.get("position", "PG"))
            name = str(row.get("name", f"Player {pid}"))
            try:
                pred      = self.ensemble.predict(pid, team_id, self.data, season)
                p_feats   = compute_player_features(pid, self.data)
                role      = self._classify_role_advanced(p_feats, pos)
                cur_tid   = row.get("current_team_id")
                cur_team  = str(team_dict.get(
                    _normalize_id(cur_tid) if cur_tid else None, {}
                ).get("name", "Unknown"))
                all_scored.append({
                    "player_id":       pid,
                    "player_name":     name,
                    "position":        pos,
                    "predicted_rating": round(pred.predicted_rating, 3),
                    "role":            role,
                    "current_team":    cur_team,
                    "style_compat":    round(
                        get_style_position_compat(
                            str(team_dict.get(_normalize_id(team_id), {}).get("playing_style", "")), pos
                        ), 3
                    ),
                })
            except Exception:
                continue

        # Build per-role candidate lists
        role_candidates: List[RoleCandidateResult] = []
        for role in desired_roles:
            matches = [p for p in all_scored if p["role"] == role]
            matches.sort(key=lambda p: p["predicted_rating"], reverse=True)
            role_candidates.append(RoleCandidateResult(
                role=role,
                description=ADVANCED_ROLES.get(role, ""),
                candidates=matches[:top_n_per_role],
            ))

        # Optimal lineup: best unique player per role (greedy)
        used: set = set()
        optimal: List[Dict[str, Any]] = []
        for rc in role_candidates:
            for cand in rc.candidates:
                if cand["player_id"] not in used:
                    optimal.append({**cand, "target_role": rc.role})
                    used.add(cand["player_id"])
                    break

        avg_r = float(np.mean([p["predicted_rating"] for p in optimal])) if optimal else 0.0

        return LineupByRolesResult(
            target_player_id=_normalize_id(target_player_id),
            target_player_name=target_name,
            team_id=team_id,
            season=season,
            role_candidates=role_candidates,
            optimal_lineup=optimal,
            estimated_avg_rating=round(avg_r, 3),
        )


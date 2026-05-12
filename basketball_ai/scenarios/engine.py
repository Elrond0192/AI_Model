"""Basketball What-If scenario engine."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from basketball_ai.models.ensemble import EnsembleModel, PredictionResult
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
        player      = self.data["player_dict"].get(int(player_id), {})
        current_age = int(player.get("age", 25))
        position    = str(player.get("position", "PG"))

        if team_id is None:
            ct = player.get("current_team_id")
            team_id = int(ct) if ct else 1

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
            team   = team_dict.get(int(tid), {})
            league = league_dict.get(int(team.get("league_id", 1)), {})
            scenarios.append({
                "team_id":        tid,
                "team_name":      str(team.get("name", f"Team {tid}")),
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
            tid = int(row["id"])
            try:
                pred   = self.ensemble.predict(player_id, tid, self.data, season)
                ctx    = compute_context_features(player_id, tid, self.data)
                team   = team_dict.get(tid, {})
                league = league_dict.get(int(row["league_id"]), {})
                results.append(TeamFitResult(
                    team_id=tid,
                    team_name=str(team.get("name", f"Team {tid}")),
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
            pid = int(row["id"])
            try:
                pred     = self.ensemble.predict(pid, team_id, self.data, season)
                cur_team = team_dict.get(int(row.get("current_team_id") or 0), {})
                results.append(PlayerFitResult(
                    player_id=pid,
                    player_name=str(row["name"]),
                    position=str(row["position"]),
                    predicted_rating=pred.predicted_rating,
                    current_team=str(cur_team.get("name", "Unknown")),
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
    def predict_peak(self, player_id: int, team_id: Optional[int] = None) -> PeakPrediction:
        """Predict a player's career peak rating."""
        player      = self.data["player_dict"].get(int(player_id), {})
        current_age = int(player.get("age", 25))
        position    = str(player.get("position", "PG"))
        name        = str(player.get("name", f"Player {player_id}"))

        if team_id is None:
            ct = player.get("current_team_id")
            team_id = int(ct) if ct else 1

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
        """Heuristic role label from computed player features."""
        pm   = float(player_feats.get("playmaking_score", 0.0))
        df   = float(player_feats.get("defensive_score", 0.0))
        usg  = float(player_feats.get("avg_usg_pct", 0.0))
        prof = str(player_feats.get("scoring_profile", "efficient_scorer"))
        if pm > 0.35:
            return "Playmaker"
        if df > 4.0:
            return "Defender"
        if prof == "3pt_specialist":
            return "3pt Specialist"
        if usg > 24:
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
        target_name = str(player_dict.get(int(player_id), {}).get("name", f"Player {player_id}"))
        team_row    = team_dict.get(int(team_id), {})
        team_style  = str(team_row.get("playing_style", "motion_offense"))

        # --- 1. Profile each lineup member --------------------------------
        profiles: List[LineupMemberProfile] = []
        for pid in lineup_player_ids:
            p_row = player_dict.get(int(pid), {})
            pos   = str(p_row.get("position", "PG"))
            name  = str(p_row.get("name", f"Player {pid}"))
            try:
                pred = self.ensemble.predict(pid, team_id, self.data, season)
                r    = pred.predicted_rating
            except Exception:
                r = 6.0
            p_feats    = compute_player_features(pid, self.data)
            role       = self._assign_role(p_feats)
            style_compat = get_style_position_compat(team_style, pos)
            profiles.append(LineupMemberProfile(
                player_id=pid, player_name=name, position=pos,
                predicted_rating=round(r, 3), role=role,
                style_compat=round(style_compat, 3),
            ))

        # --- 2. Position coverage (all 5 slots incl. target player) ------
        target_pos = str(player_dict.get(int(player_id), {}).get("position", "PG"))
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

"""Basketball What-If scenario engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models.ensemble import EnsembleModel, PredictionResult
from src.models.age_curve import age_performance_factor, PEAK_AGES, peak_age_window
from src.features.context_features import compute_context_features
from src.features.team_features import compute_team_features


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
# Engine
# ---------------------------------------------------------------------------

class WhatIfEngine:
    """High-level basketball scenario engine wrapping EnsembleModel."""

    def __init__(self, ensemble: EnsembleModel, data: Dict[str, Any]) -> None:
        self.ensemble = ensemble
        self.data     = data

    # ------------------------------------------------------------------
    def predict_in_team(
        self, player_id: int, target_team_id: int, season: int = 2024
    ) -> PredictionResult:
        """Predict how a player would perform at a given team."""
        return self.ensemble.predict(player_id, target_team_id, self.data, season)

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
    ) -> ComparisonResult:
        """Compare player performance across multiple team scenarios."""
        team_dict   = self.data["team_dict"]
        league_dict = self.data["league_dict"]

        scenarios = []
        for tid in team_ids:
            pred   = self.ensemble.predict(player_id, tid, self.data, season)
            team   = team_dict.get(int(tid), {})
            league = league_dict.get(int(team.get("league_id", 1)), {})
            scenarios.append({
                "team_id":        tid,
                "team_name":      str(team.get("name", f"Team {tid}")),
                "league_name":    str(league.get("name", "Unknown")),
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
        self, player_id: int, from_team_id: int, to_team_id: int, season: int = 2024,
    ) -> TransferImpactResult:
        """Simulate the performance impact of a transfer between two teams."""
        pred_before = self.ensemble.predict(player_id, from_team_id, self.data, season)
        pred_after  = self.ensemble.predict(player_id, to_team_id,   self.data, season)
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

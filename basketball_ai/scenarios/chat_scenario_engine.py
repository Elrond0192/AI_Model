"""Composable, data-backed basketball scenario engine used by Chat V3.

The engine deliberately separates supervised model outputs from empirical or
analytical scenario estimates. Every response carries method/support metadata so
Chat V3 can explain uncertainty instead of presenting heuristics as trained ML.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from basketball_ai.bb_rating.competition_tiers import (
    competition_tier,
    competition_tier_label,
)
from basketball_ai.data.loader import _to_int
from basketball_ai.features.context_features import compute_context_features
from basketball_ai.models.competition_training import (
    normalize_competition,
    scope_prediction_context,
)
from basketball_ai.models.production_training import season_year
from basketball_ai.models.strict_production import (
    StrictProductionEnsembleModel,
    StrictWhatIfEngine,
    build_historical_snapshot,
)
from basketball_ai.scenarios.advanced_simulation_engine import AdvancedSimulationEngine

_PLAYER_METRICS = (
    "rating", "games_played", "minutes_per_game", "points", "rebounds", "assists",
    "steals", "blocks", "ts_pct", "usg_pct", "bpm", "obpm", "dbpm", "net_rtg",
    "ortg", "drtg", "three_point_pct", "three_par", "ast_pct", "tov_pct",
    "raptor_total", "lebron_total", "vorp",
)
_TEAM_METRICS = (
    "pace", "offensive_rating", "defensive_rating", "net_rtg",
    "three_point_attempt_rate", "assists_per_game", "star_player_usage",
)
_STYLE_BOUNDS = {
    "pace": (50.0, 130.0),
    "three_point_attempt_rate": (0.0, 1.0),
    "assists_per_game": (0.0, 50.0),
    "star_player_usage": (0.0, 1.0),
    "offensive_rating": (50.0, 160.0),
    "defensive_rating": (50.0, 160.0),
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _ratio(value: Any, default: float = 0.0) -> float:
    number = _finite(value, default)
    return number / 100.0 if abs(number) > 1.5 else number


def _year(value: Any) -> int:
    return season_year(value)


def _serialise(row: pd.Series | dict[str, Any], metrics: Iterable[str]) -> dict[str, Any]:
    source = row if isinstance(row, dict) else row.to_dict()
    out: dict[str, Any] = {}
    for metric in metrics:
        value = source.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out[metric] = value
    return out


def _confidence(samples: int) -> str:
    if samples >= 40:
        return "high"
    if samples >= 15:
        return "medium"
    if samples >= 5:
        return "low-medium"
    return "low"


class ChatScenarioEngine:
    """Evaluate structured basketball questions for the conversational layer."""

    def __init__(
        self,
        ensemble: StrictProductionEnsembleModel,
        data: dict[str, Any],
        bb_rating_engine: Any | None = None,
        bb_rating_uncertainty: Any | None = None,
        future_performance_model: Any | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.data = data
        self.advanced = AdvancedSimulationEngine(data)
        self.bb_rating_engine = bb_rating_engine
        self.bb_rating_uncertainty = bb_rating_uncertainty
        self.future_performance_model = future_performance_model

    def evaluate(
        self,
        spec: dict[str, Any],
        player_ids: list[int],
        team_ids: list[int],
        source_league_id: int | None,
        target_league_id: int | None,
    ) -> dict[str, Any]:
        scenario = str(spec["scenario"])
        dispatch = {
            "player_competition": self._player_competition,
            "player_intelligence": self._player_intelligence,
            "team_competition": self._team_competition,
            "player_trend": self._player_trend,
            "performance_decomposition": self._performance_decomposition,
            "metric_explanation": self._metric_explanation,
            "role_analysis": self._role_analysis,
            "performance_stability": self._performance_stability,
            "team_usage_analysis": self._team_usage_analysis,
            "regression_risk": self._regression_risk,
            "potential_synthesis": self._potential_synthesis,
            "shooting_decomposition": self._shooting_decomposition,
            "defensive_decomposition": self._defensive_decomposition,
            "team_trend": self._team_trend,
            "player_compare": self._player_compare,
            "team_compare": self._team_compare,
            "player_team": self._player_team,
            "playoff_role": self._playoff_role,
            "league_transfer": self._league_transfer,
            "player_pair": self._player_pair,
            "lineup_fit": self._lineup_fit,
            "style_change": self._style_change,
            "player_role_change": self._player_role_change,
            "role_minutes_projection": self._player_role_change,
            "clutch_analysis": self._clutch_analysis,
            "team_add_player": self._team_add_player,
            "team_replace_player": self._team_replace_player,
            "best_team_fit": self._best_team_fit,
            "best_player_fit": self._best_player_fit,
            "player_similarity": self._player_similarity,
            "age_trajectory": self._age_trajectory,
            "probabilistic_boxscore": self.advanced.probabilistic_boxscore,
            "opponent_matchup": self.advanced.opponent_matchup,
            "defensive_matchup": self.advanced.defensive_matchup,
            "play_type_matchup": self.advanced.play_type_matchup,
            "shot_profile_counterfactual": self.advanced.shot_counterfactual,
            "lineup_synergy": self.advanced.lineup_synergy,
            "lineup_optimizer": self.advanced.lineup_optimizer,
            "roster_optimizer": self.advanced.roster_optimizer,
            "composite_scenario": self.advanced.composite,
            "causal_effect": self.advanced.causal_effect,
            "causal_team_effect": self._causal_team_effect,
            "model_evidence": self._model_evidence,
        }
        if scenario not in dispatch:
            raise ValueError(f"Unsupported scenario {scenario!r}")
        return dispatch[scenario](
            spec, player_ids, team_ids, source_league_id, target_league_id
        )

    # ------------------------------------------------------------------
    # Identity/context helpers
    # ------------------------------------------------------------------
    def _player_name(self, player_id: int) -> str:
        row = self.data.get("player_dict", {}).get(_to_int(player_id), {})
        return str(row.get("name", f"Player {player_id}"))

    def _team_name(self, team_id: int) -> str:
        row = self.data.get("team_dict", {}).get(_to_int(team_id), {})
        return str(row.get("name", f"Team {team_id}"))

    def _league_name(self, league_id: int | None) -> str:
        if league_id is None:
            return ""
        row = self.data.get("league_dict", {}).get(_to_int(league_id), {})
        return str(row.get("name", row.get("league_key", league_id)))

    def _league_key(self, league_id: int | None) -> str:
        if league_id is None:
            return ""
        row = self.data.get("league_dict", {}).get(_to_int(league_id), {})
        value = row.get("league_key", row.get("code", row.get("name", league_id)))
        return str(value).strip().upper()


    def _league_key(self, league_id: int | None) -> str:
        if league_id is None:
            return ""
        row = self.data.get("league_dict", {}).get(_to_int(league_id), {})
        value = row.get("league_key", row.get("code", row.get("name", league_id)))
        return str(value).strip().upper()

    def _player_rows(
        self,
        player_id: int,
        league_id: int | None,
        competition: str,
        through_season: int,
    ) -> pd.DataFrame:
        frame = self.data.get("player_stats", pd.DataFrame()).copy()
        if frame.empty:
            return frame
        frame["_year"] = frame["season"].map(_year)
        mask = (
            (frame["player_id"].map(_to_int) == _to_int(player_id))
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (frame["_year"] <= int(through_season))
        )
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        return frame[mask].sort_values("_year").reset_index(drop=True)

    def _team_rows(
        self,
        team_id: int,
        league_id: int | None,
        competition: str,
        through_season: int,
    ) -> pd.DataFrame:
        frame = self.data.get("team_season_stats", pd.DataFrame()).copy()
        if frame.empty:
            return frame
        frame["_year"] = frame["season"].map(_year)
        mask = (
            (frame["team_id"].map(_to_int) == _to_int(team_id))
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (frame["_year"] <= int(through_season))
        )
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        return frame[mask].sort_values("_year").reset_index(drop=True)

    @staticmethod
    def _latest(rows: pd.DataFrame) -> pd.Series:
        if rows.empty:
            raise ValueError("No data is available for the requested context")
        return rows.iloc[-1]

    def _direct_prediction(
        self,
        player_id: int,
        team_id: int,
        league_id: int,
        season: int,
        competition: str,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        snapshot = snapshot or build_historical_snapshot(self.data, season)
        scoped = scope_prediction_context(
            snapshot,
            player_id,
            team_id,
            league_id,
            competition,
            season,
        )
        engine = StrictWhatIfEngine(self.ensemble, scoped)
        result = engine.predict_in_team(
            player_id, team_id, season, competition=competition
        )
        return result, scoped

    # ------------------------------------------------------------------
    # Descriptive analysis
    # ------------------------------------------------------------------
    def _player_competition(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        competition = normalize_competition(spec.get("competition"))
        comparison = normalize_competition(spec.get("comparison_competition") or "RS")
        season = int(spec["season"])
        rows = self._player_rows(players[0], source_league, competition, season)
        latest = self._latest(rows)
        compare_rows = self._player_rows(players[0], source_league, comparison, season)
        compare = compare_rows.iloc[-1] if not compare_rows.empty else None
        deltas: dict[str, float] = {}
        if compare is not None and competition != comparison:
            for metric in ("rating", "minutes_per_game", "points", "assists", "ts_pct", "usg_pct", "bpm", "net_rtg"):
                if metric in latest and metric in compare and pd.notna(latest.get(metric)) and pd.notna(compare.get(metric)):
                    deltas[metric] = round(_finite(latest[metric]) - _finite(compare[metric]), 4)
        games = int(pd.to_numeric(rows.get("games_played", 0), errors="coerce").fillna(0).sum())
        return {
            "result": {
                "player": self._player_name(players[0]),
                "league": self._league_name(source_league),
                "competition": competition,
                "latest_season": int(latest["_year"]),
                "latest": _serialise(latest, _PLAYER_METRICS),
                "comparison_competition": comparison if compare is not None else None,
                "comparison": _serialise(compare, _PLAYER_METRICS) if compare is not None else None,
                "deltas": deltas,
            },
            "evidence": [{"type": "historical_seasons", "count": int(len(rows)), "games": games}],
            "support": {"method": "observed_data", "samples": games, "confidence": _confidence(games)},
            "limitations": [],
        }

    def _team_competition(self, spec, players, teams, source_league, target_league):
        del players, target_league
        if not teams:
            raise ValueError("team is required")
        competition = normalize_competition(spec.get("competition"))
        comparison = normalize_competition(spec.get("comparison_competition") or "RS")
        season = int(spec["season"])
        rows = self._team_rows(teams[0], source_league, competition, season)
        latest = self._latest(rows)
        compare_rows = self._team_rows(teams[0], source_league, comparison, season)
        compare = compare_rows.iloc[-1] if not compare_rows.empty else None
        deltas: dict[str, float] = {}
        if compare is not None and competition != comparison:
            for metric in _TEAM_METRICS:
                if pd.notna(latest.get(metric)) and pd.notna(compare.get(metric)):
                    deltas[metric] = round(_finite(latest[metric]) - _finite(compare[metric]), 4)
        return {
            "result": {
                "team": self._team_name(teams[0]),
                "league": self._league_name(source_league),
                "competition": competition,
                "latest_season": int(latest["_year"]),
                "latest": _serialise(latest, _TEAM_METRICS),
                "comparison_competition": comparison if compare is not None else None,
                "comparison": _serialise(compare, _TEAM_METRICS) if compare is not None else None,
                "deltas": deltas,
            },
            "evidence": [{"type": "historical_seasons", "count": int(len(rows))}],
            "support": {"method": "observed_data", "samples": int(len(rows)), "confidence": _confidence(len(rows) * 5)},
            "limitations": [],
        }

    def _performance_decomposition(self, spec, players, teams, source_league, target_league):
        """Decompose season-over-season player change into volume, efficiency, role and impact."""
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        season = int(spec["season"])
        competition = normalize_competition(spec["competition"])
        rows = self._player_rows(players[0], source_league, competition, season)
        if len(rows) < 2:
            raise ValueError("At least two consecutive observed seasons are required")

        current = rows.iloc[-1]
        previous = rows.iloc[-2]
        current_year = int(current["_year"])
        previous_year = int(previous["_year"])
        if current_year != previous_year + 1:
            raise ValueError("Performance decomposition requires consecutive seasons")

        metric_groups = {
            "volume": ("minutes_per_game", "points", "assists", "rebounds"),
            "efficiency": ("ts_pct", "three_point_pct", "ft_pct", "tov_pct"),
            "role": ("usg_pct", "ast_pct", "three_par"),
            "impact": ("rating", "bpm", "obpm", "dbpm", "net_rtg"),
        }
        deltas: dict[str, dict[str, float]] = {}
        for group, metrics in metric_groups.items():
            group_delta: dict[str, float] = {}
            for metric in metrics:
                a = current.get(metric)
                b = previous.get(metric)
                if pd.notna(a) and pd.notna(b):
                    group_delta[metric] = round(_finite(a) - _finite(b), 4)
            deltas[group] = group_delta

        minutes_delta = deltas["volume"].get("minutes_per_game", 0.0)
        points_delta = deltas["volume"].get("points", 0.0)
        ts_delta = deltas["efficiency"].get("ts_pct", 0.0)
        rating_delta = deltas["impact"].get("rating", 0.0)

        if abs(minutes_delta) > 1.5 and abs(points_delta) > 0:
            volume_interpretation = "production_change is partly associated with a meaningful minutes change"
        else:
            volume_interpretation = "production change is not primarily explained by minutes"

        if abs(ts_delta) >= 0.02:
            efficiency_interpretation = "efficiency changed materially"
        else:
            efficiency_interpretation = "efficiency was relatively stable"

        if abs(rating_delta) >= 1.0:
            impact_interpretation = "overall impact rating changed materially"
        else:
            impact_interpretation = "overall impact rating was relatively stable"

        return {
            "result": {
                "player": self._player_name(players[0]),
                "league": self._league_name(source_league),
                "competition": competition,
                "previous_season": previous_year,
                "current_season": current_year,
                "previous": _serialise(previous, _PLAYER_METRICS),
                "current": _serialise(current, _PLAYER_METRICS),
                "deltas": deltas,
                "interpretation": {
                    "volume": volume_interpretation,
                    "efficiency": efficiency_interpretation,
                    "impact": impact_interpretation,
                },
            },
            "evidence": [
                {"type": "consecutive_seasons", "count": 2},
                {"type": "metric_groups", "groups": list(metric_groups)},
            ],
            "support": {
                "method": "observed_season_over_season_decomposition",
                "samples": int(current.get("games_played", 0) or 0),
                "confidence": _confidence(int(current.get("games_played", 0) or 0)),
            },
            "limitations": [
                "This decomposition describes changes; it does not establish causal effects."
            ],
        }

    def _player_trend(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        rows = self._player_rows(players[0], source_league, spec["competition"], int(spec["season"]))
        if rows.empty:
            raise ValueError("No player history is available")
        series = [
            {"season": int(row["_year"]), **_serialise(row, _PLAYER_METRICS)}
            for row in rows.to_dict("records")
        ]
        first, last = rows.iloc[0], rows.iloc[-1]
        return {
            "result": {
                "player": self._player_name(players[0]),
                "competition": normalize_competition(spec["competition"]),
                "series": series,
                "rating_change": round(_finite(last.get("rating")) - _finite(first.get("rating")), 3),
                "peak_rating": round(float(pd.to_numeric(rows["rating"], errors="coerce").max()), 3) if "rating" in rows else None,
            },
            "evidence": [{"type": "seasons", "count": len(series)}],
            "support": {"method": "observed_data", "samples": len(series), "confidence": _confidence(len(series) * 6)},
            "limitations": [],
        }

    def _team_trend(self, spec, players, teams, source_league, target_league):
        del players, target_league
        if not teams:
            raise ValueError("team is required")
        rows = self._team_rows(teams[0], source_league, spec["competition"], int(spec["season"]))
        if rows.empty:
            raise ValueError("No team history is available")
        series = [
            {"season": int(row["_year"]), **_serialise(row, _TEAM_METRICS)}
            for row in rows.to_dict("records")
        ]
        return {
            "result": {"team": self._team_name(teams[0]), "competition": normalize_competition(spec["competition"]), "series": series},
            "evidence": [{"type": "seasons", "count": len(series)}],
            "support": {"method": "observed_data", "samples": len(series), "confidence": _confidence(len(series) * 6)},
            "limitations": [],
        }

    def _player_compare(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if len(players) < 2:
            raise ValueError("at least two players are required")
        season, competition = int(spec["season"]), normalize_competition(spec["competition"])
        compared = []
        for player_id in players[:8]:
            latest = self._latest(self._player_rows(player_id, source_league, competition, season))
            compared.append({"player": self._player_name(player_id), "season": int(latest["_year"]), "metrics": _serialise(latest, _PLAYER_METRICS)})
        return {"result": {"competition": competition, "players": compared}, "evidence": [], "support": {"method": "observed_data", "samples": len(compared), "confidence": "high"}, "limitations": []}

    def _team_compare(self, spec, players, teams, source_league, target_league):
        del players, target_league
        if len(teams) < 2:
            raise ValueError("at least two teams are required")
        season, competition = int(spec["season"]), normalize_competition(spec["competition"])
        compared = []
        for team_id in teams[:8]:
            latest = self._latest(self._team_rows(team_id, source_league, competition, season))
            compared.append({"team": self._team_name(team_id), "season": int(latest["_year"]), "metrics": _serialise(latest, _TEAM_METRICS)})
        return {"result": {"competition": competition, "teams": compared}, "evidence": [], "support": {"method": "observed_data", "samples": len(compared), "confidence": "high"}, "limitations": []}

    # ------------------------------------------------------------------
    # Supervised/direct prediction and cross-league transfer
    # ------------------------------------------------------------------
    def _clutch_analysis(self, spec, players, teams, source_league, target_league):
        if not players:
            raise ValueError("player is required")
        season = int(spec["season"])
        competition = normalize_competition(spec["competition"])
        rows = self._player_rows(players[0], source_league, competition, season)
        if rows.empty:
            raise ValueError("No player history is available for the requested clutch context")
        metrics = ("clutch_games", "clutch_pts", "clutch_ts_pct", "clutch_net_rtg", "clutch_ast_to_tov")
        latest = _serialise(self._latest(rows), metrics)
        history = [
            {"season": int(row["_year"]), **_serialise(row, metrics)}
            for row in rows.tail(8).to_dict("records")
        ]
        samples = int(pd.to_numeric(rows.get("clutch_games"), errors="coerce").fillna(0).sum()) if "clutch_games" in rows else 0
        return {
            "result": {"player": self._player_name(players[0]), "competition": competition, "through_season": season, "latest_clutch": latest, "clutch_history": history},
            "evidence": [{"type": "observed_clutch_advanced_stats", "seasons_considered": len(rows), "clutch_games": samples}],
            "support": {"method": "observed_clutch_advanced_stats", "samples": samples, "confidence": _confidence(samples)},
            "limitations": ["This reports observed clutch aggregates. Possession-level last-five-minute simulations require PBP clock and score fields for the requested competition."],
        }

    def _causal_team_effect(self, spec, players, teams, source_league, target_league):
        """Estimate team-context association without claiming causal identification.

        The question is intentionally answered as an observational attribution:
        we compare a player's own rating across team-season observations and
        quantify whether changes in team context move with changes in the player's
        observed rating. A causal effect is never produced by this method.
        """
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        if source_league is None:
            raise ValueError("league context is required")

        season = int(spec["season"])
        competition = normalize_competition(spec["competition"])
        player_rows = self._player_rows(players[0], source_league, competition, season)
        if player_rows.empty:
            raise ValueError("No player history is available for the requested team-effect context")
        if "team_id" not in player_rows.columns:
            raise ValueError("Player team history is unavailable for the requested context")

        team_stats = self.data.get("team_season_stats", pd.DataFrame()).copy()
        if team_stats.empty or "team_id" not in team_stats.columns:
            raise ValueError("Team context is unavailable for the requested context")
        team_stats["_year"] = team_stats["season"].map(_year)
        team_stats["_team_int"] = team_stats["team_id"].map(_to_int)
        team_stats["_league_int"] = team_stats["league_id"].map(_to_int)
        team_stats["_comp"] = team_stats["competition"].map(normalize_competition)
        team_stats = team_stats[
            (team_stats["_league_int"] == _to_int(source_league))
            & (team_stats["_comp"] == competition)
            & (team_stats["_year"] <= season)
        ].copy()

        context_metrics = (
            "offensive_rating",
            "defensive_rating",
            "assists_per_game",
            "three_point_attempt_rate",
            "pace",
        )
        records = []
        for row in player_rows.to_dict("records"):
            year = int(row["_year"])
            team_id = _to_int(row.get("team_id"))
            if team_id is None:
                continue
            candidates = team_stats[
                (team_stats["_team_int"] == team_id)
                & (team_stats["_year"] == year)
            ]
            if candidates.empty:
                continue
            team = candidates.iloc[-1]
            values = {metric: _finite(team.get(metric), np.nan) for metric in context_metrics}
            if all(not math.isfinite(value) for value in values.values()):
                continue

            # Convert team context to within-league-year percentiles. Offensive
            # efficiency, creation and spacing are positive; defensive rating is
            # inverted; pace is treated as a style dimension around league median.
            peers = team_stats[team_stats["_year"] == year]
            def pct(metric, value):
                vals = pd.to_numeric(peers.get(metric), errors="coerce").dropna()
                if vals.empty or not math.isfinite(value):
                    return 0.5
                return float((vals <= value).mean())

            offense = pct("offensive_rating", values["offensive_rating"])
            defense = 1.0 - pct("defensive_rating", values["defensive_rating"])
            creation = pct("assists_per_game", values["assists_per_game"])
            spacing = pct("three_point_attempt_rate", values["three_point_attempt_rate"])
            pace_values = pd.to_numeric(peers.get("pace"), errors="coerce").dropna()
            pace_fit = 0.5
            if not pace_values.empty and math.isfinite(values["pace"]):
                scale = float(pace_values.std(ddof=0)) or 1.0
                pace_fit = float(np.clip(1.0 - abs(values["pace"] - float(pace_values.median())) / (2.0 * scale), 0.0, 1.0))
            context_score = float(np.clip(
                offense * 0.30 + defense * 0.30 + creation * 0.15 + spacing * 0.15 + pace_fit * 0.10,
                0.0, 1.0,
            ))
            rating = _finite(row.get("rating"), np.nan)
            if math.isfinite(rating):
                records.append({
                    "season": year,
                    "team_id": team_id,
                    "team": self._team_name(team_id),
                    "rating": rating,
                    "team_context_score": context_score,
                })

        if not records:
            raise ValueError("Team context is unavailable for the player's observed history")

        history = pd.DataFrame(records).sort_values(["season", "team_id"]).reset_index(drop=True)
        correlation = None
        if len(history) >= 3 and history["rating"].nunique() >= 2 and history["team_context_score"].nunique() >= 2:
            correlation = float(history["rating"].corr(history["team_context_score"]))

        switches = []
        previous = None
        for row in history.sort_values("season").to_dict("records"):
            if previous is not None and row["team_id"] != previous["team_id"]:
                switches.append({
                    "from_season": previous["season"],
                    "to_season": row["season"],
                    "from_team": previous["team"],
                    "to_team": row["team"],
                    "rating_change": round(row["rating"] - previous["rating"], 3),
                    "team_context_change": round(row["team_context_score"] - previous["team_context_score"], 3),
                })
            previous = row

        latest = history.iloc[-1]
        if correlation is None:
            association_label = "insufficient_evidence"
        elif correlation >= 0.35:
            association_label = "positive_association"
        elif correlation <= -0.35:
            association_label = "negative_association"
        else:
            association_label = "weak_or_mixed_association"

        limitations = [
            "Questa analisi misura un'associazione osservata, non un effetto causale della squadra sul giocatore.",
            "Il contesto squadra può essere confuso con ruolo, compagni, allenatore, infortuni, calendario e selezione del campione.",
            "L'identificazione causale resta non disponibile senza un controfattuale e un controllo esplicito dei confondenti.",
        ]
        if len(switches) < 2:
            limitations.append("Ci sono pochi cambi di squadra osservati per separare meglio l'effetto del contesto dalla variabilità individuale.")

        return {
            "result": {
                "player": self._player_name(players[0]),
                "league": self._league_name(source_league),
                "competition": competition,
                "through_season": season,
                "causal_identification": "unavailable",
                "causal_effect": None,
                "association": {
                    "label": association_label,
                    "rating_vs_team_context_correlation": round(correlation, 3) if correlation is not None else None,
                },
                "current_context": {
                    "team": str(latest["team"]),
                    "season": int(latest["season"]),
                    "rating": round(float(latest["rating"]), 3),
                    "team_context_score": round(float(latest["team_context_score"]), 3),
                },
                "team_switches": switches[-8:],
                "observations": int(len(history)),
            },
            "evidence": [
                {"type": "within_player_team_seasons", "count": int(len(history))},
                {"type": "observed_team_switches", "count": int(len(switches))},
            ],
            "support": {
                "method": "within_player_observational_team_context",
                "samples": int(len(history)),
                "confidence": _confidence(len(history)),
            },
            "limitations": limitations,
        }

    def _player_team(self, spec, players, teams, source_league, target_league):
        """Player-team counterfactual with an optional historical source cutoff."""
        if not players or not teams:
            raise ValueError("player and team are required")
        source_league = source_league or target_league
        target_league = target_league or source_league
        if source_league is None or target_league is None:
            raise ValueError("league context is required")
        parameters = spec.get("parameters", {}) or {}
        player_season = int(parameters.get("player_season", spec["season"]))
        target_season = int(parameters.get("target_season", player_season + 1))
        if target_season <= player_season:
            raise ValueError("target_season must be later than player_season")
        target_competition = normalize_competition(spec.get("target_competition") or spec["competition"])
        source_competition = normalize_competition(spec["competition"])

        if source_league != target_league:
            return self._league_transfer(
                {**spec, "season": player_season,
                 "parameters": {**parameters, "player_season": player_season, "target_season": target_season}},
                players, teams, source_league, target_league
            )

        snapshot = build_historical_snapshot(self.data, player_season)
        player_rows = self._player_rows(players[0], source_league, source_competition, player_season)
        if player_rows.empty:
            raise ValueError("Player has no source-season history for the requested cutoff")
        team_rows = self._team_rows(teams[0], target_league, target_competition, player_season)
        if team_rows.empty:
            raise ValueError("Target team has no historical context available before the target season")

        scoped = dict(snapshot)
        scoped["player_stats"] = player_rows.copy()
        scoped["team_season_stats"] = team_rows.copy()
        scoped["_as_of_season"] = player_season
        scoped["_prediction_league_id"] = _to_int(target_league)
        scoped["_prediction_competition"] = target_competition
        scoped["_prediction_team_id"] = _to_int(teams[0])
        scoped["_competition_support"] = {
            "mode": "historical_counterfactual",
            "competition": target_competition,
            "source_season": player_season,
            "target_season": target_season,
            "source_player_seasons": int(len(player_rows)),
        }

        player_dict = dict(scoped.get("player_dict", {}))
        player_meta = dict(player_dict.get(_to_int(players[0]), {}))
        birth_date = player_meta.get("birth_date", player_meta.get("date_of_birth"))
        try:
            age = target_season - int(str(birth_date)[:4])
        except (TypeError, ValueError):
            try:
                age = int(player_meta.get("age")) + (target_season - player_season)
            except (TypeError, ValueError):
                age = int(player_meta.get("age", 25))
        player_meta["age"] = int(age)
        player_meta["current_team_id"] = _to_int(teams[0])
        player_meta["current_league_id"] = _to_int(target_league)
        player_dict[_to_int(players[0])] = player_meta
        scoped["player_dict"] = player_dict

        team_dict = dict(scoped.get("team_dict", {}))
        target_team_meta = dict(team_dict.get(_to_int(teams[0]), self.data.get("team_dict", {}).get(_to_int(teams[0]), {})))
        target_team_meta["league_id"] = _to_int(target_league)
        team_dict[_to_int(teams[0])] = target_team_meta
        scoped["team_dict"] = team_dict

        result = StrictWhatIfEngine(self.ensemble, scoped).predict_in_team(
            players[0], teams[0], target_season, competition=target_competition
        )
        player_latest = self._latest(player_rows)
        observed_rating = _finite(player_latest.get("rating"), result.base_rating)
        rating_delta = float(result.predicted_rating) - observed_rating
        team_latest = self._latest(team_rows)

        current_team_id = _to_int(player_latest.get("team_id"))
        current_team_rows = self._team_rows(current_team_id, source_league, source_competition, player_season) if current_team_id is not None else pd.DataFrame()
        current_team = self._latest(current_team_rows) if not current_team_rows.empty else None
        environment_delta = {}
        for metric in _TEAM_METRICS:
            target_value = _finite(team_latest.get(metric), math.nan)
            current_value = _finite(current_team.get(metric), math.nan) if current_team is not None else math.nan
            if math.isfinite(target_value) and math.isfinite(current_value):
                environment_delta[metric] = round(target_value - current_value, 4)

        drivers = dict(getattr(result, "shap_values", {}) or {})
        ranked_drivers = [{"feature": key, "impact": round(float(value), 4)} for key, value in sorted(drivers.items(), key=lambda item: abs(_finite(item[1])), reverse=True) if math.isfinite(_finite(value))][:8]
        compatibility = float(result.compatibility_factor)
        fit_label = "strong_fit" if compatibility >= 1.0 else "good_fit" if compatibility >= 0.9 else "mixed_fit" if compatibility >= 0.8 else "weak_fit"
        outcome_label = "meaningful_positive_change" if rating_delta >= 0.5 else "meaningful_negative_change" if rating_delta <= -0.5 else "limited_change"

        what_if = {
            "baseline": {"profile_season": player_season, "current_team": self._team_name(current_team_id) if current_team_id is not None else None, "current_rating": round(observed_rating, 3), "current_minutes_per_game": round(_finite(player_latest.get("minutes_per_game")), 3), "current_usage_pct": round(_ratio(player_latest.get("usg_pct")), 4), "competition_tier": competition_tier(self._league_key(source_league)), "competition_tier_label": competition_tier_label(self._league_key(source_league))},
            "target": {"team": self._team_name(teams[0]), "scenario_season": target_season, "context_as_of_season": player_season, "competition_tier": competition_tier(self._league_key(target_league)), "competition_tier_label": competition_tier_label(self._league_key(target_league))},
            "changes": {"predicted_rating_delta": round(rating_delta, 3), "team_environment_delta": environment_delta, "compatibility": round(compatibility, 4), "compatibility_label": fit_label, "league_factor": round(float(result.league_factor), 4), "context_adjustment": round(float(result.context_adjustment), 4)},
            "outcome": {"label": outcome_label, "predicted_rating": round(float(result.predicted_rating), 3), "confidence_low": round(float(result.confidence_low), 3), "confidence_high": round(float(result.confidence_high), 3)},
            "drivers": ranked_drivers,
            "interpretation": f"Scenario costruito dal profilo osservato fino al {player_season} e proiettato alla stagione {target_season}. Il contesto target usa solo informazioni disponibili al cutoff, evitando leakage della stagione target."
        }
        return {
            "result": {"player": self._player_name(players[0]), "team": self._team_name(teams[0]), "league": self._league_name(target_league), "competition": result.competition, "competition_tier": competition_tier(self._league_key(target_league)), "competition_tier_label": competition_tier_label(self._league_key(target_league)), "source_season": player_season, "target_season": target_season, "predicted_rating": result.predicted_rating, "confidence_low": result.confidence_low, "confidence_high": result.confidence_high, "base_rating": result.base_rating, "compatibility": result.compatibility_factor, "league_factor": result.league_factor, "context_adjustment": result.context_adjustment, "drivers": result.shap_values, "what_if": what_if},
            "evidence": [{"type": "historical_counterfactual", "player_profile_through": player_season, "scenario_season": target_season}, {"type": "target_team_context", "team": self._team_name(teams[0]), "context_through": player_season}],
            "support": {"method": "strict_supervised_historical_counterfactual", **dict(scoped.get("_competition_support", {}))},
            "limitations": ["Il contesto della squadra target è valutato con le informazioni disponibili al cutoff del profilo, non con i risultati reali della stagione target."]
        }

    def _league_transfer(self, spec, players, teams, source_league, target_league):
        if not players:
            raise ValueError("player is required")
        if source_league is None or target_league is None:
            raise ValueError("source_league and target_league are required")
        parameters = spec.get("parameters", {}) or {}
        player_season = int(parameters.get("player_season", spec["season"]))
        target_season = int(parameters.get("target_season", player_season + 1))
        if target_season <= player_season:
            raise ValueError("target_season must be later than player_season")
        source_comp = normalize_competition(spec["competition"])
        target_comp = normalize_competition(spec.get("target_competition") or source_comp)
        source_rows = self._player_rows(players[0], source_league, source_comp, player_season)
        source = self._latest(source_rows)
        source_rating = _finite(source.get("rating"), 6.5)
        samples = self._transfer_samples(source_league, target_league, source_comp, target_comp, player_season)
        fallback_delta = self._league_quality_fallback(source_league, target_league)
        if samples:
            empirical = float(np.median(np.asarray(samples, dtype=float)))
            weight = len(samples) / (len(samples) + 8.0)
            transfer_delta = empirical * weight + fallback_delta * (1.0 - weight)
            method = "empirical_cross_league_transition"
        else:
            empirical = None
            transfer_delta = fallback_delta
            method = "league_quality_fallback"
        context_score, context = (0.5, {})
        if teams:
            context_score, context = self._target_context_score(
                players[0], teams[0], target_league, target_comp, player_season
            )
        fit_delta = float(np.clip((context_score - 0.65) * 0.70, -0.30, 0.25)) if teams else 0.0
        projected = float(np.clip(source_rating + transfer_delta + fit_delta, 0.0, 10.0))
        residual_scale = float(np.std(samples, ddof=1)) if len(samples) >= 2 else 0.75
        half_width = max(0.45, residual_scale * (1.35 if len(samples) < 10 else 1.0))
        limitations = []
        if len(samples) < 5:
            limitations.append("Cross-league support is sparse; the estimate relies materially on league-quality priors.")
        limitations.append(
            f"Il contesto target è limitato alle informazioni disponibili fino al {player_season}; "
            f"i dati reali del {target_season} non entrano nella previsione."
        )
        what_if = {
            "baseline": {
                "profile_season": player_season,
                "current_rating": round(source_rating, 3),
                "competition_tier": competition_tier(self._league_key(source_league)),
                "competition_tier_label": competition_tier_label(self._league_key(source_league)),
            },
            "target": {
                "team": self._team_name(teams[0]) if teams else None,
                "scenario_season": target_season,
                "context_as_of_season": player_season,
                "competition_tier": competition_tier(self._league_key(target_league)),
                "competition_tier_label": competition_tier_label(self._league_key(target_league)),
            },
            "changes": {
                "predicted_rating_delta": round(projected - source_rating, 3),
                "league_transfer_delta": round(transfer_delta, 3),
                "league_prior_delta": round(fallback_delta, 3),
                "team_fit_delta": round(fit_delta, 3),
                "target_context": context,
            },
            "outcome": {
                "predicted_rating": round(projected, 3),
                "confidence_low": round(max(0.0, projected - half_width), 3),
                "confidence_high": round(min(10.0, projected + half_width), 3),
            },
            "interpretation": (
                f"Scenario costruito dal profilo di {player_season} e proiettato al {target_season}, "
                "senza usare i dati osservati della stagione target."
            ),
        }
        return {
            "result": {
                "player": self._player_name(players[0]),
                "source_league": self._league_name(source_league),
                "target_league": self._league_name(target_league),
                "source_competition": source_comp,
                "target_competition": target_comp,
                "source_competition_tier": competition_tier(self._league_key(source_league)),
                "source_competition_tier_label": competition_tier_label(self._league_key(source_league)),
                "target_competition_tier": competition_tier(self._league_key(target_league)),
                "target_competition_tier_label": competition_tier_label(self._league_key(target_league)),
                "source_season": int(source["_year"]),
                "target_season": target_season,
                "source_rating": round(source_rating, 3),
                "projected_rating": round(projected, 3),
                "confidence_low": round(max(0.0, projected - half_width), 3),
                "confidence_high": round(min(10.0, projected + half_width), 3),
                "empirical_transfer_delta": round(empirical, 3) if empirical is not None else None,
                "league_prior_delta": round(fallback_delta, 3),
                "team_fit_delta": round(fit_delta, 3),
                "team": self._team_name(teams[0]) if teams else None,
                "context": context,
                "what_if": what_if,
            },
            "evidence": [
                {"type": "historical_cross_league_transitions", "count": len(samples), "source_season": player_season, "target_season": target_season}
            ],
            "support": {"method": method, "samples": len(samples), "confidence": _confidence(len(samples))},
            "limitations": limitations,
        }

    def _playoff_role(self, spec, players, teams, source_league, target_league):
        del target_league
        if not players:
            raise ValueError("player is required")
        season = int(spec["season"])
        po = self._player_rows(players[0], source_league, "PO", season)
        rs = self._player_rows(players[0], source_league, "RS", season)
        po_latest = self._latest(po)
        rs_latest = rs.iloc[-1] if not rs.empty else None
        rating_delta = _finite(po_latest.get("rating")) - (_finite(rs_latest.get("rating")) if rs_latest is not None else _finite(po_latest.get("rating")))
        ts_delta = _ratio(po_latest.get("ts_pct")) - (_ratio(rs_latest.get("ts_pct")) if rs_latest is not None else _ratio(po_latest.get("ts_pct")))
        usage = _ratio(po_latest.get("usg_pct"), 0.18)
        resilience = float(np.clip(0.50 + rating_delta * 0.16 + ts_delta * 1.4, 0.0, 1.0))
        role_label = str(po_latest.get("ruolo_combinato", "") or "")
        if teams:
            team_po = self._player_team(
                {**spec, "competition": "PO", "target_competition": "PO"},
                players,
                teams,
                source_league,
                source_league,
            )
            projected = team_po["result"].get("predicted_rating")
        else:
            projected = None
        if usage >= 0.27 and resilience >= 0.58:
            role_signal = "PRIMARY_OPTION"
        elif usage >= 0.22 and resilience >= 0.48:
            role_signal = "SECONDARY_OPTION"
        elif usage >= 0.16:
            role_signal = "ROTATION_ROLE"
        else:
            role_signal = "LOW_USAGE_ROLE"
        games = int(pd.to_numeric(po.get("games_played", 0), errors="coerce").fillna(0).sum())
        return {
            "result": {
                "player": self._player_name(players[0]),
                "competition": "PO",
                "historical_role": role_label or None,
                "role_signal": role_signal,
                "resilience_score": round(resilience, 3),
                "playoff_rating": round(_finite(po_latest.get("rating")), 3),
                "rs_to_po_rating_delta": round(rating_delta, 3),
                "rs_to_po_ts_delta": round(ts_delta, 4),
                "playoff_usage": round(usage, 4),
                "projected_team_rating": projected,
                "team": self._team_name(teams[0]) if teams else None,
            },
            "evidence": [{"type": "playoff_games", "count": games}, {"type": "playoff_seasons", "count": len(po)}],
            "support": {"method": "observed_playoff_resilience_plus_role_rules", "samples": games, "confidence": _confidence(games)},
            "limitations": ["Role signal is an evidence-based classification, not a calibrated probability."] if games < 25 else [],
        }

    def _player_vector(self, player_id: int, league_id: int | None, competition: str, season: int) -> tuple[np.ndarray, dict[str, Any]]:
        row = self._latest(self._player_rows(player_id, league_id, competition, season))
        vector = np.asarray([
            _ratio(row.get("usg_pct"), 0.18),
            _ratio(row.get("ts_pct"), 0.52),
            _finite(row.get("assists"), 3.0) / 10.0,
            _ratio(row.get("three_point_pct"), 0.33),
            _finite(row.get("dbpm"), 0.0) / 6.0,
            _finite(row.get("bpm"), 0.0) / 10.0,
        ], dtype=float)
        return vector, row.to_dict()

    def _pair_score(self, first_id: int, second_id: int, league_id: int | None, competition: str, season: int) -> dict[str, Any]:
        a, ar = self._player_vector(first_id, league_id, competition, season)
        b, br = self._player_vector(second_id, league_id, competition, season)
        usage_overlap = max(0.0, a[0] + b[0] - 0.50)
        spacing = float(np.clip((a[3] + b[3]) / 0.80, 0.0, 1.0))
        creation = float(np.clip((a[2] + b[2]) / 1.2, 0.0, 1.0))
        defense = float(np.clip(0.5 + (a[4] + b[4]) / 2.0, 0.0, 1.0))
        positions = [
            str(self.data.get("player_dict", {}).get(_to_int(pid), {}).get("position", ""))
            for pid in (first_id, second_id)
        ]
        overlap = 1.0 if positions[0] and positions[0].split("/")[0] == positions[1].split("/")[0] else 0.0
        score = float(np.clip(0.50 + spacing * 0.18 + creation * 0.14 + defense * 0.12 - usage_overlap * 0.42 - overlap * 0.10, 0.0, 1.0))
        return {
            "score": round(score, 3),
            "usage_overlap": round(usage_overlap, 3),
            "spacing": round(spacing, 3),
            "creation": round(creation, 3),
            "defensive_complementarity": round(defense, 3),
            "positional_overlap": bool(overlap),
            "first_usage": round(a[0], 3),
            "second_usage": round(b[0], 3),
            "first_rating": round(_finite(ar.get("rating")), 3),
            "second_rating": round(_finite(br.get("rating")), 3),
        }

    def _player_pair(self, spec, players, teams, source_league, target_league):
        del target_league
        if len(players) < 2:
            raise ValueError("two players are required")
        competition, season = normalize_competition(spec["competition"]), int(spec["season"])
        pair = self._pair_score(players[0], players[1], source_league, competition, season)
        baseline = None
        adjusted = None
        if teams:
            direct = self._player_team(spec, [players[0]], teams, source_league, source_league)
            baseline = direct["result"].get("predicted_rating")
            if baseline is not None:
                adjusted = round(float(np.clip(float(baseline) + (pair["score"] - 0.5) * 0.45, 0.0, 10.0)), 3)
        return {
            "result": {
                "players": [self._player_name(players[0]), self._player_name(players[1])],
                "team": self._team_name(teams[0]) if teams else None,
                "pair_fit": pair,
                "baseline_rating": baseline,
                "rating_with_pair_context": adjusted,
            },
            "evidence": [],
            "support": {"method": "analytical_pair_complementarity", "samples": 2, "confidence": "medium"},
            "limitations": ["Pair fit is analytical unless lineup/on-off evidence for the exact pair exists in the source data."],
        }

    def _lineup_fit(self, spec, players, teams, source_league, target_league):
        del target_league
        if len(players) < 2:
            raise ValueError("at least two players are required")
        competition, season = normalize_competition(spec["competition"]), int(spec["season"])
        pairs = []
        for index, first in enumerate(players[:5]):
            for second in players[index + 1 : 5]:
                score = self._pair_score(first, second, source_league, competition, season)
                pairs.append({"players": [self._player_name(first), self._player_name(second)], **score})
        overall = float(np.mean([row["score"] for row in pairs])) if pairs else 0.5
        return {
            "result": {
                "players": [self._player_name(pid) for pid in players[:5]],
                "team": self._team_name(teams[0]) if teams else None,
                "overall_fit": round(overall, 3),
                "pairwise": pairs,
            },
            "evidence": [{"type": "pairwise_combinations", "count": len(pairs)}],
            "support": {"method": "analytical_lineup_complementarity", "samples": len(pairs), "confidence": "medium"},
            "limitations": ["This is a fit estimate, not a possession-level lineup Net Rating forecast."],
        }

    def _distribution_step(self, column: str, league_id: int, competition: str, season: int) -> float:
        frame = self.data.get("team_season_stats", pd.DataFrame()).copy()
        if frame.empty or column not in frame.columns:
            return 1.0
        frame["_year"] = frame["season"].map(_year)
        rows = frame[
            (frame["league_id"].map(_to_int) == _to_int(league_id))
            & (frame["competition"].map(normalize_competition) == competition)
            & (frame["_year"] == season)
        ]
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        step = float(values.std(ddof=0)) if len(values) >= 2 else 0.0
        if step > 0:
            return step
        defaults = {"pace": 4.0, "three_point_attempt_rate": 0.05, "assists_per_game": 3.0, "star_player_usage": 0.04, "offensive_rating": 4.0, "defensive_rating": 4.0}
        return defaults.get(column, 1.0)

    def _apply_style_overrides(self, scoped: dict[str, Any], team_id: int, league_id: int, competition: str, season: int, overrides: dict[str, Any]) -> dict[str, Any]:
        modified = dict(scoped)
        history = scoped["team_season_stats"].copy()
        team_dict = dict(scoped["team_dict"])
        applied: dict[str, float] = {}
        team_mask = history["team_id"].map(_to_int) == _to_int(team_id)
        if not team_mask.any():
            raise ValueError("Target team context is unavailable")
        index = history[team_mask].index[-1]
        team = dict(team_dict[_to_int(team_id)])
        for key, raw in overrides.items():
            current = _finite(history.at[index, key] if key in history.columns else team.get(key))
            if isinstance(raw, str):
                direction = raw.lower()
                step = self._distribution_step(key, league_id, competition, season)
                target = current + step if direction == "higher" else current - step if direction == "lower" else current
            else:
                target = float(raw)
            low, high = _STYLE_BOUNDS[key]
            target = float(np.clip(target, low, high))
            if key not in history.columns:
                history[key] = np.nan
            history.at[index, key] = target
            team[key] = target
            applied[key] = round(target, 4)
        modified["team_season_stats"] = history
        team_dict[_to_int(team_id)] = team
        modified["team_dict"] = team_dict
        modified["_historical_style_map"] = None
        return {"data": modified, "applied": applied}

    def _style_change(self, spec, players, teams, source_league, target_league):
        if not players or not teams:
            raise ValueError("player and team are required")
        target_league = target_league or source_league
        if target_league is None:
            raise ValueError("target league is required")
        if source_league is not None and source_league != target_league:
            transfer = self._league_transfer(spec, players, teams, source_league, target_league)
            transfer["limitations"].append("Cross-league style changes are reported through contextual fit; exact synthetic-team XGBoost inference requires target-league player history.")
            return transfer
        competition = normalize_competition(spec.get("target_competition") or spec["competition"])
        season = int(spec["season"])
        baseline, scoped = self._direct_prediction(players[0], teams[0], target_league, season, competition)
        patched = self._apply_style_overrides(scoped, teams[0], target_league, competition, season, spec.get("style_overrides", {}))
        self.ensemble.clear_cache()
        engine = StrictWhatIfEngine(self.ensemble, patched["data"])
        changed = engine.predict_in_team(players[0], teams[0], season, competition=competition)
        self.ensemble.clear_cache()
        return {
            "result": {
                "player": self._player_name(players[0]),
                "team": self._team_name(teams[0]),
                "competition": competition,
                "applied_style": patched["applied"],
                "baseline_rating": baseline.predicted_rating,
                "scenario_rating": changed.predicted_rating,
                "rating_delta": round(changed.predicted_rating - baseline.predicted_rating, 3),
                "confidence_low": changed.confidence_low,
                "confidence_high": changed.confidence_high,
            },
            "evidence": [],
            "support": {"method": "strict_ensemble_counterfactual_team_context", "samples": 1, "confidence": "medium"},
            "limitations": ["Directional style changes use one within-context standard-deviation step and report the applied numeric value."],
        }

    def _player_role_change(self, spec, players, teams, source_league, target_league):
        del target_league
        if not players or not teams or source_league is None:
            raise ValueError("player, team and league are required")
        competition, season = normalize_competition(spec["competition"]), int(spec["season"])
        baseline, scoped = self._direct_prediction(players[0], teams[0], source_league, season, competition)
        stats = scoped["player_stats"].copy()
        mask = stats["player_id"].map(_to_int) == _to_int(players[0])
        if not mask.any():
            raise ValueError("Player context is unavailable")
        index = stats[mask].index[-1]
        applied: dict[str, Any] = {}
        for key, value in spec.get("player_overrides", {}).items():
            if key == "role":
                stats.at[index, "ruolo_combinato"] = str(value)
                applied[key] = str(value)
                continue
            numeric = float(value)
            if key == "minutes_per_game":
                numeric = float(np.clip(numeric, 0.0, 48.0))
            elif key == "usg_pct":
                numeric = float(np.clip(numeric, 0.0, 100.0 if numeric > 1.5 else 1.0))
            stats.at[index, key] = numeric
            applied[key] = numeric
        modified = dict(scoped)
        modified["player_stats"] = stats
        self.ensemble.clear_cache()
        changed = StrictWhatIfEngine(self.ensemble, modified).predict_in_team(players[0], teams[0], season, competition=competition)
        self.ensemble.clear_cache()
        return {
            "result": {"player": self._player_name(players[0]), "team": self._team_name(teams[0]), "applied_player_context": applied, "baseline_rating": baseline.predicted_rating, "scenario_rating": changed.predicted_rating, "rating_delta": round(changed.predicted_rating - baseline.predicted_rating, 3)},
            "evidence": [],
            "support": {"method": "strict_ensemble_counterfactual_player_context", "samples": 1, "confidence": "medium"},
            "limitations": ["Role labels unseen during training are encoded as out-of-vocabulary rather than treated as known roles."],
        }


    def _context_percentile(self, frame, metric, value, *, lower_better=False):
        if frame.empty or metric not in frame.columns or pd.isna(value):
            return None
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if len(values) < 5:
            return None
        percentile = float((values <= float(value)).mean())
        return round(1.0 - percentile if lower_better else percentile, 4)

    def _player_rows_frame_context(self, league_id, competition, season):
        frame = self.data.get("player_stats", pd.DataFrame()).copy()
        if frame.empty:
            return frame
        frame["_year"] = frame["season"].map(_year)
        mask = ((frame["competition"].map(normalize_competition) == normalize_competition(competition)) & (frame["_year"] == int(season)))
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        return frame[mask].copy()

    @staticmethod
    def _percentile_label(percentile):
        if percentile is None:
            return "insufficient context"
        if percentile >= .90:
            return "elite relative to context"
        if percentile >= .75:
            return "very strong relative to context"
        if percentile >= .60:
            return "above average relative to context"
        if percentile >= .40:
            return "around average relative to context"
        if percentile >= .25:
            return "below average relative to context"
        return "low relative to context"

    def _metric_explanation(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        season = int(spec["season"])
        competition = normalize_competition(spec["competition"])
        rows = self._player_rows(players[0], source_league, competition, season)
        current = self._latest(rows)
        requested = spec.get("parameters", {}).get("metrics")
        metrics = requested if isinstance(requested, list) else ["rating", "ts_pct", "usg_pct", "ast_pct", "tov_pct", "three_point_pct", "bpm", "dbpm"]
        meanings = {
            "rating": ("Overall player rating", False),
            "ts_pct": ("True shooting efficiency", False),
            "usg_pct": ("Share of team possessions used", False),
            "ast_pct": ("Share of teammate field goals assisted", False),
            "tov_pct": ("Turnover rate on possessions used", True),
            "three_point_pct": ("Three-point shooting accuracy", False),
            "bpm": ("Box-score impact estimate", False),
            "dbpm": ("Box-score defensive impact estimate", False),
        }
        context = self._player_rows_frame_context(source_league, competition, season)
        explanations = []
        for metric in metrics[:12]:
            if metric not in meanings or metric not in current.index or pd.isna(current.get(metric)):
                continue
            meaning, lower_better = meanings[metric]
            value = _finite(current.get(metric))
            pct = self._context_percentile(context, metric, value, lower_better=lower_better)
            explanations.append({"metric": metric, "meaning": meaning, "value": round(value, 4), "context_percentile": pct, "interpretation": self._percentile_label(pct)})
        return {"result": {"player": self._player_name(players[0]), "league": self._league_name(source_league), "season": season, "competition": competition, "metrics": explanations},
                "evidence": [{"type": "same_league_season_competition_pool", "count": len(context)}],
                "support": {"method": "contextual_metric_explanation", "samples": len(context), "confidence": _confidence(len(context))},
                "limitations": ["Percentiles describe the observed context; they are not causal effects."]}

    def _role_analysis(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        rows = self._player_rows(players[0], source_league, spec["competition"], int(spec["season"]))
        if rows.empty:
            raise ValueError("No player history is available")
        history = []
        for _, row in rows.iterrows():
            usage, ast, minutes = _ratio(row.get("usg_pct")), _ratio(row.get("ast_pct")), _finite(row.get("minutes_per_game"))
            if usage >= .27: role = "primary_creator"
            elif usage >= .22 and ast >= .18: role = "secondary_creator"
            elif usage >= .20: role = "scoring_role"
            elif ast >= .18: role = "connector"
            elif minutes >= 20: role = "rotation"
            else: role = "depth"
            history.append({"season": int(row["_year"]), "role": role, "usage": round(usage,4), "ast_pct": round(ast,4), "minutes_per_game": round(minutes,2), "three_par": round(_ratio(row.get("three_par")),4)})
        changes = [{"from_season": a["season"], "to_season": b["season"], "from": a["role"], "to": b["role"]} for a,b in zip(history,history[1:]) if a["role"] != b["role"]]
        return {"result": {"player": self._player_name(players[0]), "season": int(spec["season"]), "competition": normalize_competition(spec["competition"]), "current_role": history[-1]["role"], "current_profile": history[-1], "history": history, "role_changes": changes},
                "evidence": [{"type": "historical_role_profile", "count": len(history)}],
                "support": {"method": "usage_creation_minutes_role_classification", "samples": len(history)*6, "confidence": _confidence(len(history)*6)},
                "limitations": ["Role labels are analytical classifications, not causal assignments."]}

    def _performance_stability(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players:
            raise ValueError("player is required")
        rows = self._player_rows(players[0], source_league, spec["competition"], int(spec["season"]))
        if len(rows) < 2:
            raise ValueError("At least two observed seasons are required")
        valid = rows[["rating","_year"]].copy()
        valid["rating"] = pd.to_numeric(valid["rating"], errors="coerce")
        valid = valid.dropna()
        ratings, years = valid["rating"].to_numpy(float), valid["_year"].to_numpy(float)
        slope = float(np.polyfit(years, ratings, 1)[0]) if len(ratings) > 1 and len(set(years)) > 1 else 0.0
        std, median, peak, recent = float(np.std(ratings)), float(np.median(ratings)), float(np.max(ratings)), float(ratings[-1])
        stability = float(np.clip(np.exp(-std/1.25) * min(1.0, len(ratings)/4.0), 0.0, 1.0))
        return {"result": {"player": self._player_name(players[0]), "seasons": len(ratings), "rating_mean": round(float(np.mean(ratings)),3), "rating_median": round(median,3), "rating_std": round(std,3), "peak_rating": round(peak,3), "recent_rating": round(recent,3), "trend_slope_per_season": round(slope,4), "peak_premium": round(peak-median,3), "stability_score": round(stability,3)},
                "evidence": [{"type":"rating_history","count":len(ratings)}],
                "support": {"method":"historical_dispersion_and_trend","samples":len(ratings),"confidence":_confidence(len(ratings)*6)},
                "limitations":["The stability score is a descriptive index, not a probability of future consistency."]}

    def _team_usage_analysis(self, spec, players, teams, source_league, target_league):
        del target_league
        if not players or not teams:
            raise ValueError("player and team are required")
        season, competition = int(spec["season"]), normalize_competition(spec["competition"])
        player, team = self._latest(self._player_rows(players[0], source_league, competition, season)), self._latest(self._team_rows(teams[0], source_league, competition, season))
        usage, star = _ratio(player.get("usg_pct")), _ratio(team.get("star_player_usage"))
        three, team_three = _ratio(player.get("three_par")), _ratio(team.get("three_point_attempt_rate"))
        ast, team_ast = _ratio(player.get("ast_pct")), _finite(team.get("assists_per_game"))
        signals=[]
        if usage-star > .04: signals.append("player carries more usage than the team star-usage reference")
        elif usage-star < -.04: signals.append("player carries materially less usage than the team star-usage reference")
        if three > team_three + .08: signals.append("player is more perimeter-oriented than the team baseline")
        if ast >= .18 and team_ast >= 22: signals.append("creation profile aligns with a high-assist team environment")
        if not signals: signals.append("observed role is broadly aligned with the team context")
        return {"result":{"player":self._player_name(players[0]),"team":self._team_name(teams[0]),"season":season,"competition":competition,"player_usage":round(usage,4),"team_star_usage":round(star,4),"usage_gap_vs_team_reference":round(usage-star,4),"player_three_point_rate":round(three,4),"team_three_point_rate":round(team_three,4),"player_ast_pct":round(ast,4),"team_assists_per_game":round(team_ast,2),"team_pace":round(_finite(team.get("pace")),2),"signals":signals},
                "evidence":[{"type":"player_team_context","count":1}],"support":{"method":"observed_role_vs_team_style","samples":1,"confidence":"medium"},
                "limitations":["This evaluates contextual alignment; it does not establish that team usage caused performance."]}

    def _regression_risk(self, spec, players, teams, source_league, target_league):
        stability = self._performance_stability(spec, players, teams, source_league, target_league)
        r = stability["result"]; deviation = r["recent_rating"]-r["rating_median"]; peak_gap = r["recent_rating"]-r["peak_rating"]
        risk = float(np.clip(.40*max(0.0,deviation)/2.0 + .35*(1.0-r["stability_score"]) + .25*max(0.0,-peak_gap)/1.5,0,1))
        level = "high" if risk >= .65 else "moderate" if risk >= .35 else "low"
        drivers=[x for x in ["recent rating above historical baseline" if deviation>.5 else None, "historical dispersion" if r["rating_std"]>.75 else None, "recent level below historical peak" if peak_gap<-.5 else None] if x]
        return {"result":{"player":r["player"],"risk_level":level,"risk_index":round(risk,3),"recent_vs_median":round(deviation,3),"recent_vs_peak":round(peak_gap,3),"stability_score":r["stability_score"],"drivers":drivers},
                "evidence":stability["evidence"],"support":{"method":"descriptive_regression_risk_index","samples":stability["support"]["samples"],"confidence":stability["support"]["confidence"]},
                "limitations":["This is not a calibrated probability of regression and does not replace Prediction Model."]}

    def _potential_synthesis(self, spec, players, teams, source_league, target_league):
        del target_league
        if not players:
            raise ValueError("player is required")
        season=int(spec["season"]); rows=self._player_rows(players[0],source_league,spec["competition"],season)
        if rows.empty: raise ValueError("No player history is available")
        latest=self._latest(rows); ratings=pd.to_numeric(rows["rating"],errors="coerce").dropna().to_numpy(float)
        slope=float(np.polyfit(np.arange(len(ratings)),ratings,1)[0]) if len(ratings)>1 else 0.0
        p=self.data.get("player_dict",{}).get(_to_int(players[0]),{}); birth=p.get("birth_date",p.get("date_of_birth"))
        try: age=season-int(str(birth)[:4])
        except (TypeError,ValueError): age=int(_finite(p.get("age"),float("nan"))) if np.isfinite(_finite(p.get("age"),float("nan"))) else None
        peak_age=26 if str(p.get("position","")).upper() in {"PG","SG"} else 28
        age_signal="unknown" if age is None else "development_window" if age<peak_age-2 else "near_peak" if age<=peak_age+2 else "post_peak"
        signal="upside" if age_signal=="development_window" and slope>=0 else "established" if age_signal!="development_window" and abs(slope)<.15 else "mixed"
        return {"result":{"player":self._player_name(players[0]),"season":season,"current_rating":round(_finite(latest.get("rating")),3),"historical_peak":round(float(np.max(ratings)),3),"rating_trend_per_season":round(slope,4),"age":age,"age_signal":age_signal,"potential_signal":signal},
                "evidence":[{"type":"rating_history","count":len(ratings)},{"type":"age_context","available":age is not None}],
                "support":{"method":"rating_trend_plus_age_synthesis","samples":len(ratings),"confidence":_confidence(len(ratings)*6)},
                "limitations":["This synthesis uses observed trajectory and age; it is not a replacement for season-ahead Prediction or Future Performance."]}

    def _shooting_decomposition(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players: raise ValueError("player is required")
        rows=self._player_rows(players[0],source_league,spec["competition"],int(spec["season"]))
        if len(rows)<2: raise ValueError("At least two observed seasons are required")
        current,previous=rows.iloc[-1],rows.iloc[-2]; metrics=("ts_pct","three_point_pct","ft_pct","three_par","three_point_attempts","free_throw_attempts")
        deltas={m:round(_finite(current.get(m))- _finite(previous.get(m)),4) for m in metrics if pd.notna(current.get(m)) and pd.notna(previous.get(m))}
        return {"result":{"player":self._player_name(players[0]),"previous_season":int(previous["_year"]),"current_season":int(current["_year"]),"deltas":deltas},
                "evidence":[{"type":"consecutive_seasons","count":2}],"support":{"method":"observed_shooting_profile_decomposition","samples":int(current.get("games_played",0) or 0),"confidence":_confidence(int(current.get("games_played",0) or 0))},
                "limitations":["Without shot-level data this cannot separate shot quality, location and defensive pressure."]}

    def _defensive_decomposition(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players: raise ValueError("player is required")
        rows=self._player_rows(players[0],source_league,spec["competition"],int(spec["season"]))
        if len(rows)<2: raise ValueError("At least two observed seasons are required")
        current,previous=rows.iloc[-1],rows.iloc[-2]; metrics=("dbpm","raptor_def","lebron_def","drtg","stl_pct","blk_pct","def_rebounds")
        deltas={m:round(_finite(current.get(m))- _finite(previous.get(m)),4) for m in metrics if pd.notna(current.get(m)) and pd.notna(previous.get(m))}
        return {"result":{"player":self._player_name(players[0]),"previous_season":int(previous["_year"]),"current_season":int(current["_year"]),"deltas":deltas},
                "evidence":[{"type":"consecutive_seasons","count":2}],"support":{"method":"observed_defensive_profile_decomposition","samples":int(current.get("games_played",0) or 0),"confidence":_confidence(int(current.get("games_played",0) or 0))},
                "limitations":["Box-score defensive metrics do not isolate individual causal defensive impact."]}


    def _model_evidence(self, spec, players, teams, source_league, target_league):
        """Compose model evidence and make the prediction context explicit.

        Future Performance is a player/league/competition season-ahead model.
        It uses player role, minutes, usage, age, position and recent history,
        but it does not use a target-team counterfactual. The intelligence layer
        therefore exposes that scope instead of presenting a per-36 prediction
        as an unconditional per-game expectation.
        """
        if not players:
            raise ValueError("player is required")
        player = players[0]
        global_id = str(player.get("global_id", ""))
        league = self._league_key(source_league)
        season = _year(spec.get("season"))
        competition = normalize_competition(spec.get("competition", "RS"))
        if not global_id or not league:
            raise ValueError("player model context is incomplete")

        result = {
            "bb_rating": {"available": False},
            "bb_rating_uncertainty": {"available": False},
            "future_performance": {"available": False},
            "prediction_model": {"available": False},
        }

        if self.bb_rating_engine is not None:
            try:
                rated = self.bb_rating_engine.rate_player(
                    global_id, league=league, season=season, phase=competition
                )
                result["bb_rating"] = {
                    "available": True,
                    "score": int(rated.score),
                    "quality": getattr(rated, "quality", None),
                    "bb_rating_version": getattr(rated, "bb_rating_version", None),
                }
                if self.bb_rating_uncertainty is not None:
                    result["bb_rating_uncertainty"] = self.bb_rating_uncertainty.for_player(
                        self.bb_rating_engine.frame,
                        player_global_id=global_id,
                        league=league,
                        season=season,
                        phase=competition,
                        score=rated.score,
                    )
            except (ValueError, RuntimeError, KeyError) as exc:
                result["bb_rating"]["reason"] = str(exc)
                result["bb_rating_uncertainty"]["reason"] = "bb_rating_unavailable"

        if self.future_performance_model is not None:
            try:
                future = self.future_performance_model.predict_player(
                    self.data,
                    player_id=int(player["id"]),
                    league_key=league,
                    season=season,
                    competition=competition,
                )
                future = dict(future)
                future["available"] = True

                # Explicitly expose the context in which the trained model made
                # the prediction. These are model inputs/derived interpretation,
                # not a new prediction model and not a team counterfactual.
                rows = self._player_rows(
                    int(player["id"]), source_league, competition, season
                )
                latest = self._latest(rows)
                current_minutes = _finite(latest.get("minutes_per_game"), 0.0)
                current_usage = _ratio(latest.get("usg_pct"), 0.0)
                current_rating = _finite(latest.get("rating"), 0.0)
                prior_rating = (
                    _finite(rows.iloc[-2].get("rating"), current_rating)
                    if len(rows) >= 2 else current_rating
                )
                position = str(
                    self.data.get("player_dict", {})
                    .get(_to_int(player["id"]), {})
                    .get("position", "")
                    or ""
                )

                targets = future.get("targets", {})
                projected_minutes = targets.get("minutes_per_game", {}).get("value")
                per_game_projection = {}
                for key, label in (
                    ("pts_per_36", "points"),
                    ("ast_per_36", "assists"),
                    ("reb_per_36", "rebounds"),
                    ("stl_per_36", "steals"),
                    ("blk_per_36", "blocks"),
                ):
                    item = targets.get(key)
                    if isinstance(item, dict) and item.get("value") is not None:
                        per_game = item.get("derived_per_game")
                        if per_game is not None:
                            per_game_projection[label] = round(float(per_game), 2)

                current_team_id = _to_int(latest.get("team_id"))
                future["prediction_context"] = {
                    "source_league": league,
                    "source_competition": competition,
                    "source_season": season,
                    "target_season": int(future.get("target_season", season + 1)),
                    "competition_tier": competition_tier(league),
                    "competition_tier_label": competition_tier_label(league),
                    "position": position or None,
                    "current_minutes_per_game": round(current_minutes, 2),
                    "predicted_minutes_per_game": (
                        round(float(projected_minutes), 2)
                        if projected_minutes is not None else None
                    ),
                    "current_usage_pct": round(current_usage * 100.0, 2),
                    "current_rating": round(current_rating, 3),
                    "rating_change_vs_previous_season": round(
                        current_rating - prior_rating, 3
                    ) if len(rows) >= 2 else None,
                    "current_team": (
                        self._team_name(current_team_id)
                        if current_team_id is not None else None
                    ),
                    "team_used_as_prediction_input": False,
                    "interpretation": (
                        "Le previsioni per-36 sono produzione normalizzata. "
                        "La stima per partita deriva dai minuti previsti; "
                        "non rappresenta un valore garantito ogni gara."
                    ),
                }
                future["projected_per_game"] = per_game_projection
                future["prediction_scope"] = (
                    "player_role_league_competition_history"
                )
                future["team_counterfactual"] = {
                    "available": False,
                    "reason": (
                        "Future Performance non modifica la previsione in base "
                        "a una squadra target. Per un cambio squadra va usato "
                        "uno scenario player_team/league_transfer."
                    ),
                }
                result["future_performance"] = future
            except (ValueError, RuntimeError, KeyError) as exc:
                result["future_performance"]["reason"] = str(exc)

        # Prediction Model: observed current-team context, not a future-team counterfactual.
        try:
            rows = self._player_rows(int(player["id"]), source_league, competition, season)
            latest = self._latest(rows)
            current_team_id = _to_int(latest.get("team_id"))
            observed_rating = _finite(latest.get("rating"), np.nan)
            if current_team_id is not None and source_league is not None:
                prediction, _ = self._direct_prediction(int(player["id"]), current_team_id, source_league, season, competition)
                result["prediction_model"] = {"available": True, "scope": "current_team_context", "team": self._team_name(current_team_id), "predicted_rating": round(float(prediction.predicted_rating), 3), "confidence_low": round(float(prediction.confidence_low), 3), "confidence_high": round(float(prediction.confidence_high), 3), "base_rating": round(float(prediction.base_rating), 3), "age_factor": round(float(prediction.age_factor), 4), "compatibility_factor": round(float(prediction.compatibility_factor), 4), "league_factor": round(float(prediction.league_factor), 4), "context_adjustment": round(float(prediction.context_adjustment), 4), "delta_vs_observed": round(float(prediction.predicted_rating) - observed_rating, 3) if math.isfinite(observed_rating) else None, "interpretation": "Stima del livello nel contesto della squadra corrente; non è una previsione di cambio squadra né una proiezione per-36."}
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            result["prediction_model"]["reason"] = str(exc)

        return result

    def _intelligence_context(self, spec, players, source_league):
        """Build descriptive context used by the Player Intelligence synthesis layer."""
        if not players:
            raise ValueError("player is required")
        player_id = int(players[0]["id"])
        season = int(spec["season"])
        competition = normalize_competition(spec.get("competition", "RS"))
        rows = self._player_rows(player_id, source_league, competition, season)
        latest = self._latest(rows)
        games = int(_finite(latest.get("games_played"), 0.0))
        minutes = _finite(latest.get("minutes_per_game"), 0.0)
        usage = _ratio(latest.get("usg_pct"), 0.0)
        ast_pct = _ratio(latest.get("ast_pct"), 0.0)
        ts_pct = _ratio(latest.get("ts_pct"), 0.0)
        rating = _finite(latest.get("rating"), np.nan)
        previous_rating = _finite(rows.iloc[-2].get("rating"), np.nan) if len(rows) >= 2 else np.nan
        rating_delta = round(rating - previous_rating, 3) if math.isfinite(rating) and math.isfinite(previous_rating) else None
        rating_series = pd.to_numeric(rows.get("rating"), errors="coerce").dropna()
        trend = float(np.polyfit(rows.loc[rating_series.index, "_year"].astype(float), rating_series.to_numpy(float), 1)[0]) if len(rating_series) >= 2 else 0.0
        rating_std = float(np.std(rating_series)) if len(rating_series) else None
        stability_score = float(np.clip(np.exp(-rating_std / 1.25) * min(1.0, len(rating_series) / 4.0), 0.0, 1.0)) if rating_std is not None else None
        team_context = {}
        team_id = _to_int(latest.get("team_id"))
        if team_id is not None:
            team_rows = self._team_rows(team_id, source_league, competition, season)
            if not team_rows.empty:
                team = self._latest(team_rows)
                team_frame = self.data.get("team_season_stats", pd.DataFrame()).copy()
                percentiles = {}
                if not team_frame.empty:
                    team_frame["_year"] = team_frame["season"].map(_year)
                    team_frame = team_frame[(team_frame["league_id"].map(_to_int) == _to_int(source_league)) & (team_frame["competition"].map(normalize_competition) == competition) & (team_frame["_year"] == season)]
                    for metric, lower_better in (("net_rtg", False), ("offensive_rating", False), ("defensive_rating", True), ("pace", False)):
                        if metric in team_frame.columns and pd.notna(team.get(metric)):
                            percentiles[metric] = self._context_percentile(team_frame, metric, team.get(metric), lower_better=lower_better)
                team_context = {"team": self._team_name(team_id), "net_rtg": round(_finite(team.get("net_rtg")), 2), "offensive_rating": round(_finite(team.get("offensive_rating")), 2), "defensive_rating": round(_finite(team.get("defensive_rating")), 2), "pace": round(_finite(team.get("pace")), 2), "team_star_usage": round(_ratio(team.get("star_player_usage")), 4), "three_point_attempt_rate": round(_ratio(team.get("three_point_attempt_rate")), 4), "assists_per_game": round(_finite(team.get("assists_per_game")), 2), "league_percentiles": percentiles}
        player_dict = self.data.get("player_dict", {}).get(player_id, {})
        role_label = latest.get("ruolo_combinato") or latest.get("role") or latest.get("role_label")
        position = str(player_dict.get("position") or latest.get("position") or "")
        return {
            "league": self._league_key(source_league), "competition": competition, "season": season,
            "competition_tier": competition_tier(self._league_key(source_league)),
            "competition_tier_label": competition_tier_label(self._league_key(source_league)),
            "role": {"position": position or None, "role_label": str(role_label) if role_label else None, "minutes_per_game": round(minutes, 2), "usage_pct": round(usage * 100.0, 2), "assist_pct": round(ast_pct * 100.0, 2), "offensive_responsibility": "high" if usage >= 0.24 or (usage >= 0.21 and ast_pct >= 0.18) else "medium" if usage >= 0.17 or ast_pct >= 0.14 else "low", "creation_signal": "primary_or_secondary_creator" if ast_pct >= 0.18 else "secondary_creator" if ast_pct >= 0.14 else "finisher_or_off_ball"},
            "opportunity": {"minutes_per_game": round(minutes, 2), "usage_pct": round(usage * 100.0, 2), "team_pace": team_context.get("pace"), "estimated_player_possession_share": round(usage, 4), "interpretation": "USG% è un proxy della quota di possessi offensivi, non un conteggio PBP dei possessi individuali."},
            "efficiency_vs_volume": {"ts_pct": round(ts_pct * 100.0, 2), "usage_pct": round(usage * 100.0, 2), "points_per_game": round(_finite(latest.get("points")), 2), "three_point_pct": round(_ratio(latest.get("three_point_pct")) * 100.0, 2), "profile": "efficiency_favored" if ts_pct >= 0.60 and usage < 0.22 else "volume_favored" if usage >= 0.24 else "balanced"},
            "trend": {"current_rating": round(rating, 3) if math.isfinite(rating) else None, "change_vs_previous_season": rating_delta, "trend_slope_per_season": round(trend, 4)},
            "stability": {"observed_seasons": int(len(rating_series)), "games_current_season": games, "rating_std": round(rating_std, 3) if rating_std is not None else None, "stability_score": round(stability_score, 3) if stability_score is not None else None, "confidence": _confidence(games), "interpretation": "limited_current_sample" if games < 10 else "stable_profile" if stability_score is not None and stability_score >= 0.65 else "variable_profile"},
            "team_environment": team_context,
            "evidence_strength": {"current_games": games, "historical_rating_seasons": int(len(rating_series)), "confidence": _confidence(max(games, len(rating_series) * 6))},
        }

    def _player_intelligence_context_leagues(self, spec, player_id: int, season: int, competition: str, source_league: int | None) -> list[int]:
        """Return concrete league contexts for one canonical player identity."""
        if source_league is not None:
            return [_to_int(source_league)]
        frame = self.data.get("player_stats", pd.DataFrame()).copy()
        if frame.empty or "player_id" not in frame.columns or "league_id" not in frame.columns:
            return []
        frame["_year"] = frame["season"].map(_year)
        mask = (
            (frame["player_id"].map(_to_int) == _to_int(player_id))
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (frame["_year"] == int(season))
        )
        values = pd.to_numeric(frame.loc[mask, "league_id"], errors="coerce").dropna().astype(int).drop_duplicates().tolist()
        return sorted({_to_int(value) for value in values})

    def _player_intelligence(self, spec, players, teams, source_league, target_league):
        if not players:
            raise ValueError("player is required")
        parameters = spec.get("parameters", {}) or {}
        raw_keys = parameters.get("question_keys")
        if isinstance(raw_keys, (list, tuple)):
            keys = [str(item).strip().lower() for item in raw_keys if str(item).strip()]
        else:
            keys = [str(parameters.get("question_key", "current_level")).strip().lower()]
        keys = list(dict.fromkeys(keys))[:8]

        bundles = {
            "current_level": ("player_competition", "model_evidence"),
            "why_performing": ("performance_decomposition", "metric_explanation"),
            "change_vs_last_season": ("performance_decomposition", "role_analysis"),
            "real_improvement": ("performance_decomposition", "metric_explanation"),
            "current_role": ("role_analysis",),
            "role_fit": ("role_analysis", "performance_decomposition"),
            "team_usage": ("team_usage_analysis",),
            "stability": ("performance_stability",),
            "potential": ("potential_synthesis", "age_trajectory"),
            "regression_risk": ("regression_risk",),
            "team_counterfactual": ("player_team", "compatibility"),
            "shot_profile": ("shooting_decomposition",),
            "defensive_reason": ("defensive_decomposition",),
            "clutch_value": ("clutch_analysis",),
            "causal_team_effect": ("causal_team_effect",),
        }
        selected_bundles = []
        for key in keys:
            scenarios = bundles.get(key)
            if scenarios is None:
                raise ValueError(f"unsupported Player Intelligence question_key: {key}")
            selected_bundles.extend(scenarios)
        scenarios = tuple(dict.fromkeys(selected_bundles))

        # When no league was explicitly requested, a canonical player identity
        # may have multiple current league contexts. Player Intelligence owns
        # that context discovery and evaluates the selected evidence per league.
        context_leagues = self._player_intelligence_context_leagues(
            spec,
            int(players[0]),
            int(spec["season"]),
            normalize_competition(spec.get("competition", "RS")),
            source_league,
        )
        if source_league is None and len(context_leagues) == 1:
            source_league = context_leagues[0]
        if source_league is None and len(context_leagues) > 1:
            dispatch_multi = {
                "player_competition": self._player_competition,
                "performance_decomposition": self._performance_decomposition,
                "metric_explanation": self._metric_explanation,
                "role_analysis": self._role_analysis,
                "performance_stability": self._performance_stability,
                "team_usage_analysis": self._team_usage_analysis,
                "regression_risk": self._regression_risk,
                "potential_synthesis": self._potential_synthesis,
                "shooting_decomposition": self._shooting_decomposition,
                "defensive_decomposition": self._defensive_decomposition,
                "age_trajectory": self._age_trajectory,
                "player_team": self._player_team,
                "clutch_analysis": self._clutch_analysis,
                "causal_team_effect": self._causal_team_effect,
            }
            contexts = []
            for league_id in context_leagues:
                context_analyses = {}
                for name in scenarios:
                    if name == "compatibility":
                        continue
                    try:
                        context_analyses[name] = dispatch_multi[name](
                            spec, players, teams, league_id, target_league
                        )
                    except (ValueError, KeyError) as exc:
                        context_analyses[name] = {"available": False, "reason": str(exc)}
                contexts.append({
                    "league": self._league_name(league_id),
                    "league_key": self._league_key(league_id),
                    "season": int(spec["season"]),
                    "analyses": context_analyses,
                })
            return {
                "result": {
                    "player": self._player_name(players[0]),
                    "question_key": keys[0] if len(keys) == 1 else None,
                    "question_keys": keys,
                    "context": None,
                    "contexts": contexts,
                    "answer_mode": "multi_context_synthesis",
                },
                "evidence": [
                    {"type": "question_analysis_registry", "question_keys": keys},
                    {"type": "league_contexts", "count": len(contexts)},
                    {
                        "type": "composed_analysis_layers",
                        "count": sum(len(item["analyses"]) for item in contexts),
                    },
                ],
                "support": {
                    "method": "player_intelligence_multi_context_evidence_composition",
                    "samples": len(contexts),
                    "confidence": "medium",
                },
                "limitations": [
                    "Player Intelligence compone le evidenze separatamente per ogni contesto di lega; non fonde automaticamente metriche appartenenti a leghe diverse."
                ],
            }

        dispatch = {
            "player_competition": self._player_competition,
            "performance_decomposition": self._performance_decomposition,
            "metric_explanation": self._metric_explanation,
            "role_analysis": self._role_analysis,
            "performance_stability": self._performance_stability,
            "team_usage_analysis": self._team_usage_analysis,
            "regression_risk": self._regression_risk,
            "potential_synthesis": self._potential_synthesis,
            "shooting_decomposition": self._shooting_decomposition,
            "defensive_decomposition": self._defensive_decomposition,
            "age_trajectory": self._age_trajectory,
            "player_team": self._player_team,
            "clutch_analysis": self._clutch_analysis,
            "causal_team_effect": self._causal_team_effect,
        }
        analyses = {}
        limitations = [
            "Player Intelligence compone evidenze descrittive e predittive esistenti; non trasforma una correlazione in causalità."
        ]
        for name in scenarios:
            if name == "compatibility":
                continue
            fn = dispatch[name]
            try:
                payload = fn(spec, players, teams, source_league, target_league)
            except (ValueError, KeyError) as exc:
                payload = {"available": False, "reason": str(exc)}
            analyses[name] = payload
        return {
            "result": {
                "player": self._player_name(players[0]),
                "question_key": keys[0] if len(keys) == 1 else None,
                "question_keys": keys,
                "context": self._intelligence_context(spec, players, source_league) if "current_level" in keys else None,
                "analyses": analyses,
                "answer_mode": "contextual_synthesis",
            },
            "evidence": [
                {"type": "question_analysis_registry", "question_keys": keys},
                {"type": "composed_analysis_layers", "count": len(analyses)},
            ],
            "support": {
                "method": "player_intelligence_evidence_composition",
                "samples": len(analyses),
                "confidence": "medium",
            },
            "limitations": limitations,
        }

    # ------------------------------------------------------------------
    # Team impact / replacement / fit searches
    # ------------------------------------------------------------------
    def _rotation_ratings(self, team_id: int, league_id: int, competition: str, season: int) -> list[tuple[int, float]]:
        frame = self.data.get("player_stats", pd.DataFrame()).copy()
        if frame.empty:
            return []
        frame["_year"] = frame["season"].map(_year)
        rows = frame[
            (frame["team_id"].map(_to_int) == _to_int(team_id))
            & (frame["league_id"].map(_to_int) == _to_int(league_id))
            & (frame["competition"].map(normalize_competition) == competition)
            & (frame["_year"] == season)
        ]
        valid = rows.loc[rows["rating"].notna(), ["player_id", "rating"]]
        result = [
            (_to_int(player_id), _finite(rating))
            for player_id, rating in valid.itertuples(index=False, name=None)
        ]
        return sorted(result, key=lambda item: item[1], reverse=True)

    def _team_impact_slope(self, league_id: int, competition: str, through_season: int) -> tuple[float | None, int]:
        teams = self.data.get("team_season_stats", pd.DataFrame()).copy()
        stats = self.data.get("player_stats", pd.DataFrame()).copy()
        if teams.empty or stats.empty:
            return None, 0
        teams["_year"] = teams["season"].map(_year)
        stats["_year"] = stats["season"].map(_year)
        teams = teams[(teams["league_id"].map(_to_int) == _to_int(league_id)) & (teams["competition"].map(normalize_competition) == competition) & (teams["_year"] <= through_season)]
        stats = stats[(stats["league_id"].map(_to_int) == _to_int(league_id)) & (stats["competition"].map(normalize_competition) == competition) & (stats["_year"] <= through_season)]
        samples: list[tuple[float, float]] = []
        for team in teams.to_dict("records"):
            if pd.isna(team.get("net_rtg")):
                continue
            roster = stats[(stats["team_id"].map(_to_int) == _to_int(team["team_id"])) & (stats["_year"] == int(team["_year"]))]
            ratings = sorted(pd.to_numeric(roster.get("rating"), errors="coerce").dropna().tolist(), reverse=True)[:8]
            if len(ratings) < 5:
                continue
            samples.append((float(np.mean(ratings)), _finite(team["net_rtg"])))
        if len(samples) < 5:
            return None, len(samples)
        x = np.asarray([row[0] for row in samples], dtype=float)
        y = np.asarray([row[1] for row in samples], dtype=float)
        if float(np.std(x)) < 1e-6:
            return None, len(samples)
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        return float(np.clip(slope, -8.0, 8.0)), len(samples)

    def _project_player_for_team(self, spec, player_id: int, team_id: int, source_league: int, target_league: int) -> dict[str, Any]:
        result = self._player_team(spec, [player_id], [team_id], source_league, target_league)
        payload = result["result"]
        rating = payload.get("predicted_rating", payload.get("projected_rating"))
        return {"rating": float(rating), "analysis": result}

    def _team_add_player(self, spec, players, teams, source_league, target_league):
        if not players or not teams:
            raise ValueError("player and team are required")
        target_league = target_league or source_league
        source_league = source_league or target_league
        if target_league is None or source_league is None:
            raise ValueError("league context is required")
        competition = normalize_competition(spec.get("target_competition") or spec["competition"])
        season = int(spec["season"])
        projection = self._project_player_for_team(spec, players[0], teams[0], source_league, target_league)
        rotation = self._rotation_ratings(teams[0], target_league, competition, season)
        if not rotation:
            raise ValueError("Target team rotation is unavailable")
        top = rotation[:8]
        before = float(np.mean([rating for _, rating in top]))
        incoming = projection["rating"]
        replaced = min(top, key=lambda item: item[1])
        after_values = [rating for pid, rating in top if pid != replaced[0]] + [incoming]
        after = float(np.mean(after_values[:8]))
        slope, samples = self._team_impact_slope(target_league, competition, season)
        team_row = self._latest(self._team_rows(teams[0], target_league, competition, season))
        base_net = _finite(team_row.get("net_rtg"))
        net_delta = (after - before) * slope if slope is not None else None
        return {
            "result": {
                "team": self._team_name(teams[0]),
                "player": self._player_name(players[0]),
                "competition": competition,
                "projected_player_rating": round(incoming, 3),
                "rotation_rating_before": round(before, 3),
                "rotation_rating_after": round(after, 3),
                "rotation_delta": round(after - before, 3),
                "replacement_level_rating": round(replaced[1], 3),
                "team_net_rating_before": round(base_net, 3),
                "estimated_net_rating_delta": round(net_delta, 3) if net_delta is not None else None,
                "estimated_net_rating_after": round(base_net + net_delta, 3) if net_delta is not None else None,
                "player_projection": projection["analysis"]["result"],
            },
            "evidence": [{"type": "team_impact_regression_samples", "count": samples}, {"type": "rotation_players", "count": len(top)}],
            "support": {"method": "rotation_replacement_plus_empirical_team_impact", "samples": samples, "confidence": _confidence(samples)},
            "limitations": [] if slope is not None else ["Too few historical team samples to translate rotation improvement into Net Rating."],
        }

    def _team_replace_player(self, spec, players, teams, source_league, target_league):
        if len(players) < 2 or not teams:
            raise ValueError("incoming player, outgoing player and team are required")
        target_league = target_league or source_league
        source_league = source_league or target_league
        if target_league is None or source_league is None:
            raise ValueError("league context is required")
        competition, season = normalize_competition(spec.get("target_competition") or spec["competition"]), int(spec["season"])
        projection = self._project_player_for_team(spec, players[0], teams[0], source_league, target_league)
        rotation = self._rotation_ratings(teams[0], target_league, competition, season)
        outgoing = next((rating for pid, rating in rotation if pid == _to_int(players[1])), None)
        if outgoing is None:
            raise ValueError("Outgoing player is not in the target context rotation")
        top = rotation[:8]
        before = float(np.mean([rating for _, rating in top]))
        values = [rating for pid, rating in top if pid != _to_int(players[1])]
        values.append(projection["rating"])
        after = float(np.mean(sorted(values, reverse=True)[:8]))
        slope, samples = self._team_impact_slope(target_league, competition, season)
        net_delta = (after - before) * slope if slope is not None else None
        return {
            "result": {"team": self._team_name(teams[0]), "incoming_player": self._player_name(players[0]), "outgoing_player": self._player_name(players[1]), "incoming_rating": round(projection["rating"], 3), "outgoing_rating": round(outgoing, 3), "rotation_delta": round(after - before, 3), "estimated_net_rating_delta": round(net_delta, 3) if net_delta is not None else None},
            "evidence": [{"type": "team_impact_regression_samples", "count": samples}],
            "support": {"method": "explicit_rotation_replacement", "samples": samples, "confidence": _confidence(samples)},
            "limitations": [] if slope is not None else ["Net Rating delta is unavailable because historical support is insufficient."],
        }

    def _best_team_fit(self, spec, players, teams, source_league, target_league):
        del teams
        if not players:
            raise ValueError("player is required")
        target_league = target_league or source_league
        source_league = source_league or target_league
        if target_league is None or source_league is None:
            raise ValueError("league context is required")
        candidates = self.data.get("teams", pd.DataFrame())
        candidates = candidates[candidates["league_id"].map(_to_int) == _to_int(target_league)]
        results = []
        for row in candidates.head(80).to_dict("records"):
            tid = _to_int(row["id"])
            try:
                analysis = self._player_team(spec, players, [tid], source_league, target_league)
                payload = analysis["result"]
                rating = payload.get("predicted_rating", payload.get("projected_rating"))
                results.append({"team": self._team_name(tid), "rating": float(rating), "method": analysis["support"].get("method")})
            except (ValueError, RuntimeError):
                continue
        results.sort(key=lambda row: row["rating"], reverse=True)
        return {"result": {"player": self._player_name(players[0]), "target_league": self._league_name(target_league), "teams": results[: int(spec.get("top_n", 5))]}, "evidence": [{"type": "teams_evaluated", "count": len(results)}], "support": {"method": "scenario_rank", "samples": len(results), "confidence": _confidence(len(results))}, "limitations": []}

    def _best_player_fit(self, spec, players, teams, source_league, target_league):
        del players
        if not teams:
            raise ValueError("team is required")
        target_league = target_league or source_league
        if target_league is None:
            raise ValueError("target league is required")
        season, competition = int(spec["season"]), normalize_competition(spec.get("target_competition") or spec["competition"])
        stats = self.data.get("player_stats", pd.DataFrame()).copy()
        stats["_year"] = stats["season"].map(_year)
        candidates = stats[(stats["league_id"].map(_to_int) == _to_int(target_league)) & (stats["competition"].map(normalize_competition) == competition) & (stats["_year"] == season)].copy()
        if "rating" in candidates.columns:
            candidates = candidates.sort_values("rating", ascending=False).head(120)
        results = []
        for pid in candidates["player_id"].map(_to_int).drop_duplicates().tolist():
            try:
                analysis = self._player_team(spec, [pid], teams, target_league, target_league)
                rating = analysis["result"]["predicted_rating"]
                results.append({"player": self._player_name(pid), "rating": float(rating)})
            except (ValueError, RuntimeError):
                continue
        results.sort(key=lambda row: row["rating"], reverse=True)
        return {"result": {"team": self._team_name(teams[0]), "competition": competition, "players": results[: int(spec.get("top_n", 5))]}, "evidence": [{"type": "players_evaluated", "count": len(results)}], "support": {"method": "strict_ensemble_rank", "samples": len(results), "confidence": _confidence(len(results))}, "limitations": []}

    def _player_similarity(self, spec, players, teams, source_league, target_league):
        del teams, target_league
        if not players or source_league is None:
            raise ValueError("player and source league are required")
        season, competition = int(spec["season"]), normalize_competition(spec["competition"])
        stats = self.data.get("player_stats", pd.DataFrame()).copy()
        stats["_year"] = stats["season"].map(_year)
        rows = stats[(stats["league_id"].map(_to_int) == _to_int(source_league)) & (stats["competition"].map(normalize_competition) == competition) & (stats["_year"] == season)].copy()
        features = ["usg_pct", "ts_pct", "assists", "three_point_pct", "dbpm", "bpm", "rating"]
        if rows.empty:
            raise ValueError("Comparison pool is unavailable")
        matrix = rows[features].apply(pd.to_numeric, errors="coerce").fillna(rows[features].apply(pd.to_numeric, errors="coerce").median()).fillna(0.0)
        mean, std = matrix.mean(), matrix.std(ddof=0).replace(0, 1.0)
        z = (matrix - mean) / std
        target_index = rows.index[rows["player_id"].map(_to_int) == _to_int(players[0])]
        if len(target_index) == 0:
            raise ValueError("Target player is unavailable in the comparison pool")
        target = z.loc[target_index[0]].to_numpy(dtype=float)
        distances = np.linalg.norm(z.to_numpy(dtype=float) - target, axis=1)
        rows = rows.assign(_distance=distances)
        rows = rows[rows["player_id"].map(_to_int) != _to_int(players[0])].sort_values("_distance")
        similar = [
            {
                "player": self._player_name(_to_int(row["player_id"])),
                "distance": round(float(row["_distance"]), 3),
                "rating": round(_finite(row.get("rating")), 3),
            }
            for row in rows.head(int(spec.get("top_n", 5))).to_dict("records")
        ]
        return {"result": {"player": self._player_name(players[0]), "competition": competition, "similar_players": similar}, "evidence": [{"type": "comparison_pool", "count": len(rows)}], "support": {"method": "standardised_player_profile_distance", "samples": len(rows), "confidence": _confidence(len(rows))}, "limitations": []}

    def _age_trajectory(self, spec, players, teams, source_league, target_league):
        del target_league
        if not players or not teams or source_league is None:
            raise ValueError("player, team and league are required")
        competition = normalize_competition(spec["competition"])
        season = int(spec["season"])
        snapshot = build_historical_snapshot(self.data, season)
        scoped = scope_prediction_context(snapshot, players[0], teams[0], source_league, competition, season)
        engine = StrictWhatIfEngine(self.ensemble, scoped)
        age_range = spec.get("parameters", {}).get("age_range")
        parsed_range = None
        if isinstance(age_range, list) and len(age_range) == 2:
            parsed_range = (max(18, int(age_range[0])), min(45, int(age_range[1])))
        trajectory = engine.predict_age_trajectory(players[0], season_base=season, age_range=parsed_range, team_id=teams[0])
        return {"result": {"player": self._player_name(players[0]), "team": self._team_name(teams[0]), "trajectory": [asdict(point) for point in trajectory]}, "evidence": [], "support": {"method": "strict_supervised_age_trajectory", "samples": len(trajectory), "confidence": "medium"}, "limitations": ["Long-horizon uncertainty grows beyond the one-season training target."]}
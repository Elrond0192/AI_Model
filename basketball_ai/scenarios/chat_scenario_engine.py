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
    ) -> None:
        self.ensemble = ensemble
        self.data = data
        self.advanced = AdvancedSimulationEngine(data)

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
            "team_competition": self._team_competition,
            "player_trend": self._player_trend,
            "performance_decomposition": self._performance_decomposition,
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

    def _player_team(self, spec, players, teams, source_league, target_league):
        if not players or not teams:
            raise ValueError("player and team are required")
        source_league = source_league or target_league
        target_league = target_league or source_league
        if source_league is None or target_league is None:
            raise ValueError("league context is required")
        if source_league != target_league:
            return self._league_transfer(spec, players, teams, source_league, target_league)
        result, scoped = self._direct_prediction(
            players[0], teams[0], target_league, int(spec["season"]), normalize_competition(spec.get("target_competition") or spec["competition"])
        )
        support = dict(scoped.get("_competition_support", {}))
        player_history = self._player_rows(
            players[0], source_league, spec["competition"], int(spec["season"])
        )
        team_history = self._team_rows(
            teams[0], target_league, normalize_competition(spec.get("target_competition") or spec["competition"]), int(spec["season"])
        )
        player_latest = _serialise(self._latest(player_history), _PLAYER_METRICS) if not player_history.empty else {}
        team_latest = _serialise(self._latest(team_history), _TEAM_METRICS) if not team_history.empty else {}
        player_ratings = pd.to_numeric(player_history.get("rating"), errors="coerce").dropna() if "rating" in player_history else pd.Series(dtype=float)
        return {
            "result": {
                "player": self._player_name(players[0]),
                "team": self._team_name(teams[0]),
                "league": self._league_name(target_league),
                "competition": result.competition,
                "target_season": int(spec["season"]) + 1,
                "predicted_rating": result.predicted_rating,
                "confidence_low": result.confidence_low,
                "confidence_high": result.confidence_high,
                "base_rating": result.base_rating,
                "compatibility": result.compatibility_factor,
                "league_factor": result.league_factor,
                "context_adjustment": result.context_adjustment,
                "drivers": result.shap_values,
            },
            "evidence": [
                {"type": "competition_support", **support},
                {
                    "type": "player_historical_profile",
                    "player": self._player_name(players[0]),
                    "seasons_considered": int(len(player_history)),
                    "through_season": int(spec["season"]),
                    "latest": player_latest,
                    "rating_history": [
                        {"season": int(row["_year"]), "rating": round(float(row["rating"]), 3)}
                        for row in player_history.tail(8).to_dict("records")
                        if pd.notna(row.get("rating"))
                    ],
                    "rating_average": round(float(player_ratings.mean()), 3) if not player_ratings.empty else None,
                },
                {
                    "type": "target_team_profile",
                    "team": self._team_name(teams[0]),
                    "seasons_considered": int(len(team_history)),
                    "through_season": int(spec["season"]),
                    "latest": team_latest,
                },
            ],
            "support": {"method": "strict_supervised_ensemble", **support},
            "limitations": [],
        }

    def _league_quality_fallback(self, source_league: int, target_league: int) -> float:
        leagues = self.data.get("league_dict", {})
        source = leagues.get(_to_int(source_league), {})
        target = leagues.get(_to_int(target_league), {})
        source_q = _finite(source.get("competitiveness_score"), 1.0)
        target_q = _finite(target.get("competitiveness_score"), 1.0)
        maximum = max(
            [_finite(row.get("competitiveness_score"), 1.0) for row in leagues.values()] or [1.0]
        )
        if maximum <= 0:
            return 0.0
        return float(np.clip((source_q - target_q) / maximum * 1.25, -1.25, 1.25))

    def _transfer_samples(
        self,
        source_league: int,
        target_league: int,
        source_comp: str,
        target_comp: str,
        through_season: int,
    ) -> list[float]:
        stats = self.data.get("player_stats", pd.DataFrame()).copy()
        if stats.empty:
            return []
        stats["_year"] = stats["season"].map(_year)
        stats["_comp"] = stats["competition"].map(normalize_competition)
        stats = stats[stats["_year"] <= int(through_season) + 1]
        stats["_pid_int"] = stats["player_id"].map(_to_int)
        stats["_league_int"] = stats["league_id"].map(_to_int)
        stats["_row_order"] = np.arange(len(stats), dtype=int)
        source_rows = stats[
            (stats["_league_int"] == _to_int(source_league))
            & (stats["_comp"] == source_comp)
        ][["_pid_int", "_year", "rating", "_row_order"]].copy()
        target_rows = stats[
            (stats["_league_int"] == _to_int(target_league))
            & (stats["_comp"] == target_comp)
        ][["_pid_int", "_year", "rating", "_row_order"]].copy()
        if source_rows.empty or target_rows.empty:
            return []

        # The legacy per-player dict retained the last target row for a year.
        target_rows = target_rows.drop_duplicates(
            ["_pid_int", "_year"], keep="last"
        ).rename(columns={"_year": "_target_year", "rating": "_target_rating"})
        source_rows["_target_year"] = source_rows["_year"] + 1
        paired = source_rows.merge(
            target_rows[["_pid_int", "_target_year", "_target_rating"]],
            on=["_pid_int", "_target_year"],
            how="inner",
            sort=False,
            validate="many_to_one",
        )
        paired["_source_rating"] = pd.to_numeric(paired["rating"], errors="coerce")
        paired["_target_rating"] = pd.to_numeric(
            paired["_target_rating"], errors="coerce"
        )
        paired = paired.dropna(subset=["_source_rating", "_target_rating"])
        paired = paired.sort_values(["_pid_int", "_row_order"], kind="stable")
        return (paired["_target_rating"] - paired["_source_rating"]).astype(float).tolist()

    def _target_context_score(self, player_id: int, team_id: int, target_league: int, competition: str, season: int) -> tuple[float, dict[str, float]]:
        snapshot = build_historical_snapshot(self.data, season)
        team_rows = self._team_rows(team_id, target_league, competition, season)
        if team_rows.empty:
            return 0.5, {}
        latest = team_rows.iloc[-1]
        patched = dict(snapshot)
        team_dict = dict(snapshot.get("team_dict", {}))
        team = dict(team_dict.get(_to_int(team_id), self.data.get("team_dict", {}).get(_to_int(team_id), {})))
        for metric in _TEAM_METRICS:
            if pd.notna(latest.get(metric)):
                team[metric] = latest[metric]
        team["league_id"] = target_league
        team_dict[_to_int(team_id)] = team
        patched["team_dict"] = team_dict
        player_dict = dict(patched.get("player_dict", {}))
        player = dict(player_dict.get(_to_int(player_id), {}))
        if player:
            player_dict[_to_int(player_id)] = player
            patched["player_dict"] = player_dict
        patched["_as_of_season"] = season
        ctx = compute_context_features(player_id, team_id, patched)
        score = (
            ctx["position_team_fit"] * 0.25
            + ctx["style_compatibility"] * 0.30
            + ctx["role_opportunity"] * 0.20
            + ctx["league_adaptation_factor"] * 0.15
            + ctx["spacing_fit"] * 0.10
        )
        return float(np.clip(score, 0.0, 1.0)), ctx

    def _league_transfer(self, spec, players, teams, source_league, target_league):
        if not players:
            raise ValueError("player is required")
        if source_league is None or target_league is None:
            raise ValueError("source_league and target_league are required")
        season = int(spec["season"])
        source_comp = normalize_competition(spec["competition"])
        target_comp = normalize_competition(spec.get("target_competition") or source_comp)
        source_rows = self._player_rows(players[0], source_league, source_comp, season)
        source = self._latest(source_rows)
        source_rating = _finite(source.get("rating"), 6.5)
        samples = self._transfer_samples(source_league, target_league, source_comp, target_comp, season)
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
                players[0], teams[0], target_league, target_comp, season
            )
        fit_delta = float(np.clip((context_score - 0.65) * 0.70, -0.30, 0.25)) if teams else 0.0
        projected = float(np.clip(source_rating + transfer_delta + fit_delta, 0.0, 10.0))
        residual_scale = float(np.std(samples, ddof=1)) if len(samples) >= 2 else 0.75
        half_width = max(0.45, residual_scale * (1.35 if len(samples) < 10 else 1.0))
        limitations = []
        if len(samples) < 5:
            limitations.append("Cross-league support is sparse; the estimate relies materially on league-quality priors.")
        return {
            "result": {
                "player": self._player_name(players[0]),
                "source_league": self._league_name(source_league),
                "target_league": self._league_name(target_league),
                "source_competition": source_comp,
                "target_competition": target_comp,
                "source_season": int(source["_year"]),
                "target_season": season + 1,
                "source_rating": round(source_rating, 3),
                "projected_rating": round(projected, 3),
                "confidence_low": round(max(0.0, projected - half_width), 3),
                "confidence_high": round(min(10.0, projected + half_width), 3),
                "empirical_transfer_delta": round(empirical, 3) if empirical is not None else None,
                "league_prior_delta": round(fallback_delta, 3),
                "team_fit_delta": round(fit_delta, 3),
                "team": self._team_name(teams[0]) if teams else None,
                "context": context,
            },
            "evidence": [{"type": "historical_cross_league_transitions", "count": len(samples)}],
            "support": {"method": method, "samples": len(samples), "confidence": _confidence(len(samples))},
            "limitations": limitations,
        }

    # ------------------------------------------------------------------
    # Playoffs, pair/lineup and counterfactuals
    # ------------------------------------------------------------------
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

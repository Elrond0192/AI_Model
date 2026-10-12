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

from basketball_ai.scenarios.scouting_archetypes import ARCHETYPE_METRICS
from basketball_ai.data.nationality_codes import nationality_matches, normalize_nationality
from basketball_ai.scenarios.acquisition_impact import (
    fit_acquisition_impact_model,
    predict_acquisition_impact,
)

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
            "player_scouting": self._player_scouting,
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

    def _player_identity_ids(self, player_id: int) -> set[int]:
        """Return every internal player ID belonging to the same canonical global identity."""
        pid = _to_int(player_id)
        if pid is None:
            return set()

        ids = {pid}
        players_dict = self.data.get("player_dict", {}) or {}
        global_id = ""

        player_row = players_dict.get(pid, {})
        if isinstance(player_row, dict):
            global_id = str(player_row.get("global_id", "")).strip()

        # The lookup dictionary is normally authoritative, but keep a DataFrame
        # fallback because some runtime/test loaders populate `players` without
        # rebuilding `player_dict` after canonical identity enrichment.
        players_frame = self.data.get("players", pd.DataFrame())
        if not global_id and isinstance(players_frame, pd.DataFrame) and not players_frame.empty:
            if "id" in players_frame.columns and "global_id" in players_frame.columns:
                matches = players_frame[players_frame["id"].map(_to_int) == pid]
                if not matches.empty:
                    global_id = str(matches.iloc[0].get("global_id", "")).strip()

        if not global_id:
            return ids

        for raw_id, row in players_dict.items():
            if not isinstance(row, dict):
                continue
            if str(row.get("global_id", "")).strip() != global_id:
                continue
            other_id = _to_int(raw_id)
            if other_id is not None:
                ids.add(other_id)

        if isinstance(players_frame, pd.DataFrame) and not players_frame.empty:
            if {"id", "global_id"}.issubset(players_frame.columns):
                matches = players_frame.loc[
                    players_frame["global_id"].astype(str).str.strip() == global_id
                ]
                for row in matches.to_dict("records"):
                    other_id = _to_int(row.get("id"))
                    if other_id is not None:
                        ids.add(other_id)

        return ids

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
        identity_ids = self._player_identity_ids(player_id)
        global_id = ""
        players_dict = self.data.get("player_dict", {}) or {}
        player_meta = players_dict.get(_to_int(player_id), {})
        if isinstance(player_meta, dict):
            global_id = str(player_meta.get("global_id", "")).strip()

        # Resolve the canonical identity from Players itself as the final
        # authority. This also handles runtimes where player_dict was built from
        # only one representative row of a multi-ID player.
        players_frame = self.data.get("players", pd.DataFrame())
        if not global_id and isinstance(players_frame, pd.DataFrame) and not players_frame.empty:
            if {"id", "global_id"}.issubset(players_frame.columns):
                matches = players_frame[players_frame["id"].map(_to_int) == _to_int(player_id)]
                if not matches.empty:
                    global_id = str(matches.iloc[0].get("global_id", "")).strip()

        if global_id and isinstance(players_frame, pd.DataFrame) and not players_frame.empty:
            if {"id", "global_id"}.issubset(players_frame.columns):
                canonical_map = players_frame[["id", "global_id"]].copy()
                canonical_map["id"] = canonical_map["id"].map(_to_int)
                canonical_map["global_id"] = canonical_map["global_id"].astype(str).str.strip()
                canonical_map = canonical_map[canonical_map["global_id"] != ""].drop_duplicates("id", keep="last")
                canonical_by_id = canonical_map.set_index("id")["global_id"]
                row_identity = frame["player_id"].map(_to_int).map(canonical_by_id)
                identity_mask = row_identity == global_id
            else:
                identity_mask = frame["player_id"].map(_to_int).isin(identity_ids)
        elif global_id and "global_id" in frame.columns:
            identity_mask = frame["global_id"].fillna("").astype(str).str.strip() == global_id
        else:
            identity_mask = frame["player_id"].map(_to_int).isin(identity_ids)

        mask = (
            identity_mask
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
                "league_key": self._league_key(source_league),
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
        player_row = self.data.get("player_dict", {}).get(_to_int(player), {})
        global_id = str(player_row.get("global_id", ""))
        player_id = _to_int(player)
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

        # A generic/current Player Intelligence request is descriptive only.
        # Future Performance and Prediction must never leak into the semantic
        # evidence unless the user explicitly asks for a forward-looking view.
        requested_keys = [
            str(value).strip().lower()
            for value in (spec.get("parameters", {}) or {}).get("question_keys", [])
            if str(value).strip()
        ]
        current_level_only = "current_level" in requested_keys or not requested_keys

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
                    "dimensions": {
                        str(key): int(value)
                        for key, value in (getattr(rated, "dimensions", {}) or {}).items()
                        if value is not None
                    },
                    "strengths": list(getattr(rated, "strengths", ()) or ()),
                    "limitations": list(getattr(rated, "limitations", ()) or ()),
                    "explanation": getattr(rated, "explanation", None),
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

        if self.future_performance_model is not None and not current_level_only:
            try:
                future = self.future_performance_model.predict_player(
                    self.data,
                    player_id=player_id,
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
                    player_id, source_league, competition, season
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
                    .get(player_id, {})
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

        # Prediction Model is also forward-looking for Chat V3 purposes;
        # keep it out of a generic/current Player Intelligence answer.
        if current_level_only:
            return result

        # Prediction Model: observed current-team context, not a future-team counterfactual.
        try:
            rows = self._player_rows(player_id, source_league, competition, season)
            latest = self._latest(rows)
            current_team_id = _to_int(latest.get("team_id"))
            observed_rating = _finite(latest.get("rating"), np.nan)
            if current_team_id is not None and source_league is not None:
                prediction, _ = self._direct_prediction(player_id, current_team_id, source_league, season, competition)
                result["prediction_model"] = {"available": True, "scope": "current_team_context", "team": self._team_name(current_team_id), "predicted_rating": round(float(prediction.predicted_rating), 3), "confidence_low": round(float(prediction.confidence_low), 3), "confidence_high": round(float(prediction.confidence_high), 3), "base_rating": round(float(prediction.base_rating), 3), "age_factor": round(float(prediction.age_factor), 4), "compatibility_factor": round(float(prediction.compatibility_factor), 4), "league_factor": round(float(prediction.league_factor), 4), "context_adjustment": round(float(prediction.context_adjustment), 4), "delta_vs_observed": round(float(prediction.predicted_rating) - observed_rating, 3) if math.isfinite(observed_rating) else None, "interpretation": "Stima del livello nel contesto della squadra corrente; non è una previsione di cambio squadra né una proiezione per-36."}
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            result["prediction_model"]["reason"] = str(exc)

        return result

    def _intelligence_context(self, spec, players, source_league):
        """Build descriptive context used by the Player Intelligence synthesis layer."""
        if not players:
            raise ValueError("player is required")
        raw_player = players[0]
        player_id = int(raw_player["id"]) if isinstance(raw_player, dict) else int(raw_player)
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

    def _player_intelligence_context_leagues(
        self,
        spec,
        player_id: int,
        season: int,
        competition: str,
        source_league: int | None,
        team_ids: list[int] | None = None,
    ) -> list[int]:
        """Return concrete league contexts for one canonical player identity."""
        if source_league is not None:
            return [_to_int(source_league)]
        frame = self.data.get("player_stats", pd.DataFrame()).copy()
        if frame.empty or "player_id" not in frame.columns or "league_id" not in frame.columns:
            return []
        frame["_year"] = frame["season"].map(_year)
        identity_ids = self._player_identity_ids(player_id)
        global_id = ""
        player_meta = (self.data.get("player_dict", {}) or {}).get(_to_int(player_id), {})
        if isinstance(player_meta, dict):
            global_id = str(player_meta.get("global_id", "")).strip()
        if not global_id and "global_id" in frame.columns:
            candidate = frame.loc[
                frame["player_id"].map(_to_int) == _to_int(player_id), "global_id"
            ]
            candidate = candidate.dropna().astype(str).str.strip()
            if not candidate.empty:
                global_id = candidate.iloc[0]

        if global_id:
            # Players is the canonical identity authority. Do not let a stale or
            # missing statistic-level global_id hide a valid competition row.
            players_frame = self.data.get("players", pd.DataFrame())
            if isinstance(players_frame, pd.DataFrame) and not players_frame.empty and {"id", "global_id"}.issubset(players_frame.columns):
                canonical_map = players_frame[["id", "global_id"]].copy()
                canonical_map["id"] = canonical_map["id"].map(_to_int)
                canonical_map["global_id"] = canonical_map["global_id"].astype(str).str.strip()
                canonical_map = canonical_map[
                    (canonical_map["id"].notna()) & (canonical_map["global_id"] != "")
                ].drop_duplicates("id", keep="last")
                canonical_by_id = canonical_map.set_index("id")["global_id"]
                row_identity = frame["player_id"].map(_to_int).map(canonical_by_id)
                identity_mask = row_identity == global_id
            elif "global_id" in frame.columns:
                identity_mask = frame["global_id"].fillna("").astype(str).str.strip() == global_id
            else:
                identity_mask = frame["player_id"].map(_to_int).isin(identity_ids)
        else:
            identity_mask = frame["player_id"].map(_to_int).isin(identity_ids)
        # A player can move competitions between seasons. For a generic
        # cross-context profile, include this season and the immediately previous
        # season so one league's newly-started season does not hide another
        # league's latest completed context (e.g. EL 2025 alongside ITA1 2026).
        latest_context_start = max(2000, int(season) - 1)
        mask = (
            identity_mask
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (frame["_year"] <= int(season))
            & (frame["_year"] >= latest_context_start)
        )
        if team_ids:
            if "team_id" not in frame.columns:
                return []
            allowed_team_ids = {_to_int(value) for value in team_ids}

            # A club can have one canonical global_id but distinct internal team
            # rows per league. Expand the requested team IDs to every team row
            # sharing that global identity before filtering PlayerCompetitionStats.
            teams = self.data.get("teams", pd.DataFrame())
            if not teams.empty and {"id", "global_id"} <= set(teams.columns):
                selected_globals = {
                    str(row.get("global_id")).strip()
                    for row in teams.to_dict("records")
                    if _to_int(row.get("id")) in allowed_team_ids and str(row.get("global_id", "")).strip()
                }
                if selected_globals:
                    allowed_team_ids.update(
                        _to_int(row.get("id"))
                        for row in teams.to_dict("records")
                        if str(row.get("global_id", "")).strip() in selected_globals
                        and row.get("id") is not None
                    )

            mask &= frame["team_id"].map(_to_int).isin(allowed_team_ids)
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
            "current_level": ("player_competition", "model_evidence", "metric_explanation"),
            "why_performing": ("performance_decomposition", "metric_explanation"),
            "change_vs_last_season": ("performance_decomposition", "role_analysis", "metric_explanation"),
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
        # cross_context=true is an explicit Chat V3 contract: ignore any
        # accidentally inherited source league and discover every matching
        # competition context for the selected canonical player.
        cross_context = bool(parameters.get("cross_context", False))
        effective_source_league = None if cross_context else source_league
        context_leagues = self._player_intelligence_context_leagues(
            spec,
            int(players[0]),
            int(spec["season"]),
            normalize_competition(spec.get("competition", "RS")),
            effective_source_league,
            team_ids=teams,
        )
        if cross_context:
            source_league = None
        if source_league is None and len(context_leagues) == 1:
            source_league = context_leagues[0]
        if source_league is None and len(context_leagues) > 1:
            dispatch_multi = {
                "player_competition": self._player_competition,
                "model_evidence": self._model_evidence,
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
                season_comparison = {
                    "available": False,
                    "reason": "Season-over-season comparison is unavailable in this league context.",
                }
                decomposition = context_analyses.get("performance_decomposition")
                if isinstance(decomposition, dict):
                    decomposition_result = decomposition.get("result")
                    if isinstance(decomposition_result, dict):
                        previous = decomposition_result.get("previous") or {}
                        current = decomposition_result.get("current") or {}
                        previous_rating = _finite(previous.get("rating"), np.nan)
                        current_rating = _finite(current.get("rating"), np.nan)
                        previous_team_id = previous.get("team_id")
                        current_team_id = current.get("team_id")
                        previous_team_id = (
                            _to_int(previous_team_id)
                            if previous_team_id is not None
                            else None
                        )
                        current_team_id = (
                            _to_int(current_team_id)
                            if current_team_id is not None
                            else None
                        )
                        season_comparison = {
                            "available": (
                                decomposition_result.get("previous_season") is not None
                                and decomposition_result.get("current_season") is not None
                            ),
                            "league": self._league_name(league_id),
                            "league_key": self._league_key(league_id),
                            "previous_season": decomposition_result.get("previous_season"),
                            "current_season": decomposition_result.get("current_season"),
                            "previous_rating": round(previous_rating, 3) if math.isfinite(previous_rating) else None,
                            "current_rating": round(current_rating, 3) if math.isfinite(current_rating) else None,
                            "rating_delta": (
                                round(current_rating - previous_rating, 3)
                                if math.isfinite(previous_rating) and math.isfinite(current_rating)
                                else None
                            ),
                            "previous_stats": {
                                key: round(_finite(previous.get(key)), 2)
                                for key in ("points", "assists", "rebounds", "minutes_per_game")
                                if previous.get(key) is not None and math.isfinite(_finite(previous.get(key)))
                            },
                            "current_stats": {
                                key: round(_finite(current.get(key)), 2)
                                for key in ("points", "assists", "rebounds", "minutes_per_game")
                                if current.get(key) is not None and math.isfinite(_finite(current.get(key)))
                            },
                            "previous_team": (
                                self._team_name(previous_team_id)
                                if previous_team_id is not None
                                else None
                            ),
                            "current_team": (
                                self._team_name(current_team_id)
                                if current_team_id is not None
                                else None
                            ),
                        }
                context_season = int(spec["season"])
                competition_layer = context_analyses.get("player_competition")
                if isinstance(competition_layer, dict):
                    competition_result = competition_layer.get("result")
                    if isinstance(competition_result, dict):
                        latest_season = competition_result.get("latest_season")
                        if latest_season is not None:
                            try:
                                context_season = int(latest_season)
                            except (TypeError, ValueError):
                                pass

                contexts.append({
                    "league": self._league_name(league_id),
                    "league_key": self._league_key(league_id),
                    "season": context_season,
                    "analyses": context_analyses,
                    "season_comparison": season_comparison,
                })

            season_comparisons = [
                item["season_comparison"]
                for item in contexts
                if item.get("season_comparison", {}).get("available")
            ]
            return {
                "result": {
                    "player": self._player_name(players[0]),
                    "question_key": keys[0] if len(keys) == 1 else None,
                    "question_keys": keys,
                    "context": None,
                    "contexts": contexts,
                    "season_comparisons": season_comparisons,
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
            "model_evidence": self._model_evidence,
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

    def _team_roster_size(
        self, team_id: int, league_id: int, competition: str, season: int
    ) -> int:
        relations = self.data.get("team_player_relations", pd.DataFrame())
        if isinstance(relations, pd.DataFrame) and not relations.empty and {
            "team_id", "player_id", "season"
        }.issubset(relations.columns):
            relation_year = relations["season"].map(_year)
            selected = relations[
                (relations["team_id"].map(_to_int) == _to_int(team_id))
                & (relation_year == int(season))
            ]
            if "league_id" in selected.columns:
                selected = selected[selected["league_id"].map(_to_int) == _to_int(league_id)]
            size = int(selected["player_id"].map(_to_int).dropna().nunique())
            if size > 0:
                return size

        stats = self.data.get("player_stats", pd.DataFrame())
        if not isinstance(stats, pd.DataFrame) or stats.empty or not {
            "team_id", "player_id", "league_id", "season", "competition"
        }.issubset(stats.columns):
            return 0
        stat_year = stats["season"].map(_year)
        rows = stats[
            (stats["team_id"].map(_to_int) == _to_int(team_id))
            & (stats["league_id"].map(_to_int) == _to_int(league_id))
            & (stats["competition"].map(normalize_competition) == normalize_competition(competition))
            & (stat_year == int(season))
        ]
        return int(rows["player_id"].map(_to_int).dropna().nunique())

    def _team_acquisition_forecast(
        self,
        team_id: int,
        league_id: int,
        competition: str,
        season: int,
        rotation_quality_delta: float,
        roster_size: int,
        impact_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a fail-closed, walk-forward-validated acquisition estimate."""
        team_rows = self._team_rows(team_id, league_id, competition, season)
        if team_rows.empty or int(_year(team_rows.iloc[-1]["season"])) != int(season):
            return {
                "status": "team_context_unavailable",
                "reason": "Net Rating e roster della stagione target non sono disponibili.",
                "current_net_rtg": None,
                "impact": {"status": "team_context_unavailable", "available": False},
            }

        current_row = self._latest(team_rows)
        current_net = _finite(current_row.get("net_rtg"), float("nan"))
        if len(team_rows) < 2:
            return {
                "status": "team_history_unavailable",
                "reason": "Servono almeno due stagioni consecutive di Net Rating per impostare il forecast.",
                "current_net_rtg": current_net if math.isfinite(current_net) else None,
                "impact": {"status": "team_history_unavailable", "available": False},
            }
        prior_row = team_rows.iloc[-2]
        if int(_year(prior_row["season"])) != int(season) - 1:
            return {
                "status": "team_history_gap",
                "reason": "Manca la stagione immediatamente precedente per calcolare il trend della squadra.",
                "current_net_rtg": current_net if math.isfinite(current_net) else None,
                "impact": {"status": "team_history_gap", "available": False},
            }
        prior_net = _finite(prior_row.get("net_rtg"), float("nan"))
        if not math.isfinite(current_net) or not math.isfinite(prior_net):
            return {
                "status": "team_net_rating_unavailable",
                "reason": "Net Rating mancante nella serie storica della squadra.",
                "current_net_rtg": None,
                "impact": {"status": "team_net_rating_unavailable", "available": False},
            }

        prior_change = current_net - prior_net
        model = impact_model if isinstance(impact_model, dict) else fit_acquisition_impact_model(
            self.data, league_id, competition, through_season=season
        )
        transaction_turnover = min(1.0, 2.0 / max(1, int(roster_size)))
        impact = predict_acquisition_impact(
            model,
            roster_quality_delta=float(rotation_quality_delta),
            transaction_turnover_delta=transaction_turnover,
            prior_net_rtg=current_net,
            prior_net_rtg_change=prior_change,
        )
        return {
            "status": str(model.get("status", "uncalibrated")),
            "reason": str(model.get("reason", "")),
            "current_net_rtg": current_net,
            "target_season": int(season) + 1,
            "roster_size": int(roster_size),
            "rotation_quality_delta": round(float(rotation_quality_delta), 4),
            "transaction_turnover_delta": round(transaction_turnover, 4),
            "model": {
                "status": str(model.get("status", "uncalibrated")),
                "reason": str(model.get("reason", "")),
                "method": str(model.get("method", "")),
                "training_samples": int(model.get("training_samples", 0)),
                "training_seasons": int(model.get("training_seasons", 0)),
                "validation": model.get("validation", {}),
            },
            "impact": impact,
        }

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
        if len(rotation) < 5:
            raise ValueError("Target team needs at least five rotation ratings for an acquisition what-if")
        if any(pid == _to_int(players[0]) for pid, _ in rotation):
            raise ValueError("The incoming player is already in the target team's observed rotation")
        top = rotation[:8]
        before = float(np.mean([rating for _, rating in top]))
        incoming = projection["rating"]
        replaced = min(top, key=lambda item: item[1])
        after_values = [rating for pid, rating in top if pid != replaced[0]] + [incoming]
        after = float(np.mean(after_values))
        roster_size = self._team_roster_size(teams[0], target_league, competition, season) or int(len({pid for pid, _ in rotation}))
        forecast = self._team_acquisition_forecast(
            teams[0], target_league, competition, season,
            rotation_quality_delta=after - before, roster_size=roster_size,
        )
        impact = forecast.get("impact", {}) if isinstance(forecast.get("impact"), dict) else {}
        estimate_available = bool(impact.get("available"))
        current_net = forecast.get("current_net_rtg")
        net_change_with = impact.get("predicted_team_net_rating_change_with_acquisition") if estimate_available else None
        net_change_without = impact.get("predicted_team_net_rating_change_without_acquisition") if estimate_available else None
        estimated_after = (
            float(current_net) + float(net_change_with)
            if estimate_available and current_net is not None and net_change_with is not None else None
        )
        limitations = [
            "L'impatto stimato è un'associazione osservazionale ricavata da transizioni storiche di roster; non identifica l'effetto causale del singolo acquisto.",
            "Il forecast considera la qualità relativa della rotazione, il turnover storico e il Net Rating precedente; non modella direttamente infortuni, minuti promessi, compatibilità nello spogliatoio o decisioni dell'allenatore.",
        ]
        if not estimate_available:
            limitations.append(
                "Nessun valore numerico dell'impatto viene pubblicato perché il modello storico non ha superato i requisiti minimi di validazione: "
                + str(forecast.get("reason") or impact.get("reason") or "validazione non disponibile")
            )
        return {
            "result": {
                "team": self._team_name(teams[0]),
                "player": self._player_name(players[0]),
                "competition": competition,
                "source_season": season,
                "target_season": int(season) + 1,
                "projected_player_rating": round(incoming, 3),
                "rotation_rating_before": round(before, 3),
                "rotation_rating_after": round(after, 3),
                "rotation_delta": round(after - before, 3),
                "replacement_level_rating": round(replaced[1], 3),
                "team_net_rating_before": round(float(current_net), 3) if current_net is not None else None,
                "acquisition_impact_status": str(impact.get("status", forecast.get("status", "uncalibrated"))),
                "estimated_net_rating_delta": impact.get("estimated_incremental_net_rating") if estimate_available else None,
                "estimated_net_rating_delta_interval_90": impact.get("incremental_net_rating_interval_90") if estimate_available else None,
                "predicted_team_net_rating_change_without_acquisition": net_change_without,
                "predicted_team_net_rating_change_with_acquisition": net_change_with,
                "predicted_team_net_rating_change_interval_80": impact.get("predicted_team_net_rating_change_interval_80") if estimate_available else None,
                "estimated_net_rating_after": round(estimated_after, 3) if estimated_after is not None else None,
                "impact_model": forecast.get("model", {
                    "status": forecast.get("status"),
                    "reason": forecast.get("reason"),
                }),
                "player_projection": projection["analysis"]["result"],
            },
            "evidence": [
                {"type": "historical_roster_transitions", "count": int(forecast.get("model", {}).get("training_samples", 0))},
                {"type": "walk_forward_validation_predictions", "count": int(forecast.get("model", {}).get("validation", {}).get("samples", 0))},
                {"type": "rotation_players", "count": len(top)},
            ],
            "support": {
                "method": "walk_forward_validated_acquisition_impact" if estimate_available else "acquisition_impact_not_validated",
                "samples": int(forecast.get("model", {}).get("training_samples", 0)),
                "confidence": _confidence(int(forecast.get("model", {}).get("validation", {}).get("samples", 0))) if estimate_available else "low",
            },
            "limitations": limitations,
        }

    def _team_replace_player(self, spec, players, teams, source_league, target_league):
        if len(players) < 2 or not teams:
            raise ValueError("incoming player, outgoing player and team are required")
        target_league = target_league or source_league
        source_league = source_league or target_league
        if target_league is None or source_league is None:
            raise ValueError("league context is required")
        competition = normalize_competition(spec.get("target_competition") or spec["competition"])
        season = int(spec["season"])
        projection = self._project_player_for_team(spec, players[0], teams[0], source_league, target_league)
        rotation = self._rotation_ratings(teams[0], target_league, competition, season)
        if len(rotation) < 5:
            raise ValueError("Target team needs at least five rotation ratings for a replacement what-if")
        outgoing_id = _to_int(players[1])
        incoming_id = _to_int(players[0])
        outgoing = next((rating for pid, rating in rotation if pid == outgoing_id), None)
        if outgoing is None:
            raise ValueError("Outgoing player is not in the target context rotation")
        if incoming_id != outgoing_id and any(pid == incoming_id for pid, _ in rotation):
            raise ValueError("The incoming player is already in the target team's observed rotation")
        top = rotation[:8]
        if not any(pid == outgoing_id for pid, _ in top):
            raise ValueError("Outgoing player is outside the top-eight modeled rotation")
        before = float(np.mean([rating for _, rating in top]))
        values = [rating for pid, rating in top if pid != outgoing_id]
        values.append(projection["rating"])
        after = float(np.mean(values))
        relations = self.data.get("team_player_relations", pd.DataFrame())
        roster_size = 0
        if isinstance(relations, pd.DataFrame) and not relations.empty and {
            "team_id", "player_id", "season"
        }.issubset(relations.columns):
            relation_year = relations["season"].map(_year)
            selected = relations[
                (relations["team_id"].map(_to_int) == _to_int(teams[0]))
                & (relation_year == season)
            ]
            roster_size = int(selected["player_id"].map(_to_int).dropna().nunique())
        if roster_size == 0:
            roster_size = int(len({pid for pid, _ in rotation}))
        forecast = self._team_acquisition_forecast(
            teams[0], target_league, competition, season,
            rotation_quality_delta=after - before, roster_size=roster_size,
        )
        impact = forecast.get("impact", {}) if isinstance(forecast.get("impact"), dict) else {}
        estimate_available = bool(impact.get("available"))
        current_net = forecast.get("current_net_rtg")
        net_change_with = impact.get("predicted_team_net_rating_change_with_acquisition") if estimate_available else None
        net_change_without = impact.get("predicted_team_net_rating_change_without_acquisition") if estimate_available else None
        estimated_after = (
            float(current_net) + float(net_change_with)
            if estimate_available and current_net is not None and net_change_with is not None else None
        )
        limitations = [
            "L'impatto stimato è un'associazione osservazionale ricavata da transizioni storiche di roster; non identifica l'effetto causale del singolo acquisto.",
            "Il forecast considera la qualità relativa della rotazione, il turnover storico e il Net Rating precedente; non modella direttamente infortuni, minuti promessi, compatibilità nello spogliatoio o decisioni dell'allenatore.",
        ]
        if not estimate_available:
            limitations.append(
                "Nessun valore numerico dell'impatto viene pubblicato perché il modello storico non ha superato i requisiti minimi di validazione: "
                + str(forecast.get("reason") or impact.get("reason") or "validazione non disponibile")
            )
        return {
            "result": {
                "team": self._team_name(teams[0]),
                "incoming_player": self._player_name(players[0]),
                "outgoing_player": self._player_name(players[1]),
                "source_season": season,
                "target_season": int(season) + 1,
                "incoming_rating": round(projection["rating"], 3),
                "outgoing_rating": round(outgoing, 3),
                "rotation_delta": round(after - before, 3),
                "acquisition_impact_status": str(impact.get("status", forecast.get("status", "uncalibrated"))),
                "estimated_net_rating_delta": impact.get("estimated_incremental_net_rating") if estimate_available else None,
                "estimated_net_rating_delta_interval_90": impact.get("incremental_net_rating_interval_90") if estimate_available else None,
                "predicted_team_net_rating_change_without_acquisition": net_change_without,
                "predicted_team_net_rating_change_with_acquisition": net_change_with,
                "predicted_team_net_rating_change_interval_80": impact.get("predicted_team_net_rating_change_interval_80") if estimate_available else None,
                "estimated_net_rating_after": round(estimated_after, 3) if estimated_after is not None else None,
                "impact_model": forecast.get("model", {
                    "status": forecast.get("status"),
                    "reason": forecast.get("reason"),
                }),
            },
            "evidence": [
                {"type": "historical_roster_transitions", "count": int(forecast.get("model", {}).get("training_samples", 0))},
                {"type": "walk_forward_validation_predictions", "count": int(forecast.get("model", {}).get("validation", {}).get("samples", 0))},
                {"type": "rotation_players", "count": len(top)},
            ],
            "support": {
                "method": "walk_forward_validated_acquisition_impact" if estimate_available else "acquisition_impact_not_validated",
                "samples": int(forecast.get("model", {}).get("training_samples", 0)),
                "confidence": _confidence(int(forecast.get("model", {}).get("validation", {}).get("samples", 0))) if estimate_available else "low",
            },
            "limitations": limitations,
        }


    def _project_player_for_team(self, spec, player_id: int, team_id: int, source_league: int, target_league: int) -> dict[str, Any]:
        result = self._player_team(spec, [player_id], [team_id], source_league, target_league)
        payload = result["result"]
        rating = payload.get("predicted_rating", payload.get("projected_rating"))
        return {"rating": float(rating), "analysis": result}

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

    def _player_scouting(self, spec, players, teams, source_league, target_league):
        """Rank a league-scoped player pool against explicit scouting filters and archetypes.

        This is a transparent percentile-based fit score, not a trained prediction
        or a replacement for BB-Rating. Filters are strict; unavailable identity
        fields never silently satisfy nationality/age requirements.
        """
        del players
        parameters = spec.get("parameters", {}) or {}
        league_id = target_league if target_league is not None else source_league
        if league_id is None:
            raise ValueError("source_league or target_league is required to keep candidate comparisons in one league")
        season = int(spec["season"])
        competition = normalize_competition(spec.get("target_competition") or spec["competition"])
        archetype = str(parameters.get("archetype", "balanced")).strip().lower()
        archetype_metrics = ARCHETYPE_METRICS
        if archetype not in archetype_metrics:
            raise ValueError(
                f"unsupported scouting archetype {archetype!r}; "
                f"choose from {sorted(archetype_metrics)}"
            )

        stats = self.data.get("player_stats", pd.DataFrame()).copy()
        if stats.empty:
            raise ValueError("Player competition statistics are unavailable")
        required = {"player_id", "league_id", "season", "competition"}
        if not required.issubset(stats.columns):
            raise ValueError("Canonical player statistics are missing required columns")
        stats["_year"] = stats["season"].map(_year)
        pool = stats[
            (stats["league_id"].map(_to_int) == _to_int(league_id))
            & (stats["competition"].map(normalize_competition) == competition)
            & (stats["_year"] == season)
        ].copy()
        if pool.empty:
            raise ValueError("No player scouting pool exists for the requested league, season and competition")

        # Resolve duplicate team rows to one candidate observation per player.
        sort_cols = [column for column in ("games_played", "minutes_per_game", "rating") if column in pool.columns]
        if sort_cols:
            pool = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        pool = pool.drop_duplicates(subset=["player_id"], keep="first").copy()

        requested_position = str(parameters.get("role", "") or "").strip().upper()
        nationality = str(parameters.get("nationality", "") or "").strip()
        age_range = parameters.get("age_range")
        min_age = parameters.get("min_age")
        max_age = parameters.get("max_age")
        if isinstance(age_range, (list, tuple)) and len(age_range) == 2:
            min_age = age_range[0] if min_age is None else min_age
            max_age = age_range[1] if max_age is None else max_age
        min_games = int(parameters.get("min_games", 5))
        min_minutes = float(parameters.get("min_minutes_per_game", 0.0))
        if min_games < 0 or min_games > 82 or min_minutes < 0 or min_minutes > 48:
            raise ValueError("min_games must be 0..82 and min_minutes_per_game must be 0..48")
        if min_age is not None and not 16 <= int(min_age) <= 50:
            raise ValueError("min_age must be between 16 and 50")
        if max_age is not None and not 16 <= int(max_age) <= 50:
            raise ValueError("max_age must be between 16 and 50")
        if min_age is not None and max_age is not None and int(min_age) > int(max_age):
            raise ValueError("min_age cannot exceed max_age")

        metadata = self.data.get("player_dict", {}) or {}
        accepted = []
        excluded_missing_identity = 0
        for row in pool.to_dict("records"):
            pid = _to_int(row.get("player_id"))
            meta = metadata.get(pid, {}) or {}
            position = str(meta.get("position", meta.get("ruolo", row.get("position", ""))) or "").upper()
            if requested_position and requested_position not in {part.strip() for part in position.replace("-", "/").split("/")}:
                continue
            player_nationality_raw = str(meta.get("nationality", meta.get("nationality_code", "")) or "").strip()
            player_nationality = normalize_nationality(player_nationality_raw)
            if nationality:
                if not player_nationality:
                    excluded_missing_identity += 1
                    continue
                if not nationality_matches(nationality, player_nationality_raw):
                    continue
            birth_date = meta.get("birth_date", meta.get("date_of_birth"))
            age = None
            try:
                age = season - int(str(birth_date)[:4])
            except (TypeError, ValueError):
                try:
                    age = int(meta.get("age"))
                except (TypeError, ValueError):
                    pass
            if min_age is not None or max_age is not None:
                if age is None:
                    excluded_missing_identity += 1
                    continue
                if min_age is not None and age < int(min_age):
                    continue
                if max_age is not None and age > int(max_age):
                    continue
            games = _finite(row.get("games_played"), 0)
            minutes = _finite(row.get("minutes_per_game"), 0)
            if games < min_games or minutes < min_minutes:
                continue
            accepted.append({**row, "_pid": pid, "_name": self._player_name(pid), "_position": position or None,
                             "_nationality": player_nationality or None, "_age": age})

        if not accepted:
            raise ValueError("No candidates match the scouting filters; verify role, nationality, age and minimum playing-time requirements")
        candidates = pd.DataFrame(accepted)
        metrics = archetype_metrics[archetype]
        weighted_metrics = [(metric, weight) for metric, weight in metrics.items() if metric in candidates.columns]
        if not weighted_metrics:
            raise ValueError("None of the metrics required by this scouting archetype are available")

        # Keep the individual scouting score and its metric evidence. When a team
        # is supplied, this score becomes one component of a separate roster-aware
        # score rather than being misrepresented as team fit by itself.
        percentile_values: dict[str, pd.Series] = {}
        for metric, weight in weighted_metrics:
            values = pd.to_numeric(candidates[metric], errors="coerce")
            if values.notna().sum() == 0:
                continue
            pct = values.rank(method="average", pct=True)
            if weight < 0:
                pct = 1.0 - pct
            percentile_values[metric] = pct
        if not percentile_values:
            raise ValueError("Scouting metrics are present but contain no usable values")
        scores = pd.Series(0.0, index=candidates.index)
        available_weight = pd.Series(0.0, index=candidates.index)
        metric_percentiles: dict[str, pd.Series] = {}
        for metric, weight in weighted_metrics:
            if metric not in percentile_values:
                continue
            pct = percentile_values[metric]
            magnitude = abs(weight)
            scores = scores.add(pct.fillna(0.0) * magnitude, fill_value=0.0)
            available_weight = available_weight.add(pct.notna().astype(float) * magnitude, fill_value=0.0)
            metric_percentiles[metric] = pct
        candidates["_scouting_fit_score"] = (
            scores / available_weight.replace(0, np.nan) * 100.0
        ).fillna(50.0).clip(0, 100)
        candidates["_fit_score"] = candidates["_scouting_fit_score"]

        target_team = self._team_name(teams[0]) if teams else None
        team_fit_status = "not_requested" if not teams else "unavailable"
        team_fit_reason = "Nessuna squadra target indicata." if not teams else ""
        roster_source = None
        roster_member_ids: set[int] = set()
        roster_stats = pd.DataFrame()
        roster_strengths: dict[str, float] = {}
        roster_needs: list[dict[str, Any]] = []
        position_depth: dict[str, int] = {}
        positional_needs: dict[str, float] = {}
        candidate_team_fit: dict[Any, dict[str, Any]] = {}
        total_complementarity_pairs = 0
        core_positions = ("PG", "SG", "SF", "PF", "C")

        def safe_season_year(value: Any) -> int | None:
            try:
                return _year(value)
            except (TypeError, ValueError, OverflowError):
                return None

        def position_tokens(value: Any) -> set[str]:
            raw = str(value or "").upper().replace("-", "/").replace(",", "/")
            tokens = {part.strip() for part in raw.split("/") if part.strip()}
            mapped: set[str] = set()
            for token in tokens:
                if token in core_positions:
                    mapped.add(token)
                elif token in {"G", "GUARD", "GUARDS"}:
                    mapped.update(("PG", "SG"))
                elif token in {"F", "FORWARD", "FORWARDS"}:
                    mapped.update(("SF", "PF"))
                elif token in {"C", "CENTER", "CENTRE"}:
                    mapped.add("C")
            return mapped

        def listed_position(player_id: int, fallback: Any = "") -> str:
            meta = metadata.get(_to_int(player_id), {}) or {}
            value = meta.get("position", meta.get("ruolo", fallback))
            return str(value or "").strip().upper()

        def unique_player_rows(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or "player_id" not in frame.columns:
                return frame.iloc[0:0].copy()
            ordered = frame.copy()
            order_columns = [
                column for column in ("games_played", "minutes_per_game", "rating")
                if column in ordered.columns
            ]
            if order_columns:
                ordered = ordered.sort_values(
                    order_columns, ascending=[False] * len(order_columns), na_position="last"
                )
            return ordered.drop_duplicates(subset=["player_id"], keep="first").copy()

        # Prefer the season-specific roster relation. PlayerCompetitionStats
        # supplies the same-context performance observations used in the fit.
        # If no season relation exists, fall back to players with observed stats
        # explicitly attached to this team; the response declares that fallback.
        if teams:
            team_id = _to_int(teams[0])
            raw_stats = self.data.get("player_stats", pd.DataFrame()).copy()
            context_stats = pd.DataFrame()
            exact_team_stats = pd.DataFrame()
            if not raw_stats.empty and {"player_id", "league_id", "season", "competition"}.issubset(raw_stats.columns):
                raw_stats["_year"] = raw_stats["season"].map(safe_season_year)
                stats_mask = (
                    (raw_stats["league_id"].map(_to_int) == _to_int(league_id))
                    & (raw_stats["competition"].map(normalize_competition) == competition)
                    & (raw_stats["_year"] == season)
                )
                context_stats = raw_stats[stats_mask].copy()
                if "team_id" in raw_stats.columns:
                    exact_team_stats = context_stats[
                        context_stats["team_id"].map(_to_int) == team_id
                    ].copy()

            relations = self.data.get("team_player_relations", pd.DataFrame())
            if (
                team_id is not None
                and isinstance(relations, pd.DataFrame)
                and not relations.empty
                and {"player_id", "team_id", "season"}.issubset(relations.columns)
            ):
                relation_rows = relations.copy()
                relation_rows["_year"] = relation_rows["season"].map(safe_season_year)
                relation_rows = relation_rows[
                    (relation_rows["team_id"].map(_to_int) == team_id)
                    & (relation_rows["_year"] == season)
                ]
                if not relation_rows.empty:
                    roster_member_ids.update(
                        pid for pid in relation_rows["player_id"].map(_to_int).tolist()
                        if pid is not None
                    )
                    roster_source = "season_roster_relations"

            observed_team_ids = {
                pid for pid in exact_team_stats.get("player_id", pd.Series(dtype=object)).map(_to_int).tolist()
                if pid is not None
            }
            if observed_team_ids:
                roster_member_ids.update(observed_team_ids)
                roster_source = (
                    "season_roster_relations_plus_observed_stats"
                    if roster_source == "season_roster_relations"
                    else "observed_player_competition_stats"
                )
            if roster_member_ids and not context_stats.empty:
                roster_stats = unique_player_rows(
                    context_stats[context_stats["player_id"].map(_to_int).isin(roster_member_ids)]
                )
                if "games_played" in roster_stats.columns:
                    appeared = pd.to_numeric(
                        roster_stats["games_played"], errors="coerce"
                    ).fillna(0) > 0
                    roster_stats = roster_stats[appeared].copy()
            if not roster_member_ids:
                team_fit_reason = "Il roster della squadra non è disponibile per la stagione richiesta."
            elif len(roster_stats) < 2:
                team_fit_reason = (
                    "Sono disponibili meno di due giocatori del roster con statistiche "
                    "nella stessa lega, stagione e competizione; il fit individuale resta disponibile."
                )
            else:
                has_authoritative_roster = roster_source in {
                    "season_roster_relations",
                    "season_roster_relations_plus_observed_stats",
                }
                has_full_statistics_coverage = len(roster_stats) >= len(roster_member_ids)
                if has_authoritative_roster and has_full_statistics_coverage:
                    team_fit_status = "available"
                else:
                    team_fit_status = "partial"
                    if not has_authoritative_roster:
                        team_fit_reason = (
                            "La relazione stagionale del roster non è disponibile: "
                            "il fit può usare soltanto i giocatori osservati nelle statistiche "
                            "della squadra e non può certificare la copertura dell'intero roster."
                        )
                    else:
                        team_fit_reason = (
                            f"Fit parziale: statistiche utilizzabili per {len(roster_stats)} "
                            f"di {len(roster_member_ids)} membri del roster."
                        )

        # Compute league-relative percentiles for every archetype, not just the
        # requested one. Those profiles identify which dimensions look weakest
        # in the current roster and whether a candidate can help address them.
        all_archetype_metrics = sorted({
            metric for weights in archetype_metrics.values() for metric in weights
        })
        metric_distributions: dict[str, np.ndarray] = {}
        for metric in all_archetype_metrics:
            if metric not in pool.columns:
                continue
            values = pd.to_numeric(pool[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                metric_distributions[metric] = np.sort(values)

        def metric_percentile(metric: str, value: Any) -> float | None:
            distribution = metric_distributions.get(metric)
            number = _finite(value, float("nan"))
            if distribution is None or distribution.size == 0 or not math.isfinite(number):
                return None
            left = int(np.searchsorted(distribution, number, side="left"))
            right = int(np.searchsorted(distribution, number, side="right"))
            return float((left + (right - left) / 2.0) / distribution.size)

        def archetype_strength(record: dict[str, Any], label: str) -> float | None:
            weighted_values: list[tuple[float, float]] = []
            for metric, weight in archetype_metrics[label].items():
                pct = metric_percentile(metric, record.get(metric))
                if pct is None:
                    continue
                if weight < 0:
                    pct = 1.0 - pct
                weighted_values.append((pct, abs(weight)))
            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight <= 0:
                return None
            return 100.0 * sum(value * weight for value, weight in weighted_values) / total_weight

        if team_fit_status in {"available", "partial"}:
            roster_records = roster_stats.to_dict("records")
            for label in archetype_metrics:
                values = [
                    strength for record in roster_records
                    if (strength := archetype_strength(record, label)) is not None
                ]
                if values:
                    roster_strengths[label] = float(np.mean(values))
            roster_needs = [
                {
                    "archetype": label,
                    "roster_strength_score": round(strength, 1),
                    "need_score": round(max(0.0, 100.0 - strength), 1),
                }
                for label, strength in roster_strengths.items()
            ]
            roster_needs.sort(key=lambda item: item["need_score"], reverse=True)

            for pid in roster_member_ids:
                pos = listed_position(pid)
                for token in position_tokens(pos):
                    position_depth[token] = position_depth.get(token, 0) + 1
            # Two listed players at a primary position is a transparent depth
            # heuristic, not a claim about an ideal rotation for every coach.
            positional_needs = {
                pos: round(max(0.0, (2.0 - position_depth.get(pos, 0)) / 2.0 * 100.0), 1)
                for pos in core_positions
            }

            def row_pair_score(first: dict[str, Any], second: dict[str, Any], first_pos: str, second_pos: str) -> float:
                first_usage = _ratio(first.get("usg_pct"), 0.18)
                second_usage = _ratio(second.get("usg_pct"), 0.18)
                usage_overlap = max(0.0, first_usage + second_usage - 0.50)
                first_three = _ratio(first.get("three_point_pct"), 0.33)
                second_three = _ratio(second.get("three_point_pct"), 0.33)
                spacing = float(np.clip((first_three + second_three) / 0.80, 0.0, 1.0))
                creation = float(np.clip(
                    (_finite(first.get("assists"), 3.0) / 10.0 + _finite(second.get("assists"), 3.0) / 10.0) / 1.20,
                    0.0, 1.0,
                ))
                defense = float(np.clip(
                    0.5 + (_finite(first.get("dbpm"), 0.0) / 6.0 + _finite(second.get("dbpm"), 0.0) / 6.0) / 2.0,
                    0.0, 1.0,
                ))
                first_primary = first_pos.split("/")[0].strip() if first_pos else ""
                second_primary = second_pos.split("/")[0].strip() if second_pos else ""
                positional_overlap = 1.0 if first_primary and first_primary == second_primary else 0.0
                return float(np.clip(
                    0.50 + spacing * 0.18 + creation * 0.14 + defense * 0.12
                    - usage_overlap * 0.42 - positional_overlap * 0.10,
                    0.0, 1.0,
                ))

            for candidate_index, candidate in candidates.iterrows():
                candidate_id = _to_int(candidate.get("_pid"))
                if candidate_id is None:
                    continue
                candidate_meta = metadata.get(candidate_id, {}) or {}
                candidate_global_id = str(candidate_meta.get("global_id", "") or "").strip()
                candidate_position = listed_position(candidate_id, candidate.get("_position"))
                pair_values: list[tuple[float, str]] = []
                for teammate in roster_records:
                    teammate_id = _to_int(teammate.get("player_id"))
                    if teammate_id is None:
                        continue
                    teammate_meta = metadata.get(teammate_id, {}) or {}
                    teammate_global_id = str(teammate_meta.get("global_id", "") or "").strip()
                    if teammate_id == candidate_id or (
                        candidate_global_id and teammate_global_id
                        and candidate_global_id == teammate_global_id
                    ):
                        continue
                    teammate_position = listed_position(teammate_id, teammate.get("position"))
                    pair_score = row_pair_score(
                        candidate.to_dict(), teammate, candidate_position, teammate_position
                    )
                    pair_values.append((pair_score, self._player_name(teammate_id)))
                pair_values.sort(key=lambda pair: pair[0])
                pair_count = len(pair_values)
                total_complementarity_pairs += pair_count
                complementarity_score = (
                    100.0 * float(np.mean([value for value, _ in pair_values]))
                    if pair_values else None
                )

                candidate_arch_scores = {
                    label: archetype_strength(candidate.to_dict(), label)
                    for label in archetype_metrics
                }
                need_terms = [
                    (max(0.0, 100.0 - strength), candidate_arch_scores.get(label))
                    for label, strength in roster_strengths.items()
                    if candidate_arch_scores.get(label) is not None
                    and max(0.0, 100.0 - strength) > 0
                ]
                need_weight = sum(weight for weight, _ in need_terms)
                need_alignment_score = (
                    sum(weight * float(value) for weight, value in need_terms) / need_weight
                    if need_weight > 0 else None
                )
                candidate_positions = position_tokens(candidate_position)
                position_values = [
                    positional_needs[pos] for pos in candidate_positions if pos in positional_needs
                ]
                position_need_score = float(np.mean(position_values)) if position_values else None

                score_components: list[tuple[float, float]] = [
                    (0.40, _finite(candidate.get("_scouting_fit_score"), 50.0))
                ]
                if complementarity_score is not None:
                    score_components.append((0.30, complementarity_score))
                if need_alignment_score is not None:
                    score_components.append((0.20, need_alignment_score))
                if position_need_score is not None:
                    score_components.append((0.10, position_need_score))
                component_weight = sum(weight for weight, _ in score_components)
                team_fit_score = (
                    sum(weight * value for weight, value in score_components) / component_weight
                    if component_weight > 0 and complementarity_score is not None else None
                )

                candidate_team_fit[candidate_index] = {
                    "team_fit_score": round(team_fit_score, 1) if team_fit_score is not None else None,
                    "complementarity_score": round(complementarity_score, 1) if complementarity_score is not None else None,
                    "complementarity_pairs": pair_count,
                    "need_alignment_score": round(need_alignment_score, 1) if need_alignment_score is not None else None,
                    "position_need_score": round(position_need_score, 1) if position_need_score is not None else None,
                    "best_roster_match": (
                        {"player": pair_values[-1][1], "score": round(pair_values[-1][0] * 100.0, 1)}
                        if pair_values else None
                    ),
                    "weakest_roster_match": (
                        {"player": pair_values[0][1], "score": round(pair_values[0][0] * 100.0, 1)}
                        if pair_values else None
                    ),
                }

            candidates["_team_fit_score"] = pd.Series(
                {index: values["team_fit_score"] for index, values in candidate_team_fit.items()},
                dtype=float,
            ).reindex(candidates.index)
            candidates["_fit_score"] = candidates["_team_fit_score"].where(
                candidates["_team_fit_score"].notna(), candidates["_scouting_fit_score"]
            )

        if not teams:
            team_fit_payload: dict[str, Any] = {
                "status": "not_requested",
                "team": None,
                "reason": "Indicare una squadra per valutare il roster e la complementarità.",
            }
        elif team_fit_status not in {"available", "partial"}:
            team_fit_payload = {
                "status": "unavailable",
                "team": target_team,
                "season": season,
                "competition": competition,
                "roster_source": roster_source,
                "roster_player_count": len(roster_member_ids),
                "roster_players_with_stats": int(len(roster_stats)),
                "reason": team_fit_reason or "Dati roster insufficienti nel contesto richiesto.",
            }
        else:
            roster_player_names = [
                self._player_name(pid) for pid in sorted(roster_member_ids)
            ]
            members_with_stats = {
                _to_int(pid) for pid in roster_stats.get("player_id", pd.Series(dtype=object)).map(_to_int)
                if _to_int(pid) is not None
            }
            team_fit_payload = {
                "status": team_fit_status,
                "completeness": "full" if team_fit_status == "available" else "partial",
                "team": target_team,
                "league": self._league_name(league_id),
                "season": season,
                "competition": competition,
                "method": "roster_pairwise_complementarity_plus_archetype_and_position_needs",
                "roster_source": roster_source,
                "roster_player_count": len(roster_member_ids),
                "roster_players_with_stats": int(len(roster_stats)),
                "roster_stats_coverage": {
                    "registered_members_in_resolved_scope": int(len(roster_member_ids)),
                    "members_with_usable_stats": int(len(roster_stats)),
                    "members_without_usable_stats": int(max(0, len(roster_member_ids) - len(roster_stats))),
                    "statistics_coverage_pct": round(
                        100.0 * len(roster_stats) / len(roster_member_ids), 1
                    ) if roster_member_ids else 0.0,
                    "membership_source_is_authoritative": roster_source in {
                        "season_roster_relations",
                        "season_roster_relations_plus_observed_stats",
                    },
                    "complete": team_fit_status == "available",
                },
                "completeness_note": (
                    "Copertura roster completa nel contesto richiesto."
                    if team_fit_status == "available"
                    else team_fit_reason
                ),
                "roster_members_without_stats": [
                    self._player_name(pid) for pid in sorted(roster_member_ids - members_with_stats)[:10]
                ],
                "roster_players": roster_player_names[:15],
                "archetype_strengths": {
                    label: round(value, 1) for label, value in roster_strengths.items()
                },
                "roster_needs": roster_needs[:3],
                "position_depth": {pos: int(position_depth.get(pos, 0)) for pos in core_positions},
                "positional_needs": [
                    {"position": pos, "need_score": score}
                    for pos, score in sorted(positional_needs.items(), key=lambda item: item[1], reverse=True)[:3]
                ],
                "positional_depth_target": 2,
                "complementarity_pairs_evaluated": int(total_complementarity_pairs),
                "score_components": {
                    "individual_scouting": 0.40,
                    "roster_complementarity": 0.30,
                    "roster_need_alignment": 0.20,
                    "positional_depth": 0.10,
                },
                "score_note": (
                    "Punteggio analitico 0-100; pesi normalizzati se un componente manca. "
                    "Non è una probabilità né una previsione ML calibrata."
                ),
            }

        ranked = candidates.sort_values(["_fit_score", "rating"], ascending=[False, False]).head(int(spec.get("top_n", 5)))
        acquisition_impact_model: dict[str, Any] | None = None
        acquisition_impact_summary: dict[str, Any] = {
            "status": "not_requested" if not teams else "unavailable",
            "reason": "Nessuna squadra target indicata." if not teams else "Modello d'impatto non valutato.",
            "training_samples": 0,
            "training_seasons": 0,
            "validation": {"samples": 0, "seasons": 0},
        }
        candidate_acquisition_impacts: dict[Any, dict[str, Any]] = {}
        if teams:
            team_id = _to_int(teams[0])
            if team_id is not None:
                acquisition_impact_model = fit_acquisition_impact_model(
                    self.data, int(league_id), competition, through_season=season
                )
                acquisition_impact_summary = {
                    "status": str(acquisition_impact_model.get("status", "uncalibrated")),
                    "reason": str(acquisition_impact_model.get("reason", "")),
                    "method": str(acquisition_impact_model.get("method", "")),
                    "training_samples": int(acquisition_impact_model.get("training_samples", 0)),
                    "training_seasons": int(acquisition_impact_model.get("training_seasons", 0)),
                    "validation": acquisition_impact_model.get("validation", {}),
                    "numeric_estimates_allowed": acquisition_impact_model.get("status") == "validated",
                }
                rotation = self._rotation_ratings(team_id, int(league_id), competition, season)
                top_rotation = rotation[:8]
                if acquisition_impact_model.get("status") == "validated" and len(top_rotation) >= 5:
                    replacement = min(top_rotation, key=lambda item: item[1])
                    replacement_rating = float(replacement[1])
                    rotation_slots = max(1, len(top_rotation))
                    roster_size = self._team_roster_size(
                        team_id, int(league_id), competition, season
                    ) or len({pid for pid, _ in rotation})
                    global_id_by_player: dict[int, str] = {}
                    players_frame = self.data.get("players", pd.DataFrame())
                    if isinstance(players_frame, pd.DataFrame) and {"id", "global_id"}.issubset(players_frame.columns):
                        for identity_row in players_frame[["id", "global_id"]].to_dict("records"):
                            identity_pid = _to_int(identity_row.get("id"))
                            raw_global_id = identity_row.get("global_id")
                            global_id = "" if raw_global_id is None or pd.isna(raw_global_id) else str(raw_global_id).strip()
                            if global_id.lower() in {"nan", "none", "null"}:
                                global_id = ""
                            if identity_pid is not None and global_id:
                                global_id_by_player[identity_pid] = global_id
                    for identity_pid, identity_meta in metadata.items():
                        parsed_pid = _to_int(identity_pid)
                        if parsed_pid is None or parsed_pid in global_id_by_player or not isinstance(identity_meta, dict):
                            continue
                        raw_global_id = identity_meta.get("global_id")
                        global_id = "" if raw_global_id is None or pd.isna(raw_global_id) else str(raw_global_id).strip()
                        if global_id.lower() in {"nan", "none", "null"}:
                            global_id = ""
                        if global_id:
                            global_id_by_player[parsed_pid] = global_id
                    roster_global_ids = {
                        global_id_by_player[pid]
                        for raw_pid in roster_member_ids
                        if (pid := _to_int(raw_pid)) is not None and pid in global_id_by_player
                    }
                    for candidate_index, candidate in ranked.iterrows():
                        candidate_id = _to_int(candidate.get("_pid"))
                        if candidate_id is None:
                            candidate_acquisition_impacts[candidate_index] = {
                                "status": "candidate_identity_unavailable",
                                "estimated_incremental_net_rating": None,
                            }
                            continue
                        candidate_global_id = global_id_by_player.get(candidate_id, "")
                        if candidate_id in roster_member_ids or (
                            candidate_global_id and candidate_global_id in roster_global_ids
                        ):
                            candidate_acquisition_impacts[candidate_index] = {
                                "status": "already_on_roster",
                                "estimated_incremental_net_rating": None,
                                "reason": "Il giocatore risulta già nel roster osservato della squadra.",
                            }
                            continue
                        try:
                            projection = self._project_player_for_team(
                                spec, candidate_id, team_id, int(league_id), int(league_id)
                            )
                            projected_rating = float(projection["rating"])
                            quality_delta = (projected_rating - replacement_rating) / rotation_slots
                            forecast = self._team_acquisition_forecast(
                                team_id, int(league_id), competition, season,
                                rotation_quality_delta=quality_delta,
                                roster_size=roster_size,
                                impact_model=acquisition_impact_model,
                            )
                            impact = forecast.get("impact", {}) if isinstance(forecast.get("impact"), dict) else {}
                            candidate_acquisition_impacts[candidate_index] = {
                                "status": str(impact.get("status", forecast.get("status", "unavailable"))),
                                "estimated_incremental_net_rating": impact.get("estimated_incremental_net_rating") if impact.get("available") else None,
                                "incremental_net_rating_interval_90": impact.get("incremental_net_rating_interval_90") if impact.get("available") else None,
                                "predicted_team_net_rating_change_without_acquisition": impact.get("predicted_team_net_rating_change_without_acquisition") if impact.get("available") else None,
                                "predicted_team_net_rating_change_with_acquisition": impact.get("predicted_team_net_rating_change_with_acquisition") if impact.get("available") else None,
                                "predicted_team_net_rating_change_interval_80": impact.get("predicted_team_net_rating_change_interval_80") if impact.get("available") else None,
                                "target_season": int(season) + 1,
                                "validation": acquisition_impact_summary["validation"],
                            }
                        except (ValueError, RuntimeError, TypeError, KeyError):
                            candidate_acquisition_impacts[candidate_index] = {
                                "status": "candidate_projection_unavailable",
                                "estimated_incremental_net_rating": None,
                                "reason": "Dati previsionali insufficienti per valutare il candidato nel contesto target.",
                            }
                elif acquisition_impact_model.get("status") == "validated":
                    acquisition_impact_summary["status"] = "team_rotation_unavailable"
                    acquisition_impact_summary["reason"] = (
                        "Il modello supera il backtest, ma la rotazione osservata della squadra "
                        "non contiene almeno cinque valutazioni utilizzabili."
                    )
            if isinstance(team_fit_payload, dict):
                team_fit_payload["acquisition_impact_model"] = acquisition_impact_summary

        impact_rankable = [
            index for index, item in candidate_acquisition_impacts.items()
            if item.get("status") == "validated"
            and item.get("estimated_incremental_net_rating") is not None
        ]
        if impact_rankable:
            ranked = ranked.copy()
            ranked["_acquisition_impact_sort"] = ranked.index.map(
                lambda index: _finite(
                    candidate_acquisition_impacts.get(index, {}).get("estimated_incremental_net_rating"),
                    float("-inf"),
                )
            )
            ranked = ranked.sort_values(
                ["_acquisition_impact_sort", "_fit_score", "rating"],
                ascending=[False, False, False],
                na_position="last",
            ).drop(columns=["_acquisition_impact_sort"])
            ranking_basis = "team_fit_plus_acquisition_impact"
        else:
            ranking_basis = (
                "team_fit" if team_fit_status == "available"
                else "team_fit_partial" if team_fit_status == "partial"
                else "individual_scouting"
            )

        results = []
        for _, row in ranked.iterrows():
            strengths = []
            evidence_metrics = {}
            for metric, pct in metric_percentiles.items():
                value = pct.loc[row.name]
                if pd.isna(value):
                    continue
                evidence_metrics[metric] = {"value": round(_finite(row.get(metric)), 4), "percentile": round(float(value), 3)}
                if value >= .75:
                    strengths.append(metric)
            team_detail = candidate_team_fit.get(row.name, {})
            individual_score = _finite(row.get("_scouting_fit_score"), 50.0)
            archetype_scores = []
            for label in archetype_metrics:
                score = archetype_strength(row.to_dict(), label)
                if score is None:
                    continue
                supported_metrics = sum(
                    1 for metric in archetype_metrics[label]
                    if metric in metric_distributions
                    and _finite(row.get(metric), float("nan")) == _finite(row.get(metric), float("nan"))
                )
                expected_metrics = len(archetype_metrics[label])
                archetype_scores.append({
                    "archetype": label,
                    "score": round(score, 1),
                    "evidence_metrics": supported_metrics,
                    "metrics_defined": expected_metrics,
                    "evidence_coverage_pct": round(100.0 * supported_metrics / expected_metrics, 1) if expected_metrics else 0.0,
                    "confidence": "high" if supported_metrics >= 4 else "medium" if supported_metrics >= 2 else "low",
                })
            archetype_scores.sort(key=lambda item: item["score"], reverse=True)
            primary_archetypes = [
                item["archetype"] for item in archetype_scores
                if item["score"] >= 65.0
            ][:3]
            results.append({
                "player": row["_name"],
                "fit_score": round(_finite(row.get("_fit_score"), individual_score), 1),
                "scouting_fit_score": round(individual_score, 1),
                "team_fit_score": team_detail.get("team_fit_score"),
                "complementarity_score": team_detail.get("complementarity_score"),
                "complementarity_pairs": int(team_detail.get("complementarity_pairs", 0)),
                "need_alignment_score": team_detail.get("need_alignment_score"),
                "position_need_score": team_detail.get("position_need_score"),
                "best_roster_match": team_detail.get("best_roster_match"),
                "weakest_roster_match": team_detail.get("weakest_roster_match"),
                "acquisition_impact": candidate_acquisition_impacts.get(row.name, {
                    "status": str(acquisition_impact_summary.get("status", "unavailable")),
                    "estimated_incremental_net_rating": None,
                    "reason": str(acquisition_impact_summary.get("reason", "")),
                }) if teams else None,
                "position": row["_position"],
                "nationality": row["_nationality"],
                "age": int(row["_age"]) if row["_age"] is not None and pd.notna(row["_age"]) else None,
                "bb_rating_source_score": round(_finite(row.get("rating")), 3) if pd.notna(row.get("rating")) else None,
                "games_played": int(_finite(row.get("games_played"))),
                "minutes_per_game": round(_finite(row.get("minutes_per_game")), 1),
                "strengths": strengths[:5],
                "archetype_profiles": archetype_scores[:8],
                "primary_archetypes": primary_archetypes,
                "archetype_evidence": "league_season_competition_percentiles",
                "metrics": evidence_metrics,
            })
        limitations = [
            "Il punteggio individuale per archetipo è un ranking percentile trasparente, non una probabilità calibrata né il BB-Rating.",
            "Nazionalità ed età sono filtri rigidi solo quando i relativi dati d'identità sono disponibili; i record con dato mancante vengono esclusi.",
        ]
        if team_fit_status in {"available", "partial"}:
            limitations.extend([
                "Il Team Fit usa complementarità analitica candidato-compagni, lacune di archetipi e una stima indicativa della profondità per ruolo; non è un modello ML calibrato.",
                "La complementarità confronta il candidato con tutti i membri risolti del roster che dispongono di statistiche utilizzabili nella stessa lega, stagione e competizione.",
                "Il target di profondità di due giocatori per ruolo è euristico e non descrive minuti, sistema tattico o preferenze dello staff.",
            ])
            if team_fit_status == "partial":
                limitations.append(
                    "Team Fit parziale: la classifica usa solo la porzione di roster con evidenze utilizzabili "
                    "e non va interpretata come valutazione completa della squadra."
                )
            if roster_source == "observed_player_competition_stats":
                limitations.append(
                    "Il roster proviene dalle statistiche osservate della squadra perché non è disponibile una relazione stagionale completa."
                )
        else:
            limitations.append(
                "La squadra definisce il contesto, ma il Team Fit non è stato calcolato perché i dati di roster/statistiche non sono sufficienti."
            )

        if teams:
            validated_estimate_count = sum(
                1 for item in candidate_acquisition_impacts.values()
                if item.get("status") == "validated"
                and item.get("estimated_incremental_net_rating") is not None
            )
            if validated_estimate_count > 0:
                limitations.append(
                    "Per alcuni candidati della shortlist è disponibile una stima osservazionale dell'impatto incrementale sul Net Rating della stagione successiva. "
                    "La shortlist è pre-selezionata per Team Fit e ordinata per impatto soltanto tra i candidati con stima valida; non è una prova causale del singolo acquisto."
                )
            elif acquisition_impact_summary.get("status") == "validated":
                limitations.append(
                    "Il modello storico supera il backtest, ma nessun candidato in questa shortlist dispone di una stima d'acquisizione utilizzabile; l'ordinamento resta basato sul Team Fit."
                )
            else:
                limitations.append(
                    "La previsione dell'impatto d'acquisto non ha superato i requisiti di validazione per questo contesto; i campi numerici restano null e il ranking usa soltanto lo score analitico."
                )

        evidence = [
            {"type": "same_league_season_competition_pool", "count": int(len(pool))},
            {"type": "eligible_candidates", "count": int(len(accepted))},
            {"type": "missing_identity_exclusions", "count": excluded_missing_identity},
        ]
        if teams:
            evidence.append({"type": "team_roster_members", "count": int(len(roster_member_ids))})
            evidence.append({"type": "roster_members_with_statistics", "count": int(len(roster_stats))})
            evidence.append({
                "type": "validated_acquisition_impact_estimates",
                "count": sum(
                    1 for item in candidate_acquisition_impacts.values()
                    if item.get("status") == "validated" and item.get("estimated_incremental_net_rating") is not None
                ),
            })
            if team_fit_status in {"available", "partial"}:
                evidence.append({"type": "roster_complementarity_pairs", "count": int(total_complementarity_pairs)})
        return {
            "result": {
                "archetype": archetype,
                "role_filter": requested_position or None,
                "nationality_filter": parameters.get("nationality") or None,
                "age_range": {"min": int(min_age) if min_age is not None else None, "max": int(max_age) if max_age is not None else None},
                "team_context": target_team,
                "team_fit": team_fit_payload,
                "ranking_basis": ranking_basis,
                "acquisition_impact_model": acquisition_impact_summary,
                "acquisition_impact_note": (
                    "La shortlist viene pre-selezionata dal Team Fit e poi ordinata per impatto incrementale stimato fra i candidati valutati."
                    if ranking_basis == "team_fit_plus_acquisition_impact"
                    else "L'impatto d'acquisto non è disponibile o non ha superato la validazione; il ranking usa soltanto lo score di scouting/Team Fit."
                ),
                "league": self._league_name(league_id),
                "season": season,
                "competition": competition,
                "candidates": results,
                "candidates_considered": len(accepted),
            },
            "evidence": evidence,
            "support": {
                "method": (
                    "roster_aware_scouting_fit_plus_walk_forward_acquisition_impact"
                    if ranking_basis == "team_fit_plus_acquisition_impact"
                    else "roster_aware_scouting_fit" if team_fit_status == "available"
                    else "partial_roster_aware_scouting_fit" if team_fit_status == "partial"
                    else "transparent_weighted_percentile_scouting"
                ),
                "samples": len(accepted),
                "confidence": _confidence(len(roster_stats)) if team_fit_status in {"available", "partial"} else _confidence(len(accepted)),
            },
            "limitations": limitations,
        }

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
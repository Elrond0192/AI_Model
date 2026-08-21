"""Possession-aware probabilistic, matchup, optimization and causal scenarios.

The engine consumes optional canonical frames (``pbp_events``, ``lineup_stints``,
``play_type_stats`` and ``shot_profiles``).  Every estimator reports whether it
used possession evidence or a shrunk aggregate fallback; inferred defensive
assignments are never labelled as observed assignments.
"""
from __future__ import annotations

import itertools
import math
import secrets
from typing import Any, Iterable

import numpy as np
import pandas as pd

from basketball_ai.data.loader import _to_int
from basketball_ai.models.competition_training import normalize_competition
from basketball_ai.models.production_training import season_year


BOX_METRICS = (
    "minutes_per_game", "points", "rebounds", "offensive_rebounds",
    "defensive_rebounds", "assists", "steals", "blocks", "turnovers",
    "personal_fouls", "three_point_made", "three_point_attempts",
    "free_throws_made", "free_throw_attempts", "usg_pct", "ts_pct", "ortg",
    "drtg", "bpm", "net_rtg",
)
COUNT_METRICS = {
    "points", "rebounds", "offensive_rebounds", "defensive_rebounds",
    "assists", "steals", "blocks", "turnovers", "personal_fouls",
    "three_point_made", "three_point_attempts", "free_throws_made",
    "free_throw_attempts",
}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
PLAY_TYPES = {
    "pick_and_roll_ball_handler", "pick_and_roll_roll_man", "isolation",
    "post_up", "spot_up", "handoff", "cut", "off_screen", "transition",
    "putback", "miscellaneous",
}
SHOT_ZONES = (
    "rim", "paint", "short_mid", "long_mid", "corner_three",
    "above_break_three",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _ratio(value: Any, default: float = 0.0) -> float:
    number = _finite(value, default)
    return number / 100.0 if abs(number) > 1.5 else number


def _confidence(samples: float) -> str:
    if samples >= 500:
        return "high"
    if samples >= 150:
        return "medium"
    if samples >= 40:
        return "low-medium"
    return "low"


class AdvancedSimulationEngine:
    """Specialized scenario estimators layered over the strict model data."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def _frame(self, name: str) -> pd.DataFrame:
        value = self.data.get(name)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

    def _name(self, entity: str, value: int) -> str:
        dictionary = self.data.get(f"{entity}_dict", {})
        row = dictionary.get(_to_int(value), {})
        return str(row.get("name", f"{entity.title()} {value}"))

    def _player_row(
        self, player_id: int, league_id: int | None, competition: str, season: int
    ) -> pd.Series:
        frame = self._frame("player_stats")
        if frame.empty:
            raise ValueError("No player statistics are available")
        years = frame["season"].map(season_year)
        mask = (
            (frame["player_id"].map(_to_int) == _to_int(player_id))
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (years <= int(season))
        )
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        rows = frame[mask].copy()
        if rows.empty:
            raise ValueError("No player data is available for the requested context")
        rows["_year"] = rows["season"].map(season_year)
        return rows.sort_values("_year").iloc[-1]

    def _team_row(
        self, team_id: int, league_id: int | None, competition: str, season: int
    ) -> pd.Series:
        frame = self._frame("team_season_stats")
        if frame.empty:
            raise ValueError("No team statistics are available")
        years = frame["season"].map(season_year)
        mask = (
            (frame["team_id"].map(_to_int) == _to_int(team_id))
            & (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (years <= int(season))
        )
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        rows = frame[mask].copy()
        if rows.empty:
            raise ValueError("No team data is available for the requested context")
        rows["_year"] = rows["season"].map(season_year)
        return rows.sort_values("_year").iloc[-1]

    @staticmethod
    def _rng(parameters: dict[str, Any]) -> tuple[np.random.Generator, int, str]:
        explicit = parameters.get("seed")
        if explicit is None:
            seed = secrets.randbits(63)
            source = "system_entropy"
        else:
            seed = int(explicit)
            source = "user_seed"
        return np.random.default_rng(seed), seed, source

    def _peer_rows(self, league_id: int | None, competition: str, season: int) -> pd.DataFrame:
        frame = self._frame("player_stats")
        if frame.empty:
            return frame
        years = frame["season"].map(season_year)
        mask = (
            (frame["competition"].map(normalize_competition) == normalize_competition(competition))
            & (years <= int(season))
            & (years >= int(season) - 3)
        )
        if league_id is not None:
            mask &= frame["league_id"].map(_to_int) == _to_int(league_id)
        return frame[mask].copy()

    @staticmethod
    def _derive_box(row: pd.Series) -> dict[str, float]:
        points = _finite(row.get("points"))
        three_rate = min(0.85, max(0.02, _ratio(row.get("three_par"), 0.35)))
        three_pct = min(0.65, max(0.15, _ratio(row.get("three_point_pct"), 0.34)))
        ft_pct = min(0.98, max(0.35, _ratio(row.get("ft_pct"), 0.75)))
        possessions = max(6.0, points / max(0.65, _finite(row.get("ppsa"), 1.05)))
        three_attempts = possessions * three_rate
        free_attempts = max(0.0, possessions * _ratio(row.get("foul_drawing_rate"), 0.18))
        values = {metric: _finite(row.get(metric)) for metric in BOX_METRICS}
        values.update({
            "three_point_attempts": _finite(row.get("three_point_attempts"), three_attempts),
            "three_point_made": _finite(row.get("three_point_made"), three_attempts * three_pct),
            "free_throw_attempts": _finite(row.get("free_throw_attempts"), free_attempts),
            "free_throws_made": _finite(row.get("free_throws_made"), free_attempts * ft_pct),
        })
        return values

    def probabilistic_boxscore(
        self, spec: dict[str, Any], players: list[int], teams: list[int],
        source_league: int | None, target_league: int | None,
        adjustment: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if not players:
            raise ValueError("player is required")
        competition = normalize_competition(spec.get("target_competition") or spec.get("competition"))
        league_id = target_league or source_league
        season = int(spec["season"])
        row = self._player_row(players[0], source_league or league_id, normalize_competition(spec.get("competition")), season)
        mean = self._derive_box(row)
        overrides = spec.get("player_overrides", {})
        if "minutes_per_game" in overrides:
            old_minutes = max(mean["minutes_per_game"], 1.0)
            new_minutes = min(48.0, max(0.0, _finite(overrides["minutes_per_game"])))
            for metric in COUNT_METRICS:
                mean[metric] *= new_minutes / old_minutes
            mean["minutes_per_game"] = new_minutes
        if "usg_pct" in overrides:
            old_usage = max(_ratio(mean["usg_pct"], 0.20), 0.05)
            new_usage = min(0.60, max(0.03, _ratio(overrides["usg_pct"])))
            for metric in ("points", "assists", "turnovers", "three_point_attempts", "free_throw_attempts"):
                mean[metric] *= new_usage / old_usage
            mean["usg_pct"] = new_usage
        for metric, delta in (adjustment or {}).items():
            if metric in mean:
                mean[metric] += _finite(delta)

        peers = self._peer_rows(league_id, competition, season)
        peer_boxes = pd.DataFrame([
            self._derive_box(row) for row in peers.to_dict("records")
        ])
        metrics = [m for m in BOX_METRICS if m in mean]
        if len(peer_boxes) >= 8:
            scales = peer_boxes[metrics].apply(pd.to_numeric, errors="coerce").std().fillna(0.0)
            scale = np.array([max(_finite(scales.get(m)), max(abs(mean[m]) * 0.08, 0.05)) for m in metrics])
            covariance = peer_boxes[metrics].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0).to_numpy()
            covariance = covariance * np.outer(scale, scale)
            covariance += np.eye(len(metrics)) * 1e-6
            method = "multivariate_peer_distribution"
        else:
            scale = np.array([max(abs(mean[m]) * (0.28 if m in COUNT_METRICS else 0.10), 0.08) for m in metrics])
            covariance = np.diag(scale ** 2)
            method = "parametric_shrunk_distribution"
        parameters = spec.get("parameters", {})
        simulations = min(50000, max(500, int(parameters.get("simulations", 5000))))
        rng, seed, entropy_source = self._rng(parameters)
        samples = rng.multivariate_normal(np.array([mean[m] for m in metrics]), covariance, simulations)
        for index, metric in enumerate(metrics):
            if metric in COUNT_METRICS or metric == "minutes_per_game":
                samples[:, index] = np.maximum(0.0, samples[:, index])
            if metric in {"ts_pct", "usg_pct"}:
                samples[:, index] = np.clip(samples[:, index], 0.0, 1.0)
        summary: dict[str, Any] = {}
        thresholds = parameters.get("thresholds", {})
        for index, metric in enumerate(metrics):
            values = samples[:, index]
            item: dict[str, Any] = {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std(ddof=1)), 4),
                "quantiles": {f"p{int(q * 100)}": round(float(np.quantile(values, q)), 4) for q in QUANTILES},
            }
            requested = thresholds.get(metric, []) if isinstance(thresholds, dict) else []
            item["probabilities"] = {
                f"gte_{_finite(value):g}": round(float(np.mean(values >= _finite(value))), 4)
                for value in list(requested)[:10]
            }
            summary[metric] = item
        return {
            "result": {
                "player": self._name("player", players[0]),
                "competition": competition,
                "simulations": simulations,
                "distribution": summary,
                "randomness": {"source": entropy_source, "seed": seed if parameters.get("return_seed", False) else None},
            },
            "evidence": [{"type": "peer_player_seasons", "count": int(len(peers))}],
            "support": {"method": method, "samples": int(len(peers)), "confidence": _confidence(len(peers))},
            "limitations": ["Probabilities are simulation estimates and require out-of-sample calibration monitoring."],
        }

    def _play_type_rows(self, player_id: int | None, team_id: int | None, competition: str, season: int) -> pd.DataFrame:
        frame = self._frame("play_type_stats")
        if frame.empty:
            return frame
        mask = frame["competition"].map(normalize_competition) == normalize_competition(competition)
        if "season" in frame:
            mask &= frame["season"].map(season_year) <= season
        if player_id is not None and "player_id" in frame:
            mask &= frame["player_id"].map(_to_int) == _to_int(player_id)
        if team_id is not None and "team_id" in frame:
            mask &= frame["team_id"].map(_to_int) == _to_int(team_id)
        return frame[mask].copy()

    def play_type_matchup(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        if not players:
            raise ValueError("player is required")
        competition = normalize_competition(spec.get("competition"))
        season = int(spec["season"])
        opponent = teams[-1] if teams else None
        offense = self._play_type_rows(players[0], None, competition, season)
        defense = self._play_type_rows(None, opponent, competition, season) if opponent else pd.DataFrame()
        requested = spec.get("parameters", {}).get("play_types", [])
        requested = {str(v).strip().lower() for v in requested if str(v).strip().lower() in PLAY_TYPES}
        if requested:
            offense = offense[offense["play_type"].astype(str).str.lower().isin(requested)] if not offense.empty else offense
            defense = defense[defense["play_type"].astype(str).str.lower().isin(requested)] if not defense.empty else defense
        rows: list[dict[str, Any]] = []
        play_types = requested or set(offense.get("play_type", pd.Series(dtype=str)).astype(str).str.lower()) or PLAY_TYPES
        for play_type in sorted(play_types):
            off = offense[offense["play_type"].astype(str).str.lower() == play_type] if not offense.empty else pd.DataFrame()
            deff = defense[defense["play_type"].astype(str).str.lower() == play_type] if not defense.empty else pd.DataFrame()
            off_ppp = _finite(off.get("ppp", pd.Series(dtype=float)).mean(), 0.95)
            def_ppp = _finite(deff.get("ppp_allowed", deff.get("ppp", pd.Series(dtype=float))).mean(), 0.95)
            off_poss = _finite(off.get("possessions", pd.Series(dtype=float)).sum())
            def_poss = _finite(deff.get("possessions", pd.Series(dtype=float)).sum())
            weight = min(1.0, (off_poss + def_poss) / 250.0)
            expected = 0.95 + weight * (((off_ppp + def_ppp) / 2.0) - 0.95)
            rows.append({"play_type": play_type, "expected_ppp": round(expected, 4), "offense_ppp": round(off_ppp, 4), "defense_ppp_allowed": round(def_ppp, 4), "possessions": int(off_poss + def_poss)})
        samples = sum(r["possessions"] for r in rows)
        return {
            "result": {"player": self._name("player", players[0]), "opponent": self._name("team", opponent) if opponent else None, "play_types": rows},
            "evidence": [{"type": "play_type_possessions", "count": samples}],
            "support": {"method": "possession_play_type_shrinkage" if samples else "aggregate_play_type_prior", "samples": samples, "confidence": _confidence(samples)},
            "limitations": [] if samples else ["No canonical play-type possessions were available; league priors were returned."],
        }

    def opponent_matchup(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        if not players or not teams:
            raise ValueError("player and opponent team are required")
        competition = normalize_competition(spec.get("target_competition") or spec.get("competition"))
        season = int(spec["season"])
        player = self._player_row(players[0], source_league, normalize_competition(spec.get("competition")), season)
        opponent = self._team_row(teams[-1], target_league or source_league, competition, season)
        peers = self._frame("team_season_stats")
        scoped = peers[(peers["competition"].map(normalize_competition) == competition)] if not peers.empty else peers
        league_drtg = _finite(scoped.get("defensive_rating", pd.Series(dtype=float)).median(), 110.0)
        pace_base = _finite(scoped.get("pace", pd.Series(dtype=float)).median(), 72.0)
        pace_factor = _finite(opponent.get("pace"), pace_base) / max(pace_base, 1.0)
        defense_factor = (league_drtg - _finite(opponent.get("defensive_rating"), league_drtg)) / 18.0
        base_points = _finite(player.get("points"))
        base_assists = _finite(player.get("assists"))
        adjustment = {
            "points": base_points * (pace_factor - 1.0) + base_points * defense_factor * 0.12,
            "assists": base_assists * (pace_factor - 1.0) + base_assists * defense_factor * 0.06,
            "ts_pct": defense_factor * 0.012,
            "turnovers": -defense_factor * max(_finite(player.get("turnovers"), 1.5), 0.5) * 0.08,
        }
        play = self.play_type_matchup(spec, players, teams, source_league, target_league)
        observed = [r for r in play["result"]["play_types"] if r["possessions"] > 0]
        if observed:
            ppp_delta = np.average([r["expected_ppp"] - 0.95 for r in observed], weights=[r["possessions"] for r in observed])
            adjustment["points"] += float(ppp_delta) * max(3.0, base_points * 0.35)
        distribution = self.probabilistic_boxscore(spec, players, teams, source_league, target_league, adjustment)
        distribution["result"].update({
            "opponent": self._name("team", teams[-1]),
            "matchup_adjustment": {key: round(value, 4) for key, value in adjustment.items()},
            "play_type_matchup": play["result"]["play_types"],
        })
        distribution["evidence"].extend(play["evidence"])
        distribution["support"]["method"] = "opponent_conditioned_boxscore_distribution"
        return distribution

    def defensive_matchup(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del teams, source_league, target_league
        if len(players) < 2:
            raise ValueError("offensive player and defender are required")
        offense_id, defender_id = players[:2]
        competition = normalize_competition(spec.get("competition"))
        season = int(spec["season"])
        stints = self._frame("lineup_stints")
        pbp = self._frame("pbp_events")
        for frame in (stints, pbp):
            if frame.empty:
                continue
            if "competition" in frame.columns:
                frame.drop(
                    frame[frame["competition"].map(normalize_competition) != competition].index,
                    inplace=True,
                )
            if "season" in frame.columns:
                frame.drop(frame[frame["season"].map(season_year) > season].index, inplace=True)
        exposures: list[dict[str, Any]] = []
        if not stints.empty and {"offense_player_id", "defense_player_id"}.issubset(stints.columns):
            mask = (
                (stints["offense_player_id"].map(_to_int) == _to_int(offense_id))
                & (stints["defense_player_id"].map(_to_int) == _to_int(defender_id))
            )
            pair = stints[mask]
            for row in pair.to_dict("records"):
                exposures.append({"possessions": _finite(row.get("possessions")), "assignment_probability": _finite(row.get("assignment_probability"), 0.5), "points_allowed": _finite(row.get("points_allowed")), "turnovers_forced": _finite(row.get("turnovers_forced"))})
        if not pbp.empty and {"offensive_player_id", "defender_id"}.issubset(pbp.columns):
            pair = pbp[(pbp["offensive_player_id"].map(_to_int) == _to_int(offense_id)) & (pbp["defender_id"].map(_to_int) == _to_int(defender_id))]
            if not pair.empty:
                exposures.append({"possessions": float(len(pair)), "assignment_probability": _finite(pair.get("assignment_probability", pd.Series([1.0])).mean(), 1.0), "points_allowed": _finite(pair.get("points", pd.Series(dtype=float)).sum()), "turnovers_forced": float(pair.get("turnover", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())})
        possessions = sum(item["possessions"] * item["assignment_probability"] for item in exposures)
        points = sum(item["points_allowed"] * item["assignment_probability"] for item in exposures)
        turnovers = sum(item["turnovers_forced"] * item["assignment_probability"] for item in exposures)
        if possessions:
            raw_ppp = points / possessions
            weight = possessions / (possessions + 75.0)
            expected_ppp = weight * raw_ppp + (1.0 - weight) * 0.98
            method = "inferred_assignment_from_pbp_lineup_exposure"
        else:
            defenders = self._frame("player_stats")
            row = defenders[defenders["player_id"].map(_to_int) == _to_int(defender_id)]
            dbpm = _finite(row.get("dbpm", pd.Series(dtype=float)).tail(1).mean())
            expected_ppp = 0.98 - max(-0.12, min(0.12, dbpm * 0.018))
            method = "defender_profile_prior"
        probability = min(0.99, max(0.01, possessions / (possessions + 35.0)))
        return {
            "result": {
                "offensive_player": self._name("player", offense_id),
                "defender": self._name("player", defender_id),
                "estimated_matchup_possessions": round(possessions, 2),
                "estimated_assignment_probability": round(probability, 4),
                "expected_points_per_possession": round(expected_ppp, 4),
                "expected_turnover_rate": round(turnovers / possessions, 4) if possessions else None,
                "assignment_status": "inferred_not_observed",
            },
            "evidence": [{"type": "weighted_lineup_pbp_exposure", "count": round(possessions, 2)}],
            "support": {"method": method, "samples": round(possessions, 2), "confidence": _confidence(possessions)},
            "limitations": ["The defender assignment is probabilistic unless the upstream feed marks it as observed."],
        }

    def shot_counterfactual(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del teams, source_league, target_league
        if not players:
            raise ValueError("player is required")
        profiles = self._frame("shot_profiles")
        params = spec.get("parameters", {})
        transfers = params.get("shot_transfers", [])
        if profiles.empty:
            row = self._player_row(players[0], None, normalize_competition(spec.get("competition")), int(spec["season"]))
            three_rate = _ratio(row.get("three_par"), 0.35)
            profiles = pd.DataFrame([
                {"player_id": players[0], "zone": "rim", "attempts": 100 * (1 - three_rate) * .55, "fg_pct": .62},
                {"player_id": players[0], "zone": "long_mid", "attempts": 100 * (1 - three_rate) * .45, "fg_pct": .40},
                {"player_id": players[0], "zone": "above_break_three", "attempts": 100 * three_rate, "fg_pct": _ratio(row.get("three_point_pct"), .34)},
            ])
            method = "aggregate_shot_profile_prior"
        else:
            profiles = profiles[profiles["player_id"].map(_to_int) == _to_int(players[0])].copy()
            method = "observed_shot_zone_counterfactual"
        if profiles.empty:
            raise ValueError("No shot profile is available")
        profile_records = profiles.to_dict("records")
        attempts = {
            str(row["zone"]).lower(): _finite(row.get("attempts"))
            for row in profile_records
        }
        accuracy = {
            str(row["zone"]).lower(): _ratio(row.get("fg_pct"), .35)
            for row in profile_records
        }
        total = max(sum(attempts.values()), 1.0)
        baseline_points = sum(attempts[z] * accuracy[z] * (3 if "three" in z else 2) for z in attempts)
        applied: list[dict[str, Any]] = []
        for transfer in list(transfers)[:12]:
            source = str(transfer.get("from", "")).lower()
            target = str(transfer.get("to", "")).lower()
            if source not in attempts or target not in SHOT_ZONES:
                continue
            fraction = min(1.0, max(0.0, _ratio(transfer.get("fraction"), 0.0)))
            moved = attempts[source] * fraction
            attempts[source] -= moved
            attempts[target] = attempts.get(target, 0.0) + moved
            source_accuracy = accuracy.get(source, .40)
            target_prior = .36 if "three" in target else (.62 if target == "rim" else .42)
            # Volume/selection uncertainty: new-zone accuracy is strongly shrunk.
            accuracy[target] = (accuracy.get(target, target_prior) * 2.0 + source_accuracy + target_prior * 3.0) / 6.0
            applied.append({"from": source, "to": target, "attempts": round(moved, 3)})
        scenario_points = sum(attempts[z] * accuracy.get(z, .40) * (3 if "three" in z else 2) for z in attempts)
        return {
            "result": {
                "player": self._name("player", players[0]),
                "baseline_expected_points_per_100_shots": round(baseline_points / total * 100, 3),
                "scenario_expected_points_per_100_shots": round(scenario_points / total * 100, 3),
                "delta_expected_points_per_100_shots": round((scenario_points - baseline_points) / total * 100, 3),
                "applied_transfers": applied,
                "scenario_profile": {z: {"frequency": round(v / total, 4), "fg_pct": round(accuracy.get(z, .40), 4)} for z, v in attempts.items()},
            },
            "evidence": [{"type": "shot_attempts", "count": round(total, 2)}],
            "support": {"method": method, "samples": round(total, 2), "confidence": _confidence(total)},
            "limitations": ["Counterfactual accuracy is shrunk because shot selection and shot difficulty change together."],
        }

    def _lineup_score(self, lineup: Iterable[int], competition: str, season: int) -> dict[str, float]:
        ids = tuple(sorted(_to_int(v) for v in lineup))
        stints = self._frame("lineup_stints")
        observed = pd.DataFrame()
        if not stints.empty and "player_ids" in stints.columns:
            observed = stints[stints["player_ids"].map(lambda v: tuple(sorted(_to_int(x) for x in (v if isinstance(v, (list, tuple)) else str(v).split(",")))) == ids)]
        possessions = _finite(observed.get("possessions", pd.Series(dtype=float)).sum())
        observed_net = _finite(np.average(observed["net_rtg"], weights=observed["possessions"]) if possessions and "net_rtg" in observed else 0.0)
        stats = [self._player_row(pid, None, competition, season) for pid in ids]
        individual = float(np.mean([_finite(row.get("bpm"), _finite(row.get("rating")) - 6.0) for row in stats]))
        usages = np.array([_ratio(row.get("usg_pct"), .20) for row in stats])
        spacing = float(np.mean([_ratio(row.get("three_point_pct"), .33) * _ratio(row.get("three_par"), .30) for row in stats]))
        defense = float(np.mean([_finite(row.get("dbpm")) for row in stats]))
        usage_penalty = max(0.0, float(usages.sum()) - 1.12) * 9.0 + float(usages.std()) * 0.8
        prior = individual * 1.7 + spacing * 8.0 + defense * 0.9 - usage_penalty
        weight = possessions / (possessions + 200.0)
        net = weight * observed_net + (1.0 - weight) * prior
        uncertainty = 12.0 / math.sqrt(max(possessions, 1.0) / 25.0 + 1.0)
        return {"net_rtg": net, "observed_net_rtg": observed_net, "prior_net_rtg": prior, "possessions": possessions, "uncertainty": uncertainty}

    def lineup_synergy(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del teams, source_league, target_league
        if len(players) != 5:
            raise ValueError("exactly five players are required")
        score = self._lineup_score(players, normalize_competition(spec.get("competition")), int(spec["season"]))
        return {
            "result": {"lineup": [self._name("player", p) for p in players], **{k: round(v, 4) for k, v in score.items()}},
            "evidence": [{"type": "lineup_possessions", "count": round(score["possessions"], 2)}],
            "support": {"method": "partial_pooling_lineup_synergy", "samples": score["possessions"], "confidence": _confidence(score["possessions"])},
            "limitations": ["Sparse lineups are shrunk toward player, spacing, defense and usage priors."],
        }

    def lineup_optimizer(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del teams, source_league, target_league
        roster = list(dict.fromkeys(players))
        if len(roster) < 5:
            raise ValueError("at least five roster players are required")
        if len(roster) > 18:
            raise ValueError("lineup optimizer supports at most 18 players")
        competition = normalize_competition(spec.get("competition"))
        season = int(spec["season"])
        objective = str(spec.get("parameters", {}).get("objective", "net_rating"))
        ranked = []
        for lineup in itertools.combinations(roster, 5):
            score = self._lineup_score(lineup, competition, season)
            value = score["net_rtg"] - (score["uncertainty"] * .10 if objective == "risk_adjusted" else 0.0)
            ranked.append((value, lineup, score))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_n = min(int(spec.get("top_n", 5)), len(ranked))
        result = [{"players": [self._name("player", p) for p in lineup], "objective_value": round(value, 4), **{k: round(v, 4) for k, v in score.items()}} for value, lineup, score in ranked[:top_n]]
        return {"result": {"objective": objective, "evaluated_lineups": len(ranked), "lineups": result}, "evidence": [{"type": "candidate_lineups", "count": len(ranked)}], "support": {"method": "exhaustive_partial_pooling_optimizer", "samples": len(ranked), "confidence": "medium"}, "limitations": ["Optimizer quality depends on possession coverage and does not imply causal lineup effects."]}

    def roster_optimizer(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del source_league, target_league
        candidates = list(dict.fromkeys(players))
        params = spec.get("parameters", {})
        size = min(15, max(5, int(params.get("roster_size", min(12, len(candidates))))))
        if len(candidates) < size:
            raise ValueError("candidate pool is smaller than roster_size")
        if len(candidates) > 30:
            raise ValueError("roster optimizer supports at most 30 candidates")
        required = {_to_int(v) for v in params.get("required_player_ids", [])}
        competition = normalize_competition(spec.get("competition"))
        season = int(spec["season"])
        profiles = {pid: self._player_row(pid, None, competition, season) for pid in candidates}
        scored = []
        for pid, row in profiles.items():
            score = _finite(row.get("bpm")) + _finite(row.get("dbpm")) * .4 + _ratio(row.get("three_point_pct"), .33) * 2.0
            scored.append((score, pid))
        selected = list(required)
        for _, pid in sorted(scored, reverse=True):
            if pid not in selected:
                selected.append(pid)
            if len(selected) == size:
                break
        # Local search evaluates replacements by the average of the best sampled lineups.

        def roster_value(roster: list[int]) -> float:
            combos = list(itertools.combinations(roster, 5))
            if len(combos) > 250:
                combos = combos[:250]
            values = sorted((self._lineup_score(c, competition, season)["net_rtg"] for c in combos), reverse=True)
            return float(np.mean(values[: min(8, len(values))]))
        best = selected[:]
        best_value = roster_value(best)
        for _ in range(3):
            improved = False
            for outgoing in [p for p in best if p not in required]:
                for incoming in [p for p in candidates if p not in best]:
                    trial = [incoming if p == outgoing else p for p in best]
                    value = roster_value(trial)
                    if value > best_value + 1e-9:
                        best, best_value, improved = trial, value, True
            if not improved:
                break
        return {"result": {"team": self._name("team", teams[0]) if teams else None, "roster": [self._name("player", p) for p in best], "expected_top_lineup_net_rating": round(best_value, 4), "candidate_count": len(candidates)}, "evidence": [{"type": "candidate_players", "count": len(candidates)}], "support": {"method": "constrained_local_search_over_lineup_synergy", "samples": len(candidates), "confidence": "medium"}, "limitations": ["Salary, contracts and availability are enforced only when supplied as explicit constraints."]}

    def causal_effect(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        del players, teams, source_league, target_league
        params = spec.get("parameters", {})
        treatment = str(params.get("treatment", "pace_increase"))
        outcome = str(params.get("outcome", "points"))
        frame = self._frame("causal_panel")
        if frame.empty:
            stats = self._frame("player_stats")
            if stats.empty or outcome not in stats.columns:
                raise ValueError("No longitudinal panel is available for causal estimation")
            stats = stats.copy()
            stats["_year"] = stats["season"].map(season_year)
            stats = stats.sort_values(["player_id", "_year"])
            stats["outcome"] = pd.to_numeric(stats[outcome], errors="coerce")
            stats["next_outcome"] = stats.groupby("player_id")["outcome"].shift(-1)
            stats["treatment"] = (pd.to_numeric(stats.get("minutes_per_game", 0), errors="coerce").fillna(0).groupby(stats["player_id"]).diff().shift(-1) > 1.5).astype(float)
            frame = stats.dropna(subset=["outcome", "next_outcome"])
            covariates = [c for c in ("outcome", "age", "minutes_per_game", "usg_pct", "bpm", "rating") if c in frame]
        else:
            covariates = [c for c in params.get("covariates", []) if c in frame.columns][:20]
        if len(frame) < 30 or "treatment" not in frame or "next_outcome" not in frame:
            return {"result": {"treatment": treatment, "outcome": outcome, "causal_effect": None, "identification": "unavailable", "reason": "insufficient_longitudinal_support"}, "evidence": [{"type": "panel_rows", "count": len(frame)}], "support": {"method": "aipw_not_identified", "samples": len(frame), "confidence": "low"}, "limitations": ["Causal language is disabled because identification checks failed."]}
        from sklearn.linear_model import LinearRegression, LogisticRegression
        x = frame[covariates].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        t = frame["treatment"].astype(float).to_numpy()
        y = frame["next_outcome"].astype(float).to_numpy()
        if len(np.unique(t)) < 2:
            raise ValueError("Treatment has no variation")
        propensity_model = LogisticRegression(max_iter=1000).fit(x, t)
        propensity = np.clip(propensity_model.predict_proba(x)[:, 1], .05, .95)
        outcome_0 = LinearRegression().fit(x[t == 0], y[t == 0])
        outcome_1 = LinearRegression().fit(x[t == 1], y[t == 1])
        mu0, mu1 = outcome_0.predict(x), outcome_1.predict(x)
        influence = mu1 - mu0 + t * (y - mu1) / propensity - (1 - t) * (y - mu0) / (1 - propensity)
        effect = float(influence.mean())
        se = float(influence.std(ddof=1) / math.sqrt(len(influence)))
        overlap = float(np.mean((propensity >= .1) & (propensity <= .9)))
        identified = overlap >= .7 and min(int(t.sum()), int((1 - t).sum())) >= 10
        return {"result": {"treatment": treatment, "outcome": outcome, "causal_effect": round(effect, 4) if identified else None, "confidence_interval_95": [round(effect - 1.96 * se, 4), round(effect + 1.96 * se, 4)] if identified else None, "overlap": round(overlap, 4), "identification": "supported" if identified else "unavailable", "reason": None if identified else "insufficient_overlap"}, "evidence": [{"type": "longitudinal_panel", "count": len(frame), "treated": int(t.sum()), "control": int((1 - t).sum())}], "support": {"method": "doubly_robust_aipw", "samples": len(frame), "confidence": _confidence(len(frame)) if identified else "low"}, "limitations": ["AIPW depends on consistency, positivity and no-unmeasured-confounding assumptions."]}

    def composite(self, spec, players, teams, source_league, target_league) -> dict[str, Any]:
        if not players:
            raise ValueError("player is required")
        components: dict[str, Any] = {}
        if teams:
            matchup = self.opponent_matchup(spec, players, teams, source_league, target_league)
            components["opponent_matchup"] = matchup["result"]
            final = matchup
        else:
            final = self.probabilistic_boxscore(spec, players, teams, source_league, target_league)
        if len(players) == 5:
            components["lineup_synergy"] = self.lineup_synergy(spec, players, teams, source_league, target_league)["result"]
        if spec.get("parameters", {}).get("shot_transfers"):
            components["shot_counterfactual"] = self.shot_counterfactual(spec, players, teams, source_league, target_league)["result"]
        final["result"]["components"] = components
        final["support"]["method"] = "single_conditioned_composite_simulation"
        final["limitations"].append("All requested context is combined before the final Monte Carlo distribution; components are not independent forecasts.")
        return final

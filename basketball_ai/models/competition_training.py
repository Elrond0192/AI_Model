"""Competition-aware training and inference primitives.

Competition is part of the statistical identity of a sample.  A production
forecast therefore pairs only the same player, league and competition across
consecutive seasons and scopes online inference to that exact context.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from basketball_ai.constants import LEAGUE_MAX_GAMES_BY_NAME, LEAGUE_MAX_GAMES_DEFAULT
from basketball_ai.data.loader import _to_int
from basketball_ai.models.compatibility_model import _POSITION_BUCKET
from basketball_ai.models.performance_model import FEATURE_COLS, METRIC_CATALOG
from basketball_ai.models.production_training import (
    SeasonAheadPerformanceModel,
    TemporalCompatibilityModel,
    _model_threads,
    _numeric_seasons,
    season_year,
)

_COMP_ALIASES = {
    "": "RS",
    "REGULAR": "RS",
    "REGULAR SEASON": "RS",
    "REG SEASON": "RS",
    "RS": "RS",
    "PLAYOFF": "PO",
    "PLAYOFFS": "PO",
    "POSTSEASON": "PO",
    "PO": "PO",
    "TOTAL": "TOT",
    "ALL": "TOT",
    "TOT": "TOT",
    "SUPER CUP": "SUPERCUP",
    "SUPERCUP": "SUPERCUP",
    "CUP": "CUP",
}
_PREFERRED_COMPETITION_ORDER = ("RS", "PO", "CUP", "SUPERCUP", "TOT")


def normalize_competition(value: Any) -> str:
    text = re.sub(r"[\s_-]+", " ", str(value or "RS").strip().upper())
    canonical = _COMP_ALIASES.get(text)
    if canonical:
        return canonical
    return re.sub(r"[^A-Z0-9_]", "", text.replace(" ", "_")) or "RS"


def build_competition_encoding(values: pd.Series | List[Any]) -> Dict[str, int]:
    observed = {normalize_competition(value) for value in list(values)}
    ordered = [value for value in _PREFERRED_COMPETITION_ORDER if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return {value: index for index, value in enumerate(ordered)}


def apply_competition_encoding(mapping: Dict[str, int]) -> None:
    """Keep legacy feature builders aligned with the trained vocabulary."""
    from basketball_ai.models import ensemble as ensemble_module
    from basketball_ai.models import performance_model as performance_module

    performance_module.COMPETITION_ENCODING.clear()
    performance_module.COMPETITION_ENCODING.update(mapping)
    ensemble_module.COMPETITION_ENCODING.clear()
    ensemble_module.COMPETITION_ENCODING.update(mapping)


def resolve_league_id(data: Dict[str, Any], league: str | int) -> int:
    try:
        raw = int(league)
        if raw in data.get("league_dict", {}):
            return raw
    except (TypeError, ValueError):
        pass

    wanted = str(league or "").strip().upper()
    leagues = data.get("leagues", pd.DataFrame())
    if leagues is None or leagues.empty:
        raise ValueError(f"League {league!r} is unavailable")
    for row in leagues.to_dict("records"):
        candidates = {
            str(row.get("name", "") or "").strip().upper(),
            str(row.get("league_key", "") or "").strip().upper(),
        }
        if wanted in candidates:
            return _to_int(row["id"])
    raise ValueError(f"League {league!r} is unavailable")


class CompetitionSeasonAheadPerformanceModel(SeasonAheadPerformanceModel):
    """Season-ahead XGBoost samples isolated by league and competition."""

    competition_encoding: Dict[str, int]

    def prepare_features(
        self,
        data: Dict[str, Any],
        extra_metrics: Optional[List[str]] = None,
        split_season: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if extra_metrics is None:
            extra_metrics = []
        player_stats = data["player_stats"].copy()
        players = data["players"]
        player_stats["_season_year"] = _numeric_seasons(player_stats)
        player_stats = player_stats.dropna(
            subset=["_season_year", "player_id", "league_id", "competition"]
        )
        player_stats["_season_year"] = player_stats["_season_year"].astype(int)
        player_stats["_competition"] = player_stats["competition"].map(
            normalize_competition
        )

        if split_season is not None:
            cutoff = season_year(split_season)
            player_stats = player_stats[player_stats["_season_year"] <= cutoff]

        key_columns = ["player_id", "league_id", "_season_year", "_competition"]
        duplicated = player_stats.duplicated(key_columns, keep=False)
        if duplicated.any():
            examples = (
                player_stats.loc[
                    duplicated,
                    ["player_id", "league_id", "season", "competition"],
                ]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                "player_stats must contain one row per "
                "player+league+season+competition; duplicates: "
                f"{examples}"
            )

        self.competition_encoding = build_competition_encoding(
            player_stats["_competition"]
        )
        apply_competition_encoding(self.competition_encoding)

        def build_encoding(column: str) -> Dict[str, int]:
            if column not in player_stats.columns:
                return {}
            values = sorted(
                {
                    str(value).strip()
                    for value in player_stats[column].dropna()
                    if str(value).strip()
                }
            )
            return {value: index + 1 for index, value in enumerate(values)}

        self.role_encoding = build_encoding("ruolo_combinato")
        self.role_off_encoding = build_encoding("ruolo_offensivo")
        self.role_def_encoding = build_encoding("ruolo_difensivo")

        extra_feature_names: List[str] = []
        for column in extra_metrics:
            info = METRIC_CATALOG.get(column)
            if info is None:
                continue
            name = f"{column}_per_36" if info.use_per36 else f"avg_{column}"
            if name not in extra_feature_names:
                extra_feature_names.append(name)
        feature_names = FEATURE_COLS + extra_feature_names

        latest_data_year = int(player_stats["_season_year"].max())
        birth_year_map: Dict[int, int] = {}
        position_map: Dict[int, str] = {}
        for player in players.to_dict("records"):
            pid = _to_int(player["id"])
            position_map[pid] = str(player.get("position", "PG") or "PG")
            birth_date = player.get("birth_date", player.get("date_of_birth"))
            try:
                birth_year_map[pid] = int(str(birth_date)[:4])
                continue
            except (TypeError, ValueError):
                pass
            age = player.get("age")
            if age is not None and not pd.isna(age):
                birth_year_map[pid] = latest_data_year - int(age)

        league_max_games: Optional[Dict[int, int]] = None
        leagues = data.get("leagues")
        if leagues is not None and not leagues.empty and "id" in leagues.columns:
            if "max_games" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): int(row["max_games"])
                    for row in leagues.to_dict("records")
                    if row.get("max_games") and not pd.isna(row.get("max_games"))
                }
            elif "name" in leagues.columns:
                league_max_games = {
                    int(_to_int(row["id"])): LEAGUE_MAX_GAMES_BY_NAME.get(
                        str(row["name"]).strip(), LEAGUE_MAX_GAMES_DEFAULT
                    )
                    for row in leagues.to_dict("records")
                }

        rows: List[Dict[str, float]] = []
        targets: List[float] = []
        target_years: List[int] = []
        source_years: List[int] = []
        competitions: List[str] = []
        league_ids: List[int] = []
        skipped_gaps = 0

        grouping = ["player_id", "league_id", "_competition"]
        for (raw_pid, raw_league_id, competition), group in player_stats.groupby(
            grouping, dropna=False, sort=False
        ):
            pid = _to_int(raw_pid)
            league_id = _to_int(raw_league_id)
            birth_year = birth_year_map.get(pid)
            if birth_year is None:
                continue
            position = position_map.get(pid, "PG")
            group = group.sort_values("_season_year").reset_index(drop=True)
            years = [int(value) for value in group["_season_year"].tolist()]
            year_to_index = {year: index for index, year in enumerate(years)}
            max_year = years[-1]
            for source_index, source_year in enumerate(years):
                target_year = source_year + 1
                target_index = year_to_index.get(target_year)
                if target_index is None:
                    if source_year < max_year:
                        skipped_gaps += 1
                    continue
                source = group.iloc[source_index].copy()
                target = group.iloc[target_index]
                source_age = source_year - birth_year
                if source_age < 14 or source_age > 44:
                    continue

                history = group.iloc[: source_index + 1].drop(
                    columns=["_season_year", "_competition"]
                )
                source_clean = source.drop(labels=["_season_year", "_competition"])
                source_clean["competition"] = competition
                rows.append(
                    self._build_row(
                        source_clean,
                        source_age,
                        position,
                        history,
                        extra_metrics=extra_metrics,
                        league_max_games=league_max_games,
                    )
                )
                targets.append(float(target["rating"]))
                source_years.append(source_year)
                target_years.append(target_year)
                competitions.append(str(competition))
                league_ids.append(league_id)

        if not rows:
            raise ValueError(
                "No consecutive t -> t+1 player/league/competition samples are available"
            )

        X = pd.DataFrame(rows, columns=feature_names).fillna(0.0)
        y = np.asarray(targets, dtype=float)
        self._last_source_years = source_years
        self._last_season_years = target_years
        self._last_competitions = competitions
        self._last_league_ids = league_ids
        self._skipped_gap_pairs = skipped_gaps
        return X, y


class CompetitionTemporalCompatibilityModel(TemporalCompatibilityModel):
    """Compatibility model whose historical vectors use one league/competition."""

    @staticmethod
    def _context(data: Dict[str, Any]) -> tuple[Optional[int], Optional[str]]:
        league_id = data.get("_prediction_league_id")
        competition = data.get("_prediction_competition")
        return (
            _to_int(league_id) if league_id is not None else None,
            normalize_competition(competition) if competition is not None else None,
        )

    @staticmethod
    def _filter_context(
        frame: pd.DataFrame,
        league_id: Optional[int],
        competition: Optional[str],
    ) -> pd.DataFrame:
        rows = frame
        if league_id is not None and "league_id" in rows.columns:
            rows = rows[rows["league_id"].map(_to_int) == _to_int(league_id)]
        if competition is not None and "competition" in rows.columns:
            rows = rows[
                rows["competition"].map(normalize_competition) == competition
            ]
        return rows

    def _player_style_through_context(
        self,
        player_id: int,
        data: Dict[str, Any],
        through_season: Optional[int],
        league_id: Optional[int],
        competition: Optional[str],
    ) -> np.ndarray:
        pid = _to_int(player_id)
        player = data["player_dict"].get(pid, {})
        position = str(player.get("position", "PG"))
        bucket = _POSITION_BUCKET.get(
            position, _POSITION_BUCKET.get(position.split("/")[0], 0.5)
        )
        stats = data["player_stats"]
        p_stats = stats[stats["player_id"].map(_to_int) == pid].copy()
        p_stats = self._filter_context(p_stats, league_id, competition)
        if not p_stats.empty:
            p_stats["_season_year"] = _numeric_seasons(p_stats)
            if through_season is not None:
                p_stats = p_stats[p_stats["_season_year"] <= int(through_season)]

        def mean(column: str, fallback: float) -> float:
            if p_stats.empty or column not in p_stats.columns:
                return fallback
            values = pd.to_numeric(p_stats[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else fallback

        return np.asarray(
            [
                bucket,
                np.clip(mean("usg_pct", 18.0) / 40.0, 0.0, 1.0),
                np.clip(mean("ts_pct", 0.52), 0.0, 1.0),
                np.clip(mean("points", 12.0) / 40.0, 0.0, 1.0),
                np.clip(mean("three_par", 0.30), 0.0, 1.0),
                np.clip((mean("dbpm", 0.0) + 5.0) / 10.0, 0.0, 1.0),
            ],
            dtype=float,
        )

    def _team_row_context(
        self,
        team_id: int,
        data: Dict[str, Any],
        season: int,
        league_id: Optional[int],
        competition: Optional[str],
        *,
        strict_before: bool,
    ) -> Optional[Dict[str, Any]]:
        history = data.get("team_season_stats")
        if history is None or history.empty:
            return None
        rows = history[history["team_id"].map(_to_int) == _to_int(team_id)].copy()
        rows = self._filter_context(rows, league_id, competition)
        if rows.empty:
            return None
        rows["_season_year"] = _numeric_seasons(rows)
        rows = rows[
            rows["_season_year"] < int(season)
            if strict_before
            else rows["_season_year"] <= int(season)
        ]
        if rows.empty:
            return None
        return rows.sort_values("_season_year").iloc[-1].to_dict()

    @staticmethod
    def _prefix_mean(values: pd.Series, fallback: float) -> tuple[np.ndarray, np.ndarray, float]:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(numeric)
        sums = np.cumsum(np.where(valid, numeric, 0.0))
        counts = np.cumsum(valid.astype(np.int64))
        return sums, counts, fallback

    @staticmethod
    def _mean_before(prefix: tuple[np.ndarray, np.ndarray, float], index: int) -> float:
        sums, counts, fallback = prefix
        if index <= 0:
            return fallback
        count = int(counts[index - 1])
        return float(sums[index - 1] / count) if count else fallback

    def train(self, data: Dict[str, Any]) -> None:
        history = data.get("team_season_stats")
        if history is None or history.empty:
            raise RuntimeError(
                "team competition history is required for compatibility training"
            )

        stats = data["player_stats"].copy()
        stats["_season_year"] = _numeric_seasons(stats)
        stats["_competition"] = stats["competition"].map(normalize_competition)
        stats = stats.dropna(
            subset=["_season_year", "team_id", "player_id", "league_id"]
        )
        stats["_pid_int"] = stats["player_id"].map(_to_int)
        stats["_tid_int"] = stats["team_id"].map(_to_int)
        stats["_lid_int"] = stats["league_id"].map(_to_int)

        team_history = history.copy()
        team_history["_season_year"] = _numeric_seasons(team_history)
        team_history["_competition"] = team_history["competition"].map(
            normalize_competition
        )
        team_history = team_history.dropna(
            subset=["_season_year", "team_id", "league_id"]
        )
        team_history["_tid_int"] = team_history["team_id"].map(_to_int)
        team_history["_lid_int"] = team_history["league_id"].map(_to_int)

        team_context: Dict[tuple[int, int, str], tuple[np.ndarray, List[np.ndarray]]] = {}
        for key, group in team_history.groupby(
            ["_tid_int", "_lid_int", "_competition"], sort=False
        ):
            ordered = group.sort_values("_season_year")
            years = ordered["_season_year"].astype(int).to_numpy()
            vectors = [
                self._team_style_vector(row)
                for row in ordered.to_dict("records")
            ]
            team_context[(int(key[0]), int(key[1]), str(key[2]))] = (
                years,
                vectors,
            )

        X_rows: List[np.ndarray] = []
        y_values: List[float] = []
        samples_by_competition: Dict[str, int] = {}
        grouping = ["_pid_int", "_lid_int", "_competition"]

        for (pid, league_id, competition), group in stats.groupby(
            grouping, sort=False
        ):
            ordered = group.sort_values("_season_year").reset_index(drop=True)
            if len(ordered) < 2:
                continue
            player = data["player_dict"].get(int(pid), {})
            position = str(player.get("position", "PG"))
            bucket = _POSITION_BUCKET.get(
                position, _POSITION_BUCKET.get(position.split("/")[0], 0.5)
            )
            prefixes = {
                "usg_pct": self._prefix_mean(ordered.get("usg_pct", pd.Series(dtype=float)), 18.0),
                "ts_pct": self._prefix_mean(ordered.get("ts_pct", pd.Series(dtype=float)), 0.52),
                "points": self._prefix_mean(ordered.get("points", pd.Series(dtype=float)), 12.0),
                "three_par": self._prefix_mean(ordered.get("three_par", pd.Series(dtype=float)), 0.30),
                "dbpm": self._prefix_mean(ordered.get("dbpm", pd.Series(dtype=float)), 0.0),
                "rating": self._prefix_mean(ordered["rating"], 0.0),
            }

            for index in range(1, len(ordered)):
                stat = ordered.iloc[index]
                target_season = int(stat["_season_year"])
                tid = int(stat["_tid_int"])
                team_data = team_context.get((tid, int(league_id), str(competition)))
                if team_data is None:
                    continue
                team_years, team_vectors = team_data
                team_index = int(np.searchsorted(team_years, target_season, side="left")) - 1
                if team_index < 0:
                    continue

                player_vector = np.asarray(
                    [
                        bucket,
                        np.clip(self._mean_before(prefixes["usg_pct"], index) / 40.0, 0.0, 1.0),
                        np.clip(self._mean_before(prefixes["ts_pct"], index), 0.0, 1.0),
                        np.clip(self._mean_before(prefixes["points"], index) / 40.0, 0.0, 1.0),
                        np.clip(self._mean_before(prefixes["three_par"], index), 0.0, 1.0),
                        np.clip((self._mean_before(prefixes["dbpm"], index) + 5.0) / 10.0, 0.0, 1.0),
                    ],
                    dtype=float,
                )
                vector = np.concatenate([player_vector, team_vectors[team_index]])
                prior_mean = self._mean_before(prefixes["rating"], index)
                compatibility = float(
                    np.clip(
                        (float(stat["rating"]) - prior_mean + 1.5) / 3.0,
                        0.0,
                        1.0,
                    )
                )
                X_rows.append(vector)
                y_values.append(compatibility)
                samples_by_competition[str(competition)] = (
                    samples_by_competition.get(str(competition), 0) + 1
                )

        if not X_rows:
            raise RuntimeError(
                "No leakage-free competition-specific compatibility samples are available"
            )
        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_values, dtype=float)
        effective_neighbors = max(1, min(self.n_neighbors, len(X)))
        self.knn = KNeighborsRegressor(
            n_neighbors=effective_neighbors,
            metric="euclidean",
            n_jobs=_model_threads(),
        )
        scaled = self.scaler.fit_transform(X)
        self.knn.fit(scaled, y)
        self.is_trained = True
        self.training_samples = len(X)
        self.training_samples_by_competition = samples_by_competition

    def score(self, player_id: int, team_id: int, data: Dict[str, Any]) -> float:
        if not self.is_trained:
            raise RuntimeError("Compatibility model is not trained")
        league_id, competition = self._context(data)
        as_of = data.get("_as_of_season")
        if as_of is None:
            raise ValueError("Competition-aware prediction requires an as-of season")
        team = self._team_row_context(
            team_id,
            data,
            int(as_of),
            league_id,
            competition,
            strict_before=False,
        )
        if team is None:
            raise ValueError("No team context exists for the requested competition")
        player_vector = self._player_style_through_context(
            player_id,
            data,
            int(as_of),
            league_id,
            competition,
        )
        vector = np.concatenate([player_vector, self._team_style_vector(team)])
        scaled = self.scaler.transform(vector.reshape(1, -1))
        return float(np.clip(self.knn.predict(scaled)[0], 0.50, 1.0))


def competition_support(
    data: Dict[str, Any],
    player_id: int,
    league_id: int,
    competition: str,
    source_season: int,
) -> Dict[str, Any]:
    competition = normalize_competition(competition)
    stats = data.get("player_stats", pd.DataFrame())
    if stats.empty:
        return {
            "mode": "isolated",
            "competition": competition,
            "source_seasons": 0,
            "source_games": 0,
            "exact_source_games": 0,
        }
    rows = stats[
        (stats["player_id"].map(_to_int) == _to_int(player_id))
        & (stats["league_id"].map(_to_int) == _to_int(league_id))
        & (stats["competition"].map(normalize_competition) == competition)
    ].copy()
    if rows.empty:
        years: List[int] = []
        games = 0
        exact_games = 0
    else:
        rows["_season_year"] = _numeric_seasons(rows)
        rows = rows[rows["_season_year"] <= int(source_season)]
        years = sorted({int(value) for value in rows["_season_year"].dropna()})
        games = int(pd.to_numeric(rows.get("games_played", 0), errors="coerce").fillna(0).sum())
        exact = rows[rows["_season_year"] == int(source_season)]
        exact_games = int(
            pd.to_numeric(exact.get("games_played", 0), errors="coerce").fillna(0).sum()
        )
    return {
        "mode": "isolated",
        "competition": competition,
        "source_seasons": len(years),
        "source_games": games,
        "exact_source_games": exact_games,
    }


def scope_prediction_context(
    data: Dict[str, Any],
    player_id: int,
    team_id: int,
    league_id: int,
    competition: str,
    source_season: int,
    *,
    require_exact_player_source: bool = True,
) -> Dict[str, Any]:
    """Return a shallow isolated league/competition view for one prediction."""
    competition = normalize_competition(competition)
    source_season = int(source_season)
    league_id = _to_int(league_id)
    player_id = _to_int(player_id)
    team_id = _to_int(team_id)

    scoped = dict(data)
    stats = data["player_stats"].copy()
    stats["_season_year"] = _numeric_seasons(stats)
    stats = stats[
        (stats["league_id"].map(_to_int) == league_id)
        & (stats["competition"].map(normalize_competition) == competition)
        & (stats["_season_year"] <= source_season)
    ].copy()
    player_source = stats[
        (stats["player_id"].map(_to_int) == player_id)
        & (stats["_season_year"] == source_season)
    ]
    if require_exact_player_source and player_source.empty:
        raise ValueError(
            f"Player has no isolated {competition} data in the requested league "
            f"for source season {source_season}"
        )
    stats["season"] = stats["_season_year"].astype(int).astype(str)
    scoped["player_stats"] = stats.drop(columns=["_season_year"]).reset_index(drop=True)

    history = data["team_season_stats"].copy()
    history["_season_year"] = _numeric_seasons(history)
    history = history[
        (history["league_id"].map(_to_int) == league_id)
        & (history["competition"].map(normalize_competition) == competition)
        & (history["_season_year"] <= source_season)
    ].copy()
    target_team_history = history[history["team_id"].map(_to_int) == team_id]
    if target_team_history.empty:
        raise ValueError(
            f"Target team has no {competition} context in the requested league "
            f"through source season {source_season}"
        )
    history["season"] = history["_season_year"].astype(int).astype(str)
    scoped["team_season_stats"] = history.drop(columns=["_season_year"]).reset_index(drop=True)

    latest = (
        history.sort_values(["team_id", "_season_year"])
        .groupby("team_id", as_index=False)
        .tail(1)
    )
    current_meta = {
        _to_int(row["id"]): row
        for row in data.get("teams", pd.DataFrame()).to_dict("records")
        if not pd.isna(row.get("id"))
    }
    team_rows: List[Dict[str, Any]] = []
    for row in latest.to_dict("records"):
        tid = _to_int(row["team_id"])
        base = dict(current_meta.get(tid, {}))
        base.update(
            {
                "id": tid,
                "global_id": row.get("global_id", base.get("global_id")),
                "name": row.get("name", base.get("name", f"Team {tid}")),
                "league_id": league_id,
                "league_key": row.get("league_key", base.get("league_key", "")),
                "pace": float(row.get("pace", 75.0)),
                "offensive_rating": float(row.get("offensive_rating", 110.0)),
                "defensive_rating": float(row.get("defensive_rating", 110.0)),
                "three_point_attempt_rate": float(row.get("three_point_attempt_rate", 0.35)),
                "assists_per_game": float(row.get("assists_per_game", 20.0)),
                "star_player_usage": float(row.get("star_player_usage", 0.25)),
                "net_rtg": float(row.get("net_rtg", 0.0)),
                "short_name": row.get("short_name", base.get("short_name", "")),
                "playing_style": "",
                "formation": "",
                "league_tier": int(base.get("league_tier", 1) or 1),
            }
        )
        team_rows.append(base)
    teams = pd.DataFrame(team_rows)
    scoped["teams"] = teams
    scoped["team_dict"] = {
        _to_int(row["id"]): row for row in teams.to_dict("records")
    }
    scoped["league_teams"] = {league_id: sorted(scoped["team_dict"])}

    players = data.get("players", pd.DataFrame()).copy()
    if not players.empty and not player_source.empty:
        source_row = player_source.sort_values("_season_year").iloc[-1]
        mask = players["id"].map(_to_int) == player_id
        if mask.any():
            players.loc[mask, "current_league_id"] = league_id
            if not pd.isna(source_row.get("team_id")):
                players.loc[mask, "current_team_id"] = _to_int(source_row["team_id"])
    scoped["players"] = players
    scoped["player_dict"] = {
        _to_int(row["id"]): row
        for row in players.to_dict("records")
        if not pd.isna(row.get("id"))
    }

    scoped["_as_of_season"] = source_season
    scoped["_prediction_league_id"] = league_id
    scoped["_prediction_competition"] = competition
    scoped["_competition_support"] = competition_support(
        data, player_id, league_id, competition, source_season
    )
    return scoped


__all__ = [
    "CompetitionSeasonAheadPerformanceModel",
    "CompetitionTemporalCompatibilityModel",
    "apply_competition_encoding",
    "build_competition_encoding",
    "competition_support",
    "normalize_competition",
    "resolve_league_id",
    "scope_prediction_context",
]

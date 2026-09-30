"""Contextual BB-Rating engine.

BB-Rating answers a different question from the season-ahead Prediction Model:

    "How is this player performing relative to comparable players in this
     league/season/competition?"

The score is deterministic. It is built from contextual percentiles of observed
metrics, with peer selection based on position, age band and role where the
population is large enough. No post-processing is applied to the prediction
model and no WordPress rating logic is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from basketball_ai.bb_rating.semantics import METRIC_SEMANTICS, metric_interpretation

BB_RATING_VERSION = "1.1"

# Minimum peer population before using a narrower peer definition. The engine
# deliberately falls back to broader cohorts rather than producing unstable
# tiny-sample percentiles.
MIN_PEER_SAMPLES = 25
MIN_CONTEXT_SAMPLES = 10

AGE_BANDS = (
    (18, 21, "18-21"),
    (22, 25, "22-25"),
    (26, 29, "26-29"),
    (30, 33, "30-33"),
    (34, 99, "34+"),
)

BAND_LABELS = (
    (0.00, "Very Low"),
    (0.10, "Low"),
    (0.25, "Below Average"),
    (0.40, "Average"),
    (0.60, "Good"),
    (0.75, "Very Good"),
    (0.90, "Elite"),
    (0.975, "Superstar"),
)

RATE_COLUMNS = frozenset(
    {
        "ts_pct",
        "usg_pct",
        "tov_pct",
        "ast_pct",
        "stl_pct",
        "blk_pct",
        "orb_pct",
        "drb_pct",
        "reb_pct",
        "starter_pct",
    }
)

@dataclass(frozen=True)
class RatingMetricSpec:
    key: str
    source_column: str
    dimension: str
    weight: float
    direction: str = "higher_better"
    description: str = ""


BB_RATING_METRICS: tuple[RatingMetricSpec, ...] = (
    RatingMetricSpec("RAPTOR", "raptor_total", "impact", 0.18, description="RAPTOR total"),
    RatingMetricSpec("LEBRON", "lebron_total", "impact", 0.14, description="LEBRON total"),
    RatingMetricSpec("VORP", "vorp", "impact", 0.08, description="Value Over Replacement Player"),
    RatingMetricSpec("PER", "per", "offense", 0.06, description="Player efficiency"),
    RatingMetricSpec("TS%", "ts_pct", "offense", 0.075, description="True Shooting Percentage"),
    RatingMetricSpec(
        "SCORING_EFFICIENCY",
        "scoring_efficiency",
        "offense",
        0.075,
        description="Scoring efficiency index",
    ),
    RatingMetricSpec("AST%", "ast_pct", "offense", 0.045, description="Assist Percentage"),
    # USG% is an explicit role/involvement signal. It is deliberately excluded
    # from the composite weight: high usage is not inherently good or bad.
    RatingMetricSpec("USG%", "usg_pct", "role_context", 0.0, description="Usage Rate"),
    RatingMetricSpec(
        "TOV%",
        "tov_pct",
        "offense",
        0.045,
        direction="lower_better",
        description="Turnover Percentage",
    ),
    RatingMetricSpec("RAPTOR_DEF", "raptor_def", "defense", 0.09, description="RAPTOR defensive component"),
    RatingMetricSpec("DBPM", "dbpm", "defense", 0.06, description="Defensive Box Plus/Minus"),
    RatingMetricSpec("NET_RTG_DIFF", "net_rtg_diff", "defense", 0.03, description="On/off net rating differential"),
    RatingMetricSpec("STL%", "stl_pct", "defense", 0.01, description="Steal Percentage"),
    RatingMetricSpec("BLK%", "blk_pct", "defense", 0.01, description="Block Percentage"),
    RatingMetricSpec("REB%", "reb_pct", "versatility", 0.045, description="Rebound Percentage"),
    RatingMetricSpec("HUSTLE", "hustle_index", "versatility", 0.04, description="Hustle index"),
    RatingMetricSpec("FOUL_DRAWING", "foul_drawing_rate", "versatility", 0.015, description="Foul drawing rate"),

    # Explain-only evidence. These metrics are returned with contextual
    # percentiles and semantic interpretation but have zero score weight.
    RatingMetricSpec("POINTS", "points", "explanation_only", 0.0, direction="contextual", description="Points per game"),
    RatingMetricSpec("ASSISTS", "assists", "explanation_only", 0.0, direction="contextual", description="Assists per game"),
    RatingMetricSpec("REBOUNDS", "rebounds", "explanation_only", 0.0, direction="contextual", description="Rebounds per game"),
    RatingMetricSpec("OFFENSIVE_REBOUNDS", "offensive_rebounds", "explanation_only", 0.0, direction="contextual", description="Offensive rebounds per game"),
    RatingMetricSpec("DEFENSIVE_REBOUNDS", "defensive_rebounds", "explanation_only", 0.0, direction="contextual", description="Defensive rebounds per game"),
    RatingMetricSpec("STEALS", "steals", "explanation_only", 0.0, direction="contextual", description="Steals per game"),
    RatingMetricSpec("BLOCKS", "blocks", "explanation_only", 0.0, direction="contextual", description="Blocks per game"),
    RatingMetricSpec("TURNOVERS", "turnovers", "explanation_only", 0.0, direction="lower_better", description="Turnovers per game"),
    RatingMetricSpec("FG%", "fg_pct", "explanation_only", 0.0, direction="higher_better", description="Field goal percentage"),
    RatingMetricSpec("3P%", "three_point_pct", "explanation_only", 0.0, direction="higher_better", description="Three-point percentage"),
    RatingMetricSpec("FT%", "ft_pct", "explanation_only", 0.0, direction="higher_better", description="Free-throw percentage"),
    RatingMetricSpec("eFG%", "efg_pct", "explanation_only", 0.0, direction="higher_better", description="Effective field-goal percentage"),
    RatingMetricSpec("2P%", "two_point_pct", "explanation_only", 0.0, direction="higher_better", description="Two-point percentage"),
    RatingMetricSpec("MINUTES", "minutes_per_game", "explanation_only", 0.0, direction="contextual", description="Minutes per game"),
    RatingMetricSpec("GAMES", "games_played", "explanation_only", 0.0, direction="contextual", description="Games played"),
    RatingMetricSpec("STARTER%", "starter_pct", "explanation_only", 0.0, direction="contextual", description="Starter percentage"),
    RatingMetricSpec("PLUS_MINUS", "plus_minus", "explanation_only", 0.0, direction="contextual", description="Plus/minus"),
    RatingMetricSpec("PTS_PER_40", "pts_per_40", "explanation_only", 0.0, direction="contextual", description="Points per 40"),
    RatingMetricSpec("AST_PER_40", "ast_per_40", "explanation_only", 0.0, direction="contextual", description="Assists per 40"),
)



@dataclass(frozen=True)
class MetricEvidence:
    metric: str
    value: Optional[float]
    percentile: Optional[float]
    label: Optional[str]
    dimension: str
    weight: float
    sample_size: int
    population_source: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": None if self.value is None else round(float(self.value), 4),
            "percentile": None if self.percentile is None else round(float(self.percentile), 4),
            "label": self.label,
            "dimension": self.dimension,
            "weight": round(float(self.weight), 4),
            "sample_size": int(self.sample_size),
            "population_source": self.population_source,
            "direction": self.direction,
            "meaning": METRIC_SEMANTICS.get(
                self.metric,
                None,
            ).meaning if METRIC_SEMANTICS.get(self.metric) else self.metric,
            "interpretation": metric_interpretation(
                METRIC_SEMANTICS[self.metric],
                self.value,
                self.percentile,
            ) if self.metric in METRIC_SEMANTICS
              and self.value is not None
              and self.percentile is not None
            else None,
        }


@dataclass(frozen=True)
class BBRatingResult:
    player_global_id: str
    player_name: Optional[str]
    league: str
    season: int
    phase: str
    score: int
    score_band: str
    dimensions: dict[str, int]
    metrics: dict[str, MetricEvidence]
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    explanation: str
    peer_group: dict[str, Any]
    quality: str
    metric_coverage: float
    version: str = BB_RATING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_global_id": self.player_global_id,
            "player_name": self.player_name,
            "league": self.league,
            "season": self.season,
            "phase": self.phase,
            "bb_rating": self.score,
            "score_band": self.score_band,
            "dimensions": dict(self.dimensions),
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "explanation": self.explanation,
            "peer_group": dict(self.peer_group),
            "quality": self.quality,
            "metric_coverage": round(float(self.metric_coverage), 4),
            "bb_rating_version": self.version,
        }


class BBRatingEngine:
    """Deterministic contextual rating over an in-memory canonical stats frame."""

    def __init__(
        self,
        player_stats: pd.DataFrame,
        players: Optional[pd.DataFrame] = None,
        *,
        min_peer_samples: int = MIN_PEER_SAMPLES,
    ) -> None:
        self.min_peer_samples = int(max(5, min_peer_samples))
        self.frame = self._prepare_frame(player_stats, players)
        self._population_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    @staticmethod
    def _prepare_frame(
        player_stats: pd.DataFrame,
        players: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        frame = player_stats.copy()
        if frame.empty:
            return frame

        if "league_key" not in frame.columns:
            if "league_id" in frame.columns:
                frame["league_key"] = frame["league_id"].astype(str)
            else:
                frame["league_key"] = pd.Series("UNKNOWN", index=frame.index, dtype=object)
        frame["league_key"] = frame["league_key"].astype(str).str.strip().str.upper()

        if "season" in frame.columns:
            frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        else:
            frame["season"] = pd.Series(np.nan, index=frame.index, dtype=float)
        if "competition" not in frame.columns:
            frame["competition"] = "RS"
        frame["competition"] = (
            frame["competition"]
            .fillna("RS")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if players is not None and not players.empty and "player_id" in frame.columns:
            metadata_columns = [
                column
                for column in ("id", "global_id", "name", "age", "position")
                if column in players.columns
            ]
            if metadata_columns:
                meta = players[metadata_columns].copy()
                if "id" in meta.columns:
                    meta = meta.rename(columns={"id": "_player_meta_id"})
                    frame = frame.merge(
                        meta,
                        left_on="player_id",
                        right_on="_player_meta_id",
                        how="left",
                        suffixes=("", "_player"),
                    )
                    frame = frame.drop(columns=["_player_meta_id"], errors="ignore")
                    for column in ("global_id", "name", "age", "position"):
                        player_column = f"{column}_player"
                        if player_column in frame.columns:
                            if column in frame.columns:
                                frame[column] = frame[column].where(
                                    frame[column].notna()
                                    & frame[column].astype(str).ne(""),
                                    frame[player_column],
                                )
                                frame = frame.drop(columns=[player_column])
                            else:
                                frame = frame.rename(columns={player_column: column})

        if "player_global_id" not in frame.columns:
            frame["player_global_id"] = frame.get("global_id", pd.Series(index=frame.index, dtype=object))
        frame["player_global_id"] = frame["player_global_id"].astype(str)

        for column in ("position", "ruolo_combinato"):
            if column not in frame.columns:
                frame[column] = ""

        frame["position_family"] = frame["position"].map(_position_family)
        frame["age"] = pd.to_numeric(frame.get("age"), errors="coerce")
        frame["age_band"] = frame["age"].map(_age_band)

        return frame

    @staticmethod
    def _normalise_value(column: str, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if not np.isfinite(number):
            return float("nan")
        if column in RATE_COLUMNS and abs(number) > 1.0:
            number /= 100.0
        return number

    def _context_frame(self, league: str, season: int, phase: str) -> pd.DataFrame:
        key = ("context", league, int(season), phase)
        if key not in self._population_cache:
            mask = (
                (self.frame["league_key"] == league)
                & (self.frame["season"] == int(season))
                & (self.frame["competition"] == phase)
            )
            self._population_cache[key] = self.frame.loc[mask].copy()
        return self._population_cache[key]

    def _peer_population(
        self,
        context: pd.DataFrame,
        *,
        position_family: str,
        age_band: str,
        role: str,
    ) -> tuple[pd.DataFrame, str]:
        if context.empty:
            return context, "none"

        candidates = context
        if role:
            role_mask = (
                candidates["position_family"].eq(position_family)
                & candidates["age_band"].eq(age_band)
                & candidates["ruolo_combinato"].astype(str).str.strip().eq(role)
            )
            role_group = candidates.loc[role_mask]
            if len(role_group) >= self.min_peer_samples:
                return role_group, "position+age+role"

        pa_group = candidates.loc[
            candidates["position_family"].eq(position_family)
            & candidates["age_band"].eq(age_band)
        ]
        if len(pa_group) >= self.min_peer_samples:
            return pa_group, "position+age"

        position_group = candidates.loc[candidates["position_family"].eq(position_family)]
        if len(position_group) >= self.min_peer_samples:
            return position_group, "position"

        if len(context) >= MIN_CONTEXT_SAMPLES:
            return context, "league+season+phase"
        return context, "limited_context"

    def _percentile(self, values: pd.Series, value: float, direction: str) -> Optional[float]:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if clean.size == 0 or not np.isfinite(value):
            return None
        clean.sort()
        right = int(np.searchsorted(clean, value, side="right"))
        left = int(np.searchsorted(clean, value, side="left"))
        # Mid-rank ties provide a stable percentile for repeated zero/average
        # values while retaining the empirical interpretation.
        pct = (left + right) / 2.0 / float(clean.size)
        pct = float(np.clip(pct, 0.0, 1.0))
        if direction == "lower_better":
            pct = 1.0 - pct
        return pct

    @staticmethod
    def _band(percentile: Optional[float]) -> Optional[str]:
        if percentile is None:
            return None
        result = BAND_LABELS[0][1]
        for lower, label in BAND_LABELS:
            if percentile >= lower:
                result = label
        return result

    @staticmethod
    def _score(percentile: float) -> int:
        return int(np.clip(np.rint(1.0 + 99.0 * percentile), 1, 100))

    def _dimension_scores(
        self,
        evidences: Iterable[MetricEvidence],
    ) -> dict[str, int]:
        grouped: dict[str, list[tuple[float, float]]] = {}
        for evidence in evidences:
            if evidence.percentile is None:
                continue
            grouped.setdefault(evidence.dimension, []).append(
                (float(evidence.percentile), float(evidence.weight))
            )
        out: dict[str, int] = {}
        for dimension, values in grouped.items():
            total_weight = sum(weight for _, weight in values)
            if total_weight <= 0:
                continue
            pct = sum(percentile * weight for percentile, weight in values) / total_weight
            out[dimension] = self._score(pct)
        return out

    @staticmethod
    def _quality(peer_sample: int, coverage: float) -> str:
        if peer_sample >= 50 and coverage >= 0.80:
            return "high"
        if peer_sample >= 25 and coverage >= 0.60:
            return "medium"
        if peer_sample >= 10 and coverage >= 0.40:
            return "low"
        return "insufficient"

    def rate_player(
        self,
        player_global_id: str,
        *,
        league: str,
        season: int,
        phase: str,
    ) -> BBRatingResult:
        league_key = str(league).strip().upper()
        phase_key = str(phase).strip().upper()
        season_year = int(str(season).split("-")[0])

        context = self._context_frame(league_key, season_year, phase_key)
        if context.empty:
            raise ValueError(
                f"No BB-Rating context for league={league_key} "
                f"season={season_year} phase={phase_key}"
            )

        matches = context.loc[context["player_global_id"].astype(str) == str(player_global_id)]
        if matches.empty:
            raise ValueError("Player has no row in the requested BB-Rating context")

        player = matches.iloc[0]
        position_family = str(player.get("position_family", "OTHER") or "OTHER")
        age_band = str(player.get("age_band", "") or "")
        role = str(player.get("ruolo_combinato", "") or "").strip()
        peer, peer_source = self._peer_population(
            context,
            position_family=position_family,
            age_band=age_band,
            role=role,
        )

        evidences: list[MetricEvidence] = []
        for spec in BB_RATING_METRICS:
            if spec.source_column not in peer.columns:
                evidences.append(
                    MetricEvidence(
                        spec.key, None, None, None, spec.dimension, spec.weight,
                        0, "missing_column", spec.direction
                    )
                )
                continue

            raw_value = self._normalise_value(spec.source_column, player.get(spec.source_column))
            if not np.isfinite(raw_value):
                evidences.append(
                    MetricEvidence(
                        spec.key, None, None, None, spec.dimension, spec.weight,
                        0, peer_source, spec.direction
                    )
                )
                continue

            peer_values = peer[spec.source_column].map(
                lambda value: self._normalise_value(spec.source_column, value)
            )
            valid = peer_values.replace([np.inf, -np.inf], np.nan).dropna()
            percentile = self._percentile(valid, raw_value, spec.direction)
            evidences.append(
                MetricEvidence(
                    spec.key,
                    raw_value,
                    percentile,
                    self._band(percentile),
                    spec.dimension,
                    spec.weight,
                    int(valid.shape[0]),
                    peer_source,
                    spec.direction,
                )
            )

        usable = [e for e in evidences if e.percentile is not None]
        total_weight = sum(spec.weight for spec in BB_RATING_METRICS)
        used_weight = sum(e.weight for e in usable)
        coverage = float(used_weight / total_weight) if total_weight else 0.0

        if not usable:
            raise ValueError("BB-Rating cannot be computed: no usable metrics in context")

        composite_pct = sum(
            float(e.percentile) * float(e.weight) for e in usable
        ) / used_weight
        score = self._score(composite_pct)
        dimensions = self._dimension_scores(usable)
        quality = self._quality(len(peer), coverage)

        strength_candidates = sorted(
            (
                e
                for e in usable
                if e.weight > 0.0 and e.percentile >= 0.75
            ),
            key=lambda e: (float(e.percentile), float(e.weight)),
            reverse=True,
        )
        limitation_candidates = sorted(
            (
                e
                for e in usable
                if e.weight > 0.0 and e.percentile <= 0.35
            ),
            key=lambda e: (float(e.percentile), -float(e.weight)),
        )

        strengths = tuple(
            _strength_text(e.metric, float(e.percentile), position_family, age_band)
            for e in strength_candidates[:3]
        )
        limitations = tuple(
            _limitation_text(e.metric, float(e.percentile))
            for e in limitation_candidates[:3]
        )

        player_name = player.get("name")
        if player_name is not None and (pd.isna(player_name) or str(player_name).strip() == ""):
            player_name = None

        impact = dimensions.get("impact")
        offense = dimensions.get("offense")
        defense = dimensions.get("defense")
        versatility = dimensions.get("versatility")
        usg_percentile = next(
            (
                float(e.percentile)
                for e in usable
                if e.metric == "USG%" and e.percentile is not None
            ),
            None,
        )
        explanation = _build_explanation(
            score,
            impact=impact,
            offense=offense,
            defense=defense,
            versatility=versatility,
            quality=quality,
            role=role,
            usg_percentile=usg_percentile,
        )

        metrics_dict = {e.metric: e for e in evidences}
        return BBRatingResult(
            player_global_id=str(player_global_id),
            player_name=None if player_name is None else str(player_name),
            league=league_key,
            season=season_year,
            phase=phase_key,
            score=score,
            score_band=self._band(composite_pct) or "Average",
            dimensions=dimensions,
            metrics=metrics_dict,
            strengths=strengths,
            limitations=limitations,
            explanation=explanation,
            peer_group={
                "position_family": position_family,
                "age_band": age_band or None,
                "role": role or None,
                "definition": peer_source,
                "sample_size": int(len(peer)),
                "context_sample_size": int(len(context)),
            },
            quality=quality,
            metric_coverage=coverage,
        )


def _position_family(value: Any) -> str:
    token = str(value or "").strip().upper()
    if not token:
        return "OTHER"
    primary = token.split("/", 1)[0]
    if primary in {"PG", "SG"}:
        return "GUARD"
    if primary == "SF":
        return "WING"
    if primary in {"PF", "C"}:
        return "BIG"
    return token


def _age_band(value: Any) -> str:
    try:
        age = int(round(float(value)))
    except (TypeError, ValueError):
        return ""
    for lower, upper, label in AGE_BANDS:
        if lower <= age <= upper:
            return label
    return "34+" if age >= 34 else ""


def _strength_text(metric: str, percentile: float, position_family: str, age_band: str) -> str:
    place = f"nel gruppo {position_family.lower()}"
    if age_band:
        place += f" {age_band}"
    return f"{metric} si colloca al {round(percentile * 100)}° percentile {place}."


def _limitation_text(metric: str, percentile: float) -> str:
    return f"{metric} è al {round(percentile * 100)}° percentile del gruppo di confronto."


def _build_explanation(
    score: int,
    *,
    impact: Optional[int],
    offense: Optional[int],
    defense: Optional[int],
    versatility: Optional[int],
    quality: str,
    role: str,
    usg_percentile: Optional[float] = None,
) -> str:
    parts = [
        f"Il BB-Rating di {score}/100 sintetizza le performance osservate nel contesto richiesto.",
    ]
    available = {
        "Impact": impact,
        "Offense": offense,
        "Defense": defense,
        "Versatility": versatility,
    }
    available = {key: value for key, value in available.items() if value is not None}
    if available:
        strongest = max(available.items(), key=lambda item: item[1])
        weakest = min(available.items(), key=lambda item: item[1])
        parts.append(
            f"La dimensione più forte è {strongest[0]} ({strongest[1]}/100), "
            f"mentre quella relativamente più contenuta è {weakest[0]} ({weakest[1]}/100)."
        )
    if role:
        parts.append(
            f"Il confronto delle metriche usa anche il ruolo osservato '{role}' "
            "quando la numerosità lo consente."
        )
    if usg_percentile is not None and usg_percentile >= 0.90:
        parts.append(
            f"Il suo USG% si colloca al {round(usg_percentile * 100)}° percentile: "
            "indica un coinvolgimento offensivo molto alto; questo segnale non "
            "aumenta il BB-Rating da solo e viene letto insieme all'efficienza."
        )
    if quality != "high":
        parts.append(f"La qualità del confronto è {quality}: il risultato va letto con maggiore cautela.")
    return " ".join(parts)


__all__ = [
    "BB_RATING_VERSION",
    "BB_RATING_METRICS",
    "BBRatingEngine",
    "BBRatingResult",
    "MetricEvidence",
]

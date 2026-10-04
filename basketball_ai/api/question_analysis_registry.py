"""Question-to-analysis capability registry for Player Intelligence.

This module is deliberately declarative. It does not calculate ratings or alter
Prediction/BB-Rating models; it describes which evidence is available to answer
common basketball questions and which analysis layers are still missing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


CapabilityStatus = Literal["available", "partial", "missing"]


@dataclass(frozen=True)
class QuestionCapability:
    key: str
    question: str
    status: CapabilityStatus
    analyses: tuple[str, ...]
    missing_analyses: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    limitation: str | None = None


QUESTION_CAPABILITIES: tuple[QuestionCapability, ...] = (
    QuestionCapability(
        key="current_level",
        question="Come sta giocando il giocatore?",
        status="available",
        analyses=("bb_rating", "performance_vs_expectation"),
        evidence=("observed_context_rating", "expected_rating"),
    ),
    QuestionCapability(
        key="why_performing",
        question="Perché sta giocando meglio o peggio?",
        status="partial",
        analyses=("bb_rating", "performance_vs_expectation", "performance_decomposition"),
        missing_analyses=("metric_explanation",),
        evidence=("metric_strengths", "metric_limitations", "expectation_gap"),
    ),
    QuestionCapability(
        key="change_vs_last_season",
        question="Cosa è cambiato rispetto all'anno scorso?",
        status="available",
        analyses=("player_trend", "bb_rating", "performance_decomposition", "role_analysis"),
        missing_analyses=(),
        evidence=("historical_metrics", "role_history"),
    ),
    QuestionCapability(
        key="real_improvement",
        question="È migliorato davvero o gioca semplicemente più minuti?",
        status="available",
        analyses=("player_trend", "performance_decomposition"),
        evidence=("minutes", "per36_production", "efficiency", "impact"),
    ),
    QuestionCapability(
        key="current_role",
        question="Che ruolo sta realmente avendo?",
        status="available",
        analyses=("role_analysis", "player_trend"),
        evidence=("usage", "creation", "position", "dbpm"),
    ),
    QuestionCapability(
        key="role_fit",
        question="Il suo rendimento è coerente con il ruolo?",
        status="available",
        analyses=("role_analysis", "bb_rating", "performance_vs_expectation", "performance_decomposition"),
        evidence=("role", "rating", "role_metrics"),
    ),
    QuestionCapability(
        key="team_usage",
        question="La squadra lo sta utilizzando bene?",
        status="available",
        analyses=("compatibility", "role_analysis", "performance_vs_expectation", "team_usage_analysis"),
        evidence=("team_style", "role", "compatibility", "performance_gap"),
        limitation="Il fit descrive compatibilità; non dimostra un effetto causale dell'utilizzo.",
    ),
    QuestionCapability(
        key="stability",
        question="È affidabile o sta vivendo un picco?",
        status="available",
        analyses=("player_trend", "bb_rating", "performance_stability"),
        evidence=("historical_ratings", "trend"),
    ),
    QuestionCapability(
        key="potential",
        question="Qual è il suo potenziale?",
        status="partial",
        analyses=("prediction", "future_performance", "age_trajectory", "bb_rating"),
        missing_analyses=("potential_synthesis",),
        evidence=("next_season_prediction", "future_stat_profile", "age_curve"),
    ),
    QuestionCapability(
        key="regression_risk",
        question="Quanto è a rischio di regressione?",
        status="available",
        analyses=("bb_rating", "prediction", "future_performance", "performance_stability", "regression_risk"),
        evidence=("uncertainty", "trend", "historical_distribution"),
    ),
    QuestionCapability(
        key="team_counterfactual",
        question="Come renderebbe in un'altra squadra?",
        status="partial",
        analyses=("compatibility", "player_team", "league_transfer"),
        missing_analyses=("causal_team_effect",),
        evidence=("team_fit", "what_if_prediction"),
        limitation="Il sistema può stimare fit e scenario; non va presentato come effetto causale certo.",
    ),
    QuestionCapability(
        key="causal_team_effect",
        question="La squadra sta causando il suo rendimento?",
        status="missing",
        analyses=("compatibility",),
        missing_analyses=("causal_team_effect",),
        evidence=("team_context", "player_context"),
        limitation="Serve un disegno causale con confondenti e controfattuale identificabile.",
    ),
    QuestionCapability(
        key="shot_profile",
        question="Perché il suo rendimento al tiro è cambiato?",
        status="available",
        analyses=("shot_profile_counterfactual", "player_trend", "shooting_decomposition"),
        evidence=("shot_profile", "efficiency"),
    ),
    QuestionCapability(
        key="defensive_reason",
        question="Perché la sua difesa è migliorata o peggiorata?",
        status="available",
        analyses=("defensive_matchup", "bb_rating", "player_trend", "defensive_decomposition"),
        evidence=("dbpm", "defensive_matchup", "trend"),
    ),
    QuestionCapability(
        key="clutch_value",
        question="Quanto vale nei momenti decisivi?",
        status="available",
        analyses=("clutch_analysis",),
        evidence=("clutch_metrics",),
    ),
)


def capability_dicts() -> list[dict]:
    return [asdict(item) for item in QUESTION_CAPABILITIES]


def get_capability(key: str) -> QuestionCapability | None:
    return next((item for item in QUESTION_CAPABILITIES if item.key == key), None)


def registry_summary() -> dict[str, int]:
    return {
        "total": len(QUESTION_CAPABILITIES),
        "available": sum(item.status == "available" for item in QUESTION_CAPABILITIES),
        "partial": sum(item.status == "partial" for item in QUESTION_CAPABILITIES),
        "missing": sum(item.status == "missing" for item in QUESTION_CAPABILITIES),
    }

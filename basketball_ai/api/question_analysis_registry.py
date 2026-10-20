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
    required_data: tuple[str, ...] = ()
    optional_data: tuple[str, ...] = ()
    minimum_evidence: tuple[str, ...] = ()
    limitation: str | None = None


QUESTION_CAPABILITIES: tuple[QuestionCapability, ...] = (
    QuestionCapability(
        key="current_level",
        question="Come sta giocando il giocatore?",
        status="available",
        analyses=("player_competition", "metric_explanation", "model_evidence"),
        evidence=("canonical_bb_rating_1_100", "calibrated_expectation_if_available"),
    ),
    QuestionCapability(
        key="why_performing",
        question="Perché sta giocando meglio o peggio?",
        status="available",
        analyses=("performance_decomposition", "metric_explanation", "model_evidence"),
        evidence=("metric_strengths", "metric_limitations", "expectation_gap"),
    ),
    QuestionCapability(
        key="change_vs_last_season",
        question="Cosa è cambiato rispetto all'anno scorso?",
        status="available",
        analyses=("performance_decomposition", "role_analysis", "metric_explanation", "model_evidence"),
        missing_analyses=(),
        evidence=("historical_metrics", "role_history"),
    ),
    QuestionCapability(
        key="real_improvement",
        question="È migliorato davvero o gioca semplicemente più minuti?",
        status="available",
        analyses=("performance_decomposition", "metric_explanation", "model_evidence"),
        evidence=("minutes", "per36_production", "efficiency", "impact"),
    ),
    QuestionCapability(
        key="current_role",
        question="Che ruolo sta realmente avendo?",
        status="available",
        analyses=("role_analysis", "model_evidence"),
        evidence=("usage", "creation", "position", "dbpm"),
    ),
    QuestionCapability(
        key="role_fit",
        question="Il suo rendimento è coerente con il ruolo?",
        status="available",
        analyses=("role_analysis", "performance_decomposition", "model_evidence"),
        evidence=("observed_role", "canonical_bb_rating_1_100", "role_metrics"),
    ),
    QuestionCapability(
        key="team_usage",
        question="La squadra lo sta utilizzando bene?",
        status="available",
        analyses=("role_analysis", "performance_decomposition", "team_usage_analysis", "metric_explanation", "model_evidence"),
        evidence=("team_style", "role_and_usage_changes", "team_context", "performance_decomposition"),
        limitation="La valutazione richiede un contesto squadra-stagione risolvibile; descrive l'allineamento osservato e non dimostra che l'utilizzo abbia causato il rendimento.",
    ),
    QuestionCapability(
        key="stability",
        question="È affidabile o sta vivendo un picco?",
        status="available",
        analyses=("performance_stability", "model_evidence"),
        evidence=("historical_bb_rating_1_100", "trend"),
    ),
    QuestionCapability(
        key="potential",
        question="Qual è il suo potenziale?",
        status="available",
        analyses=("potential_synthesis", "age_trajectory", "model_evidence"),
        evidence=("observed_bb_rating_trajectory_1_100", "age_curve"),
        limitation="Il potenziale sintetizza traiettoria osservata ed età; le previsioni numeriche future restano responsabilità dei modelli Prediction/Future Performance.",
    ),
    QuestionCapability(
        key="regression_risk",
        question="Quanto è a rischio di regressione?",
        status="available",
        analyses=("regression_risk", "model_evidence"),
        evidence=("uncertainty", "trend", "historical_distribution"),
    ),
    QuestionCapability(
        key="team_counterfactual",
        question="Come renderebbe in un'altra squadra?",
        status="available",
        analyses=("player_team", "model_evidence"),
        evidence=("team_fit", "what_if_prediction"),
        limitation="Richiede una squadra target esplicita e un contesto storico valido per la stagione richiesta. Se la lega cambia, il player-team scenario usa la transizione storica cross-league; non è un effetto causale certo e non va interpretato come BB-Rating osservato.",
    ),
    QuestionCapability(
        key="performance_attribution",
        question="Il cambiamento dipende dal giocatore, dal ruolo o dal contesto della squadra?",
        status="partial",
        analyses=(
            "performance_decomposition",
            "role_analysis",
            "team_usage_analysis",
            "causal_team_effect",
            "lineup_context_association",
            "observed_teammate_lineup_associations",
            "metric_explanation",
            "model_evidence",
        ),
        missing_analyses=("causal_teammate_effect_attribution", "causal_identification"),
        evidence=(
            "season_over_season_player_metrics",
            "role_and_usage_changes",
            "observed_team_context_association",
            "within_team_season_lineup_net_rating",
            "teammate_lineup_presence_within_player_minutes",
        ),
        required_data=(
            "PlayerCompetitionStats with observed role, minutes and usage",
            "TeamCompetitionStats in the same league, season and competition",
            "canonical BB-Rating engine on the public 1-100 scale",
        ),
        optional_data=(
            "ScenarioLineupStats sourced from SimulationLineupStints",
        ),
        minimum_evidence=(
            "Team-context association: at least 5 valid BB-Rating/team-context seasons",
            "Adjusted association: at least max(8, design_columns + 3) complete observations and non-degenerate residuals",
            "Pooled lineup comparison: at least 2 matched team-seasons and 100 possessions per group",
            "Teammate comparison: at least 100 possessions in each teammate-presence group within a team-season",
        ),
        limitation=(
            "La composizione distingue cambiamenti individuali, contesto squadra e Net Rating osservato "
            "nei quintetti con/senza il giocatore. Non identifica da sola il contributo causale di compagni "
            "o giocatore; selezione dei quintetti e confondenti restano rilevanti."
        ),
    ),
    QuestionCapability(
        key="causal_team_effect",
        question="La squadra sta causando il suo rendimento?",
        status="partial",
        analyses=(
            "team_context_association",
            "team_switch_analysis",
            "bb_rating_context_adjustment",
            "lineup_context_association",
            "observed_teammate_lineup_associations",
            "model_evidence",
        ),
        missing_analyses=("causal_identification", "causal_teammate_effect_attribution"),
        evidence=(
            "canonical_bb_rating_1_100",
            "team_context",
            "team_switches",
            "minutes",
            "usage",
            "observed_role",
            "within_team_season_lineup_net_rating",
            "teammate_lineup_presence_within_player_minutes",
        ),
        required_data=(
            "PlayerCompetitionStats with canonical global player identity",
            "TeamCompetitionStats in the requested league and competition",
            "BB-Rating history from the canonical BB-Rating engine",
        ),
        optional_data=(
            "ScenarioLineupStats sourced from SimulationLineupStints",
        ),
        minimum_evidence=(
            "Team-context association: at least 5 valid BB-Rating/team-context seasons",
            "Adjusted association: enough complete observations for season, minutes, usage and role controls",
            "Pooled lineup comparison: at least 2 matched team-seasons and 100 possessions per group",
            "Teammate comparison: at least 100 possessions in each teammate-presence group within a team-season",
        ),
        limitation=(
            "L'esito individuale resta il BB-Rating canonico 1-100. Il confronto dei quintetti descrive "
            "il Net Rating di squadra osservato nelle lineup con/senza il giocatore, non il suo BB-Rating "
            "né un effetto individuale. Le associazioni per compagno confrontano il Net Rating della squadra "
            "nei quintetti con il giocatore e quel compagno insieme rispetto ai quintetti con il giocatore "
            "senza quel compagno; restano descrittive e possono risentire della selezione dei quintetti. "
            "Tutti i risultati restano osservazionali e non dimostrano causalità."
        ),
    ),
    QuestionCapability(
        key="shot_profile",
        question="Perché il suo rendimento al tiro è cambiato?",
        status="available",
        analyses=("performance_decomposition", "shooting_decomposition", "metric_explanation", "model_evidence"),
        evidence=("season_over_season_shooting_metrics", "rate_adjusted_production", "metric_context"),
    ),
    QuestionCapability(
        key="defensive_reason",
        question="Perché la sua difesa è migliorata o peggiorata?",
        status="available",
        analyses=("performance_decomposition", "role_analysis", "defensive_decomposition", "metric_explanation", "model_evidence"),
        evidence=("defensive_metric_changes", "role_and_usage_changes", "canonical_bb_rating_1_100", "metric_context"),
    ),
    QuestionCapability(
        key="clutch_value",
        question="Quanto vale nei momenti decisivi?",
        status="available",
        analyses=("clutch_analysis", "model_evidence"),
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

"""Typed Chat V3 scenario contract for data-backed basketball analysis."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from basketball_ai.models.competition_training import normalize_competition


SCENARIO_TYPES = {
    "player_competition",
    "team_competition",
    "player_trend",
    "performance_decomposition",
    "team_trend",
    "player_compare",
    "team_compare",
    "player_team",
    "playoff_role",
    "league_transfer",
    "player_pair",
    "lineup_fit",
    "style_change",
    "player_role_change",
    "role_minutes_projection",
    "clutch_analysis",
    "team_add_player",
    "team_replace_player",
    "best_team_fit",
    "best_player_fit",
    "player_similarity",
    "age_trajectory",
    "probabilistic_boxscore",
    "opponent_matchup",
    "defensive_matchup",
    "play_type_matchup",
    "shot_profile_counterfactual",
    "lineup_synergy",
    "lineup_optimizer",
    "roster_optimizer",
    "composite_scenario",
    "causal_effect",
}

_STYLE_KEYS = {
    "pace",
    "three_point_attempt_rate",
    "assists_per_game",
    "star_player_usage",
    "offensive_rating",
    "defensive_rating",
}
_PLAYER_OVERRIDE_KEYS = {"minutes_per_game", "usg_pct", "role"}


class ScenarioRequestV2(BaseModel):
    scenario: str = Field(min_length=1, max_length=64)
    player_global_ids: list[str] = Field(default_factory=list, max_length=30)
    team_global_ids: list[str] = Field(default_factory=list, max_length=8)
    source_league: str | None = Field(default=None, max_length=32)
    target_league: str | None = Field(default=None, max_length=32)
    season: int = Field(ge=2000, le=2100)
    competition: str = Field(default="RS", min_length=1, max_length=32)
    target_competition: str | None = Field(default=None, max_length=32)
    comparison_competition: str | None = Field(default=None, max_length=32)
    style_overrides: dict[str, float | str] = Field(default_factory=dict)
    player_overrides: dict[str, float | str] = Field(default_factory=dict)
    top_n: int = Field(default=5, ge=1, le=20)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("player_global_ids", "team_global_ids")
    @classmethod
    def unique_global_ids(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entity identifiers must be unique")
        return cleaned

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, value: str) -> str:
        scenario = value.strip().lower()
        if scenario not in SCENARIO_TYPES:
            raise ValueError(f"unsupported scenario {scenario!r}")
        return scenario

    @field_validator("competition", "target_competition", "comparison_competition")
    @classmethod
    def normalise_competitions(cls, value: str | None) -> str | None:
        return normalize_competition(value) if value is not None else None

    @field_validator("source_league", "target_league")
    @classmethod
    def clean_league(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("style_overrides")
    @classmethod
    def validate_style_overrides(cls, value: dict[str, float | str]) -> dict[str, float | str]:
        unknown = set(value) - _STYLE_KEYS
        if unknown:
            raise ValueError(f"unsupported style overrides: {sorted(unknown)}")
        for key, raw in value.items():
            if isinstance(raw, str) and raw.lower() not in {"higher", "lower", "same"}:
                raise ValueError(f"{key} must be numeric or higher/lower/same")
        return value

    @field_validator("player_overrides")
    @classmethod
    def validate_player_overrides(cls, value: dict[str, float | str]) -> dict[str, float | str]:
        unknown = set(value) - _PLAYER_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"unsupported player overrides: {sorted(unknown)}")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        # The planner contract remains bounded even though scenario-specific
        # parameters are extensible.
        if len(value) > 24:
            raise ValueError("too many scenario parameters")
        encoded = str(value)
        if len(encoded) > 12000:
            raise ValueError("scenario parameters are too large")
        if "simulations" in value and not 500 <= int(value["simulations"]) <= 50000:
            raise ValueError("simulations must be between 500 and 50000")
        if "seed" in value and not -(2**63) < int(value["seed"]) < 2**63:
            raise ValueError("seed is out of range")
        return value


class ScenarioResponseV2(BaseModel):
    model_run_id: str
    model_version: str
    feature_version: str
    data_cutoff: str
    scenario: str
    generated_at: datetime
    result: dict[str, Any]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    support: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

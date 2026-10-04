# Player Intelligence — Question / Analysis Registry

Player Intelligence is an interpretation layer above the existing basketball models.
It does not retrain or modify Prediction Model, BB-Rating, Compatibility, or the
Future Performance model.

## Capability statuses

- **available** — the requested question has a dedicated, usable evidence path.
- **partial** — existing evidence can answer part of the question, but one or more analysis layers are missing.
- **missing** — the current evidence is not sufficient for a responsible answer.

The registry is exposed at:

`GET /api/v2/player-intelligence/capabilities`

## Implemented analysis layers

1. **performance_decomposition** — volume, efficiency, role and impact changes.
2. **metric_explanation** — metric meaning plus same-context percentile and interpretation.
3. **role_analysis** — reconstructs observed role and role changes over time.
4. **performance_stability** — historical dispersion, trend and peak-vs-baseline.
5. **team_usage_analysis** — compares player role with team style and usage context.
6. **regression_risk** — descriptive risk index from baseline deviation and stability.
7. **shooting_decomposition** — season-over-season shooting profile changes.
8. **defensive_decomposition** — season-over-season defensive profile changes.
9. **potential_synthesis** — age + observed trajectory synthesis; it remains distinct from future forecasting.

## Still intentionally limited

- **causal_team_effect** remains unavailable as a causal claim. It requires explicit identification, confounder control and a defensible counterfactual design.
- **potential_synthesis** does not replace the dedicated Prediction/Future Performance models.
- **team_counterfactual** remains a fit/scenario estimate, not a causal statement.

## Design rule

A question must not be promoted from partial to available merely because a numeric estimate exists.
Compatibility and scenario prediction can describe or estimate fit; they do not by themselves establish that a team caused a player's observed performance.

This separation is intentional because basketball performance is affected by player, teammates, strategy, context and sampling variation, and causal questions require explicit counterfactual reasoning.

## Evidence orchestrator

The `player_intelligence` scenario now composes the appropriate analysis layers from a `question_key`. Examples:

- `why_performing` → performance decomposition + metric explanation
- `change_vs_last_season` → decomposition + role analysis
- `real_improvement` → decomposition + metric explanation
- `current_role` → role analysis
- `team_usage` → team usage analysis
- `stability` → performance stability
- `regression_risk` → regression risk
- `potential` → potential synthesis + age trajectory when team context is available
- `team_counterfactual` → player-in-team scenario
- `shot_profile` → shooting decomposition
- `defensive_reason` → defensive decomposition
- `clutch_value` → clutch analysis

This keeps the analytical layers independent while giving Chat V3 a single evidence-composition entry point.

## Final boundary

The only intentionally unavailable Player Intelligence question is a **causal team-effect claim**. A statement such as “Team X caused this player to perform worse” requires an explicit intervention/counterfactual design and control of confounding. Existing compatibility and scenario models must therefore be presented as modeled fit/scenario estimates, not causal effects.

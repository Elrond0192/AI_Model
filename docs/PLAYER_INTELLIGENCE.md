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

## First missing analysis layers

1. **performance_decomposition** — separates volume, minutes, efficiency, role and impact.
2. **role_analysis** — reconstructs role and role changes over time.
3. **performance_stability** — measures historical dispersion, trend and peak-vs-baseline.
4. **team_usage_analysis** — compares role, team style, expected and observed performance.
5. **regression_risk** — combines stability, sample size, uncertainty and recent deviation.
6. **potential_synthesis** — combines current BB-Rating, Prediction, Future Performance and age trajectory.
7. **causal_team_effect** — reserved for a future causal design with explicit confounders and counterfactual identification.

## Design rule

A question must not be promoted from partial to available merely because a numeric estimate exists.
Compatibility and scenario prediction can describe or estimate fit; they do not by themselves establish that a team caused a player's observed performance.

This separation is intentional because basketball performance is affected by player, teammates, strategy, context and sampling variation, and causal questions require explicit counterfactual reasoning.
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

- **causal_team_effect** is now partially answerable: it measures the observed association between a player’s team context and his own rating across team-season observations, including team switches when available. The causal effect itself remains unavailable because explicit identification, confounder control and a defensible counterfactual design are still missing.
- **potential_synthesis** does not replace the dedicated Prediction/Future Performance models.
- **team_counterfactual** remains a fit/scenario estimate, not a causal statement.

## Design rule

A question must not be promoted from partial to available merely because a numeric estimate exists.
Compatibility and scenario prediction can describe or estimate fit; they do not by themselves establish that a team caused a player's observed performance.

This separation is intentional because basketball performance is affected by player, teammates, strategy, context and sampling variation, and causal questions require explicit counterfactual reasoning.

## Multi-question composition

The `player_intelligence` scenario accepts either a single `question_key` or an ordered `question_keys` list (up to 8). Multiple keys are deduplicated and their analysis layers are composed into one evidence bundle. The response exposes `question_keys` and sets `question_key` to null when multiple intents are requested.

This is intended for Chat V3 questions such as "is he improving, what is his role, and how stable is he?" without requiring independent scenario calls for each sub-question.

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
- `causal_team_effect` → team-context association + observed team-switch analysis

This keeps the analytical layers independent while giving Chat V3 a single evidence-composition entry point.

## Historical season comparison across league contexts

When the selected player has multiple current league contexts, Player Intelligence keeps the historical comparison isolated by league. The player identity is expanded through `global_id`, so a previous season can be resolved even when its internal `player_id` differs from the current season.

For season-over-season questions such as `change_vs_last_season`, the multi-context response exposes:

- one `season_comparison` object per league context;
- `previous_season`, `current_season`, `previous_rating`, `current_rating` and `rating_delta`;
- a top-level `season_comparisons` collection ordered with the discovered league contexts.

Chat V3 should therefore report the comparison explicitly per league (for example, "in ITA1 ... while in EL ...") whenever more than one context has a valid consecutive-season series. Metrics from different leagues must never be merged into a single season comparison.

## Final boundary

The exact **causal team-effect claim** remains unavailable, but Chat V3 can now answer the user-facing question with an observational attribution. A statement such as “Team X caused this player to perform worse” is not supported; instead the response reports the player’s observed rating, the team-context association, observed team switches, and the remaining causal limitation.

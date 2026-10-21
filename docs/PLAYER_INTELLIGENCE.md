# Player Intelligence — Question / Analysis Registry

Player Intelligence is an interpretation layer above the existing basketball models.
It does not retrain or modify Prediction Model, BB-Rating, Compatibility, or the
Future Performance model.

## Capability statuses

- **available** — the requested question has a dedicated, usable evidence path.
- **partial** — existing evidence can answer part of the question, but one or more analysis layers are missing.
- **missing** — the current evidence is not sufficient for a responsible answer.

The registry is exposed at:

`GET /api/v2/player-intelligence/capabilities` (metadata contract v1.1).

For attribution capabilities, the registry includes `required_data`, `optional_data` and `minimum_evidence`. These describe source dependencies and numeric support gates; they do not claim that a source is populated in every requested league/season. Runtime responses must still report `unavailable` or `insufficient_evidence` when the requested context lacks data.

## Implemented analysis layers

1. **performance_decomposition** — volume, efficiency, role and impact changes; includes per-36-minute production so role/minute changes can be separated descriptively from rate-adjusted production.
2. **metric_explanation** — metric meaning plus same-context percentile and interpretation.

For season-over-season comparisons, the public rating is always the canonical BB-Rating (1–100), never the source-statistics `rating` field. A comparison for the active calendar-year season is marked provisional and must not be presented as a completed-season result. Per-36-minute rates help describe volume changes; they are not causal attribution and do not by themselves explain a BB-Rating delta.
3. **role_analysis** — reconstructs observed role and role changes over time.
4. **performance_stability** — historical dispersion, trend and peak-vs-baseline.
5. **team_usage_analysis** — compares player role with team style and usage context.
6. **regression_risk** — descriptive risk index from baseline deviation and stability.
7. **shooting_decomposition** — season-over-season shooting profile changes.
8. **defensive_decomposition** — season-over-season defensive profile changes.
9. **potential_synthesis** — age + observed trajectory synthesis; it remains distinct from future forecasting.

## Still intentionally limited

- **performance_attribution** is available as a descriptive capability: it combines season-over-season BB-Rating decomposition, role/usage changes, team-style alignment, observed team-context association and, when data are present, lineup-presence association. The API loads bounded lineup stints only for attribution questions and restricts them to the player's observed team-season contexts in the recent four-season window. It compares team Net Rating in lineups with versus without the player within the same team-season, reporting possessions and per-season support. Within lineups containing the player, it compares stints with a teammate present versus absent; named teammates are included only where identity lookup is available. Teammate comparisons are returned per team-season, and numeric differences are suppressed unless both groups have at least 100 possessions. These remain descriptive team outcomes, not the player's BB-Rating or isolated individual contribution. The separate causal_team_effect capability remains partial: a guarded team-switch cohort ATT is now computed only when sample size, propensity overlap and weighted covariate balance pass; it is explicitly cohort-level and conditional on observational assumptions, never a causal effect for the named player.
- **causal_team_effect** is partially answerable: the individual outcome is the canonical observed BB-Rating (1–100), never the source-statistics rating. It measures association between team context and BB-Rating across player-season observations, reports BB-Rating changes at team switches, and shows changes in minutes, usage and role. When enough history exists, it computes an exploratory partial correlation after controlling for season trend, minutes, usage and observed role. When lineup data are available, it also compares observed team Net Rating for lineups with versus without the player within matching team-season contexts. Each layer states its outcome and support explicitly; insufficient evidence does not produce a pooled numeric conclusion. The causal effect remains unavailable because lineup selection is endogenous, some confounders are unobserved and a defensible counterfactual design is not identified.
- **potential_synthesis** does not replace the dedicated Prediction/Future Performance models.
- **team_counterfactual** is composed through `player_team`; that scenario internally delegates to the cross-league transition layer when the requested source and target leagues differ. It requires an explicit target team and usable historical target-team context. The common model-evidence layer may also expose BB Prediction/Future Performance when forward evidence is requested, but Future Performance itself is not conditioned on the target team. The counterfactual remains an estimate, not a causal statement.

## Causal attribution readiness

The implementation gate, proposed estimands, panel requirements, identification diagnostics and promotion criteria for `performance_attribution` / `causal_team_effect` are specified in [`CAUSAL_PLAYER_ATTRIBUTION_ROADMAP.md`](CAUSAL_PLAYER_ATTRIBUTION_ROADMAP.md). Keep both capabilities `partial` until the defined causal design is implemented and passes out-of-time, balance/overlap, placebo and sensitivity checks. The existing lineup comparisons remain descriptive and must not be presented as causal effects.

## Design rule

A question must not be promoted from partial to available merely because a numeric estimate exists.
Compatibility and scenario prediction can describe or estimate fit; they do not by themselves establish that a team caused a player's observed performance.

This separation is intentional because basketball performance is affected by player, teammates, strategy, context and sampling variation, and causal questions require explicit counterfactual reasoning.

## Multi-question composition

The `player_intelligence` scenario accepts either a single `question_key` or an ordered `question_keys` list (up to 8). Multiple keys are deduplicated and their analysis layers are composed into one evidence bundle. The response exposes `question_keys` and sets `question_key` to null when multiple intents are requested.

Every selected question also receives the common `model_evidence` layer. For `potential`, `regression_risk` and `team_counterfactual`, that layer opts into the dedicated Future Performance and BB Prediction Model outputs; for current/observed questions it remains descriptive. The registry lists the dedicated layers actually composed rather than treating a skipped placeholder (for example, an uncalled standalone compatibility layer) as executed evidence.

This is intended for Chat V3 questions such as "is he improving, what is his role, and how stable is he?" without requiring independent scenario calls for each sub-question.

## Evidence orchestrator

The `player_intelligence` scenario now composes the appropriate analysis layers from a `question_key`. Examples:

- `why_performing` → performance decomposition + metric explanation
- `performance_attribution` → decomposition + role analysis + team usage + observed team-context association + metric explanation; teammate/lineup attribution remains explicitly unsupported
- `change_vs_last_season` → decomposition + role analysis
- `real_improvement` → decomposition + metric explanation
- `current_role` → role analysis
- `team_usage` → team usage analysis
- `stability` → performance stability
- `regression_risk` → regression risk
- `potential` → potential synthesis + age trajectory when team context is available
- `team_counterfactual` → player-in-team scenario
- `shot_profile` → season-over-season performance decomposition + shooting decomposition + metric explanation
- `defensive_reason` → performance decomposition + observed role analysis + defensive decomposition + metric explanation
- `clutch_value` → observed `AdvanceStats_Clutch_<league_key>` aggregates (clutch games, points, 2P/3P/FT volume and efficiency, rebounds, assists, steals, blocks, turnovers, ORtg/DRtg/NetRtg and native ClutchnessRating when available). This path comes through `PlayerCompetitionStats`, not the optional `Scenario*` tables.
- `causal_team_effect` → team-context association + observed team-switch analysis

This keeps the analytical layers independent while giving Chat V3 a single evidence-composition entry point.

## Historical season comparison across league contexts

When the selected player has multiple current league contexts, Player Intelligence keeps the historical comparison isolated by league. The player identity is expanded through `global_id`, so a previous season can be resolved even when its internal `player_id` differs from the current season.

For season-over-season questions such as `change_vs_last_season`, the multi-context response exposes:

- one `season_comparison` object per league context;
- `previous_season`, `current_season`, `previous_rating`, `current_rating` and `rating_delta`;
- a top-level `season_comparisons` collection ordered with the discovered league contexts.

Chat V3 should therefore report the comparison explicitly per league (for example, "in ITA1 ... while in EL ...") whenever more than one context has a valid consecutive-season series. Metrics from different leagues must never be merged into a single season comparison.

When the canonical BB-Rating engine exposes a component breakdown, the
`performance_decomposition` result and its `season_comparison` also include
`bb_rating_contribution_decomposition`. This compares the arithmetic
contributions of scored metrics and dimensions between the two seasons, plus
the integer-rounding residual. Chat should prefer the readable
`display_name` labels, describe only the largest positive/negative component
deltas, and explicitly frame them as a mathematical score decomposition, not
as causes of a player's real-world performance change. Since percentiles are
relative to each season's own peer population and missing inputs can change
effective weights, component deltas may also reflect changes in the comparison
context. The scoring formula and versioned calibration are not changed.

## Final boundary

The exact **causal team-effect claim** remains unavailable, but Chat V3 can now answer the user-facing question with an observational attribution. A statement such as “Team X caused this player to perform worse” is not supported; instead the response reports the player’s observed rating, the team-context association, observed team switches, and the remaining causal limitation.


## Scouting & Player Fit

The scenario registry now includes `player_scouting`: a first-stage shortlist
that filters a single league/season/competition pool by position, nationality,
age and playing-time requirements, then ranks eligible candidates against a
transparent archetype profile (`playmaker`, `scorer`, `connector`,
`defender`, `big`, `balanced`). The output includes fit score, supporting
metric percentiles, strengths, sample support and explicit limitations.

Nationality values follow the database's basketball-code convention: `ITA` remains `ITA`, while `DEU` is normalized to `DE` as a configured Germany alias. Other three-letter codes are preserved unless an explicit alias is defined.

When a target team is supplied, Player Scouting attempts a roster-aware
Team Fit: it resolves season membership from TeamPlayerRelations, compares each
eligible candidate with every roster member who has usable statistics in the
same league/season/competition, estimates archetype needs, and adds an
indicative positional-depth signal. The result separates the individual
scouting score from the combined Team Fit score and reports roster coverage,
pair counts and missing evidence.

The fit status is deliberately explicit. `available` requires a season roster
relation and usable matching-context statistics for every resolved member.
`partial` is returned if the relation is missing or some members cannot be
evaluated; the candidate ranking may still be useful, but must not be called a
complete roster fit. `unavailable` means fewer than two roster members have
usable statistics. The scouting Team Fit remains a transparent analytical
ranking, not the acquisition-impact forecast.

For an explicit acquisition what-if, `team_add_player` and
`team_replace_player` now use a separate historical roster-transition model
with an expanding-window walk-forward evaluation. Numeric estimates are
returned only when the model has at least 40 historical transitions, at least
20 out-of-time predictions across at least three seasons, and improves on a
zero-change Net Rating baseline without excessive prediction bias. Otherwise
the output marks the impact model as unvalidated and leaves the numerical
acquisition estimate empty. Even when validation passes, it is an observational
forecast of team Net Rating change, not proof of the acquisition's causal
effect. Cross-league transferability, contract status and cost still require
additional evidence layers.


### Reuse of HoopmetricsEngine advanced roles

Player Scouting reads the already-calculated `ruolo_offensivo`, `ruolo_difensivo` and `ruolo_combinato` fields exposed through AI_Source. These are the authoritative source for primary archetype labels; the scouting layer maps known role labels to one or more readable archetypes without recomputing the upstream role classification. The original role labels are returned in `advanced_roles`, mapped labels in `archetypes_from_advanced_roles`, and unknown labels are preserved and reported rather than silently discarded. Team Fit also compares archetype coverage from these existing role labels across the target roster; where role fields are unavailable, it falls back to metric-derived need alignment. Statistical `archetype_profiles` remain supporting, cohort-relative evidence and are not a replacement for the advanced roles. If role fields are missing for a player/context, metric-derived profiles remain available and the role mapping coverage is zero.


## Lineup evidence and next validation gates

The first attribution extension uses the existing `ScenarioLineupStats` serving feed. It is requested only for `causal_team_effect` and Player Intelligence questions `performance_attribution` / `causal_team_effect`; ordinary profile requests do not load lineup data. Reads are bounded to the recent four-season window, competition and available league/team scope.

The observed outcome is team `net_rtg` by lineup, not the individual BB-Rating. The pooled with/without-player comparison is returned only when at least two matched team-seasons and at least 100 possessions in each group are available. Teammate-presence comparisons are scoped to a single team-season and to stints where the player is on court; each numeric difference requires at least 100 possessions in both teammate-presence groups. Otherwise, the per-season evidence (if any) is exposed as insufficient rather than turned into a headline effect. These thresholds are minimum evidence guards, not a claim of statistical significance.

Next validation gates remain: (1) measure the coverage of the serving feed by league/season and verify `player_ids` membership is complete; (2) quantify lineup selection bias and account for opponent, venue, teammate combinations and game state where fields exist; (3) perform out-of-time stability checks; (4) build a defensible comparison design before using causal language. Lineup-presence and teammate-presence associations do not satisfy those causal gates by themselves.


## Clutch source and optional scenario-feed coverage

Clutch aggregates are sourced from `Analisi.AdvanceStats_Clutch_<league_key>` (the older `AdvancedStats_Clutch_<league_key>` spelling remains a fallback) and joined to the matching player-season-competition row when `ai_source_full.sql` builds `AI_Source.PlayerCompetitionStats`. The projected fields include counting stats, shooting makes/attempts and percentages, ORtg/DRtg/NetRtg, and the source `ClutchnessRating` category. The latter is a source label, not the canonical BB-Rating scale.

The clutch analysis distinguishes a real source row from default zeros. If the detail fields are absent or null for the requested context, `result.status` is `unavailable` rather than presenting zeroes as observed clutch performance. It exposes the sample size and a per-clutch-game points rate when possible; ratios/ratings are flagged as unstable below five clutch games. The source is a seasonal aggregate for the last five minutes of close games, not possession-by-possession evidence.

Separately, `ai_scenario_serving.sql` copies data from optional `AI_Source.Simulation*` feeds into the physical `Scenario*` tables. It does not generate these upstream feeds. A successful refresh can therefore leave a Scenario table empty when the matching Simulation source has no rows for that league/season/competition. The refresh procedure now emits an explicit notice in this case. Do not fabricate ScenarioLineupStats, PBP matchups, play-type stats or shot profiles to fill that gap; those analyses must remain unavailable or prior-only until their real source feeds are populated.

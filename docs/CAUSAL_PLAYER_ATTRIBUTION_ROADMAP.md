# Causal player-performance attribution: implementation gate

## Decision

Do **not** promote `performance_attribution` or `causal_team_effect` from
`partial` to `available` yet. The current implementation can provide:
- canonical BB-Rating (1–100) history and mathematical score decomposition;
- observed changes in minutes, usage and role;
- associations between player-season BB-Rating and observed team context;
- descriptive team Net Rating comparisons across lineups with/without a player
  or a teammate, where possession support passes the existing minimum gates.

These are useful analyses, but none alone identifies the causal effect of a team,
teammate or lineup on an individual's BB-Rating. A numeric coefficient or an
AIPW estimate does not establish causality unless the treatment, outcome,
confounders, overlap and identifying assumptions are defensible.

## Current data readiness

The repository has a schema for AI_Source.SimulationCausalPanel, and the seasonal source-refresh design leaves it unpopulated. The first implemented estimator therefore builds an in-memory cohort from canonical player/team season sources rather than treating that schema-only table as evidence. This cohort is not yet a persisted, refresh-audited causal panel. The existing `causal_effect` scenario can fall back to a
constructed next-season panel based on changes in minutes; this is not an
acceptable source for team/teammate attribution and must not be reused as proof
of the requested causal effect.

The current lineup source can support descriptive stint-level comparisons. It
does not by itself provide a randomized assignment of teammates or lineups, and
the observed team Net Rating is not the player's BB-Rating. The public outcome
for player-performance attribution must remain the canonical BB-Rating 1–100;
never substitute the native source-statistics `rating`.

## Proposed estimands (keep separate)

1. **Team-switch effect (player-level ATT):** for players who switch teams,
   estimate the difference in subsequent BB-Rating under the observed move
   versus a credible counterfactual of the same player not switching, relative
   to comparable non-switchers in the same league/season/competition and with
   similar pre-treatment trajectories.
2. **Teammate exposure effect (within player/team-season):** estimate the
   effect of exposure to a named teammate on the focal player's subsequent
   individual outcome only if exposure can be defined before the outcome and
   there is adequate within-context variation. Shared minutes and lineup Net
   Rating are supporting exposure evidence, not the individual outcome.
3. **Lineup context effect:** estimate team Net Rating for a defined lineup
   exposure, as a team outcome. Do not describe it as an individual player's
   causal BB-Rating contribution.
4. **Role/minutes mediation:** keep this separate from the total team effect.
   Role and minutes may be mediators of a team move; adjusting for them in the
   primary total-effect model can remove part of the effect being estimated.
   Report a controlled/direct effect only as a separately defined secondary
   estimand.

## Candidate identification strategy

Start with a longitudinal player-season panel, not a cross-sectional
correlation. Build a target-trial-style specification before fitting any model:

- **Unit:** canonical global player identity × league × competition × season.
- **Treatment:** a precisely defined team switch or predeclared teammate/lineup
  exposure. Do not infer treatment from the outcome or from future minutes.
- **Outcome:** next completed season's canonical BB-Rating, with the source
  cutoff and target season explicit. Current/in-progress seasons are excluded.
- **Pre-treatment covariates:** prior BB-Rating trajectory, age/experience,
  role/position, prior minutes and usage, team strength/style, league/competition,
  prior availability/injury indicators where actually available, and schedule/
  opportunity measures where reliable.
- **Primary estimator:** matched/weighted difference-in-differences or a
  doubly-robust AIPW estimator with treatment and outcome models fit only on
  pre-treatment covariates. Player and season structure must be respected.
  Cluster uncertainty at the treatment-assignment level (typically player or
  team, depending on the estimand); do not use naive independent-row standard
  errors.
- **Identification checks:** treatment/control counts, propensity overlap,
  covariate balance before/after weighting, effective sample size, missingness,
  common support, pre-trend/event-study diagnostics for DiD, and sensitivity
  to unobserved confounding.
- **Falsification:** placebo treatment dates, pre-treatment outcomes as negative
  controls, alternative comparison cohorts, leave-one-team/season-out tests,
  and checks that results are not driven by league transitions or changing
  rating calibration.
- **Validation:** rolling-origin/out-of-time evaluation; no random split across
  the same player timeline. Freeze the BB-Rating calibration release used to
  construct outcomes, and test sensitivity to calibration version.
- **Reporting:** point estimate, confidence interval, treated/control support,
  overlap/balance diagnostics, identifying assumptions and explicit failure
  reason. If any critical gate fails, return `causal_effect: null` and
  `identification: unavailable`, not a headline causal estimate.

This is a candidate design, not a guarantee that observational data identify
the effect. If key confounders are unavailable or overlap fails, the correct
result is that the causal effect is not identified for that question.

## Requirements before capability status changes

### `performance_attribution` can become `available` only when
- the decomposition and team/role/lineup evidence paths have contract tests;
- each evidence layer returns an explicit outcome, source, sample size and
  unavailable/insufficient state;
- the causal teammate/team component either passes its identification gates or
  is clearly optional and the capability definition is narrowed to descriptive
  attribution. Do not silently equate association with attribution.

### `causal_team_effect` can become `available` only when
- the declared estimand and treatment/outcome timeline are implemented;
- the panel is populated from canonical, refresh-audited source relations;
- the causal estimator and all diagnostics above are implemented and tested;
- out-of-time and placebo tests pass prespecified acceptance thresholds;
- production responses expose the estimate only when all mandatory gates pass.

A capability can be operationally available while a specific player's result is
`insufficient_evidence`; the endpoint must still say so for that query.

## Implementation status

- Implemented in basketball_ai/scenarios/causal_attribution.py: guarded cohort-level team-switch ATT using propensity and outcome models on pre-treatment BB-Rating, minutes, usage, age, team strength, role and transition season.
- The in-memory builder uses canonical global player identity and the canonical BB-Rating engine only. It excludes the current calendar year as potentially in progress, requires adjacent completed seasons, and keeps one latest transition per player to avoid repeated-player pseudo-replication.
- The estimator suppresses the estimate when treated/control counts, propensity overlap or weighted covariate balance fail. It returns diagnostics and an interval only when the gates pass. The estimate is explicitly cohort-level and conditional on observational assumptions.
- This is an initial guarded estimator, not proof of causality and not a player-specific estimate. Its uncertainty calculation, calibration comparability, residual confounding and out-of-time/placebo performance still require validation against production data.

## Remaining implementation sequence

1. Persist a schema/versioned causal panel with canonical player/team identities, treatment date, outcome season, pre-treatment covariates, calibration release and source provenance; validate uniqueness and temporal ordering during refresh.
2. Run rolling-origin/out-of-time, placebo, sensitivity and leave-one-team/season-out tests on real completed-season data with prespecified acceptance thresholds.
3. Add teammate exposure only after treatment timing and selection are modeled without treating shared minutes as randomized.
4. Promote causal_team_effect only after these tests pass. Until then its status remains partial.

## Immediate safety rule

The descriptive performance_attribution capability is now marked available because it is explicitly scoped to observed decomposition and associations, with causal claims excluded. Keep causal_team_effect marked partial. Return the existing BB-Rating decomposition and observed associations, and expose the cohort ATT only when its support and balance gates pass; explicitly state that this is not a player-specific causal effect and that causal identification remains conditional on observational assumptions. Never call an observed difference
"caused by the team", "caused by the quintet" or "caused by a teammate".

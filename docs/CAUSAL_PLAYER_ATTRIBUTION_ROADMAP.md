# Causal player-performance attribution: implementation gate

## Decision

The descriptive `performance_attribution` capability is now `available` within a narrowed scope: decomposition and observed associations only. Keep `causal_team_effect` `partial` until causal identification and production validation pass. The implementation can provide:
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

The `AI_Source.SimulationCausalPanel` contract stores canonical focal-player
identity, optional named-teammate identity, league/competition, pre-treatment
BB-Rating, minutes, usage, age, team strength, role, transition season and
refresh timestamp. An incremental migration preserves installations with the
original schema-only table, and the data loader reads the expanded contract.
The guarded team-switch and teammate-exposure ATT estimators build completed
adjacent-season cohorts and can persist them through the canonical BB-Rating
engine and atomic writer.

The **Model Releases** page exposes explicit bulk-refresh actions for team-switch
and teammate-exposure panels. Each action discovers all canonical
league/competition contexts with completed adjacent seasons, chooses the latest
completed transition available per context, records per-context outcomes and
row counts, and audits the start and completion/failure. This is a manual,
operator-triggered maintenance workflow: it is not a chat side effect and is not
one of the recurring scheduler task types. A panel must not be assumed populated
because its schema exists; inspect the refresh status and persisted-row counts.

The existing `causal_effect` scenario can fall back to a constructed next-season
panel based on changes in minutes; that fallback is not an acceptable source
for team/teammate attribution and must never be reused as proof of the requested
causal effect. Production rollout still requires confirming SQL migration,
database permissions, persisted panel coverage and row-level invariants in the
target database.

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

## Capability scope and promotion gates

### `performance_attribution` — available, descriptive scope
- The capability is limited to observed BB-Rating decomposition, role/usage changes, team-context associations and descriptive lineup/team-Net-Rating comparisons.
- Each optional evidence layer must expose its outcome, source, sample size and unavailable/insufficient state.
- It must never describe an observed association as a cause. Requests for causal language must also invoke `causal_team_effect`.

### `causal_team_effect` — remains partial
- The guarded team-switch ATT and persisted-panel refresh path are implemented. The operator can refresh all eligible canonical contexts from Model Releases; the resulting cohort is observational, not a named player's causal effect.
- The bulk refresh records per-context status/counts, exposes a 0–100 progress bar and allows cooperative operator cancellation at the next context boundary; it writes audit events and never promotes a model. Causal cohort builders ignore null/unparseable season and local-ID values instead of raising on pandas `NA` values.
- `ai_scenario_serving.sql` grants the configured runtime roles SELECT/INSERT/UPDATE/DELETE on `SimulationCausalPanel` for the guarded persistence path only. Apply that SQL with the schema owner after this release; canonical player/team statistics remain read-only.
- Promotion requires validated uncertainty and calibration comparability, out-of-time and placebo tests, sensitivity/leave-one-team-season-out checks, prespecified thresholds, and production contract tests.
- The teammate-exposure panel compares focal players only with candidate teammates who have an observed row in the same outcome/transition season. Historical-only identities are excluded from the control pool; otherwise they create artificial treatment=0 rows and inflate counts without supplying eligible controls. The corrected builder must be deployed and the bulk teammate-exposure refresh rerun to replace previously persisted rows.
- A guarded teammate-exposure estimator and bulk refresh path now exist, but teammate-specific causal attribution remains partial until its treatment/exposure design and estimates pass empirical validation. Do not represent implementation presence as validated causality.

A capability can be operationally available while a specific player's result is `insufficient_evidence`; the endpoint must still say so for that query.

## Implementation status

- Implemented in basketball_ai/scenarios/causal_attribution.py: guarded cohort-level team-switch ATT using propensity and outcome models on pre-treatment BB-Rating, minutes, usage, age, team strength, role and transition season.
- The in-memory builder uses canonical global player identity and the canonical BB-Rating engine only. It excludes the current calendar year as potentially in progress, requires adjacent completed seasons, and keeps one latest transition per player to avoid repeated-player pseudo-replication.
- `performance_attribution` now includes an explicit adjacent-season descriptive decomposition: BB-Rating point change, team-context score change, minutes and usage changes, team switch and role change. It labels this as non-causal and does not assign percentages of responsibility to individual/team factors.
- The estimator suppresses the estimate when treated/control counts, propensity overlap or weighted covariate balance fail. It returns diagnostics and an interval only when the gates pass. The estimate is explicitly cohort-level and conditional on observational assumptions.
- This is an initial guarded estimator, not proof of causality and not a player-specific estimate. Its uncertainty calculation, calibration comparability, residual confounding and out-of-time/placebo performance still require validation against production data.

## Remaining validation sequence

1. On the deployment database, apply/verify the incremental `SimulationCausalPanel`
   migration and the required table/procedure permissions by rerunning
   `basketball_ai/data/ai_scenario_serving.sql` as the PostgreSQL schema owner (normally
   `hm_admin`). Run both bulk refresh actions and verify persisted rows, unique player transitions, temporal ordering,
   per-context status and audit history.
2. Run rolling-origin/out-of-time, placebo, sensitivity and leave-one-team/season-out
   tests on real completed-season data with prespecified acceptance thresholds.
3. Validate teammate exposure timing, selection and overlap; shared minutes and
   lineup presence must not be treated as randomized assignment.
4. Promote `causal_team_effect` beyond partial only after all empirical, uncertainty,
   database-contract and production-readiness gates pass.

### Teammate exposure support diagnostics

- Model Releases now offers an explicit diagnostic of the persisted `teammate_exposure` panel grouped by `teammate_global_id × league_key × competition × transition_season`; it does not pool a named teammate across seasons to make a small cohort appear larger.
- The report checks canonical focal identity integrity, complete-case covariates and the existing minimum gates (60 complete rows, at least 20 treated and 20 controls). Only groups that pass those raw gates proceed to propensity overlap and weighted-covariate-balance evaluation.
- At most 50 raw-supported groups are fitted in one run to bound operational cost. Any raw-supported groups beyond that cap are explicitly marked not evaluated; the CSV is the audit detail. No effect estimate is published by this diagnostic.
- Even groups that pass the statistical gates remain observational. The treatment may represent sharing a team in the outcome season rather than a newly acquired teammate; temporal treatment definition, calibration comparability, out-of-time/placebo and sensitivity checks remain required. Keep `causal_team_effect` partial.

## Immediate safety rule

The descriptive performance_attribution capability is now marked available because it is explicitly scoped to observed decomposition and associations, with causal claims excluded. Keep causal_team_effect marked partial. Return the existing BB-Rating decomposition and observed associations, and expose the cohort ATT only when its support and balance gates pass; explicitly state that this is not a player-specific causal effect and that causal identification remains conditional on observational assumptions. Never call an observed difference
"caused by the team", "caused by the quintet" or "caused by a teammate".

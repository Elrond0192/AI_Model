# Changelog

All notable changes to Basketball Performance AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Fixed
- Preserve the canonical `team_global_id` on each `PlayerCompetitionStats` row and require it in the PostgreSQL loader contract. Teammate-exposure causal refresh now resolves pre-treatment team strength by that explicit identity even when local numeric team IDs differ between player and team-stat sources.
### Fixed
- Corrected the missing blank line in the causal panel persistence helper; no runtime behavior changes.
### Added
- Added arm-specific propensity-overlap failure diagnostics to the teammate-exposure estimator and Model Releases report: gate-specific failure codes, kept treated/control counts, tail counts and p05/p50/p95 score quantiles are available in the API and CSV. Existing overlap thresholds and the observational-only status are unchanged.
- Added per-row diagnostics for missing pre-treatment team strength in teammate-exposure panels. Model Releases now distinguishes identity-match failures, absent pre-treatment team seasons, incompatible competition contexts, stale team data, and unavailable rating metrics; older persisted rows are explicitly flagged for refresh. This is diagnostic-only and does not relax causal support, overlap, or balance gates.
- Extended teammate-exposure diagnostics with row-level missingness counts and percentages for identities, treatment, outcome and every model covariate, globally and per teammate/context group. This separates low raw treatment support from complete-case attrition before any estimator threshold is changed.
- Added a Model Releases diagnostic for the persisted teammate-exposure panel, pooled by named teammate × league × competition across eligible transitions (while retaining season coverage in the CSV). It reports complete-case treated/control support, identity integrity, propensity overlap and weighted covariate balance for up to 50 raw-supported cohorts per run; passing gates remains observational and does not promote `causal_team_effect`.
- Redefined teammate treatment as a newly observed shared-team relationship between adjacent completed seasons. Continuing relationships and pairs without an observed teammate baseline team are excluded from the initiation cohort; treatment is not inferred from outcome-season co-membership alone.

### Fixed
- Fixed teammate-exposure `team_strength_pre` lookup to prefer canonical team global identity before numeric team IDs. The builder now falls back from the exact competition to `TOT`/`RS`, then to the same pre-treatment team season or at most a one-season-lagged value; it never uses future or stale multi-season team strength. Added regression tests for team-ID drift, aggregate-comp coverage and temporal leakage. Persisted teammate panels must be refreshed.
- Corrected teammate-exposure panel construction: treatment/control candidates are now restricted to players with observed statistics in each focal transition's outcome season. Previously, every player ever observed in historical data could be counted as a control even when inactive in that season, artificially inflating the control count. Existing persisted panels must be rebuilt with the bulk teammate-exposure refresh after deployment.
- Added a regression test proving that a historical-only player is excluded from later-season teammate-exposure controls.
- Hardened team-switch and teammate-exposure cohort builders against nullable pandas IDs and missing/unparseable season values. Invalid rows are excluded before integer conversion; usable transitions in the same context continue to be processed.
- Fixed `refresh_context_seasonal` PostgreSQL permission failures by granting the configured runtime roles DML only on the physical derived Simulation/Training/Scenario serving tables that the allow-listed refresh procedures rebuild. Canonical player/team facts and upstream PBP/anagraphic/analysis inputs remain read-only.
- Fixed admin navigation after login: repeated `.nav-item`, `.admin-only` and `.jump` controls now use `querySelectorAll` before `forEach`, preventing a runtime TypeError from blocking the shell and menu. Bumped the JavaScript asset version to `2.6.9` and added a regression test.
- Fixed the JavaScript syntax error that prevented the admin console scripts, including the login handler, from running; separated causal refresh button state from the existing job-state variable and bumped static asset cache versions.
- Prevented login credentials from being submitted in URL query strings: the form now uses POST, the backend safely handles a native form POST fallback with clean redirects, and admin assets are cache-busted.
- Added a BB Prediction Model-style progress bar and a cooperative stop control to
  the global causal-panel refresh. Cancellation stops at a safe context boundary;
  persistence permission failures are now counted as errors rather than missing cohorts.
- Granted the runtime roles scoped DML on `AI_Source.SimulationCausalPanel` through
  the schema-owner SQL contract, fixing persisted cohort refresh permission failures.
- Reset the main document and content scroll positions to the top on every page
  navigation. Serialized the initial overview render and PostgreSQL auto-load on
  login/session restore, then refreshes overview from the loaded state to avoid
  concurrent renders racing and leaving the dashboard blank or stale.
- Fixed scheduler HTTP 500s for restricted PostgreSQL runtime roles: the web
  process now validates access without attempting privileged runtime `ALTER TABLE`
  operations, and the serving SQL grants CRUD only on scheduler metadata tables
  and their sequences. Missing schema/permissions now return an actionable HTTP 503.
- Removed the obsolete PR #76 implementation-label copy from Model Releases while
  retaining a concise description of the causal-panel refresh action.
- Fixed Model Releases promotion and rollback button binding: operate on the full
  button collection, and submit the exact validated candidate ID shown in the UI.
- Populate release cutoff/season metadata from the schema used by BB-Rating and
  Future Performance artifacts.
- Prefer the lifecycle-managed BB-Rating release over a stale legacy uncertainty
  JSON when loading the runtime artifact.
- Remove orphaned release-candidate directories when the required primary artifact
  was never written; show/promote the newest valid candidate instead of a stale,
  incomplete directory. Prediction calibration failures now retain the real error
  and clean up candidates that never produced their JSON artifact.

### Documentation
- Added `docs/SCHEDULER.md` and synchronized README, RUNBOOK and Player
  Intelligence/causal roadmaps with the current scheduler, causal-panel refresh
  implementation and operational boundaries. Updated scheduler migration
  ownership and least-privilege runtime permissions guidance.
- Added a release-documentation policy and automated checks: runtime/API/data
  changes in a pull request must include documentation or changelog updates,
  and CI checks for known stale capability statements.
- Repository-wide Markdown audit: corrected the PostgreSQL installation order, replaced the obsolete `PlayerStats` references with the competition-preserving `PlayerCompetitionStats` contract, documented the current API route families and health/restart behavior, refreshed the first-prediction tutorial and architecture overview, and added the referenced WordPress/Chat V3 integration guide.

### Added
- Persistent scheduler controls for recurring tasks: interval-in-hours or daily
  local clock time (`Europe/Rome`), optional inclusive start/end dates, and
  enable/disable switches. Existing schedules migrate as interval-based jobs.
- Model Releases bulk refresh for persisted team-switch and teammate-exposure
  observational cohorts across eligible canonical league/competition contexts,
  with per-context result counts and audit events. These actions do not train or
  promote models; `causal_team_effect` remains partial pending empirical gates.
- Model Center cancellation for Prediction training, BB-Rating calibration,
  Future Performance training and Prediction 1–100 calibration. Cancellation
  is cooperative at safe checkpoints; incomplete candidates are removed and
  production artifacts remain unchanged.
- Model Center P0/P1 model lifecycle controls: isolated candidate artifacts for
  Prediction 1–100 calibration, BB-Rating and Future Performance; artifact
  validation; atomic promotion; retained previous releases and rollback; and
  runtime readiness/version diagnostics. The native Prediction Model registry
  remains a separate lifecycle.
- Control Center promotions and rollbacks now request an authenticated API
  restart and wait for `/health/ready`. Candidate training/calibration does
  not restart production. CLI workflows remain separate and require the
  documented manual restart when they change serving artifacts.
- BB-Rating 1.9 introduces a competition-tier context registry for cross-competition analysis: Tier 1 European elite (EL/EC), Tier 2 national first divisions (including BEL1), and Tier 3 national second division (ITA2). Tier is metadata for competitive level/transferability and does not modify the within-league BB-Rating score.
- Player Future Performance Model v1.0: independent player-centric season-ahead multivariate forecasts for per-36 production, shooting/usage rates, role metrics and minutes, with expanding walk-forward OOS validation, persistence baselines, target-level uncertainty and a separate serving artifact. Prediction Model 2.6.0 and BB-Rating 1.8 remain unchanged.
- Player Future Performance is exposed through the API at `POST /api/v2/future-performance/player`, the AI Model Control Center and the WordPress player profile proxy.
- AI_Model Control Center v2.5.0: dedicated BB-Rating operations page with
  calibration/artifact status, dataset and uncertainty diagnostics, runtime
  version checks, and a separate calibration job. BB-Rating remains outside
  the Prediction Model candidate/promotion lifecycle.
- BB-Rating calibration can now persist `calibration_version` and
  `bb_rating_version` directly in `bb_rating_uncertainty.json`.
- BB-Rating calibration 1.14: separates between-league variation from the
  within-league exposure signal using league-adjusted rank diagnostics and
  fixed-effect incremental R². Diagnostic-only; BB-Rating 1.8 and Prediction
  Model outputs are unchanged.
- BB-Rating calibration 1.13 league/exposure uncertainty decomposition:
  empirical league × exposure variation tables and rank-based incremental R²
  diagnostics testing whether league context adds information beyond exposure.

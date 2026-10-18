# Changelog

All notable changes to Basketball Performance AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Fixed
- Fixed Model Releases promotion and rollback button binding: operate on the full
  button collection, and submit the exact validated candidate ID shown in the UI.
- Populate release cutoff/season metadata from the schema used by BB-Rating and
  Future Performance artifacts.

### Added
- Model Center cancellation for Prediction training, BB-Rating calibration,
  Future Performance training and Prediction 1–100 calibration. Cancellation
  is cooperative at safe checkpoints; incomplete candidates are removed and
  production artifacts remain unchanged.

### Documentation
- Repository-wide Markdown audit: corrected the PostgreSQL installation order, replaced the obsolete `PlayerStats` references with the competition-preserving `PlayerCompetitionStats` contract, documented the current API route families and health/restart behavior, refreshed the first-prediction tutorial and architecture overview, and added the referenced WordPress/Chat V3 integration guide.
### Added
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
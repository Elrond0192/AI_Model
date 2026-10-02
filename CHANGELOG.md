# Changelog

All notable changes to Basketball Performance AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
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
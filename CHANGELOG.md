# Changelog

All notable changes to Basketball Performance AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
- AI_Model Control Center v2.5.0: dedicated BB-Rating operations page with
  calibration/artifact status, dataset and uncertainty diagnostics, runtime
  version checks, and a separate calibration job. BB-Rating remains outside
  the Prediction Model candidate/promotion lifecycle.
- BB-Rating calibration can now persist `calibration_version` and
  `bb_rating_version` directly in `bb_rating_uncertainty.json`.

### Previously added
- BB-Rating calibration 1.14: separates between-league variation from the
  within-league exposure signal using league-adjusted rank diagnostics and
  fixed-effect incremental R². Diagnostic-only; BB-Rating 1.8 and Prediction
  Model outputs are unchanged.
- BB-Rating calibration 1.13 league/exposure uncertainty decomposition:
  empirical league × exposure variation tables and rank-based incremental R²
  diagnostics testing whether league context adds information beyond exposure.
  Diagnostic-only; BB-Rating 1.8 and Prediction Model outputs are unchanged.

- BB-Rating calibration 1.12 empirical uncertainty diagnostics: absolute
  next-season rating-change distributions by exposure, peer-context sample
  size, metric coverage and league, plus exposure-controlled association
  diagnostics. Diagnostic-only; BB-Rating 1.8 and Prediction Model outputs are
  unchanged.

- Quality-v2 forecast contract: explicit `RS/PO/TOT` training allow-list,
  minimum-games gate, reliability sample weights, and training diagnostics.
- Automatic fast-source selection: indexed serving tables are used when
  populated, with a safe canonical-view fallback.
- Basketball Simulation & Causal Engine for Chat V3 scenarios:
  probabilistic joint boxscore distributions, quantiles, and threshold
  probabilities with seeded or system-entropy Monte Carlo runs.
- Opponent-conditioned matchup projections using pace, defensive profile, and
  possession-level play-type context.
- Dedicated player-vs-player defensive exposure inference from PBP and lineup
  stints. Inferred assignments are explicitly labelled as inferred rather than
  observed.
- Play-type matchup estimates, shot-profile counterfactuals with
  destination-efficiency shrinkage, and composable multi-condition scenarios.
- Partial-pooled lineup synergy, exhaustive five-player lineup optimization,
  and constrained roster optimization.
- AIPW causal-effect estimates with propensity-overlap and identification
  gates; unsupported causal questions return an unavailable identification
  result instead of a causal claim.
- Optional PostgreSQL simulation source adapter for canonical PBP, lineup,
  play-type, shot-profile, and causal-panel frames, with empty-view fallbacks
  when physical feeds are unavailable.
- `docs/SIMULATION_ENGINE.md` documenting simulation, data, and causal
  contracts.
- `has_po_history` feature to distinguish players with no playoff history from those
  with equal RS/PO performance.
- League-aware `durability_score` using per-league `max_games` (no longer hardcoded to 82).
- Account lockout after configurable failed login attempts (`LOGIN_MAX_ATTEMPTS`,
  `LOGIN_LOCKOUT_SECONDS` env vars).
- Admin one-time credentials written to `.admin_credentials` (chmod 0600) instead of
  displayed in the browser UI.
- SQL injection guard: schema identifier validated in `sql_loader.py`.
- Error boundaries on all 8 Streamlit tabs (`_safe_tab` context manager).
- `/health/live` (liveness) and `/health/ready` (readiness) API endpoints.
- CORS fail-fast at startup when `API_ENV=production` and `ALLOWED_ORIGINS` is unset.
- Coverage reporting in CI (`--cov=basketball_ai`).
- Lint failures now fail the CI build (removed `|| true`).
- `python_requires = ">=3.11,<3.13"` in `pyproject.toml`.
- `MODEL_CARD.md` documenting model characteristics, limitations, and bias notes.
- Time-based chronological train/val split (no temporal leakage).
- Baseline models (mean, linear, RandomForest) logged in every training run.
- Dataset MD5 signature saved alongside model artifact.
- Data lineage (input hashes, timestamp, sample count) in training metadata.
- Schema validation on DataFrames at loader boundary.
- SQL dataset parquet caching with configurable TTL.
- Structured JSON logging (`LOG_FORMAT=json`).
- Optional Prometheus metrics (`/metrics` endpoint).
- Simple model registry (`registry.json`) updated on every `model.save()`.
- `/predictions/{id}/explain` endpoint returning top-5 SHAP contributions.
- `/predictions/batch` endpoint for bulk predictions.
- `X-Total-Count` header on list endpoints (players, teams).
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, HSTS).
- Argon2id password hashing (PBKDF2 fallback).
- Persistent audit log (`audit.log`) for all auth/admin actions.
- `pip-audit` security job in CI.
- Centralised `Settings` class via `pydantic-settings`.
- `.pre-commit-config.yaml` with ruff + pre-commit-hooks.
- Intent handler registry (`@register_handler` decorator) for extensible chat intents.
- Onboarding empty-state in the Data tab for new users.
- `CONTRIBUTING.md` with development workflow and code conventions.
- `CHANGELOG.md` (this file).
- `docs/architecture/` directory with C4 overview.

### Fixed
- `Home`/`Away` descriptive splits no longer create duplicated season-ahead
  forecast samples; they remain available to generic observed-data analysis.
- `ValLegaPerGame` is mapped to the documented 1–10 target scale within each
  league/season/competition cohort.
- Clutch scoring now uses points per clutch game plus sample reliability,
  instead of dividing clutch totals by ordinary minutes per game.
- `datetime.utcnow()` deprecated call replaced with `datetime.now(timezone.utc)`.
- Module-level `random.seed(42)` / `np.random.seed(42)` removed from `generator.py`.
- `_safe_model_dir` now uses `Path.resolve().is_relative_to()` to prevent path traversal.

### Changed
- CI lint steps (`ruff`, `flake8`) no longer use `|| true`; lint failures fail the build.
- Admin bootstrap: password no longer displayed in browser; written to file instead.

---

## [1.0.0] — 2024-01-01

### Added
- Initial release: XGBoost performance model, EnsembleModel, WhatIfEngine.
- FastAPI REST API with JWT + API-key auth.
- Streamlit multi-user GUI with RBAC.
- Azure SQL Server + CSV dual-backend loader.
- Chat engine with 15 intents.
- Conformal prediction intervals.
- WordPress JSON export.

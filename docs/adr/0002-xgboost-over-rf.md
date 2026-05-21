# ADR 0002 – XGBoost as primary model (over Random Forest)

**Status:** Accepted  
**Date:** 2025-05-21  

## Context

The PerformanceModel needs to predict next-season WAR (BPM-based) from
58 basketball features. Both Random Forest and XGBoost were evaluated.

## Decision

Use XGBoost as the primary predictor, with Random Forest as a baseline
for comparison in `compute_baselines()`.

Rationale:
- Lower val_rmse than RF in backtesting (typically 8–12% improvement).
- Native support for missing features (`eval_set`, `early_stopping_rounds`).
- Faster inference than deep ensembles for real-time API use.
- `TimeSeriesSplit` CV prevents look-ahead bias.

## Consequences

- **Good:** Better accuracy, faster serving.
- **Bad:** Less interpretable than RF (mitigated by SHAP integration).
- **Good:** `_get_shap_values()` provides feature importance for each prediction.

# Model Card — BBallstat AI_Model

## Model

| Field | Value |
|---|---|
| Version | 2.1.0 |
| Feature contract | `forecast-t-plus-1-v2` |
| Primary target | next-season player rating (`t -> t+1`) |
| Base estimator | XGBoost regressor |
| Team/context layer | temporally trained k-NN compatibility + bounded context adjustments |
| Output | predicted rating 0–10 + empirically validated interval |
| Production source | PostgreSQL `ai_source` canonical views |

AI_Model is an inference/training service, not a conversational model. WordPress Chat V3/Bax owns intent detection, entity resolution and natural-language responses.

## Training unit

The forecast unit is strictly:

```text
features available at season t  ->  observed rating at season t+1
```

A pair is created only when `target_season == source_season + 1`. Missing intermediate seasons are excluded rather than treated as one-year forecasts. The canonical PostgreSQL adapter provides one target row per global player and season.

## Temporal split and estimator selection

Target seasons are blocked as whole seasons:

- older target seasons: estimator training;
- penultimate target season: validation and XGBoost tree-count selection;
- latest target season: untouched final-ensemble interval calibration.

After validation selects the number of boosting trees, XGBoost is refit on train + validation. The calibration target remains unseen by estimator fitting.

Optional cross-validation is season-blocked; row-level random CV is not used for production evidence.

## Historical team/player context

`ai_source.team_season_stats` preserves team style by season. Compatibility training for a target season can use only player history and team context available before that target. Historical backtests and historical API requests create an as-of snapshot that removes future player stats, future roster state and future team style.

## Feature families

- age and position;
- per-36 volume: points, assists, rebounds, steals, blocks;
- efficiency/usage: PIE, TS%, USG%, OBPM, DBPM;
- RAPTOR, LEBRON, SPM, OWS/DWS, FIC and related advanced metrics when present;
- career form, consistency and trajectory;
- league-aware durability;
- role labels;
- on/off, clutch and starter context when available;
- bounded interaction features.

Missing optional advanced fields may use neutral defaults; target identity, chronology and required source-contract fields may not.

## Production prediction path

```text
historical player state through t
historical team state as of t
             |
             v
        XGBoost t+1
             |
             +-- compatibility
             +-- league/context
             +-- minutes / age scenario logic
             v
      final production rating
             |
             v
 empirically calibrated interval
```

Supported dynamic endpoint: `POST /api/v2/predictions/player-team`.

## Evaluation

Promotion evidence comes from an expanding walk-forward backtest of the **same final ensemble behavior used in production**, not only the base estimator. Every fold trains on information available before its target season and predicts the untouched next season.

Reported metrics include:

- RMSE, MAE, R² and bias;
- final interval coverage and mean width;
- base XGBoost RMSE;
- previous-season persistence RMSE;
- ensemble improvement over both baselines;
- per-league, per-position and age-band performance;
- fold-level target seasons and sample counts.

Fold failures invalidate the backtest; target values are never used as fallback predictions.

## Candidate / production lifecycle

Training creates immutable artifacts under `models_saved/runs/<model_run_id>/`. A run is registered as **candidate** and cannot alter production directly.

Promotion requires the configured quality gates, including valid OOT evaluation, sample minimum, acceptable interval coverage, ensemble improvement over base and persistence, and bounded per-league regressions. When a production run already exists, the candidate must also improve its OOT RMSE by the configured threshold.

Only promoted artifacts are copied atomically to `models_saved/production/`, which is the directory loaded by the API container. Rollback reactivates the previous immutable run.

## Explainability and monitoring

The base tree model exposes SHAP values when the optional SHAP runtime is available. Training records a data signature and lineage metadata. Drift monitoring stores a training reference and supports PSI/segment checks; drift is a monitoring signal, not a substitute for retraining/backtesting.

## Target comparability

`ValLegaPerGame` / `rating` is treated as the supervised target. `validate-data` reports target distributions by league and warns when league means diverge substantially. AI_Model does not automatically normalize the target across leagues because that transformation must be demonstrated to improve real OOT performance rather than assumed.

## Limitations

- Team-fit effects remain observational/associative rather than causal transfer estimates.
- Injury, contract, coaching and off-court information are not explicitly modelled.
- Sparse-history players may be excluded from OOT folds or have weaker career features.
- The season-level target intentionally collapses multiple same-season competition rows. Competition-specific API context is a scenario feature, not a separate competition-specific supervised target.
- New leagues or material source-schema changes require rebuilding and validating the PostgreSQL adapter before retraining.
- Real effectiveness must be judged on the deployment dataset's untouched future seasons; synthetic metrics are not production quality claims.

## Privacy and security

- Source schemas are read-only to AI_Model.
- Writes are restricted to model-owned schema `ai`.
- PostgreSQL credentials and API secrets are runtime secrets and are never committed.
- Browser clients never receive database credentials, `IdGlobal`, or generic SQL access.
- WordPress communicates with AI_Model server-to-server over authenticated HTTPS.

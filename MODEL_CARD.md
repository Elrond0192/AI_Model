# Model Card — BBallstat AI_Model

## Model

| Field | Value |
|---|---|
| Version | 2.2.0 |
| Feature contract | `forecast-t-plus-1-competition-v1` |
| Primary target | same-league, same-competition next-season rating (`t -> t+1`) |
| Competition support | data-driven; any normalized competition with supervised consecutive history |
| Base estimator | pooled XGBoost regressor with explicit competition feature |
| Team/context layer | temporally trained competition-aware k-NN + bounded context adjustments |
| Output | predicted rating 0–10 + calibrated interval + competition support metadata |
| Production source | PostgreSQL `ai_source` competition-preserving views |

AI_Model is an inference/training service, not a conversational model. WordPress Chat V3/Bax owns intent detection, entity resolution and natural-language responses.

## Training unit

The forecast unit is strictly:

```text
player P / league L / competition C / season t
                         ->
player P / league L / competition C / season t+1
```

Missing intermediate seasons, league switches and competition switches do not form training pairs. `RS -> PO` and domestic-league -> EuroLeague pairs are invalid. The model is pooled across valid contexts so it can learn shared basketball relationships, while `competition` remains an explicit learned feature and every row's historical features are isolated to its own league/competition.

Known labels are normalized (`PLAYOFFS -> PO`, `REGULAR SEASON -> RS`). New labels are retained in normalized form. A competition is not serveable merely because its label exists: it must have produced supervised consecutive pairs in the promoted training run.

## Temporal split and estimator selection

Target seasons are blocked as whole seasons:

- older targets: estimator training;
- penultimate target season: validation/tree-count selection;
- latest target season: untouched final-ensemble interval calibration.

XGBoost is refit on train + validation after tree selection. Calibration outcomes remain unseen by estimator fitting. Role vocabularies are temporally bounded and row-level random CV is not used as production evidence.

## Historical player/team/competition context

`ai_source.player_competition_stats` and `ai_source.team_competition_stats` preserve league, season and competition. Historical inference constructs an as-of source-season snapshot and then scopes it to the requested league/competition.

For a playoff request, player form, target-team pace/ORtg/DRtg/style, compatibility and persistence come from playoff rows in that league. Regular-season statistics are not silently substituted. If the required isolated source context is missing, inference fails explicitly.

Team-style normalization bounds and the trained competition vocabulary are run-owned state persisted in `production_state.joblib` and restored before inference.

## Feature families

- source-season age and roster position;
- explicit competition encoding;
- per-36 volume and efficiency/usage;
- PIE, TS%, USG%, OBPM, DBPM, RAPTOR, LEBRON, SPM, OWS/DWS, FIC when available;
- career form/trajectory **inside the requested league/competition**;
- temporally bounded role labels;
- on/off, clutch and starter context when available for that competition;
- competition-specific team style/context;
- bounded interactions.

## Production prediction path

```text
league + competition + source season
             |
             v
isolated player history + team context
             |
             v
       XGBoost t+1
             |
       compatibility
       context / league
       minutes / age scenario
             |
             v
    final production rating
             |
             v
competition conformal interval if sufficiently calibrated
otherwise explicitly reported global conformal fallback
```

The global interval is calibrated on final-ensemble absolute residuals. Per-competition finite-sample quantiles are used only when that competition reaches the configured calibration minimum. The API reports which calibration scope was used.

## Evaluation

Promotion evidence comes from expanding walk-forward backtests of the same final ensemble used online. Reported metrics include RMSE, MAE, R², bias, coverage/width, base-XGBoost and persistence comparisons, plus segmentation by league, **competition**, position and age band.

A candidate cannot hide a serious PO/CUP regression behind a better global average: promotion compares competition segments against production when both have enough samples.

## Candidate / production lifecycle

Training creates immutable `models_saved/runs/<model_run_id>/` artifacts and registers a candidate without replacing production. `production_state.joblib` contains style calibration, competition vocabulary and per-competition interval state. Promotion requires valid OOT evidence, sufficient samples, interval coverage, ensemble improvement and bounded league/competition regressions. Rollback reactivates a previous immutable run.

## Sparse competition evidence

`competition_support` reports:

- number of historical source seasons in that league/competition;
- total games represented;
- exact source-season games;
- number of calibration samples;
- whether confidence bounds use competition-specific or global calibration.

The model never claims an unseen competition is trained. A newly ingested competition becomes available only after data validation, training/backtesting and promotion of a model whose competition vocabulary contains it.

## Target comparability

`ValLegaPerGame` / `rating` is the supervised target. The validator reports rating distributions by league and competition. Cross-league normalization is not applied automatically; it must be justified by OOT evidence.

## Limitations

- Team-fit effects remain observational rather than causal transfer estimates.
- Injury, contract, coaching and off-court information are not explicitly modelled.
- Sparse-history players/competitions produce less evidence; competition-specific interval calibration may fall back to the global conformal interval.
- If a player has no exact requested source-season competition row, the service refuses the isolated forecast rather than mixing contexts.
- Real effectiveness must be judged on untouched future seasons from the deployment dataset.

## Privacy and security

Source schemas are read-only; writes are restricted to schema `ai`; secrets are runtime-only; browser clients never receive DB credentials or `IdGlobal`; WordPress calls AI_Model server-to-server over authenticated HTTPS.

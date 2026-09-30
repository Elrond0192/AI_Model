# Model Card — BBallstat AI_Model

## Model

| Field | Value |
|---|---|
| Version | 2.6.0 |
| Feature contract | `forecast-t-plus-1-persistence-delta-v1` |
| Primary target | same-league, same-competition next-season rating (`t -> t+1`), learned as `rating(t+1) - rating(t)` and reconstructed to 0–10 at inference |
| Competition support | explicit allow-list; `RS,PO,TOT` by default |
| Base estimator | pooled XGBoost regressor with nominal competition/role one-hot features |
| Team/context layer | temporally trained competition-aware k-NN + bounded context adjustments |
| Output | predicted rating 0–10 + calibrated interval + competition support metadata |
| Production source | PostgreSQL `"AI_Source"` competition-preserving views |

AI_Model is an inference/training service, not a conversational model. WordPress Chat V3/Bax owns intent detection, entity resolution and natural-language responses.

## Training unit

The forecast unit is strictly:

```text
player P / league L / competition C / season t
                         ->
player P / league L / competition C / season t+1
```

Missing intermediate seasons, league switches and competition switches do not form training pairs. `RS -> PO` and domestic-league -> EuroLeague pairs are invalid. The model is pooled across valid contexts so it can learn shared basketball relationships, while `competition` remains an explicit learned feature and every row's historical features are isolated to its own league/competition.

Known labels are normalized (`PLAYOFFS -> PO`, `REGULAR SEASON -> RS`). `Home`
and `Away` remain available to observed-data chat answers but are excluded from
forecast training because they duplicate partial regular-season evidence. A
competition is not serveable merely because its label exists: it must be in
`MODEL_TRAINING_COMPETITIONS` and produce supervised consecutive pairs.

## Temporal split and estimator selection

Target seasons are blocked as whole seasons:

- older targets: estimator training;
- penultimate target season: validation/tree-count selection;
- latest target season: untouched final-ensemble interval calibration.

XGBoost is refit on train + validation after tree selection. Calibration outcomes remain unseen by estimator fitting. Role vocabularies are temporally bounded and row-level random CV is not used as production evidence.

## Historical player/team/competition context

`"AI_Source"."PlayerCompetitionStats"` and `"AI_Source"."TeamCompetitionStats"` preserve league, season and competition. Historical inference constructs an as-of source-season snapshot and then scopes it to the requested league/competition.

For a playoff request, player form, target-team pace/ORtg/DRtg/style, compatibility and persistence come from playoff rows in that league. Regular-season statistics are not silently substituted. If the required isolated source context is missing, inference fails explicitly.

Team-style normalization bounds and the trained competition vocabulary are run-owned state persisted in `production_state.joblib` and restored before inference.

## Feature families

- source-season age and roster position;
- explicit persistence: last rating, one-step/two-step rating deltas, recent rating mean and volatility;
- explicit competition encoding;
- per-36 volume and efficiency/usage;
- PIE, TS%, USG%, OBPM, DBPM, RAPTOR, LEBRON, SPM, OWS/DWS, FIC when available;
- career form/trajectory **inside the requested league/competition**;
- temporally bounded role labels;
- on/off, clutch and starter context when available for that competition;
- clutch points per clutch game plus an explicit clutch-sample reliability signal;
- competition-specific team style/context;
- bounded interactions.

## Production prediction path

The production contract for `delta_vs_prior` is deliberately narrower than the
scenario/diagnostic layer. Persistence, competition, role, performance and
source-season context are learned by XGBoost from the source-season feature
vector. They are not reapplied as heuristic multipliers after inference.

```text
league + competition + source season
             |
             v
isolated player history + team context
             |
             v
       XGBoost t+1 delta
             |
       + source-season rating
             |
             v
   raw learned t+1 rating
             |
      bounded to [0,10]
             |
             v
competition conformal interval if sufficiently calibrated
otherwise explicitly reported global conformal fallback
```

Compatibility, age, league/context and minutes signals remain available as
metadata/diagnostics and for scenario analysis; they are not post-processing
corrections to the learned delta production prediction.

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

`ValLegaPerGame` is retained as the source score and converted by the SQL
contract to a stable 1–10 percentile rating inside each
league/season/competition cohort. This makes the documented target scale true
without mixing leagues. Ties receive the same percentile score.

## Limitations

- Team-fit effects remain observational rather than causal transfer estimates.
- Injury, contract, coaching and off-court information are not explicitly modelled.
- Sparse-history players/competitions produce less evidence; competition-specific interval calibration may fall back to the global conformal interval.
- If a player has no exact requested source-season competition row, the service refuses the isolated forecast rather than mixing contexts.
- Real effectiveness must be judged on untouched future seasons from the deployment dataset.

## Privacy and security

Source schemas are read-only; writes are restricted to schema `ai`; secrets are runtime-only; browser clients never receive DB credentials or `IdGlobal`; WordPress calls AI_Model server-to-server over authenticated HTTPS.

## BB-Rating layer

The season-ahead Prediction Model and BB-Rating are separate contracts.

- Prediction Model 2.6.0 forecasts the next-season rating using the native
  delta_vs_prior target. It is evaluated with OOS forecasting metrics.
- BB-Rating 1.0 is descriptive: it evaluates observed player performance
  relative to a contextual peer population and returns a 1–100 score with
  dimension scores, metric percentiles and deterministic explanations.
- BB-Rating does not modify, shrink, floor or otherwise post-process Prediction
  Model outputs.
- USG% is a contextual role/involvement signal and has no standalone positive
  weight in the composite score.

The initial BB-Rating implementation lives in basketball_ai/bb_rating and is
API-exposed at POST /api/v2/bb-rating/player. Its weights and peer thresholds
must be validated on real production data before the methodology is frozen.

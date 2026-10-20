# Direct future BB-Rating prediction

## Contract

The player timeline has one consistent scale:
- past and present: observed BB-Rating from the pinned `BBRatingEngine`;
- future: predicted observed BB-Rating for target season `t+1`, on the same 1-100 scale.

The native statistical rating (0-10) is not a BB-Rating label and is never the target of this model. The legacy native Prediction Model endpoint and its 1–100 display calibration are no longer exposed as user-facing prediction features; this model predicts BB-Rating directly.

## Model and labels

`basketball_ai.bb_rating.future_prediction.BBRatingFuturePredictionModel` builds target labels by calling the official `BBRatingEngine` for each available player + target season + destination league + competition context. Player features and destination-team context are restricted to information available at or before the source season; team context uses the destination team's latest available historical season, never its target-season performance. Source BB-Rating is itself calculated from that source-season context. Canonical player global identity is used to join seasons when provider-specific player IDs change.

The model uses an expanding walk-forward evaluation by target season. Each fold trains only on target seasons earlier than its test season. Calibration for OOF evaluation is itself temporal: a target-season fold can use only calibration rows from earlier OOF target seasons. The final serving calibrator is then fit on the complete OOF record set.

## Scenario coverage and pairing rules

The model stores source and destination league, resolved team identity, target statistical context and competition tier. Codes such as `HOME`, `AWAY`, `RS`, `TOT`, `PO` and `FINAL` can represent statistical views/phases; they are not treated as an independent competition identifier. Without an authoritative competition ID separate from these codes, the model does not infer a competition change from their text.

Training constructs at most one record for each player + target season + destination league + target context. A source row must have the same statistical-context code as the target row. If the destination league also exists in the source season, that source row is preferred; otherwise a single deterministic source context is selected by usage (games, minutes, then league key). This avoids the many-to-many cross-product and repeated weighting of one target context against multiple source contexts.

Scenarios describe only what the available identity data can support:
- same resolved team and league;
- team change within a league, only when both team global IDs resolve;
- league change, with or without team change, split by authoritative tier direction when available;
- unknown/uncertain when either team identity is unresolved, which uses the safe fallback rather than interpreting local IDs as stable identities.

Scenario calibration first attempts destination league + scenario, then scenario-only, then global. Cohort calibration requires at least 40 distinct players when canonical player IDs are available (40 rows is not sufficient); reports also expose rows evaluated, players, player-season transitions and unique target contexts. MAE/RMSE and training/calibration weights balance each player-season across its target contexts. Returned metadata keeps row volume separate from effective support.

## Baselines and release gate

The walk-forward report includes MAE, RMSE, bias and calibration error overall and by scenario and destination league. It compares the calibrated model with a context-aware baseline:
- source BB-Rating persistence when source and destination league/competition contexts match;
- otherwise the destination league + competition historical mean from the training fold, falling back to destination league mean and then the fold-wide mean.

The CLI does not save a candidate unless the player-season-balanced walk-forward MAE beats this baseline overall, at least three OOF target seasons and at least 40 distinct OOF players are available, and no supported scenario/destination cohort (at least 40 distinct players) regresses by more than 10% plus 0.25 BB-Rating points. The 40-player threshold is unchanged; it is now measured on independent player identities instead of raw rows. This is a minimum gate, not proof that every scenario is reliable. Before promoting a release, inspect per-scenario/per-league row counts, unique players, player-season transitions, unique target contexts and metrics. Under-supported cohorts remain clearly labelled and do not count as adequately validated.

## Train a candidate

Run against a complete canonical training snapshot or the intended analysis database profile:

```bash
python -m basketball_ai.cli --mode bb-rating-prediction-train \
  --database-profile production \
  --bb-rating-prediction-min-oof-samples 500 \
  --bb-rating-prediction-output-dir models_saved
```

The command excludes the latest observed season as a target by default (it can still be used as source history) to avoid training on an in-progress season. Only pass `--bb-rating-prediction-include-latest-target-season` when that season has been confirmed complete. It writes a lifecycle-managed candidate under `models_saved/candidates/bb-rating-prediction/<candidate_id>/` only if enough walk-forward OOF predictions exist and the MAE gate passes. It does not automatically promote the candidate. Inspect it in Model Center → Model Releases → BB Prediction Model (1–100), then use the normal candidate validation/promotion/rollback controls. The old native Prediction Model is not used as a target, intermediate conversion or user-facing prediction. Historical artifacts may remain on disk for audit/rollback until downstream consumers have been migrated.

After independent review, promote the candidate through Model Center or set `BB_RATING_FUTURE_MODEL_DIR` to an explicitly approved release directory. The API loads only the promoted release path, never the candidate directory.

## API

`POST /api/v2/bb-rating-prediction/player`

Example request:

```json
{
  "player_global_id": "174019",
  "source_season": 2026,
  "source_league": "EL",
  "source_competition": "RS",
  "target_league": "ITA1",
  "target_competition": "RS",
  "target_team_global_id": "TEAM_GLOBAL_ID"
}
```

The response contains a direct 1-100 BB-Rating prediction, empirical OOF 10th-90th residual interval, scenario, calibration level and support counts. The source league is optional; if omitted, the model deterministically selects the highest-usage source context in the requested source season/phase. Specify it when a player has simultaneous league contexts. The target team is optional; when omitted, the team transition is unknown and the model must not pretend it has calibrated a same-team or transfer scenario.

The health response exposes `bb_rating_future_prediction_loaded` and the loaded model version. The old `/api/v2/predictions/player-team` endpoint is no longer mounted; use the direct BB Prediction endpoint for future BB-Rating. A missing release disables only this new endpoint; it does not silently substitute the native-rating Prediction Model.

## Important limitations

- This PR adds the model path and API contract but does not train against the production database or prove that each destination/scenario cohort has sufficient history.
- The MAE release gate is necessary but not sufficient: review all scenario metrics and uncertainty coverage before promotion.
- Confidence intervals are empirical OOF residual intervals, not guaranteed frequentist coverage; their observed coverage must be monitored by scenario.
- Future team roster/role changes and causal acquisition impact are not inferred unless their features are known at prediction time. This model predicts the player's future contextual BB-Rating, not the causal impact of signing him.
- BB-Rating is contextual. A score of 75 in two different leagues means a similar relative standing in each peer population, not necessarily equal absolute player quality.

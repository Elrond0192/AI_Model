# Direct future BB-Rating prediction

## Contract

The player timeline has one consistent scale:
- past and present: observed BB-Rating from the pinned `BBRatingEngine`;
- future: predicted observed BB-Rating for target season `t+1`, on the same 1-100 scale.

The native statistical rating (0-10) is not a BB-Rating label and is never the target of this model. The legacy native Prediction Model endpoint and its 1–100 display calibration are no longer exposed as user-facing prediction features; this model predicts BB-Rating directly.

## Model and labels

`basketball_ai.bb_rating.future_prediction.BBRatingFuturePredictionModel` builds target labels by calling the official `BBRatingEngine` for each available player + target season + destination league + competition context. Player features and destination-team context are restricted to information available at or before the source season; team context uses the destination team's latest available historical season, never its target-season performance. Source BB-Rating is itself calculated from that source-season context. Canonical player global identity is used to join seasons when provider-specific player IDs change.

The model uses an expanding walk-forward evaluation by target season. Each fold trains only on target seasons earlier than its test season. Calibration for OOF evaluation is itself temporal: a target-season fold can use only calibration rows from earlier OOF target seasons. The final serving calibrator is then fit on the complete OOF record set.

## Scenario coverage

The model stores and evaluates source and destination league, team, competition/phase, and competition tier. It distinguishes:
- same team, league and competition;
- team change within the same league and competition;
- league change, with or without team change;
- regular-season ↔ playoff ↔ cup/other competition changes, including combined team/league/phase changes;
- promotion/tier-up, relegation/tier-down and same-tier league transfers when authoritative tier metadata exists;
- unknown or missing context, which uses explicit calibration fallback rather than guessing a same-team transfer.

Scenario calibration first attempts destination league + scenario, then scenario-only, then global. Sparse groups are not fitted. Returned predictions expose the selected calibration level and sample count.

## Baselines and release gate

The walk-forward report includes MAE, RMSE, bias and calibration error overall and by scenario and destination league. It compares the calibrated model with a context-aware baseline:
- source BB-Rating persistence when source and destination league/competition contexts match;
- otherwise the destination league + competition historical mean from the training fold, falling back to destination league mean and then the fold-wide mean.

The CLI does not save a candidate unless calibrated walk-forward MAE beats this baseline overall and no supported scenario/destination cohort (at least 40 OOF rows) regresses by more than 10% plus 0.25 BB-Rating points. This is a minimum gate, not proof that every scenario is reliable. Before promoting a release, inspect per-scenario/per-league sample sizes and metrics, especially combined transfers, new leagues, playoffs/cups and tier changes. Under-supported cohorts must remain clearly labelled and must not be treated as equally reliable.

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

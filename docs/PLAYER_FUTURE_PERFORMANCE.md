# Player Future Performance Model

## Purpose

Player Future Performance is the player-level season-ahead forecasting layer of AI_Model. It predicts a vector of next-season performance measures from data available through the player's current season.

It is intentionally independent from:

- **Prediction Model 2.6.0** — predicts next-season overall player rating.
- **BB-Rating 1.8** — describes observed performance relative to peers and is not a predictive model.

The three layers can therefore answer three different questions:

| Layer | Question |
| --- | --- |
| BB-Rating | How strong was the player's observed performance relative to his context? |
| Prediction Model | What overall rating can be expected next season? |
| Player Future Performance | What stat-level performance profile can be expected next season? |

## Target contract

The initial target vector is:

- PTS / 36
- AST / 36
- REB / 36
- STL / 36
- BLK / 36
- TS%
- USG%
- TOV%
- AST%
- ORB%
- DRB%
- 3P%
- FT%
- 3PA / 36
- FTA / 36
- Minutes / game

Per-36 targets separate production rate from playing time. Minutes are predicted as their own target and are then used to derive per-game estimates for points, assists, rebounds, steals and blocks. Derived per-game values are never treated as independently trained targets.

## Statistical identity

Training pairs are formed only when the same player has consecutive seasons within the same league and competition:

`player + league + competition + season(t) -> season(t+1)`

No gap is bridged. For example, 2019 -> 2021 does not create a 2019 -> 2021 training sample.

Features are calculated from source-season information only. The history used for a sample stops at season `t`, so target-season observations cannot leak into the feature vector.

## Features

The model uses current values, recent-history averages/variability and trend signals for the target metrics, plus:

- age and age-vs-peak
- current minutes and games played
- starter share
- current overall rating
- rating deltas and recent rating stability
- season experience
- position-family one-hot features
- competition one-hot features
- league one-hot features

The model does not assume the player's future team is known. This keeps the first version player-centric and suitable for off-season what-if forecasting.

## Validation

Production evaluation is an expanding walk-forward backtest by target season.

For each target season:

1. train only on earlier target seasons;
2. predict the target season out of sample;
3. record RMSE, MAE, bias and correlation;
4. compare with a persistence baseline equal to the source-season metric;
5. accumulate OOS absolute residuals for P50/P75/P90 uncertainty.

A metric is eligible for fitted serving only when it has at least 40 training samples. OOS uncertainty requires at least 20 OOS observations.

This is an evaluation gate, not an automatic quality claim: the Control Center must show the observed OOS results for each target before the model is used.

## Lifecycle and current-season handling

The default training path excludes the latest observed season as a target. This matches the live-data lifecycle where the latest season may still be **In corso**.

The latest season can still be used as the source season for serving. A future 2027 forecast can therefore use observed 2026 features without making 2026 a training target until that season is complete.

## Artifact

The production artifact is stored separately from the Prediction Model:

`models_saved/future_performance/`

It contains:

- `metadata.json`
- `models/<target>.joblib` for every trained target

The public API loads this sibling artifact. The existing Prediction Model production directory remains untouched.

## API

`POST /api/v2/future-performance/player`

Request:

```json
{
  "player_global_id": "PLAYER-GLOBAL-ID",
  "league": "ITA1",
  "season": 2026,
  "competition": "RS"
}
```

The response contains the target-season forecast, P50/P75/P90 historical OOS absolute-error levels and a bounded P90 range for each target.

## Training

CLI:

```bash
docker compose run --rm admin \
  python main.py \
  --mode future-performance-train \
  --database-profile production \
  --model-dir /app/models_saved
```

By default, the latest observed season is excluded as a target.

The Model Center exposes the same operation as **Future Performance → Genera / aggiorna modello**.

## Deployment

The API reads the artifact from the shared runtime model volume, but it remains outside `/app/models_saved/production`. Code changes still require the normal Docker rebuild/redeploy workflow.
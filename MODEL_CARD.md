# Model Card — BBallstat AI_Model

## Model

| Field | Value |
|---|---|
| Version | 2.0.0 |
| Primary target | next-season player rating (`t -> t+1`) |
| Base estimator | XGBoost regressor |
| Team/context layer | k-NN compatibility + bounded context adjustments |
| Output | predicted rating 0–10 + confidence interval |
| Production source | PostgreSQL `ai_source` canonical views |

AI_Model is an inference/training service. It is not a conversational model;
WordPress Chat V3/Bax owns intent detection, entity resolution and responses.

## Training unit

The base forecast is trained on chronological player-season pairs:

```text
features available through season t  ->  observed rating in season t+1
```

The canonical PostgreSQL adapter exposes **one row per global player and
season**. Domestic/European and RS/PO/TOT rows from the same season are not
allowed to become false chronological samples.

Training uses season-blocked partitions:

- older target seasons: training;
- penultimate target season: validation / early stopping;
- latest target season: conformal calibration;
- expanding walk-forward backtest: promotion evidence.

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

Missing optional advanced fields fall back to neutral defaults; the target
rating and chronological identity fields do not.

## Prediction path

```text
season-t player history
        |
        v
XGBoost t+1 forecast
        |
        +-- age/context adjustment
        +-- team compatibility
        +-- league/team context
        v
final rating + interval
```

Dynamic player/team scenarios are served through
`POST /api/v2/predictions/player-team`. Batch current-team forecasts are stored
in PostgreSQL `ai.player_forecasts` with model run/version metadata.

## Identity

Physical per-league IDs are never treated as global identities. The database
adapter resolves player/team `IdGlobal` and generates stable internal numeric
keys for the Python model. Raw global IDs stay server-side and are used only in
the authenticated service contract.

## Evaluation and promotion

Each candidate records RMSE/MAE/R², lineage, feature version, data cutoff and
walk-forward backtest results. An invalid/empty backtest is not promotable.
The registry maintains candidate, production and previous states for rollback.

Synthetic CSV data may be used for isolated development tests, but production
training is PostgreSQL-backed and real-data metrics must be evaluated on the
actual deployment dataset. No synthetic benchmark should be interpreted as a
production quality claim.

## Limitations

- Team-fit adjustments are associative, not causal estimates of transfer value.
- Injury, contract, coaching and off-court information are not explicitly
  modelled.
- Sparse-history players have wider uncertainty and weaker career features.
- A player-season spanning several competitions is represented by the canonical
  most-complete season aggregate so chronology remains valid; this intentionally
  sacrifices competition-specific target modelling.
- New leagues or major source-schema changes require rebuilding
  `ai_source_schema.sql` views and validating the data contract before retraining.

## Privacy and security

- Source schemas are read-only to AI_Model.
- Writes are restricted to model-owned schema `ai`.
- PostgreSQL credentials and API secrets are runtime secrets and are never
  committed.
- Browser clients never receive database credentials or generic SQL access.
- WordPress communicates with AI_Model server-to-server over authenticated
  HTTPS.

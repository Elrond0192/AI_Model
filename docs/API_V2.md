# AI_Model prediction contract v2

`AI_Model` is an inference service, not a chatbot. WordPress resolves player/team entities and calls `POST /api/v2/predictions/player-team` with server-side global IDs.

## Request

```json
{
  "player_global_id": "313267",
  "team_global_id": "313271",
  "league": "ITA1",
  "season": 2025,
  "competition": "PO"
}
```

`season` is the **source season** and the target is `season + 1`.

`competition` is open-ended. Known aliases are normalized (`playoffs -> PO`, `regular season -> RS`, `Super Cup -> SUPERCUP`); other labels are normalized to uppercase identifiers. A competition is accepted by a promoted model only if that model was trained on real consecutive samples for it. Merely sending a new label never enables an unsupported forecast.

## Isolation semantics

A request is resolved as:

```text
player + league + competition + source season
team   + league + competition + source season
```

Player history and team context are filtered to that league/competition before prediction. Therefore `competition=PO` does not silently substitute RS rows. The request returns `422` when the player has no exact source-season row for the requested competition/league, when the target team lacks the requested competition context, or when the promoted model has no trained competition vocabulary entry.

## Response

Every successful response includes model lineage, source/target season, prediction and confidence bounds plus:

```json
{
  "competition_support": {
    "mode": "isolated",
    "competition": "PO",
    "source_seasons": 3,
    "source_games": 24,
    "exact_source_games": 8,
    "calibration_scope": "competition",
    "calibration_samples": 42
  }
}
```

`calibration_scope="competition"` means the finite-sample conformal half-width was calibrated on that competition. `global` means that competition did not reach the configured calibration minimum, so the service uses the global final-ensemble conformal interval and says so explicitly.

## Training contract

Supervised pairs require:

```text
same player
same league
same competition
target season = source season + 1
```

The pooled model can learn shared basketball patterns, but `RS -> PO`, domestic -> European, and gap-year pairs are never training targets.

## BB-Rating uncertainty serving

`POST /api/v2/bb-rating/player` returns the existing deterministic BB-Rating 1–100
and, when the Calibration 1.16 artifact is loaded, an additional `uncertainty`
object.

The uncertainty object contains:
- the calibration and BB-Rating versions;
- the selected source (normally `league+exposure`, with documented fallbacks);
- empirical P50/P75/P90 thresholds for absolute next-season rating change;
- the exposure minutes and exposure band used;
- a bounded 1–100 P90 range centered on the current BB-Rating.

The uncertainty layer is observational: it does not modify, shrink or replace the
BB-Rating. The API can continue serving the rating when the uncertainty artifact
is temporarily unavailable, with `uncertainty.available=false`.

The production API loads the final artifact from `BB_RATING_UNCERTAINTY_PATH` when
configured, then from the model directory or its parent calibration directory. In
the standard Compose layout this also supports
`/app/models_saved/bb_rating_calibration/bb_rating_uncertainty.json` while the
prediction model remains isolated under `/app/models_saved/production`.

## Production lifecycle

Production Docker loads only explicitly promoted artifacts from `models_saved/production`. `production_state.joblib` includes run-owned style calibration, competition vocabulary and per-competition conformal state. Backtests report `by_competition`; promotion can reject a candidate that regresses substantially in a competition with enough OOT samples.

Precomputed forecasts are stored in `"AI"."PlayerForecasts"`, whose key already includes `competition`.

## Metric Rating Engine endpoints (FASE I)

WordPress / Chat consume the **precomputed, versioned** distributions
(`AI.MetricDistribution`) — percentile/tier are never recomputed per request
and no rating logic lives in WordPress.

### `POST /api/v2/metric-rating/rate`

Rate a raw advanced-metric value against the stored population:

```json
{
  "metric": "RAPTOR",
  "value": 5.82,
  "league": "ITA1",
  "season": "2025-26",
  "phase": "RS"
}
```

Response (same shape as the engine): `metric, value, percentile, zscore,
tier, label, sample_size, quality, population_source, fallback_used,
population_type, distribution_version, rating_version, above_reference,
below_reference, warnings`. `season` accepts an integer year or `"2025-26"`;
`phase` is the competition code (`RS`, `PO`, `TOT`, …). Errors: `422` for an
unknown metric, a non-finite value or a context without a stored distribution
(the fallback used by the engine is always reported, never hidden); `503`
when the distributions are not loaded (run the FASE G backfill first).

### `POST /api/v2/metric-rating/player-snapshot`

Per-metric rating snapshot for a player in a context (Chat use):

```json
{
  "player_global_id": "46238",
  "league": "ITA1",
  "season": 2025,
  "phase": "RS",
  "metrics": ["RAPTOR", "LEBRON", "VORP"]
}
```

Response: `{..., "ratings": {"RAPTOR": {value, percentile, zscore, tier,
label, quality, population_source, fallback_used, distribution_version},
"LEBRON": {...}, "VORP": {...}}}`. `404` when the player is not resolved,
`503` when distributions/data are not loaded.

## Authentication and network

WordPress calls server-to-server using `X-API-Key` or an externally provisioned Bearer token. Global database IDs must not be returned to the browser. PostgreSQL on the Docker host is reached through `host.docker.internal`; port 5432 must remain private.


## BB-Rating contextual endpoint (FASE I)

The BB-Rating is a separate descriptive layer from the season-ahead prediction
model. It evaluates observed performance against a contextual peer population
and returns a deterministic 1–100 score plus metric evidence.

### POST /api/v2/bb-rating/player

Request:

```json
{
  "player_global_id": "46238",
  "league": "ITA1",
  "season": 2025,
  "phase": "RS"
}
```

The primary peer population is the requested league + season + competition.
When that context contains at least the preferred minimum peer population
(default 25), the full context is used for scoring percentiles. Smaller
contexts remain in the same league/season/competition population down to the
minimum context size (default 10), with lower quality reported rather than
silently changing the comparison universe. Position family, age band and
combined role remain available for explanation and diagnostics but do not
replace the primary competition population. USG% is exposed as a
role/involvement signal but has no direct composite weight.

The response contains:

- bb_rating (1–100);
- dimension scores for impact, offense, defense and versatility;
- raw metric values and contextual percentiles;
- strengths and limitations derived from those percentiles;
- deterministic explanation text;
- peer-group definition, sample size, quality and metric coverage;
- bb_rating_version.

WordPress / Chat consumes the result and does not implement the rating formula,
percentile calculation or peer selection.

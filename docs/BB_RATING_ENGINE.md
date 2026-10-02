# BB-Rating Engine

The BB-Rating is the descriptive/contextual layer for player performance. It is
separate from the season-ahead Prediction Model.

The Prediction Model answers:

> What rating is this player expected to have next season?

BB-Rating answers:

> How is this player performing relative to comparable players in this
> league, season and competition?

## Contract

BBRatingEngine.rate_player() returns a deterministic score from **1 to 100**.

The score is built from contextual percentiles of observed metrics. The default
dimensions are:

| Dimension | Main evidence |
|---|---|
| Impact | RAPTOR total, VORP |
| Offense | PER, TS%, AST%, TOV% |
| Defense | RAPTOR defensive component, DBPM, net rating differential, STL%, BLK% |
| Versatility | REB%, hustle, foul-drawing rate |

The current weights are explicit in basketball_ai/bb_rating/engine.py. They are
a versioned product contract and should be recalibrated only after real-data
validation.

## Contextual peer population

The primary percentile population is:

league + season + competition

When that context has at least 25 rows, the full requested context is used for
every scoring percentile. The 25-row threshold is a quality threshold, not a
reason to substitute a different comparison universe.

For smaller contexts, the engine keeps the same league/season/competition
population down to 10 rows and reports lower quality. Below 10 rows the result
is explicitly marked limited_context.

Position family, age band and observed combined role remain part of the
peer-group metadata and explanation layer, but in v1.8 they do not replace the
primary competition population used for scoring.

Age bands are 18–21, 22–25, 26–29, 30–33 and 34+.

Position families are GUARD, WING and BIG, derived from the canonical position
labels. Position family, age band and observed ruolo_combinato remain descriptive
context; they no longer replace the primary competition population.

## Metric semantics and explanations

Every registered BB-Rating metric has a semantic definition independent of its
weight in the composite score. The API therefore returns, for each available
metric:

- raw value;
- contextual percentile;
- relative band;
- metric meaning;
- a deterministic human-readable interpretation.

This separation is deliberate. A metric can be important for explanation
without contributing to the 1–100 score.

USG% is a canonical example. A 34% USG can be described as very high offensive
involvement relative to peers, but USG% has zero composite weight because usage
describes role/load rather than effectiveness by itself. The same semantic
approach is applied to efficiency, creation, turnovers, shooting, rebounding,
defense, playing time and other registered metrics.

The semantic registry lives in basketball_ai/bb_rating/semantics.py. Adding a
new explainable metric is therefore a metadata operation first, not a new
algorithm.

## Missing data and quality

The engine renormalises composite weights across usable metrics while exposing
metric_coverage.

Quality is:

| Condition | Quality |
|---|---|
| peer sample >= 50 and coverage >= 80% | high |
| peer sample >= 25 and coverage >= 60% | medium |
| peer sample >= 10 and coverage >= 40% | low |
| otherwise | insufficient |

A low-quality result remains a numerical result, but the response explicitly
contains the quality flag so Chat/UI can qualify the explanation.

## API

POST /api/v2/bb-rating/player

Request:

{
  "player_global_id": "46238",
  "league": "ITA1",
  "season": 2025,
  "phase": "RS"
}

Response shape:

{
  "player_global_id": "46238",
  "player_name": "Example Player",
  "league": "ITA1",
  "season": 2025,
  "phase": "RS",
  "bb_rating": 87,
  "score_band": "Elite",
  "dimensions": {
    "impact": 91,
    "offense": 84,
    "defense": 79,
    "versatility": 70
  },
  "metrics": {},
  "strengths": [],
  "limitations": [],
  "explanation": "...",
  "peer_group": {},
  "quality": "high",
  "metric_coverage": 0.93,
  "bb_rating_version": "1.8"
}

WordPress remains a pure consumer: no percentile, peer selection or BB-Rating
logic belongs in PHP.

## Current status

This is **FASE I of the BB-Rating layer** (version 1.8). The implementation is deterministic
and API-ready, but the weights and peer thresholds must still be validated
against real production data before treating the 1–100 score as the final
public methodology.

The next validation should measure score distributions, stability across
leagues/seasons, position and age groups, missing-metric coverage and whether
the explanations agree with the underlying percentiles.


## Calibration v1.8

The calibration layer is diagnostic-only and does not train an ML model. It uses the
the same primary-context peer-selection contract and empirical percentile convention as BBRatingEngine,
but evaluates the complete real-data frame in batch. The BB-Rating reads the canonical
observed contract; On/Off fields are resolved upstream by ai_source_full.sql.

Run from the repository/container:

```bash
docker compose run --rm admin \
  python main.py --mode bb-rating-calibrate \
  --database-profile production \
  --output-dir /app/models_saved/bb_rating_calibration
```

The command reads the canonical observed contract with the analysis loader
and writes:

- `bb_rating_calibration.json`
- `bb_rating_calibration.md`

The report contains:

- registry/semantic source audit;
- dataset coverage by league, season and competition;
- peer population source and fallback frequency;
- metric availability, missingness and zero-rate diagnostics;
- contextual percentile distributions;
- effective weights after missing-data renormalisation;
- BB-Rating score distribution by league, season and competition;
- consecutive player-season stability;
- heuristic diagnostic warnings.

These warnings are investigation signals, not promotion gates. No BB-Rating weight,
threshold or public methodology should be frozen from the report alone; the next step
is to review the real-data output and decide whether the peer model, metric set and
weights need calibration.


## Calibration v1.10 — empirical reliability diagnostics

Calibration 1.10 is diagnostic-only. The public BB-Rating remains version 1.8:
no score weights, peer rules or Prediction Model outputs are changed.

The v1.10 report adds:
- Pearson and Spearman stability for the rounded 1–100 score;
- Pearson and Spearman stability for the continuous composite percentile before rounding;
- exposure-stratified stability using minutes total when available, otherwise games played × minutes per game;
- stability by league, position family and age band;
- stability of each active scoring metric's contextual percentile;
- an empirical reliability proxy by exposure and active-metric persistence, with no automatic score correction.

The purpose is to identify whether remaining year-to-year noise is concentrated in
low-exposure observations or in specific metrics before introducing any reliability
adjustment into the scoring contract.


### Empirical reliability

Calibration 1.10 estimates persistence diagnostically rather than changing the
public score. Consecutive-season Spearman persistence is used as a test-retest
proxy; exposure quartiles are fitted with a saturating curve of the form
`reliability_proxy = asymptote × exposure / (exposure + half_exposure)`.
Active metric persistence is also aggregated using configured weight and metric
availability. These outputs are evidence for a future reliability policy and
are not applied as score shrinkage or weight adjustments.


## Calibration v1.11 — empirical stability profile

Calibration 1.11 keeps the BB-Rating public contract at version 1.8 and remains
diagnostic-only. It does not modify the 1–100 score, metric weights, peer
selection, source data or Prediction Model.

In addition to the v1.10 persistence diagnostics, the report exposes an empirical
stability profile by exposure quartile:

- `stability_score`: a relative persistence index where the highest observed
  exposure-quartile persistence in the calibration dataset is 100;
- `confidence_level`: the empirical exposure band (`low`, `moderate`,
  `high`, `very_high`) mapped from Q1–Q4;
- `expected_rating_variation`: the median absolute next-season BB-Rating change
  observed within that exposure quartile.

This profile is deliberately descriptive. It provides a candidate foundation
for a future UI confidence/uncertainty layer without shrinking or otherwise
altering the BB-Rating itself.

The 1.11 JSON therefore contains:

```json
"stability_profile": {
  "status": "diagnostic_only",
  "stability_score_definition": "...",
  "confidence_level_definition": "...",
  "expected_rating_variation_definition": "...",
  "bands": []
}
```

No reliability or stability correction is applied automatically.


## Calibration v1.12 — empirical uncertainty profile

Calibration 1.12 remains diagnostic-only and leaves the public BB-Rating at
version 1.8. It does not change production weights, peer selection, the 1–100
score, source data, or Prediction Model outputs.

The report adds an empirical uncertainty profile based on consecutive
player-season absolute BB-Rating changes:

- overall P25/P50/P75/P90 variation;
- variation by exposure quartile;
- variation by peer-context sample-size quartile when there is sufficient
  observed variation;
- variation by metric-coverage quartile when there is sufficient observed
  variation;
- variation by league;
- Spearman association between each signal and absolute rating change;
- exposure-controlled rank association for peer sample size and metric coverage.

The exposure-controlled association is a diagnostic partial Spearman-style
measure: rank the outcome and predictor, remove the linear rank component
explained by exposure, then correlate the residuals. It is used only to test
whether context size or metric coverage contains information beyond exposure;
it does not assign production weights.

The uncertainty profile intentionally reports empirical variation rather than
claiming a probability that a rating is correct. In particular,
`expected_rating_variation` from calibration 1.11 is retained as a historical
median change by exposure band, while 1.12 adds the broader empirical
distribution needed for future uncertainty intervals.

No uncertainty correction is applied automatically to the public BB-Rating.


## Calibration v1.14 — within-league uncertainty decomposition

Calibration 1.14 remains diagnostic-only and keeps the public BB-Rating at
version 1.8. It separates between-league variation from within-league exposure
variation when studying next-season absolute rating change.

The report adds:
- a league × exposure table with empirical P25/P50/P75/P90 absolute rating change;
- a rank-based in-sample variance decomposition for exposure alone;
- the incremental R² obtained by adding league indicators;
- the additional incremental R² from league × exposure interactions.

The decomposition is descriptive rather than causal and is not converted into
production weights. The main decision signal is the incremental contribution of
league after exposure: if it is negligible, future uncertainty can remain
exposure-conditioned only; if it is material, a league-aware uncertainty layer
can be evaluated separately.

The cell table uses a minimum support threshold of 50 consecutive player-season
pairs per league × exposure cell. Smaller cells are omitted rather than being
presented as stable estimates.

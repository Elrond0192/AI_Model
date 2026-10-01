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
labels. The exact observed ruolo_combinato is used only when its population is
large enough.

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
  "bb_rating_version": "1.7"
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

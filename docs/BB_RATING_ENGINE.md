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

This is **FASE I of the BB-Rating layer (version 1.8)**. The implementation is deterministic
and API-ready. Calibration validates the existing methodology against real production
data without changing the public 1–100 rating.

## Calibration 1.16

The calibration command performs two separate operations:

1. **Temporal OOS validation** of uncertainty using expanding walk-forward empirical
   P50/P75/P90 thresholds.
2. **Final production fit** of the validated league × exposure uncertainty structure
   on all completed target seasons available before the latest observed target season.

Run from the repository/container:

\`\`\`bash
docker compose run --rm admin \
  python main.py --mode bb-rating-calibrate \
  --database-profile production \
  --output-dir /app/models_saved/bb_rating_calibration
\`\`\`

The command writes:

- bb_rating_calibration.json — complete diagnostic and validation report;
- bb_rating_calibration.md — human-readable report;
- bb_rating_uncertainty.json — final serving artifact for the uncertainty layer.

### Final uncertainty artifact

The production fit keeps the public **BB-Rating 1.8** unchanged. It estimates the
absolute next-season BB-Rating change with empirical P50/P75/P90 thresholds.

The fitted structure is:

league + exposure_band

where exposure is divided into four quartile bands. League-specific exposure
thresholds are used when enough historical exposure observations exist; otherwise the
serving logic follows the documented fallback order:

league+exposure → league → exposure → global

Each persisted table includes its empirical sample size n and P50/P75/P90 values.
The artifact also records the exposure thresholds used to assign bands, the completed
target seasons used for fitting, the excluded latest target season, source/current
seasons, available exact cells and fallback usage.

By default the latest observed target season is excluded because it may still be
incomplete. For the current production dataset this means that 2026 is excluded from
the final fit and the completed target history through 2025 is used.

No uncertainty correction, shrinkage or alteration of the 1–100 BB-Rating is applied
by the calibration command itself. The persisted uncertainty artifact is a separate
serving layer.

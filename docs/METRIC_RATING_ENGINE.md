# Metric Rating Engine

Centralized interpretation layer for advanced metrics produced by
HoopmetricsEngine. The engine consumes metrics that **already exist** in the
PostgreSQL `"AI_Source"` contract and turns them into an interpretable rating:

```text
HoopmetricsEngine
    ↓
AdvancedStats  (Analisi.AdvancedStats_Player_*)
    ↓
PostgreSQL
    ↓
"AI_Source"."PlayerCompetitionStats"     ← read-only contract (AI_Model adapter)
    ↓
AI_Model → MetricRatingEngine
    ↓
MetricRating (percentile / z-score / tier / label / quality)
    ↓
API / Chat / AI / Frontend (WordPress is consumer only)
```

The engine never modifies how HoopmetricsEngine computes AdvancedStats and
never recomputes advanced metrics from raw possessions. It only interprets
values that are already stored.

## Design principles

- One engine for every metric. Adding a metric is **registration + direction +
  value source**, never a new algorithm.
- Distributions are **precomputed and versioned**, independent of any single
  API request.
- The percentile is **always relative to a contextual population**
  (metric + league + season + phase + population type), never computed against
  every player in the database.
- Fallbacks and quality are **explicitly reported**, never hidden from the
  user or the AI.
- Never produce `NaN` / `Infinity`. A degenerate or missing population yields
  `None` rating fields, not a fake-precise number.

## Repository layout

```
basketball_ai/metric_rating/
├── __init__.py        public exports
├── definitions.py     MetricDefinition + METRIC_DEFINITIONS registry
├── tiers.py           percentile tier table (configurable)
├── distribution.py    DistributionStats + quantile builder + quality
├── population.py      PopulationBuilder (context + qualification + fallback)
├── rating.py          MetricRatingEngine.rate(...) → MetricRating
└── poc.py             FASE D POC report generator (CLI + library)
```

## Metric definitions (FASE 1)

`MetricDefinition` is a frozen dataclass:

| field | meaning |
|---|---|
| `metric` | canonical key, e.g. `RAPTOR` |
| `source_column` | column in `"AI_Source"."PlayerCompetitionStats"` that carries the value |
| `direction` | `higher_better` \| `lower_better` |
| `zero_reference` | interpretable zero point (RAPTOR/LEBRON/VORP = 0) |
| `rating_enabled` | whether a rating can be produced |
| `replacement_reference` | optional reference for above/below semantics (VORP: 0.0) |

First version:

| metric | source_column | direction | zero_reference | replacement_reference |
|---|---|---|---|---|
| RAPTOR | `raptor_total` | higher_better | 0 | — |
| LEBRON | `lebron_total` | higher_better | 0 | — |
| VORP | `vorp` | higher_better | 0 | 0 |

VORP keeps the *above/below replacement* concept as metadata
(`above_reference` / `below_reference` in the result). This reference is
explanatory and **never substitutes the percentile**.

Adding a new metric (BPM, EPM, PER, WS, WS/48, …) requires:

1. register a `MetricDefinition` (metric, direction, source column);
2. make sure the column exists in the `"AI_Source"` contract
   (`ai_source_competition.sql` maps it if HoopmetricsEngine exposes it);
3. no new algorithm.

## Population builder (FASE 2)

`PopulationBuilder` selects the rows to compare against. The population is
contextualized by:

- `metric` (source column);
- `league_key` (`ITA1`, `ESP1`, `GER1`, `FRA1`, `EL`, …);
- `season` (numeric start year, e.g. `2025` for `2025-26`);
- `phase` (competition: `RS`, `PO`, `TOT`, `CUP`, `SUPERCUP`);
- `population_type` (`qualified_players` by default).

Qualification rules are configurable (`PopulationConfig`):

| rule | default | note |
|---|---|---|
| `min_games` | 10 | uses existing `games_played` column |
| `min_minutes_per_game` | 0.0 | uses existing `minutes_per_game` column; kept at 0 until the POC verifies minutes coverage |
| `exclude_non_finite` | True | drops `NULL`/`NaN`/`±inf` values in the metric column |
| `population_type` | `qualified_players` | `all_players` skips games/minutes filters |

Thresholds are deliberately **not invented columns**: they only use
`games_played` and `minutes_per_game` already present in the contract. Values
are recalibrated in the POC (FASE D) against real data.

### Fallback chain (FASE 8)

Triggered when the exact population has fewer than `min_samples` (default 50)
valid rows. Configurable order:

1. `exact` — same league + season + phase;
2. `league_multi_season` — same league + phase, all seasons;
3. `competition_family` — same league + season, phase broadened to the
   compatible competition family (e.g. `RS → {RS, TOT}`, `PO → {PO, TOT}`);
4. `global` — all leagues / seasons / phases for the metric.

The result always reports `population_source` (which level won),
`fallback_used`, `population_type` and `quality`.

## Distribution engine (FASE 3)

`build_distribution` computes from the qualified values:

```text
sample_size  min  p10  p25  p40  p50  p60  p75  p90  p97.5  max  mean  stddev
```

- quantiles use **linear interpolation** (`numpy` `method="linear"`);
- `stddev` is the population standard deviation (`ddof=0`);
- distributions are computed in batch and cached/stored — never per request.

## Percentile (FASE 4)

The percentile is defined as the **inverse of the stored quantile function**:
linear interpolation over the distribution points
`(0, min) (10, p10) (25, p25) (40, p40) (50, p50) (60, p60) (75, p75) (90, p90)
(97.5, p97.5) (100, max)`, expressed in `[0, 1]`.

- `value` below the distribution minimum → `0.0`; `value ≥ max` → `1.0`;
- `value` equal to a stored quantile → that quantile's percentile; with ties,
  the highest percentile sharing the value is kept;
- degenerate distribution (`min == max`): any value equal to it → `0.5`.

For `lower_better` metrics the percentile is inverted (`1 − p`).

## Z-score (FASE 6)

```text
zscore = (value − mean) / stddev
```

- `stddev == 0 → zscore = None`;
- result is `None` whenever the population is insufficient;
- never `NaN` / `Infinity`.

## Tiers (FASE 5)

Tiers are **percentile thresholds**, not raw metric thresholds, and are
configured centrally (`tiers.py`). Defaults:

| tier | percentile range (fraction) | label |
|---|---|---|
| 1 | [0.00, 0.10) | Very Poor |
| 2 | [0.10, 0.25) | Poor |
| 3 | [0.25, 0.40) | Below Average |
| 4 | [0.40, 0.60) | Average |
| 5 | [0.60, 0.75) | Good |
| 6 | [0.75, 0.90) | Very Good |
| 7 | [0.90, 0.975) | Elite |
| 8 | [0.975, 1.00] | Superstar |

Ranges are half-open `[low, high)`; the last tier includes `1.0`.

## Population quality (FASE 7)

Configurable thresholds on `sample_size`:

| sample_size | quality |
|---|---|
| ≥ 200 | high |
| 100–199 | medium |
| 50–99 | low |
| < 50 | insufficient |

With `insufficient` quality the engine does **not** produce a precise-looking
rating: `percentile`, `zscore`, `tier`, `label` are `None`; the caller still
receives `sample_size`, `quality`, `population_source` and `fallback_used`.

## Rating output (FASE 4)

```python
rating_engine.rate(metric="RAPTOR", value=5.82, league="ITA1",
                   season="2025-26", phase="RS", frame=player_stats)
```

```json
{
  "metric": "RAPTOR",
  "value": 5.82,
  "percentile": 0.964,
  "zscore": 1.91,
  "tier": 8,
  "label": "Elite",
  "sample_size": 284,
  "quality": "high",
  "population_source": "exact",
  "fallback_used": false,
  "population_type": "qualified_players",
  "distribution_version": "1.0",
  "rating_version": "1.0",
  "above_reference": null
}
```

VORP additionally reports `above_reference` / `below_reference` relative to
`replacement_reference = 0`.

## Versioning (FASE 10)

- `distribution_version` — changes whenever population rules, filters or
  quantile conventions change;
- `rating_version` — changes whenever tier tables or quality thresholds
  change.

Both flow through `DistributionStats` and every `MetricRating` result, so a
stored rating can always be traced back to the exact rules that produced it.

## Storage (FASE 9)

Verified: the database contains no equivalent structures (schema `"AI"` only
has `ModelRuns`/`PlayerForecasts`). The storage contract lives in
`basketball_ai/data/ai_metric_schema.sql` (schema `"AI"`):

- `AI.MetricDefinition` — metric registry (direction, zero_reference,
  rating_enabled, replacement_reference);
- `AI.MetricDistribution` — one row per (metric, league_key, season,
  competition, population_type, distribution_version) with min/p10…p97.5/max,
  mean, stddev, sample_size, quality, population_source, fallback_used;
- `AI.MetricRating` — one row per (metric, player_global_id, league_key,
  season, competition, distribution_version, rating_version) with value,
  percentile, zscore, tier, label, quality and provenance.

Apply the DDL, then the backfill:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_schema.sql
python -m basketball_ai.metric_rating.backfill \
    --csv data/metric_poc/player_stats.csv \
    --out data/metric_poc/backfill
psql -d YOUR_DATABASE -f data/metric_poc/backfill/apply_backfill.sql
```

AdvancedStats and the `"AI_Source"` contract are **never modified**; AI_Model
only reads them and generates the interpretive layer. Indexes are minimal (a
lookup index per table) — reviewed against `docs/DB_INDEXES.md`.

## Performance (FASE 18)

- Distributions are precomputed (batch) and refreshed after the ETL, following
  the existing `"AI_Source"."RefreshContext"` pattern — never per request;
- `rate()` against a stored `DistributionStats` is O(1) over the 10-point
  quantile table (no percentile recomputation, no table scan);
- no N+1 queries; indexes are reviewed against `docs/DB_INDEXES.md` before
  adding any.

## AI model integration (FASE 14)

The rating features are **additive**: `<metric>_value`, `<metric>_percentile`,
`<metric>_zscore` and `<metric>_tier` are added per context **without
replacing** the original metric columns. Implementation in FASE H:

- `basketball_ai/metric_rating/features.py` — vectorized provider
  `add_rating_features(frame, distribution_rows)` (percentile via `np.interp`
  over the stored quantile function, z-score with `NaN` on zero stddev, tier
  via `searchsorted`), equivalent to the scalar engine; NaN where no
  distribution exists. `player_rating_snapshot(...)` returns the per-metric
  raw/percentile/tier/label/population/quality dict for Chat.
- Training hook: `SeasonAheadPerformanceModel.train/prepare_features` accept
  `rating_distributions` (the `AI.MetricDistribution` rows). When provided,
  the rating columns are added to `FEATURE_COLS`; **without them the behaviour
  is identical to today** (toggle, no OLD/NEW comparison required).
- `XGB_DEVICE=cpu|cuda` overrides GPU auto-detection (prediction-neutral) so
  training/backtests can run on any host.
- Chat: `vorp` added to `_PLAYER_METRICS`; the engine is reusable via
  `player_rating_snapshot` (raw + percentile + tier + population + quality),
  keeping percentile/tier logic out of WordPress.

The tier is primarily for explainability / Chat / UI; the model features rely
mainly on raw value + percentile + z-score.

## Chat / API (FASE 15–16)

The engine is reusable by the Chat scenario layer and exposed through a new
API endpoint (e.g. `POST /api/v2/metric-rating/rate`). WordPress is a pure
consumer; percentile/tier logic stays out of PHP/WordPress.

FASE I ships two endpoints (see `docs/API_V2.md`):

- `POST /api/v2/metric-rating/rate` — rate a raw value against the stored,
  versioned population (O(1), never recomputed per request);
- `POST /api/v2/metric-rating/player-snapshot` — per-metric
  raw/percentile/tier/label/population/quality snapshot for the Chat.

The API loads `AI.MetricDistribution` at startup
(`basketball_ai/metric_rating/store.py`); missing distributions only disable
the rating endpoints (forecasts keep working) and return `503` with a clear
message.

## Open issues

1. **NULL → 0 coalescing**: `"AI_Source"."PlayerCompetitionStats"` applies
   `COALESCE(metric, 0)`, so a missing value is indistinguishable from a real
   `0`. The population builder therefore drops non-finite values at the frame
   level, but the definitive fix belongs to the SQL contract (read the raw
   `CompetitionStatsRawInternal` view or expose a metric view that preserves
   `NULL`). Scheduled with storage/backfill (FASE F/G).
2. `vorp` is not yet exposed in `_PLAYER_METRICS` for Chat — added at
   integration time.
3. Qualification thresholds must be calibrated on real data in the POC.

## FASE D — POC on real data

1. **Dump the contract** (read-only, nothing is written). Run
   `basketball_ai/data/ai_metric_poc_dump.sql` — it creates the read-only
   `"AI_Source"."MetricPocExtract"` view **and** writes the CSV in the same
   psql session (and disables parallel query, whose workers cannot run the
   adapter's PL/pgSQL helpers):

   ```bash
   psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_poc_dump.sql
   ```

   The dump covers RAPTOR/LEBRON/VORP plus identity and qualification columns
   (`league_key`, `season`, `competition`, `games_played`,
   `minutes_per_game`, `games_started`, `starter_pct`) for every league
   available (ITA1, ESP1, GER1, FRA1, EL, …), all seasons and phases.
   Requires the `"AI_Source"` contract views; the `\copy` path is relative to
   the psql working directory (edit the file if you run psql elsewhere).

   **Fallback when the `"AI_Source"` contract is not installed** (e.g. the
   database only has the physical HoopmetricsEngine schemas
   `Analisi`/`Anagrafiche`): use
   `basketball_ai/data/ai_metric_poc_dump_physical.sql` instead. It is a
   one-off extraction that reads `Analisi.AdvancedStats_Player_*` and
   `Anagrafiche.*` directly (same key mapping as the adapter) and **preserves
   NULL metric values** instead of coalescing them to 0:

   ```bash
   psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_poc_dump_physical.sql
   ```

   then uncomment the `\copy` line in the file (adjust the output path). If
   the `"AI_Source"` contract is missing, first run the diagnostic:

   ```sql
   SELECT schema_name FROM information_schema.schemata ORDER BY 1;
   SELECT table_schema, table_name FROM information_schema.tables
   WHERE lower(table_schema) IN ('ai_source', 'analisi', 'anagrafiche')
   ORDER BY 1, 2 LIMIT 300;
   ```

   **No-psql alternative (recommended when psql scripts are not an option)**:
   export the physical tables **as-is** to CSV from any tool (pgAdmin / DBeaver
   export, or plain `\copy` of the table) and put them in
   `data/metric_poc/raw/` with the names documented in
   `data/metric_poc/raw/README.md`
   (`advancedstats_player_<LEAGUE>.csv` from `Analisi.AdvancedStats_Player_*`,
   `anagrafiche_<LEAGUE>.csv` from `Anagrafiche.*`; header row included, column
   casing does not matter). Then normalize and run the POC:

   ```bash
   python -m basketball_ai.metric_rating.ingest \
       --raw data/metric_poc/raw \
       --out data/metric_poc/player_stats.csv
   python -m basketball_ai.metric_rating.poc \
       --csv data/metric_poc/player_stats.csv \
       --out data/metric_poc/report
   ```

   The ingest mirrors the adapter's key mapping (lowercase keys + candidate
   lookup), preserves NULL metric values and dedupes rows.

2. **Generate the report**:

   ```bash
   python -m basketball_ai.metric_rating.poc \
       --csv data/metric_poc/player_stats.csv \
       --out data/metric_poc/report
   ```

   Produces `distributions.json`, `players.json` and `report.md` with:

   - distribution table: `metric, league, season, phase, sample_size, mean,
     stddev, p10, p25, p50, p75, p90, p97.5, quality` (exact context, no
     fallback — this is the FASE 13 validation artifact);
   - representative players per context (top / median / bottom):
     `player, value, percentile, zscore, tier, label, quality`,
     with `population_source` and `fallback_used` provenance;
   - data-quality signal: fraction of exact-0 values per context (NULL→0
     coalescing detector, open issue #1).

3. **Validate (FASE 13)** before touching the model: superstars in the top
   tiers, average players near the median, marginal players at the bottom,
   insufficient-sample players excluded, and sensible distributions across
   leagues and seasons. If a distribution is wrong, fix the **population**
   (qualification rules, phase selection) — never bend the tier table to make
   results look good.

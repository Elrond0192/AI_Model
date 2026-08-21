# Data Card — AI_Model PostgreSQL Contract

**Contract:** `"AI_Source" competition-v1`

The forecasting feature contract consumes the already calculated
HoopmetricsEngine/AdvanceStats player and team tables. It does not recompute
advanced metrics from Boxscore or PBP. Boxscore supplies only missing starter
information; PBP feeds the separate scenario-serving aggregates.
**Model target:** same-league, same-competition next-season player rating (`t -> t+1`)

AI_Model never queries BBallstat physical tables directly. PostgreSQL adapters expose stable read-only views; source schemas remain untouched.

## Canonical production views

### `"AI_Source"."Leagues"`
One row per discovered league with stable internal ID and league key.

### `"AI_Source"."Teams"`
One row per team `IdGlobal`, latest registry state. Used for entity lookup only.

### `"AI_Source"."Players"`
One row per player `IdGlobal`, latest identity state. `IdGlobal` remains server-side.

### `"AI_Source"."TeamPlayerRelations"`
Roster membership by season, resolved through global identities. Used to reconstruct position/team state as of the source season.

### `"AI_Source"."PlayerCompetitionStats"`
One row per:

```text
player_id + league_id + season + competition
```

This is the supervised production source. It preserves `RS`, `PO`, `CUP`, `SUPERCUP`, `TOT` and normalized future competition labels instead of collapsing them into one season row. Known aliases are normalized (`PLAYOFFS -> PO`, `REGULAR SEASON -> RS`), while unknown labels are retained as normalized uppercase identifiers.

Rows without a non-null `ValLegaPerGame` / `rating` target are excluded. Team-specific duplicates inside the same league/season/competition are resolved by preferring all-team aggregates and then the row with most games.

### `"AI_Source"."TeamCompetitionStats"`
One row per:

```text
team_id + league_id + season + competition
```

Contains competition-specific pace, ORtg, DRtg, NetRtg, three-point rate, assists and star usage. A playoff forecast therefore receives playoff team context rather than regular-season style.

## Installation order

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_competition.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Re-run the adapters after adding a league, changing source schemas or introducing a newly ingested competition, then validate and retrain.

## Physical source discovery

```text
Anagrafiche.<LEAGUE>
Anagrafiche.Team_<LEAGUE>
Analisi.AdvancedStats_Player_<LEAGUE>
Analisi.AdvancedStatsTeam_<LEAGUE>          optional
Analisi.PlayerRoles_<LEAGUE>                optional
Analisi.AdvancedStatsOnOffCourt_<LEAGUE>    optional
Analisi.AdvancedStats_Clutch_<LEAGUE>       optional
Boxscore.<LEAGUE>                            optional
```

The adapters expect numeric `Season`. Historical casing is normalized through JSONB helpers.

## Forecast pairing rules

A supervised sample is created only when all of these are equal between source and target except the year:

```text
player
league
competition
```

and:

```text
target_season == source_season + 1
```

Examples:

```text
ITA1 / PO / 2024 -> ITA1 / PO / 2025      valid
ITA1 / RS / 2024 -> ITA1 / PO / 2025      invalid
ITA1 / PO / 2024 -> EL   / PO / 2025      invalid
ITA1 / PO / 2023 -> ITA1 / PO / 2025      invalid gap
```

The pooled estimator may learn shared basketball patterns across competitions, but each row's history, target and online prediction context remain isolated to the requested league/competition.

## Temporal correctness

Production training/backtesting enforces:

- one target row per player + league + season + competition;
- one historical team row per team + league + season + competition;
- exact consecutive pairs only;
- player history and team context clipped to the source season;
- same league and same competition for source/target pairs;
- calibration target season never used for estimator fitting;
- walk-forward target seasons excluded from each fold's training snapshot.

## Data quality gates

```bash
python main.py --mode validate-data --database-profile production
```

The validator reports available seasons, consecutive pairs globally and by competition, games/ratings by competition, context gaps and rating distributions by league. A competition with observations but zero consecutive pairs remains visible but is not serveable until enough historical data exists and the model is retrained.

PostgreSQL loading fails closed on missing views/columns, empty mandatory views, null competition labels and duplicate canonical keys.

The application does **not** normalize ratings across leagues automatically; cross-league target comparability must be demonstrated using real out-of-time evidence.

## Identity and ownership

- `global_id`: server-side model/database identity, never a browser identifier;
- `"AI_Source"`: read-only adapter views;
- `ai`: model-owned outputs (`model_runs`, `player_forecasts`);
- `Anagrafiche`, `Analisi`, `Boxscore`: source schemas, never written by AI_Model.

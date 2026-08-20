# Data Card — AI_Model PostgreSQL Contract

**Contract:** `ai_source` v2  
**Model target:** season-ahead player rating (`t -> t+1`)

AI_Model never queries BBallstat physical tables directly. The PostgreSQL adapter exposes stable read-only model views, while source schemas remain untouched.

## Canonical views

### `ai_source.leagues`
One row per discovered league. Includes stable internal ID, league key, country and observed metadata used by feature engineering.

### `ai_source.teams`
One row per team `IdGlobal`, using the latest registry/team state. This view is appropriate for current entity lookup only; it must not be used as historical team context during training or backtesting.

### `ai_source.team_season_stats`
One row per global team and season. Created by `basketball_ai/data/ai_source_team_season.sql` after the base adapter. It contains historical pace, ORtg, DRtg, NetRtg, three-point attempt rate, assists and roster-independent team context. This view is mandatory for leakage-safe compatibility training and historical prediction snapshots.

### `ai_source.players`
One row per player `IdGlobal`, using the latest identity state. `IdGlobal` is the cross-season server-side identity; league-local IDs are adapter-only join keys.

### `ai_source.player_stats`
Exactly one canonical target row per player and season. Physical data can contain domestic/European, `TOT`, `RS`, `PO`, team-specific and all-team rows for the same season. Those rows are not separate chronological seasons and are collapsed before model training.

Rows without a non-null `ValLegaPerGame` / `rating` target are excluded. The production training pipeline then creates a sample only when the same player has an **exact next season**: a 2023 row may target 2024, but never 2025 if 2024 is missing.

### `ai_source.team_player_relations`
Canonical player/team membership by season, assembled from available Boxscore, player aggregate and anagraphic information and resolved through global identities.

## Installation order

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_team_season.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Re-run the source adapters after adding a new league or materially changing physical table schemas.

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

The adapter expects numeric `Season`. Column-name case is normalized through JSONB so PostgreSQL imports that preserved historical casing remain usable.

## Temporal correctness

Production training/backtesting observes these rules:

- one target observation per `player_id + season`;
- one historical team row per `team_id + season`;
- only exact consecutive `t -> t+1` samples;
- team and player state in a historical prediction is clipped to the source season;
- the calibration target season is never used for estimator fitting;
- walk-forward test target seasons are not present in the corresponding training snapshot.

## Data quality gates

`postgres_loader.py` fails closed when a required view/column is missing, when mandatory views are empty, or when player/team-season duplicates violate the contract.

```bash
python main.py --mode validate-data --database-profile production
```

The validator also reports:

- available seasons;
- number of exact consecutive forecast pairs;
- players with gaps in season history;
- rating distribution by league;
- a warning when league-level target means differ enough that `ValLegaPerGame` cross-league comparability should be reviewed.

The application does **not** normalize the target across leagues automatically: such a transformation must be justified by real data and backtests rather than assumed.

## Identity and ownership

`global_id` is a server-side model/database identity, not a browser identifier. AI_Model receives it only through authenticated server-to-server calls.

- `ai_source`: read-only adapter views;
- `ai`: model-owned outputs (`model_runs`, `player_forecasts`);
- `Anagrafiche`, `Analisi`, `Boxscore`: source schemas, never written by AI_Model.

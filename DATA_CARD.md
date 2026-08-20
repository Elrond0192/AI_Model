# Data Card — AI_Model PostgreSQL Contract

**Contract:** `ai_source` v1  
**Model target:** season-ahead player rating (`t -> t+1`)

AI_Model does not query BBallstat physical tables directly. The database adapter
in `basketball_ai/data/ai_source_schema.sql` creates five stable, read-only
views over the unified PostgreSQL tables.

## Canonical views

### `ai_source.leagues`

One row per discovered league, including stable internal ID, league key,
country, observed pace/offensive rating and `max_games` for durability features.

### `ai_source.teams`

One row per team `IdGlobal`, using the latest registry state. It exposes
`global_id`, name, league, pace, ORtg/DRtg/NetRtg, 3PA rate and assists.
Playing style and star usage are derived after load.

### `ai_source.players`

One row per player `IdGlobal`, using the latest registry state. It exposes
`global_id`, identity/demographics, position and current team/league.
`IdGlobal` is the cross-season identity; physical per-league `Id` values are
adapter-only join keys.

### `ai_source.player_stats`

Exactly one canonical row per player and season. Physical tables can contain
`TOT`, `RS`, `PO`, team-specific and all-team rows for the same season. Treating
those as separate chronological observations would corrupt a `t -> t+1` target.

Selection priority is:

1. all-team row (`TeamId IS NULL`);
2. `TOT`;
3. `RS`;
4. another competition only if neither exists.

The selected `TOT` aggregate is exposed as season context `RS` because the
production forecast is season-level, not a competition-split forecast.
Season-total volume columns are converted to per-game values. Rows without a
non-null `ValLegaPerGame` / `rating` target are excluded.

### `ai_source.team_player_relations`

Canonical player/team membership by season, assembled from Boxscore and
team-specific player aggregate rows and resolved through `IdGlobal`.

## Physical source discovery

The adapter discovers current unified tables automatically:

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

Every discovered table uses numeric `Season`. Legacy `<LEAGUE>_<YYYY>`
partition naming is not used. Column-name case is normalized through JSONB so
PostgreSQL imports that preserved SQL Server casing remain usable.

## Data quality gates

`postgres_loader.py` checks that all five views exist, required columns exist,
and players/teams/player stats are non-empty.

```bash
python main.py --mode validate-data --database-profile production
```

The operations console expects at least five seasons.

## Identity and ownership

`global_id` is a server-side database identity, not a browser identifier.
AI_Model does not receive WordPress salts, user credentials or generic SQL.

- `ai_source`: read-only adapter views.
- `ai`: model-owned outputs (`model_runs`, `player_forecasts`).
- `Anagrafiche`, `Analisi`, `Boxscore`: source schemas, never written by AI_Model.

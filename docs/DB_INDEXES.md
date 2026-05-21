# Recommended Database Indexes

Indexes for the most frequent query patterns in the Basketball Performance AI.
Apply these on the Azure SQL Server instance to improve query performance.

---

## Anagrafiche

```sql
-- Fast player lookup by IdGlobal
CREATE INDEX IX_Player_IdGlobal
    ON [Anagrafiche].[Player] (IdGlobal);

-- Name search
CREATE INDEX IX_Player_Name
    ON [Anagrafiche].[Player] (LastName, FirstName);
```

## Boxscore

```sql
-- Player stats history (most frequent join)
-- Replace ITA1_2024_25 with each {LEAGUE}_{SEASON} partition name.
CREATE INDEX IX_Boxscore_IdPlayer
    ON [Boxscore].[ITA1_2024_25] (IdPlayer);

CREATE INDEX IX_Boxscore_IdGame
    ON [Boxscore].[ITA1_2024_25] (IdGame);
```

## Analisi

```sql
-- Advanced stats by player
CREATE INDEX IX_AdvStats_Player_Id
    ON [Analisi].[AdvancedStats_Player_ITA1_2024_25] (Id);

-- Lineup analytics by game
CREATE INDEX IX_Lineups_IdGame
    ON [Analisi].[AdvancedStats_Lineups_Quarter_ITA1_2024_25] (IdGame);

-- On/Off by player
CREATE INDEX IX_OnOff_IdPlayer
    ON [Analisi].[AdvancedStatsOnOffCourt_ITA1_2024_25] (IdPlayer);
```

## Pbp

```sql
-- PBP by player
CREATE INDEX IX_Pbp_IdPlayer
    ON [Pbp].[ITA1_2024_25] (IdPlayer);

-- PBP by game
CREATE INDEX IX_Pbp_IdGame
    ON [Pbp].[ITA1_2024_25] (IdGame);

-- Shot zone filtering (common for shot chart queries)
CREATE INDEX IX_Pbp_Zone
    ON [Pbp].[ITA1_2024_25] (side_area_zone, in_area);
```

---

## Notes

- Apply indexes after bulk loads (not before) to avoid write amplification.
- Use `FILLFACTOR=80` for tables with frequent inserts.
- Monitor index usage with `sys.dm_db_index_usage_stats`.
- The application uses `INFORMATION_SCHEMA.TABLES` (no custom indexes needed there).

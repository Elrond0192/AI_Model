# Data Card – Basketball Performance AI

**Version:** 1.0  
**Last Updated:** 2025-05-21  
**Maintainer:** Basketball AI Team  

---

## Overview

This Data Card describes the data sources, schemas, and known quality issues
for the Basketball Performance AI system. Data is read exclusively from the
following SQL Server schemas (no Lookup or Configuration tables are used):

| Schema        | Purpose                                      |
|---------------|----------------------------------------------|
| `Anagrafiche` | Player registry (demographics, contracts)    |
| `Analisi`     | Advanced stats (on/off, lineups, shot charts)|
| `Boxscore`    | Per-game box scores                          |
| `Pbp`         | Play-by-play event data                      |

---

## Schema: Anagrafiche

### Anagrafiche.*Player*
| Column          | Type          | Description                          | Coverage |
|-----------------|---------------|--------------------------------------|----------|
| `IdGlobal`      | INT           | Global player identifier             | ~100%    |
| `FirstName`     | nvarchar(200) | Given name                           | ~99%     |
| `LastName`      | nvarchar(200) | Family name                          | ~99%     |
| `BirthDate`     | nvarchar(500) | Date of birth (coerced to DATE)      | ~95%     |
| `Nationality`   | nvarchar(500) | Country of citizenship               | ~95%     |
| `Cm`            | nvarchar(500) | Height in cm (coerced to INT)        | ~90%     |
| `Weight`        | nvarchar(500) | Weight in kg (coerced to INT)        | ~88%     |
| `Role`          | nvarchar(500) | Playing role (PG/SG/SF/PF/C/hybrid)  | ~92%     |
| `ShirtNumber`   | nvarchar(500) | Jersey number (coerced to INT)       | ~85%     |

**Known issues:**
- `BirthDate`, `Cm`, `Weight`, `ShirtNumber` stored as `nvarchar(500)` — require coercion.
- ~5% of players missing nationality; defaults to empty string.

---

## Schema: Boxscore

### Boxscore.*{LEAGUE}_{SEASON}* (e.g., `Boxscore.ITA1_2024_25`)

Tables are partitioned by league code and season label.

| Column              | Type    | Description                          |
|---------------------|---------|--------------------------------------|
| `IdPlayer`          | INT     | FK → Anagrafiche.IdGlobal            |
| `IdGame`            | INT     | Game identifier                      |
| `IdTeam`            | INT     | Team identifier                      |
| `Pts`               | FLOAT   | Points scored                        |
| `Reb`               | FLOAT   | Total rebounds                       |
| `Ast`               | FLOAT   | Assists                              |
| `Stl`               | FLOAT   | Steals                               |
| `Blk`               | FLOAT   | Blocks                               |
| `Fg2m` / `Fg2a`     | FLOAT   | 2-point FG made / attempted          |
| `Fg3m` / `Fg3a`     | FLOAT   | 3-point FG made / attempted          |
| `Ftm` / `Fta`       | FLOAT   | Free throws made / attempted         |
| `Min`               | FLOAT   | Minutes played                       |
| `SF`                | INT     | Games started (starter flag)         |
| `PlusMinus`         | FLOAT   | +/- for the game                     |
| `x`, `y`            | FLOAT   | Shot coordinates (if PBP joined)     |

**Partitioning:** `{LEAGUE}{TIER}_{SEASON}` where `SEASON` uses `YYYY_YY` format.  
**Supported leagues:** see `SUPPORTED_LEAGUES` in `basketball_ai/constants.py`.  
**Supported seasons:** see `SUPPORTED_SEASONS` in `basketball_ai/constants.py`.

---

## Schema: Analisi

### Analisi.AdvancedStats_Player_*
Advanced per-player statistics.

| Column         | Type  | Description                           |
|----------------|-------|---------------------------------------|
| `Id`           | INT   | FK → Anagrafiche.IdGlobal             |
| `BPM`          | FLOAT | Box Plus/Minus                        |
| `OBPM`         | FLOAT | Offensive BPM                         |
| `DBPM`         | FLOAT | Defensive BPM                         |
| `VORP`         | FLOAT | Value Over Replacement Player         |
| `WS`           | FLOAT | Win Shares                            |
| `OWS` / `DWS`  | FLOAT | Offensive / Defensive Win Shares      |
| `PER`          | FLOAT | Player Efficiency Rating              |
| `TS_pct`       | FLOAT | True Shooting Percentage              |
| `USG_pct`      | FLOAT | Usage Rate                            |
| `RAPTOR_Off`   | FLOAT | RAPTOR offensive rating               |
| `RAPTOR_Def`   | FLOAT | RAPTOR defensive rating               |
| `LEBRON_Off`   | FLOAT | LEBRON offensive component            |
| `LEBRON_Def`   | FLOAT | LEBRON defensive component            |
| `SPM`          | FLOAT | Statistical Plus/Minus                |
| `GmSc`         | FLOAT | Game Score                            |
| `FIC`          | FLOAT | Floor Impact Counter                  |

### Analisi.AdvancedStats_Lineups_Quarter_*
Five-player lineup analytics.

| Column         | Type         | Description                     |
|----------------|--------------|---------------------------------|
| `HomeLineup`   | nvarchar(500)| Comma-separated player IDs      |
| `AwayLineup`   | nvarchar(500)| Comma-separated player IDs      |
| `NetRtg`       | FLOAT        | Net rating for this lineup      |
| `ORtg`         | FLOAT        | Offensive rating                |
| `DRtg`         | FLOAT        | Defensive rating                |
| `Possessions`  | INT          | Possessions played              |

### Analisi.AdvancedStatsOnOffCourt_*
On/off differential analytics.

| Column         | Type  | Description                            |
|----------------|-------|----------------------------------------|
| `IdPlayer`     | INT   | FK → Anagrafiche.IdGlobal              |
| `NetRtg_On`    | FLOAT | Net rating when player is on court     |
| `NetRtg_Off`   | FLOAT | Net rating when player is off court    |
| `NetRtg_Diff`  | FLOAT | On − Off differential                  |
| `ORtg_Diff`    | FLOAT | Offensive rating differential          |
| `DRtg_Diff`    | FLOAT | Defensive rating differential          |

---

## Schema: Pbp

### Pbp.*{LEAGUE}_{SEASON}*
Play-by-play events.

| Column                         | Type          | Description                       |
|--------------------------------|---------------|-----------------------------------|
| `IdGame`                       | INT           | Game identifier                   |
| `IdPlayer`                     | INT           | FK → Anagrafiche.IdGlobal         |
| `x`, `y`                       | FLOAT         | Court coordinates (0–100 scale)   |
| `side_area_zone`               | nvarchar(50)  | Zone label (paint/midrange/3pt)   |
| `side_area_code`               | nvarchar(20)  | Zone code                         |
| `in_area`                      | BIT           | Shot attempted in restricted area |
| `dunk`                         | BIT           | Dunk flag                         |
| `action_1_qualifier_description` | nvarchar(200)| Primary action qualifier          |
| `action_2_qualifier_description` | nvarchar(200)| Secondary action qualifier        |
| `QuarterTime`                  | FLOAT         | Seconds elapsed in quarter        |
| `GameTime`                     | FLOAT         | Total elapsed game seconds        |

---

## Data Quality

| Check                              | Target  | Current status |
|------------------------------------|---------|----------------|
| NULL rate on `IdGlobal`            | < 1%    | ✅ < 0.1%       |
| NULL rate on `Pts`, `Reb`, `Ast`   | < 5%    | ✅ ~1%          |
| Fg2m ≤ Fg2a consistency            | 100%    | ✅              |
| Rating in [0, 10]                  | 100%    | ✅              |
| Pts ≈ 2×Fg2m + 3×Fg3m + Ftm      | ±5 pts  | ✅ ~99%         |

Run `python main.py --mode validate-data` to generate a fresh quality report.

---

## Privacy and Retention

- `BirthDate`, `Cm`, `Weight` from `Anagrafiche.*` are treated as PII.
- These fields are masked in API responses for roles other than `admin` and `analyst`.
- See `basketball_ai/api/middleware/pii.py` for masking implementation.
- Data is retained for the duration of the service operation. No automatic deletion is currently implemented.

---

## Lineage

| Source              | Ingestion method                           |
|---------------------|--------------------------------------------|
| SQL Server (live)   | `python main.py --mode api --source sql`   |
| CSV (offline)       | `python main.py --mode generate-data`      |
| Ingestion state     | `data/ingestion.db` (SQLite, idempotent)   |

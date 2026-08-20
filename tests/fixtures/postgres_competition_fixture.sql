CREATE SCHEMA IF NOT EXISTS "Anagrafiche";
CREATE SCHEMA IF NOT EXISTS "Analisi";
CREATE SCHEMA IF NOT EXISTS "Boxscore";

CREATE TABLE IF NOT EXISTS "Anagrafiche"."ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "IdGlobal" text NOT NULL,
  "PlayerName" text NOT NULL,
  "BirthDate" date,
  "Pos" text,
  "TeamName" text
);

CREATE TABLE IF NOT EXISTS "Anagrafiche"."Team_ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "IdGlobal" text NOT NULL,
  "TeamName" text NOT NULL,
  "ShortName" text
);

CREATE TABLE IF NOT EXISTS "Analisi"."AdvancedStats_Player_ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "TeamId" text,
  "Competition" text NOT NULL,
  "Games" double precision,
  "Min" double precision,
  "Pts" double precision,
  "TR" double precision,
  "Ast" double precision,
  "Stl" double precision,
  "Blk" double precision,
  "TO" double precision,
  "PF" double precision,
  "TsPct" double precision,
  "UsgPct" double precision,
  "OBpm" double precision,
  "DBpm" double precision,
  "ThreePAr" double precision,
  "ValLegaPerGame" double precision
);

CREATE TABLE IF NOT EXISTS "Analisi"."AdvancedStatsTeam_ITA1" (
  "Season" integer NOT NULL,
  "TeamId" text NOT NULL,
  "Competition" text NOT NULL,
  "Games" double precision,
  "Pace" double precision,
  "ORtg" double precision,
  "DRtg" double precision,
  "NetRtg" double precision,
  "ThreePAr" double precision,
  "AstPerGame" double precision
);

CREATE TABLE IF NOT EXISTS "Boxscore"."ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "TeamId" text,
  "Game" text,
  "SF" boolean,
  "Competition" text NOT NULL
);

TRUNCATE TABLE
  "Anagrafiche"."ITA1",
  "Anagrafiche"."Team_ITA1",
  "Analisi"."AdvancedStats_Player_ITA1",
  "Analisi"."AdvancedStatsTeam_ITA1",
  "Boxscore"."ITA1";

INSERT INTO "Anagrafiche"."ITA1"
  ("Season", "Id", "IdGlobal", "PlayerName", "BirthDate", "Pos", "TeamName")
VALUES
  (2024, 'P_LOCAL', 'P_GLOBAL', 'Test Player', DATE '2000-01-01', 'PG', 'Test Team'),
  (2025, 'P_LOCAL', 'P_GLOBAL', 'Test Player', DATE '2000-01-01', 'PG', 'Test Team');

INSERT INTO "Anagrafiche"."Team_ITA1"
  ("Season", "Id", "IdGlobal", "TeamName", "ShortName")
VALUES
  (2024, 'T_LOCAL', 'T_GLOBAL', 'Test Team', 'TST'),
  (2025, 'T_LOCAL', 'T_GLOBAL', 'Test Team', 'TST');

INSERT INTO "Analisi"."AdvancedStats_Player_ITA1"
  ("Season", "Id", "TeamId", "Competition", "Games", "Min", "Pts", "TR", "Ast", "Stl", "Blk", "TO", "PF", "TsPct", "UsgPct", "OBpm", "DBpm", "ThreePAr", "ValLegaPerGame")
VALUES
  (2024, 'P_LOCAL', 'T_LOCAL', 'Regular Season', 30, 750, 360, 120, 150, 30, 5, 60, 50, 0.580, 22.0, 2.0, 0.5, 0.35, 6.50),
  (2024, 'P_LOCAL', 'T_LOCAL', 'Playoffs',        8, 224, 112,  36,  48, 10, 2, 18, 16, 0.610, 24.0, 3.0, 0.8, 0.38, 7.00),
  (2025, 'P_LOCAL', 'T_LOCAL', 'RS',             30, 780, 390, 126, 156, 32, 6, 58, 48, 0.590, 22.5, 2.4, 0.6, 0.36, 6.70),
  (2025, 'P_LOCAL', 'T_LOCAL', 'PO',              9, 261, 135,  40,  54, 11, 2, 17, 15, 0.620, 25.0, 3.3, 0.9, 0.40, 7.20);

INSERT INTO "Analisi"."AdvancedStatsTeam_ITA1"
  ("Season", "TeamId", "Competition", "Games", "Pace", "ORtg", "DRtg", "NetRtg", "ThreePAr", "AstPerGame")
VALUES
  (2024, 'T_LOCAL', 'Regular Season', 30, 74.0, 112.0, 108.0, 4.0, 0.34, 20.0),
  (2024, 'T_LOCAL', 'Playoffs',        8, 71.0, 115.0, 106.0, 9.0, 0.39, 22.0),
  (2025, 'T_LOCAL', 'RS',             30, 75.0, 113.0, 107.0, 6.0, 0.35, 21.0),
  (2025, 'T_LOCAL', 'PO',              9, 70.0, 116.0, 105.0, 11.0, 0.41, 23.0);

INSERT INTO "Boxscore"."ITA1"
  ("Season", "Id", "TeamId", "Game", "SF", "Competition")
VALUES
  (2024, 'P_LOCAL', 'T_LOCAL', 'RS-1', true, 'RS'),
  (2024, 'P_LOCAL', 'T_LOCAL', 'PO-1', true, 'Playoffs'),
  (2025, 'P_LOCAL', 'T_LOCAL', 'RS-2', true, 'Regular Season'),
  (2025, 'P_LOCAL', 'T_LOCAL', 'PO-2', true, 'PO');

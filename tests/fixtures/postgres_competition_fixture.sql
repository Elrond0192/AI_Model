CREATE SCHEMA IF NOT EXISTS "Anagrafiche";
CREATE SCHEMA IF NOT EXISTS "Analisi";
CREATE SCHEMA IF NOT EXISTS "Boxscore";
CREATE SCHEMA IF NOT EXISTS "Pbp";

CREATE TABLE IF NOT EXISTS "Anagrafiche"."ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "IdGlobal" text NOT NULL,
  "PlayerName" text NOT NULL,
  "Team" text,
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

CREATE TABLE IF NOT EXISTS "Analisi"."AdvancedStats_Clutch_ITA1" (
  "Season" integer NOT NULL,
  "Id" text NOT NULL,
  "TeamId" text,
  "Competition" text NOT NULL,
  "ClutchGames" integer,
  "Pts" double precision,
  "Fg2m" double precision,
  "Fg2a" double precision,
  "Fg3m" double precision,
  "Fg3a" double precision,
  "Ftm" double precision,
  "Fta" double precision,
  "Ast" double precision,
  "Orb" double precision,
  "Drb" double precision,
  "Stl" double precision,
  "Blk" double precision,
  "Tov" double precision,
  "FgPct" double precision,
  "Fg2Pct" double precision,
  "Fg3Pct" double precision,
  "FtPct" double precision,
  "EfgPct" double precision,
  "TsPct" double precision,
  "Ftr" double precision,
  "ThreePAr" double precision,
  "AstToTovRatio" double precision,
  "ORtg" double precision,
  "DRtg" double precision,
  "NetRtg" double precision,
  "ClutchnessRating" text
);

CREATE TABLE IF NOT EXISTS "Pbp"."ITA1_2025" (
  "Season" integer NOT NULL,
  "Competition" text NOT NULL,
  "GameCode" text NOT NULL,
  "Player" text,
  "TeamId" text,
  "HomeClub" integer,
  "QuarterTime" text,
  "Score" text,
  "Quarter" integer,
  "PlayType" text,
  "AttemptAnd1" integer,
  "HomeLineup" text,
  "AwayLineup" text
);

TRUNCATE TABLE
  "Anagrafiche"."ITA1",
  "Anagrafiche"."Team_ITA1",
  "Analisi"."AdvancedStats_Player_ITA1",
  "Analisi"."AdvancedStatsTeam_ITA1",
  "Analisi"."AdvancedStats_Clutch_ITA1",
  "Boxscore"."ITA1",
  "Pbp"."ITA1_2025";

INSERT INTO "Anagrafiche"."ITA1"
  ("Season", "Id", "IdGlobal", "PlayerName", "Team", "BirthDate", "Pos", "TeamName")
VALUES
  (2024, 'P_LOCAL', 'P_GLOBAL', 'Test Player', '1001', DATE '2000-01-01', 'PG', 'Test Team'),
  (2025, 'P_LOCAL', 'P_GLOBAL', 'Test Player', '1001', DATE '2000-01-01', 'PG', 'Test Team'),
  (2025, 'P2_LOCAL', 'P2_GLOBAL', 'Test Player 2', NULL, DATE '2000-02-01', 'SG', NULL),
  (2025, 'P3_LOCAL', 'P3_GLOBAL', 'Test Player 3', NULL, DATE '2000-03-01', 'SF', NULL),
  (2025, 'P4_LOCAL', 'P4_GLOBAL', 'Test Player 4', NULL, DATE '2000-04-01', 'PF', NULL),
  (2025, 'P5_LOCAL', 'P5_GLOBAL', 'Test Player 5', NULL, DATE '2000-05-01', 'C', NULL),
  (2025, 'P6_LOCAL', 'P6_GLOBAL', 'Test Player 6', NULL, DATE '2000-06-01', 'PG', NULL),
  (2025, 'P7_LOCAL', 'P7_GLOBAL', 'Test Player 7', NULL, DATE '2000-07-01', 'SG', NULL),
  (2025, 'P8_LOCAL', 'P8_GLOBAL', 'Test Player 8', NULL, DATE '2000-08-01', 'SF', NULL),
  (2025, 'P9_LOCAL', 'P9_GLOBAL', 'Test Player 9', NULL, DATE '2000-09-01', 'PF', NULL),
  (2025, 'P10_LOCAL', 'P10_GLOBAL', 'Test Player 10', NULL, DATE '2000-10-01', 'C', NULL);

INSERT INTO "Anagrafiche"."Team_ITA1"
  ("Season", "Id", "IdGlobal", "TeamName", "ShortName")
VALUES
  (2024, '1001', 'T_GLOBAL', 'Test Team', 'TST'),
  (2025, '1001', 'T_GLOBAL', 'Test Team', 'TST'),
  (2025, '1002', 'T_GLOBAL2', 'Away Test Team', 'AWY');

INSERT INTO "Analisi"."AdvancedStats_Player_ITA1"
  ("Season", "Id", "TeamId", "Competition", "Games", "Min", "Pts", "TR", "Ast", "Stl", "Blk", "TO", "PF", "TsPct", "UsgPct", "OBpm", "DBpm", "ThreePAr", "ValLegaPerGame")
VALUES
  (2024, 'P_LOCAL', '1001.0', 'Regular Season', 30, 750, 360, 120, 150, 30, 5, 60, 50, 0.580, 22.0, 2.0, 0.5, 0.35, 6.50),
  (2024, 'P_LOCAL', '1001.0', 'Playoffs',        8, 224, 112,  36,  48, 10, 2, 18, 16, 0.610, 24.0, 3.0, 0.8, 0.38, 7.00),
  (2025, 'P_LOCAL', NULL,       'RS',             30, 780, 390, 126, 156, 32, 6, 58, 48, 0.590, 22.5, 2.4, 0.6, 0.36, 6.70),
  (2025, 'P_LOCAL', '1001.0', 'PO',              9, 261, 135,  40,  54, 11, 2, 17, 15, 0.620, 25.0, 3.3, 0.9, 0.40, 7.20);

INSERT INTO "Analisi"."AdvancedStatsTeam_ITA1"
  ("Season", "TeamId", "Competition", "Games", "Pace", "ORtg", "DRtg", "NetRtg", "ThreePAr", "AstPerGame")
VALUES
  (2024, '1001.0', 'Regular Season', 30, 74.0, 112.0, 108.0, 4.0, 0.34, 20.0),
  (2024, '1001.0', 'Playoffs',        8, 71.0, 115.0, 106.0, 9.0, 0.39, 22.0),
  (2025, '1001.0', 'RS',             30, 75.0, 113.0, 107.0, 6.0, 0.35, 21.0),
  (2025, '1001.0', 'PO',              9, 70.0, 116.0, 105.0, 11.0, 0.41, 23.0);

INSERT INTO "Boxscore"."ITA1"
  ("Season", "Id", "TeamId", "Game", "SF", "Competition")
VALUES
  (2024, 'P_LOCAL', '1001.0', 'RS-1', true, 'RS'),
  (2024, 'P_LOCAL', '1001.0', 'PO-1', true, 'Playoffs'),
  (2025, 'P_LOCAL', '1001.0', 'RS-2', true, 'Regular Season'),
  (2025, 'P_LOCAL', '1001.0', 'PO-2', true, 'PO');


INSERT INTO "Analisi"."AdvancedStats_Clutch_ITA1"
  ("Season", "Id", "TeamId", "Competition", "ClutchGames", "Pts",
   "Fg2m", "Fg2a", "Fg3m", "Fg3a", "Ftm", "Fta", "Ast", "Orb", "Drb",
   "Stl", "Blk", "Tov", "FgPct", "Fg2Pct", "Fg3Pct", "FtPct", "EfgPct",
   "TsPct", "Ftr", "ThreePAr", "AstToTovRatio", "ORtg", "DRtg", "NetRtg",
   "ClutchnessRating")
VALUES
  (2025, 'P_LOCAL', '1001', 'RS', 6, 12, 3, 4, 1, 4, 2, 2, 2, 1, 3,
   1, 1, 0, 0.5, 0.75, 0.25, 1.0, 0.5625, 0.667, 0.25, 0.5, NULL, 115.0, 103.0, 12.0, 'B'),
  (2025, 'P_LOCAL', '1001', 'PO', 3, 4, 1, 2, 0, 1, 2, 2, 1, 0, 1,
   0, 0, 1, 0.333, 0.5, 0.0, 1.0, 0.333, 0.444, 0.667, 0.333, 1.0, 100.0, 112.0, -12.0, 'C');

INSERT INTO "Pbp"."ITA1_2025"
  ("Season", "Competition", "GameCode", "Player", "TeamId", "HomeClub",
   "QuarterTime", "Score", "Quarter", "PlayType", "AttemptAnd1", "HomeLineup", "AwayLineup")
VALUES
  (2025, 'RS', 'ITA1-RS-TEST-1', 'P_LOCAL', '1001', 1, '04:20', '2-0', 4, '2FGM', 0,
   'P_LOCAL - P2_LOCAL - P3_LOCAL - P4_LOCAL - P5_LOCAL',
   'P6_LOCAL - P7_LOCAL - P8_LOCAL - P9_LOCAL - P10_LOCAL'),
  (2025, 'RS', 'ITA1-RS-TEST-1', 'P6_LOCAL', '1002', 0, '03:58', '2-0', 4, '3FGA', 0,
   'P_LOCAL - P2_LOCAL - P3_LOCAL - P4_LOCAL - P5_LOCAL',
   'P6_LOCAL - P7_LOCAL - P8_LOCAL - P9_LOCAL - P10_LOCAL'),
  (2025, 'RS', 'ITA1-RS-TEST-1', 'P7_LOCAL', '1002', 0, '03:43', '2-0', 4, 'TOV', 0,
   'P_LOCAL - P2_LOCAL - P3_LOCAL - P4_LOCAL - P5_LOCAL',
   'P6_LOCAL - P7_LOCAL - P8_LOCAL - P9_LOCAL - P10_LOCAL');

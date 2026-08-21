CREATE SCHEMA IF NOT EXISTS "AI";
CREATE TABLE IF NOT EXISTS "AI"."ModelRuns" (
  model_run_id text PRIMARY KEY, model_version text NOT NULL, feature_version text NOT NULL,
  data_cutoff date NOT NULL, trained_at timestamptz NOT NULL DEFAULT now(), status text NOT NULL,
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS "AI"."PlayerForecasts" (
  model_run_id text NOT NULL REFERENCES "AI"."ModelRuns"(model_run_id), player_global_id text NOT NULL,
  team_global_id text NOT NULL DEFAULT '', league text NOT NULL, season integer NOT NULL,
  competition text NOT NULL, target_season integer NOT NULL, predicted_rating double precision NOT NULL,
  confidence_low double precision NOT NULL, confidence_high double precision NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_run_id, player_global_id, team_global_id, league, season, competition)
);
CREATE INDEX IF NOT EXISTS player_forecasts_lookup ON "AI"."PlayerForecasts"(player_global_id, league, season, competition);

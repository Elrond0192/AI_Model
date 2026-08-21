-- One-time migration for installations that used unquoted ai / ai_source.
-- AI_Source contains rebuildable adapters/serving data, so it is recreated.
-- AI contains model run history and is renamed in place when possible.

BEGIN;

DROP SCHEMA IF EXISTS ai_source CASCADE;
DROP SCHEMA IF EXISTS "AI_Source" CASCADE;
CREATE SCHEMA "AI_Source";

DO $$
BEGIN
    IF to_regnamespace('ai') IS NOT NULL AND to_regnamespace('"AI"') IS NULL THEN
        ALTER SCHEMA ai RENAME TO "AI";
    ELSIF to_regnamespace('ai') IS NOT NULL AND to_regnamespace('"AI"') IS NOT NULL THEN
        RAISE EXCEPTION 'Both ai and AI schemas exist; merge model history before migration';
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('"AI".model_runs') IS NOT NULL
       AND to_regclass('"AI"."ModelRuns"') IS NULL THEN
        ALTER TABLE "AI".model_runs RENAME TO "ModelRuns";
    END IF;
    IF to_regclass('"AI".player_forecasts') IS NOT NULL
       AND to_regclass('"AI"."PlayerForecasts"') IS NULL THEN
        ALTER TABLE "AI".player_forecasts RENAME TO "PlayerForecasts";
    END IF;
END;
$$;

COMMIT;

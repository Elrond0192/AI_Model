-- PostgreSQL parallel-safety correction for "AI_Source" parsing helpers.
--
-- _num() and _date() contain EXCEPTION handlers. PostgreSQL implements those
-- handlers with subtransactions, which cannot execute inside parallel workers.
-- _int() calls _num(), so it must inherit the same restriction.
--
-- Run this immediately after ai_source_schema.sql. It is idempotent.

BEGIN;

ALTER FUNCTION "AI_Source"."NumericValue"(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION "AI_Source"."IntValue"(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION "AI_Source"."DateValue"(jsonb, text[]) PARALLEL UNSAFE;

COMMIT;

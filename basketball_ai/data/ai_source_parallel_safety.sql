-- PostgreSQL parallel-safety correction for ai_source parsing helpers.
--
-- _num() and _date() contain EXCEPTION handlers. PostgreSQL implements those
-- handlers with subtransactions, which cannot execute inside parallel workers.
-- _int() calls _num(), so it must inherit the same restriction.
--
-- Run this immediately after ai_source_schema.sql. It is idempotent.

BEGIN;

ALTER FUNCTION ai_source._num(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION ai_source._int(jsonb, text[]) PARALLEL UNSAFE;
ALTER FUNCTION ai_source._date(jsonb, text[]) PARALLEL UNSAFE;

COMMIT;

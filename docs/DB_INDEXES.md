# PostgreSQL indexes

The production output indexes are created by
`basketball_ai/data/ai_schema.sql`. Canonical source views should be backed by
indexes on player/team global ID, numeric season, league and competition.

The AI read role receives `SELECT` on `ai_source`; the writer role receives
`SELECT`, `INSERT` and `UPDATE` only on schema `ai`.

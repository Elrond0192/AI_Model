# ADR 0003 — SQLite for local API operational state

**Status:** Accepted

PostgreSQL is the basketball data/model store. Small single-node API concerns
that do not belong in BBallstat remain local to the container runtime:

- request audit log;
- JWT revocation state;
- tenant usage counters.

These SQLite files live under ignored runtime volumes and are never committed.

The retired game-ingestion tracker is not part of AI_Model anymore. Source data
is loaded directly from the canonical PostgreSQL `ai_source` views.

If the API is scaled to multiple replicas, move this operational state to a
shared store before enabling horizontal writes.

# ADR 0003 – SQLite for local operational state (no external broker)

**Status:** Accepted  
**Date:** 2025-05-21  

## Context

Several operational concerns require persisted state:
- Ingestion idempotency (game processing status).
- Audit logging.
- Token revocation.
- Tenant usage metering.

Options considered: PostgreSQL (external), Redis (external), SQLite (embedded).

## Decision

Use SQLite for all local operational state:
- `data/ingestion.db` – ingestion state tracker.
- `audit.db` – request audit log.
- `data/revoked_tokens.db` – JWT revocation blocklist.
- `audit.db` (shared) – tenant usage counters.

## Consequences

- **Good:** Zero external dependencies; works in Docker single-container deployments.
- **Good:** Atomic writes via WAL mode; concurrent readers supported.
- **Bad:** Not horizontally scalable (single-writer). Acceptable for current deployment model.
- **Migration path:** Replace `AuditDB`, `IngestionTracker`, `TenantManager` with
  PostgreSQL adapters when horizontal scaling is required.

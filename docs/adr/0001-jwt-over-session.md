# ADR 0001 – JWT over session-based authentication

**Status:** Accepted  
**Date:** 2025-05-21  
**Deciders:** Basketball AI Team  

## Context

The API needs to authenticate requests from multiple clients (Streamlit GUI,
REST API consumers, the chat widget). Session-based auth requires sticky sessions
or a shared session store, which adds infrastructure complexity.

## Decision

Use JWT (JSON Web Tokens) as the primary authentication scheme.
- Access token: 30-minute lifetime (configurable via `JWT_ACCESS_EXPIRE`).
- Refresh token: 24-hour lifetime (configurable via `JWT_REFRESH_EXPIRE`).
- Signing: HS256 with `JWT_SECRET` / `JWT_SECRETS` (key rotation supported via `kid` header).
- Token revocation: SQLite-backed `jti` blocklist at `data/revoked_tokens.db`.

Static `API_KEY` is kept as a deprecated legacy fallback (logs a warning).

## Consequences

- **Good:** Stateless, horizontally scalable, supports multi-tenant claims.
- **Bad:** Token revocation requires blocklist (accepted cost for statelessness benefit).
- **Good:** Key rotation supported without downtime via multiple secrets.

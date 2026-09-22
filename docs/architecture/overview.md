# Architecture

```text
PostgreSQL host
  ├─ raw PBP / boxscore (source of truth, never loaded by the API)
  ├─ "AI_Source".* six canonical physical model tables
  ├─ "AI_Source"."Training*" physical feature tables
  ├─ "AI_Source"."Scenario*" physical indexed serving tables
  └─ "AI".* versioned prediction snapshots
          ▲
          │ psycopg / SQLAlchemy via host.docker.internal
AI_Model Docker
  ├─ immutable training snapshots (joblib + SHA-256 manifest)
  ├─ authenticated operations console :8501
  └─ typed inference API :8000
          ▲
          │ authenticated server-to-server HTTPS
WordPress Chat V3 / Bax
```

The operations console can store multiple PostgreSQL profiles and requires an
explicit active profile. Chat and entity resolution are owned by WordPress;
AI_Model has no mounted conversational endpoint.

Scenario serving tables are refreshed incrementally after ETL with
`CALL "AI_Source"."RefreshContext"(league_key, season, competition)`.
Chat V3 reads only these bounded aggregates. Raw possession rows are reserved
for ETL/rebuild operations and are never materialized in API memory.

Training uses an immutable snapshot selected by ID. Re-running backtests or
training from that ID does not query PostgreSQL again and records the snapshot
ID/checksum in model metadata.

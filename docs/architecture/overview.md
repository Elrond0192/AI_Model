# Architecture

```text
PostgreSQL host
  ├─ ai_source.* canonical read-only views
  └─ ai.* versioned prediction snapshots
          ▲
          │ psycopg / SQLAlchemy via host.docker.internal
AI_Model Docker
  ├─ authenticated operations console :8501
  └─ typed inference API :8000
          ▲
          │ authenticated server-to-server HTTPS
WordPress Chat V3 / Bax
```

The operations console can store multiple PostgreSQL profiles and requires an
explicit active profile. Chat and entity resolution are owned by WordPress;
AI_Model has no mounted conversational endpoint.

# Architecture overview

## Runtime boundaries

```text
PostgreSQL host
  ├─ HoopmetricsEngine-owned source schemas (read-only to AI_Model)
  ├─ "AI_Source"."Leagues", "Teams", "Players", "TeamPlayerRelations"
  ├─ "AI_Source"."PlayerCompetitionStats", "TeamCompetitionStats"
  ├─ "AI_Source"."Training*" and "AI_Source"."Scenario*" serving tables
  └─ "AI"."ModelRuns", "AI"."PlayerForecasts" and other model-owned outputs
          ▲
          │ psycopg / SQLAlchemy via host.docker.internal
AI_Model Docker Compose
  ├─ admin :8501 — authenticated operations console
  └─ api   :8000 — typed FastAPI inference and health endpoints
          ▲
          │ authenticated server-to-server HTTPS
WordPress Chat V3 / Bax
```

WordPress owns conversational orchestration and public entity resolution. AI_Model is an inference/analytics service, not a chatbot. It does not own or recompute HoopmetricsEngine AdvancedStats. It consumes canonical source contracts and keeps its forecasting, BB-Rating, Metric Rating, Future Performance, Player Intelligence and simulation/scenario responsibilities separated.

## Database contracts and refresh

Install database contracts in order:

1. `basketball_ai/data/ai_source_full.sql` — canonical entity and competition-preserving statistical source.
2. `basketball_ai/data/ai_scenario_serving.sql` — indexed training/scenario serving tables and refresh procedures.
3. `basketball_ai/data/ai_schema.sql` — model-owned prediction storage.
4. `basketball_ai/data/ai_metric_schema.sql` — Metric Rating Engine tables (`"AI"."MetricDefinition"`, `"AI"."MetricDistribution"`, `"AI"."MetricRating"`) when that feature is deployed.

After the upstream ETL has committed canonical AI_Source statistics for a season,
refresh all discovered league/competition contexts for that season through the
single public entry point:

```sql
CALL "AI_Source"."RefreshContextSeasonal"(2026);
```

There is no public single-context `RefreshContext` overload.

The refresh procedure operates on the current canonical AI_Source tables; it does not itself rebuild those tables from HoopmetricsEngine source schemas. Follow the ETL/source-contract rebuild steps in [README](../../README.md) and [RUNBOOK](../../RUNBOOK.md).

## Model lifecycle

Training consumes an immutable snapshot and creates a candidate run. Production changes only after validation and explicit promotion. Model Center promotions/rollbacks request an authenticated API restart and wait for `/health/ready`; equivalent CLI artifact changes require a manual API restart. Candidate training and calibration do not restart production.

Prediction Model, Prediction 1–100 presentation calibration, BB-Rating calibration and Future Performance are separate artifact/lifecycle concerns. Do not infer that promoting one artifact updates the others.

## Operational health

- `GET /health/live` — process liveness.
- `GET /health/ready` — readiness of required data/model dependencies.
- `GET /healthz` — admin console health.
- `POST /internal/restart` — internal, API-key-authenticated controlled restart for the Model Center; never expose as a public browser action.

See [API v2](../API_V2.md), [API integration](../API_INTEGRATION.md) and [production runbook](../../RUNBOOK.md) for request and recovery details.

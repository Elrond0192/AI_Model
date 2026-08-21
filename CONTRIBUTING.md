# Contributing to AI_Model

## Prerequisites

- Python 3.11 or 3.12
- PostgreSQL for integration testing when changing the data contract
- Docker / Docker Compose for production-like smoke tests

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Optional hooks:

```bash
pre-commit install
```

## Workflow

1. Create a feature branch from `main`.
2. Keep production data access behind `ai_source` canonical views.
3. Add tests for model/data/API contract changes.
4. Run:

```bash
pytest tests/ -q --tb=short
ruff check basketball_ai/ --select E,F,W --ignore E501
```

5. Build the Docker image before deployment-sensitive changes:

```bash
docker build -t basketball-ai:test .
```

## Project boundaries

| Area | Responsibility |
|---|---|
| `basketball_ai/data/postgres_loader.py` | load and validate the canonical PostgreSQL contract |
| `basketball_ai/data/ai_source_schema.sql` | adapt BBallstat physical tables to `ai_source.*` |
| `basketball_ai/data/ai_schema.sql` | model-owned PostgreSQL output tables |
| `basketball_ai/features/` | feature engineering |
| `basketball_ai/models/` | forecast, compatibility, calibration, registry |
| `basketball_ai/scenarios/` | bounded what-if/team context logic |
| `basketball_ai/api/` | authenticated typed inference |
| `docker-compose.yml` (`ops`) | one-shot technical operations CLI |

Conversation, natural-language intent routing and public entity resolution belong
in WordPress Chat V3/Bax, not AI_Model.

## Rules

- Never commit `.env`, database profiles, API keys, trained model artefacts,
  sessions or audit databases.
- No SQL Server/Azure SQL compatibility code in the production data layer.
- No generic SQL endpoint.
- Source BBallstat schemas are read-only to AI_Model; writes belong only in
  schema `ai`.
- New league tables should be picked up by `ai_source_schema.sql` without
  Python changes.
- Preserve season chronology: a player must not have multiple training rows for
  the same target season.
- Use type hints on public Python APIs and structured logging in library code.
- Do not weaken failed backtests/readiness into silent fallbacks.

## PostgreSQL contract changes

After modifying `ai_source_schema.sql`, run it against a representative database
and verify:

```sql
SELECT count(*) FROM ai_source.players;
SELECT count(*) FROM ai_source.teams;
SELECT min(season), max(season), count(*) FROM ai_source.player_stats;
SELECT player_id, season, count(*)
FROM ai_source.player_stats
GROUP BY player_id, season
HAVING count(*) > 1;
```

The final query must return zero rows.

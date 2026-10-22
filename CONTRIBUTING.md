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
2. Keep production data access behind `"AI_Source"` canonical views.
3. Add tests for model/data/API contract changes.
4. Update the relevant user/operator documentation and `CHANGELOG.md` in the same
   release change. Scheduler behavior belongs in `docs/SCHEDULER.md`; database
   contract changes belong in `DATA_CARD.md` and the relevant runbook/SQL notes.
5. Run:

```bash
pytest tests/ -q --tb=short
ruff check basketball_ai/ --select E,F,W --ignore E501
python .github/scripts/check_release_docs.py
```

6. Build the Docker image before deployment-sensitive changes:

```bash
docker build -t basketball-ai:test .
```

7. For pull requests that change runtime code, API/UX behavior or SQL/data
   contracts, the documentation gate requires at least one relevant documentation
   or changelog file in the PR diff. A code change is not release-ready merely
   because its tests pass.

## Project boundaries

| Area | Responsibility |
|---|---|
| `basketball_ai/data/postgres_loader.py` | load and validate the canonical PostgreSQL contract |
| `basketball_ai/data/ai_source_full.sql` | adapt BBallstat physical tables to `"AI_Source".*` |
| `basketball_ai/data/ai_schema.sql` | model-owned PostgreSQL output tables |
| `basketball_ai/features/` | feature engineering |
| `basketball_ai/models/` | forecast, compatibility, calibration, registry |
| `basketball_ai/scenarios/` | bounded what-if/team context logic |
| `basketball_ai/api/` | authenticated typed inference |
| `basketball_ai/admin_web/app.py` | technical operations console |

Conversation, natural-language intent routing and public entity resolution belong
in WordPress Chat V3/Bax, not AI_Model.

## Rules

- Never commit `.env`, database profiles, API keys, trained model artefacts,
  sessions or audit databases.
- No SQL Server/Azure SQL compatibility code in the production data layer.
- No generic SQL endpoint.
- Canonical BBallstat facts and statistics remain read-only to AI_Model. The only
  runtime write exception inside `AI_Source` is scheduler metadata in
  `AIModelScheduledJob` and `AIModelJobRun`, granted narrowly by
  `ai_scenario_serving.sql`; model outputs belong in schema `AI`.
- New league tables should be picked up by `ai_source_full.sql` without
  Python changes.
- Preserve season chronology: a player must not have multiple training rows for
  the same target season.
- Use type hints on public Python APIs and structured logging in library code.
- Do not weaken failed backtests/readiness into silent fallbacks.

## Documentation and release policy

Documentation is part of each release, not a cleanup task for later.

- Update `README.md` for capability/architecture changes and `RUNBOOK.md` for
  operational, deployment, recovery and scheduling changes.
- Update the focused document under `docs/` when a model, API contract, scheduler
  task or scenario changes. Keep `roadmap_intelligence.md` and the causal
  attribution roadmap aligned with what is implemented versus what is validated.
- Add a `CHANGELOG.md` entry under `[Unreleased]` for user-visible behavior,
  bug fixes, data-contract/schema changes and operational procedure changes.
- Never mark a capability as fully available only because code exists: explicitly
  state validation gates, partial coverage, deployment/database prerequisites and
  any manual steps still required.
- Do not leave old behavior statements next to the new contract. Replace stale
  claims in the same change; the documentation contract checker rejects known
  contradictions and the PR documentation gate rejects implementation changes
  without a docs/changelog update.

## PostgreSQL contract changes

After modifying `ai_source_full.sql`, run it against a representative database
and verify:

```sql
SELECT count(*) FROM "AI_Source"."Players";
SELECT count(*) FROM "AI_Source"."Teams";
SELECT min(season), max(season), count(*) FROM "AI_Source"."PlayerCompetitionStats";
SELECT player_id, league_id, season, competition, count(*)
FROM "AI_Source"."PlayerCompetitionStats"
GROUP BY player_id, league_id, season, competition
HAVING count(*) > 1;
```

The final query must return zero rows.

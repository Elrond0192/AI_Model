# AI_Model Production Runbook

## Services

| Service | Port | Purpose |
|---|---:|---|
| `admin` | 8501 | DB profile, training, backtest, promotion |
| `api` | 8000 | typed FastAPI inference |

PostgreSQL runs on the host. Containers reach it through
`host.docker.internal:5432`.

## Start / stop

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose logs -f admin
```

```bash
docker compose down
```

Runtime data is persisted under `runtime/`.

## First database setup

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verify:

```sql
SELECT count(*) FROM ai_source.players;
SELECT count(*) FROM ai_source.teams;
SELECT min(season), max(season), count(*) FROM ai_source.player_stats;
SELECT * FROM ai_source.leagues ORDER BY name;
```

Re-run `ai_source_schema.sql` after adding a new league table or materially
changing the physical source schema. It is idempotent and does not modify source
tables.

## Database connectivity incident

Symptoms: admin Test connection fails, `/health/ready` returns 503 or logs show
PostgreSQL/canonical-view errors.

```bash
docker compose exec api getent hosts host.docker.internal
docker compose logs --tail=200 api
```

On the host:

```bash
ss -ltnp | grep 5432
```

Verify PostgreSQL accepts the Docker bridge subnet/user/database, uses
`host.docker.internal` from the container profile and does not expose 5432 to
the public Internet.

## Canonical contract incident

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql

docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production
```

Required views:

```text
ai_source.leagues
ai_source.teams
ai_source.players
ai_source.player_stats
ai_source.team_player_relations
```

## Training

```bash
docker compose run --rm admin \
  python main.py --mode train \
  --database-profile production \
  --model-dir /app/models_saved
```

Training produces a candidate only if backtesting is valid. Review and promote
it from the admin Model Registry.

## Backtest only

```bash
docker compose run --rm admin \
  python main.py --mode backtest \
  --database-profile production \
  --model-dir /app/models_saved
```

A failed/empty backtest is not promotable.

## Batch publishing

```bash
docker compose run --rm admin \
  python main.py --mode publish-batch \
  --database-profile production \
  --model-dir /app/models_saved
```

Verify:

```sql
SELECT model_run_id, count(*)
FROM ai.player_forecasts
GROUP BY model_run_id
ORDER BY model_run_id DESC;
```

## Health

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
```

`/health/live` checks the process; `/health/ready` requires data and model.

## Dynamic prediction smoke test

```bash
curl -fsS -X POST \
  http://127.0.0.1:8000/api/v2/predictions/player-team \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "player_global_id": "PLAYER_GLOBAL_ID",
    "team_global_id": "TEAM_GLOBAL_ID",
    "league": "ITA1",
    "season": 2025,
    "competition": "RS"
  }'
```

When `JWT_SECRET` is configured, use a Bearer token instead; API-key auth is
disabled in JWT mode.

## First admin credentials

First boot writes one-time credentials to `runtime/auth/.admin_credentials`.
Read it once, change the password and delete the file.

## Backup

Back up runtime state, not repository artifacts:

```bash
tar -C . -czf ai-model-runtime-$(date +%F).tar.gz runtime/
```

PostgreSQL model output is covered by the normal PostgreSQL backup strategy.

## Deployment update

```bash
git pull --ff-only
docker compose build --pull
docker compose up -d
curl -fsS http://127.0.0.1:8000/health/live
```

If `ai_source_schema.sql` changed, apply it before restarting API/training.

Never commit `.env`, DB profile JSON, trained `.joblib` files, generated
registry/metadata, sessions, SQLite audit files or one-time credentials.

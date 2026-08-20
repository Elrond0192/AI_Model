# AI_Model Production Runbook

## Services

| Service | Port | Purpose |
|---|---:|---|
| `admin` | 8501 | DB profile, validation, training, backtest, promotion, rollback |
| `api` | 8000 | promoted typed FastAPI v2 inference |

PostgreSQL runs on the host. Containers reach it through `host.docker.internal:5432`.

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

Runtime state is persisted under `runtime/`.

## First database setup

Apply all contracts in order:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_team_season.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verify:

```sql
SELECT count(*) FROM ai_source.players;
SELECT count(*) FROM ai_source.teams;
SELECT min(season), max(season), count(*) FROM ai_source.player_stats;
SELECT min(season), max(season), count(*) FROM ai_source.team_season_stats;

SELECT player_id, season, count(*)
FROM ai_source.player_stats
GROUP BY player_id, season
HAVING count(*) > 1;

SELECT team_id, season, count(*)
FROM ai_source.team_season_stats
GROUP BY team_id, season
HAVING count(*) > 1;
```

Both duplicate queries must return zero rows.

Re-run `ai_source_schema.sql` and then `ai_source_team_season.sql` after adding a league or materially changing the physical source schema.

## Database connectivity incident

Symptoms: admin connection test fails, `/health/ready` returns 503, or logs show PostgreSQL/canonical-view errors.

```bash
docker compose exec api getent hosts host.docker.internal
docker compose logs --tail=200 api
```

On the host:

```bash
ss -ltnp | grep 5432
```

Verify PostgreSQL accepts only the intended Docker bridge subnet/user/database and does not expose 5432 publicly.

## Canonical contract incident

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_team_season.sql

docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production
```

Required read views:

```text
ai_source.leagues
ai_source.teams
ai_source.players
ai_source.player_stats
ai_source.team_player_relations
ai_source.team_season_stats
```

`validate-data` must report no hard issues. Review any cross-league target-distribution warning before training.

## Training a candidate

```bash
docker compose run --rm admin \
  python main.py --mode train \
  --database-profile production \
  --model-dir /app/models_saved
```

A successful run is immutable under:

```text
/app/models_saved/runs/<model_run_id>/
```

and is registered as **candidate**. Training never replaces `/app/models_saved/production`.

The run is rejected before registration if its full-ensemble walk-forward backtest is invalid.

## Backtest only

```bash
docker compose run --rm admin \
  python main.py --mode backtest \
  --database-profile production \
  --model-dir /app/models_saved
```

Review at least:

- ensemble RMSE/MAE/bias;
- base XGBoost RMSE;
- persistence RMSE;
- interval coverage/width;
- per-league regressions;
- fold sample counts and target seasons.

A failed fold invalidates the report.

## Promotion

```bash
docker compose run --rm admin \
  python main.py --mode promote \
  --model-dir /app/models_saved
```

Promotion fails closed unless configured gates pass. A successful promotion atomically activates:

```text
/app/models_saved/production/
```

Restart/reload the API after promotion:

```bash
docker compose restart api
curl -fsS http://127.0.0.1:8000/health/ready
```

## Rollback

```bash
docker compose run --rm admin \
  python main.py --mode rollback \
  --model-dir /app/models_saved

docker compose restart api
```

Rollback reactivates the previous immutable run; it does not retrain a model.

## Batch publishing

Only the promoted production model may publish:

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

`/health/live` checks process liveness. `/health/ready` requires PostgreSQL data and a fully loaded promoted model.

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

`season` is the source-season snapshot; the target is the next season. Supported competition values: `RS`, `PO`, `CUP`, `SUPERCUP`.

When `JWT_SECRET` is configured, use the externally provisioned Bearer token instead of `X-API-Key`.

## First admin credentials

First boot writes one-time credentials to `runtime/auth/.admin_credentials`. Read them once, change the password and delete that file.

## Backup

Back up runtime state, including immutable runs and active production artifacts:

```bash
tar -C . -czf ai-model-runtime-$(date +%F).tar.gz runtime/
```

PostgreSQL model output is covered by the normal PostgreSQL backup strategy.

## Deployment update

```bash
git pull --ff-only
docker compose build --pull
```

If either source SQL adapter changed, apply both before training or API restart:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_team_season.sql
```

Then:

```bash
docker compose up -d
curl -fsS http://127.0.0.1:8000/health/live
```

Never commit `.env`, DB profile JSON, trained `.joblib` files, generated registries/metadata, sessions, audit databases or one-time credentials.

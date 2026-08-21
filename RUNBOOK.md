# AI_Model Production Runbook

## Services

| Service | Port | Purpose |
|---|---:|---|
| `admin` | 8501 | DB profile, validation, training, backtest, promotion, rollback |
| `api` | 8000 | promoted typed FastAPI v2 inference |

PostgreSQL runs on the host and Docker reaches it through `host.docker.internal:5432`.

## First database setup

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_competition.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verify the production contract:

```sql
SELECT competition, count(*)
FROM ai_source.player_competition_stats
GROUP BY competition ORDER BY competition;

SELECT competition, count(*)
FROM ai_source.team_competition_stats
GROUP BY competition ORDER BY competition;

SELECT player_id, league_id, season, competition, count(*)
FROM ai_source.player_competition_stats
GROUP BY player_id, league_id, season, competition
HAVING count(*) > 1;

SELECT team_id, league_id, season, competition, count(*)
FROM ai_source.team_competition_stats
GROUP BY team_id, league_id, season, competition
HAVING count(*) > 1;
```

Both duplicate queries must return zero rows. Re-run both source adapters after league/schema/competition ingestion changes.

## Validate data

```bash
docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production
```

Check `consecutive_pairs_by_competition`. A competition can exist in PostgreSQL but is not trainable/serveable until real same-league, same-competition `t -> t+1` pairs exist.

## Train candidate

```bash
docker compose run --rm admin \
  python main.py --mode train \
  --database-profile production \
  --model-dir /app/models_saved
```

Runs are immutable under `/app/models_saved/runs/<model_run_id>/`; production is unchanged until promotion. A complete run includes `production_state.joblib` containing the competition vocabulary and conformal state.

## Backtest

```bash
docker compose run --rm admin \
  python main.py --mode backtest \
  --database-profile production \
  --model-dir /app/models_saved
```

Review global RMSE/MAE/bias/coverage, base and persistence RMSE, `by_league`, **`by_competition`**, fold samples and target seasons. A failed fold invalidates the report.

## Promotion

```bash
docker compose run --rm admin \
  python main.py --mode promote --model-dir /app/models_saved
```

Promotion fails closed unless OOT gates pass. Competition segments with sufficient samples are compared against production, so a large PO/CUP regression can block promotion even when global RMSE improves.

After promotion:

```bash
docker compose restart api
curl -fsS http://127.0.0.1:8000/health/ready
```

## Rollback

```bash
docker compose run --rm admin \
  python main.py --mode rollback --model-dir /app/models_saved
docker compose restart api
```

## Competition smoke test

For playoffs:

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
    "competition": "PO"
  }'
```

A successful response must report `competition="PO"` and `competition_support.mode="isolated"`. Verify `source_games`, `exact_source_games`, `calibration_samples` and `calibration_scope` before presenting the confidence level as competition-specific.

A `422` is expected if the player/team lacks the requested isolated source context or the promoted model was trained before that competition acquired supervised consecutive history.

## Batch publishing

```bash
docker compose run --rm admin \
  python main.py --mode publish-batch \
  --database-profile production \
  --model-dir /app/models_saved
```

Verify forecasts by competition:

```sql
SELECT model_run_id, league, competition, count(*)
FROM ai.player_forecasts
GROUP BY model_run_id, league, competition
ORDER BY model_run_id DESC, league, competition;
```

## Connectivity / health

```bash
docker compose exec api getent hosts host.docker.internal
docker compose logs --tail=200 api
ss -ltnp | grep 5432
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
```

PostgreSQL 5432 and Docker port 8501 must stay private.

## Public operations GUI

The GUI is available at `https://admin-ai.bballstat.com` through Nginx, while
the API is available at `https://ai.bballstat.com`. Point both DNS records at
the server, but proxy them only to their loopback Docker ports (`8501` and
`8000` respectively). Never publish either Docker port in the firewall.

Protect `admin-ai.bballstat.com` with both the Streamlit application login and
Nginx basic authentication:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-ai YOUR_OPERATIONS_USER
```

The GUI Nginx `location /` requires:

```nginx
auth_basic "AI_Model Operations";
auth_basic_user_file /etc/nginx/.htpasswd-ai;
proxy_pass http://127.0.0.1:8501;
```

Issue certificates separately for both hosts:

```bash
sudo certbot --nginx -d ai.bballstat.com
sudo certbot --nginx -d admin-ai.bballstat.com
```

## First admin credentials

First boot writes one-time credentials to `runtime/auth/.admin_credentials`. Read once, change the password and delete the file.

## Backup / deployment

```bash
tar -C . -czf ai-model-runtime-$(date +%F).tar.gz runtime/
git pull --ff-only
docker compose build --pull
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_competition.sql
docker compose up -d
```

Never commit `.env`, DB profile JSON, model artifacts, registries/metadata, sessions, audit DBs or one-time credentials.

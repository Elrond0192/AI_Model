# AI_Model Production Runbook

## Services

| Service | Port | Purpose |
|---|---:|---|
| `admin` | 8501 | custom operations console: DB profile, validation/load, training, backtest, promotion, rollback, audit, settings |
| `api` | 8000 | promoted typed FastAPI v2 inference |

PostgreSQL runs on the host and Docker reaches it through `host.docker.internal:5432`.
The admin console is a custom FastAPI + HTML/CSS/JS application; Streamlit is not used.

## First database setup

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_full.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_scenario_serving.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
# Optional, required when Metric Rating is enabled:
psql -d YOUR_DATABASE -f basketball_ai/data/ai_metric_schema.sql
```

`ai_source_full.sql` builds the canonical entity/statistics source contract. `ai_scenario_serving.sql` installs the indexed training/scenario serving tables and refresh procedures. `ai_schema.sql` installs the model-owned prediction outputs. Apply them in that order; reapply the relevant contract after schema/code changes.

Verify the production contract:

```sql
SELECT competition, count(*)
FROM "AI_Source"."PlayerCompetitionStats"
GROUP BY competition ORDER BY competition;

SELECT competition, count(*)
FROM "AI_Source"."TeamCompetitionStats"
GROUP BY competition ORDER BY competition;

SELECT player_id, league_id, season, competition, count(*)
FROM "AI_Source"."PlayerCompetitionStats"
GROUP BY player_id, league_id, season, competition
HAVING count(*) > 1;

SELECT team_id, league_id, season, competition, count(*)
FROM "AI_Source"."TeamCompetitionStats"
GROUP BY team_id, league_id, season, competition
HAVING count(*) > 1;
```

Both duplicate queries must return zero rows. Re-run `ai_source_full.sql` after league/schema/competition ingestion changes.

## Operations through the GUI

Start the custom admin service:

```bash
docker compose up -d admin
docker compose ps admin
curl -fsS http://127.0.0.1:8501/healthz
```

Production access is through `https://admin-ai.bballstat.com`. The main flow is:

1. **Data Sources** – select the `production` profile, test PostgreSQL, then **Select and load**. This loads only the core forecasting contract, so it remains fast even when possession-level simulation feeds are large. Loading fails closed if the canonical source contract cannot be read.
2. **Training Runs** – inspect available seasons/competitions and start a leakage-safe candidate run. Training includes the exact production ensemble plus walk-forward backtest.
3. **Backtests** – inspect OOT RMSE/MAE/coverage globally and by competition.
4. **Model Registry** – review the candidate and explicitly confirm promotion gates before promoting. Rollback is available when a previous production run exists.
5. **API & Health** – verify the inference API process and readiness endpoint.
6. **Audit Logs / Settings** – inspect operator events, runtime settings and change the application password.

The GUI runs long training in a dedicated background worker so the browser remains responsive. Production is never changed automatically by training.

## CLI fallback: validate data

```bash
docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production
```

Check `consecutive_pairs_by_competition`. A competition can exist in PostgreSQL but is not trainable/serveable until real same-league, same-competition `t -> t+1` pairs exist.

## CLI fallback: train candidate

```bash
docker compose run --rm admin \
  python main.py --mode train \
  --database-profile production \
  --model-dir /app/models_saved
```

Runs are immutable under `/app/models_saved/runs/<model_run_id>/`; production is unchanged until promotion. A complete run includes `production_state.joblib` containing the competition vocabulary, conformal state and empirical age-curve state. Compatibility training uses out-of-fold base predictions when sufficient historical data exists.

## CLI fallback: backtest

```bash
docker compose run --rm admin \
  python main.py --mode backtest \
  --database-profile production \
  --model-dir /app/models_saved
```

Review global RMSE/MAE/bias/coverage, base and persistence RMSE, `by_league`, **`by_competition`**, fold samples and target seasons. A failed fold invalidates the report.

## CLI fallback: promotion

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

## Prediction Model 1–100 calibration

The display calibration is a separate persisted artifact; it does not retrain or
change the native Prediction Model. Run it after deploying the code and whenever
a new, validated calibration is required. The API reads the shared artifact from
`/app/models_saved/bb_rating_future_prediction/metadata.json` for the promoted direct BB Prediction Model release.

```bash
docker compose run --rm admin \
  python main.py --mode prediction-calibrate \
  --database-profile production \
```

The CLI fallback writes an artifact directly and does not trigger the Model Center restart workflow; after using this fallback, restart the API manually and verify readiness.

The command performs a temporal out-of-sample backtest and requires at least 500
valid OOS records. Do not create a synthetic/default mapping when calibration
data is insufficient; fix the data/training window instead. A successful run
prints the JSON and Markdown artifact paths. Confirm startup reports
`Loaded direct BB Prediction Model from /app/models_saved/bb_rating_future_prediction`.

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
FROM "AI"."PlayerForecasts"
GROUP BY model_run_id, league, competition
ORDER BY model_run_id DESC, league, competition;
```

## Connectivity / health

```bash
docker compose exec api getent hosts host.docker.internal
docker compose logs --tail=200 api
docker compose logs --tail=200 admin
ss -ltnp | grep 5432
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8501/healthz
```

PostgreSQL 5432 and Docker ports 8000/8501 must stay private.

## Public operations GUI

The GUI is available at `https://admin-ai.bballstat.com` through Nginx, while the API is available at `https://ai.bballstat.com`. Point both DNS records at the server, but proxy them only to their loopback Docker ports (`8501` and `8000` respectively). Never publish either Docker port in the firewall.

Protect `admin-ai.bballstat.com` with both the custom application login and Nginx basic authentication:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-ai YOUR_OPERATIONS_USER
```

Recommended GUI Nginx location:

```nginx
location / {
    auth_basic "AI_Model Operations";
    auth_basic_user_file /etc/nginx/.htpasswd-ai;

    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

The custom console uses normal HTTP requests; Streamlit websocket proxy directives are no longer required.

Issue certificates separately for both hosts:

```bash
sudo certbot --nginx -d ai.bballstat.com
sudo certbot --nginx -d admin-ai.bballstat.com
```

## First admin credentials

First boot writes one-time credentials to `runtime/auth/.admin_credentials`. Read once, change the password from **Settings**, then delete the one-time file.

## Runtime permissions

The admin container runs as non-root `appuser` and must be able to write runtime config/auth/audit/model directories. Keep the runtime directories group-writable by the container GID rather than using `chmod 777`.

## Backup / deployment

```bash
tar -C . -czf ai-model-runtime-$(date +%F).tar.gz runtime/
git pull --ff-only
docker compose build --pull
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_full.sql
docker compose up -d
```

`api` and `admin` use separate image tags (`basketball-ai-api:latest` and `basketball-ai-admin:latest`) to avoid concurrent export collisions when `docker compose build` builds both services.

Never commit `.env`, DB profile JSON, model artifacts, registries/metadata, sessions, audit DBs or one-time credentials.

## Model Center: calibration candidates and safe releases

The admin console writes auxiliary model outputs to immutable candidate directories under `runtime/models/candidates/<model>/<run-id>/`. Calibration/training must not write directly to serving paths.

Supported release IDs:
- `bb-rating`: calibration report plus `bb_rating_uncertainty.json`.
- `future-performance`: fitted `metadata.json` plus every model file declared in `model_files`.

In **Model Releases**, inspect the production and candidate validation state before promoting. Promotion validates the candidate and atomically swaps the artifact directory, keeping the prior production directory under `runtime/models/releases/<model>/`. Rollback restores that previous directory. The operation is audited. Every promotion or rollback in Model Releases and the native Prediction Model registry requests an authenticated, controlled API-container restart automatically. The admin waits for `/health/ready` to succeed before reporting success. The API container uses Compose `restart: unless-stopped`; its internal restart endpoint accepts only the configured `API_KEY`. Candidate training/calibration does not restart production. If the API key is missing or readiness is not restored within 90 seconds, the operation result explicitly reports the restart failure and requires operator intervention.

Prediction 1–100 calibration only fits the display mapping from OOS records; it does not retrain or replace the native Prediction Model. The minimum 500 OOS rows is an operational artifact gate, not a guarantee of statistical quality. Review the saved report's temporal folds and raw-versus-calibrated metrics before promotion.

Manual recovery only (for example, if the Model Center reports that its automatic restart/readiness check failed):
```bash
docker compose restart api
docker compose ps
curl -fsS http://127.0.0.1:8000/health/ready
docker compose logs --tail=150 api
```

Do not promote if validation fails or if the expected uncertainty/model files are absent. Keep the prior release until the API passes its health and inference smoke tests. For CLI promotion/rollback, follow the manual restart steps in the relevant CLI section; the Model Center UI handles restart and readiness checks automatically.

# AI_Model

Production basketball forecasting service for BBallstat.

`AI_Model` is **not a chatbot**. It reads a stable PostgreSQL contract, trains a strict season-ahead model (`t -> t+1`), evaluates the exact production ensemble out of time, promotes immutable model runs, stores bounded forecasts in PostgreSQL schema `ai`, and exposes typed dynamic inference to WordPress Chat V3.

## Production architecture

```text
PostgreSQL on host
       ^
       | host.docker.internal:5432
       |
AI_Model Docker
  |- admin :8501   validate / train / backtest / promote / rollback
  `- api   :8000   promoted FastAPI v2 inference only
       ^
       | HTTPS server-to-server
       |
WordPress Chat V3 / Bax
```

There is no SSH tunnel in production and no direct browser-to-database access.

## 1. Clone and configure

```bash
git clone https://github.com/Elrond0192/AI_Model.git
cd AI_Model
cp .env.example .env
mkdir -p runtime/config runtime/models runtime/auth runtime/audit
```

Set at least:

```env
DATABASE_PROFILE=production
DATABASE_PROFILES_FILE=/app/config/database_profiles.json
DATA_SOURCE=postgres
API_KEY=<random service key>
JWT_SECRET=
SESSION_SECRET_KEY=<random session secret>
ALLOWED_ORIGINS=https://your-wordpress-host.example
```

Generate secrets with `openssl rand -hex 32`.

## 2. Install the PostgreSQL contracts

Run the source adapter, historical team context and model-owned output schema in this order:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_team_season.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Production training reads six canonical read-only views:

- `ai_source.leagues`
- `ai_source.teams`
- `ai_source.players`
- `ai_source.player_stats`
- `ai_source.team_player_relations`
- `ai_source.team_season_stats`

`team_season_stats` is mandatory for leakage-safe team compatibility and historical backtests. The adapter never writes to `Anagrafiche`, `Analisi` or `Boxscore`.

`ai_schema.sql` creates model-owned output tables such as `ai.model_runs` and `ai.player_forecasts`.

### Database permissions

```sql
GRANT CONNECT ON DATABASE your_database TO ai_model;
GRANT USAGE ON SCHEMA ai_source, ai TO ai_model;
GRANT SELECT ON ALL TABLES IN SCHEMA ai_source TO ai_model;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ai TO ai_model;
```

Do not grant write access to the statistical source schemas and do not expose PostgreSQL port 5432 publicly.

## 3. Docker -> PostgreSQL host

`docker-compose.yml` maps:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Create the runtime database profile with:

```text
Host:          host.docker.internal
Port:          5432
Source schema: ai_source
Output schema: ai
```

PostgreSQL must allow only the relevant Docker bridge subnet in `pg_hba.conf`.

## 4. Start the admin console

```bash
docker compose up -d --build admin
docker compose logs -f admin
```

Keep port `8501` private. On first boot the one-time admin credentials are written to `runtime/auth/.admin_credentials`; delete that file after changing the password.

In **Data Sources**, create/select the PostgreSQL profile and load it. Loading fails closed if any canonical view is missing, duplicated or empty where required.

## 5. Professional training lifecycle

The production path uses:

- exact consecutive player seasons only (`t -> t+1`; gaps are excluded);
- whole target-season train/validation/calibration partitions;
- validation only for XGBoost tree-count selection, followed by refit on train + validation;
- source-season roster position and temporally bounded role vocabularies;
- historical player, roster and team state only;
- run-owned team-style normalization persisted with the model;
- finite-sample split-conformal calibration on the final ensemble output;
- expanding walk-forward evaluation of the same ensemble served by the API;
- comparisons against base XGBoost and previous-season persistence;
- RMSE, MAE, R², bias, interval coverage and segment metrics;
- immutable candidate runs and explicit promotion/rollback.

CLI:

```bash
docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production

docker compose run --rm admin \
  python main.py --mode train --database-profile production \
  --model-dir /app/models_saved

docker compose run --rm admin \
  python main.py --mode backtest --database-profile production \
  --model-dir /app/models_saved
```

Training writes an immutable run under:

```text
models_saved/runs/<model_run_id>/
```

and registers it as **candidate**. It does **not** change the model used by the API. Each complete run includes `production_state.joblib`; this is required so data-derived runtime calibration cannot drift between training, backtest and serving.

Review the walk-forward report, then promote:

```bash
docker compose run --rm admin \
  python main.py --mode promote --model-dir /app/models_saved
```

Promotion gates require a valid out-of-time full-ensemble backtest, finite RMSE/MAE/bias, sufficient samples, mandatory acceptable interval coverage, no excessive league regression, and improvement over both the base forecast and persistence. When a production model already exists, the candidate must also beat its OOT RMSE by the configured threshold.

Promotion atomically materializes the active artifacts in:

```text
models_saved/production/
```

Rollback is explicit:

```bash
docker compose run --rm admin \
  python main.py --mode rollback --model-dir /app/models_saved
```

Restart/reload the API after a promotion or rollback so the process loads the newly activated immutable artifacts.

## 6. Start the prediction API

The Docker API service reads only `/app/models_saved/production`. If no model has been promoted, readiness remains unavailable.

```bash
docker compose up -d api
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

Dynamic inference:

```http
POST /api/v2/predictions/player-team
X-API-Key: <service key>
Content-Type: application/json
```

```json
{
  "player_global_id": "313267",
  "team_global_id": "313271",
  "league": "EL",
  "season": 2025,
  "competition": "RS"
}
```

`season` is the source season and the response predicts `season + 1`. The current supervised target is season-level, therefore **only `competition="RS"` is supported**. `PO`, `CUP` and `SUPERCUP` requests are rejected with `422` until a competition-specific target/model is trained. Historical requests are evaluated strictly using state available at the requested source season, not the latest player/team state.

The response includes `model_run_id`, `model_version`, `feature_version`, `data_cutoff`, confidence bounds and `target_season`.

## 7. Publish bounded batch forecasts

Only a promoted production model can publish batch forecasts:

```bash
docker compose run --rm admin \
  python main.py --mode publish-batch \
  --database-profile production \
  --model-dir /app/models_saved
```

This upserts current-team forecasts into `ai.player_forecasts`. High-cardinality team-fit/transfer scenarios remain API calls and are not precomputed as a player × team Cartesian product.

## WordPress Chat V3

WordPress resolves exact basketball entities and calls AI_Model server-to-server over HTTPS. Keep the service credential outside the WordPress database:

```php
define('HM_AI_MODEL_API_KEY', 'same-service-key');
```

See `docs/API_INTEGRATION.md` and `docs/API_V2.md`.

## Runtime state

Secrets, immutable model runs, active production artifacts, sessions and audit state are runtime data and are ignored by Git. Production state belongs under `runtime/`, not in the repository.

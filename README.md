# AI_Model

Production basketball forecasting service for BBallstat.

`AI_Model` is **not a chatbot**. It reads a stable PostgreSQL contract, trains
the season-ahead model (`t -> t+1`), stores bounded batch forecasts in
PostgreSQL schema `ai`, and exposes typed dynamic inference to WordPress Chat V3.

## Production architecture

```text
PostgreSQL on host
       ^
       | host.docker.internal:5432
       |
AI_Model Docker
  |- admin :8501   training / backtest / promotion
  `- api   :8000   FastAPI inference
       ^
       | HTTPS server-to-server
       |
WordPress Chat V3 / Bax
```

There is no SSH tunnel and no direct browser-to-database access.

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

Generate secrets with:

```bash
openssl rand -hex 32
```

## 2. Install the PostgreSQL contracts

Run both scripts once on the BBallstat PostgreSQL database:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

`ai_source_schema.sql` automatically discovers the unified per-league source
tables and creates the five read-only canonical views consumed by the model:

- `ai_source.leagues`
- `ai_source.teams`
- `ai_source.players`
- `ai_source.player_stats`
- `ai_source.team_player_relations`

It never writes to `Anagrafiche`, `Analisi` or `Boxscore`.

`ai_schema.sql` creates model-owned output tables such as:

- `ai.model_runs`
- `ai.player_forecasts`

### Database permissions

Use a dedicated PostgreSQL login. The minimum intended permissions are:

```sql
GRANT CONNECT ON DATABASE your_database TO ai_model;
GRANT USAGE ON SCHEMA ai_source, ai TO ai_model;
GRANT SELECT ON ALL TABLES IN SCHEMA ai_source TO ai_model;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ai TO ai_model;
```

Do not grant write access to the statistical source schemas.

## 3. Let Docker reach PostgreSQL on the host

`docker-compose.yml` already maps:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

The PostgreSQL profile created in the admin console should therefore use:

```text
Host:          host.docker.internal
Port:          5432
Source schema: ai_source
Output schema: ai
```

PostgreSQL must accept connections from the Docker bridge subnet in
`pg_hba.conf`. Do **not** expose port 5432 publicly.

## 4. Start the admin console

```bash
docker compose up -d --build admin
docker compose logs -f admin
```

Open port `8501` only through a reverse proxy/private network or an SSH local
port-forward for administration.

On first boot, the one-time admin credentials are written to:

```text
runtime/auth/.admin_credentials
```

Delete that file after changing the password.

In **Data Sources**:

1. create profile `production`;
2. use `host.docker.internal:5432`;
3. test the connection;
4. select and load the profile.

## 5. Validate, train, backtest and promote

From the admin UI, load data and start a training run. At least five seasons
are expected by the operations console.

Equivalent CLI commands inside the admin container:

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

Review the backtest in **Model Registry** before promoting the candidate.

## 6. Start the prediction API

```bash
docker compose up -d api
```

Health checks:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

The second endpoint returns `200` only when data and model are available.

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

The response includes `model_run_id`, `model_version`, `feature_version`,
`data_cutoff`, confidence bounds and `target_season`.

## 7. Publish bounded batch forecasts

```bash
docker compose run --rm admin \
  python main.py --mode publish-batch \
  --database-profile production \
  --model-dir /app/models_saved
```

This upserts current-team forecasts into `ai.player_forecasts`.

High-cardinality team-fit/transfer scenarios remain API calls; they are not
precomputed as a player x team Cartesian product.

## WordPress Chat V3

WordPress resolves exact basketball entities and calls AI_Model server-to-server
over HTTPS. Set the AI_Model URL in the Chat AI admin panel and keep the service
credential outside the WordPress database:

```php
define('HM_AI_MODEL_API_KEY', 'same-service-key');
```

See `docs/API_INTEGRATION.md` and `docs/API_V2.md`.

## Runtime state

Secrets, trained models, sessions and audit state are runtime data and are
ignored by Git. Production state belongs under `runtime/`, not in the
repository.

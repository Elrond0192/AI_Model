# AI_Model

Production basketball forecasting service for BBallstat.

`AI_Model` is **not a chatbot**. It reads a stable PostgreSQL contract, trains strict season-ahead forecasts (`t -> t+1`) inside the same league and competition, evaluates the exact production ensemble out of time, promotes immutable model runs, stores bounded forecasts in PostgreSQL schema `ai`, and exposes typed inference to WordPress Chat V3.

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

Apply the entity adapter, competition-preserving statistical adapter and model-owned output schema in this order:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_competition.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Production reads these canonical read-only views:

- `ai_source.leagues`
- `ai_source.teams`
- `ai_source.players`
- `ai_source.team_player_relations`
- `ai_source.player_competition_stats`
- `ai_source.team_competition_stats`

The statistical identity is **entity + league + season + competition**. `TOT`, `RS`, `PO`, cups and any newly observed competition are not collapsed together. Known aliases are normalized (`PLAYOFFS -> PO`, `REGULAR SEASON -> RS`); new competition labels are preserved in normalized form and become serveable after they have real consecutive training history and the model is retrained.

The adapters never write to `Anagrafiche`, `Analisi` or `Boxscore`. `ai_schema.sql` creates model-owned outputs such as `ai.model_runs` and `ai.player_forecasts`.

### Database permissions

```sql
GRANT CONNECT ON DATABASE your_database TO ai_model;
GRANT USAGE ON SCHEMA ai_source, ai TO ai_model;
GRANT SELECT ON ALL TABLES IN SCHEMA ai_source TO ai_model;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ai TO ai_model;
```

Do not grant write access to source schemas and do not expose PostgreSQL 5432 publicly.

## 3. Docker -> PostgreSQL host

`docker-compose.yml` maps `host.docker.internal` through `host-gateway`. Create the runtime profile with PostgreSQL host `host.docker.internal`, port `5432`, source schema `ai_source` and output schema `ai`. Restrict `pg_hba.conf` to the intended Docker bridge/user/database.

## 4. Admin console

```bash
docker compose up -d --build admin
docker compose logs -f admin
```

Keep `8501` private. First boot writes one-time admin credentials to `runtime/auth/.admin_credentials`. In **Data Sources**, create/select the PostgreSQL profile and load it. Loading fails closed on missing views, missing columns, empty mandatory data or duplicate competition keys.

## 5. Professional training lifecycle

The production path uses:

- exact consecutive pairs only: `t -> t+1`;
- pairing only inside the same `player + league + competition`;
- no `RS -> PO`, `ITA1 -> EL` or gap-year target substitution;
- whole target-season train/validation/calibration partitions;
- source-season roster position and temporally bounded role vocabularies;
- player history and team style isolated to the requested league/competition;
- competition as an explicit learned feature in one pooled model;
- finite-sample split-conformal calibration on the final ensemble, with competition-specific intervals when calibration volume is sufficient and an explicit global fallback otherwise;
- expanding walk-forward evaluation of the exact production ensemble;
- RMSE, MAE, R², bias and coverage globally, by league, by competition, by position and by age band;
- comparison against base XGBoost and previous-season persistence;
- immutable candidate runs and explicit promotion/rollback.

Validate before training:

```bash
docker compose run --rm admin \
  python main.py --mode validate-data --database-profile production
```

The report includes observations and consecutive pairs for each competition. A competition with no real `t -> t+1` pairs remains visible in the data report but is not considered trained/serveable.

Train and backtest:

```bash
docker compose run --rm admin \
  python main.py --mode train --database-profile production \
  --model-dir /app/models_saved

docker compose run --rm admin \
  python main.py --mode backtest --database-profile production \
  --model-dir /app/models_saved
```

Training writes immutable artifacts under `models_saved/runs/<model_run_id>/` and registers the run as **candidate**. It does not replace production. `production_state.joblib` stores data-derived runtime state, including the exact competition vocabulary and competition-specific conformal calibration.

Promote only after reviewing the report:

```bash
docker compose run --rm admin \
  python main.py --mode promote --model-dir /app/models_saved
```

Promotion requires valid OOT metrics, enough samples, acceptable interval coverage, improvement over base/persistence, no excessive league regression and no excessive competition regression when a segment has enough evidence. A promoted run is atomically materialized in `models_saved/production/`.

Rollback:

```bash
docker compose run --rm admin \
  python main.py --mode rollback --model-dir /app/models_saved
```

Restart/reload the API after promotion or rollback.

## 6. Prediction API

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

Example playoff forecast:

```json
{
  "player_global_id": "313267",
  "team_global_id": "313271",
  "league": "ITA1",
  "season": 2025,
  "competition": "PO"
}
```

`season` is the **source season**; the target is `season + 1`. For the example above, player history, target-team context and persistence baseline are isolated to `ITA1 + PO`. Regular-season rows are not silently substituted.

A competition is accepted only if the promoted model has actually seen supervised consecutive pairs for it. Otherwise the API returns `422` and requires source refresh + retraining. The response includes `competition_support` with source seasons/games and whether its interval is competition-specific or uses the explicitly reported global calibration fallback.

## 7. Publish bounded batch forecasts

```bash
docker compose run --rm admin \
  python main.py --mode publish-batch \
  --database-profile production \
  --model-dir /app/models_saved
```

The batch publisher emits each real latest-season player/team/league/competition context supported by the promoted model. It does not precompute a player × team Cartesian product.

## WordPress Chat V3

WordPress resolves exact basketball entities and calls AI_Model server-to-server over HTTPS. Keep the service credential outside the WordPress database:

```php
define('HM_AI_MODEL_API_KEY', 'same-service-key');
```

See `docs/API_INTEGRATION.md` and `docs/API_V2.md`.

## Runtime state

Secrets, immutable model runs, production artifacts, sessions and audit state are runtime data and ignored by Git. Production state belongs under `runtime/`, not in the repository.

# AI_Model

Production basketball forecasting service for BBallstat.

The production API also exposes a composable Basketball Simulation & Causal
Engine. See [docs/SIMULATION_ENGINE.md](docs/SIMULATION_ENGINE.md) for the
probabilistic, matchup, lineup, roster and causal contracts.

For PostgreSQL deployments, run `basketball_ai/data/ai_source_full.sql` followed
by `basketball_ai/data/ai_scenario_serving.sql`. The unified source contract
contains the entity, competition-preserving and simulation adapters plus the
PostgreSQL parallel-safety fixes.

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
       | HTTPS through Nginx
       |
WordPress Chat V3 / Bax and authenticated operations users
```

Expose neither Docker port directly. Production uses `ai.bballstat.com` for the
API and `admin-ai.bballstat.com` for the operations GUI; PostgreSQL remains
private and is never reachable from a browser.

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
# Legacy lowercase schemas must be migrated separately before this contract.
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_full.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_scenario_serving.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Production reads these canonical read-only views:

- `"AI_Source"."Leagues"`
- `"AI_Source"."Teams"`
- `"AI_Source"."Players"`
- `"AI_Source"."TeamPlayerRelations"`
- `"AI_Source"."PlayerCompetitionStats"`
- `"AI_Source"."TeamCompetitionStats"`

The statistical identity is **entity + league + season + competition**. `TOT`, `RS`, `PO`, cups and any newly observed competition are not collapsed together. Known aliases are normalized (`PLAYOFFS -> PO`, `REGULAR SEASON -> RS`); new competition labels are preserved in normalized form and become serveable after they have real consecutive training history and the model is retrained.

HoopmetricsEngine/AdvanceStats tables in `Analisi.AdvancedStats_Player_*` and
`Analisi.AdvancedStatsTeam_*` are the authoritative model-feature source. Their
already calculated BPM, ratings, efficiency, role, on/off and related metrics
are adapted, not recomputed from raw possessions. `Boxscore` is consulted only
for missing starter counts; when AdvancedStats exposes `GamesStarted` or
`StarterPct`, those values take precedence. PBP is used only to build the
physical scenario aggregates and is never a direct XGBoost training input.

The adapters never write to `Anagrafiche`, `Analisi` or `Boxscore`. `ai_schema.sql` creates model-owned outputs such as `"AI"."ModelRuns"` and `"AI"."PlayerForecasts"`.

### Database permissions

```sql
GRANT CONNECT ON DATABASE your_database TO ai_model;
GRANT USAGE ON SCHEMA "AI_Source", "AI" TO ai_model;
GRANT SELECT ON ALL TABLES IN SCHEMA "AI_Source" TO ai_model;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA "AI" TO ai_model;
```

Do not grant write access to source schemas and do not expose PostgreSQL 5432 publicly.

## 3. Docker -> PostgreSQL host

`docker-compose.yml` maps `host.docker.internal` through `host-gateway`. Create the runtime profile with PostgreSQL host `host.docker.internal`, port `5432`, source schema `AI_Source` and output schema `AI`. Restrict `pg_hba.conf` to the intended Docker bridge/user/database.

## 4. Admin console

The admin service serves `AI Model Control Center`, a self-contained
dashboard (vanilla HTML/CSS/JS, no framework) under
`basketball_ai/admin_web/static/`: login, overview KPIs, data/snapshot
lifecycle, training, backtests, Scenario Lab, model registry, health and
audit pages. Icons are Phosphor Icons, self-hosted under
`static/assets/phosphor/` (regular + fill weights, woff2/woff only) —
no external CDN calls, so the console keeps working fully offline/air-gapped.

```bash
docker compose up -d --build admin
docker compose logs -f admin
```

Keep `8501` private. First boot writes one-time admin credentials to `runtime/auth/.admin_credentials`. In **Data Sources**, create/select the PostgreSQL profile and load it. Loading fails closed on missing views, missing columns, empty mandatory data or duplicate competition keys.

### Publish the GUI securely

For BBallstat, create DNS records for `ai.bballstat.com` and
`admin-ai.bballstat.com` pointing to the server. Keep Compose bound to
`127.0.0.1`; Nginx is the only public entry point. Protect the GUI both with
the application login and HTTP basic authentication:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-ai YOUR_OPERATIONS_USER
```

Create an Nginx virtual host for `admin-ai.bballstat.com`:

```nginx
server {
    listen 80;
    server_name admin-ai.bballstat.com;

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
}
```

Enable the site, obtain TLS, and verify it:

```bash
sudo ln -s /etc/nginx/sites-available/admin-ai.bballstat.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d admin-ai.bballstat.com
curl -fsS https://admin-ai.bballstat.com
```

Use a separate Nginx virtual host for `ai.bballstat.com`, proxying only to
`127.0.0.1:8000`. Do not use basic authentication on the API because WordPress
authenticates with `X-API-Key` server-to-server.

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

Prepare one immutable snapshot, then train and backtest without re-reading the
growing PostgreSQL history:

```bash
docker compose run --rm admin \
  python main.py --mode prepare-snapshot --database-profile production \
  --snapshot-dir /app/models_saved/snapshots

# Use the snapshot_id printed by the previous command.
docker compose run --rm admin \
  python main.py --mode train --snapshot-id SNAPSHOT_ID \
  --snapshot-dir /app/models_saved/snapshots \
  --model-dir /app/models_saved

docker compose run --rm admin \
  python main.py --mode backtest --snapshot-id SNAPSHOT_ID \
  --snapshot-dir /app/models_saved/snapshots \
  --model-dir /app/models_saved
```

After each ETL context completes, refresh only its serving partition:

```sql
CALL "AI_Source"."RefreshContext"('ITA1', 2025, 'RS');
```

The same hook is available to n8n/systemd/cron without embedding SQL:

```bash
python main.py --mode refresh-serving --database-profile production \
  --league-key ITA1 --season 2025 --competition RS
```

Run it after HoopmetricsEngine/AdvanceStats commits each changed context. Entity
views reflect Anagrafiche changes immediately; create a new immutable snapshot
after the daily ETL completes, and reload/restart the API so newly added
players, teams and competitions enter the in-memory identity dictionaries.

After the initial serving backfill, `POSTGRES_TRAINING_SOURCE=auto` lets the
training/admin path select physical serving tables when both contain data and
safely falls back to canonical views otherwise. The API analysis path stays on
canonical views so generic questions retain every observed split. Set
`POSTGRES_TRAINING_SOURCE=serving` only to make
the indexed store mandatory. Snapshot preparation then reads the indexed
physical feature store instead of rebuilding features from the growing raw
boxscore history.

Generic chat analysis continues to query every observed context, including
`Home` and `Away`. Forecast training is deliberately narrower: by default only
`RS`, `PO` and `TOT` form supervised season-ahead pairs, and both source and
target must have at least three games. Configure these gates with
`MODEL_TRAINING_COMPETITIONS` and `MODEL_MIN_TRAIN_GAMES`.

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

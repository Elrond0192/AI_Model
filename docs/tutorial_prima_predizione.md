# Prima predizione con AI_Model

Questa guida descrive il percorso di produzione corrente: PostgreSQL
sull'host, AI_Model in Docker e Chat V3 via API.

## 1. Prepara PostgreSQL

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_competition.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_simulation.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verifica:

```sql
SELECT count(*) FROM ai_source.players;
SELECT count(*) FROM ai_source.teams;
SELECT min(season), max(season), count(*) FROM ai_source.player_stats;
```

## 2. Configura il profilo PostgreSQL

```bash
cp .env.example .env
mkdir -p runtime/config runtime/models runtime/audit
```

Scrivi `runtime/config/database_profiles.json`:

```json
{
  "production": {
    "host": "host.docker.internal",
    "port": 5432,
    "database": "YOUR_DATABASE",
    "user": "ai_model",
    "password": "YOUR_PASSWORD"
  }
}
```

## 3. Training e promozione

Il dataset deve avere almeno cinque stagioni. Esegui validazione, training,
backtest e promozione dalla CLI:

```bash
docker compose run --rm ops python main.py --mode validate-data --database-profile production
docker compose run --rm ops python main.py --mode train --database-profile production --model-dir /app/models_saved
docker compose run --rm ops python main.py --mode backtest --database-profile production --model-dir /app/models_saved
docker compose run --rm ops python main.py --mode promote --model-dir /app/models_saved
```

## 4. Avvia FastAPI

```bash
docker compose up -d api
curl http://127.0.0.1:8000/health/ready
```

La risposta deve indicare `data: true` e `model: true`.

## 5. Prima predizione

```bash
curl -X POST \
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

`season=2025` significa usare lo stato disponibile nel 2025 per il forecast
della stagione target 2026.

Per WordPress usa `docs/API_INTEGRATION.md`.

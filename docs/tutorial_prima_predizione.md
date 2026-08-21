# Prima predizione con AI_Model

Questa guida descrive il percorso di produzione corrente: PostgreSQL
sull'host, AI_Model in Docker e Chat V3 via API.

## 1. Prepara PostgreSQL

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_schema.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verifica:

```sql
SELECT count(*) FROM "AI_Source"."Players";
SELECT count(*) FROM "AI_Source"."Teams";
SELECT min(season), max(season), count(*) FROM "AI_Source"."PlayerStats";
```

## 2. Avvia la console

```bash
cp .env.example .env
mkdir -p runtime/config runtime/models runtime/auth runtime/audit
docker compose up -d --build admin
```

La GUI resta in ascolto solo su `127.0.0.1:8501`. In produzione pubblicala
dietro Nginx su `https://admin-ai.bballstat.com`, con certificato TLS e HTTP
basic authentication aggiuntiva; non aprire la porta 8501 nel firewall.

Crea il profilo PostgreSQL con:

```text
host = host.docker.internal
port = 5432
source_schema = "AI_Source"
output_schema = ai
```

Testa la connessione e carica i dati.

## 3. Training e promozione

In **Training Runs** avvia il training. Il dataset deve avere almeno cinque
stagioni. Controlla **Backtests**, quindi promuovi il candidato da
**Model Registry**.

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

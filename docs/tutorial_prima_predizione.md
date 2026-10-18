# Prima predizione con AI_Model

Questa guida avvia lo stack corrente: PostgreSQL sull'host, AI_Model in Docker e WordPress Chat V3 via API. Per l'elenco degli endpoint e i contratti completi vedere [API v2](API_V2.md) e [integrazione WordPress](API_INTEGRATION.md).

## 1. Prepara PostgreSQL

Esegui gli script nell'ordine indicato:

```bash
psql -d YOUR_DATABASE -f basketball_ai/data/ai_source_full.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_scenario_serving.sql
psql -d YOUR_DATABASE -f basketball_ai/data/ai_schema.sql
```

Verifica le viste canoniche correnti:

```sql
SELECT count(*) FROM "AI_Source"."Players";
SELECT count(*) FROM "AI_Source"."Teams";
SELECT min(season), max(season), count(*) FROM "AI_Source"."PlayerCompetitionStats";
SELECT competition, count(*) FROM "AI_Source"."PlayerCompetitionStats"
GROUP BY competition ORDER BY competition;
```

Le statistiche sono separate per giocatore/squadra, lega, stagione e competizione. Non usare il vecchio nome `AI_Source.PlayerStats`: la vista canonica è `"AI_Source"."PlayerCompetitionStats"`.

## 2. Configura e avvia la console

```bash
cp .env.example .env
mkdir -p runtime/config runtime/models runtime/auth runtime/audit
docker compose up -d --build admin
docker compose ps admin
curl -fsS http://127.0.0.1:8501/healthz
```

La GUI è una console FastAPI + HTML/CSS/JS, non Streamlit. In produzione pubblicala dietro Nginx e TLS su `https://admin-ai.bballstat.com`, con HTTP Basic Authentication aggiuntiva; non esporre direttamente la porta 8501.

Crea il profilo PostgreSQL con host `host.docker.internal`, porta `5432`, schema sorgente `AI_Source` e schema di output `AI`. Carica il profilo dalla sezione **Data Sources** e verifica che il controllo del contratto sia superato.

## 3. Valida, allena e valuta

In **Training Runs** esegui prima la validazione dei dati, poi crea uno snapshot immutabile e avvia training/backtest. Non assumere che il semplice numero di stagioni renda il modello idoneo: il training richiede coppie consecutive reali `t -> t+1` nello stesso giocatore, lega e competizione, oltre ai gate di qualità e di campionamento. Le competizioni senza coppie valide restano non supportate.

Esamina i risultati OOT per stagione, lega e competizione prima di promuovere. Il training crea un candidato; non sostituisce automaticamente il modello in produzione.

## 4. Avvia e verifica l'API

```bash
docker compose up -d api
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
```

La readiness deve indicare che dati e modello richiesti per il servizio sono disponibili. L'API usa `X-API-Key` per le chiamate server-to-server; non esporre la chiave nel browser.

## 5. Prima predizione

```bash
curl -X POST http://127.0.0.1:8000/api/v2/predictions/player-team \
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

`season` è la stagione sorgente: il target è `season + 1`. Il giocatore e la squadra devono avere il contesto esatto lega/stagione/competizione e il modello promosso deve supportare quella competizione.

## 6. Rilasci e restart

La pagina **Model Releases** gestisce i candidati di calibrazione Prediction 1–100, BB-Rating e Future Performance; la registry nativa del Prediction Model è un ciclo di vita separato. Le promozioni/rollback effettuati dalla GUI richiedono il restart autenticato dell'API e attendono `/health/ready`. Le operazioni CLI che modificano artefatti serviti richiedono invece il restart manuale documentato in [RUNBOOK.md](../RUNBOOK.md).

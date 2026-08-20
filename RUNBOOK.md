# Runbook – Basketball Performance AI

**Version:** 1.0  
**Last Updated:** 2025-05-21  

This runbook covers common operational procedures and incident responses.

---

## Service Overview

| Component        | Port  | Technology           |
|------------------|-------|----------------------|
| REST API         | 8000  | FastAPI / uvicorn    |
| Admin panel      | 8501  | Streamlit            |
| Chat widget      | 8502  | Streamlit embed      |

```bash
docker-compose up -d          # start all services
docker-compose logs -f api    # follow API logs
```

---

## Startup & Health Checks

```bash
# Liveness (always 200 when process is running)
curl http://localhost:8000/health/live

# Readiness (200 when data + model loaded, 503 otherwise)
curl http://localhost:8000/health/ready
```

Expected ready response:
```json
{"status": "ready", "data": true, "model": true}
```

---

## Incident: Model fails to load

**Symptoms:** `/health/ready` returns `{"model": false}`.

**Steps:**
1. Check model files exist: `ls models_saved/`
2. Check for pickle corruption: `python -c "import joblib; joblib.load('models_saved/performance_model.joblib')"`
3. Retrain: `python main.py --mode train`
4. Check model promotion status: `GET /api/v1/internal/models/status`
5. If candidate is available, promote: `POST /api/v1/internal/models/promote`

---

## Incident: Database unreachable

**Symptoms:** `/health/ready` returns `{"data": false}` with SQL source.

**Steps:**
1. Check `DATABASE_PROFILE` selects an existing PostgreSQL profile and verify it from the operations console.
2. Check network connectivity: `nc -zv <host> 1433`
3. Check circuit breaker state in logs: look for `[CircuitBreaker:postgresql]`.
4. Check ingestion error log: `python -c "from basketball_ai.data.ingestion import IngestionTracker; print(IngestionTracker().list_failed())"`
5. Restart with file-based data as fallback: `DATA_SOURCE=file docker-compose up -d api`

---

## Incident: Feature drift detected

**Symptoms:** Log line `[DriftDetection] SIGNIFICANT DRIFT detected for 'X': PSI=Y`.

**Steps:**
1. Check drift report: `GET /api/v1/internal/metrics` (includes route latencies).
2. Identify drifted features from PSI report.
3. Re-train model with fresh data: `python main.py --mode train --source sql`
4. Run backtest: see `basketball_ai/models/backtest.py`.
5. Promote new model if improved: `POST /api/v1/internal/models/promote`

---

## Incident: Admin account locked

**Symptoms:** Can't log in as admin.

**Steps:**
1. Lockout clears automatically after 15 minutes (900 seconds).
2. Override: edit `users.json`, set `"failed_attempts": 0` and remove `"locked_until"`.
3. If password is lost, delete `users.json` — admin account will be recreated on next start.
4. New credentials are written to `.admin_credentials` (chmod 0600).

---

## Incident: High latency

**Symptoms:** p95 response time > 500ms.

**Steps:**
1. Check in-process metrics: `GET /api/v1/internal/metrics`
2. Look for slow queries: grep logs for `[SlowQuery]`.
3. Check prediction cache hit rate — if cache is cold, first requests will be slow.
4. Scale horizontally: add uvicorn workers (`--workers 4`).

---

## Backup & Recovery

```bash
# Create backup
python main.py --mode backup
# Output: backups/backup_YYYYMMDDTHHMMSS.zip + .sha256

# Verify backup integrity
sha256sum -c backups/backup_YYYYMMDDTHHMMSS.sha256

# Restore models
unzip backups/backup_YYYYMMDDTHHMMSS.zip models_saved/ -d /
```

---

## Data Validation

```bash
python main.py --mode validate-data
# Generates: data/quality_reports/quality_YYYYMMDDTHHMMSS.json
```

---

## Tenant Management

```bash
# Check usage for a tenant
python -c "from basketball_ai.tenancy import TenantManager; print(TenantManager().export_usage_csv('acme'))"

# Set quota
python -c "from basketball_ai.tenancy import TenantManager; TenantManager().set_quota('acme', '/api/v1/predictions', 5000)"
```

---

## JWT Key Rotation

1. Generate new secret: `python -c "import secrets; print(secrets.token_hex(32))"`
2. Add to `JWT_SECRETS` env var as a new first entry (keep old secret for active sessions):
   ```json
   [{"kid": "v2", "secret": "NEW_SECRET"}, {"kid": "v1", "secret": "OLD_SECRET"}]
   ```
3. Old tokens signed with `v1` will still validate during the transition window.
4. Once all `v1` tokens have expired, remove the old entry.

---

## Audit Log

```bash
# Last 100 requests by user
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/internal/audit?user=alice&limit=100"
```

---

## Environment Variables Reference

| Variable                  | Default      | Description                               |
|---------------------------|--------------|-------------------------------------------|
| `JWT_SECRET`              | (none)       | JWT signing secret (single-key mode)      |
| `JWT_SECRETS`             | (none)       | JSON list of `{kid, secret}` (rotation)   |
| `ALLOWED_ORIGINS`         | (none)       | Comma-separated CORS allowlist            |
| `API_ENV`                 | `production` | `development` allows `*` CORS             |
| `DATA_SOURCE`             | `file`       | `file` or `sql`                           |
| `DATA_DIR`                | `data/sample`| CSV data directory                        |
| `MODEL_DIR`               | `models_saved`| Model save directory                     |
| `AUDIT_DB_PATH`           | `audit.db`   | SQLite audit log path                     |
| `INGESTION_DB`            | `data/ingestion.db`| Ingestion state DB                  |
| `TENANTS_FILE`            | `tenants.json`| Tenant quotas file                       |
| `LOG_FORMAT`              | `text`       | `text` or `json`                          |
| `SLOW_QUERY_THRESHOLD_S`  | `2.0`        | Seconds before logging a slow SQL query   |
| `DB_MAX_RETRIES`          | `3`          | DB connection retry count                 |
| `CB_FAILURE_THRESHOLD`    | `5`          | Circuit-breaker failure threshold         |
| `MODEL_PROMOTION_THRESHOLD_PCT` | `2.0`  | Min improvement % to auto-promote model   |

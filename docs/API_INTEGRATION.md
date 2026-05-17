# Basketball Performance AI – API Integration Guide

This guide explains how to integrate the **Basketball Performance AI** REST API
into your application.  It covers authentication, multi-tenancy, all available
endpoints, error handling, and SDK generation.

---

## Base URL

| Environment | URL |
|---|---|
| Local development | `http://localhost:8000` |
| Docker Compose (API service) | `http://api:8000` (internal) / `http://localhost:8000` (host) |
| Production | `https://your-api.example.com` |

Interactive documentation (Swagger UI) is always available at `{BASE_URL}/docs`.
OpenAPI JSON spec is at `{BASE_URL}/openapi.json`.

---

## Authentication

The API uses **JWT Bearer tokens** (HMAC-SHA256 / HS256).

### 1. Obtain a token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "admin",
  "password": "your-password"
}
```

**Response**

```json
{
  "access_token":  "<jwt>",
  "refresh_token": "<jwt>",
  "token_type":    "bearer",
  "expires_in":    1800
}
```

### 2. Use the token

Include the access token in every subsequent request:

```http
GET /api/v1/players
Authorization: Bearer <access_token>
```

### 3. Refresh an expired token

Access tokens expire after 30 minutes (configurable via `JWT_ACCESS_EXPIRE`).
Use the refresh token (valid 24 h by default) to obtain a new pair without
re-entering credentials:

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{ "refresh_token": "<refresh_token>" }
```

### 4. Inspect the current caller

```http
GET /api/v1/auth/me
Authorization: Bearer <access_token>
```

```json
{
  "username":   "admin",
  "tenant_id":  "acme-sports",
  "role":       "analyst",
  "token_kind": "access"
}
```

---

## Multi-tenancy

Every token carries a `tenant_id` claim that identifies the customer
organisation.  This allows a single API deployment to serve multiple customers
with isolated data scopes.

| Deployment model | Recommended setup |
|---|---|
| **SaaS hosted** (you manage infra) | One running instance per tenant; set `JWT_TENANT_ID=<slug>` per container |
| **On-premise** (customer installs) | Customer sets `JWT_TENANT_ID` in their `.env` |
| **Shared instance** | Future: `tenant_id` used to partition DB queries |

The `tenant_id` is visible in the `/auth/me` response and in all structured
log entries (when `LOG_FORMAT=json`).

---

## Endpoints

### Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/token` | Obtain access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh access token |
| `GET`  | `/api/v1/auth/me` | Current caller identity |

### Players

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/players` | List players (filters: `position`, `nationality`, `min_age`, `max_age`, `limit`, `offset`) |
| `GET` | `/api/v1/players/{player_id}` | Player details (ETag cached) |
| `GET` | `/api/v1/players/{player_id}/stats` | Season stats |
| `GET` | `/api/v1/players/{player_id}/profile` | Enriched profile (form, consistency, trajectory) |

### Teams

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/teams` | List teams |
| `GET` | `/api/v1/teams/{team_id}` | Team details |
| `GET` | `/api/v1/teams/{team_id}/analysis` | Team analysis (squad rating, top performers) |

### Predictions

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/predictions/{player_id}` | Predict player rating at a given team |
| `GET` | `/api/v1/predictions/{player_id}/trajectory` | Age trajectory |
| `GET` | `/api/v1/predictions/{player_id}/peak` | Peak prediction |
| `POST` | `/api/v1/predictions/batch` | Batch ratings for up to 50 (player, team) pairs |
| `GET` | `/api/v1/predictions/{player_id}/explain` | Top-5 SHAP feature contributions |

### Scenarios

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/scenarios/compare` | Compare player at multiple teams |
| `GET` | `/api/v1/scenarios/best-teams` | Best-fit teams for a player |
| `GET` | `/api/v1/scenarios/best-players` | Best players for a team |
| `POST` | `/api/v1/scenarios/transfer-impact` | Transfer impact analysis |
| `POST` | `/api/v1/scenarios/lineup` | Lineup what-if |
| `POST` | `/api/v1/scenarios/teammates` | Teammate quality what-if |

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Natural-language query with session context |

### WordPress / Embeds

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/wordpress/player-card/{player_id}` | Compact player snapshot for CMS shortcodes |

---

## Error handling

All errors follow a consistent JSON envelope:

```json
{ "detail": "Human-readable error message" }
```

| HTTP status | Meaning |
|---|---|
| `400` | Bad request (invalid parameters) |
| `401` | Missing, invalid, or expired token |
| `404` | Resource not found |
| `422` | Request body validation failed |
| `429` | Rate limit exceeded |
| `503` | Model or data not yet loaded |

---

## Rate limits

In production (`API_ENV=production`) the `slowapi` middleware limits each
client to **60 requests / minute** by default.  Configure via the `RATE_LIMIT`
env var (see `.env.example`).

---

## SDK generation

The API exposes a standard OpenAPI 3.0 specification.  You can generate a
typed client in any language:

### Python (with `openapi-python-client`)

```bash
pip install openapi-python-client
openapi-python-client generate --url http://localhost:8000/openapi.json
```

This creates a `basketball-performance-ai-client/` package with typed models
and async/sync client methods.

### TypeScript / JavaScript (with `openapi-typescript-codegen`)

```bash
npx openapi-typescript-codegen \
  --input http://localhost:8000/openapi.json \
  --output ./src/api-client \
  --client fetch
```

### Using the raw OpenAPI spec

Download the spec once and commit it to your project:

```bash
curl http://localhost:8000/openapi.json -o openapi.json
```

Then regenerate clients whenever the API version changes.

---

## Quick start (Python)

```python
import requests

BASE = "http://localhost:8000"

# 1. Authenticate
r = requests.post(f"{BASE}/api/v1/auth/token",
                  json={"username": "admin", "password": "your-password"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. List players
players = requests.get(f"{BASE}/api/v1/players?limit=10", headers=headers).json()

# 3. Predict rating
prediction = requests.get(
    f"{BASE}/api/v1/predictions/1",
    params={"team_id": 3},
    headers=headers,
).json()
print(prediction["predicted_rating"])

# 4. Chat
reply = requests.post(
    f"{BASE}/api/v1/chat",
    json={"message": "How good is Player 1 at Team 3?"},
    headers=headers,
).json()
print(reply["reply"])
```

---

## Deployment options

### Option A – Docker Compose (self-hosted)

```bash
# 1. Copy and fill in secrets
cp .env.example .env
# Edit JWT_SECRET, JWT_TENANT_ID, ALLOWED_ORIGINS, etc.

# 2. Train the model (first time only)
python main.py --mode generate-data
python main.py --mode train

# 3. Start all services
docker compose up -d

# API:   http://localhost:8000
# Admin: http://localhost:8501
# Embed: http://localhost:8502
```

### Option B – API only (Docker)

```bash
docker build -t basketball-ai .
docker run -d \
  --name basketball-ai \
  -p 8000:8000 \
  -e JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -e JWT_TENANT_ID="my-org" \
  -e ALLOWED_ORIGINS="https://my-app.example.com" \
  -e API_ENV=production \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/models_saved:/app/models_saved:ro" \
  basketball-ai
```

### Option C – Azure App Service / Cloud Run

Set the environment variables in the service configuration panel and mount the
data/models directories via a storage account or Cloud Storage bucket.

---

## Security checklist

- [ ] Set `JWT_SECRET` to a strong random value (64+ hex chars)
- [ ] Set `ALLOWED_ORIGINS` to your exact frontend origin(s)
- [ ] Set `API_ENV=production`
- [ ] Enable HTTPS via a reverse proxy (nginx, Caddy, Azure Front Door, etc.)
- [ ] Rotate `JWT_SECRET` periodically and revoke old sessions
- [ ] Limit container network exposure (bind admin panel to `localhost` only)

# AI_Model

Prediction service for BBallstat. The model reads PostgreSQL, trains a true
season-ahead forecast (`t → t+1`), publishes bounded snapshots to schema `ai`
and exposes high-cardinality scenarios through FastAPI. Conversation remains
in WordPress Chat V3.

## Deployment

1. Copy `.env.example` to `.env` and set API/JWT secrets.
2. Run `docker compose up -d --build`.
3. Open the authenticated operations console at `http://SERVER:8501`, or expose
   it behind an HTTPS reverse proxy.
4. Create one or more PostgreSQL profiles and choose the active database.
5. Load data, run training/backtest, then promote the candidate.

PostgreSQL stays on the host. Docker reaches it as `host.docker.internal`; the
compose file maps that name to `host-gateway`. No SSH tunnel exists in this
application.

The database must expose read-only canonical views:

- `ai_source.leagues`
- `ai_source.teams`
- `ai_source.players`
- `ai_source.player_stats`
- `ai_source.team_player_relations`

Apply `basketball_ai/data/ai_schema.sql` once for output tables.

## Interfaces

- `POST /api/v2/predictions/player-team`: typed dynamic inference.
- `python main.py --mode publish-batch --database-profile NAME`: publish
  current-team forecasts to `ai.player_forecasts`.
- JSON/GZIP export is optional portability output, not the primary transport.

Database profile secrets live in the ignored `runtime/config` volume (mode
`0600` on Linux), or can be injected with `DATABASE_PROFILES_JSON`.

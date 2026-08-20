# AI_Model prediction contract v2

`AI_Model` is an inference service, not a chatbot. WordPress resolves player and team entities itself and calls `POST /api/v2/predictions/player-team` with server-side global IDs.

## Request

```json
{
  "player_global_id": "313267",
  "team_global_id": "313271",
  "league": "EL",
  "season": 2025,
  "competition": "RS"
}
```

`season` is the **source season**. The forecast target is always `season + 1`.

Supported competition values are:

- `RS`
- `PO`
- `CUP`
- `SUPERCUP`

The API does not silently coerce arbitrary competition strings.

## Historical boundary

Before inference the service builds an as-of snapshot for the requested source season. Player history, roster relations and team style after that season are excluded from the calculation. This prevents a historical request/backtest from reading current team/player state.

The request is rejected if the player or target team has no resolvable state for the requested source-season context.

## Response metadata

Every successful response includes:

- `model_run_id`
- `model_version`
- `feature_version`
- `data_cutoff`
- source `season`
- `target_season`
- `predicted_rating`
- confidence bounds
- `generated_at`

Production Docker loads only the explicitly promoted artifacts in `models_saved/production`; candidate runs are never served by the API.

Precomputed forecasts belong in PostgreSQL `ai.player_forecasts`; high-cardinality player/team scenarios are evaluated dynamically.

## Authentication and network

WordPress calls the endpoint server-to-server using `X-API-Key` or an externally provisioned Bearer token. Global database IDs must not be returned to the browser by the WordPress layer.

For Docker with PostgreSQL on the host, the compose service maps `host.docker.internal` to `host-gateway`; PostgreSQL must allow only the intended Docker bridge subnet and must not expose port 5432 publicly.

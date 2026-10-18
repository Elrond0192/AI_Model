# WordPress / Chat V3 API integration

AI_Model is a server-to-server analytics service. WordPress owns conversation orchestration and resolves public/player/team entities; AI_Model resolves the supplied global identifiers internally. Do not expose internal database IDs, API keys, or database connectivity to the browser.

## Authentication and transport

- Use HTTPS to the API host (production: `https://ai.bballstat.com`).
- Send `X-API-Key: <service key>` from trusted server-side code. A configured Bearer-token integration is also supported where explicitly provisioned.
- Keep PostgreSQL port 5432 private. Do not put API credentials in JavaScript, HTML, public REST responses, or logs.
- Requests must preserve league, season and competition context. Do not merge a player's statistics across leagues or competitions.

## Available API entry points

| Purpose | Method and path | Contract |
|---|---|---|
| Season-ahead player/team forecast | `POST /api/v2/predictions/player-team` | Source season and exact league/competition; target is source season + 1 |
| BB-Rating | `POST /api/v2/bb-rating/player` | Contextual descriptive rating from 1–100; not a forecast |
| Metric Rating | `POST /api/v2/metric-rating/rate` | Rates one metric value against a stored distribution |
| Player metric snapshot | `POST /api/v2/metric-rating/player-snapshot` | Returns requested metric ratings for a player/context |
| Player Future Performance | `POST /api/v2/future-performance/player` | Separate multivariate next-season player-stat forecast |
| Player Intelligence capabilities | `GET /api/v2/player-intelligence/capabilities` | Lists available/partial/missing analysis capabilities |
| Scenario evaluation | `POST /api/v2/scenarios/evaluate` | Typed scenario analysis; preserve support and limitation metadata |

See [API_V2.md](API_V2.md) for request/response contracts and [PLAYER_INTELLIGENCE.md](PLAYER_INTELLIGENCE.md), [PLAYER_FUTURE_PERFORMANCE.md](PLAYER_FUTURE_PERFORMANCE.md), [BB_RATING_ENGINE.md](BB_RATING_ENGINE.md), [METRIC_RATING_ENGINE.md](METRIC_RATING_ENGINE.md) and [SIMULATION_ENGINE.md](SIMULATION_ENGINE.md) for semantics.

## Error and evidence handling

- Treat `422` as a request/context/model-support validation failure; do not retry unchanged requests as though they were transient.
- Treat `401/403` as authentication/authorization failures and do not leak the rejected credential.
- Treat `503` or failed readiness as unavailable model/data dependencies; return a controlled service error or retry according to the WordPress retry policy.
- Preserve each response's model/version lineage, sample support, calibration scope, warnings and limitations. A numeric scenario output is not automatically a causal claim.
- If the API returns a partial/unavailable evidence status, Chat V3 must not present it as a validated conclusion.

## Operational endpoints

- `GET /health/live` and `GET /health/ready` are operational checks, not public player endpoints.
- `POST /internal/restart` is an internal API-key-authenticated endpoint used by Model Center release operations. Do not call it from WordPress or expose it to end users.

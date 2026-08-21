# WordPress integration

WordPress Chat V3 resolves exact player/team entities, league, numeric season
and competition before calling AI_Model over authenticated HTTPS.

Dynamic prediction:

```http
POST /api/v2/predictions/player-team
Authorization: Bearer <service-token>
Content-Type: application/json

{"player_global_id":"...","team_global_id":"...","league":"ITA1","season":2026,"competition":"RS"}
```

Every response includes `model_run_id`, `model_version`, `feature_version` and
`data_cutoff`. WordPress never receives PostgreSQL credentials and AI_Model
does not expose a chat endpoint.

For frequent forecasts, run `main.py --mode publish-batch`; results are upserted
into `"AI"."PlayerForecasts"`. JSON export is optional and not used by Chat V3 as
its primary transport.

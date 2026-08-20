# AI_Model prediction contract v2

`AI_Model` is an inference service, not a chatbot.  WordPress resolves player
and team entities itself and calls `POST /api/v2/predictions/player-team` with
their server-side global IDs, numeric season, league and competition.

The response always carries `model_run_id`, `model_version`,
`feature_version` and `data_cutoff`.  Precomputed forecasts belong in
PostgreSQL `ai.player_forecasts`; JSON export is retained only as a portable
offline artifact.

For Docker with PostgreSQL on the host, set `DATABASE_URL` to
`postgresql+psycopg://…@host.docker.internal:5432/…`.  The compose service maps
that name to `host-gateway`; PostgreSQL must allow the Docker bridge subnet.

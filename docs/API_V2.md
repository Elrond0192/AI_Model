# AI_Model prediction contract v2

`AI_Model` is an inference service, not a chatbot. WordPress resolves player/team entities and calls `POST /api/v2/predictions/player-team` with server-side global IDs.

## Request

```json
{
  "player_global_id": "313267",
  "team_global_id": "313271",
  "league": "ITA1",
  "season": 2025,
  "competition": "PO"
}
```

`season` is the **source season** and the target is `season + 1`.

`competition` is open-ended. Known aliases are normalized (`playoffs -> PO`, `regular season -> RS`, `Super Cup -> SUPERCUP`); other labels are normalized to uppercase identifiers. A competition is accepted by a promoted model only if that model was trained on real consecutive samples for it. Merely sending a new label never enables an unsupported forecast.

## Isolation semantics

A request is resolved as:

```text
player + league + competition + source season
team   + league + competition + source season
```

Player history and team context are filtered to that league/competition before prediction. Therefore `competition=PO` does not silently substitute RS rows. The request returns `422` when the player has no exact source-season row for the requested competition/league, when the target team lacks the requested competition context, or when the promoted model has no trained competition vocabulary entry.

## Response

Every successful response includes model lineage, source/target season, prediction and confidence bounds plus:

```json
{
  "competition_support": {
    "mode": "isolated",
    "competition": "PO",
    "source_seasons": 3,
    "source_games": 24,
    "exact_source_games": 8,
    "calibration_scope": "competition",
    "calibration_samples": 42
  }
}
```

`calibration_scope="competition"` means the finite-sample conformal half-width was calibrated on that competition. `global` means that competition did not reach the configured calibration minimum, so the service uses the global final-ensemble conformal interval and says so explicitly.

## Training contract

Supervised pairs require:

```text
same player
same league
same competition
target season = source season + 1
```

The pooled model can learn shared basketball patterns, but `RS -> PO`, domestic -> European, and gap-year pairs are never training targets.

## Production lifecycle

Production Docker loads only explicitly promoted artifacts from `models_saved/production`. `production_state.joblib` includes run-owned style calibration, competition vocabulary and per-competition conformal state. Backtests report `by_competition`; promotion can reject a candidate that regresses substantially in a competition with enough OOT samples.

Precomputed forecasts are stored in `ai.player_forecasts`, whose key already includes `competition`.

## Authentication and network

WordPress calls server-to-server using `X-API-Key` or an externally provisioned Bearer token. Global database IDs must not be returned to the browser. PostgreSQL on the Docker host is reached through `host.docker.internal`; port 5432 must remain private.

# Chat V3 Basketball Scenario Engine

AI_Model exposes a composable server-side scenario surface at `POST /api/v2/scenarios/evaluate`.
The goal is not to map every possible natural-language question to a bespoke endpoint. Chat V3
maps questions onto reusable basketball dimensions: player(s), team(s), source/target league,
source/target competition, season, style, role, usage/minutes and roster changes.

## Supported scenario families

| Scenario | Purpose | Primary method |
| --- | --- | --- |
| `player_competition` | Player performance in PO/RS/CUP/etc. and optional competition comparison | observed data |
| `team_competition` | Team performance by competition | observed data |
| `player_trend`, `team_trend` | Multi-season evolution | observed data |
| `player_compare`, `team_compare` | Rich same-context comparison | observed data |
| `player_team` | Player performance in a target team | strict supervised ensemble when same-league |
| `playoff_role` | RS→PO resilience and likely basketball role signal | observed PO evidence + role classification |
| `league_transfer` | Projection into another league/competition | real historical cross-league transitions, shrunk toward league-quality prior |
| `player_pair` | Two-player/archetype complementarity | analytical basketball fit |
| `lineup_fit` | Pairwise complementarity across a proposed group | analytical basketball fit |
| `style_change` | Counterfactual pace/3P rate/creation/usage/team efficiency | strict ensemble with a synthetic bounded team context when supported |
| `player_role_change` | Counterfactual minutes, usage or role | strict ensemble with a synthetic bounded player context |
| `team_add_player` | Rotation change and estimated team impact after adding a player | player projection + empirical rotation→Net Rating relation |
| `team_replace_player` | Explicit incoming/outgoing player replacement | explicit rotation replacement + empirical team impact |
| `best_team_fit` | Rank target-league teams for a player | scenario ranking |
| `best_player_fit` | Rank players for a target team | strict ensemble ranking in supported context |
| `player_similarity` | Find statistically similar player profiles | standardised profile distance |
| `age_trajectory` | Age/peak trajectory | strict model + age curve; long-horizon caveat |

Competition labels are open-ended and use the same normalisation contract as training (`RS`, `PO`,
`CUP`, `SUPERCUP`, `TOT`, `FINAL_FOUR`, or newly observed labels). Direct strict-model inference only
serves competitions present in the promoted model vocabulary. Descriptive and empirical scenario
methods can use newly available competition data without pretending it was part of a prior model run.

## Evidence semantics

Every scenario response contains:

- `result`: basketball result only; no database identifiers are required by Chat V3.
- `evidence`: counts/context used to produce the result.
- `support.method`: identifies the methodology.
- `support.samples`: direct support count when meaningful.
- `support.confidence`: qualitative support tier, not a calibrated probability.
- `limitations`: material caveats that the answer layer should mention when relevant.

Methods beginning with `strict_supervised_` are promoted-model outputs. `observed_data` is historical
fact. `empirical_` uses historical population transitions/relationships. `analytical_` is a transparent
basketball fit calculation and must not be described as a calibrated ML probability.

## Cross-league behavior

A player does not need prior target-league history to answer questions such as "what could he do in
EuroLeague?". The transfer layer searches real consecutive-season transitions from the requested
source league/competition into the target league/competition. The robust median observed delta is
shrunk toward a league-quality prior when support is small. If no matching transitions exist, the
response explicitly reports `league_quality_fallback` and low support rather than silently reusing a
same-league model.

## Style counterfactuals

Numeric overrides are applied exactly within hard basketball bounds. Qualitative `higher`/`lower`
overrides are converted to one standard-deviation move calculated from teams in the same
league/competition/season, with a documented fallback step if that distribution is unavailable. Chat V3
receives the applied number and must not invent its own pace/usage change.

## Privacy boundary

WordPress resolves `hm_` / `hmg_` IDs to database `IdGlobal` only inside the authenticated
server-to-server client. Scenario requests can include multiple players/teams, but real global IDs are
never exposed to the browser or conversational context. Responses are recursively scrubbed of internal
and global identifiers before Chat V3 receives them.

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
| `team_add_player` | What-if acquisition into a target team's rotation | walk-forward-validated historical roster-transition model when validation passes |
| `team_replace_player` | Explicit incoming/outgoing player replacement | same validated acquisition-impact model with an explicit outgoing player |
| `best_team_fit` | Rank target-league teams for a player | scenario ranking |
| `best_player_fit` | Rank players for a target team | strict ensemble ranking in supported context |
| `player_scouting` | Filter and rank candidates by position, nationality, age and archetype; when a team is supplied, assess roster-aware fit | individual percentiles + analytical pairwise roster complementarity, roster-need alignment and indicative positional depth; not a calibrated prediction |
| `player_similarity` | Find statistically similar player profiles | standardised profile distance |
| `age_trajectory` | Age/peak trajectory | strict model + age curve; long-horizon caveat |


### Roster-aware Team Fit in Player Scouting

When player_scouting receives a target team, the engine no longer treats that team as context alone. It attempts to reconstruct the requested season roster from team_player_relations and combines it with player statistics from the exact league, season and competition. If the season relation is missing, observed player-stat rows explicitly attached to the target team are used as a declared fallback.

Each eligible candidate is compared analytically with every roster member who has matching-context statistics, excluding the candidate's own canonical identity when already on the roster. The score combines individual archetype fit (40%), average candidate-to-roster complementarity (30%), alignment with the roster's weaker archetype profiles (20%) and an indicative positional-depth signal (10% when position data is available). Components with missing evidence are omitted and the remaining weights are renormalized. The reported component scores and pair counts make the calculation inspectable.

Roster needs are relative to the same league/season/competition statistical pool; positional depth uses a transparent target of two listed players per main position. These are analytical heuristics, not learned coefficients, calibrated probabilities, causal impact estimates or a coach-specific rotation model. The result reports roster size separately from members with usable statistics and exposes statistics coverage. A full `available` Team Fit requires a season-specific roster relation and usable statistics for every resolved roster member. If the membership relation is absent, or some roster members lack matching-context statistics, the result is explicitly `partial`; if fewer than two members have usable statistics, it is `unavailable`. Partial scores can rank candidates provisionally but must never be described as a complete roster fit.

Competition labels are open-ended and use the same normalisation contract as training (`RS`, `PO`,
`CUP`, `SUPERCUP`, `TOT`, `FINAL_FOUR`, or newly observed labels). Direct strict-model inference only
serves competitions present in the promoted model vocabulary. Descriptive and empirical scenario
methods can use newly available competition data without pretending it was part of a prior model run.


## Acquisition impact model

The `team_add_player` and `team_replace_player` scenarios estimate next-season
team Net Rating change from historical team-season roster transitions. For each
training example, roster-quality and turnover features are built from membership
changes plus player ratings available in the prior season; the target is the
observed Net Rating change in the following season. The model uses ridge
regression with an expanding-window walk-forward evaluation, and reports OOS
RMSE, MAE, bias, sample counts, and empirical residual intervals.

The numeric acquisition estimate is **fail-closed**: the model must have at
least 40 complete transitions, at least 20 out-of-time predictions spanning at
least three held-out seasons, and beat a zero-change baseline on walk-forward
RMSE without excessive OOS bias. Otherwise the scenario returns
`acquisition_impact_status` and validation diagnostics but leaves numeric
acquisition-impact fields null. This status is evaluated from the data loaded
for the requested league, competition, and source-season cutoff.

The response separates the estimated incremental Net Rating association from
the predicted team Net Rating change with/without the hypothetical acquisition.
The 80% outcome interval is derived from walk-forward residual quantiles; the
90% interval around the incremental coefficient effect expresses model
coefficient uncertainty. Neither interval makes the estimate causal. Observed
team changes also reflect coaching, injuries, minutes, player availability,
schedule, and other transactions, so the output must be described as a
historically validated observational forecast—not as the real causal effect
of a single signing. The estimator does not infer salary, contract status,
promised minutes, or locker-room fit.

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


## Player Scouting filters and Team Fit

The `player_scouting` scenario accepts `parameters.archetype` (`playmaker`,
`scorer`, `connector`, `defender`, `big`, or `balanced`) and optional strict
filters: `role` (position code), `nationality`, `min_age`, `max_age`,
`min_games`, and `min_minutes_per_game`. The request must identify a source or
target league so candidates are compared inside one league/season/competition
cohort. Missing nationality or age excludes a candidate when that filter is
requested.

The individual `scouting_fit_score` is a transparent weighted average of
within-cohort metric percentiles, distinct from BB-Rating, Prediction Model,
and Future Performance. When a target team is supplied, the engine also tries
to compute roster complementarity, archetype needs, and positional depth.
Inspect `result.team_fit.status`: `available` means the resolved season roster
has full matching-context statistical coverage; `partial` means membership
or player evidence is incomplete; `unavailable` means there are too few
usable roster profiles to score complementarity. Only `available` should be
described as a full roster-aware fit. All versions are analytical heuristics,
not a calibrated probability, causal estimate, or coach-specific rotation model.

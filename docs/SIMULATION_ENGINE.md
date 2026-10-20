# Basketball Simulation & Causal Engine

The `/api/v2/scenarios/evaluate` endpoint supports probabilistic boxscores,
opponent and defensive matchups, play-type interactions, shot-profile
counterfactuals, lineup/roster optimization, composite scenarios and AIPW causal
estimates.

## Optional possession contract

The strict forecasting service depends only on the canonical physical tables
created by `ai_source_full.sql`. Simulation feeds and serving tables are owned by
`ai_scenario_serving.sql`. After both SQL contracts have been installed,
`RefreshContextSeasonal(season)` rebuilds Simulation inputs once per league-season
and then refreshes the indexed Scenario/Training aggregates. The process records
a per-league-season outcome in `AI_Source.SimulationSourceRefreshState`.

| Serving table | Required fields | Purpose |
|---|---|---|
| `scenario_defender_matchups` | Indexed player-pair/context aggregates | Defensive assignment evidence without loading raw PBP |
| `scenario_lineup_stats` | Indexed lineup/context aggregates | Partial-pooled lineup and matchup exposure |
| `scenario_play_type_stats` | Indexed player/team/context aggregates | Offensive strength × defensive weakness |
| `scenario_shot_profiles` | Indexed player/context aggregates | Shot-location counterfactuals |
| `simulation_causal_panel` | `treatment`, `next_outcome`, configured covariates | Explicit longitudinal AIPW panel |

All entity columns contain the same stable internal integer keys used by
`player_competition_stats` and `team_competition_stats`. They are not exposed to
Chat V3. WordPress sends only `hm_`/`hmg_` opaque identifiers and resolves
`IdGlobal` inside the authenticated server-to-server client; AI_Model resolves
those global identifiers to internal keys.

## Statistical behavior

- Boxscore simulations use a joint peer covariance matrix when enough
  competition-specific history exists and a shrunk parametric distribution
  otherwise. A system-entropy seed is generated for every unseeded request;
  callers can supply a seed for reproducible tests.
- Opponent scenarios condition one final distribution on pace, defense and
  play-type interactions.
- Defensive assignments are always returned as `inferred_not_observed` unless
  the upstream contract explicitly provides observed assignments. Exposure is
  weighted by assignment probability and shrunk toward a league prior.
- Lineup observations are partially pooled toward individual quality, spacing,
  defense and usage-balance priors. The optimizer exhaustively enumerates
  five-player combinations for pools up to 18.
- Shot transfers shrink efficiency in the destination zone; moving volume does
  not assume unchanged shot quality.
- Causal effects are returned only when treatment/control support and propensity
  overlap pass identification checks. Otherwise `causal_effect` is null and the
  response states the reason.

Every response carries `support.method`, sample counts and material
`limitations`; Chat V3 must preserve those distinctions.


## Extensible player archetypes

Player Scouting uses the catalog in `basketball_ai/scenarios/scouting_archetypes.py`.
The catalog defines overlapping statistical archetypes across creation, scoring,
shooting, defense, rebounding/interior play and two-way/utility roles. It is not a
closed set of mutually exclusive player types: a player can score highly in
several archetypes at once, and new definitions can be added in the catalog
without adding a branch to the scenario engine.

Each candidate returns up to eight `archetype_profiles`, each with a
league/season/competition-relative score, count of available metric evidence,
coverage and a qualitative evidence-confidence label. `primary_archetypes`
contains up to three profiles scoring at least 65/100. These values are
descriptive percentiles, not calibrated probabilities or causal conclusions.
Metrics absent from the source cohort are omitted and weights are renormalized;
the output must not be interpreted as evidence for an archetype whose underlying
metrics are unavailable. Traditional box-score proxies cannot fully identify
tactical roles such as screen-setting, off-ball movement, switch defense or
pick-and-roll coverage; those need play-by-play, tracking or lineup evidence
before being presented as directly measured.

 
## Source refresh and interpretation limits

`AI_Source.RefreshSimulationSources(league_key, season)` reads
`Pbp.{LEAGUE}` filtered by `Season`, canonical player/team identity mappings and, when
available, `Analisi.AdvancedStats_Lineups_Quarter_{LEAGUE}`. The seasonal refresh
calls it before producing `Scenario*` tables. Inspect
`AI_Source.SimulationSourceRefreshState` to distinguish `available`,
`partial` and `unavailable`; a successful procedure call alone does not mean
every optional feed has data.

- Defensive player matchups are probabilistic exposures derived from the opposing
  five-player lineup unless the source explicitly provides an observed defender
  assignment. They are not direct tracking labels.
- When PBP has no shot coordinates, shot profiles use `unknown_two_point` and
  `unknown_three_point`. No specific shot location is inferred.
- Tactical play type statistics are populated only when the source contains a
  separate, recognized tactical classification. PBP event codes such as
  `2FGM`, `3FGA` and `TOV` are not themselves play-type labels, so this feed
  can legitimately remain empty and the state be `partial`.
- `SimulationCausalPanel` is schema-only until the repository defines and
  validates a causal treatment/control design. Do not populate it with simple
  pre/post deltas.
- Clutch totals do not rely on Simulation: they are sourced by
  `ai_source_full.sql` from `Analisi.AdvancedStats_Clutch_{LEAGUE}` into
  `AI_Source.PlayerCompetitionStats`.

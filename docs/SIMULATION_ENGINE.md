# Basketball Simulation & Causal Engine

The `/api/v2/scenarios/evaluate` endpoint supports probabilistic boxscores,
opponent and defensive matchups, play-type interactions, shot-profile
counterfactuals, lineup/roster optimization, composite scenarios and AIPW causal
estimates.

## Optional possession contract

The strict forecasting service still starts when only the competition-aware
core views exist. The following `ai_source` views are additive and loaded when
available:

| View | Required fields | Purpose |
|---|---|---|
| `simulation_pbp_events` | `season`, `competition`, `offensive_player_id`; optional `defender_id`, `assignment_probability`, `points`, `turnover` | Event evidence for defensive assignments |
| `simulation_lineup_stints` | `player_ids`, `possessions`, `net_rtg`; optional `offense_player_id`, `defense_player_id`, `assignment_probability`, `points_allowed`, `turnovers_forced` | Partial-pooled lineup and matchup exposure |
| `simulation_play_type_stats` | `season`, `competition`, `play_type`, `possessions`; `player_id` or `team_id`, plus `ppp` or `ppp_allowed` | Offensive strength × defensive weakness |
| `simulation_shot_profiles` | `player_id`, `zone`, `attempts`, `fg_pct` | Shot-location counterfactuals |
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

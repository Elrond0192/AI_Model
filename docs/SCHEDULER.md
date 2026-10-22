# Recurring scheduler

The AI_Model Admin Control Center includes a PostgreSQL-backed recurring scheduler.
It executes only the allow-listed task types below; it is not a general SQL/job
runner. Scheduling a task does not mean its result is scientifically validated
or that a model is automatically promoted.

## Scheduling modes

Each scheduled job can use one of two modes:

- **Every X hours** — executes at an interval measured from the scheduled
  occurrence. Existing jobs keep this mode when the schema is upgraded.
- **Daily at a time** — executes once per local calendar day at the selected
  time, using the `Europe/Rome` timezone. The schedule follows Italian daylight
  saving time; `tzdata` is installed explicitly for containers without system
  timezone data.

The optional **start date** and **end date** are selected with calendar controls.
Both boundaries are inclusive in the `Europe/Rome` calendar. When no start date
is provided, the schedule starts from its next calculated occurrence. When no
end date is provided, it has no date-based expiry. Once the end date has passed,
the scheduler disables the job. If there are no future occurrences inside the
selected date window, the API rejects creation or reactivation.

The activation switch controls whether a new schedule is enabled immediately.
Each saved schedule has its own switch in the list; switching it off pauses future
dispatches, and switching it on recalculates the next occurrence. Deleting a job
removes its schedule, not its historical run records.

## Supported operations

| Task in the UI | Key | Parameters / behavior |
|---|---|---|
| `AI_Source.RefreshContextSeasonal` | `refresh_context_seasonal` | Requires `season`; discovers all canonical league-season-competition contexts for that season and refreshes Simulation, Training, Scenario and `ServingRefreshState`. |
| `AI_Source.RefreshSimulationSources` | `refresh_simulation_sources` | Requires `season` and one or more leagues (or `Tutte`); refreshes Simulation source feeds once per selected league-season. |
| `AI_Source.RefreshScenarioServing` | `refresh_scenario_serving` | Requires `season` and one or more leagues (or `Tutte`); refreshes all canonical competitions found for those leagues. |
| `AI_Source.RefreshTrainingServing` | `refresh_training_serving` | Requires `season` and one or more leagues (or `Tutte`); refreshes Training serving for all available competitions in those leagues. |
| BB Prediction Model training | `bb_prediction_training` | Submits the existing training pipeline; does not itself promote a candidate. |
| BB-Rating calibration | `bb_rating_calibration` | Submits a calibration job; does not itself promote the resulting candidate. |
| Future Performance training | `future_performance_training` | Submits a training job; does not itself promote a candidate. |

For routine post-ETL context refreshes, prefer a single
`refresh_context_seasonal` job for the season that receives new data. It already
calls the lower-level Simulation, Training and Scenario refresh procedures in
the correct sequence. Do not separately schedule those three lower-level refreshes
for the same data window unless there is a specific operational reason.

The causal-panel buttons on **Model Releases** are explicit manual maintenance
operations and are **not** keys in the recurring scheduler. They refresh the
team-switch or teammate-exposure observational cohorts across the canonical
league/competition contexts with completed adjacent seasons. They do not train or
promote models and are not triggered by chat.

## Suggested daily data-refresh workflow

1. Finish the upstream BBallstat/HoopmetricsEngine ETL and commit updated
   statistics and source tables.
2. Load/rebuild the canonical `AI_Source` source contract if that ETL does not
   already update it.
3. Run `RefreshContextSeasonal(season)` after the canonical player and team
   competition tables have the new rows. In the scheduler, choose the daily mode
   and a time after the upstream ETL is expected to finish (for example, 02:00
   Europe/Rome only if that is after the real ETL completion time).
4. Review **Pianificazioni ricorrenti** and **Esecuzioni recenti**. A failed or
   partial refresh must be investigated; a completed scheduler task is not proof
   that every optional PBP/tactical source contained data.
5. Refresh model candidates only when the new data warrants training and follow
   their validation/promotion gates separately. A context refresh never implies
   training, calibration, or promotion.

Do not create a nightly training job merely because new current-season statistics
arrive. In-progress seasons must remain excluded from completed-season training
targets where required by the model contract.

## Database and runtime behavior

The scheduler stores persistent jobs in
`"AI_Source"."AIModelScheduledJob"` and run history in
`"AI_Source"."AIModelJobRun"`. Its schema is defined in
`basketball_ai/data/ai_scenario_serving.sql`. The SQL file creates the columns
for new installations and includes an idempotent `ALTER TABLE ... ADD COLUMN IF
NOT EXISTS` migration. The admin process does not perform schema DDL as the runtime role. After a release
that changes this contract, reapply `basketball_ai/data/ai_scenario_serving.sql`
as the PostgreSQL schema/table owner (normally `hm_admin`) so its idempotent
migration adds `schedule_mode`, `daily_time`, `timezone_name`, `start_date`
and `end_date` without deleting existing jobs. The same script grants
`ai_model`, `hm_admin`, `hm_local_workers` and the configured
`app.ai_source_read_role` (when supplied) `SELECT/INSERT/UPDATE/DELETE` on the
two scheduler metadata tables and the guarded `SimulationCausalPanel`, plus
usage on the scheduler sequences. It does not grant write access to canonical
player/team statistics.
The runtime preflights `PlayerCompetitionStats` and both scheduler tables and
returns HTTP 503 with the PostgreSQL detail if the schema or grants are missing.

The worker polls at a short interval and executes due jobs one at a time per
worker process, using PostgreSQL row locking to avoid selecting the same job
twice. Failed executions are written to run history. The `refresh_context_seasonal`
procedure is one SQL transaction; league-by-league refresh jobs record which
contexts succeeded and which failed.

## Operational caveats

- `Tutte` for a league selector means the leagues currently discoverable from
  canonical source data at execution time, not a frozen static league list.
- Scenario/Training jobs use competitions currently available in canonical source
  data for the selected leagues. The scheduler does not invent missing contexts.
- PBP and tactical sources can legitimately be empty or unavailable. Inspect
  `"AI_Source"."SimulationSourceRefreshState"` and the serving refresh status
  tables rather than judging success from job status alone.
- The scheduler stores timestamps in PostgreSQL; daily schedule interpretation
  and the UI display use `Europe/Rome`.
- The scheduler does not retrain, recalibrate, promote, or restart production
  automatically after a refresh. Use the relevant model lifecycle procedure and
  gates when a new release is warranted.
- This documentation describes the repository contract. After deploying code
  changes, rebuild/restart the `admin` container so the new scheduler code and
  static UI are active, then verify `/healthz` and the next-run timestamp.

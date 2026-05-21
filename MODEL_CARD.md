# Model Card – Basketball Player Performance AI

## 1. Model Details

| Field              | Value                                                                     |
|--------------------|---------------------------------------------------------------------------|
| **Name**           | Basketball Player Performance Ensemble                                    |
| **Version**        | 2.0.0                                                                     |
| **Type**           | XGBoost regressor + k-NN compatibility model                              |
| **Output**         | Player performance rating (continuous, 0–10 scale)                        |
| **Trained on**     | Per-player seasonal stats (regular season, playoff, cup, supercup)        |
| **Framework**      | XGBoost 2.x, scikit-learn 1.3+                                           |
| **Created**        | 2024                                                                      |
| **Contact**        | Repository owner (@Elrond0192)                                           |

---

## 2. Intended Use

### Primary Use Cases
- **Transfer impact analysis**: estimate a player's expected rating in a target team and league.
- **Career trajectory forecasting**: project future ratings by applying age-curve adjustments.
- **Lineup optimisation**: rank candidate players by fit with a given team's playing style.
- **Scouting**: identify undervalued players in lower-tier leagues relative to projected ratings.

### Out-of-Scope Uses
- Medical or contractual decisions in isolation (the model does not capture injury history,
  mental health, or contractual disputes).
- Player comparisons across eras not represented in training data.
- Use outside of basketball (the feature space and ratings are basketball-specific).

---

## 3. Training Data

| Aspect                   | Description                                                               |
|--------------------------|---------------------------------------------------------------------------|
| **Source type**          | Synthetic (generated) or real data loaded from Azure SQL Server           |
| **Real DB schema**       | Multi-schema SQL Server: `Analisi`, `Boxscore`, `Anagrafiche`             |
| **Leagues covered**      | Up to 20 leagues (NBA, EuroLeague, BCL, ACB, Bundesliga, Lega Basket, …) |
| **Seasons**              | Multiple seasons (typically 2019-20 through 2023-24)                      |
| **Competition types**    | Regular Season (RS), Playoffs (PO), Cup (CUP), Super Cup (SUPERCUP)       |
| **Players**              | 5,000+ player-seasons in synthetic mode; real data varies by deployment   |
| **Train/Val/Test split** | 70 % train / 15 % validation (early stopping) / 15 % conformal holdout   |

### Feature Categories (57 total)
- Identity/context: age, position, competition type
- DB role labels: `ruolo_combinato`, `ruolo_offensivo`, `ruolo_difensivo`
- Per-36 stats: points, assists, rebounds, steals, blocks
- Advanced metrics: PER, TS%, USG%, BPM, OBPM, DBPM, RAPTOR, LEBRON, OWS, DWS, FIC, SPM
- Career signals: form score, consistency, career trajectory, age vs. peak
- Efficiency trends: TS% slope, durability (league-normalised)
- Playoff context: `po_vs_rs_delta`, `po_games_played`, `has_po_history`
- Clutch performance: clutch TS%, clutch net rating
- On/Off differentials
- Interaction features: `obpm_x_usg`, `dbpm_x_reb`, `two_way_score`

---

## 4. Model Architecture

```
Input features (57+)
       │
       ▼
 StandardScaler
       │
       ▼
  XGBoost Regressor
  (n_estimators=300, max_depth=5, learning_rate=0.05)
       │
       ▼
  Base rating (0–10)
       │   ┌─────────────────┐
       ├───│ Age-curve ratio  │
       ├───│ Compat. (k-NN)  │
       └───│ League quality  │
           └─────────────────┘
                  │
                  ▼
         Final rating + PI
         (conformal intervals)
```

---

## 5. Evaluation Metrics

| Metric            | Typical value (synthetic data) |
|-------------------|--------------------------------|
| Val RMSE          | ~0.28–0.45                     |
| Val MAE           | ~0.22–0.36                     |
| Val R²            | ~0.92–0.97                     |
| CV RMSE (5-fold)  | ~0.30–0.48                     |

> **Note**: metrics on real-world data will differ. Calibrate with
> `--mode train` on your own data and inspect the metadata.json output.

---

## 6. Limitations

1. **Synthetic data**: the default generator produces plausible but artificial player
   statistics. Models trained solely on synthetic data may not generalise to real leagues
   without re-training on actual data.

2. **League coverage**: ratings are relative to the leagues represented in training data.
   A player from an unrepresented league will receive features that are out-of-distribution.

3. **No injury/availability data**: `durability_score` is a proxy (games played / league max),
   not a direct injury model.

4. **Static age curve**: the age-curve adjustment uses empirical peak-age priors. Individual
   players may peak earlier or later.

5. **Historical data dependency**: players with fewer than 2 seasons of history receive less
   reliable predictions (form score, trajectory, and trend features fall back to defaults).

6. **Playoff delta (`po_vs_rs_delta`)**: if a player has no playoff history
   (`has_po_history = 0`), the delta is 0.0 by construction, not a measured value.
   The `has_po_history` flag disambiguates these two cases.

7. **Not causal**: the model is associative. A predicted rating increase from moving to a
   team does not account for causally unidentifiable factors (chemistry, coaching, etc.).

---

## 7. Bias and Fairness

- The model is trained on performance statistics, which may encode historical biases
  present in how basketball roles have been assigned (e.g. positional expectations).
- No explicit debiasing has been applied. Users deploying the model for player decisions
  should audit outputs across demographic groups.
- The `nationality` field is used only in the data generation process; it is not a model
  feature (no discriminatory nationality-based predictions).

---

## 8. Privacy and Data Governance

- When using the Azure SQL backend, all data access is through read-only parameterised
  queries via SQLAlchemy.
- No player PII is stored in model artefacts (`.joblib` files contain only numeric weights
  and scaler parameters).
- Connection strings and API keys must be stored in environment variables or `.env`, never
  in source code.

---

## 9. Citation and Versioning

If you use this model in a research or production context, please reference the repository
and version (`basketball-ai v2.0.0`).  Save the `metadata.json` output alongside any
exported predictions to ensure reproducibility.

---

## Limitations by League

| League Code | Notes                                                               |
|-------------|---------------------------------------------------------------------|
| ITA1        | Well-represented in training data; reliable predictions.            |
| GRC1        | Moderate sample size; accuracy slightly lower for young players.    |
| Other       | Predictions less reliable for leagues with < 2 seasons of history. |

## Known Biases by Role / Age

| Segment       | Known Issue                                                       |
|---------------|-------------------------------------------------------------------|
| Centers (C)   | WAR target underestimates defensive contribution.                 |
| Age ≤ 20      | High variance; age-curve prior dominates over sparse data.        |
| Age ≥ 35      | Retirement risk not modeled; ratings may be optimistic.           |
| PO vs RS      | Playoff performance predictor requires ≥ 5 PO games in history.  |

## Data Sources

| Source          | Description                                                      |
|-----------------|------------------------------------------------------------------|
| Boxscore.*      | Per-game stats (Pts, Reb, Ast, Fg2/3/Ft made/attempted, Min)     |
| Analisi.*       | Advanced stats (BPM, RAPTOR, LEBRON, WS, OWS/DWS, FIC, GmSc)   |
| Anagrafiche.*   | Player demographics (age, position, nationality, height, weight) |
| Pbp.*           | Play-by-play (shot coordinates, clutch stats, zone distribution) |

## Fairness Evaluation

RMSE is monitored per league and per role during walk-forward backtesting
(see `basketball_ai/models/backtest.py`). A promotion is blocked if the
per-segment gap exceeds the global RMSE by more than 20%.

## Privacy

Player PII (birth_date, weight_kg, height_cm, nationality) is masked in API
responses for roles below `analyst`. See `basketball_ai/api/middleware/pii.py`.

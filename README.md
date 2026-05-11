# AI_Model — Football Player Performance Prediction System

A production-quality AI system for predicting football (soccer) player performance, built with XGBoost, KNN compatibility modelling, and a FastAPI REST backend.

---

## Features

- **Synthetic data generator** — 20 leagues, 380 teams, 5 200 players, 10 seasons of stats
- **Feature engineering** — form score, consistency, per-90 metrics, career trajectory, age-vs-peak
- **XGBoost regressor** — predicts season rating from 15 engineered features; SHAP explanations included
- **KNN compatibility model** — scores a player's fit to a team's tactical style vector
- **Ensemble** — combines base rating, league-tier factor, and tactical context into a final prediction with confidence intervals
- **What-if scenario engine** — trajectory, transfer impact, peak prediction, best-team-fit, and more
- **FastAPI REST API** — full CRUD + prediction endpoints under `/api/v1`
- **77 pytest tests** — feature, model, and API coverage

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python main.py --mode generate-data

# 3. Train models
python main.py --mode train

# 4. Run demo (CLI predictions & scenarios)
python main.py --mode demo

# 5. Run tests
python -m pytest tests/ -v

# 6. Start REST API
python main.py --mode api        # http://127.0.0.1:8000
```

---

## Project structure

```
AI_Model/
├── main.py                     # CLI entry-point (generate-data | train | demo | api)
├── requirements.txt
├── data/sample/                # Generated CSVs (auto-created)
├── models_saved/               # Trained model files (auto-created)
├── notebooks/exploration.ipynb
├── src/
│   ├── data/
│   │   ├── generator.py        # Synthetic data generation
│   │   ├── loader.py           # CSV → DataFrames + lookup dicts
│   │   └── models.py           # Dataclasses: Player, Team, League …
│   ├── features/
│   │   ├── player_features.py  # form, consistency, per-90, trajectory
│   │   ├── team_features.py    # style vector, tier factor, similarity
│   │   └── context_features.py # position fit, style compat, adaptation
│   ├── models/
│   │   ├── age_curve.py        # Asymmetric Gaussian peak curve
│   │   ├── performance_model.py# XGBRegressor + StandardScaler
│   │   ├── compatibility_model.py # KNeighborsRegressor on style vectors
│   │   └── ensemble.py         # Combines all signals → PredictionResult
│   ├── scenarios/
│   │   └── engine.py           # WhatIfEngine: 8 scenario methods
│   └── api/
│       ├── main.py             # FastAPI app with lifespan
│       ├── schemas.py          # Pydantic v2 request/response models
│       └── routes/             # players | teams | predictions | scenarios
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_api.py
```

---

## REST API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/players` | List players (filter by position / age) |
| GET | `/api/v1/players/{id}` | Player profile |
| GET | `/api/v1/players/{id}/stats` | Season statistics |
| GET | `/api/v1/teams` | List teams |
| GET | `/api/v1/teams/{id}/analysis` | Team squad analysis |
| POST | `/api/v1/predictions/player/{id}/team/{tid}` | Predict rating |
| GET | `/api/v1/predictions/player/{id}/trajectory` | Age trajectory |
| GET | `/api/v1/predictions/player/{id}/peak` | Peak prediction |
| POST | `/api/v1/scenarios/what-if` | Custom what-if scenario |
| GET | `/api/v1/scenarios/best-teams/{id}` | Top team fits |
| GET | `/api/v1/scenarios/best-players/{tid}` | Top players for a team |
| POST | `/api/v1/scenarios/compare` | Compare multiple scenarios |
| POST | `/api/v1/scenarios/transfer-impact` | Transfer impact simulation |
| POST | `/api/v1/scenarios/what-if-teammates` | Teammate quality effect |

Interactive docs available at `http://127.0.0.1:8000/docs` once the API is running.

---

## Ensemble prediction formula

```
adjusted = base_rating × compat_mult × league_factor × ctx_mult
```

- **base_rating** — XGBoost prediction (captures age, position, per-90 stats)
- **compat_mult** — KNN compatibility score remapped to [0.90, 1.10]
- **league_factor** — tier-1 → 1.00, tier-5 → 0.86 (harder leagues are a bigger test)
- **ctx_mult** — tactical context (position fit × style × role × adaptation) → [0.95, 1.05]
- For **trajectory** projections: `base_rating` is additionally scaled by the age-curve ratio `target_af / current_af`

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list (XGBoost, scikit-learn, FastAPI, pandas, NumPy, SHAP, …)

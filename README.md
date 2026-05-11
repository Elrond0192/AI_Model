# AI_Model — Basketball Player Performance Prediction System

A professional-grade AI system for estimating basketball player performance
across leagues and teams, with full **What-If scenario analysis**.

Built with XGBoost, KNN compatibility modelling, and a FastAPI REST backend.

---

## Features

| Capability | Details |
|---|---|
| **Synthetic data generator** | 20 leagues (NBA, EuroLeague, ACB, Lega Basket, …), 300 teams, 5 000 players, 5 seasons of per-game & advanced stats |
| **Mixed / hybrid positions** | PG, SG, SF, PF, C, PG/SG, SG/SF, SF/PF, PF/C, SG/PF — full support |
| **Feature engineering** | Per-36 stats, PER, BPM, TS%, USG%, form score, consistency, age-vs-peak, versatility, playmaking & defensive scores |
| **XGBoost performance model** | 15-feature regressor; SHAP explanations available |
| **KNN compatibility model** | Scores player-team style fit using 6-d basketball style vectors |
| **Asymmetric age curves** | Position-specific peak ages with faster post-peak decline for guards vs bigs |
| **Ensemble prediction** | XGBoost base × age-curve ratio × style compatibility × league tier × context adjustments |
| **What-If engine** | Trajectory, transfer impact, peak prediction, best-team-fit, teammate quality scenarios |
| **FastAPI REST API** | Full CRUD + prediction endpoints under `/api/v1`, Pydantic v2 validated |
| **73 pytest tests** | Feature engineering, model, and API endpoint coverage |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic basketball dataset
python main.py --mode generate-data

# 3. Train models
python main.py --mode train

# 4. Run interactive demo
python main.py --mode demo

# 5. Start REST API server
python main.py --mode api
```

---

## REST API overview

Base URL: `http://localhost:8000/api/v1`

### Players
| Method | Endpoint | Description |
|---|---|---|
| GET | `/players` | List players (filter by position, nationality, age) |
| GET | `/players/{id}` | Get player details |
| GET | `/players/{id}/stats` | All season stats for a player |
| GET | `/players/{id}/profile` | Enriched profile (form, consistency, trajectory) |

### Teams
| Method | Endpoint | Description |
|---|---|---|
| GET | `/teams` | List teams (filter by league, tier, style) |
| GET | `/teams/{id}` | Get team details |
| GET | `/teams/{id}/roster` | Current roster |
| GET | `/teams/{id}/analysis` | Squad stats + style strengths |

### Predictions
| Method | Endpoint | Description |
|---|---|---|
| POST | `/predictions/player/{id}/team/{id}` | Predict player rating at team |
| GET | `/predictions/player/{id}/trajectory` | Age-trajectory projection |
| GET | `/predictions/player/{id}/peak` | Career peak prediction |

### What-If Scenarios
| Method | Endpoint | Description |
|---|---|---|
| POST | `/scenarios/what-if` | Predict performance in a specific team |
| GET | `/scenarios/best-teams/{player_id}` | Top-N teams for a player |
| GET | `/scenarios/best-players/{team_id}` | Top-N players for a team |
| POST | `/scenarios/compare` | Compare performance across multiple teams |
| POST | `/scenarios/transfer-impact` | Simulate transfer performance delta |
| POST | `/scenarios/what-if-teammates` | What if teammates had a different quality? |

---

## Architecture

```
src/
├── data/
│   ├── models.py          # Dataclasses: League, Team, Player, PlayerStats, ...
│   ├── generator.py       # Synthetic basketball data generator
│   └── loader.py          # CSV loaders + load_all_data() dict API
├── features/
│   ├── player_features.py # Per-36, PER, BPM, form, consistency, age-vs-peak
│   ├── team_features.py   # 6-d style vector, teammate quality
│   └── context_features.py# Position-style fit, role opportunity, league adaptation
├── models/
│   ├── age_curve.py       # Asymmetric Gaussian age curves per position
│   ├── performance_model.py # XGBoost rating predictor
│   ├── compatibility_model.py # KNN style compatibility
│   └── ensemble.py        # Prediction pipeline
├── scenarios/
│   └── engine.py          # WhatIfEngine: trajectory, transfer, peak, best-fit
├── api/
│   ├── main.py            # FastAPI app factory + lifespan
│   ├── schemas.py         # Pydantic v2 request/response models
│   └── routes/            # players, teams, predictions, scenarios
└── utils/
    └── helpers.py         # Formatting, position grouping, normalization
```

---

## Basketball positions supported

Pure: `PG · SG · SF · PF · C`  
Hybrid: `PG/SG · SG/SF · SF/PF · PF/C · SG/PF`

All positions have their own:
- Peak age (guards peak 24-26, wings 26, bigs 27-28)
- Age curve width (bigs develop & decline more slowly)
- Style-position compatibility matrix
- Per-36 stat generation distributions

---

## Playing styles

`pace_and_space · pick_and_roll · isolation · defensive · motion_offense · post_up`

Each style has a full compatibility matrix across all 10 positions.

---

## Running tests

```bash
pytest tests/ -q
```

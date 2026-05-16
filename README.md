# AI_Model — Basketball Player Performance Prediction System

A production-oriented basketball AI platform to estimate player performance across teams and leagues, simulate scenarios, and expose results via API and chat.

Built around an XGBoost-based ensemble, compatibility modelling, and a FastAPI backend.

---

## Features

| Capability | Details |
|---|---|
| **Package structure** | Codebase is now under the `basketball_ai` package namespace (legacy `src` layout removed) |
| **Synthetic data generator** | Multi-league dataset generation with players, teams, stats, and relations |
| **Advanced feature engineering** | Per-36/Per-40 stats, RAPTOR/LEBRON/SPM, OWS/DWS, FIC, clutch/on-off metrics, role signals, durability and interaction features |
| **Performance model** | XGBoost regressor with an expanded feature set (57 features in `FEATURE_COLS`) and optional SHAP explainability |
| **Competition-aware predictions** | Handles competition context (`RS`, `PO`, `CUP`, `SUPERCUP`) in training and inference |
| **Compatibility model** | Team-style compatibility via 6D style vectors with data-calibrated normalization bounds |
| **Ensemble prediction** | Combines performance model, age curve, style fit, league context, and scenario modifiers |
| **What-If engine** | Trajectory, transfer impact, peak, best-team/best-player fit, teammate quality, and lineup what-if analysis |
| **FastAPI REST API** | Endpoints under `/api/v1` for players, teams, predictions, scenarios, auth, chat, and WordPress export |
| **Chat interface** | Natural-language endpoint (`/api/v1/chat`) with intent detection and session continuity |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (optional) install package in editable mode
pip install -e .

# 3. Generate synthetic basketball dataset
python main.py --mode generate-data

# 4. Train models (CSV source by default)
python main.py --mode train

# 5. Run interactive demo
python main.py --mode demo

# 6. Start REST API server
python main.py --mode api
```

### CLI modes

```bash
python main.py --mode generate-data
python main.py --mode train [--source file|sql]
python main.py --mode demo [--source file|sql]
python main.py --mode api [--host 0.0.0.0] [--port 8000] [--source file|sql]
python main.py --mode export-wordpress [--source file|sql] [--out-dir wp_export]
```

---

## REST API overview

Base URL: `http://localhost:8000/api/v1`

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/token` | Obtain access + refresh tokens |
| POST | `/auth/refresh` | Refresh access token |

### Players
| Method | Endpoint | Description |
|---|---|---|
| GET | `/players` | List players |
| GET | `/players/{player_id}` | Get player details |
| GET | `/players/{player_id}/stats` | Season stats for player |
| GET | `/players/{player_id}/profile` | Enriched player profile |

### Teams
| Method | Endpoint | Description |
|---|---|---|
| GET | `/teams` | List teams |
| GET | `/teams/{team_id}` | Get team details |
| GET | `/teams/{team_id}/roster` | Team roster |
| GET | `/teams/{team_id}/analysis` | Team analysis |

### Predictions
| Method | Endpoint | Description |
|---|---|---|
| POST | `/predictions/player/{player_id}/team/{team_id}` | Predict rating in team context |
| GET | `/predictions/player/{player_id}/trajectory` | Age trajectory |
| GET | `/predictions/player/{player_id}/peak` | Peak projection |

### Scenarios
| Method | Endpoint | Description |
|---|---|---|
| POST | `/scenarios/what-if` | One-team scenario prediction |
| GET | `/scenarios/best-teams/{player_id}` | Best team fits for player |
| GET | `/scenarios/best-players/{team_id}` | Best player fits for team |
| POST | `/scenarios/compare` | Compare multiple team scenarios |
| POST | `/scenarios/transfer-impact` | Transfer impact simulation |
| POST | `/scenarios/what-if-teammates` | Teammate-quality what-if |
| POST | `/scenarios/what-if-lineup` | Custom lineup what-if |

### Chat & WordPress
| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Natural-language analysis endpoint |
| GET | `/wordpress/player-card/{player_id}` | WordPress-ready player card payload |

---

## Architecture

```
basketball_ai/
├── api/
│   ├── main.py            # FastAPI app and lifecycle
│   ├── schemas.py         # Pydantic request/response models
│   └── routes/            # auth, players, teams, predictions, scenarios, chat, wordpress
├── auth/                  # RBAC/auth helpers
├── chat/                  # Intent detection and chat engine
├── data/                  # Data models, generator, loaders, schema mapping
├── features/              # Player/team/context feature engineering
├── models/                # Performance, compatibility, age curves, ensemble
├── monitoring/            # Monitoring utilities
├── scenarios/             # What-if simulation engine
└── utils/                 # Shared helpers
```

---

## Basketball positions and styles

**Positions:** `PG · SG · SF · PF · C · PG/SG · SG/SF · SF/PF · PF/C · SG/PF`  
**Styles:** `pace_and_space · pick_and_roll · isolation · defensive · motion_offense · post_up`

Position-specific modeling includes:
- peak-age behavior by role archetype
- different development/decline curve widths by position family
- style-position compatibility matrices (including hybrid positions)
- per-minute distribution signals used in feature generation

---

## Running tests

```bash
pytest tests/ -q
```

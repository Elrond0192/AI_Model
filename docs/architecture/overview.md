# Architecture Overview

This document uses the C4 model notation to describe the Basketball Performance AI system.

## C1 – System Context

```
[Analyst / Scout]  ──→  [Basketball AI GUI (Streamlit)]
[External App]     ──→  [Basketball AI REST API (FastAPI)]
[Basketball AI]    ──→  [Azure SQL Server]  (real data backend)
[Basketball AI]    ──→  [CSV Files]         (synthetic / test backend)
```

## C2 – Container Diagram

| Container | Technology | Purpose |
|-----------|-----------|---------|
| GUI | Streamlit | Multi-user web interface; RBAC-gated tabs |
| REST API | FastAPI + Uvicorn | Programmatic access; WordPress integration |
| AI Engine | XGBoost + sklearn | Player rating prediction + What-If scenarios |
| Chat Engine | TF-IDF intent classifier | Natural language queries |
| Data Loader | pandas + pyodbc | Dual CSV / Azure SQL backend |
| Auth Service | PBKDF2/Argon2id + JWT | User management, sessions, RBAC |

## C3 – Component Diagram (AI Engine)

```
prepare_features()
   ↓
PerformanceModel (XGBoost)  ←→  StandardScaler
   ↓                              ↓
age_performance_factor()    compute_baselines()  [A4]
   ↓
CompatibilityModel (k-NN)
   ↓
league_quality_factor()
   ↓
conformal_prediction_interval()
   ↓
PredictionResult(rating, ci, shap_values)
```

## Key design decisions (ADR summary)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | XGBoost over neural net | Tabular data, interpretability, fast training |
| 2 | Conformal prediction for CI | Distribution-free, well-calibrated intervals |
| 3 | Streamlit for GUI | Rapid prototyping, Python-native, no frontend build |
| 4 | Dual CSV/SQL backend | Testability (CSV) + production (Azure SQL) |
| 5 | pydantic-settings for config | Single source of truth for all env vars |
| 6 | Intent handler registry | Extensible chat without modifying engine core |

# Prima Predizione in 5 Minuti

Una guida pratica per ottenere la tua prima predizione con **Basketball Performance AI** in meno di 5 minuti.

---

## Passo 1 – Installazione

```bash
# Clona il repository
git clone https://github.com/your-org/basketball-ai.git
cd basketball-ai

# Installa le dipendenze
pip install -e ".[dev]"
```

> **Requisiti**: Python 3.11–3.12, pip ≥ 23.

---

## Passo 2 – Genera i dati di esempio

```bash
python main.py --mode generate-data
```

Questo crea la directory `data/sample/` con file CSV sintetici:
- `players.csv` – 200 giocatori
- `teams.csv` – 40 squadre
- `player_stats.csv` – statistiche per stagione

---

## Passo 3 – Addestra il modello

```bash
python main.py --mode train
```

Il modello viene salvato in `models_saved/`. Il training richiede circa 30-60 secondi.

---

## Passo 4 – Prima predizione

### Via CLI

```bash
python main.py --mode predict --player-id 1 --team-id 1
```

Output di esempio:
```
Player 1 @ Team 1
Rating predetto: 7.42 / 10
Intervallo di confidenza: [6.98 – 7.86]
```

### Via Python

```python
from basketball_ai.data.loader import load_all_data
from basketball_ai.models.ensemble import EnsembleModel
from basketball_ai.scenarios.engine import WhatIfEngine

# Carica dati
data = load_all_data("data/sample")

# Carica modello pre-addestrato
ensemble = EnsembleModel()
ensemble.load("models_saved")

# Inizializza il motore di scenari
engine = WhatIfEngine(ensemble, data)

# Esegui predizione
result = engine.predict_in_team(player_id=1, team_id=1)
print(f"Rating: {result.predicted_rating:.2f}")
print(f"CI: [{result.confidence_low:.2f} – {result.confidence_high:.2f}]")
print(result.explanation)
```

---

## Passo 5 – Avvia la GUI

```bash
streamlit run gui/app.py
```

Apri il browser su [http://localhost:8501](http://localhost:8501).

1. Vai alla sezione **📂 Dati** e verifica che i dati siano caricati
2. Vai su **🎯 Predizioni**, seleziona un giocatore e una squadra
3. Clicca **Calcola predizione**

---

## Troubleshooting

| Problema | Causa probabile | Soluzione |
|----------|-----------------|-----------|
| `data not found` | Dati non generati | Esegui `python main.py --mode generate-data` |
| `Model not loaded` | Modello non addestrato | Esegui `python main.py --mode train` |
| `Player X not found` | ID non valido | Verifica gli ID in `data/sample/players.csv` |
| `ImportError: joblib` | Dipendenze mancanti | Esegui `pip install -e ".[dev]"` |
| Port 8501 già in uso | Un'altra istanza gira | Usa `streamlit run gui/app.py --server.port 8502` |
| `503 Model not loaded` (API) | Server avviato senza training | Avvia il server dopo il training |

---

## Prossimi passi

- Leggi [`docs/API.md`](API.md) per le API REST
- Esplora gli scenari What-If nella GUI
- Consulta [`CONTRIBUTING.md`](../CONTRIBUTING.md) per contribuire al progetto

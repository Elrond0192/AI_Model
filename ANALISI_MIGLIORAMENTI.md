# Analisi funzionale critica e piano di miglioramento — Basketball Performance AI

> Analisi condotta dal punto di vista combinato **UX Designer · ML/AI Engineer · Senior Python Engineer**.
> Approccio: mettere in dubbio le scelte fatte, evidenziare debito tecnico, proporre interventi concreti
> con **priorità (P0 critico → P3 nice-to-have)**, **effort indicativo (S/M/L)** e **impatto atteso**.
>
> Questo documento non descrive *cosa* fa già il sistema, ma *cosa non funziona*, *cosa è discutibile*
> e *cosa migliorerebbe in modo misurabile* la qualità del prodotto, del modello e dell'esperienza utente.

---

## 0. Sintesi esecutiva — i 10 punti più importanti

| #  | Area              | Problema                                                                                   | Priorità |
|----|-------------------|--------------------------------------------------------------------------------------------|----------|
| 1  | AI/ML             | Il modello è addestrato e validato su **dati sintetici**: nessuna prova di validità reale. | **P0**   |
| 2  | AI/ML             | Target `rating 0–10` è una **costruzione circolare**: PER/BPM → rating → si predice rating. | **P0**   |
| 3  | UX                | `gui/app.py` è un **monolite di ~3000 righe**, ingestibile e fragile.                       | **P0**   |
| 4  | Sicurezza         | Password admin temporanea **mostrata in pagina**; sessioni JSON su disco senza rotazione.  | **P0**   |
| 5  | Sicurezza         | `roles.json` e `users.json` su filesystem, nessun audit log, nessun lockout brute-force.   | **P1**   |
| 6  | Architettura      | `WhatIfEngine` (1053 LOC) e `chat/engine.py` (1091 LOC) sono **god-object**.               | **P1**   |
| 7  | AI/ML              | Nessun **MLOps**: no model registry, no versioning, no A/B, drift solo abbozzato.          | **P1**   |
| 8  | Chat / NLP        | Intent detection TF-IDF su esempi hard-coded multilingua: fragile, no fallback LLM.        | **P1**   |
| 9  | Test               | 73 test ma quasi tutti su dati sintetici → coverage di logica, non di correttezza modello. | **P1**   |
| 10 | Performance       | Predizioni single-threaded, no batching API, cache LRU in-process (non condivisa).         | **P2**   |

---

## 1. UX / Frontend (Streamlit GUI)

### 1.1 Cosa metto in dubbio

- **Perché Streamlit per un prodotto multi-utente con RBAC?** Streamlit nasce per prototipi mono-utente.
  Stiamo forzando autenticazione, cookie session, RBAC, multi-tab e cache in uno strumento che ri-esegue
  tutto lo script ad ogni interazione. Ogni `st.rerun()` è un costo nascosto.
- **8 tab in un'unica pagina** (`Dati`, `Training`, `Predizioni`, `Scenari`, `Chat`, `Scouting AI`,
  `Mapping`, `Utenti`) → l'utente non ha un *journey*, ha un cruscotto da pilota di Boeing.
- **Lingua mista IT/EN** (etichette in italiano, log e messaggi tecnici in inglese): incoerenza
  cognitiva, peggiora accessibilità e traduzione.
- **Nessun design system**: emoji come icone, spacing inconsistente, niente theming, niente dark mode,
  niente componenti riusabili.
- **Form lunghi senza progressive disclosure**: l'utente deve scorrere centinaia di righe per arrivare
  al bottone "Esegui". Non c'è onboarding, non c'è empty state utile.
- **Nessuna gestione degli errori user-facing**: la maggior parte dei `try/except` ingoia l'errore
  silenziosamente o stampa stacktrace tecnici.
- **Nessuna telemetria UX**: non sappiamo quali funzioni usano gli utenti, dove abbandonano, quanto
  tempo ci mettono. Stiamo migliorando alla cieca.

### 1.2 Piano di miglioramento — UX

| #  | Intervento                                                                                                | Prio | Effort | Impatto |
|----|----------------------------------------------------------------------------------------------------------|------|--------|---------|
| U1 | **Split `gui/app.py` (~3000 LOC)** in moduli per tab + componenti condivisi (`gui/tabs/`, `gui/components/`). | P0   | M      | Alto    |
| U2 | Introdurre un **design system minimo**: token CSS (colori, spacing, tipografia), un `theme.toml`, palette per stati (success/warn/error). | P1 | S | Medio |
| U3 | **Unificare la lingua** (italiano per tutta la UI, inglese per i log) e introdurre file di traduzione (`i18n/it.json`, `i18n/en.json`). | P1 | M | Medio |
| U4 | **Empty state e onboarding**: nuovo utente vede una guida step-by-step (Carica dati → Allena → Predici). | P1 | S | Alto    |
| U5 | **Skeleton loader e spinner contestuali** invece di blocchi globali; mostrare progress bar reale durante training. | P2 | S | Medio |
| U6 | Sostituire i 5 `st.file_uploader` separati con **un singolo drop-zone** che accetta uno zip o auto-rileva i nomi. | P2 | S | Medio |
| U7 | Introdurre **shortcut da tastiera** e ricerca globale (Cmd+K) per giocatore/squadra. | P3 | M | Medio |
| U8 | Aggiungere **dark mode** e contrasto AA (WCAG 2.1) — attualmente non testato. | P2 | S | Medio |
| U9 | **Valutare migrazione a Next.js + FastAPI** per la versione 3.0 (Streamlit resta per il pannello "Analyst", il pubblico vede una SPA). | P2 | L | Molto alto |
| U10 | Standardizzare gli **error boundary**: ogni tab cattura le sue eccezioni e mostra messaggi UX (`st.error("Dati non disponibili: …")`) invece di stacktrace. | P0 | S | Alto |
| U11 | **Auto-save dello stato dei form** in `st.session_state` con TTL, così un reload non azzera 10 minuti di configurazione. | P2 | M | Medio |
| U12 | **Visualizzazioni**: oggi si usa solo `st.dataframe`. Aggiungere grafici (radar style-fit, trajectory chart con CI, heatmap compatibilità) usando `plotly` o `altair`. | P1 | M | Alto |
| U13 | **Telemetria UX**: instrumentare con un semplice logger (anonimo) di eventi (`event=tab_change`, `event=predict_run`) verso file o PostHog self-hosted. | P2 | S | Alto |

---

## 2. Modello AI / Machine Learning

### 2.1 Cosa metto in dubbio (sezione critica)

#### Problemi metodologici fondamentali
- **Target circolare**: la `rating` è calcolata in `generator.py` come funzione lineare di
  PER, BPM, WS, modulata da `age_factor`. Poi il modello impara a predire `rating` usando in input
  PER, BPM, WS e *57 feature derivate*. Il modello sta apprendendo una funzione che già esiste in
  forma chiusa → R² altissimi *che non significano nulla* sulla realtà.
- **Dati sintetici come unico training set**: tutta la pipeline è ottimizzata e valutata su dati
  generati da `generator.py`. Non c'è alcuna evidenza che i pesi appresi generalizzino su dati NBA
  reali, EuroLeague reale, o sulle leghe SQL Azure target.
- **57 feature, ~poche migliaia di sample**: rapporto feature/sample sospetto. Rischio overfit
  altissimo, regolarizzazione di XGBoost non basta. Non c'è feature selection sistematica
  (es. SHAP-based pruning, mutual information).
- **Nessuna baseline**: non confrontiamo il modello con (a) media stagionale, (b) regressione lineare,
  (c) modello ad albero singolo. Senza baseline non sappiamo se XGBoost stia davvero aggiungendo valore.
- **Cross-validation assente**: si usa `train_test_split` + un holdout conformal. Non c'è
  k-fold stratificato per posizione/lega, né time-based CV (essenziale per dati stagionali).
- **Leakage temporale potenziale**: split casuale su `player_stats` mischia stagioni diverse dello
  stesso giocatore. Modello vede 2023 e predice 2024 dello stesso atleta → leakage.
- **`age_factor` hard-coded a curva quadratica**: la PEAK_AGES dict è una scelta arbitraria
  (PG:26, C:28...). Letteratura (Silver, Berri, Hollinger) suggerisce curve diverse per skill type
  (scoring picca prima, rim protection dopo). Andrebbe stimato da dati, non hard-coded.
- **Compatibility model = 6D style vector con norm bounds calibrati**: non è chiaro se i 6 stili
  (`pace_and_space`, `pick_and_roll`, ...) siano effettivamente ortogonali o ridondanti.
  Manca un'analisi PCA / fattoriale che giustifichi la scelta.

#### Problemi di confidenza e calibrazione
- **Conformal prediction implementato ma su residui di poche centinaia di sample** → intervalli
  larghi e pessimisticamente stabili. Da rivalutare con normalized conformal (es. Mondrian).
- **Nessuna calibrazione di probabilità** sui modelli secondari (style-fit, role-fit) che
  producono "score" sotto forma di moltiplicatori senza unità statistica.

#### Problemi di explainability
- SHAP è opzionale (`try: import shap`). Per un sistema che propone "best team fit" e "transfer
  impact" a un utente, l'explainability *non può essere opzionale*: è un requisito legale (AI Act EU)
  e di fiducia.
- Nessuna spiegazione contro-fattuale ("se il giocatore avesse +1 di TS%, il rating salirebbe a X").

#### Problemi di feature engineering
- **`form_score`, `consistency_score`, `career_trajectory`** sono indicatori derivati ma la loro
  formula non è documentata in modo riproducibile (è dentro `player_features.py`).
- **`durability_score = games_played / 82`**: ma le leghe europee giocano 30-40 partite. La
  normalizzazione "82" è NBA-centric. → bug funzionale.
- **`po_vs_rs_delta`**: se un giocatore non gioca i playoff, il feature è 0 → indistinguibile da
  "gioca uguale in PO e RS". Bisognerebbe avere un flag separato `has_po_history`.

### 2.2 Piano di miglioramento — AI/ML

| #   | Intervento                                                                                                                                          | Prio | Effort | Impatto    |
|-----|-----------------------------------------------------------------------------------------------------------------------------------------------------|------|--------|------------|
| A1  | **Validare su dati reali**: pipeline ETL da fonti pubbliche (basketball-reference, hashtag basketball, EuroLeague API). Misurare RMSE/MAE su holdout reale 2023-24. | P0 | L | Critico |
| A2  | **Eliminare la circolarità del target**: scegliere un target indipendente (es. `WAR` calcolato post-hoc, oppure `next-season VORP`) e ri-allenare. | P0 | L | Critico |
| A3  | **Time-based split obbligatorio**: train su stagioni ≤ N-2, validation N-1, test N. Sostituire `train_test_split` casuale. | P0 | S | Alto |
| A4  | **Baseline obbligatorie**: aggiungere `BaselineMeanModel`, `LinearRegression`, `RandomForest` con stesse feature; loggare delta vs XGBoost in ogni training. | P1 | S | Alto |
| A5  | **Feature pruning sistematico**: SHAP global importance + correlation matrix; ridurre 57 → 25-30 feature più informative. | P1 | M | Alto |
| A6  | **K-fold stratificato per posizione e lega** in addition al time-split per stime di varianza dei risultati. | P1 | S | Medio |
| A7  | **Stimare le peak ages dai dati** invece di hard-coding (`POSITIONAL_PEAK_AGES`): fit di una loess/polinomiale per posizione e per skill cluster. | P1 | M | Alto |
| A8  | **Analisi fattoriale sugli style vector**: PCA o NMF sui dati di team-play-by-play per verificare che i 6 stili siano interpretabili e non ridondanti. | P2 | M | Medio |
| A9  | **Explainability come first-class**: rendere SHAP non opzionale, aggiungere endpoint `/predictions/.../explain` che restituisce top-5 fattori per ogni predizione. | P0 | M | Alto |
| A10 | **Counterfactuals**: integrare DiCE o uno script custom per "what-if su singola feature" (es. "se TS% salisse del 3%, rating predetto = …"). | P2 | M | Medio |
| A11 | **Calibrare conformal su residui per sotto-gruppo** (Mondrian: per posizione e per lega) — intervalli più stretti dove c'è densità. | P2 | M | Medio |
| A12 | **`durability_score` league-aware**: usare il numero massimo di partite della lega del giocatore, non 82 fisso. **Bug funzionale**. | P0 | XS | Medio |
| A13 | **`po_vs_rs_delta`**: aggiungere `has_po_history: bool` come feature separata. | P1 | XS | Basso |
| A14 | **Hyper-parameter tuning sistematico**: oggi gli iperparametri XGBoost sembrano hard-coded; introdurre Optuna o `GridSearchCV` con CV time-based. | P1 | M | Medio |
| A15 | **Ensembling reale**: oggi `EnsembleModel` orchestra moltiplicatori, non vere predizioni. Aggiungere stacking con un meta-learner. | P2 | L | Medio |
| A16 | **Quantile regression** per i bordi della CI invece di +/- residui conformali su MAE: ottiene CI eteroschedastiche corrette. | P2 | M | Medio |
| A17 | **Salvare le firme del dataset** (hash + schema) insieme al modello (joblib), in modo da rifiutare predizioni su dati incompatibili. | P1 | S | Alto |
| A18 | **Bayesian / probabilistic head** opzionale (NGBoost / pymc) per produrre distribuzioni complete, non solo CI 90%. | P3 | L | Medio |

---

## 3. Architettura Python e qualità del codice

### 3.1 Cosa metto in dubbio

- **`basketball_ai/scenarios/engine.py` (1053 LOC)** e **`basketball_ai/chat/engine.py` (1091 LOC)**
  sono *god object*: troppi responsabilità, difficili da testare in isolamento.
- **`basketball_ai/data/sql_loader.py` (802 LOC) + `schema_mapping.py` (801 LOC)**: la logica di
  auto-discovery dello schema è "magica" — vantaggio operativo, ma ipotesi implicite che rompono in
  silenzio se i nomi tabella cambiano leggermente.
- **`from __future__ import annotations`** ovunque ma type hints **non controllati**: nessun mypy/pyright
  in CI. Le annotazioni servono solo per leggere.
- **Pydantic v2 + ConfigDict da v1**: alcune sintassi miste (`model_config = ConfigDict(...)`) sono ok
  ma altri schemi non sfruttano i validator v2.
- **`logging.basicConfig` chiamato in `lifespan`** (api/main.py): se l'app è importata altrove
  (test, gunicorn) la config logging viene ricalcolata o ignorata.
- **`global` mutable `app_state: Dict[str, Any]`** nel modulo `api/main.py`: anti-pattern, rende
  l'app non thread-safe e non isolabile nei test.
- **Path traversal mitigation** in `gui/app.py` (`_safe_model_dir`) è reinventata; meglio
  `Path.resolve().is_relative_to(...)`.
- **Pin di dipendenze troppo lassi** (`>=`): non c'è `lock file` (poetry/pip-tools). Build non riproducibile.
- **`random.seed(42)` + `np.random.seed(42)` come side-effect a livello modulo** in `generator.py`:
  modifica il seed globale di ogni programma che importa il modulo. Bug latente.
- **`sys.path.insert(0, ...)` nel GUI**: hack per importare il package. Va sostituito con install
  `pip install -e .` (già presente) e import puliti.
- **Mix `src/` vs `basketball_ai/` nei docstring e nelle memorie**: il legacy `src/` è stato rimosso
  ma alcuni commenti citano ancora `src.chat.session` (es. `basketball_ai/chat/engine.py:7`).
- **`datetime.utcnow()`** usato in `gui/app.py:124` — deprecato in Python 3.12, usare
  `datetime.now(timezone.utc)`.
- **`os.environ.setdefault`** in `main.py:227-229` accoppia CLI e processo API tramite env-var
  globale: fragile in test e con reloader.

### 3.2 Piano di miglioramento — Python / Architettura

| #   | Intervento                                                                                              | Prio | Effort | Impatto |
|-----|--------------------------------------------------------------------------------------------------------|------|--------|---------|
| P1  | **Decomporre `scenarios/engine.py`** in: `TrajectoryService`, `TransferService`, `FitService`, `LineupService` (single-responsibility). | P1 | M | Alto |
| P2  | **Decomporre `chat/engine.py`** estraendo un *intent handler registry* (`@handler(Intent.PREDICT)`) per evitare l'if-else gigante. | P1 | M | Alto |
| P3  | **Introdurre mypy/pyright in CI** con `strict-optional`; obiettivo type coverage >80% in `models/` e `api/`. | P1 | M | Alto |
| P4  | **Lock-file delle dipendenze** (`uv pip compile` o `poetry lock`); pin esatti in `requirements.lock`. | P0 | S | Alto |
| P5  | **Rimuovere lo stato globale `app_state`**: usare `request.app.state` o un container DI (es. `dependency_injector`). | P1 | M | Alto |
| P6  | **Rimuovere `random.seed` modulo-level** in `generator.py`; passare un `np.random.default_rng(seed)` esplicito. | P0 | S | Medio |
| P7  | **Sostituire `_safe_model_dir`** con `Path.resolve().relative_to(allowed_root)`. | P1 | S | Medio |
| P8  | **Eliminare `sys.path.insert`** dal GUI; documentare `pip install -e .` come prerequisito. | P2 | XS | Basso |
| P9  | **Sostituire `datetime.utcnow()`** con `datetime.now(timezone.utc)` ovunque. | P1 | XS | Basso |
| P10 | **Modello configurazione**: introdurre `pydantic-settings` per leggere tutte le env (DATA_SOURCE, MODEL_DIR, AZURE_SQL_*, JWT_*) in un'unica `Settings`. | P1 | S | Alto |
| P11 | **Pre-commit hooks**: ruff (oggi non blocca), mypy, end-of-file fixer, detect-secrets. | P1 | S | Medio |
| P12 | **Docstring convention** uniforme (Google style, già parzialmente adottata) + pydocstyle in lint. | P3 | S | Basso |
| P13 | **Refactor naming**: `_to_int`, `_normalize_id`, `_primary_pos` con underscore leading sono "privati" ma importati da molti moduli — promuoverli a API pubbliche. | P2 | S | Basso |
| P14 | **Async I/O**: la API è `async` ma le chiamate al modello sono sincrone (XGBoost CPU). Considerare `run_in_executor` o offloading a worker (Celery/RQ) per training. | P2 | M | Medio |

---

## 4. Dati e pipeline ETL

### 4.1 Cosa metto in dubbio

- **Il generator produce dati che si auto-confermano**: pesi delle distribuzioni scelti dall'autore,
  poi il modello impara la stessa funzione. Non è validation, è tautologia.
- **`sql_loader.py` auto-discovery**: euristica basata su nomi colonne — se domani Azure rinomina
  `Player_Stats` in `PlayerStats`, l'auto-match potrebbe scegliere la tabella sbagliata senza warning.
- **CSV + Azure SQL come backend intercambiabili**: contratto fragile. I tipi numerici/datetime
  non sono garantiti uguali (es. ID hex vs int).
- **Nessuna data validation a livello di schema**: niente pandera, niente `great-expectations`. Un
  CSV malformato passa indisturbato fino al modello.
- **`schema_db.sql`** è committato ma non c'è migration tool (Alembic, sqlmesh). Evoluzione schema = manuale.
- **`team_player_relations`** non ha gestione esplicita di stagioni: difficile distinguere "rosa
  attuale" da "ha giocato in passato".
- **Nessun PII redaction**: i nomi dei giocatori sono reali (potenzialmente). Per i dati reali serve
  policy GDPR (anonimizzazione opzionale, diritto all'oblio).

### 4.2 Piano di miglioramento — Dati

| #   | Intervento                                                                                                  | Prio | Effort | Impatto |
|-----|-------------------------------------------------------------------------------------------------------------|------|--------|---------|
| D1  | **Schema validation** con `pandera` sui DataFrame all'ingresso (loader file e SQL).                          | P0   | S      | Alto    |
| D2  | **Migration tool**: introdurre Alembic per `schema_db.sql`; versionare le migrazioni.                       | P1   | M      | Alto    |
| D3  | **Auto-discovery con fail-safe**: se il match score < soglia, raise eccezione esplicita (oggi fallback silenzioso). | P1 | S | Alto |
| D4  | **Test di contract** tra CSV loader e SQL loader: stesso fixture deve produrre lo stesso `dict` output.     | P1   | M      | Alto    |
| D5  | **Connection pooling esplicito** SQLAlchemy con `pool_pre_ping=True` e retry esponenziale.                  | P1   | S      | Medio   |
| D6  | **Data lineage**: per ogni run di training salvare hash dei dataset di input + timestamp.                   | P1   | S      | Alto    |
| D7  | **Pipeline ETL idempotente** (Prefect, Dagster o anche un Makefile chiaro): oggi tutto è ad-hoc CLI.        | P2   | L      | Alto    |
| D8  | **Anonymization layer** opzionale (`--anonymize`) per export WordPress e demo pubbliche.                    | P2   | S      | Medio   |
| D9  | **Caching dei dataset SQL** su disco (parquet) con TTL — oggi ogni avvio API ricarica tutto da SQL.        | P1   | S      | Alto    |
| D10 | **Versioning del dataset** (DVC o Delta Lake) per riprodurre training storici.                              | P3   | L      | Medio   |

---

## 5. API REST (FastAPI)

### 5.1 Cosa metto in dubbio

- **`API_KEY` statica via env-var** è anacronistica per un sistema multi-tenant; meglio JWT con
  refresh (parzialmente già presente in `auth/`).
- **CORS wildcard in dev**: ok, ma la fallback `[]` in production *blocca silenziosamente* il browser
  — meglio fallire fast con messaggio esplicito.
- **`@app.middleware("http")` per la API-key**: precede il rate limiter quindi un attaccante può
  ancora consumare CPU del middleware. Riordinare.
- **Niente versioning vero**: `/api/v1` ma nessun pattern per `/v2`. Schema breaking changes saranno
  dolorosi.
- **Health check superficiale**: `/health` ritorna solo `data_loaded`. Manca `readiness` (modello
  caricato? db raggiungibile?) e `liveness` separati.
- **No OpenAPI metadata custom**: tag descritti, ma esempi richiesta/risposta assenti → DX scadente
  per integratori (WordPress widget, mobile).
- **No pagination su `/players`, `/teams`**: liste potenzialmente di migliaia di record.
- **No ETag / cache headers**: ogni GET `/players/{id}` ricalcola tutto.
- **Rate limiting opzionale (`SLOWAPI_AVAILABLE`)**: in produzione **deve** essere obbligatorio.

### 5.2 Piano di miglioramento — API

| #   | Intervento                                                                                | Prio | Effort | Impatto |
|-----|------------------------------------------------------------------------------------------|------|--------|---------|
| AP1 | Rendere **JWT l'unico schema auth**, deprecare API_KEY statica.                          | P1   | M      | Alto    |
| AP2 | **`/health/live` e `/health/ready`** separati; ready controlla `engine is not None and ensemble.is_trained`. | P0 | S | Alto |
| AP3 | **Pagination** standard (`limit`, `offset` o cursor) su tutte le list endpoint.          | P1   | S      | Alto    |
| AP4 | **OpenAPI examples** per ogni endpoint (`response_model=…, responses={…}`).              | P1   | M      | Medio   |
| AP5 | **ETag** sui GET deterministici (players, teams).                                        | P2   | S      | Medio   |
| AP6 | **Rate limiting obbligatorio** in produzione (non opzionale); aggiungere quote per JWT user. | P1 | S | Alto |
| AP7 | **CORS fallback fail-fast**: se `ALLOWED_ORIGINS` non è set e `API_ENV=production`, log ERROR e raise. | P1 | XS | Medio |
| AP8 | **Batch endpoints**: `/predictions/batch` con lista di `(player_id, team_id)` per evitare N+1 chiamate. | P1 | M | Alto |
| AP9 | **Idempotency-Key** sugli endpoint POST scenari, per retry safe.                         | P2   | S      | Medio   |
| AP10| **Strutturare `app_state` con DI** (vedi P5 sezione 3).                                  | P1   | M      | Alto    |

---

## 6. Chat / NLP

### 6.1 Cosa metto in dubbio

- **Intent detection è TF-IDF + Logistic Regression su esempi hard-coded multilingua** (`_EXAMPLES`
  con ~14 lingue). Manutenzione: aggiungere un intent significa scrivere ~80 esempi a mano. Non scala.
- **Soglia di confidenza arbitraria** (`CONF_THRESHOLD`, `AMBIG_RATIO`): no calibrazione.
- **Nessun memory di lungo termine** nelle sessioni (`chat/session.py` 86 LOC).
- **Nessun fallback a LLM** quando intent = UNKNOWN: l'utente riceve "non ho capito".
- **Nessuna gestione del *grounding*** delle risposte: l'engine combina template + dati. Bene per
  factualità, ma rigido (ogni intent ha il suo template). Difficile esprimere risposte composite.
- **`engine.py` 1091 LOC** con if-else giganti per intent → manutenzione lenta.
- **Niente eval suite per la chat**: come misuriamo se una risposta è "buona"?

### 6.2 Piano di miglioramento — Chat

| #   | Intervento                                                                                   | Prio | Effort | Impatto |
|-----|---------------------------------------------------------------------------------------------|------|--------|---------|
| C1  | **Architettura RAG ibrida**: TF-IDF rimane come router veloce; UNKNOWN → fallback a LLM con tool calling che invoca `WhatIfEngine` come tool. | P1 | L | Molto alto |
| C2  | **Eval suite** con set di Q&A annotate (intent atteso, slot attesi, risposta canonica). Metriche: intent accuracy, slot F1, response BLEU/ROUGE. | P0 | M | Alto |
| C3  | **Slot filling esplicito**: oggi l'entity extraction è euristica string-match; passare a modello NER (spaCy multilingua) o LLM. | P1 | M | Alto |
| C4  | **Memory conversazionale**: salvare ultimi N turn in `session.py` e iniettare come contesto. | P2 | S | Medio |
| C5  | **Calibrare soglie** UNKNOWN/AMBIG su validation set, non a colpo d'occhio.                | P1 | S | Medio |
| C6  | **Handler registry per intent** (vedi P2) per ridurre `engine.py` a ~300 LOC.              | P1 | M | Alto |
| C7  | **Streaming responses** SSE su `/chat` per percezione di latenza migliore.                  | P2 | M | Medio |
| C8  | **Guardrail di sicurezza**: prompt injection mitigation (se introduco LLM), filtro topic out-of-domain. | P1 | S | Alto |
| C9  | **Localizzazione output**: rispondere nella lingua del messaggio utente (oggi è inglese mistato).     | P2 | M | Medio |

---

## 7. Sicurezza

### 7.1 Cosa metto in dubbio

- **Password admin di default mostrata in chiaro a video** (`gui/app.py:106-111`): chiunque
  apra il browser durante il primo deploy la vede. Anti-pattern.
- **`users.json`, `roles.json`, `sessions.json` su filesystem**: nessun lock, race condition
  possibili sotto carico. Bisogna usare SQLite o DB.
- **Hash PBKDF2-HMAC-SHA-256 a 260k iterazioni**: ok nel 2017, oggi si raccomanda Argon2id o
  scrypt. PBKDF2 resta accettabile ma è inferiore.
- **Nessun lockout** dopo N tentativi falliti → vulnerabile a brute force.
- **Sessioni server-side senza rotazione** del token al login successivo.
- **Cookie session senza flag esplicito `Secure`/`HttpOnly`/`SameSite`**: dipende da
  `extra_streamlit_components.CookieManager` defaults.
- **`secrets/credentials` in `.env`** in chiaro: ok per dev, ma serve un secret manager
  (Azure Key Vault, già che il backend è Azure SQL!).
- **Nessun audit log** delle azioni admin (chi ha creato/cancellato utenti, chi ha cambiato ruoli).
- **`schema_db.sql`** non esegue alcun controllo di SQL injection visibile per le query custom
  in `sql_loader.py` (verificare uso di parametrizzazione).

### 7.2 Piano di miglioramento — Sicurezza

| #   | Intervento                                                                                      | Prio | Effort | Impatto |
|-----|------------------------------------------------------------------------------------------------|------|--------|---------|
| S1  | **Non mostrare mai la password admin in pagina**: scriverla in un file `.admin_credentials` con permessi 0600 e dirlo nella UI. | P0 | S | Alto |
| S2  | **Migrare a Argon2id** (`argon2-cffi`) per le password.                                         | P1 | S | Alto |
| S3  | **Account lockout**: 5 tentativi falliti → 15 min lock, log in audit.                          | P0 | S | Alto |
| S4  | **Audit log persistente** (tabella DB) di tutte le azioni admin e auth.                         | P1 | M | Alto |
| S5  | **Migrare users/roles/sessions da JSON a SQLite/Postgres** (concurrency safe).                  | P1 | M | Alto |
| S6  | **Forzare cookie `Secure`/`HttpOnly`/`SameSite=Lax`** e rotazione token al login.               | P0 | S | Alto |
| S7  | **Secret manager** (Azure Key Vault) per `JWT_SECRET`, `AZURE_SQL_PASSWORD`, `API_KEY`.        | P1 | M | Alto |
| S8  | **Security headers middleware**: HSTS, CSP, X-Content-Type-Options, Referrer-Policy.            | P1 | S | Alto |
| S9  | **Verificare uso parametrizzato** in `sql_loader.py` (no `f-string` su SQL).                    | P0 | S | Alto |
| S10 | **CSRF protection** sulle form Streamlit (token nascosto) — oggi non c'è.                      | P2 | M | Medio |
| S11 | **Dependency scanning** continuo (Dependabot già su GitHub? Verificare). Aggiungere `pip-audit` in CI. | P1 | XS | Alto |
| S12 | **Penetration test interno** su endpoint API con OWASP ZAP / nuclei.                            | P2 | M | Alto |

---

## 8. Test e CI/CD

### 8.1 Cosa metto in dubbio

- **73 test su dati sintetici** = test della pipeline, non del modello. Un modello che impara `y = y`
  passerebbe tutti i test.
- **`pytest tests/ -q`** non riporta coverage. Non sappiamo dove sono i buchi.
- **`ruff` e `flake8` in CI con `|| true`**: i lint failure sono ignorati. → linting di facciata.
- **Nessun smoke test della GUI** Streamlit (es. con `playwright` o `pytest-playwright`).
- **Nessun test di carico** dell'API.
- **Build Docker solo su main**: niente test di startup del container negli altri branch.
- **Nessun integration test SQL** (giustificabile per Azure SQL, ma serve un test docker mssql).
- **Python 3.11 + 3.12 in CI**: ok, ma `requirements.txt` non dichiara `python_requires`.

### 8.2 Piano di miglioramento — Test/CI

| #   | Intervento                                                                                      | Prio | Effort | Impatto |
|-----|------------------------------------------------------------------------------------------------|------|--------|---------|
| T1  | **Coverage report obbligatorio** (`pytest --cov` + soglia minima 75% per `models/` e `api/`).   | P0 | S | Alto |
| T2  | **Rimuovere `|| true`** da lint in CI → lint fallisce la build se ci sono errori.              | P0 | XS | Alto |
| T3  | **Property-based testing** (Hypothesis) per i feature engineers: invarianti su input random valido. | P1 | M | Alto |
| T4  | **Test di regressione del modello**: salvare 50 predizioni di riferimento; se nuovo training devia >X%, fallire. | P1 | M | Alto |
| T5  | **Smoke test della GUI** con `playwright` (login → tab Predizioni → predict).                   | P2 | M | Medio |
| T6  | **Test di startup Docker** in CI (build + run + curl `/health`) anche sui feature branch.       | P1 | S | Medio |
| T7  | **Test integrazione SQL** con `testcontainers-mssql` o sqlite-equivalente.                      | P2 | M | Medio |
| T8  | **Load testing** con `locust` su endpoint `/predictions/*` (target: p95 < 200ms).               | P2 | M | Medio |
| T9  | **Dichiarare `python_requires=">=3.11,<3.13"`** in `pyproject.toml`.                            | P1 | XS | Basso |
| T10 | **Matrix `os: [ubuntu, windows, macos]`**: la GUI Streamlit gira su tutti, validare.            | P3 | S | Basso |

---

## 9. Osservabilità e Operations

### 9.1 Cosa metto in dubbio

- **Logging configurato in `lifespan`** con `basicConfig`: ok per CLI, sub-ottimale in produzione.
- **No structured logging** (JSON): difficile aggregare in Datadog/Grafana.
- **No tracing distribuito** (OpenTelemetry).
- **No metriche Prometheus**: non sappiamo richiesta/sec, latenza p95, cache hit rate del modello.
- **Drift detection (`monitoring/drift.py`)** esiste ma non c'è scheduler che lo lancia automaticamente.
- **Nessun alerting**: se il modello degrada o l'API è down, nessuno lo sa.
- **`models_saved/` è una dir locale**: in K8s un restart perde il modello.

### 9.2 Piano di miglioramento — Ops

| #   | Intervento                                                                              | Prio | Effort | Impatto |
|-----|----------------------------------------------------------------------------------------|------|--------|---------|
| O1  | **Structured logging** JSON con `structlog`; correlation-id per request.                | P1   | S      | Alto    |
| O2  | **Prometheus metrics** (`prometheus-fastapi-instrumentator`).                            | P1   | S      | Alto    |
| O3  | **OpenTelemetry tracing** (FastAPI + SQLAlchemy auto-instrumentation).                  | P2   | M      | Alto    |
| O4  | **Scheduler drift detection** (CronJob/APScheduler) settimanale; metric `psi_max` esposta. | P1 | M | Alto |
| O5  | **Model registry** (MLflow o un semplice bucket S3/Azure Blob con metadata JSON).        | P1   | M      | Alto    |
| O6  | **Alerting** (Grafana/Alertmanager) su: error rate >1%, p95 >500ms, modello non caricato. | P1 | M | Alto |
| O7  | **Backup automatico** di `users.json`/`roles.json`/`sessions.json` (o DB equivalente).  | P2   | S      | Medio   |
| O8  | **Runbook** documentato per: deploy, rollback, incident, retraining.                    | P1   | S      | Alto    |

---

## 10. Documentazione

### 10.1 Cosa metto in dubbio

- **README** descrive *cosa fa* ma non *perché* è strutturato così. Mancano: architecture decision
  records (ADR), guida contributor, esempi end-to-end con dati reali.
- **Nessun changelog** (`CHANGELOG.md`).
- **Nessuna API reference generata** (es. mkdocs-material + mkdocstrings).
- **Notebook `notebooks/`** presente ma non ho viste tracce dei notebook nel listing → forse vuoto.
- **Mancano diagrammi**: data flow, sequence diagram chat, model training pipeline.

### 10.2 Piano di miglioramento — Docs

| #   | Intervento                                                                  | Prio | Effort | Impatto |
|-----|---------------------------------------------------------------------------|------|--------|---------|
| DOC1| `CONTRIBUTING.md`, `CHANGELOG.md` (Keep a Changelog), `ADR/` directory.    | P1   | S      | Alto    |
| DOC2| API reference auto-generata con `mkdocs-material` + `mkdocstrings`.       | P2   | M      | Medio   |
| DOC3| Diagrammi C4 (context, container, component) in `docs/architecture/`.     | P2   | M      | Alto    |
| DOC4| Tutorial "First prediction in 5 min" con dati pubblici reali.             | P1   | S      | Alto    |
| DOC5| Sezione "Model card" (caratteristiche, limiti, bias, dati di training).   | P0   | S      | Alto    |

---

## 11. Roadmap proposta (90 giorni)

### Sprint 1 (settimane 1-2) — *Critical bugs & hygiene*
- A12 (`durability_score` league-aware) — **bug funzionale**
- P6 (rimuovere `random.seed` modulo-level)
- P9 (`datetime.utcnow`)
- S1 (no password in UI), S3 (lockout), S6 (cookie flags), S9 (verifica SQL injection)
- T1 (coverage), T2 (lint blocca)
- U10 (error boundary)
- AP2 (health live/ready), AP7 (CORS fail-fast)
- P4 (lock-file), DOC5 (model card)

### Sprint 2 (settimane 3-6) — *MLOps foundation*
- A1 (dataset reale), A2 (target non circolare), A3 (time-based split), A4 (baseline)
- A9 (explainability), A17 (data signature)
- D1 (pandera), D6 (lineage), D9 (parquet cache)
- O1, O2, O5 (structured log, metrics, model registry)

### Sprint 3 (settimane 7-10) — *Refactor & architecture*
- U1 (split GUI), P1+P2 (split engines), P5 (DI), P10 (settings)
- C1 (RAG ibrido), C2 (eval suite), C6 (handler registry)
- AP1 (JWT only), AP3 (pagination), AP8 (batch)

### Sprint 4 (settimane 11-12) — *UX polish & docs*
- U2-U13 (design system, dark mode, charts, telemetry)
- DOC1-DOC4 (changelog, ADR, diagrammi, tutorial)
- T3-T6 (property tests, regression model, playwright)

---

## 12. Indicatori di successo (KPI proposti)

| Area      | KPI                                                                       | Baseline | Target 90gg |
|-----------|---------------------------------------------------------------------------|----------|-------------|
| AI        | MAE su dataset reale holdout                                              | n/a      | misurato + < baseline statistica |
| AI        | Calibration coverage 90% CI                                               | n/a      | 88-92%      |
| UX        | Tempo medio "login → prima predizione"                                    | n/a      | < 60 s      |
| UX        | NPS interno (analisti)                                                    | n/a      | ≥ 30        |
| API       | p95 latency `/predictions/*`                                              | n/a      | < 200 ms    |
| API       | Error rate                                                                | n/a      | < 0.5%      |
| Qualità   | Test coverage `basketball_ai/`                                            | n/a      | ≥ 80%       |
| Qualità   | Type coverage (mypy)                                                      | 0%       | ≥ 80%       |
| Sicurezza | High/Critical CVE in deps                                                 | n/a      | 0           |
| Ops       | MTTR incident                                                             | n/a      | < 1 h       |

---

## 13. Cosa NON cambierei

Per onestà intellettuale, ecco le scelte attuali che ritengo *corrette*:

- Pacchetto monolitico `basketball_ai/` con sub-moduli chiari: giusto compromesso a questa scala.
- XGBoost come base learner: appropriato per tabular data di queste dimensioni.
- FastAPI + Pydantic v2: scelta moderna e performante.
- Separazione `loader.py` / `sql_loader.py` dietro la stessa interfaccia: pattern adapter ben fatto.
- Conformal prediction per le CI: scelta seria, da affinare ma direzione giusta.
- Multi-tier league factor data-derived (non hard-coded): bene, va in questa direzione.
- RBAC con sezioni configurabili: progettazione corretta, va solo persistito meglio.

---

*Fine analisi. Documento vivo: aggiornare a ogni sprint con risultati misurati e nuove evidenze.*

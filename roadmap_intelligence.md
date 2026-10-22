# Player Intelligence — roadmap operativa

Ultimo aggiornamento: 2026-10-21  
Repository di riferimento: `Elrond0192/AI_Model`  
Stato complessivo: **estensione implementata in codice; refresh del database e verifica dei dati reali ancora da eseguire**.

Questa roadmap distingue l'esistenza del codice dalla disponibilità effettiva dei dati e dalla validazione del risultato. Una capacità non va promossa a pienamente disponibile solo perché esiste una funzione Python o una tabella SQL.

## Regole architetturali

- **WordPress** orchestra le domande, applica il contratto di presentazione e rende i risultati leggibili; non implementa l'engine delle metriche.
- **HoopmetricsEngine** produce AdvancedStats e le relative tabelle, incluse le statistiche clutch e le statistiche avanzate delle lineup.
- **AI_Model** normalizza le sorgenti, compone Player Intelligence e gestisce BB-Rating, previsioni e scenari.
- Il rendimento individuale osservato deve essere espresso tramite il **BB-Rating canonico 1–100**. Il rating statistico nativo non va esposto come se fosse una valutazione direttamente interpretabile.
- Un'associazione osservazionale, una previsione e un controfattuale non dimostrano da soli causalità.
- I dati mancanti devono essere dichiarati come non disponibili o insufficienti. Non vanno sostituiti con valori fittizi né con statistiche calcolate in un contesto diverso.

## Completato nel codice

### 1. BB-Rating canonico e spiegabilità
- La chat usa BB-Rating 1–100 come valutazione osservata canonica.
- La scomposizione matematica del BB-Rating espone i contributi dei componenti senza cambiare formula, pesi o calibrazione.
- Le comparazioni fra stagioni possono esporre BB-Rating precedente, corrente e variazione, tenendoli distinti dalle statistiche grezze.

### 2. Attribuzione del rendimento individuale e contesto squadra
- `causal_team_effect` dispone di un ATT osservazionale a livello di coorte, con controlli di supporto, overlap e bilanciamento; resta `partial` finché le verifiche empiriche fuori campione non superano i gate.
- I metodi di refresh persistono pannelli team-switch e teammate-exposure usando identità globali canoniche e outcome BB-Rating 1–100.
- La pagina Model Releases espone refresh manuali globali per tutte le combinazioni lega/competizione con transizioni completate; ogni esecuzione riporta contesti aggiornati, saltati, falliti e righe scritte.
- È stato aggiunto il confronto del Net Rating di squadra nelle lineup con/senza il giocatore, entro gli stessi contesti squadra-stagione quando possibile.
- È stato aggiunto il confronto descrittivo per compagno: quintetti in cui il giocatore è affiancato dal compagno rispetto a quintetti in cui quel compagno non è presente.
- L'evidenza di lineup è opzionale e il motore degrada a `unavailable`/`insufficient_evidence` senza interrompere l'intera analisi se il feed manca o non è leggibile.
- Le risposte devono esplicitare che il Net Rating della squadra non è il BB-Rating individuale e non costituisce una stima causale del contributo del giocatore o del compagno.

### 3. Statistiche clutch
- La sorgente canonica è **`Analisi.AdvancedStats_Clutch_{LEAGUE}`**.
- `ai_source_full.sql` proietta le statistiche clutch nella tabella canonica `AI_Source.PlayerCompetitionStats`, tra cui partite clutch, punti, tiri e percentuali, rimbalzi, assist, palle rubate, stoppate, palle perse, ORtg, DRtg, NetRtg e ClutchnessRating.
- La variante storica `AdvanceStats_Clutch_{LEAGUE}` è mantenuta solo come fallback di compatibilità.
- `clutch_analysis` esiste già in Player Intelligence. Non deve dipendere dalle tabelle Simulation per leggere questi aggregati.

### 4. Sorgenti Simulation e Scenario serving
- `ai_scenario_serving.sql` ora possiede il DDL delle tabelle `Simulation*`, invece di aspettarsi che siano state create da un bootstrap separato.
- È stato aggiunto `AI_Source.RefreshSimulationSources(league_key, season)` per ricostruire i feed sorgente partendo da `Pbp.{LEAGUE}`, filtrata per `Season`, dalle anagrafiche canoniche e, quando disponibile, da `Analisi.AdvancedStats_Lineups_Quarter_{LEAGUE}`.
- Il refresh stagionale richiama la procedura Simulation una volta per lega/stagione prima di aggiornare i contesti `Scenario*`.
- `SimulationPbpEvents` salva esposizioni PBP e assegna probabilisticamente gli eventi offensivi ai giocatori della lineup avversaria; la difesa individuale **non è osservata direttamente** dal feed PBP.
- `SimulationLineupStints` privilegia le statistiche di quintetto AdvancedStats; se non sono utilizzabili, usa aggregati derivati dalle lineup nel PBP.
- `SimulationShotProfiles` usa bucket `unknown_two_point` e `unknown_three_point` quando non ci sono coordinate o zone di tiro. Non inventa zone come ferro o angolo.
- `SimulationPlayTypeStats` si popola soltanto se esiste un campo tattico separato che contiene effettivamente la tassonomia riconosciuta. I codici evento come `2FGM` e `TOV` non vengono rinominati in pick-and-roll o isolation.
- `SimulationCausalPanel` ha ora un contratto persistente e viene popolata soltanto tramite gli espliciti refresh osservazionali di Model Releases; la presenza di righe non dimostra causalità e non rende `causal_team_effect` una capacità validata.
- Il refresh dei pannelli causali è manuale e separato dallo scheduler ricorrente: non è eseguito da chat e non esegue training o promozioni.
- `SimulationSourceRefreshState` registra lo stato e il conteggio delle righe per lega/stagione, così un feed mancante non appare come un refresh pienamente riuscito.

### 5. Composizione delle capacità
- Il registro Player Intelligence è alla versione 1.1 ed espone dati obbligatori, dati opzionali e soglie minime per le analisi di attribuzione.
- Le domande su tiro, difesa e utilizzo squadra compongono effettivamente i livelli di analisi che dichiarano.
- Le capacità di attribuzione squadra/compagni restano **parziali** fino a quando i dati reali e le verifiche fuori campione non supportano conclusioni più forti.

## Da fare — priorità immediata

### P0 — Applicare lo schema e rigenerare i dati nel database
Stato: **codice committato; non ancora applicato al database di produzione da questa sessione**.

1. Eseguire `basketball_ai/data/ai_source_full.sql` per rigenerare il contesto canonico, compreso il join con `AdvancedStats_Clutch_{LEAGUE}`.
2. Eseguire `basketball_ai/data/ai_scenario_serving.sql` per creare le nuove tabelle Simulation, le procedure e i permessi.
3. Richiamare `AI_Source.RefreshContextSeasonal(season)` per ciascuna stagione che ha contesti canonici da servire. La procedura ricrea i feed Simulation e aggiorna Training/Scenario per lega, stagione e competizione.
4. Verificare `AI_Source.SimulationSourceRefreshState` per identificare leghe/stagioni `available`, `partial` o `unavailable`.
5. Verificare che `ScenarioLineupStats`, `ScenarioShotProfiles` e `ScenarioDefenderMatchups` non restino vuote quando le sorgenti necessarie sono presenti.

Esempio per una stagione specifica, dopo l'applicazione dei due file SQL:

```sql
CALL "AI_Source"."RefreshContextSeasonal"(2026);

SELECT "LeagueKey", "Season", "Status",
       "PbpRows", "PbpEventRows", "LineupRows",
       "ShotProfileRows", "PlayTypeRows", "Details"
FROM "AI_Source"."SimulationSourceRefreshState"
WHERE "Season" = 2026
ORDER BY "LeagueKey";
```

Per le stagioni storiche, ripetere la chiamata per ogni anno per cui sono presenti contesti canonici. Il refresh stagionale non va sostituito da una chiamata a una singola procedura legacy.

### Nota di compatibilità dello schema Training (hotfix 2026-10-21)

- Le tabelle fisiche `TrainingPlayerCompetitionStats` e `TrainingTeamCompetitionStats` possono sopravvivere a un aggiornamento delle viste canoniche; `CREATE TABLE IF NOT EXISTS` da solo non aggiunge le nuove colonne.
- La versione aggiornata di `ai_scenario_serving.sql` aggiunge in modo incrementale le colonne canoniche mancanti e `RefreshTrainingServing` genera liste esplicite di colonne, evitando la dipendenza dall'ordine fisico.
- Se il refresh fallisce con `INSERT has more expressions than target columns`, applicare di nuovo la versione aggiornata di `ai_scenario_serving.sql` e poi rilanciare `RefreshContextSeasonal(season)`. Se il rebuild canonico è già riuscito, non occorre rieseguire `ai_source_full.sql` soltanto per questo errore.
- Lo stato del refresh 2025 resta da confermare dopo il nuovo tentativo; il commit del fix non implica che il database di produzione sia già stato aggiornato.

### P0 — Validare clutch su dati reali
1. Controllare il numero di righe in `Analisi.AdvancedStats_Clutch_{LEAGUE}` per stagione e competizione.
2. Dopo il rebuild di `ai_source_full.sql`, verificare che i valori non null di clutch siano arrivati in `AI_Source.PlayerCompetitionStats` e che il join per player/season/competition sia corretto.
3. Testare almeno alcuni casi con `ClutchGames > 0`, percentuali nulle legittime e `ClutchnessRating = 'N/A'`. Una percentuale nulla con zero tentativi non deve essere trasformata in uno zero osservato.
4. Verificare una richiesta API/chat `clutch_value` contro una riga nota del database, includendo il numero di partite clutch e la competizione.

### P1 — Audit copertura di PBP e lineup
1. Misurare la copertura di `Pbp.{LEAGUE}` filtered by `Season`, `AdvancedStats_Lineups_Quarter_{LEAGUE}` e delle mappe di identità canoniche per lega/stagione.
2. Identificare righe PBP senza giocatore canonico, squadra canonica o quintetto completo; non correggerle per inferenza arbitraria.
3. Controllare che il numero di giocatori in ogni lineup canonica sia esattamente cinque.
4. Verificare se esistono dati di tracking difensivo reali. Finché non esistono, `defender_id` resta un'esposizione inferita dalla lineup avversaria con probabilità, non un assegnamento difensivo osservato.
5. Verificare la definizione dei possessi per i matchup: gli eventi PBP e i possessi non sono sinonimi. Le stime basate su esposizioni evento devono mantenere espliciti metodo e limiti.

### P1 — Validare i feed Scenario
1. Verificare i conteggi per lega/stagione/competizione nelle tabelle `Scenario*` e in `ServingRefreshState`.
2. Confermare che i permessi di lettura siano presenti per `ai_model`, `hm_admin` e `hm_local_workers`.
3. Verificare cache e risposta API dopo l'aggiornamento dei dati.
4. Se `SimulationPlayTypeStats` rimane vuota, controllare se la sorgente contiene una colonna con vera classificazione tattica. Non popolarla con codici di risultato del tiro.
5. Se esistono coordinate dei tiri, estendere la proiezione shot-profile per zone reali; altrimenti mantenere la classificazione sconosciuta.

## Da fare — validazione scientifica e prodotto

### P2 — Stabilità e controfattuali
- Eseguire backtest temporali su stagioni non usate per lo sviluppo per valutare quanto le associazioni di lineup siano stabili.
- Aggiungere controlli per avversario, stato gara, compagni, ruolo e selezione delle lineup solo quando le colonne e la copertura lo consentono.
- Costruire un disegno controfattuale con gruppi di controllo e verifica dei trend precedenti prima di parlare di effetto causale squadra/compagni.
- Non popolare `SimulationCausalPanel` con una semplice differenza prima/dopo.

### P2 — Contratti, routing e UX
- Mantenere il registro delle capacità allineato ai livelli effettivamente eseguiti dal motore.
- Verificare che le risposte clutch citino stagione, competizione, numero di partite e qualità dell'evidenza.
- Verificare che tiro/difesa distinguano cambiamenti osservati, contesto di ruolo e ipotesi controfattuali.
- Mantenere il BB-Rating su scala 1–100 come unica valutazione individuale presentata all'utente; non esporre il rating statistico sorgente.
- Tenere visibile lo stato `partial` quando lineup, PBP, tassonomia tattica o dati di tracking sono assenti.

### P3 — Rilascio
- Ottenere CI verde su test Python, integrazione PostgreSQL, lint e security audit per l'ultima revisione.
- Applicare le SQL al database effettivo, controllare conteggi e log, poi ricostruire/restartare l'API soltanto secondo la normale procedura di rilascio.
- Eseguire uno smoke test API e chat con un giocatore e una lega reali.
- Registrare in questa roadmap data, commit, stato CI e risultato del controllo dati dopo ogni deploy.

## Verifica automatizzata del codice

- La pipeline PostgreSQL esegue un fixture con `Analisi.AdvancedStats_Clutch_ITA1` e `Pbp.ITA1` e richiama `RefreshContextSeasonal(2025)`.
- Il controllo end-to-end verifica il trasferimento dei campi clutch verso `PlayerCompetitionStats`, il conteggio PBP, la presenza di righe Simulation per eventi/lineup/shot e il popolamento delle tabelle `ScenarioLineupStats`, `ScenarioShotProfiles` e `ScenarioDefenderMatchups`.
- Al commit `f5ec39e8c5b45f1f058a31229d58b169a7c47fe9`, la verifica PostgreSQL end-to-end, i test Python 3.11/3.12, lint e security audit risultano superati. Il build Docker era ancora in esecuzione nell'ultima lettura.
- Il test PostgreSQL usa dati sintetici, ma verifica realmente il join clutch, `RefreshContextSeasonal(2025)`, lo stato per lega/stagione e il popolamento delle tabelle `Scenario*`. Non sostituisce una verifica dei conteggi nel database di produzione.
- La SQL di produzione e i refresh live restano da applicare ed eseguire.

## Verifiche e limiti da non dimenticare

- Le tabelle `AdvancedStats_Clutch_{LEAGUE}` contengono aggregati clutch già calcolati; non richiedono la ricostruzione delle Simulation per essere usate da `clutch_analysis`.
- `SimulationSourceRefreshState.Status = 'available'` indica che i feed previsti sono stati popolati per quel contesto, non che tutte le fonti di tracking tattico/difensivo esistano.
- `SimulationPlayTypeStats` può legittimamente contenere zero righe in assenza di una classificazione tattica.
- `SimulationCausalPanel` va aggiornata tramite gli entry point espliciti e guardati di refresh osservazionale; prima di interpretarla verificare copertura, invarianti e audit. Il modello `causal_team_effect` resta parziale finché i gate scientifici non sono superati.
- Le modifiche SQL presenti su GitHub non equivalgono a dati già rigenerati in produzione: la verifica finale richiede l'esecuzione delle procedure nel database e controllo dei conteggi reali.

## Riferimenti implementativi

- `basketball_ai/data/ai_source_full.sql` — sorgenti canoniche, incluse le colonne clutch.
- `basketball_ai/data/ai_scenario_serving.sql` — DDL delle Simulation/Scenario, refresh dei feed e stato dei refresh.
- `basketball_ai/data/postgres_loader.py` — lettura bounded dei feed Scenario.
- `basketball_ai/scenarios/chat_scenario_engine.py` — analisi clutch, contesto squadra, associazioni lineup/compagni e composizione Player Intelligence.
- `basketball_ai/api/question_analysis_registry.py` — capacità dichiarate, dati richiesti e soglie minime.
- `docs/PLAYER_INTELLIGENCE.md` — specifica funzionale delle analisi.

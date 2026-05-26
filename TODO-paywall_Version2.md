# TODO — Paywall HoopMetrics + Chat AI Premium

> **Documento operativo di pianificazione.** Ogni sezione è auto-contenuta e pensata per essere implementata da un agente AI (es. Claude Sonnet 4.6) **senza rompere il codice esistente**.
>
> **Repository coinvolti:**
> - `Elrond0192/Wordpress` — tema `court-analytics-pro` + plugin `hoopmetrics-api` (PHP/JS/CSS)
> - `Elrond0192/AI_Model` — FastAPI + modello XGBoost + chat engine (Python)
>
> **Regola d'oro per l'implementatore:** ogni modifica deve essere **additiva**. Non rinominare classi/funzioni esistenti, non cambiare firme pubbliche, non modificare schema cache `hm_*_v{N}_*` se non aggiungendo segmenti. Tutto ciò che è "esistente" deve continuare a funzionare identico per gli utenti `hm_bench`.

---

## 0. Decisioni consolidate (NON MODIFICARE)

### 0.1 Piani

| Piano | WP role | Donazione suggerita | Feature incluse |
|---|---|---|---|
| **Bench** | `hm_bench` | 0€ | Leaderboard completa, **tutte le metriche** (RAPTOR/LEBRON/BPM/VORP/WS/PIE), **shot chart**, **percentili**, profilo giocatore, profilo squadra, ricerca, confronti 1v1 ad hoc (non salvati), **lineup analysis**, **on/off court**, **giocatori simili** |
| **Starter** | `hm_starter` | 10€/mese | Tutto Bench + **watchlist personale** (max 50 entità) + **confronti salvati** (max 20) + **chat AI 200 msg/mese** |
| **All-Star** | `hm_allstar` | 20€/mese | Tutto Starter + **chat AI 500 msg/mese** + **what-if engine** (`/scenarios/*`) + watchlist illimitata + confronti illimitati |

> ⚠️ **Il pagamento è una donazione, non un abbonamento.** L'utente dona liberamente per sostenere il progetto; in cambio gli vengono sbloccate le feature del piano corrispondente. Non ci sono vincoli contrattuali. Le donazioni ricorrenti si rinnovano automaticamente ma possono essere annullate in qualsiasi momento.

### 0.2 Mapping scope token

| Piano | Scope token aggiunti |
|---|---|
| `hm_bench` | `read:basic`, `read:premium`, `read:sensitive` *(metriche avanzate, shot chart, lineups, oncourt e giocatori simili sono Bench per scelta strategica)* |
| `hm_starter` | `read:basic`, `read:premium`, `read:sensitive`, `read:userdata`, `read:chat` |
| `hm_allstar` | `read:basic`, `read:premium`, `read:sensitive`, `read:userdata`, `read:scenarios`, `read:chat` |

> ⚠️ Lo scope `read:sensitive` (oggi richiesto da `shots`, `lineups`, `oncourt`, `similar`) diventa **Bench** (accessibile a tutti). Implementatore: assicurarsi che `read:sensitive` sia incluso nello scope token di tutti i piani, compreso `hm_bench`.

### 0.3 Capability WordPress

| Capability | Bench | Starter | All-Star |
|---|---|---|---|
| `hm_view_basic` | ✅ | ✅ | ✅ |
| `hm_view_advanced_metrics` | ✅ | ✅ | ✅ |
| `hm_view_shot_chart` | ✅ | ✅ | ✅ |
| `hm_view_percentiles` | ✅ | ✅ | ✅ |
| `hm_view_full_leaderboard` | ✅ | ✅ | ✅ |
| `hm_use_watchlist` | ❌ | ✅ (max 50) | ✅ (illimitata) |
| `hm_save_comparisons` | ❌ | ✅ (max 20) | ✅ (illimitati) |
| `hm_view_lineups` | ✅ | ✅ | ✅ |
| `hm_view_oncourt` | ✅ | ✅ | ✅ |
| `hm_view_similar_players` | ✅ | ✅ | ✅ |
| `hm_use_scenarios` | ❌ | ❌ | ✅ |
| `hm_use_chat` | ❌ | ✅ (200 msg/mese) | ✅ (500 msg/mese) |

### 0.4 Vincoli non negoziabili

- **NIENTE export CSV/Excel/PDF** in nessun piano (sarebbe data leak verso scraping).
- **Chat Starter e All-Star** (no chat Bench).
- **Quota chat: 200 msg/mese per Starter, 500 msg/mese per All-Star**, reset il 1° del mese.
- **Il pagamento è una donazione**, non un abbonamento: nessun vincolo contrattuale, rinnovo automatico annullabile in qualsiasi momento, nessun rimborso dovuto.
- **Plugin chat separato** (`hoopmetrics-chat`), non dentro `hoopmetrics-api`.
- **Backend chat pluggabile** (Local/FastAPI/LLM), switch da admin senza redeploy.
- **Fallback automatico** del backend chat con messaggio esplicito all'utente.
- **Lingua chat: auto-detect** — la chat risponde nella lingua in cui l'utente scrive.

---

## 1. Fix di sicurezza preliminari (BLOCCANTI)

**Obiettivo:** rendere sicura l'architettura attuale *prima* di introdurre dati a pagamento.

### 1.1 Scope nella cache key — repo `Elrond0192/Wordpress`

**Problema:** i transient HoopMetrics oggi sono globali (`hm_<entity>_v{N}_<params>`). Quando introdurremo dati premium-only, un utente Bench potrebbe ricevere risposta cached generata per un All-Star.

**File da modificare:**
- `hoopmetrics-api/includes/class-leaderboard.php`
- `hoopmetrics-api/includes/class-players.php`
- `hoopmetrics-api/includes/class-teams.php`
- `hoopmetrics-api/includes/class-search.php`
- `hoopmetrics-api/includes/class-oncourt.php`

**Azione:** ogni funzione che costruisce una transient key DEVE includere uno scope hash.

```php
// PRIMA
$key = sprintf('hm_leaderboard_v%d_%s_%s', $cache_v, $nation, $metric);

// DOPO
$scope_hash = HM_Token_Service::scope_hash_for_current_request(); // es. 'sha8_basic'
$key = sprintf('hm_leaderboard_v%d_%s_%s_%s', $cache_v, $scope_hash, $nation, $metric);
```

- [ ] Aggiungere metodo `HM_Token_Service::scope_hash_for_current_request(): string` che restituisce hash short-8 degli scope ordinati alfabeticamente.
- [ ] Refactor delle 5 classi sopra (solo le righe `set_transient`/`get_transient`).
- [ ] Test di regressione: con utente anonimo (solo `read:basic`+`read:premium`) le risposte devono restare identiche byte-per-byte a oggi.
- [ ] Documentare in `ARCHITECTURE.md` la nuova convenzione.

### 1.2 Separazione scope shot/lineup — repo `Elrond0192/Wordpress`

**File:** `hoopmetrics-api/includes/class-rest-api.php`

- [ ] Trovare la registrazione delle route `/player/{hm_xxx}/shots` e `/player/{hm_xxx}/lineups`.
- [ ] Cambiare il `required_scope` di `shots` da `read:sensitive` a `read:basic` (diventa Bench).
- [ ] Lasciare `lineups` con `read:sensitive` (ora Bench — `read:sensitive` è incluso in tutti i piani).
- [ ] Assicurarsi che `oncourt` e `similar` usino `read:sensitive` (Bench).
- [ ] Aggiornare `API_DOCS.md` di conseguenza.

### 1.3 Rate limiter per-utente — repo `Elrond0192/Wordpress`

**File:** `hoopmetrics-api/includes/class-rate-limiter.php`

- [ ] Aggiungere metodo `check_user(int $user_id, string $bucket, int $limit, int $window_seconds): bool` accanto al check per IP esistente.
- [ ] Storage: WP transient `hm_rl_user_{user_id}_{bucket}`.
- [ ] Bucket previsti: `chat`, `scenarios`.
- [ ] Default limits (configurabili in `wp-config.php`):
  - `chat`: gestito dalla quota mensile (non da rate limit a finestra)
  - `scenarios`: 60 richieste/ora per All-Star

### 1.4 Audit log esteso — repo `Elrond0192/Wordpress`

**File:** `hoopmetrics-api/includes/class-hm-access-log.php`

- [ ] Aggiungere colonna/campo `user_id` (nullable) e `feature` (es. `chat`, `scenario`).
- [ ] Migration sicura: se la tabella esiste senza queste colonne, fare `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

---

## 2. Foundation paywall (Fase 1) — repo `Elrond0192/Wordpress`

**Obiettivo:** Starter e All-Star acquistabili e funzionanti (senza chat).

### 2.1 Plugin billing

- [ ] Installare **Paid Memberships Pro** (free) + **Stripe Gateway** (free, incluso).
- [ ] Configurare 3 livelli PMP con ID stabili:
  - Level ID 1: `Bench` (0€)
  - Level ID 2: `Starter` (donazione suggerita 10€/mese EUR)
  - Level ID 3: `All-Star` (donazione suggerita 20€/mese EUR)
- [ ] Configurare email transazionali PMP (benvenuto, rinnovo, scadenza, cancellazione). Le email devono comunicare chiaramente che si tratta di una donazione, non di un abbonamento.
- [ ] Pagina checkout PMP localizzata in italiano con wording "donazione".

### 2.2 Ruoli e capability — nuovo file

**Nuovo file:** `hoopmetrics-api/includes/class-hm-roles.php`

- [ ] Classe `HM_Roles` con:
  - `register_roles()` — chiamato su `register_activation_hook` del plugin
  - `unregister_roles()` — chiamato su `register_deactivation_hook`
  - `sync_user_role(int $user_id, int $pmp_level_id)` — hook a `pmpro_after_change_membership_level`
  - Costanti `ROLE_BENCH = 'hm_bench'`, `ROLE_STARTER = 'hm_starter'`, `ROLE_ALLSTAR = 'hm_allstar'`
  - Mappa PMP Level → WP Role
- [ ] Implementare assegnazione capability come da matrice §0.3.
- [ ] **Importante:** se PMP non è attivo, fallback a tutti utenti = `hm_bench`. **NON crashare.**

### 2.3 Helper PHP per i template

**Nuovo file:** `hoopmetrics-api/includes/hm-capabilities.php`

```php
function hm_user_can_feature(string $feature, ?int $user_id = null): bool;
function hm_user_plan(?int $user_id = null): string; // 'bench'|'starter'|'allstar'
function hm_is_premium(?int $user_id = null): bool;  // starter OR allstar
function hm_is_allstar(?int $user_id = null): bool;
```

- [ ] Tutte le funzioni devono restituire valori safe anche per utente non loggato (= bench).
- [ ] Cache in memoria per-request (static var) per evitare query ripetute.

### 2.4 Estensione `HM_Proxy::resolve_scope()`

**File:** `hoopmetrics-api/includes/class-hm-proxy.php`

- [ ] Metodo esistente `resolve_scope()`: aggiungere lettura di `hm_user_plan()` e mappatura piano→scope come da §0.2.
- [ ] **Backward compat:** se nessun ruolo paywall presente (es. PMP disattivato), comportamento attuale invariato.
- [ ] Unit test: utente anonimo riceve esattamente `['read:basic', 'read:premium']`.

### 2.5 Token service

**File:** `hoopmetrics-api/includes/class-hm-token-service.php`

- [ ] Aggiungere costanti `SCOPE_USERDATA`, `SCOPE_SCENARIOS`, `SCOPE_CHAT`.
- [ ] Aggiungere metodo `revoke_user(int $user_id): void` che invalida tutti i token cached per quell'utente (utile su downgrade/cancel).
- [ ] Hook `pmpro_after_change_membership_level` → `HM_Token_Service::revoke_user($user_id)`.

### 2.6 Watchlist & Comparisons — nuovo modulo

**Nuovi file:**
- `hoopmetrics-api/includes/class-hm-userdata.php`
- `hoopmetrics-api/includes/schema/userdata-tables.php`

**Schema DB:**
```sql
CREATE TABLE {$wpdb->prefix}hm_watchlist (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NOT NULL,
  entity_type ENUM('player','team') NOT NULL,
  entity_hm VARCHAR(20) NOT NULL,
  nation VARCHAR(8) NULL,
  season VARCHAR(10) NULL,
  added_at DATETIME NOT NULL,
  note TEXT NULL,
  UNIQUE KEY uniq_user_entity (user_id, entity_type, entity_hm),
  INDEX idx_user (user_id)
) {$charset_collate};

CREATE TABLE {$wpdb->prefix}hm_saved_comparisons (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NOT NULL,
  title VARCHAR(120) NOT NULL,
  payload LONGTEXT NOT NULL,  -- JSON
  created_at DATETIME NOT NULL,
  INDEX idx_user (user_id)
) {$charset_collate};
```

- [ ] Migration via `dbDelta()` su attivazione plugin.
- [ ] Limiti enforcement lato server:
  - Starter: max 50 watchlist, max 20 comparisons → al 51°/21° insert restituire 402 + messaggio "Upgrade to All-Star for unlimited".
  - All-Star: nessun limite.
- [ ] REST endpoint `/wp-json/hm-user/v1/`:
  - `GET/POST/DELETE /watchlist` (richiede capability `hm_use_watchlist`)
  - `GET/POST/PUT/DELETE /comparisons` (richiede `hm_save_comparisons`)
- [ ] **Nonce + login required** su tutti.
- [ ] Rate limit: 60 write/min per utente (anti-abuse).

### 2.7 Gating UI nei template

**File da modificare** (in `page-templates/`):

- [ ] `player.php` + `team.php`: bottoni "★ Aggiungi a watchlist" visibili solo a Starter/All-Star.
- [ ] `compare.php`: bottone "💾 Salva confronto" visibile solo a Starter/All-Star; confronti >2 giocatori solo All-Star.
- [ ] `dashboard.php`: aggiungere widget "La mia watchlist" se Starter/All-Star.

### 2.8 Componente paywall overlay riusabile

**Nuovo file:** `template-parts/paywall-overlay.php`

- [ ] Accetta parametri: `required_plan` (`'starter'|'allstar'`), `feature_name` (label visibile), `cta_url` (default: pagina pricing).
- [ ] CSS namespace `hm-paywall-*` (coerente con conventions).
- [ ] Versione "blur" (overlay traslucido su contenuto) e versione "card" (sostituisce il contenuto).
- [ ] **NON deve mai mostrare i dati reali sottostanti, anche blurred** (rischio leak via DevTools). Render server-side condizionale.

### 2.9 Pagine

- [ ] **Pagina `/sostieni/`** — template `page-templates/pricing.php`:
  - 3 card affiancate con feature comparison (Bench / Starter / All-Star)
  - Bottoni CTA che linkano al checkout PMP del livello giusto
  - Wording chiaro: "Sostieni il progetto con una donazione e sblocca le feature premium"
  - FAQ in fondo (includere: "La donazione è un abbonamento?" → "No, è un contributo volontario ricorrente annullabile in qualsiasi momento")
- [ ] **Pagina `/account/`** — template `page-templates/account.php`:
  - Piano attivo + data prossima donazione
  - Quota chat residua (placeholder finché Fase 2 non è live)
  - Link "Gestisci donazione" → PMP account
  - Link "Annulla donazione"
  - Sezione "La mia watchlist" e "I miei confronti salvati"
- [ ] **Footer**: aggiungere link "❤️ Sostieni il progetto" → pagina `/sostieni/`.

### 2.10 Test di non-regressione Fase 1

- [ ] Utente anonimo: tutte le pagine attuali funzionano identiche a oggi (leaderboard, shot chart, percentili visibili).
- [ ] Utente `hm_starter`: vede watchlist/save buttons, non vede lineup/scenarios.
- [ ] Utente `hm_allstar`: vede tutto tranne chat (ancora non implementata).
- [ ] Disattivazione PMP: sito torna a stato pre-paywall senza errori PHP.

---

## 3. Plugin `hoopmetrics-chat` (Fase 2) — NUOVO PLUGIN — repo `Elrond0192/Wordpress`

**Obiettivo:** chat AI pluggabile con 3 backend (Local/FastAPI/DeepSeek), switch da admin, fallback automatico.

### 3.1 Struttura plugin

```
hoopmetrics-chat/
├── hoopmetrics-chat.php                    # Bootstrap + activation
├── readme.txt
├── includes/
│   ├── class-hm-chat-plugin.php            # Singleton init
│   ├── class-hm-chat-rest.php              # REST endpoints
│   ├── class-hm-chat-quota.php             # Quota mensile
│   ├── class-hm-chat-router.php            # Selettore backend + fallback
│   ├── class-hm-chat-response.php          # DTO risposta normalizzata
│   ├── class-hm-chat-logger.php            # Audit log
│   ├── class-hm-chat-admin.php             # Pannello WP Admin
│   ├── backends/
│   │   ├── interface-hm-chat-backend.php   # Contract
│   │   ├── class-hm-chat-backend-local.php
│   │   ├── class-hm-chat-backend-fastapi.php
│   │   └── class-hm-chat-backend-deepseek.php
│   ├── importers/
│   │   └── class-hm-chat-model-importer.php
│   └── schema/
│       ├── predictions-table.php
│       └── chat-log-table.php
├── assets/
│   ├── js/hm-chat-widget.js
│   └── css/hm-chat-widget.css
├── templates/
│   └── chat-widget.php                     # Shortcode [hm_chat]
└── languages/
    └── hoopmetrics-chat-it_IT.po
```

### 3.2 Contract backend (interfaccia)

**File:** `includes/backends/interface-hm-chat-backend.php`

```php
interface HM_Chat_Backend_Interface {
    /** Nome univoco: 'local' | 'fastapi' | 'deepseek' */
    public function get_name(): string;

    /** True se il backend è raggiungibile/configurato. NO call costose. */
    public function health_check(): bool;

    /** Esegue la richiesta. DEVE essere idempotente in caso di retry. */
    public function ask(string $message, array $context, int $user_id): HM_Chat_Response;

    /** Capabilities dichiarate (per UI condizionale). */
    public function supports_followup(): bool;
    public function supports_freeform(): bool;
    public function supports_suggestions(): bool;
}
```

**DTO risposta** — `class-hm-chat-response.php`:

```php
final class HM_Chat_Response {
    public string $reply;                    // markdown
    public string $session_id;
    public ?string $intent = null;
    public array $data = [];                 // dati strutturati
    public array $suggestions = [];          // follow-up chips
    public string $backend_used;             // per trasparenza
    public bool $was_fallback = false;
    public ?string $fallback_reason = null;  // se was_fallback=true
}
```

> **Vincolo:** tutti e 3 i backend DEVONO popolare `reply` + `session_id` + `backend_used`. Gli altri campi sono opzionali.

### 3.3 Quota mensile

**File:** `class-hm-chat-quota.php`

User meta:
- `hm_chat_quota_limit` (int, default 200 per Starter, 500 per All-Star, 0 altrimenti)
- `hm_chat_quota_used` (int)
- `hm_chat_quota_period` (string `YYYY-MM`)

API:
```php
HM_Chat_Quota::get_remaining(int $user_id): int;
HM_Chat_Quota::consume(int $user_id, int $n = 1): bool; // false se quota esaurita
HM_Chat_Quota::reset_all_for_new_period(): int; // WP-Cron
```

- [ ] WP-Cron `hm_chat_quota_monthly_reset` registrato su attivazione, schedulato il 1° del mese 00:05.
- [ ] Su downgrade a Bench: `quota_limit = 0` (token chat invalidato).
- [ ] Su upgrade a Starter: `quota_limit = 200`, `quota_used = 0` se nuovo periodo, altrimenti preservato.
- [ ] Su upgrade a All-Star: `quota_limit = 500`, `quota_used = 0` se nuovo periodo, altrimenti preservato.

### 3.4 Router + fallback

**File:** `class-hm-chat-router.php`

```php
class HM_Chat_Router {
    public function get_active_backend(): HM_Chat_Backend_Interface;
    public function get_fallback_backend(): ?HM_Chat_Backend_Interface;
    public function dispatch(string $message, array $context, int $user_id): HM_Chat_Response;
}
```

Opzioni WP:
- `hm_chat_backend_active`: `'local'` (default) | `'fastapi'` | `'deepseek'`
- `hm_chat_backend_fallback`: `'local'` | `''` (none) — default `'local'`
- `hm_chat_fallback_message`: stringa custom (default: "⚠️ Servizio AI temporaneamente in modalità ridotta. Risposta basata su dati pre-calcolati.")

Logica dispatch:
1. Prova backend attivo con timeout 10s.
2. Se `health_check()` falso O eccezione O timeout → log + usa fallback.
3. Se fallback usato → response include `was_fallback=true`, `fallback_reason="..."` e UI deve mostrare il messaggio configurato sopra la risposta.
4. Se anche il fallback fallisce → 503 con messaggio chiaro all'utente.

### 3.5 REST endpoints

**File:** `class-hm-chat-rest.php`

| Metodo | Path | Auth | Capability |
|---|---|---|---|
| POST | `/wp-json/hm-chat/v1/ask` | login + nonce | `hm_use_chat` |
| GET | `/wp-json/hm-chat/v1/quota` | login | `hm_use_chat` |
| GET | `/wp-json/hm-chat/v1/sessions/{id}` | login + ownership | `hm_use_chat` |
| DELETE | `/wp-json/hm-chat/v1/sessions/{id}` | login + ownership | `hm_use_chat` |

Request `/ask`:
```json
{ "message": "How good is Player X at Team Y?", "session_id": "uuid-or-null" }
```

Response (esempio):
```json
{
  "reply": "**Player X** at **Team Y**...",
  "session_id": "abc-123",
  "intent": "predict",
  "data": { "predicted_rating": 7.42 },
  "suggestions": ["When will Player X peak?"],
  "backend_used": "local",
  "was_fallback": false,
  "quota_remaining": 472
}
```

- [ ] `quota_remaining` SEMPRE incluso (utile per UI).
- [ ] 402 se quota esaurita.
- [ ] 403 se manca capability.
- [ ] 503 se tutti i backend down.

### 3.6 Backend A — `local` (Scenario C, **default attivo al lancio**)

**File:** `class-hm-chat-backend-local.php`

**Idea:** zero hosting esterno. Tutto in WordPress.

#### 3.6.1 Tabella `wp_hm_predictions`
```sql
CREATE TABLE {$wpdb->prefix}hm_predictions (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  intent VARCHAR(32) NOT NULL,           -- 'predict'|'trajectory'|'peak'|'best_team'|'best_player'
  player_hm VARCHAR(20) NULL,
  team_hm VARCHAR(20) NULL,
  season VARCHAR(10) NULL,
  payload LONGTEXT NOT NULL,             -- JSON
  computed_at DATETIME NOT NULL,
  model_version VARCHAR(32) NOT NULL,
  INDEX idx_lookup (intent, player_hm, team_hm, season),
  INDEX idx_model_version (model_version)
) {$charset_collate};
```

#### 3.6.2 Intent detection PHP
- [ ] Regex/keyword matching per intent: `predict|how good`, `trajectory|career`, `peak|when peak`, `best team|fit for`, `compare|vs`.
- [ ] Entity extraction:
  - Player: lookup full-text contro `Anagrafiche` tramite `HM_Search::search()` (riusa codice esistente).
  - Team: stesso meccanismo.
- [ ] Se entità non risolte → risposta di clarification ("Quale giocatore intendi?") + suggestions.

#### 3.6.3 Importer modello pre-calcolato
**File:** `class-hm-chat-model-importer.php`

- [ ] WP-CLI command: `wp hm-chat import-model <path-to-json>`
- [ ] Endpoint admin: upload file `.json.gz` da pannello admin.
- [ ] Il JSON atteso ha struttura prodotta da `AI_Model` (vedi §6.1).
- [ ] Import in batch (500 righe/transaction) per non saturare memoria.
- [ ] Versioning: `wp_options: hm_chat_local_model_version` = checksum SHA256 del file.
- [ ] Soft delete: nuovo import non cancella subito le predizioni vecchie; switch atomico via `model_version` solo dopo import OK.

### 3.7 Backend B — `fastapi` (Scenario B)

**File:** `class-hm-chat-backend-fastapi.php`

- [ ] Legge `HM_CHAT_FASTAPI_URL` da `wp-config.php` (constants > options per sicurezza).
- [ ] Legge `HM_CHAT_FASTAPI_KEY` da `wp-config.php` (header `X-API-Key`).
- [ ] Chiama `POST {URL}/api/v1/chat` con payload `{message, session_id, user_context}`.
- [ ] `user_context` include `user_id_hash` (sha256 user_id) per persistenza sessione lato server.
- [ ] Timeout 10s, retry 1x.
- [ ] `health_check()`: `GET {URL}/health` con cache 60s.
- [ ] **NESSUNA api_key in `data-*` HTML** (fix critico identificato nel README di `AI_Model/wp_chat_widget`).

### 3.8 Backend C — `deepseek` (Scenario A)

**File:** `class-hm-chat-backend-deepseek.php`

**Pattern:** chiama prima il backend dati custom (FastAPI di `AI_Model`) per ottenere i numeri, poi DeepSeek per il "wrapping" linguistico.

- [ ] Legge `HM_CHAT_DEEPSEEK_API_KEY` da `wp-config.php`.
- [ ] Modello: `deepseek-chat` (configurabile).
- [ ] System prompt RAG-style:
  ```
  Sei un analista di pallacanestro europeo. Rispondi SOLO usando i dati
  numerici forniti nel context. Se manca un dato, dillo esplicitamente.
  NON inventare statistiche. Lingua: rispondi nella stessa lingua usata
  dall'utente (auto-detect).
  ```
- [ ] Tracking token in/out → log per cost monitoring.
- [ ] Timeout 20s.
- [ ] Costo medio atteso: ~0,002-0,005€/msg → 500 msg = ~1-2,5€/mese per utente All-Star → margine >85%.

### 3.9 Widget chat — shortcode `[hm_chat]`

**Files:**
- `templates/chat-widget.php`
- `assets/js/hm-chat-widget.js`
- `assets/css/hm-chat-widget.css`

- [ ] Shortcode: `[hm_chat height="520" title="Basketball AI"]`
- [ ] Se utente non ha `hm_use_chat` → mostra card "Upgrade a Starter per usare la chat AI" con CTA.
- [ ] Header widget mostra: titolo + badge "X / Y messaggi rimasti" (Y = 200 per Starter, 500 per All-Star).
- [ ] Bubble assistant mostra badge backend usato (`local`/`fastapi`/`deepseek`) — utile per debug, nascondibile in prod.
- [ ] Se `was_fallback=true` → banner giallo sopra la risposta con `fallback_message` configurato.
- [ ] Chips di suggestion sotto ogni risposta assistant.
- [ ] Session ID in `localStorage` (`hm_chat_session_id`).
- [ ] **NO chiamata diretta a backend esterno** — sempre via `/wp-json/hm-chat/v1/ask`.

### 3.10 Pannello admin

**File:** `class-hm-chat-admin.php`

Menu: `Impostazioni → HoopMetrics Chat`

Tab 1 — **Backend**:
- Radio: backend attivo (Local / FastAPI / DeepSeek)
- Radio: backend fallback (Local / Nessuno)
- Textarea: messaggio fallback custom
- Tabella health check live (con bottone "Test now")

Tab 2 — **Quote**:
- Stat globale: msg totali questo mese, top 10 consumer
- Bottone "Reset quote ora" (con conferma)

Tab 3 — **Modello locale**:
- Versione modello attivo + data import
- Form upload nuovo modello JSON
- Bottone "Verifica integrità"

Tab 4 — **Log**:
- Ultimi 100 messaggi (paginati) con: timestamp, user, intent, backend, was_fallback, latency_ms
- Filtro per data/utente/backend
- Export CSV admin-only (per debug, NON esposto a utenti)

### 3.11 Test plugin chat

- [ ] Unit test: ogni backend implementa l'interfaccia correttamente.
- [ ] Integration test: router fa fallback corretto quando backend principale lancia eccezione.
- [ ] E2E test: utente Starter consuma 200 msg → al 201° riceve 402 con CTA upgrade All-Star.
- [ ] E2E test: utente All-Star consuma 500 msg → al 501° riceve 402 con CTA.
- [ ] Smoke test: utente Bench chiama `/ask` → riceve 403.

---

---

## 4. (Rimossa — Scouting video non previsto in questa versione)

---

## 5. Donazioni (modello di pagamento)

> ⚠️ **Il pagamento per i piani Starter e All-Star è strutturato come donazione, NON come abbonamento.** Questo implica: nessun vincolo contrattuale, nessuna fatturazione automatica vincolante (annullabile in qualsiasi momento), nessun rimborso dovuto. L'utente dona liberamente e in cambio riceve accesso alle feature premium.

- [ ] Creare **Stripe Payment Link** per le donazioni:
  - Donazione ricorrente Starter: 10€/mese (annullabile in qualsiasi momento)
  - Donazione ricorrente All-Star: 20€/mese (annullabile in qualsiasi momento)
  - Donazione una tantum libera: 5/10/25/50€
- [ ] Aggiungere pagina `/sostieni/` con embed del payment link + thank-you text.
- [ ] Link in footer + nel pannello account.
- [ ] Wording chiaro ovunque: "Donazione", "Sostieni il progetto", "Contributo volontario". **Mai** usare "abbonamento", "subscription", "piano a pagamento".

---

## 6. Modifiche a `Elrond0192/AI_Model`

### 6.1 Export modello per backend Local (Scenario C)

**Nuovo CLI mode:**
```bash
python main.py --mode export-chat-model --out chat_model.json.gz
```

**File da creare:** `basketball_ai/export/chat_model_export.py`

Struttura JSON output:
```json
{
  "model_version": "sha256-...",
  "generated_at": "2026-01-15T00:00:00Z",
  "predictions": [
    {
      "intent": "predict",
      "player_hm": "hm_abc123",
      "team_hm": "hm_xyz789",
      "season": "2024-25",
      "payload": { "predicted_rating": 7.42, "confidence": 0.81 }
    },
    {
      "intent": "trajectory",
      "player_hm": "hm_abc123",
      "season": "2024-25",
      "payload": { "ages": [20,21,...], "ratings": [6.1, 6.5, ...] }
    },
    { "intent": "peak", ... },
    { "intent": "best_team", ... },
    { "intent": "best_player", ... }
  ]
}
```

- [ ] Generare predizioni per top N giocatori × top N team (es. 500 × 100 = 50k righe).
- [ ] Comprimere gzip (target file ≤ 50MB).
- [ ] Schema versionato (`schema_version: 1`).

### 6.2 Improvements FastAPI (per scenari B/A)

- [ ] **Persistenza sessioni chat su SQLite/Redis** anziché in-memory (`basketball_ai/chat/session_store.py`).
- [ ] Endpoint `/api/v1/chat` accetta header `X-User-Hash` (lega sessioni a utente WP).
- [ ] Aggiungere rate limit per IP+user_hash.

### 6.3 ~~Deprecazione~~ Rimozione `wp_chat_widget` ✅

- [x] `AI_Model/wp_chat_widget/` **eliminato** dal repository.
- [x] Sostituito da `hoopmetrics-chat` (vedi §3).

---

## 7. Roadmap completa

### Fase 0 — Sicurezza preliminare (1 settimana)
- [ ] §1.1 Scope nella cache
- [ ] §1.2 Separazione scope shots/lineups
- [ ] §1.3 Rate limiter per-utente
- [ ] §1.4 Audit log esteso

### Fase 1 — Paywall base + watchlist/comparisons (2 settimane)
- [ ] §2.1 PMP + Stripe configurati (donazioni)
- [ ] §2.2 Ruoli + capability
- [ ] §2.3 Helper PHP
- [ ] §2.4 `HM_Proxy::resolve_scope()` esteso
- [ ] §2.5 Token service esteso
- [ ] §2.6 Watchlist + Comparisons
- [ ] §2.7 Gating UI template
- [ ] §2.8 Paywall overlay
- [ ] §2.9 Pagine pricing (donazioni) + account
- [ ] §2.10 Test non-regressione
- [ ] §5 Donazioni
- 🚀 **LANCIO Starter (donazione 10€) e All-Star (donazione 20€ ma senza chat)**

### Fase 2 — Chat AI pluggabile (3 settimane)
- [ ] §6.1 Export modello AI_Model
- [ ] §3.1-3.5 Plugin `hoopmetrics-chat` (core + REST + quota + router)
- [ ] §3.6 Backend Local (default attivo)
- [ ] §3.7 Backend FastAPI (codice pronto, non attivato)
- [ ] §3.8 Backend DeepSeek (codice pronto, non attivato)
- [ ] §3.9 Widget shortcode
- [ ] §3.10 Pannello admin
- [ ] §3.11 Test
- 🚀 **Starter e All-Star completi con chat (backend Local)**

### Fase 3 — Scaling (opzionale, post-validation)
- [ ] Migrazione backend chat: Local → FastAPI → DeepSeek (cambio da admin, no codice).
- [ ] §6.2 Persistenza sessioni FastAPI.
- [x] §6.3 Rimozione `wp_chat_widget` ✅ (già completato).
- [ ] Analytics conversion funnel.

---

## 8. Linee guida per l'agente implementatore (Claude Sonnet 4.6)

### 8.1 Vincoli tecnici inderogabili
1. **Mai modificare** firme pubbliche di classi/metodi esistenti in `hoopmetrics-api`. Aggiungere parametri sempre come opzionali con default.
2. **Mai cambiare** la struttura della cache `hm_*_v{N}_*` se non aggiungendo segmenti (§1.1).
3. **Mai introdurre** chiamate HTTP sincrone bloccanti nel critical path delle pagine (usare WP-Cron / async).
4. **Mai esporre** dati DB raw via API senza passare da `HM_Anonymizer`.
5. **Mai stampare** API key, token, password in HTML/JS lato client.
6. **Mai eliminare** file/classi esistenti — usare flag `@deprecated` se necessario.

### 8.2 Convenzioni codice (da `README.md` esistente)
- PHP: `esc_html__()`, `esc_attr()`, `esc_url()`, `sanitize_text_field()`, `wp_unslash()`.
- JS: vanilla ES2017+, no framework, no bundler.
- CSS: variabili custom, prefisso namespace (`hm-`, `hm-chat-`, `hm-paywall-`).
- DB: sempre `dbDelta()` per migrations, mai `CREATE TABLE` raw.
- Translation-ready: ogni stringa user-facing in `__()` con text-domain del plugin.

### 8.3 Ordine di esecuzione consigliato
Eseguire le fasi **strettamente in ordine** (0 → 1 → 2 → 3). Ogni fase è un set di PR separabili.
**Ogni task** della checklist dovrebbe essere una commit atomica con messaggio descrittivo.
**Ogni sezione** (§1, §2.x, §3.x, §4.x) dovrebbe diventare una PR separata per facilità di review.

### 8.4 Test obbligatori prima di ogni merge
- [ ] Sito accessibile senza login (utente anonimo) funziona identico a prima.
- [ ] Disattivazione di PMP non causa fatal errors.
- [ ] Disattivazione di `hoopmetrics-chat` non causa fatal errors in `hoopmetrics-api` o nel tema.
- [ ] Tutti i nonce check + capability check presenti su endpoint REST premium.

### 8.5 Cose che l'agente NON deve fare
- ❌ Implementare export CSV/Excel/PDF.
- ❌ Esporre chat al di fuori dei piani Starter e All-Star.
- ❌ Mettere API key in `data-*` HTML o in `wp_options`.
- ❌ Modificare il behavior degli utenti anonimi (deve restare = oggi).
- ❌ Aprire PR multi-fase: una PR per sezione, max.
- ❌ Skippare le migrations `dbDelta()`.
- ❌ Aggiungere dipendenze JS frontend (no React/Vue/etc).

---

## 9. Open questions (da risolvere on-the-fly o in revisione)

- [x] ~~Branding piani: tenere "Rookie/Pro" o usare "Bench/Starter/All-Star"?~~ → **Deciso: Bench/Starter/All-Star**
- [ ] Rollover messaggi chat non usati: sì o no?
- [ ] Fatturazione: serve fattura italiana con P.IVA? (Stripe Tax + plugin "WP Invoice Italian") — N.B. essendo donazione, valutare se serve ricevuta di donazione.
- [x] ~~Lingua chat: solo IT, solo EN, o auto-detect?~~ → **Deciso: auto-detect**
- [ ] Privacy policy: serve update per chat (DeepSeek = trasferimento dati extra-UE) — consultare legale.

---

## 10. Changelog di questo documento

| Data | Modifica |
|---|---|
| Iniziale | Definito paywall a 3 piani con plugin chat pluggabile e scouting video BB_Analyzer |
| v2.1 | Rimosso scouting video. Piani rinominati Bench/Starter/All-Star. Lingua chat: auto-detect. Pagamento come donazione (non abbonamento). |
| v2.2 | Lineup analysis, on/off court e giocatori simili spostati nel piano Bench (accessibili a tutti). `read:sensitive` ora incluso in tutti i piani. |
| v2.3 | Chat AI estesa al piano Starter con quota 200 msg/mese (All-Star resta 500 msg/mese). |
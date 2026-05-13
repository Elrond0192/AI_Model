"""Intent detection for the basketball chat engine.

Uses a TF-IDF character-ngram nearest-neighbour classifier over a curated
bank of multilingual example utterances.  No regular expressions are used,
making the classifier language-agnostic and easy to extend: just add more
examples to ``_EXAMPLES`` in any language.

The classifier is initialised lazily on the first call to ``detect_intent``
and then cached for the lifetime of the process.
"""
from __future__ import annotations

import functools
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


class Intent(str, Enum):
    PREDICT      = "predict"        # How will X perform at Y?
    TRAJECTORY   = "trajectory"     # X's career arc / age curve
    PEAK         = "peak"           # When will X reach their peak?
    TRANSFER     = "transfer"       # What if X moved from A to B?
    LINEUP       = "lineup"         # X at team Y with lineup [A, B, C, D]
    COMPARE      = "compare"        # Compare X across teams
    BEST_TEAMS   = "best_teams"     # Best teams for X?
    BEST_PLAYERS = "best_players"   # Best players for team Y?
    TEAMMATES    = "teammates"      # What if X had elite teammates?
    CLUTCH       = "clutch"         # How is X in clutch situations?
    MARKET_VALUE = "market_value"   # What is X's market value?
    INDIVIDUAL   = "individual"     # X's individual stats profile
    ROLE_FIT     = "role_fit"       # Is X a good stretch big / point forward?
    HELP         = "help"           # What can you do?
    UNKNOWN      = "unknown"


# ---------------------------------------------------------------------------
# Multilingual example bank  (IT · EN · ES · FR · DE)
# Each list should contain at least 6–10 diverse phrasings per language.
# ---------------------------------------------------------------------------

_EXAMPLES: Dict[Intent, List[str]] = {
    Intent.PREDICT: [
        # IT
        "Che statistiche potrebbe avere nel Panathinaikos?",
        "Come si comporterebbe in questa squadra?",
        "Che rendimento avrebbe in quella squadra?",
        "Predici il rating di questo giocatore",
        "Quanto potrebbe rendere alla Virtus Bologna?",
        "Quanto segnerebbe all'Olimpia Milano?",
        "Che numeri potrebbe fare nel Real Madrid?",
        "Come giocherebbe all'ASVEL?",
        "Qual è il rating previsto per questo giocatore?",
        "Stimami le prestazioni al Fenerbahce",
        "Che voto gli daresti in quella squadra?",
        "Come renderebbe al Bayern Monaco?",
        # EN
        "How good is he at the Lakers?",
        "Predict his performance at this team",
        "What rating would he get at Real Madrid?",
        "How well would she play at that club?",
        "Estimate his performance if he joins this team",
        "What score would this player get here?",
        "How would he do at Barcelona?",
        "What would his stats look like at that team?",
        "Give me his predicted rating",
        "How good would he be at Bayern?",
        # ES
        "Cómo rendiría en el Barcelona?",
        "Cuál sería su rendimiento en este equipo?",
        "Predice su rating en el Real Madrid",
        "Cuántos puntos podría meter en este equipo?",
        # FR
        "Comment se comporterait-il dans cette équipe?",
        "Quelle serait sa performance à l'ASVEL?",
        "Prédis ses statistiques dans ce club",
        # DE
        "Wie gut wäre er bei Bayern München?",
        "Was wären seine Statistiken bei diesem Team?",
    ],
    Intent.TRAJECTORY: [
        # IT
        "Mostrami la traiettoria di carriera",
        "Come evolverà con l'età?",
        "Come migliorerà nel tempo?",
        "Mostrami la curva età del giocatore",
        "Come cambierà il suo rendimento con gli anni?",
        "Proiezione carriera per i prossimi anni",
        "Evoluzione del rendimento nel corso della carriera",
        "Curva di sviluppo per età",
        "Come sarà il suo arco di carriera?",
        # EN
        "Show me his career trajectory",
        "What is his age curve?",
        "How will he evolve over time?",
        "Career arc projection",
        "How does his performance change with age?",
        "Show me the aging curve",
        "Career development over time",
        "How will he develop in the next five years?",
        # ES
        "Muéstrame su trayectoria de carrera",
        "Cómo evolucionará con la edad?",
        # FR
        "Montre-moi la trajectoire de sa carrière",
        "Comment évoluera-t-il avec l'âge?",
        # DE
        "Zeig mir seine Karrierekurve",
    ],
    Intent.PEAK: [
        # IT
        "Quando raggiungerà il picco della carriera?",
        "A che età sarà al massimo?",
        "Quando sarà al suo meglio?",
        "Qual è il suo picco previsto?",
        "Quando toccherà il massimo delle prestazioni?",
        "Quanti anni ha ancora prima del picco?",
        "Stagione migliore prevista",
        "Quando arriverà al prime?",
        # EN
        "When will he reach his peak?",
        "At what age will he be best?",
        "What is his career high projection?",
        "When is his prime?",
        "How many seasons until his peak?",
        "Best age prediction",
        "When will he be at his best?",
        # ES
        "Cuándo llegará a su pico de rendimiento?",
        "A qué edad estará en su mejor momento?",
        # FR
        "Quand atteindra-t-il son pic de performance?",
        # DE
        "Wann erreicht er seinen Leistungshöhepunkt?",
    ],
    Intent.TRANSFER: [
        # IT
        "Se andasse alla Virtus Bologna cosa succederebbe?",
        "Se trasferisse al Panathinaikos quanto renderebbe?",
        "Impatto del trasferimento dal Barcelona al Real Madrid",
        "Cosa succederebbe se passasse all'Olimpia Milano?",
        "Simula il trasferimento all'ALBA Berlin",
        "Come cambierebbe le prestazioni spostandosi?",
        "Analisi del cambio squadra",
        "Effetto di un trasferimento al Fenerbahce",
        # EN
        "What if he moved from Lakers to Celtics?",
        "Simulate his transfer to Real Madrid",
        "Transfer impact from Barcelona to Valencia",
        "What would happen if he joined this team?",
        "How would his performance change if he transferred?",
        "What if he left his current team and joined another?",
        "Analyze the move from team A to team B",
        # ES
        "Qué pasaría si se transfiriera al Barcelona?",
        "Simula el traspaso al Real Madrid",
        # FR
        "Que se passerait-il s'il était transféré?",
        # DE
        "Was wäre wenn er zu Bayern wechseln würde?",
    ],
    Intent.LINEUP: [
        # IT
        "Costruisci un quintetto con lui al Fenerbahce",
        "Analizza la formazione con questi giocatori",
        "Come si integrerebbe in questo quintetto?",
        "Analisi con compagni specifici in squadra",
        "Quintetto composto da questi giocatori",
        "Come giocherebbe con questi compagni di squadra?",
        "Formazione con giocatore A, B, C e D",
        "Analizza il quintetto titolare",
        # EN
        "Build a starting five with him at Real Madrid",
        "Analyze the lineup with these players",
        "What if he played with player A and player B on that team?",
        "Lineup analysis for this roster",
        "Starting five composed of these players",
        "How would he fit in this lineup?",
        "Lineup with player A, B, C and D at this team",
        # ES
        "Analiza el quinteto con estos jugadores",
        "Forma el cinco inicial con él en el equipo",
        # FR
        "Analyse le cinq de départ avec ces joueurs",
        # DE
        "Analysiere das Starting Five mit diesen Spielern",
    ],
    Intent.COMPARE: [
        # IT
        "Confronta le prestazioni in diverse squadre",
        "Paragona le opzioni di squadra per questo giocatore",
        "Quale squadra è meglio per lui tra queste?",
        "Metti a confronto le squadre",
        "Comparazione tra più scenari di squadra",
        # EN
        "Compare his performance across teams",
        "Compare different team scenarios for him",
        "Which team is better for this player?",
        "Comparison across multiple teams",
        "Show me a comparison of his options",
        "Compare teams side by side",
        # ES
        "Compara su rendimiento en diferentes equipos",
        # FR
        "Compare ses performances dans différentes équipes",
        # DE
        "Vergleiche seine Leistung bei verschiedenen Teams",
    ],
    Intent.BEST_TEAMS: [
        # IT
        "Qual è la squadra migliore per lui?",
        "Migliori squadre per questo giocatore",
        "Dove potrebbe giocare al meglio?",
        "Dove renderebbe di più?",
        "Top squadre adatte a lui",
        "In quale squadra si adatterebbe meglio?",
        "Trovami le squadre più adatte",
        # EN
        "What are the best teams for him?",
        "Top team fits for this player",
        "Which team would suit him best?",
        "Best teams where he would thrive",
        "Where should he play?",
        "Find the best team fit for this player",
        # ES
        "Cuáles son los mejores equipos para él?",
        "Dónde podría rendir mejor?",
        # FR
        "Quelles sont les meilleures équipes pour lui?",
        # DE
        "Welches Team wäre am besten für ihn?",
    ],
    Intent.BEST_PLAYERS: [
        # IT
        "Quali sono i migliori giocatori per questa squadra?",
        "Chi si adatterebbe meglio a questa squadra?",
        "Migliori giocatori per il Fenerbahce",
        "Chi dovrebbe ingaggiare questa squadra?",
        # EN
        "Who are the best players for this team?",
        "Top players that would fit this team",
        "Who would suit this team best?",
        "Best player fits for this club",
        "Which players should this team sign?",
        # ES
        "Cuáles son los mejores jugadores para este equipo?",
        # FR
        "Quels joueurs conviendraient le mieux à cette équipe?",
        # DE
        "Welche Spieler würden am besten zu diesem Team passen?",
    ],
    Intent.TEAMMATES: [
        # IT
        "Come renderebbe con compagni più forti?",
        "Cosa succederebbe con compagni di élite?",
        "Impatto di compagni di squadra migliori",
        "Come cambierebbe con un team più forte?",
        "Cosa succede se ha compagni peggiori?",
        # EN
        "What if he had elite teammates?",
        "How would he perform with better teammates?",
        "Impact of teammate quality on his performance",
        "What if his teammates were worse?",
        "How does teammate quality affect him?",
        # ES
        "Cómo rendiría con mejores compañeros?",
        # FR
        "Comment se comporterait-il avec de meilleurs coéquipiers?",
        # DE
        "Wie würde er mit besseren Mitspielern abschneiden?",
    ],
    Intent.CLUTCH: [
        # IT
        "Come si comporta nei momenti cruciali?",
        "Prestazioni nei minuti finali",
        "È un giocatore clutch?",
        "Come gioca sotto pressione?",
        "Rendimento nelle situazioni decisive",
        "Statistiche nell'ultimo quarto",
        "Come si comporta nei momenti importanti?",
        # EN
        "How does he perform in clutch situations?",
        "Is he a clutch player?",
        "How does he play under pressure?",
        "Performance in final minutes",
        "Clutch statistics",
        "How is he in decisive moments?",
        "Big game performance",
        # ES
        "Cómo rinde en momentos decisivos?",
        # FR
        "Comment se comporte-t-il dans les moments décisifs?",
        # DE
        "Wie spielt er in entscheidenden Momenten?",
    ],
    Intent.MARKET_VALUE: [
        # IT
        "Quanto vale questo giocatore?",
        "Stima il valore di mercato",
        "Qual è il suo stipendio ideale?",
        "Quanto costerebbe acquistarlo?",
        "Valore di mercato attuale",
        "Quanto vale sul mercato?",
        # EN
        "What is his market value?",
        "Estimate his transfer value",
        "How much is he worth?",
        "What salary should he earn?",
        "Market value estimate",
        "How much would he cost?",
        # ES
        "Cuánto vale en el mercado?",
        # FR
        "Quelle est sa valeur marchande?",
        # DE
        "Was ist sein Marktwert?",
    ],
    Intent.INDIVIDUAL: [
        # IT
        "Statistiche individuali del giocatore",
        "Dimmi del profilo di questo giocatore",
        "Raccontami di lui",
        "Profilo statistico individuale",
        "Mostrami i suoi numeri",
        "Dati individuali di questo giocatore",
        "Chi è questo giocatore?",
        "Quanti punti segna in media?",
        "Quanti assist fa a partita?",
        # EN
        "Show me his individual stats",
        "Tell me about this player",
        "Individual player profile",
        "What are his stats?",
        "Give me his career numbers",
        "Player statistics overview",
        "How many points does he average?",
        # ES
        "Muéstrame sus estadísticas individuales",
        "Cuáles son sus estadísticas?",
        # FR
        "Montre-moi ses statistiques individuelles",
        # DE
        "Zeig mir seine persönlichen Statistiken",
    ],
    Intent.ROLE_FIT: [
        # IT
        "È adatto a fare il playmaker?",
        "Può fare il lungo da tre punti?",
        "Si adatta al ruolo di ala difensiva?",
        "Può giocare come tiratore?",
        "Qual è il suo ruolo migliore?",
        "È adatto come pivot?",
        "Può fare il play basso?",
        # EN
        "Is he a good fit as a stretch big?",
        "Can he play point forward?",
        "Is he suited for the role of playmaker?",
        "Is he a good spacer?",
        "Can he play as lead guard?",
        "What role suits him best?",
        "Is he good as a two-way wing?",
        # ES
        "Es adecuado para el rol de base?",
        # FR
        "Est-il adapté au rôle de meneur?",
        # DE
        "Passt er als Spielmacher?",
    ],
    Intent.HELP: [
        # IT
        "Aiuto",
        "Cosa puoi fare?",
        "Come funzioni?",
        "Dimmi cosa sai fare",
        "Lista dei comandi",
        "Cosa posso chiederti?",
        "Mostrami le funzionalità",
        # EN
        "Help",
        "What can you do?",
        "How do I use this?",
        "Show me what you can do",
        "What commands are available?",
        "I need help",
        # ES
        "Ayuda",
        "Qué puedes hacer?",
        # FR
        "Aide",
        "Que peux-tu faire?",
        # DE
        "Hilfe",
        "Was kannst du?",
    ],
}


# ---------------------------------------------------------------------------
# Classifier  (lazy-initialised, cached for the process lifetime)
# ---------------------------------------------------------------------------

class _NLUClassifier:
    """TF-IDF character-ngram nearest-neighbour intent classifier."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        # Build the flat corpus with labels
        self._corpus: List[str] = []
        self._labels: List[Intent] = []
        for intent, examples in _EXAMPLES.items():
            for ex in examples:
                self._corpus.append(ex.lower())
                self._labels.append(intent)

        # Character bi- to 4-gram TF-IDF works across languages and handles
        # typos, morphological variations, and unseen words.
        self._vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )
        self._X = self._vec.fit_transform(self._corpus)
        self._cos = _cos

    def predict(self, text: str, top_k: int = 5, threshold: float = 0.08) -> Intent:
        """Return the best-matching Intent for *text*.

        The method vectorises the query, computes cosine similarity against
        all examples, and takes a weighted-vote among the top-*k* matches.
        Returns ``Intent.UNKNOWN`` when the best similarity is below
        *threshold* (query is too different from all examples).
        """
        q_vec = self._vec.transform([text.lower()])
        sims  = self._cos(q_vec, self._X).flatten()

        top_idx = np.argsort(sims)[-top_k:][::-1]
        best_sim = float(sims[top_idx[0]])
        if best_sim < threshold:
            return Intent.UNKNOWN

        # Weighted vote: each of the top-k examples votes for its intent
        # weighted by its similarity score.
        votes: Dict[Intent, float] = {}
        for idx in top_idx:
            intent = self._labels[idx]
            votes[intent] = votes.get(intent, 0.0) + float(sims[idx])

        return max(votes, key=lambda k: votes[k])


@functools.lru_cache(maxsize=1)
def _get_classifier() -> _NLUClassifier:
    """Return the singleton NLU classifier (built once, cached forever)."""
    return _NLUClassifier()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_intent(text: str) -> Intent:
    """Return the most appropriate Intent for *text* using natural language.

    The function uses a TF-IDF nearest-neighbour classifier over a bank of
    multilingual example utterances.  No regular expressions are involved.
    """
    if not text or not text.strip():
        return Intent.UNKNOWN
    return _get_classifier().predict(text.strip())


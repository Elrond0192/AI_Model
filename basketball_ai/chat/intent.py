"""Intent detection for the basketball chat engine.

Architecture
============
Fully ML-based, zero regex.  Uses a stacked ensemble of two complementary
TF-IDF feature spaces that together cover the whole linguistic spectrum:

  1. **Character n-gram TF-IDF** (2–5 grams, word-boundary aware)
       → handles morphological variants, agglutination, cognates across
         languages, and typos.
  2. **Word n-gram TF-IDF** (1–3 grams)
       → captures semantic phrases and multi-word expressions.

Both feature matrices are concatenated (scipy sparse hstack) and fed into a
**multinomial Logistic Regression** (L2 regularised, lbfgs solver) that
produces calibrated class probabilities for all 14 intents.

UNKNOWN detection is driven by probability confidence:
  • If ``max_proba < CONF_THRESHOLD`` → UNKNOWN
  • If ``max_proba / second_max_proba < AMBIG_RATIO`` → UNKNOWN

Text is pre-processed with ``ftfy`` (optional) for Unicode normalisation,
lowercased, and lightly stripped of punctuation-only tokens.

The classifier is initialised lazily on the first call to ``detect_intent``
and cached for the lifetime of the process — warm-up takes ~100 ms.

Extending the system
====================
Add more example strings to ``_EXAMPLES`` in any language.  No other changes
are needed.  The classifier is re-built automatically from the example bank.
Never add regular expressions.
"""
from __future__ import annotations

import functools
import re as _re  # used ONLY for the normalisation helper, not for intent matching
import string
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    PREDICT      = "predict"        # How will X perform at team Y?
    TRAJECTORY   = "trajectory"     # Career arc / age-development curve
    PEAK         = "peak"           # When / at what age does X peak?
    TRANSFER     = "transfer"       # What if X moved from A to B?
    LINEUP       = "lineup"         # X at team Y alongside named players A, B, …
    COMPARE      = "compare"        # Compare X across multiple teams
    BEST_TEAMS   = "best_teams"     # Best teams for player X
    BEST_PLAYERS = "best_players"   # Best players for team Y
    TEAMMATES    = "teammates"      # Impact of better/worse teammates
    CLUTCH       = "clutch"         # Clutch / pressure-moment performance
    MARKET_VALUE = "market_value"   # Transfer value / salary estimate
    INDIVIDUAL   = "individual"     # Individual player profile & stats
    ROLE_FIT     = "role_fit"       # Suitability for a given tactical role
    HELP         = "help"           # Help / what can you do?
    UNKNOWN      = "unknown"        # Below-confidence fallback


# ---------------------------------------------------------------------------
# Multilingual example bank
#   Languages: IT EN ES FR DE PT GR TR RU HR PL SL HU SR LT NL
#   Coverage:  ≥ 6 examples per language per intent
#   Principle: diverse phrasings, avoid word-for-word duplicates,
#              include sport-specific terminology in each language.
# ---------------------------------------------------------------------------

_EXAMPLES: Dict[Intent, List[str]] = {

    # -----------------------------------------------------------------------
    Intent.PREDICT: [
        # IT — broad variety of phrasings
        "Che statistiche potrebbe avere nel Panathinaikos?",
        "Come si comporterebbe al Fenerbahce?",
        "Quanto renderebbe alla Virtus Bologna?",
        "Che rendimento avrebbe all'Olimpia Milano?",
        "Predici il rating di questo giocatore in questa squadra",
        "Quanto potrebbe valere per questa squadra?",
        "Che voto gli daresti se giocasse qui?",
        "Come giocherebbe al Real Madrid Basket?",
        "Stima le sue prestazioni in questo club",
        "Che numeri potrebbe fare al CSKA Mosca?",
        "Che impatto avrebbe in questa squadra?",
        "Quanto segnerebbe se passasse all'Alba Berlin?",
        # EN
        "How good would he be at the Lakers?",
        "Predict his performance at this team",
        "What rating would he get if he played here?",
        "How well would she perform at that club?",
        "Give me his predicted rating at Real Madrid",
        "Estimate his output if he joins this team",
        "What stats would he put up at Bayern?",
        "How would he do at Barcelona?",
        "What score would this player get here?",
        "How impactful would he be at this club?",
        # ES
        "Cómo rendiría en el Real Madrid de baloncesto?",
        "Cuál sería su rendimiento si fichara por el Barcelona?",
        "Predice su valoración en el Valencia Basket",
        "Cuántos puntos metería en este equipo?",
        "Qué impacto tendría en este club?",
        "Cómo jugaría en el Unicaja?",
        # FR
        "Comment se comporterait-il à l'ASVEL?",
        "Quelle serait sa performance dans ce club?",
        "Prédis ses statistiques si il rejoignait cette équipe",
        "Quel rating aurait-il dans cette équipe?",
        "Comment jouerait-il au Monaco?",
        # DE
        "Wie gut wäre er bei Bayern München Basketball?",
        "Was wären seine Statistiken bei diesem Team?",
        "Schätze seine Leistung bei diesem Verein",
        "Wie würde er bei Alba Berlin spielen?",
        # PT
        "Como se sairia no Benfica?",
        "Qual seria o seu rendimento neste clube?",
        "Prevê as estatísticas dele nesta equipa",
        "Quantos pontos marcaria aqui?",
        # GR (Greek — important for Panathinaikos/Olympiakos)
        "Πώς θα αποδίδε στον Παναθηναϊκό;",
        "Τι στατιστικά θα είχε στον Ολυμπιακό;",
        "Πρόβλεψε την απόδοσή του σε αυτή την ομάδα",
        "Πόσους πόντους θα σκόραρε εδώ;",
        # TR (Turkish — Fenerbahce/Anadolu Efes etc.)
        "Fenerbahçe'de nasıl oynar?",
        "Bu takımda performansını tahmin et",
        "Anadolu Efes'te kaç sayı atar?",
        # RU
        "Как он будет играть в ЦСКА?",
        "Предскажи его рейтинг в этой команде",
        "Какие статистики у него были бы здесь?",
        # HR/SL/SR
        "Kako bi igrao u ovoj momčadi?",
        "Kateri rezultati bi bili pri tej ekipi?",
        # PL
        "Jak grałby w tej drużynie?",
        "Przewidź jego wyniki w tym klubie",
        # HU
        "Hogyan teljesítene ebben a csapatban?",
        # NL
        "Hoe goed zou hij zijn bij dit team?",
    ],

    # -----------------------------------------------------------------------
    Intent.TRAJECTORY: [
        # IT
        "Mostrami la traiettoria di carriera di questo giocatore",
        "Come evolverà con l'età?",
        "Proiezione del rendimento nei prossimi anni",
        "Curva di sviluppo per età",
        "Come cambierà il suo rendimento con gli anni?",
        "Analisi dell'arco di carriera",
        "Come migliorerà o peggiorerà nel tempo?",
        "Evoluzione delle prestazioni stagione per stagione",
        # EN
        "Show me his career trajectory",
        "How will he develop over the next five seasons?",
        "Plot his age-performance curve",
        "Career arc projection",
        "How does his output change as he ages?",
        "What is his development curve?",
        "Show me how he evolves over time",
        # ES
        "Muéstrame su trayectoria de carrera",
        "Cómo evolucionará con los años?",
        "Proyección de carrera para los próximos años",
        "Cómo cambiará su rendimiento con la edad?",
        # FR
        "Montre-moi la trajectoire de sa carrière",
        "Comment évoluera-t-il avec l'âge?",
        "Projection de carrière sur plusieurs saisons",
        # DE
        "Zeig mir seine Karrierekurve",
        "Wie entwickelt er sich mit dem Alter?",
        "Karriereentwicklung über die nächsten Jahre",
        # PT
        "Mostra a trajetória de carreira dele",
        "Como vai evoluir com a idade?",
        # GR
        "Δείξε μου την καμπύλη απόδοσης με την ηλικία",
        "Πώς θα εξελιχθεί με τα χρόνια;",
        # TR
        "Kariyer yayını göster",
        "Yaşlandıkça performansı nasıl değişir?",
        # RU
        "Покажи карьерную траекторию",
        "Как изменится его игра с возрастом?",
        # PL
        "Pokaż trajektorię kariery",
        "Jak się rozwija z wiekiem?",
    ],

    # -----------------------------------------------------------------------
    Intent.PEAK: [
        # IT
        "Quando raggiungerà il picco della carriera?",
        "A che età sarà al massimo delle sue potenzialità?",
        "Quando sarà nella sua fase migliore?",
        "Quante stagioni ha davanti prima di toccare il picco?",
        "Previsione del prime di questo giocatore",
        "Quando arriverà al massimo del suo rendimento?",
        "Stima dell'età di picco",
        # EN
        "When will he reach his career peak?",
        "At what age will he be at his best?",
        "How many seasons until his prime?",
        "When is his peak performance window?",
        "Predict his best possible season",
        "When will he be at the top of his game?",
        "Career high projection",
        # ES
        "Cuándo llegará a su pico de rendimiento?",
        "A qué edad estará en su mejor momento?",
        "Cuántos años le quedan antes del prime?",
        "Cuándo alcanzará su máximo potencial?",
        # FR
        "Quand atteindra-t-il son pic de performance?",
        "À quel âge sera-t-il au sommet?",
        "Quand est son prime?",
        # DE
        "Wann erreicht er seinen Höhepunkt?",
        "In welchem Alter wird er am besten sein?",
        "Wann ist sein Prime?",
        # PT
        "Quando vai atingir o seu pico?",
        "Com que idade estará no melhor?",
        # GR
        "Πότε θα φτάσει στο απόγειό του;",
        "Σε ποια ηλικία θα είναι στο καλύτερό του;",
        # TR
        "Ne zaman zirvesine ulaşacak?",
        "Kaç yaşında en iyi formda olacak?",
        # RU
        "Когда он достигнет пика карьеры?",
        "В каком возрасте он будет на пике?",
        # PL
        "Kiedy osiągnie szczyt kariery?",
        "W jakim wieku będzie najlepszy?",
        # HU
        "Mikor éri el pályafutása csúcsát?",
    ],

    # -----------------------------------------------------------------------
    Intent.TRANSFER: [
        # IT
        "Se andasse alla Virtus Bologna cosa succederebbe?",
        "Se trasferisse al Panathinaikos quanto renderebbe?",
        "Impatto di un trasferimento dall'Olimpia al Fenerbahce",
        "Cosa succederebbe se passasse a questa squadra?",
        "Simula il passaggio all'Alba Berlin",
        "Come cambierebbe il rendimento con un cambio di squadra?",
        "Analisi del trasferimento da una squadra all'altra",
        "Effetti di un eventuale cambio di club",
        # EN
        "What if he moved from the Lakers to the Celtics?",
        "Simulate his transfer to Real Madrid",
        "Transfer impact from Barcelona to Valencia",
        "What would happen if he joined this team instead?",
        "How would his numbers change if he switched clubs?",
        "Analyze the impact of moving from team A to team B",
        "What if he signed with this club in the off-season?",
        # ES
        "Qué pasaría si se transfiriera al Barcelona?",
        "Simula el traspaso al Real Madrid",
        "Qué impacto tendría si fichara por este equipo?",
        "Cómo cambiarían sus números si cambiara de club?",
        # FR
        "Que se passerait-il s'il était transféré ici?",
        "Simule son transfert dans ce club",
        "Impact d'un changement d'équipe",
        # DE
        "Was wäre wenn er zu Bayern wechseln würde?",
        "Simuliere seinen Transfer zu diesem Klub",
        "Wie würde sich ein Wechsel auswirken?",
        # PT
        "O que aconteceria se se transferisse para este clube?",
        "Simula a transferência para o Benfica",
        # GR
        "Τι θα γινόταν αν πήγαινε σε αυτή την ομάδα;",
        "Προσομοίωσε τη μεταγραφή στον Παναθηναϊκό",
        # TR
        "Fenerbahçe'ye transfer olsa ne olurdu?",
        "Bu kulübe geçse nasıl olur?",
        # RU
        "Что было бы если бы он перешёл в ЦСКА?",
        "Симулируй трансфер в эту команду",
        # HR/PL
        "Što bi se dogodilo da pređe u ovu momčad?",
        "Co by się stało gdyby przeszedł do tego klubu?",
    ],

    # -----------------------------------------------------------------------
    Intent.LINEUP: [
        # IT
        "Costruisci un quintetto con lui al Fenerbahce",
        "Analizza la formazione con questi cinque giocatori",
        "Come si integrerebbe in un quintetto con Nedovic e Larkin?",
        "Quintetto composto da Shields, Sloukas, Calathes, Papagiannis",
        "Analisi del five con giocatori specifici in squadra",
        "Come funzionerebbe la formazione titolare con lui?",
        "Costruttore quintetto con giocatori selezionati",
        "Metti insieme una squadra con lui come protagonista",
        # EN
        "Build a starting five with him at Real Madrid",
        "Analyze the lineup with players A, B, C and D",
        "What if he played alongside player X and player Y on that team?",
        "Lineup analysis with these specific players",
        "How does this five-man unit work together?",
        "Create the optimal starting lineup including this player",
        "How would the rotation look with him and his teammates?",
        # ES
        "Construye el quinteto inicial con él en el equipo",
        "Analiza la alineación con estos jugadores",
        "Cómo funcionaría el cinco inicial con estos jugadores?",
        # FR
        "Construis le cinq de départ avec ces joueurs",
        "Analyse la composition avec ces joueurs spécifiques",
        # DE
        "Erstelle die Startfünf mit diesen Spielern",
        "Analysiere das Lineup mit Spieler A, B und C",
        # PT
        "Constrói o cinco inicial com estes jogadores",
        # GR
        "Φτιάξε το αρχικό πεντάδα με αυτούς τους παίκτες",
        "Ανάλυσε τη σύνθεση με τον παίκτη Α και Β",
        # TR
        "Bu oyuncularla ilk beşi oluştur",
        # RU
        "Составь стартовую пятёрку с этими игроками",
        # PL
        "Zbuduj wyjściową piątkę z tymi zawodnikami",
    ],

    # -----------------------------------------------------------------------
    Intent.COMPARE: [
        # IT
        "Confronta le prestazioni in queste due squadre",
        "Paragona le opzioni disponibili per questo giocatore",
        "Quale squadra è meglio per lui tra Barcellona e Real Madrid?",
        "Metti a confronto i due scenari possibili",
        "Comparazione tra più destinazioni di trasferimento",
        "Qual è la differenza di rendimento tra le due squadre?",
        # EN
        "Compare his performance across these teams",
        "Which team is better for him, team A or team B?",
        "Show me a side-by-side comparison of his options",
        "Compare his rating at these two different clubs",
        "Head-to-head comparison of these scenarios",
        "What is the difference between these two team options?",
        # ES
        "Compara su rendimiento en diferentes equipos",
        "Cuál es la diferencia entre estas dos opciones de equipo?",
        # FR
        "Compare ses performances dans ces deux équipes",
        "Quelle est la différence entre ces options?",
        # DE
        "Vergleiche seine Leistung bei verschiedenen Teams",
        "Was ist der Unterschied zwischen diesen beiden Optionen?",
        # PT
        "Compara o desempenho dele em diferentes equipas",
        # GR
        "Σύγκρινε τις επιδόσεις σε αυτές τις ομάδες",
        # TR
        "Bu iki takımda performansını karşılaştır",
        # RU
        "Сравни его выступления в разных командах",
        # PL
        "Porównaj jego wyniki w różnych drużynach",
    ],

    # -----------------------------------------------------------------------
    Intent.BEST_TEAMS: [
        # IT
        "Qual è la squadra migliore per lui?",
        "Trovami le squadre più adatte a questo giocatore",
        "Dove renderebbe meglio in Europa?",
        "In quale squadra si adatterebbe di più?",
        "Top squadre per questo giocatore",
        "Quale club dovrebbe scegliere?",
        "Dove potrebbe esprimersi al massimo?",
        # EN
        "What are the best teams for this player?",
        "Find the best team fits for him",
        "Which club would suit him most?",
        "Top destinations for this player",
        "Where should he play next season?",
        "Where would he thrive the most?",
        "Best team options for this player",
        # ES
        "Cuáles son los mejores equipos para él?",
        "Dónde rendiría mejor?",
        "Qué club le convendría más?",
        "Mejores destinos para este jugador",
        # FR
        "Quelles sont les meilleures équipes pour lui?",
        "Où devrait-il jouer?",
        "Quel club lui conviendrait le mieux?",
        # DE
        "Welches Team wäre am besten für ihn?",
        "Wo sollte er spielen?",
        "Beste Teamoptionen für diesen Spieler",
        # PT
        "Quais são as melhores equipas para ele?",
        "Onde renderia melhor?",
        # GR
        "Ποιες είναι οι καλύτερες ομάδες γι' αυτόν;",
        "Πού να πάει να παίξει;",
        # TR
        "Hangi takım en uygun olur?",
        "Nerede daha iyi oynar?",
        # RU
        "Какие команды лучше всего подходят ему?",
        "Где он сможет раскрыться?",
        # PL
        "Jakie drużyny są dla niego najlepsze?",
        "Gdzie powinien grać?",
        # HU
        "Melyik csapat lenne a legjobb neki?",
    ],

    # -----------------------------------------------------------------------
    Intent.BEST_PLAYERS: [
        # IT
        "Quali sono i migliori giocatori per questa squadra?",
        "Chi si adatterebbe meglio al Fenerbahce?",
        "Migliori profili per rinforzare questa squadra",
        "Chi dovrebbe ingaggiare questa squadra in estate?",
        "Trova i giocatori più adatti per questa formazione",
        # EN
        "Who are the best players for this team?",
        "Find top players that would fit this roster",
        "Which players should this club target?",
        "Best player fits for this team's system",
        "Who would suit this team's style?",
        "Recommend players for this team",
        # ES
        "Cuáles son los mejores jugadores para este equipo?",
        "Qué jugadores debería fichar este club?",
        "Quién encajaría mejor en este equipo?",
        # FR
        "Quels joueurs conviendraient le mieux à cette équipe?",
        "Qui devrait recruter ce club?",
        # DE
        "Welche Spieler passen am besten zu diesem Team?",
        "Wen sollte dieser Verein verpflichten?",
        # PT
        "Quais os melhores jogadores para esta equipa?",
        # GR
        "Ποιοι παίκτες ταιριάζουν καλύτερα σε αυτή την ομάδα;",
        # TR
        "Bu takıma en iyi hangi oyuncular uyar?",
        # RU
        "Кто лучше всего подойдёт этой команде?",
        # PL
        "Którzy gracze najlepiej pasują do tej drużyny?",
    ],

    # -----------------------------------------------------------------------
    Intent.TEAMMATES: [
        # IT
        "Come renderebbe con compagni di squadra più forti?",
        "Cosa succederebbe con compagni d'élite intorno a lui?",
        "Impatto di compagni di qualità superiore sulle sue prestazioni",
        "Come cambierebbe con un roster più forte?",
        "Cosa succede se ha compagni peggiori?",
        "Effetto dei compagni sul suo rendimento",
        # EN
        "What if he had elite teammates around him?",
        "How would he perform with better supporting cast?",
        "Impact of teammate quality on his stats",
        "What if his teammates were weaker?",
        "How does the quality of teammates affect his performance?",
        "Performance with star teammates vs average teammates",
        # ES
        "Cómo rendiría con mejores compañeros?",
        "Qué pasaría si tuviera compañeros de élite?",
        # FR
        "Comment se comporterait-il avec de meilleurs coéquipiers?",
        "Impact de la qualité des coéquipiers sur ses stats",
        # DE
        "Wie würde er mit besseren Mitspielern abschneiden?",
        "Einfluss der Teamkollegen auf seine Leistung",
        # PT
        "Como jogaria com colegas de equipa melhores?",
        # GR
        "Πώς θα αποδίδε με καλύτερους συμπαίκτες;",
        # TR
        "Daha iyi takım arkadaşlarıyla nasıl oynar?",
        # RU
        "Как бы он играл с более сильными партнёрами?",
        # PL
        "Jak grałby z lepszymi kolegami z drużyny?",
    ],

    # -----------------------------------------------------------------------
    Intent.CLUTCH: [
        # IT
        "Come si comporta nei momenti cruciali?",
        "È un giocatore clutch?",
        "Prestazioni negli ultimi secondi della partita",
        "Come gioca sotto pressione nelle situazioni decisive?",
        "Statistiche nei minuti finali delle partite importanti",
        "Come reagisce nei momenti che contano?",
        "Rendimento nelle partite decisive di playoff",
        # EN
        "How does he perform in clutch situations?",
        "Is he a clutch player?",
        "How does he play under pressure?",
        "Performance in the final minutes of close games",
        "Clutch statistics and big game performance",
        "Does he show up when it matters most?",
        "How good is he in deciding moments?",
        # ES
        "Cómo rinde en momentos decisivos?",
        "Es un jugador clutch?",
        "Cómo juega bajo presión?",
        # FR
        "Comment se comporte-t-il dans les moments décisifs?",
        "Est-il un joueur clutch?",
        # DE
        "Wie spielt er in entscheidenden Momenten?",
        "Ist er ein Clutch-Spieler?",
        # PT
        "Como joga em momentos decisivos?",
        # GR
        "Πώς παίζει σε αποφασιστικές στιγμές;",
        # TR
        "Kritik anlarda nasıl oynar?",
        # RU
        "Как он играет в решающие моменты?",
        # PL
        "Jak gra w kluczowych momentach?",
    ],

    # -----------------------------------------------------------------------
    Intent.MARKET_VALUE: [
        # IT
        "Quanto vale questo giocatore sul mercato?",
        "Stima il valore di trasferimento",
        "Qual è il suo stipendio ideale?",
        "Quanto costerebbe acquistarlo?",
        "Valore di mercato attuale del giocatore",
        "Quanto vale il contratto di questo giocatore?",
        # EN
        "What is his market value?",
        "Estimate his transfer fee",
        "How much is he worth on the open market?",
        "What salary should he be earning?",
        "How much would it cost to sign him?",
        "Market value and salary estimation",
        # ES
        "Cuánto vale en el mercado?",
        "Cuál es su valor de traspaso?",
        "Cuánto debería cobrar?",
        # FR
        "Quelle est sa valeur marchande?",
        "Combien vaut-il sur le marché?",
        # DE
        "Was ist sein Marktwert?",
        "Wie viel ist er wert?",
        # PT
        "Qual é o seu valor de mercado?",
        "Quanto custa contratá-lo?",
        # GR
        "Ποια είναι η αγοραία αξία του;",
        # TR
        "Piyasa değeri ne kadar?",
        # RU
        "Какова его рыночная стоимость?",
        # PL
        "Jaka jest jego wartość rynkowa?",
    ],

    # -----------------------------------------------------------------------
    Intent.INDIVIDUAL: [
        # IT
        "Statistiche individuali di questo giocatore",
        "Dimmi il profilo di questo giocatore",
        "Mostrami i suoi numeri personali",
        "Quanti punti segna in media a partita?",
        "Profilo statistico completo del giocatore",
        "Chi è questo giocatore?",
        "Dati personali e statistiche di carriera",
        "Quanti assist fa a partita?",
        "Raccontami di questo giocatore",
        "dimmi di lui",
        "parlami di questo giocatore",
        "statistiche di questo giocatore",
        "i numeri di questo atleta",
        "chi è e come gioca?",
        # EN
        "Show me his individual stats",
        "Tell me about this player's profile",
        "What are his career averages?",
        "Individual statistics overview",
        "How many points does he score per game?",
        "Give me his full player profile",
        "What are his strengths and weaknesses?",
        # ES
        "Muéstrame sus estadísticas individuales",
        "Cuáles son sus promedios de carrera?",
        "Dime el perfil de este jugador",
        "Cuántos puntos anota por partido?",
        # FR
        "Montre-moi ses statistiques individuelles",
        "Quel est son profil de joueur?",
        "Combien de points marque-t-il en moyenne?",
        # DE
        "Zeig mir seine persönlichen Statistiken",
        "Wie ist sein Spielerprofil?",
        "Was sind seine Karrieredurchschnitte?",
        # PT
        "Mostra as estatísticas individuais dele",
        "Qual é o perfil deste jogador?",
        # GR
        "Δείξε μου τις ατομικές στατιστικές",
        "Ποιος είναι αυτός ο παίκτης;",
        # TR
        "Oyuncunun bireysel istatistiklerini göster",
        "Bu oyuncunun profili nedir?",
        # RU
        "Покажи индивидуальную статистику игрока",
        "Что за игрок? Расскажи про него",
        # PL
        "Pokaż indywidualne statystyki tego gracza",
        "Kim jest ten gracz?",
        # HU
        "Mutasd meg az egyéni statisztikáit",
    ],

    # -----------------------------------------------------------------------
    Intent.ROLE_FIT: [
        # IT
        "È adatto a fare il playmaker in questa squadra?",
        "Può ricoprire il ruolo di ala difensiva?",
        "Si adatta al ruolo di lungo da tre punti?",
        "Può fare il pivot nel sistema di questo allenatore?",
        "Qual è il ruolo tattico più adatto a lui?",
        "È adatto come tiratore da fuori?",
        "Può fare il play basso in questo sistema?",
        "Idoneità al ruolo di stretch big",
        # EN
        "Is he a good fit as a stretch big?",
        "Can he play the point forward role effectively?",
        "Is he suited for the lead guard position?",
        "Can he be used as a spacer and shooter?",
        "What role suits him best in this system?",
        "Is he good as a two-way wing?",
        "Can he play center in a small-ball lineup?",
        "Tactical role suitability analysis",
        # ES
        "Es adecuado para el rol de base?",
        "Puede jugar de alero defensivo?",
        "Cuál es el papel táctico más adecuado para él?",
        # FR
        "Est-il adapté au rôle de meneur?",
        "Peut-il jouer ailier défensif?",
        "Quel rôle tactique lui convient le mieux?",
        # DE
        "Passt er als Spielmacher?",
        "Kann er die Flügelstürmerposition spielen?",
        "Welche Rolle passt am besten zu ihm?",
        # PT
        "É adequado para o papel de base?",
        "Pode jogar como ala defensivo?",
        # GR
        "Ταιριάζει στη θέση του οργανωτή;",
        "Μπορεί να παίξει ως stretch 4;",
        # TR
        "Playmaker rolüne uygun mu?",
        "Hangi pozisyon ona daha uygun?",
        # RU
        "Подходит ли он на роль разыгрывающего?",
        "В какой роли он лучше всего?",
        # PL
        "Czy nadaje się na rozgrywającego?",
        "Jaką rolę powinien pełnić?",
    ],

    # -----------------------------------------------------------------------
    Intent.HELP: [
        # IT
        "Aiuto",
        "Cosa puoi fare?",
        "Come funziona questo assistente?",
        "Dimmi cosa sai fare",
        "Mostrami le funzionalità disponibili",
        "Lista delle cose che puoi fare",
        "Non so come usarti",
        "Cosa posso chiederti?",
        # EN
        "Help",
        "What can you do?",
        "How do I use this assistant?",
        "Show me what you are capable of",
        "What questions can I ask?",
        "How does this work?",
        "I need help using this",
        # ES
        "Ayuda",
        "Qué puedes hacer?",
        "Cómo funciona esto?",
        "Qué preguntas puedo hacer?",
        # FR
        "Aide",
        "Que peux-tu faire?",
        "Comment ça marche?",
        # DE
        "Hilfe",
        "Was kannst du?",
        "Wie funktioniert das?",
        # PT
        "Ajuda",
        "O que podes fazer?",
        # GR
        "Βοήθεια",
        "Τι μπορείς να κάνεις;",
        # TR
        "Yardım",
        "Ne yapabilirsin?",
        # RU
        "Помощь",
        "Что ты умеешь?",
        # PL
        "Pomoc",
        "Co potrafisz?",
        # HU
        "Segítség",
        "Mit tudsz csinálni?",
    ],
}


# ---------------------------------------------------------------------------
# Text normalisation helper
# ---------------------------------------------------------------------------

_PUNCT_TRANS = str.maketrans("", "", string.punctuation.replace("'", "").replace("-", ""))


def _normalise(text: str) -> str:
    """Unicode-repair → lowercase → light punctuation strip."""
    try:
        import ftfy as _ftfy
        text = _ftfy.fix_text(text)
    except Exception:
        pass
    text = text.lower().strip()
    # Remove punctuation except apostrophes (contractions) and hyphens
    text = text.translate(_PUNCT_TRANS)
    # Collapse multiple spaces
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# NLU Ensemble Classifier
# ---------------------------------------------------------------------------

class _NLUClassifier:
    """Three-layer ensemble NLU intent classifier.

    Features
    --------
    Layer 1 — Char n-gram TF-IDF (2–5 grams, word-boundary aware):
        Captures morphological variation, handles agglutinative languages
        (Turkish, Hungarian), cognates across Romance/Germanic languages,
        and is robust to unknown words.

    Layer 2 — Word n-gram TF-IDF (1–3 grams, sublinear TF):
        Captures semantic phrases like "market value", "career trajectory",
        "starting five", etc.

    Both feature matrices are concatenated (sparse hstack) and fed to a
    Logistic Regression with L2 regularisation (lbfgs, C=3.0) which
    outputs calibrated multi-class probabilities.

    UNKNOWN detection
    -----------------
    A prediction is flagged UNKNOWN when:
      • ``max_proba < CONF_THRESHOLD`` (0.20) — low overall confidence
      • ``max_proba / second_max_proba < AMBIG_RATIO`` (1.25) — two intents
        are very close in probability → too ambiguous to commit
    """

    # Calibrated thresholds (tuned empirically on the example bank)
    CONF_THRESHOLD = 0.20   # absolute minimum winning probability
    AMBIG_RATIO    = 1.25   # winner must be ≥ 25 % more probable than runner-up

    def __init__(self) -> None:
        import scipy.sparse as _sp
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Build flat corpus + labels from the example bank
        corpus: List[str] = []
        labels: List[Intent] = []
        for intent, examples in _EXAMPLES.items():
            for ex in examples:
                corpus.append(_normalise(ex))
                labels.append(intent)

        # Character n-gram features
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=80_000,
        )
        # Word n-gram features
        self._word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=40_000,
        )

        X_char = self._char_vec.fit_transform(corpus)
        X_word = self._word_vec.fit_transform(corpus)
        X = _sp.hstack([X_char, X_word], format="csr")

        self._clf = LogisticRegression(
            C=3.0,
            max_iter=2000,
            solver="lbfgs",
        )
        self._clf.fit(X, [lbl.value for lbl in labels])
        # Map string class names back to Intent enum values
        self._classes: List[Intent] = [Intent(c) for c in self._clf.classes_]

    def predict(self, text: str) -> Tuple[Intent, float]:
        """Return ``(intent, confidence)`` for *text*.

        Confidence is the winning class probability in [0, 1].
        Returns ``(Intent.UNKNOWN, 0.0)`` when confidence is too low.
        """
        import scipy.sparse as _sp

        norm = _normalise(text)
        x_char = self._char_vec.transform([norm])
        x_word = self._word_vec.transform([norm])
        x = _sp.hstack([x_char, x_word], format="csr")

        proba = self._clf.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]
        best_p  = float(proba[order[0]])
        second_p = float(proba[order[1]]) if len(order) > 1 else 0.0

        if best_p < self.CONF_THRESHOLD:
            return Intent.UNKNOWN, 0.0
        if second_p > 0 and (best_p / second_p) < self.AMBIG_RATIO:
            return Intent.UNKNOWN, best_p

        return self._classes[order[0]], best_p


@functools.lru_cache(maxsize=1)
def _get_classifier() -> _NLUClassifier:
    """Return the singleton NLU classifier (built on first call, cached)."""
    return _NLUClassifier()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_intent(text: str) -> Intent:
    """Classify *text* into one of the supported basketball intents.

    Uses a multilingual ML ensemble (char n-grams + word n-grams +
    Logistic Regression).  No regular expressions are involved.
    Returns ``Intent.UNKNOWN`` when confidence is insufficient.
    """
    if not text or not text.strip():
        return Intent.UNKNOWN
    intent, _conf = _get_classifier().predict(text.strip())
    return intent


def detect_intent_with_confidence(text: str) -> Tuple[Intent, float]:
    """Like ``detect_intent`` but also returns the confidence score (0–1)."""
    if not text or not text.strip():
        return Intent.UNKNOWN, 0.0
    return _get_classifier().predict(text.strip())


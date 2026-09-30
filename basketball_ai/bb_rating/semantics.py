"""Semantic registry for player-metric explanations.

The registry is intentionally independent from BB-Rating weights: a metric can
be fully explainable without contributing to the composite score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MetricSemantic:
    key: str
    source_column: str
    label: str
    meaning: str
    unit: str = "number"
    direction: str = "contextual"


METRIC_SEMANTICS: dict[str, MetricSemantic] = {
    "RAPTOR": MetricSemantic("RAPTOR", "raptor_total", "RAPTOR", "stima dell'impatto complessivo", direction="higher_better"),
    "LEBRON": MetricSemantic("LEBRON", "lebron_total", "LEBRON", "stima dell'impatto complessivo", direction="higher_better"),
    "VORP": MetricSemantic("VORP", "vorp", "VORP", "valore prodotto rispetto a un giocatore di sostituzione", direction="higher_better"),
    "PER": MetricSemantic("PER", "per", "PER", "produttività statistica per minuto", direction="higher_better"),
    "TS%": MetricSemantic("TS%", "ts_pct", "TS%", "efficienza realizzativa considerando il valore dei diversi tipi di tiro", unit="percent", direction="higher_better"),
    "SCORING_EFFICIENCY": MetricSemantic("SCORING_EFFICIENCY", "scoring_efficiency", "Scoring Efficiency", "efficienza nella produzione di punti", direction="higher_better"),
    "AST%": MetricSemantic("AST%", "ast_pct", "AST%", "coinvolgimento nella creazione di assist", unit="percent", direction="higher_better"),
    "USG%": MetricSemantic("USG%", "usg_pct", "USG%", "coinvolgimento offensivo e quota di possessi conclusi dal giocatore", unit="percent"),
    "TOV%": MetricSemantic("TOV%", "tov_pct", "TOV%", "frequenza con cui il giocatore perde possessi", unit="percent", direction="lower_better"),
    "RAPTOR_DEF": MetricSemantic("RAPTOR_DEF", "raptor_def", "RAPTOR Def", "stima dell'impatto difensivo", direction="higher_better"),
    "DBPM": MetricSemantic("DBPM", "dbpm", "DBPM", "impatto difensivo stimato dal box score", direction="higher_better"),
    "NET_RTG_DIFF": MetricSemantic("NET_RTG_DIFF", "net_rtg_diff", "Net Rating Diff", "differenziale di rendimento della squadra con il giocatore in campo rispetto al riferimento on/off", direction="higher_better"),
    "STL%": MetricSemantic("STL%", "stl_pct", "STL%", "frequenza di recuperi palla", unit="percent", direction="higher_better"),
    "BLK%": MetricSemantic("BLK%", "blk_pct", "BLK%", "frequenza di stoppate", unit="percent", direction="higher_better"),
    "REB%": MetricSemantic("REB%", "reb_pct", "REB%", "quota di rimbalzi disponibili catturati", unit="percent", direction="higher_better"),
    "HUSTLE": MetricSemantic("HUSTLE", "hustle_index", "Hustle", "indice sintetico delle attività di energia e hustle", direction="higher_better"),
    "FOUL_DRAWING": MetricSemantic("FOUL_DRAWING", "foul_drawing_rate", "Foul Drawing", "frequenza con cui il giocatore genera falli subiti", direction="higher_better"),

    "POINTS": MetricSemantic("POINTS", "points", "Points", "produzione realizzativa per partita"),
    "ASSISTS": MetricSemantic("ASSISTS", "assists", "Assists", "creazione di gioco tramite assist per partita"),
    "REBOUNDS": MetricSemantic("REBOUNDS", "rebounds", "Rebounds", "produzione a rimbalzo per partita"),
    "OFFENSIVE_REBOUNDS": MetricSemantic("OFFENSIVE_REBOUNDS", "offensive_rebounds", "Offensive Rebounds", "produzione di rimbalzi offensivi per partita"),
    "DEFENSIVE_REBOUNDS": MetricSemantic("DEFENSIVE_REBOUNDS", "defensive_rebounds", "Defensive Rebounds", "produzione di rimbalzi difensivi per partita"),
    "STEALS": MetricSemantic("STEALS", "steals", "Steals", "recuperi palla per partita"),
    "BLOCKS": MetricSemantic("BLOCKS", "blocks", "Blocks", "stoppate per partita"),
    "TURNOVERS": MetricSemantic("TURNOVERS", "turnovers", "Turnovers", "palle perse per partita", direction="lower_better"),
    "FG%": MetricSemantic("FG%", "fg_pct", "FG%", "percentuale dal campo", unit="percent", direction="higher_better"),
    "3P%": MetricSemantic("3P%", "three_point_pct", "3P%", "percentuale da tre punti", unit="percent", direction="higher_better"),
    "FT%": MetricSemantic("FT%", "ft_pct", "FT%", "percentuale ai tiri liberi", unit="percent", direction="higher_better"),
    "eFG%": MetricSemantic("eFG%", "efg_pct", "eFG%", "efficienza al tiro che valorizza il tiro da tre", unit="percent", direction="higher_better"),
    "2P%": MetricSemantic("2P%", "two_point_pct", "2P%", "percentuale al tiro da due punti", unit="percent", direction="higher_better"),
    "MINUTES": MetricSemantic("MINUTES", "minutes_per_game", "Minutes", "volume di impiego medio per partita"),
    "GAMES": MetricSemantic("GAMES", "games_played", "Games Played", "numero di partite disputate"),
    "STARTER%": MetricSemantic("STARTER%", "starter_pct", "Starter%", "quota di partite iniziate da titolare", unit="percent"),
    "PLUS_MINUS": MetricSemantic("PLUS_MINUS", "plus_minus", "+/-", "differenziale di punti durante le presenze in campo"),
    "PTS_PER_36": MetricSemantic("PTS_PER_36", "pts_per_36", "PTS/36", "produzione di punti normalizzata per 36 minuti"),
    "AST_PER_36": MetricSemantic("AST_PER_36", "ast_per_36", "AST/36", "produzione di assist normalizzata per 36 minuti"),
    "PTS_PER_40": MetricSemantic("PTS_PER_40", "pts_per_40", "PTS/40", "produzione di punti normalizzata per 40 minuti"),
    "AST_PER_40": MetricSemantic("AST_PER_40", "ast_per_40", "AST/40", "produzione di assist normalizzata per 40 minuti"),
    "REB_PER_36": MetricSemantic("REB_PER_36", "reb_per_36", "REB/36", "produzione a rimbalzo normalizzata per 36 minuti"),
    "VOLUME_3PA": MetricSemantic("VOLUME_3PA", "three_point_attempts", "3PA", "volume di tentativi da tre"),
    "VOLUME_FTA": MetricSemantic("VOLUME_FTA", "free_throw_attempts", "FTA", "volume di tiri liberi tentati"),
    "THREE_PAR": MetricSemantic("THREE_PAR", "three_par", "3P Rate", "quota di tiri dal campo presi da tre", unit="percent"),
    "TRUE_USG": MetricSemantic("TRUE_USG", "true_usg_pct", "True USG%", "coinvolgimento offensivo normalizzato", unit="percent"),
    "CLUTCH_TS%": MetricSemantic("CLUTCH_TS%", "clutch_ts_pct", "Clutch TS%", "efficienza al tiro nelle situazioni clutch", unit="percent", direction="higher_better"),
    "CLUTCH_NET_RTG": MetricSemantic("CLUTCH_NET_RTG", "clutch_net_rtg", "Clutch Net Rating", "rendimento della squadra nelle situazioni clutch con il giocatore", direction="contextual"),
    "ORTG_DIFF": MetricSemantic("ORTG_DIFF", "ortg_diff", "ORtg Diff", "differenziale offensivo on/off", direction="contextual"),
    "NET_RTG": MetricSemantic("NET_RTG", "net_rtg", "Net Rating", "rendimento netto della squadra durante le presenze del giocatore", direction="contextual"),
    "DURABILITY": MetricSemantic("DURABILITY", "games_played", "Availability", "volume di partite disputate nella stagione", direction="contextual"),
    "FOULS": MetricSemantic("FOULS", "fouls", "Fouls", "falli personali commessi per partita", direction="contextual"),
    "3PA": MetricSemantic("3PA", "three_point_attempts", "3PA", "volume di tentativi da tre per partita", direction="contextual"),
    "3PM": MetricSemantic("3PM", "three_point_made", "3PM", "volume di triple segnate per partita", direction="contextual"),
    "FTA": MetricSemantic("FTA", "free_throw_attempts", "FTA", "volume di tiri liberi tentati per partita", direction="contextual"),
    "BPM": MetricSemantic("BPM", "bpm", "BPM", "impatto stimato attraverso il box score", direction="higher_better"),
    "WS": MetricSemantic("WS", "ws", "Win Shares", "contributo stimato alle vittorie", direction="higher_better"),
    "WS/48": MetricSemantic("WS/48", "ws_per_48", "WS/48", "Win Shares normalizzate per 48 minuti", direction="higher_better"),
    "OBPM": MetricSemantic("OBPM", "obpm", "OBPM", "impatto offensivo stimato dal box score", direction="higher_better"),
    "SPM": MetricSemantic("SPM", "spm", "SPM", "impatto statistico complessivo stimato", direction="higher_better"),
    "RAPTOR_OFF": MetricSemantic("RAPTOR_OFF", "raptor_off", "RAPTOR Off", "impatto offensivo stimato", direction="higher_better"),
    "LEBRON_OFF": MetricSemantic("LEBRON_OFF", "lebron_off", "LEBRON Off", "impatto offensivo stimato", direction="higher_better"),
    "LEBRON_DEF": MetricSemantic("LEBRON_DEF", "lebron_def", "LEBRON Def", "impatto difensivo stimato", direction="higher_better"),
    "GMSC": MetricSemantic("GMSC", "gm_sc", "GmSc", "produzione sintetica per partita", direction="higher_better"),
    "FIC": MetricSemantic("FIC", "fic", "FIC", "impatto statistico sintetico", direction="higher_better"),
    "OWS": MetricSemantic("OWS", "ows", "OWS", "contributo offensivo stimato alle vittorie", direction="higher_better"),
    "DWS": MetricSemantic("DWS", "dws", "DWS", "contributo difensivo stimato alle vittorie", direction="higher_better"),
    "PPSA": MetricSemantic("PPSA", "ppsa", "PPSA", "punti prodotti per tentativo di tiro", direction="higher_better"),
    "RF/G": MetricSemantic("RF/G", "rf_per_game", "RF/G", "falli subiti generati per partita", direction="contextual"),
    "CLUTCH_PTS": MetricSemantic("CLUTCH_PTS", "clutch_pts", "Clutch Pts", "produzione di punti nelle situazioni clutch", direction="contextual"),
    "CLUTCH_eFG%": MetricSemantic("CLUTCH_eFG%", "clutch_efg_pct", "Clutch eFG%", "efficienza al tiro nelle situazioni clutch", unit="percent", direction="higher_better"),
    "CLUTCH_A/T": MetricSemantic("CLUTCH_A/T", "clutch_ast_to_tov", "Clutch A/T", "rapporto assist/perdite nelle situazioni clutch", direction="higher_better"),
    "REB/40": MetricSemantic("REB/40", "tr_per_40", "REB/40", "produzione a rimbalzo normalizzata per 40 minuti", direction="contextual"),
    "STL/40": MetricSemantic("STL/40", "stl_per_40", "STL/40", "recuperi normalizzati per 40 minuti", direction="contextual"),
    "BLK/40": MetricSemantic("BLK/40", "blk_per_40", "BLK/40", "stoppate normalizzate per 40 minuti", direction="contextual"),
    "ORTG": MetricSemantic("ORTG", "ortg", "ORtg", "rendimento offensivo per 100 possessi", direction="contextual"),
    "DRTG": MetricSemantic("DRTG", "drtg", "DRtg", "rendimento difensivo con riferimento per 100 possessi", direction="contextual"),
    "ON_NET_RTG": MetricSemantic("ON_NET_RTG", "on_net_rtg", "NetRtg On", "net rating della squadra quando il giocatore è in campo", direction="contextual"),
}


def format_metric_value(semantic: MetricSemantic, value: float) -> str:
    number = float(value)
    if semantic.unit == "percent":
        percent = number * 100.0 if abs(number) <= 1.0 else number
        return f"{percent:.1f}%"
    return f"{number:.2f}"


def metric_interpretation(
    semantic: MetricSemantic,
    value: float,
    percentile: Optional[float],
) -> Optional[str]:
    if percentile is None:
        return None

    pct = round(float(percentile) * 100)
    value_text = format_metric_value(semantic, value)
    if pct >= 90:
        relative = "molto alto"
    elif pct >= 75:
        relative = "alto"
    elif pct >= 60:
        relative = "sopra la media"
    elif pct >= 40:
        relative = "nella fascia media"
    elif pct >= 25:
        relative = "sotto la media"
    elif pct >= 10:
        relative = "basso"
    else:
        relative = "molto basso"

    if semantic.key == "USG%":
        return (
            f"USG% {value_text}: {semantic.meaning}; si colloca al {pct}° percentile "
            f"del gruppo di confronto, quindi il coinvolgimento è {relative}. "
            "Questo descrive il ruolo offensivo e non determina da solo la qualità."
        )

    if semantic.key == "TOV%":
        return (
            f"TOV% {value_text}: {semantic.meaning}; il profilo risulta al "
            f"{pct}° percentile dopo l'inversione della direzione, quindi la "
            f"gestione dei possessi è {relative}."
        )

    if semantic.direction == "contextual":
        return (
            f"{semantic.label} {value_text}: {semantic.meaning}; si colloca "
            f"al {pct}° percentile del gruppo di confronto."
        )

    return (
        f"{semantic.label} {value_text}: {semantic.meaning}; si colloca al "
        f"{pct}° percentile, quindi il livello relativo è {relative}."
    )

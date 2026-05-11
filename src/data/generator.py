"""Synthetic data generator for football player performance system.

Generates 20 leagues, 200+ teams, 5000+ players with career histories
from 2015–2024 and saves them as CSV files to data/sample/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

LEAGUES: List[Tuple] = [
    # (id, name, country, tier, competitiveness_score)
    (1,  "Premier League",       "England",     1, 0.98),
    (2,  "La Liga",              "Spain",       1, 0.97),
    (3,  "Bundesliga",           "Germany",     1, 0.95),
    (4,  "Serie A",              "Italy",       1, 0.94),
    (5,  "Ligue 1",              "France",      1, 0.92),
    (6,  "Eredivisie",           "Netherlands", 2, 0.82),
    (7,  "Primeira Liga",        "Portugal",    2, 0.80),
    (8,  "Super Lig",            "Turkey",      2, 0.78),
    (9,  "Scottish Premiership", "Scotland",    2, 0.72),
    (10, "Belgian Pro League",   "Belgium",     2, 0.75),
    (11, "Championship",         "England",     3, 0.65),
    (12, "2. Bundesliga",        "Germany",     3, 0.67),
    (13, "Serie B",              "Italy",       3, 0.63),
    (14, "La Liga 2",            "Spain",       3, 0.64),
    (15, "Ligue 2",              "France",      3, 0.60),
    (16, "League One",           "England",     4, 0.50),
    (17, "3. Liga",              "Germany",     4, 0.48),
    (18, "Serie C",              "Italy",       4, 0.45),
    (19, "Tercera Division",     "Spain",       5, 0.35),
    (20, "National",             "France",      5, 0.38),
]

POSITIONS = ["GK", "CB", "FB", "CM", "AM", "W", "ST"]
POSITION_WEIGHTS = [0.07, 0.14, 0.14, 0.18, 0.12, 0.18, 0.17]

NATIONALITIES = [
    "English", "Spanish", "German", "Italian", "French", "Brazilian",
    "Argentine", "Portuguese", "Dutch", "Belgian", "Colombian", "Senegalese",
    "Ghanaian", "Nigerian", "Moroccan", "Japanese", "Korean", "Swedish",
    "Norwegian", "Danish", "Croatian", "Polish", "Czech", "Romanian",
    "Greek", "Turkish", "Serbian", "Austrian", "Swiss", "Uruguayan",
]

PLAYING_STYLES = ["possession", "counter", "high_press", "direct"]
FORMATIONS = ["4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "3-4-3", "5-3-2", "4-5-1", "4-1-4-1"]

FIRST_NAMES = [
    "James", "Luis", "Marco", "Antoine", "Neymar", "Kylian", "Mohamed",
    "Cristiano", "Lionel", "Kevin", "Sadio", "Robert", "Harry", "Erling",
    "Vinicius", "Pedri", "Jude", "Bukayo", "Phil", "Trent", "Mason",
    "Lautaro", "Dusan", "Ciro", "Paulo", "Federico", "Bernardo", "Bruno",
    "Ruben", "Joao", "Raphael", "Achraf", "Theo", "Kingsley", "Ferran",
    "Ansu", "Gavi", "Rodri", "Ilkay", "Joshua", "Leroy", "Serge",
    "Timo", "Kai", "Jamal", "Florian", "Thomas", "Leon", "Niklas",
    "Manuel", "David", "Alisson", "Thibaut", "Ederson", "Jan",
    "Virgil", "Ruben", "John", "Aymeric", "Marquinhos", "Presnel",
    "Kalidou", "Milan", "Stefan", "Nikola", "Luka", "Mateo", "Ivan",
    "Borna", "Marcelo", "Dani", "Jordi", "Andres", "Sergio", "Xavi",
    "Carlos", "Santi", "Fernando", "Cesc", "Juan", "Diego", "Ricardo",
    "Rafael", "Gabriel", "Willian", "Roberto", "Fabio", "Emre", "Can",
    "Granit", "Xherdan", "Remo", "Haris", "Breel", "Shaqiri", "Zuber",
    "Patrik", "Schick", "Adam", "Hlozek", "Ondrej", "Lingr", "Ladislav",
]

LAST_NAMES = [
    "Smith", "Fernandez", "Muller", "Rossi", "Dupont", "Santos", "Silva",
    "Gomez", "Martinez", "Lopez", "Rodriguez", "Garcia", "Hernandez",
    "Davies", "Jones", "Wilson", "Taylor", "Brown", "Johnson", "Williams",
    "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Moore", "Allen", "Lewis", "Hall", "Young", "Walker",
    "Scott", "King", "Green", "Baker", "Adams", "Clarke", "Mitchell",
    "Mane", "Salah", "Benzema", "Mbappe", "Griezmann", "Pogba",
    "Kante", "Varane", "Lloris", "Pavard", "Upamecano", "Camavinga",
    "Bellingham", "Sancho", "Rashford", "Wan-Bissaka", "Perisic",
    "Brozovic", "Modric", "Kovacic", "Kramaric", "Werner", "Havertz",
    "Gnabry", "Sane", "Kimmich", "Goretzka", "Rudiger", "Neuer",
    "Ramos", "Pique", "Busquets", "Alba", "Carvajal", "Kroos",
    "Casemiro", "Valverde", "Asensio", "Kane", "Son", "Firmino",
    "Robertson", "Alexander-Arnold", "Fabinho", "Henderson", "Van Dijk",
    "De Bruyne", "Sterling", "Foden", "Gundogan", "Laporte", "Dias",
    "Stones", "Ederson", "Trossard", "Saka", "Martinelli", "Odegaard",
    "Nunez", "Gakpo", "Diaz", "Mac Allister", "Szoboszlai", "Bajcetic",
]

# Position-specific stat distributions: (mean per-90, std)
POSITION_STATS: Dict[str, Dict] = {
    "GK": {
        "goals_mean": 0.01, "goals_std": 0.03,
        "assists_mean": 0.01, "assists_std": 0.03,
        "pass_acc_mean": 72.0, "pass_acc_std": 6.0,
        "dribbles_mean": 0.10, "dribbles_std": 0.08,
        "tackles_mean": 0.20, "tackles_std": 0.15,
        "interc_mean": 0.10, "interc_std": 0.10,
        "aerial_mean": 1.50, "aerial_std": 0.50,
        "xG_mean": 0.01, "xG_std": 0.01,
        "xA_mean": 0.01, "xA_std": 0.01,
        "prog_passes_mean": 1.0, "prog_passes_std": 0.50,
        "key_passes_mean": 0.10, "key_passes_std": 0.08,
        "base_rating": 6.5, "rating_std": 0.45,
    },
    "CB": {
        "goals_mean": 0.05, "goals_std": 0.06,
        "assists_mean": 0.03, "assists_std": 0.04,
        "pass_acc_mean": 82.0, "pass_acc_std": 5.0,
        "dribbles_mean": 0.50, "dribbles_std": 0.25,
        "tackles_mean": 2.50, "tackles_std": 0.70,
        "interc_mean": 1.80, "interc_std": 0.55,
        "aerial_mean": 3.50, "aerial_std": 0.90,
        "xG_mean": 0.05, "xG_std": 0.04,
        "xA_mean": 0.03, "xA_std": 0.03,
        "prog_passes_mean": 3.0, "prog_passes_std": 1.0,
        "key_passes_mean": 0.30, "key_passes_std": 0.20,
        "base_rating": 6.6, "rating_std": 0.48,
    },
    "FB": {
        "goals_mean": 0.08, "goals_std": 0.08,
        "assists_mean": 0.15, "assists_std": 0.10,
        "pass_acc_mean": 80.0, "pass_acc_std": 5.0,
        "dribbles_mean": 1.20, "dribbles_std": 0.45,
        "tackles_mean": 1.80, "tackles_std": 0.55,
        "interc_mean": 1.20, "interc_std": 0.40,
        "aerial_mean": 1.50, "aerial_std": 0.50,
        "xG_mean": 0.08, "xG_std": 0.06,
        "xA_mean": 0.15, "xA_std": 0.09,
        "prog_passes_mean": 4.0, "prog_passes_std": 1.40,
        "key_passes_mean": 0.80, "key_passes_std": 0.35,
        "base_rating": 6.7, "rating_std": 0.48,
    },
    "CM": {
        "goals_mean": 0.12, "goals_std": 0.10,
        "assists_mean": 0.18, "assists_std": 0.10,
        "pass_acc_mean": 85.0, "pass_acc_std": 4.0,
        "dribbles_mean": 1.50, "dribbles_std": 0.45,
        "tackles_mean": 1.50, "tackles_std": 0.48,
        "interc_mean": 1.20, "interc_std": 0.38,
        "aerial_mean": 1.20, "aerial_std": 0.40,
        "xG_mean": 0.12, "xG_std": 0.08,
        "xA_mean": 0.18, "xA_std": 0.10,
        "prog_passes_mean": 6.0, "prog_passes_std": 1.80,
        "key_passes_mean": 1.50, "key_passes_std": 0.55,
        "base_rating": 6.8, "rating_std": 0.55,
    },
    "AM": {
        "goals_mean": 0.25, "goals_std": 0.15,
        "assists_mean": 0.28, "assists_std": 0.15,
        "pass_acc_mean": 82.0, "pass_acc_std": 5.0,
        "dribbles_mean": 2.50, "dribbles_std": 0.70,
        "tackles_mean": 0.80, "tackles_std": 0.35,
        "interc_mean": 0.70, "interc_std": 0.28,
        "aerial_mean": 0.80, "aerial_std": 0.30,
        "xG_mean": 0.25, "xG_std": 0.13,
        "xA_mean": 0.28, "xA_std": 0.15,
        "prog_passes_mean": 5.0, "prog_passes_std": 1.60,
        "key_passes_mean": 2.50, "key_passes_std": 0.90,
        "base_rating": 7.0, "rating_std": 0.62,
    },
    "W": {
        "goals_mean": 0.30, "goals_std": 0.18,
        "assists_mean": 0.25, "assists_std": 0.13,
        "pass_acc_mean": 78.0, "pass_acc_std": 6.0,
        "dribbles_mean": 3.00, "dribbles_std": 0.90,
        "tackles_mean": 0.70, "tackles_std": 0.28,
        "interc_mean": 0.50, "interc_std": 0.25,
        "aerial_mean": 0.70, "aerial_std": 0.28,
        "xG_mean": 0.30, "xG_std": 0.16,
        "xA_mean": 0.25, "xA_std": 0.13,
        "prog_passes_mean": 3.5, "prog_passes_std": 1.30,
        "key_passes_mean": 2.00, "key_passes_std": 0.75,
        "base_rating": 7.0, "rating_std": 0.65,
    },
    "ST": {
        "goals_mean": 0.45, "goals_std": 0.22,
        "assists_mean": 0.15, "assists_std": 0.08,
        "pass_acc_mean": 73.0, "pass_acc_std": 6.0,
        "dribbles_mean": 1.50, "dribbles_std": 0.55,
        "tackles_mean": 0.40, "tackles_std": 0.25,
        "interc_mean": 0.30, "interc_std": 0.18,
        "aerial_mean": 2.50, "aerial_std": 0.90,
        "xG_mean": 0.45, "xG_std": 0.22,
        "xA_mean": 0.15, "xA_std": 0.08,
        "prog_passes_mean": 1.5, "prog_passes_std": 0.65,
        "key_passes_mean": 0.80, "key_passes_std": 0.35,
        "base_rating": 7.0, "rating_std": 0.72,
    },
}

PEAK_AGES: Dict[str, int] = {
    "GK": 31, "CB": 29, "FB": 26, "CM": 28, "AM": 27, "W": 25, "ST": 27,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_factor(age: int, position: str) -> float:
    """Return a 0–1 performance multiplier based on age vs positional peak."""
    peak = PEAK_AGES.get(position, 27)
    diff = abs(age - peak)
    if diff <= 2:
        return 1.00
    elif diff <= 4:
        return 0.95
    elif diff <= 6:
        return 0.88
    elif diff <= 8:
        return 0.78
    else:
        return 0.65


def _player_name(rng: np.random.Generator) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first} {last}"


def _gen_stat(rng: np.random.Generator, mean: float, std: float, lo: float = 0.0) -> float:
    return float(max(lo, rng.normal(mean, std)))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_data(output_dir: str = "data/sample", seed: int = 42, n_players: int = 5200) -> None:
    """Generate synthetic football data and write CSV files.

    Args:
        output_dir: Directory to write CSV files into.
        seed: NumPy random seed for reproducibility.
        n_players: Number of players to generate.
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ leagues
    leagues_df = pd.DataFrame(
        LEAGUES, columns=["id", "name", "country", "tier", "competitiveness_score"]
    )
    leagues_df.to_csv(out / "leagues.csv", index=False)

    # ------------------------------------------------------------------ teams
    TEAMS_PER_LEAGUE: Dict[int, int] = {
        1: 20, 2: 20, 3: 18, 4: 20, 5: 18,
        6: 18, 7: 18, 8: 18, 9: 12, 10: 16,
        11: 24, 12: 18, 13: 20, 14: 22, 15: 20,
        16: 24, 17: 18, 18: 20, 19: 18, 20: 18,
    }

    team_rows: List[Dict] = []
    team_id = 1
    league_teams: Dict[int, List[int]] = {}

    for lid, lname, country, tier, comp in LEAGUES:
        n_teams = TEAMS_PER_LEAGUE.get(lid, 18)
        league_teams[lid] = []
        for i in range(n_teams):
            style = rng.choice(PLAYING_STYLES)
            formation = rng.choice(FORMATIONS)
            if style == "possession":
                avg_poss = float(rng.uniform(55, 70))
            elif style == "counter":
                avg_poss = float(rng.uniform(35, 50))
            elif style == "high_press":
                avg_poss = float(rng.uniform(50, 65))
            else:  # direct
                avg_poss = float(rng.uniform(40, 55))

            pressing = float(rng.uniform(6, 10) if style == "high_press" else rng.uniform(3, 8))
            def_line = float(rng.uniform(5, 9) if style in ("high_press", "possession") else rng.uniform(3, 7))
            pass_tempo = float(rng.uniform(6, 10) if style in ("possession", "high_press") else rng.uniform(3, 7))

            # Generate an ASCII-safe team name
            suffix_letter = chr(65 + (i % 26))
            city_prefixes = [
                "North", "South", "East", "West", "Central", "Royal", "United",
                "Athletic", "Sporting", "City", "Town", "Villa", "Rovers",
            ]
            city = rng.choice(city_prefixes)
            tname = f"{country[:3]} {city} {suffix_letter}{team_id % 100:02d}"

            team_rows.append({
                "id": team_id,
                "name": tname,
                "league_id": lid,
                "playing_style": style,
                "formation": formation,
                "avg_possession": round(avg_poss, 1),
                "pressing_intensity": round(pressing, 2),
                "defensive_line": round(def_line, 2),
                "passing_tempo": round(pass_tempo, 2),
                "league_tier": tier,
            })
            league_teams[lid].append(team_id)
            team_id += 1

    teams_df = pd.DataFrame(team_rows)
    teams_df.to_csv(out / "teams.csv", index=False)
    team_tier_map: Dict[int, int] = dict(zip(teams_df["id"], teams_df["league_tier"]))
    team_league_map: Dict[int, int] = dict(zip(teams_df["id"], teams_df["league_id"]))

    # ------------------------------------------------------------------ players + stats
    SEASONS = list(range(2015, 2025))
    CURRENT_YEAR = 2024

    player_rows: List[Dict] = []
    stat_rows: List[Dict] = []
    relation_rows: List[Dict] = []

    tier_probs = [0.15, 0.20, 0.25, 0.20, 0.20]

    for pid in range(1, n_players + 1):
        # Demographics
        birth_year = int(rng.integers(1986, 2009))
        age_in_2024 = CURRENT_YEAR - birth_year
        position = rng.choice(POSITIONS, p=POSITION_WEIGHTS)
        nationality = rng.choice(NATIONALITIES)
        foot = rng.choice(["right", "left", "both"], p=[0.70, 0.25, 0.05])
        height_mean = 186.0 if position in ("GK", "CB") else 181.0
        height = float(rng.normal(height_mean, 5.5))
        weight = float(height * 0.43 + rng.normal(0, 3))

        # Current club (tier-weighted)
        p_tier = int(rng.choice([1, 2, 3, 4, 5], p=tier_probs))
        tier_leagues = [lg for lg in LEAGUES if lg[3] == p_tier]
        chosen_league = tier_leagues[int(rng.integers(0, len(tier_leagues)))]
        cur_lid = chosen_league[0]
        cur_tid = int(rng.choice(league_teams[cur_lid]))

        player_rows.append({
            "id": pid,
            "name": _player_name(rng),
            "age": age_in_2024,
            "position": position,
            "nationality": nationality,
            "foot": foot,
            "height": round(height, 1),
            "weight": round(weight, 1),
            "current_team_id": cur_tid,
            "current_league_id": cur_lid,
        })

        # Career history
        career_start_age = int(rng.integers(17, 22))
        career_end_age = int(rng.integers(33, 39))
        career_start_year = birth_year + career_start_age
        career_end_year = min(birth_year + career_end_age, CURRENT_YEAR)

        active_seasons = [s for s in SEASONS if career_start_year <= s <= career_end_year]
        if not active_seasons:
            continue

        s_tid = cur_tid
        s_lid = cur_lid
        stats_cfg = POSITION_STATS[position]

        for season in active_seasons:
            age_s = season - birth_year
            if age_s < 14:
                continue

            # Possible transfer (12 % per season)
            if season > active_seasons[0] and rng.random() < 0.12:
                new_tier = int(rng.choice([1, 2, 3, 4, 5], p=tier_probs))
                new_tl = [lg for lg in LEAGUES if lg[3] == new_tier]
                new_league = new_tl[int(rng.integers(0, len(new_tl)))]
                s_lid = new_league[0]
                s_tid = int(rng.choice(league_teams[s_lid]))

            af = _age_factor(age_s, position)
            t_tier = team_tier_map.get(s_tid, 3)
            quality_boost = (6 - t_tier) * 0.05  # tier-1 → +0.25

            # Role & minutes
            role_r = float(rng.random())
            if role_r < 0.60:
                role = "starter"
                matches = int(rng.integers(25, 38))
                mins_pct = float(rng.uniform(0.75, 0.95))
            elif role_r < 0.85:
                role = "rotation"
                matches = int(rng.integers(15, 28))
                mins_pct = float(rng.uniform(0.40, 0.70))
            else:
                role = "bench"
                matches = int(rng.integers(5, 18))
                mins_pct = float(rng.uniform(0.15, 0.35))

            total_min = matches * 90 * mins_pct
            m90 = max(total_min / 90, 0.01)

            goals = _gen_stat(rng, stats_cfg["goals_mean"] * af, stats_cfg["goals_std"]) * m90
            assists = _gen_stat(rng, stats_cfg["assists_mean"] * af, stats_cfg["assists_std"]) * m90
            pass_acc = _gen_stat(rng, stats_cfg["pass_acc_mean"], stats_cfg["pass_acc_std"], 40.0)
            pass_acc = min(pass_acc, 99.0)
            dribbles = _gen_stat(rng, stats_cfg["dribbles_mean"] * af, stats_cfg["dribbles_std"])
            tackles = _gen_stat(rng, stats_cfg["tackles_mean"], stats_cfg["tackles_std"])
            interc = _gen_stat(rng, stats_cfg["interc_mean"], stats_cfg["interc_std"])
            aerial = _gen_stat(rng, stats_cfg["aerial_mean"], stats_cfg["aerial_std"])
            xG = _gen_stat(rng, stats_cfg["xG_mean"] * af, stats_cfg["xG_std"]) * m90
            xA = _gen_stat(rng, stats_cfg["xA_mean"] * af, stats_cfg["xA_std"]) * m90
            prog = _gen_stat(rng, stats_cfg["prog_passes_mean"] * af, stats_cfg["prog_passes_std"])
            kp = _gen_stat(rng, stats_cfg["key_passes_mean"] * af, stats_cfg["key_passes_std"])

            base_r = stats_cfg["base_rating"] * af + quality_boost
            rating = float(np.clip(rng.normal(base_r, stats_cfg["rating_std"]), 4.0, 10.0))

            stat_rows.append({
                "player_id": pid,
                "season": season,
                "team_id": s_tid,
                "league_id": s_lid,
                "goals": round(goals, 2),
                "assists": round(assists, 2),
                "matches_played": matches,
                "minutes": round(total_min, 1),
                "pass_accuracy": round(pass_acc, 1),
                "dribbles": round(dribbles, 2),
                "tackles": round(tackles, 2),
                "interceptions": round(interc, 2),
                "aerial_duels_won": round(aerial, 2),
                "rating": round(rating, 2),
                "xG": round(xG, 3),
                "xA": round(xA, 3),
                "progressive_passes": round(prog, 2),
                "key_passes": round(kp, 2),
            })

            relation_rows.append({
                "team_id": s_tid,
                "player_id": pid,
                "season": season,
                "role": role,
                "jersey_number": int(rng.integers(1, 99)),
            })

    # ------------------------------------------------------------------ write CSVs
    players_df = pd.DataFrame(player_rows)
    stats_df = pd.DataFrame(stat_rows)
    rel_df = pd.DataFrame(relation_rows)

    players_df.to_csv(out / "players.csv", index=False)
    stats_df.to_csv(out / "player_stats.csv", index=False)
    rel_df.to_csv(out / "team_player_relations.csv", index=False)

    print(f"[generate] leagues   : {len(leagues_df)}")
    print(f"[generate] teams     : {len(teams_df)}")
    print(f"[generate] players   : {len(players_df)}")
    print(f"[generate] stat rows : {len(stats_df)}")
    print(f"[generate] relations : {len(rel_df)}")
    print(f"[generate] Data saved to {out.resolve()}/")


if __name__ == "__main__":
    generate_data()

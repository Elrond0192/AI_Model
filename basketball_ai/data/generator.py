"""Basketball synthetic data generator."""
import logging
import os, random, numpy as np, pandas as pd
from typing import List

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# Pure and hybrid/mixed positions
POSITIONS = ["PG","SG","SF","PF","C","PG/SG","SG/SF","SF/PF","PF/C","SG/PF"]
POS_WEIGHTS = [0.12,0.14,0.16,0.18,0.20,0.07,0.06,0.04,0.02,0.01]
STYLES = ["pace_and_space","pick_and_roll","isolation","defensive","motion_offense","post_up"]
NATIONALITIES = ["American","Spanish","French","German","Italian","Australian","Turkish","Greek","Nigerian","Canadian","Serbian","Slovenian","Argentinian","Brazilian","Croatian"]

# Fraction of player-seasons that also generate a Playoff (PO) statistics row.
_PO_GENERATION_PROBABILITY = 0.40

LEAGUES = [
    (1,"NBA","USA",1,1.0,100.0,113.0),
    (2,"EuroLeague","Europe",2,0.85,93.0,108.0),
    (3,"BCL","Europe",2,0.78,91.0,106.0),
    (4,"ACB","Spain",3,0.72,89.0,105.0),
    (5,"Bundesliga","Germany",3,0.68,88.0,104.0),
    (6,"Lega Basket","Italy",3,0.70,87.0,104.0),
    (7,"BSL","Turkey",3,0.71,88.0,105.0),
    (8,"LNB Pro A","France",3,0.65,87.0,103.0),
    (9,"NBL","Australia",4,0.60,90.0,102.0),
    (10,"VTB League","Russia",4,0.58,86.0,101.0),
    (11,"Adriatic","Balkans",4,0.55,85.0,100.0),
    (12,"Liga ACB B","Spain",5,0.45,84.0,99.0),
    (13,"Pro B","Germany",5,0.43,83.0,98.0),
    (14,"Serie A2","Italy",5,0.44,82.0,98.0),
    (15,"BSL B","Turkey",5,0.42,83.0,97.0),
    (16,"Pro B France","France",5,0.40,82.0,96.0),
    (17,"NBL1","Australia",5,0.38,81.0,95.0),
    (18,"FIBA EuroCup","Europe",4,0.62,88.0,102.0),
    (19,"CBA","China",4,0.55,87.0,101.0),
    (20,"Liga Nacional","Argentina",5,0.40,83.0,98.0),
]

FIRST_NAMES = ["James","Kevin","Stephen","LeBron","Giannis","Luka","Joel","Nikola","Jayson","Damian","Trae","Zion","Anthony","Karl","Devin","Paul","Chris","Kawhi","Jimmy","Bam","Pascal","Rudy","Draymond","Klay","Jordan","Marcus","Tyler","Miles","Brandon","Cade","Evan","OG","Scottie","Jalen","De'Aaron","Shai","Donovan","Ja","Zach","Julius","Khris","Tobias","Al","Domantas","Jonas","Kristaps","Brook","Serge","Myles","Aaron"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Davis","Miller","Wilson","Moore","Taylor","Anderson","Thomas","Jackson","White","Harris","Martin","Thompson","Garcia","Martinez","Robinson","Clark","Rodriguez","Lewis","Lee","Walker","Hall","Allen","Young","Hernandez","King","Wright","Lopez","Hill","Scott","Green","Adams","Baker","Gonzalez","Nelson","Carter","Mitchell","Perez","Roberts","Turner","Phillips","Campbell","Parker","Evans","Edwards"]

def _rng(mu, sigma, lo=None, hi=None):
    v = np.random.normal(mu, sigma)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return float(v)

def _primary_pos(pos: str) -> str:
    """Return the primary (first) position for a potentially mixed role."""
    return pos.split("/")[0]

def age_factor(age, pos):
    peaks = {"PG":26,"SG":25,"SF":26,"PF":27,"C":28,
             "PG/SG":25,"SG/SF":25,"SF/PF":26,"PF/C":27,"SG/PF":26}
    peak = peaks.get(pos, 26)
    diff = age - peak
    if diff <= 0:
        return max(0.3, 1.0 - abs(diff)*0.04)
    else:
        return max(0.3, 1.0 - diff*0.035)

def compute_rating(per, bpm, win_shares, age, pos):
    r = np.clip((per/35)*10*0.4 + (bpm+10)/25*10*0.4 + win_shares/15*10*0.2, 0, 10)
    r *= age_factor(age, pos)
    return float(np.clip(r, 0.5, 10.0))

def generate_leagues():
    rows = []
    for (lid,name,country,tier,comp,pace,ortg) in LEAGUES:
        rows.append({"id":lid,"name":name,"country":country,"tier":tier,"competitiveness_score":comp,"avg_pace":pace,"avg_offensive_rating":ortg})
    return pd.DataFrame(rows)

def generate_teams(leagues_df):
    rows = []
    tid = 1
    per_league = {1:30,2:18,3:16,4:14,5:12}
    for _,lg in leagues_df.iterrows():
        n = per_league.get(int(lg["tier"]),12)
        for i in range(n):
            style = random.choice(STYLES)
            pace = _rng(lg["avg_pace"],3,80,115)
            ortg = _rng(lg["avg_offensive_rating"],3,90,125)
            drtg = _rng(lg["avg_offensive_rating"]-5,3,88,120)
            rows.append({
                "id":tid,"name":f"Team_{tid}","league_id":int(lg["id"]),
                "playing_style":style,"formation":random.choice(["1-2-2","2-3","small_ball","big_lineup"]),
                "pace":round(pace,1),"offensive_rating":round(ortg,1),"defensive_rating":round(drtg,1),
                "three_point_attempt_rate":round(_rng(0.38,0.08,0.1,0.55),3),
                "assists_per_game":round(_rng(24,4,14,35),1),
                "star_player_usage":round(_rng(0.28,0.06,0.10,0.45),3),
                "league_tier":int(lg["tier"])
            })
            tid+=1
    return pd.DataFrame(rows)

def generate_players(n=5000):
    rows = []
    for pid in range(1,n+1):
        pos = np.random.choice(POSITIONS,p=POS_WEIGHTS)
        age = int(_rng(26,5,18,40))
        rows.append({
            "id":pid,
            "name":f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "age":age,"position":pos,
            "nationality":random.choice(NATIONALITIES),
            "height_cm":int(_rng(195,8,175,225)),
            "weight_kg":int(_rng(95,12,70,140)),
            "dominant_hand":random.choice(["right","right","right","left"]),
            "current_team_id":None,"current_league_id":None,
            "draft_year":random.choice([None]+list(range(2010,2024))),
            "draft_pick":random.choice([None]+list(range(1,61)))
        })
    return pd.DataFrame(rows)

def generate_player_stats(players_df, teams_df):
    pos_params = {
        "PG":  {"pts":(14,6),"ast":(5,3),"reb":(4,2),"stl":(1.2,0.5),"blk":(0.3,0.2),"per":(15,4),"usg":(22,6),"bpm":(0,3)},
        "SG":  {"pts":(16,7),"ast":(3,2),"reb":(4,2),"stl":(1.0,0.5),"blk":(0.4,0.2),"per":(14,4),"usg":(22,6),"bpm":(0,3)},
        "SF":  {"pts":(15,6),"ast":(3,2),"reb":(6,2),"stl":(1.1,0.4),"blk":(0.7,0.3),"per":(14,4),"usg":(21,5),"bpm":(0,3)},
        "PF":  {"pts":(13,5),"ast":(2,1.5),"reb":(7,2),"stl":(0.9,0.4),"blk":(1.0,0.5),"per":(14,4),"usg":(20,5),"bpm":(-0.5,3)},
        "C":   {"pts":(11,5),"ast":(1.5,1),"reb":(8,2.5),"stl":(0.7,0.3),"blk":(1.5,0.7),"per":(15,4),"usg":(18,5),"bpm":(-0.5,3)},
        # Hybrid positions — interpolated between component params
        "PG/SG": {"pts":(15,6),"ast":(4,2.5),"reb":(4,2),"stl":(1.1,0.5),"blk":(0.35,0.2),"per":(14.5,4),"usg":(22,6),"bpm":(0,3)},
        "SG/SF": {"pts":(15.5,6),"ast":(3,2),"reb":(5,2),"stl":(1.05,0.45),"blk":(0.55,0.25),"per":(14,4),"usg":(21.5,5.5),"bpm":(0,3)},
        "SF/PF": {"pts":(14,5.5),"ast":(2.5,1.8),"reb":(6.5,2),"stl":(1.0,0.4),"blk":(0.85,0.4),"per":(14,4),"usg":(20.5,5),"bpm":(-0.25,3)},
        "PF/C":  {"pts":(12,5),"ast":(1.75,1.2),"reb":(7.5,2.3),"stl":(0.8,0.35),"blk":(1.25,0.6),"per":(14.5,4),"usg":(19,5),"bpm":(-0.5,3)},
        "SG/PF": {"pts":(14,6),"ast":(2.5,2),"reb":(5.5,2.5),"stl":(0.95,0.4),"blk":(0.7,0.35),"per":(13.5,4),"usg":(21,5.5),"bpm":(-0.25,3)},
    }
    seasons = ["2019-20","2020-21","2021-22","2022-23","2023-24"]
    team_ids = teams_df["id"].tolist()
    team_league = dict(zip(teams_df["id"],teams_df["league_id"]))
    rows = []
    for _,pl in players_df.iterrows():
        pos = pl["position"]
        p = pos_params[pos]
        n_seasons = random.randint(1,len(seasons))
        player_seasons = random.sample(seasons,n_seasons)
        # Per-player playoff tendency: positive = elevates in PO, negative = declines
        po_tendency = float(np.random.normal(0.0, 0.12))
        for s in player_seasons:
            tid = random.choice(team_ids)
            lid = team_league[tid]
            age = pl["age"]
            pts = _rng(*p["pts"],0,50)
            ast = _rng(*p["ast"],0,15)
            reb = _rng(*p["reb"],0,20)
            stl = _rng(*p["stl"],0,4)
            blk = _rng(*p["blk"],0,5)
            per = _rng(*p["per"],0,40)
            usg = _rng(*p["usg"],5,45)
            bpm = _rng(*p["bpm"],-15,20)
            ws = _rng(per*0.3,2,0,25)
            rating = compute_rating(per,bpm,ws,age,pos)
            rows.append({
                "player_id":int(pl["id"]),"season":s,"team_id":int(tid),"league_id":int(lid),
                "games_played":int(_rng(55,18,1,82)),
                "minutes_per_game":round(_rng(24,7,5,40),1),
                "points":round(pts,1),"rebounds":round(reb,1),
                "offensive_rebounds":round(reb*0.3,1),"defensive_rebounds":round(reb*0.7,1),
                "assists":round(ast,1),"steals":round(stl,2),"blocks":round(blk,2),
                "turnovers":round(_rng(2,1,0,8),1),"personal_fouls":round(_rng(2.5,1,0,6),1),
                "fg_pct":round(_rng(0.46,0.06,0.2,0.75),3),
                "three_point_pct":round(_rng(0.35,0.08,0.0,0.6),3),
                "ft_pct":round(_rng(0.77,0.09,0.3,1.0),3),
                "plus_minus":round(_rng(0,5,-20,25),1),
                "per":round(per,2),"ts_pct":round(_rng(0.56,0.06,0.3,0.75),3),
                "usg_pct":round(usg,2),"bpm":round(bpm,2),
                "vorp":round(_rng(1,1.5,-3,10),2),"win_shares":round(ws,2),
                "ast_ratio":round(_rng(15,8,0,50),2),"reb_pct":round(_rng(10,4,1,30),2),
                "rating":round(rating,3),
                "competition":"RS",
                # Advanced metrics
                "spm":       round(_rng(bpm*0.9, 0.5), 2),
                "obpm":      round(_rng(bpm*0.6, 0.4), 2),
                "dbpm":      round(_rng(bpm*0.4, 0.4), 2),
                "gm_sc":     round(_rng(per*0.25, 1.0, 0, 50), 2),
                "fic":       round(_rng(per*0.8, 2.0, 0, 100), 2),
                "ows":       round(_rng(ws*0.55, 0.5, 0, 15), 2),
                "dws":       round(_rng(ws*0.45, 0.4, 0, 12), 2),
                "raptor_off":  round(_rng(bpm*0.7 + 1, 1.5, -15, 15), 2),
                "raptor_def":  round(_rng(bpm*0.3, 1.0, -12, 12), 2),
                "raptor_total":round(_rng(bpm*1.0, 1.5, -20, 20), 2),
                "lebron_off":  round(_rng(bpm*0.65 + 0.5, 1.2, -12, 12), 2),
                "lebron_def":  round(_rng(bpm*0.35, 0.9, -10, 10), 2),
                "lebron_total":round(_rng(bpm*0.95, 1.4, -18, 18), 2),
                "scoring_efficiency": round(_rng(1.05, 0.15, 0.5, 1.8), 3),
                "ppsa":       round(_rng(1.0, 0.15, 0.4, 1.8), 3),
                "two_point_pct": round(_rng(0.50, 0.06, 0.2, 0.75), 3),
                "tov_pct":    round(_rng(14, 4, 3, 35), 2),
                "ast_pct":    round(_rng(15, 8, 0, 50), 2),
                "stl_pct":    round(_rng(1.5, 0.5, 0, 5), 2),
                "blk_pct":    round(_rng(1.0, 0.7, 0, 8), 2),
                "orb_pct":    round(_rng(4, 3, 0, 25), 2),
                "drb_pct":    round(_rng(15, 5, 2, 40), 2),
                "three_par":  round(_rng(0.35, 0.12, 0, 0.7), 3),
                "true_usg_pct": round(usg * 1.05, 2),
                "foul_drawing_rate": round(_rng(0.25, 0.08, 0, 0.6), 3),
                "rf_per_game": round(_rng(3, 1.2, 0, 10), 2),
                "hustle_index": round(_rng(50, 15, 5, 100), 1),
                "pts_per_40":  round(pts / max(0.1, _rng(24,7,5,40)) * 40, 2),
                "ast_per_40":  round(ast / max(0.1, _rng(24,7,5,40)) * 40, 2),
                "tr_per_40":   round(reb / max(0.1, _rng(24,7,5,40)) * 40, 2),
                "stl_per_40":  round(stl / max(0.1, _rng(24,7,5,40)) * 40, 2),
                "blk_per_40":  round(blk / max(0.1, _rng(24,7,5,40)) * 40, 2),
                # On/Off
                "ortg": round(_rng(110, 5, 90, 130), 1),
                "drtg": round(_rng(108, 5, 88, 128), 1),
                "net_rtg": round(_rng(2, 4, -15, 20), 2),
                "on_net_rtg":   round(_rng(3, 5, -20, 25), 2),
                "off_net_rtg":  round(_rng(0, 4, -20, 20), 2),
                "net_rtg_diff": round(_rng(3, 3, -15, 15), 2),
                "ortg_on":  round(_rng(112, 5, 90, 130), 1),
                "ortg_off": round(_rng(109, 5, 88, 128), 1),
                "ortg_diff": round(_rng(3, 3, -15, 15), 2),
                # Clutch
                "clutch_games":     int(_rng(8, 4, 0, 25)),
                "clutch_pts":       round(_rng(pts * 0.9, 3, 0, 45), 2),
                "clutch_ts_pct":    round(_rng(0.55, 0.07, 0.2, 0.75), 3),
                "clutch_ast_to_tov":round(_rng(1.5, 0.8, 0, 6), 2),
                "clutch_net_rtg":   round(_rng(1.5, 5, -25, 25), 2),
                "clutch_efg_pct":   round(_rng(0.52, 0.07, 0.2, 0.75), 3),
                # Roles (default empty for generated data)
                "ruolo_offensivo": "", "ruolo_difensivo": "", "ruolo_combinato": "",
            })
            # ~40% of players also have a playoff (PO) row for this season
            if random.random() < _PO_GENERATION_PROBABILITY:
                _noise = lambda: float(np.random.normal(0, 0.04))
                po_mult = 1.0 + po_tendency + _noise()
                po_pts  = max(0.0, pts  * po_mult)
                po_ast  = max(0.0, ast  * (1.0 + po_tendency * 0.7 + _noise()))
                po_reb  = max(0.0, reb  * (1.0 + po_tendency * 0.5 + _noise()))
                po_stl  = max(0.0, stl  * (1.0 + po_tendency * 0.5 + _noise()))
                po_blk  = max(0.0, blk  * (1.0 + po_tendency * 0.5 + _noise()))
                po_per  = max(0.0, per  * po_mult)
                po_bpm  = bpm + po_tendency * 2.0 + float(np.random.normal(0, 0.4))
                po_usg  = max(5.0, usg  * (1.0 + po_tendency * 0.5 + _noise()))
                po_ws   = max(0.0, ws   * po_mult * 0.35)  # fewer games in PO
                po_rating = compute_rating(po_per, po_bpm, po_ws, age, pos)
                rows.append({
                    "player_id":int(pl["id"]),"season":s,"team_id":int(tid),"league_id":int(lid),
                    "games_played":int(max(1, _rng(12, 4, 1, 25))),
                    "minutes_per_game":round(_rng(26, 6, 5, 40), 1),
                    "points":round(po_pts,1),"rebounds":round(po_reb,1),
                    "offensive_rebounds":round(po_reb*0.3,1),"defensive_rebounds":round(po_reb*0.7,1),
                    "assists":round(po_ast,1),"steals":round(po_stl,2),"blocks":round(po_blk,2),
                    "turnovers":round(_rng(2,1,0,8),1),"personal_fouls":round(_rng(2.5,1,0,6),1),
                    "fg_pct":round(_rng(0.46,0.06,0.2,0.75),3),
                    "three_point_pct":round(_rng(0.35,0.08,0.0,0.6),3),
                    "ft_pct":round(_rng(0.77,0.09,0.3,1.0),3),
                    "plus_minus":round(_rng(0,5,-20,25),1),
                    "per":round(po_per,2),"ts_pct":round(_rng(0.56,0.06,0.3,0.75),3),
                    "usg_pct":round(po_usg,2),"bpm":round(po_bpm,2),
                    "vorp":round(_rng(1,1.5,-3,10),2),"win_shares":round(po_ws,2),
                    "ast_ratio":round(_rng(15,8,0,50),2),"reb_pct":round(_rng(10,4,1,30),2),
                    "rating":round(po_rating,3),
                    "competition":"PO",
                    # Advanced metrics (PO version – slightly noisy versions of RS)
                    "spm":        round(_rng(po_bpm*0.9, 0.5), 2),
                    "obpm":       round(_rng(po_bpm*0.6, 0.4), 2),
                    "dbpm":       round(_rng(po_bpm*0.4, 0.4), 2),
                    "gm_sc":      round(_rng(po_per*0.25, 1.0, 0, 50), 2),
                    "fic":        round(_rng(po_per*0.8, 2.0, 0, 100), 2),
                    "ows":        round(_rng(po_ws*0.55, 0.5, 0, 15), 2),
                    "dws":        round(_rng(po_ws*0.45, 0.4, 0, 12), 2),
                    "raptor_off":   round(_rng(po_bpm*0.7 + 1, 1.5, -15, 15), 2),
                    "raptor_def":   round(_rng(po_bpm*0.3, 1.0, -12, 12), 2),
                    "raptor_total": round(_rng(po_bpm*1.0, 1.5, -20, 20), 2),
                    "lebron_off":   round(_rng(po_bpm*0.65 + 0.5, 1.2, -12, 12), 2),
                    "lebron_def":   round(_rng(po_bpm*0.35, 0.9, -10, 10), 2),
                    "lebron_total": round(_rng(po_bpm*0.95, 1.4, -18, 18), 2),
                    "scoring_efficiency": round(_rng(1.05, 0.15, 0.5, 1.8), 3),
                    "ppsa":        round(_rng(1.0, 0.15, 0.4, 1.8), 3),
                    "two_point_pct": round(_rng(0.50, 0.06, 0.2, 0.75), 3),
                    "tov_pct":     round(_rng(14, 4, 3, 35), 2),
                    "ast_pct":     round(_rng(15, 8, 0, 50), 2),
                    "stl_pct":     round(_rng(1.5, 0.5, 0, 5), 2),
                    "blk_pct":     round(_rng(1.0, 0.7, 0, 8), 2),
                    "orb_pct":     round(_rng(4, 3, 0, 25), 2),
                    "drb_pct":     round(_rng(15, 5, 2, 40), 2),
                    "three_par":   round(_rng(0.35, 0.12, 0, 0.7), 3),
                    "true_usg_pct": round(po_usg * 1.05, 2),
                    "foul_drawing_rate": round(_rng(0.25, 0.08, 0, 0.6), 3),
                    "rf_per_game":  round(_rng(3, 1.2, 0, 10), 2),
                    "hustle_index": round(_rng(52, 15, 5, 100), 1),
                    "pts_per_40":   round(po_pts / max(0.1, _rng(26,6,5,40)) * 40, 2),
                    "ast_per_40":   round(po_ast / max(0.1, _rng(26,6,5,40)) * 40, 2),
                    "tr_per_40":    round(po_reb / max(0.1, _rng(26,6,5,40)) * 40, 2),
                    "stl_per_40":   round(po_stl / max(0.1, _rng(26,6,5,40)) * 40, 2),
                    "blk_per_40":   round(po_blk / max(0.1, _rng(26,6,5,40)) * 40, 2),
                    # On/Off
                    "ortg": round(_rng(111, 5, 90, 130), 1),
                    "drtg": round(_rng(107, 5, 88, 128), 1),
                    "net_rtg": round(_rng(4, 4, -15, 20), 2),
                    "on_net_rtg":    round(_rng(5, 5, -20, 25), 2),
                    "off_net_rtg":   round(_rng(1, 4, -20, 20), 2),
                    "net_rtg_diff":  round(_rng(4, 3, -15, 15), 2),
                    "ortg_on":  round(_rng(113, 5, 90, 130), 1),
                    "ortg_off": round(_rng(110, 5, 88, 128), 1),
                    "ortg_diff": round(_rng(3, 3, -15, 15), 2),
                    # Clutch (PO has fewer clutch games)
                    "clutch_games":      int(_rng(5, 3, 0, 15)),
                    "clutch_pts":        round(_rng(po_pts * 0.9, 3, 0, 45), 2),
                    "clutch_ts_pct":     round(_rng(0.56, 0.07, 0.2, 0.75), 3),
                    "clutch_ast_to_tov": round(_rng(1.6, 0.8, 0, 6), 2),
                    "clutch_net_rtg":    round(_rng(2.5, 5, -25, 25), 2),
                    "clutch_efg_pct":    round(_rng(0.53, 0.07, 0.2, 0.75), 3),
                    # Roles
                    "ruolo_offensivo": "", "ruolo_difensivo": "", "ruolo_combinato": "",
                })
    return pd.DataFrame(rows)

def generate_team_player_relations(players_df, teams_df, player_stats_df):
    rows = []
    seasons = ["2023-24"]
    team_ids = teams_df["id"].tolist()
    for tid in team_ids:
        n = random.randint(12,15)
        pids = player_stats_df[player_stats_df["team_id"]==tid]["player_id"].unique()
        if len(pids) < n:
            extra = players_df["id"].sample(n-len(pids)).tolist()
            pids = list(pids) + extra
        pids = list(pids)[:n]
        jerseys = random.sample(range(0,100),len(pids))
        for i,(pid,jn) in enumerate(zip(pids,jerseys)):
            role = "starter" if i<5 else ("rotation" if i<10 else "bench")
            rows.append({"team_id":tid,"player_id":int(pid),"season":"2023-24","role":role,"jersey_number":jn})
    return pd.DataFrame(rows)

def generate_and_save():
    out = "/home/runner/work/AI_Model/AI_Model/data/sample"
    os.makedirs(out,exist_ok=True)
    print("Generating leagues...")
    leagues = generate_leagues()
    leagues.to_csv(f"{out}/leagues.csv",index=False)
    print("Generating teams...")
    teams = generate_teams(leagues)
    teams.to_csv(f"{out}/teams.csv",index=False)
    print("Generating players...")
    players = generate_players(5000)
    players.to_csv(f"{out}/players.csv",index=False)
    print("Generating player stats...")
    stats = generate_player_stats(players,teams)
    stats.to_csv(f"{out}/player_stats.csv",index=False)
    print("Generating team-player relations...")
    rels = generate_team_player_relations(players,teams,stats)
    rels.to_csv(f"{out}/team_player_relations.csv",index=False)
    # update player current team/league
    latest = stats.sort_values("season").groupby("player_id").last().reset_index()
    pid_team = dict(zip(latest["player_id"],latest["team_id"]))
    pid_league = dict(zip(latest["player_id"],latest["league_id"]))
    players["current_team_id"] = players["id"].map(pid_team)
    players["current_league_id"] = players["id"].map(pid_league)
    players.to_csv(f"{out}/players.csv",index=False)
    print(f"Done. leagues={len(leagues)}, teams={len(teams)}, players={len(players)}, stats={len(stats)}, rels={len(rels)}")

def generate_data(output_dir: str = "data/sample", seed: int = 42) -> None:
    """Public entry point called by CLI and tests."""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating leagues...")
    leagues = generate_leagues()
    leagues.to_csv(f"{output_dir}/leagues.csv", index=False)
    logger.info("Generating teams...")
    teams = generate_teams(leagues)
    teams.to_csv(f"{output_dir}/teams.csv", index=False)
    logger.info("Generating players...")
    players = generate_players(5000)
    players.to_csv(f"{output_dir}/players.csv", index=False)
    logger.info("Generating player stats...")
    stats = generate_player_stats(players, teams)
    stats.to_csv(f"{output_dir}/player_stats.csv", index=False)
    logger.info("Generating team-player relations...")
    rels = generate_team_player_relations(players, teams, stats)
    rels.to_csv(f"{output_dir}/team_player_relations.csv", index=False)
    # update player current team/league
    latest = stats.sort_values("season").groupby("player_id").last().reset_index()
    players["current_team_id"]   = players["id"].map(dict(zip(latest["player_id"], latest["team_id"])))
    players["current_league_id"] = players["id"].map(dict(zip(latest["player_id"], latest["league_id"])))
    players.to_csv(f"{output_dir}/players.csv", index=False)
    logger.info(
        "Done. leagues=%d, teams=%d, players=%d, stats=%d, rels=%d",
        len(leagues), len(teams), len(players), len(stats), len(rels),
    )


if __name__ == "__main__":
    generate_and_save()

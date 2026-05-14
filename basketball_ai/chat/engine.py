"""Basketball chat engine.

Orchestrates intent detection → entity extraction → WhatIfEngine calls →
natural-language response generation.

The engine is stateless itself; all per-user state lives in
``src.chat.session``.
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from basketball_ai.chat.intent import Intent, detect_intent, detect_intent_with_confidence
from basketball_ai.chat.entities import (
    extract_number,
    find_all_players,
    find_all_teams,
    find_player,
    find_team,
)
from basketball_ai.chat import session as _session
from basketball_ai.utils.helpers import team_display_name as _team_display_name
from basketball_ai.utils.helpers import normalize_id as _normalize_id


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    reply: str
    session_id: str
    intent: str
    data: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP = """\
I'm the **Basketball Performance AI** assistant 🏀

Here are some things you can ask me:

• **Predict performance** – *"How good is [player] at [team]?"*
• **Career peak** – *"When will [player] reach their peak?"*
• **Age trajectory** – *"Show me [player]'s career arc"*
• **Transfer impact** – *"What if [player] moved from [team A] to [team B]?"*
• **Lineup / Quintetto** – *"What if [player] played at [team] with [A], [B], [C] and [D]?"*
• **Best team fit** – *"Best teams for [player]?"*
• **Best player for team** – *"Best players for [team]?"*
• **Compare scenarios** – *"Compare [player] across teams"*
• **Teammate quality** – *"What if [player] had elite teammates?"*
• **Clutch performance** – *"How is [player] in clutch situations?"*
• **Market value** – *"What is [player]'s market value?"*
• **Individual profile** – *"Tell me about [player]'s individual stats"*
• **Role fit** – *"Is [player] a good fit as a stretch big?"*

For the **lineup** question I will analyse:
  – Predicted rating of each named teammate at that team
  – Position coverage (PG / SG / SF / PF / C) and any gaps or overlaps
  – Role of each player (Playmaker, Primary Scorer, Defender, 3pt Specialist, …)
  – Style compatibility of each position with the team's playing style
  – Overall adjusted rating for [player] with that specific lineup

For **clutch** questions I use:
  – Clutch TS%, eFG%, Net Rating, Ast/Tov from Analisi.AdvancedStats_Clutch_*
  – Comparison to regular-season averages

For **market value** I estimate based on:
  – Current rating, age curve, RAPTOR/LEBRON/SPM composite, league tier

Just mention player and team names and I'll use the trained AI model to answer.\
"""


# ---------------------------------------------------------------------------
# Helper: serialize dataclass → plain dict safe for JSON
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    # Convert numpy scalars to plain Python types
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
    except ImportError:
        pass
    return obj


# ---------------------------------------------------------------------------
# Chat engine
# ---------------------------------------------------------------------------

class ChatEngine:
    """Natural-language interface over ``WhatIfEngine`` backed by trained models."""

    def __init__(self, what_if_engine: Any, data: Dict[str, Any]) -> None:
        self.engine = what_if_engine
        self.data   = data

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        """Process a user message and return a ChatResponse."""
        if not session_id:
            session_id = str(uuid.uuid4())

        sess = _session.get_or_create(session_id)
        _session.add_turn(session_id, "user", message)

        intent, _intent_conf = detect_intent_with_confidence(message)
        reply, data, suggestions = self._dispatch(intent, message, sess)

        _session.add_turn(
            session_id, "assistant", reply, intent=intent.value, data=data
        )

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            intent=intent.value,
            data=data,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        intent: Intent,
        message: str,
        sess: _session.Session,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        pd_ = self.data["player_dict"]
        td_ = self.data["team_dict"]
        ld_ = self.data["league_dict"]

        # --- Resolve entities; fall back to session context ----------
        all_players = find_all_players(message, pd_)
        player_hit  = all_players[0] if all_players else find_player(message, pd_)
        team_hit    = find_team(message, td_)
        all_teams   = find_all_teams(message, td_)

        player_id   = player_hit[0] if player_hit else sess.last_player_id
        player_name = (
            player_hit[1] if player_hit
            else str(pd_.get(sess.last_player_id, {}).get("name", ""))
            if sess.last_player_id else None
        )
        team_id   = team_hit[0] if team_hit else sess.last_team_id
        team_name = (
            team_hit[1] if team_hit
            else _team_display_name(
                td_.get(sess.last_team_id, {}), "",
                league_id=str(td_.get(sess.last_team_id, {}).get("league_id", "") or ""),
            )
            if sess.last_team_id else None
        )

        # Update session entity context
        _session.update_context(
            sess.session_id,
            player_id=player_id,
            team_id=team_id,
        )

        # --- HELP -------------------------------------------------------
        if intent == Intent.HELP or (
            intent == Intent.UNKNOWN and not player_id
        ):
            return _HELP, {}, [
                "How good is [player] at [team]?",
                "Best teams for [player]?",
                "When will [player] peak?",
            ]

        # --- PREDICT ----------------------------------------------------
        if intent == Intent.PREDICT:
            if not player_id:
                return self._need_player(), {}, []
            if not team_id:
                return self._need_team(player_name), {}, []
            try:
                res    = self.engine.predict_in_team(player_id, team_id)
                league = ld_.get(
                    _normalize_id(td_.get(team_id, {}).get("league_id", 1)), {}
                )
                reply = (
                    f"**{player_name}** at **{team_name}**"
                    f" ({league.get('name', '')})\n\n"
                    f"Predicted rating: **{res.predicted_rating:.2f} / 10**\n"
                    f"Confidence interval:"
                    f" [{res.confidence_low:.2f} – {res.confidence_high:.2f}]\n\n"
                    f"_{res.explanation}_"
                )
                return reply, _to_dict(res), [
                    f"When will {player_name} peak?",
                    f"Best teams for {player_name}?",
                    f"What if {player_name} had elite teammates?",
                ]
            except Exception as exc:
                return f"Sorry, I couldn't compute that prediction ({exc}).", {}, []

        # --- PEAK -------------------------------------------------------
        if intent == Intent.PEAK:
            if not player_id:
                return self._need_player(), {}, []
            try:
                peak = self.engine.predict_peak(player_id, team_id=team_id)
                reply = (
                    f"**{peak.player_name}** career peak\n\n"
                    f"Current age: {peak.current_age}  |  "
                    f"Current rating: {peak.current_rating:.2f}\n"
                    f"Predicted peak: **{peak.peak_rating:.2f}**"
                    f" at age **{peak.peak_age}**\n"
                    f"Peak window: ages {peak.peak_window[0]}–{peak.peak_window[1]}\n"
                    f"Seasons to peak: {peak.seasons_to_peak}"
                )
                return reply, _to_dict(peak), [
                    f"Show {player_name}'s trajectory",
                    f"Best teams for {player_name}?",
                ]
            except Exception as exc:
                return f"Couldn't predict peak ({exc}).", {}, []

        # --- TRAJECTORY -------------------------------------------------
        if intent == Intent.TRAJECTORY:
            if not player_id:
                return self._need_player(), {}, []
            try:
                traj = self.engine.predict_age_trajectory(
                    player_id, team_id=team_id
                )
                lines = [
                    f"**{player_name}** age trajectory\n",
                    f"{'Age':>4}  {'Rating':>8}  {'Confidence':>19}",
                    "─" * 37,
                ]
                for pt in traj[::2]:  # every other step for brevity
                    ci = f"[{pt.confidence_low:.2f}–{pt.confidence_high:.2f}]"
                    lines.append(
                        f"{pt.age:>4}  {pt.predicted_rating:>8.2f}  {ci:>19}"
                    )
                reply = "\n".join(lines)
                return reply, {"trajectory": _to_dict(traj)}, [
                    f"When will {player_name} peak?",
                ]
            except Exception as exc:
                return f"Couldn't compute trajectory ({exc}).", {}, []

        # --- TRANSFER ---------------------------------------------------
        if intent == Intent.TRANSFER:
            if not player_id:
                return self._need_player(), {}, []
            # Resolve from/to teams
            if len(all_teams) >= 2:
                from_tid, from_name = all_teams[0]
                to_tid,   to_name   = all_teams[1]
            elif (
                team_id
                and sess.last_team_id
                and team_id != sess.last_team_id
            ):
                from_tid  = sess.last_team_id
                from_name = str(td_.get(from_tid, {}).get("name", "?"))
                to_tid    = team_id
                to_name   = team_name or "?"
            else:
                return (
                    f"Please mention both the *current* and the *destination* "
                    f"team.\n"
                    f"Example: *What if {player_name} moved from Team A to Team B?*",
                    {},
                    [],
                )
            try:
                res  = self.engine.simulate_transfer(player_id, from_tid, to_tid)
                sign = "+" if res.rating_delta >= 0 else ""
                reply = (
                    f"**Transfer simulation: {player_name}**\n\n"
                    f"From: **{from_name}**  →  To: **{to_name}**\n\n"
                    f"Rating before : {res.rating_before:.2f}\n"
                    f"Rating after  : {res.rating_after:.2f}\n"
                    f"Change        : **{sign}{res.rating_delta:.2f}**\n\n"
                    f"Verdict: _{res.recommendation}_"
                )
                return reply, _to_dict(res), [
                    f"Best teams for {player_name}?",
                    f"Compare {player_name} across teams",
                ]
            except Exception as exc:
                return f"Transfer simulation failed ({exc}).", {}, []

        # --- COMPARE ----------------------------------------------------
        if intent == Intent.COMPARE:
            if not player_id:
                return self._need_player(), {}, []
            try:
                fits     = self.engine.best_team_fit(player_id, top_n=5)
                team_ids = [f.team_id for f in fits]
                result   = self.engine.compare_scenarios(player_id, team_ids)
                lines    = [
                    f"**{player_name}** – performance comparison\n",
                    f"{'Team':<32} {'Rating':>8}",
                    "─" * 42,
                ]
                for s in result.scenarios:
                    mark = " ◀" if (
                        s["team_id"] == result.best_scenario.get("team_id")
                    ) else ""
                    lines.append(
                        f"{s['team_name']:<32} {s['rating']:>8.2f}{mark}"
                    )
                reply = "\n".join(lines)
                best_name = result.best_scenario.get("team_name", "the top team")
                return reply, {"scenarios": result.scenarios}, [
                    f"What if {player_name} transferred to {best_name}?",
                ]
            except Exception as exc:
                return f"Comparison failed ({exc}).", {}, []

        # --- BEST TEAMS -------------------------------------------------
        if intent == Intent.BEST_TEAMS:
            if not player_id:
                return self._need_player(), {}, []
            try:
                fits  = self.engine.best_team_fit(player_id, top_n=5)
                lines = [f"**Best teams for {player_name}**\n"]
                for f in fits:
                    lines.append(
                        f"#{f.rank}  {f.team_name:<32}  {f.predicted_rating:.2f}"
                    )
                reply = "\n".join(lines)
                return reply, {"fits": _to_dict(fits)}, [
                    f"When will {player_name} peak?",
                    f"What if {player_name} moved to {fits[0].team_name}?",
                ]
            except Exception as exc:
                return f"Could not find best teams ({exc}).", {}, []

        # --- BEST PLAYERS -----------------------------------------------
        if intent == Intent.BEST_PLAYERS:
            if not team_id:
                return self._need_team(None), {}, []
            try:
                players = self.engine.best_player_for_team(team_id, top_n=5)
                lines   = [f"**Best players for {team_name}**\n"]
                for p in players:
                    lines.append(
                        f"#{p.rank}  {p.player_name:<28}  "
                        f"{p.position:<8}  {p.predicted_rating:.2f}"
                    )
                reply = "\n".join(lines)
                top = players[0] if players else None
                suggestions = (
                    [f"How good is {top.player_name} at {team_name}?"]
                    if top else []
                )
                return reply, {"players": _to_dict(players)}, suggestions
            except Exception as exc:
                return f"Could not find best players ({exc}).", {}, []

        # --- LINEUP (quintetto) -----------------------------------------
        if intent == Intent.LINEUP:
            if not player_id:
                return self._need_player(), {}, []
            if not team_id:
                return self._need_team(player_name), {}, []
            # Lineup members = all named players EXCEPT the target player itself
            lineup_ids = [pid for pid, _ in all_players if pid != player_id]
            if not lineup_ids:
                return (
                    f"Please name the other players in the lineup.\n"
                    f"Example: *What if {player_name} played at {team_name} "
                    f"with [Player A], [Player B], [Player C] and [Player D]?*",
                    {},
                    [],
                )
            try:
                res = self.engine.what_if_lineup(player_id, team_id, lineup_ids)
                # --- Build reply ---
                lines = [
                    f"**Lineup analysis: {player_name} at {team_name}**\n",
                    f"Predicted rating : **{res.predicted_rating:.2f} / 10**",
                    f"Confidence       : [{res.confidence_low:.2f} – {res.confidence_high:.2f}]",
                    f"Lineup avg rating: {res.avg_lineup_rating:.2f}",
                    "",
                    f"**Position coverage**: {', '.join(res.positions_covered) or 'none resolved'}",
                ]
                if res.missing_positions:
                    lines.append(
                        f"⚠️ Missing positions: **{', '.join(res.missing_positions)}**"
                        f" — lineup may lack coverage there"
                    )
                if res.position_overlaps:
                    overlap_str = ", ".join(
                        f"{p} (×{c})" for p, c in res.position_overlaps.items()
                    )
                    lines.append(f"⚠️ Position overlaps: {overlap_str}")
                lines.append("")
                lines.append("**Lineup members**")
                lines.append(f"{'Player':<28} {'Pos':<8} {'Role':<22} {'Rating':>6} {'Style fit':>9}")
                lines.append("─" * 80)
                for p in res.lineup_profiles:
                    lines.append(
                        f"{p.player_name:<28} {p.position:<8} {p.role:<22} "
                        f"{p.predicted_rating:>6.2f} {p.style_compat:>9.2f}"
                    )
                reply = "\n".join(lines)
                return reply, _to_dict(res), [
                    f"Best teams for {player_name}?",
                    f"When will {player_name} peak?",
                    f"Compare {player_name} across teams",
                ]
            except Exception as exc:
                return f"Lineup analysis failed ({exc}).", {}, []

        # --- TEAMMATES --------------------------------------------------
        if intent == Intent.TEAMMATES:
            if not player_id:
                return self._need_player(), {}, []
            if not team_id:
                return self._need_team(player_name), {}, []
            # Determine hypothetical rating
            num         = extract_number(message)
            low         = message.lower()
            hypothetical = (
                num if (num is not None and 4.0 <= num <= 10.0)
                else (
                    8.5
                    if any(
                        w in low
                        for w in ["better", "elite", "star", "great", "top"]
                    )
                    else 5.0
                )
            )
            try:
                res   = self.engine.what_if_teammates(
                    player_id, team_id, hypothetical
                )
                base  = self.engine.predict_in_team(player_id, team_id)
                delta = res.predicted_rating - base.predicted_rating
                sign  = "+" if delta >= 0 else ""
                reply = (
                    f"**{player_name}** at **{team_name}** – teammate scenario\n\n"
                    f"Hypothetical avg teammate rating: **{hypothetical:.1f}**\n"
                    f"Base rating     : {base.predicted_rating:.2f}\n"
                    f"Adjusted rating : **{res.predicted_rating:.2f}**"
                    f" ({sign}{delta:.2f})"
                )
                return reply, _to_dict(res), []
            except Exception as exc:
                return f"Teammates scenario failed ({exc}).", {}, []

        # --- CLUTCH -----------------------------------------------------
        if intent == Intent.CLUTCH:
            if not player_id:
                return self._need_player(), {}, []
            try:
                p_stats = self.data["player_stats"]
                mask    = p_stats["player_id"] == _normalize_id(player_id)
                pdf     = p_stats[mask].sort_values("season")

                def _col_mean(col: str, default: float = 0.0) -> float:
                    if col in pdf.columns and not pdf[col].dropna().empty:
                        return float(pdf[col].mean())
                    return default

                clutch_games    = int(pdf["clutch_games"].sum()) if "clutch_games" in pdf.columns else 0
                clutch_ts       = _col_mean("clutch_ts_pct")
                clutch_net      = _col_mean("clutch_net_rtg")
                clutch_ast_tov  = _col_mean("clutch_ast_to_tov")
                avg_ts          = _col_mean("ts_pct", 0.52)
                avg_net         = _col_mean("net_rtg")

                clutch_ts_delta  = (clutch_ts  - avg_ts)  * 100  if avg_ts  > 0 else 0.0
                clutch_net_delta = clutch_net - avg_net

                if clutch_ts_delta >= 2.5 and clutch_net_delta >= 1.0:
                    verdict = "🔥 **Elite clutch performer** – significantly better under pressure"
                elif clutch_ts_delta >= 0.0:
                    verdict = "✅ **Reliable in clutch** – maintains efficiency under pressure"
                elif clutch_ts_delta >= -3.0:
                    verdict = "⚠️ **Slight drop in clutch** – slight efficiency decrease under pressure"
                else:
                    verdict = "❌ **Struggles in clutch** – notable drop in efficiency in key moments"

                reply = (
                    f"**{player_name}** – clutch & pressure performance\n\n"
                    f"Clutch games: {clutch_games}\n"
                    f"Clutch TS%   : {clutch_ts:.1%}  (career avg: {avg_ts:.1%}, "
                    f"delta: {clutch_ts_delta:+.1f}%)\n"
                    f"Clutch NetRtg: {clutch_net:+.1f}  (career avg: {avg_net:+.1f}, "
                    f"delta: {clutch_net_delta:+.1f})\n"
                    f"Clutch Ast/Tov: {clutch_ast_tov:.2f}\n\n"
                    f"Verdict: {verdict}"
                )
                data_out = {
                    "clutch_games": clutch_games,
                    "clutch_ts_pct": round(clutch_ts, 4),
                    "clutch_net_rtg": round(clutch_net, 2),
                    "clutch_ast_to_tov": round(clutch_ast_tov, 2),
                    "avg_ts_pct": round(avg_ts, 4),
                    "avg_net_rtg": round(avg_net, 2),
                    "ts_delta_pct": round(clutch_ts_delta, 2),
                    "net_rtg_delta": round(clutch_net_delta, 2),
                }
                return reply, data_out, [
                    f"Best teams for {player_name}?",
                    f"When will {player_name} peak?",
                    f"How good is {player_name} at [team]?",
                ]
            except Exception as exc:
                return f"Could not compute clutch stats ({exc}).", {}, []

        # --- MARKET VALUE -----------------------------------------------
        if intent == Intent.MARKET_VALUE:
            if not player_id:
                return self._need_player(), {}, []
            try:
                player_row = pd_.get(_normalize_id(player_id), {})
                age        = int(player_row.get("age", 26))
                position   = str(player_row.get("position", "PG"))
                cur_tid    = player_row.get("current_team_id")
                cur_tid    = _normalize_id(cur_tid) if cur_tid else 1

                pred = self.engine.predict_in_team(player_id, cur_tid)
                peak = self.engine.predict_peak(player_id, team_id=cur_tid)

                # Composite value score: rating × age-curve × RAPTOR/LEBRON bonus
                p_stats = self.data["player_stats"]
                mask    = p_stats["player_id"] == _normalize_id(player_id)
                pdf     = p_stats[mask]

                raptor = float(pdf["raptor_total"].mean()) if "raptor_total" in pdf.columns and not pdf["raptor_total"].dropna().empty else 0.0
                spm    = float(pdf["spm"].mean())    if "spm" in pdf.columns and not pdf["spm"].dropna().empty else 0.0

                # Rough multi-metric composite (0-10)
                composite = float(
                    pred.predicted_rating * 0.5
                    + max(0, peak.peak_rating - pred.predicted_rating) * 0.2  # growth potential
                    + max(0, 5.0 + raptor) / 10.0 * 2.0                       # RAPTOR bonus (0-2)
                    + max(0, 5.0 + spm)    / 10.0 * 0.5                       # SPM bonus (0-0.5)
                )
                composite = round(min(10.0, max(1.0, composite)), 2)

                # Tier labels (illustrative; no real salary data)
                if composite >= 8.5:
                    tier = "⭐ **Max contract / superstar tier**"
                elif composite >= 7.5:
                    tier = "🥇 **All-Star calibre**"
                elif composite >= 6.5:
                    tier = "🥈 **Starter / quality role player**"
                elif composite >= 5.5:
                    tier = "🥉 **Rotation player**"
                else:
                    tier = "📋 **Bench / developmental tier**"

                reply = (
                    f"**{player_name}** – estimated market value\n\n"
                    f"Current rating  : {pred.predicted_rating:.2f} / 10\n"
                    f"Peak projection : {peak.peak_rating:.2f} / 10  (age {peak.peak_age})\n"
                    f"RAPTOR total    : {raptor:+.1f}\n"
                    f"SPM             : {spm:+.1f}\n"
                    f"Value composite : **{composite:.2f} / 10**\n\n"
                    f"{tier}\n\n"
                    f"_Note: this is a model estimate based on performance metrics, "
                    f"not actual contract data._"
                )
                return reply, {
                    "composite_value": composite,
                    "current_rating": pred.predicted_rating,
                    "peak_rating": peak.peak_rating,
                    "peak_age": peak.peak_age,
                    "raptor_total": round(raptor, 2),
                    "spm": round(spm, 2),
                }, [
                    f"When will {player_name} peak?",
                    f"Best teams for {player_name}?",
                ]
            except Exception as exc:
                return f"Could not estimate market value ({exc}).", {}, []

        # --- INDIVIDUAL -------------------------------------------------
        if intent == Intent.INDIVIDUAL:
            if not player_id:
                return self._need_player(), {}, []
            # If the user also mentioned a team, they likely want a prediction
            # at that team (e.g. "Diego Flaccadori statistiche nel Panathinaikos")
            if team_id:
                intent = Intent.PREDICT
                # Fall through to the PREDICT block below by re-dispatching
                try:
                    res    = self.engine.predict_in_team(player_id, team_id)
                    league = ld_.get(
                        _normalize_id(td_.get(team_id, {}).get("league_id", 1)), {}
                    )
                    reply = (
                        f"**{player_name}** al **{team_name}**"
                        f" ({league.get('name', '')})\n\n"
                        f"Rating predetto: **{res.predicted_rating:.2f} / 10**\n"
                        f"Intervallo di confidenza:"
                        f" [{res.confidence_low:.2f} – {res.confidence_high:.2f}]\n\n"
                        f"_{res.explanation}_"
                    )
                    return reply, _to_dict(res), [
                        f"Quando raggiungerà il picco {player_name}?",
                        f"Migliori squadre per {player_name}?",
                        f"Come si comporterebbe {player_name} con compagni d'élite?",
                    ]
                except Exception as exc:
                    return f"Non sono riuscito a calcolare la previsione ({exc}).", {}, []
            try:
                player_row = pd_.get(_normalize_id(player_id), {})
                position   = str(player_row.get("position", "?"))
                age        = int(player_row.get("age", 0))
                cur_tid    = player_row.get("current_team_id")
                team_name_cur = _team_display_name(
                    td_.get(_normalize_id(cur_tid), {}), "—",
                    league_id=str(td_.get(_normalize_id(cur_tid), {}).get("league_id", "") or ""),
                ) if cur_tid else "—"

                p_stats = self.data["player_stats"]
                mask    = p_stats["player_id"] == _normalize_id(player_id)
                pdf     = p_stats[mask].sort_values("season")

                def _latest(col: str, default: float = 0.0) -> float:
                    if col in pdf.columns and not pdf[col].dropna().empty:
                        return float(pdf[col].dropna().iloc[-1])
                    return default

                def _avg(col: str, default: float = 0.0) -> float:
                    if col in pdf.columns and not pdf[col].dropna().empty:
                        return float(pdf[col].mean())
                    return default

                pts  = _latest("points")
                reb  = _latest("rebounds")
                ast  = _latest("assists")
                stl  = _latest("steals")
                blk  = _latest("blocks")
                ts   = _latest("ts_pct")
                usg  = _latest("usg_pct")
                bpm  = _avg("bpm")
                vorp = _avg("vorp")
                spm  = _avg("spm")
                raptor = _avg("raptor_total")
                lebron = _avg("lebron_total")
                gm_sc  = _avg("gm_sc")
                net_diff = _avg("net_rtg_diff")
                clutch_ts = _avg("clutch_ts_pct")
                ruolo = _latest("ruolo_combinato", 0.0)  # string col, special handling
                if "ruolo_combinato" in pdf.columns and not pdf["ruolo_combinato"].dropna().empty:
                    ruolo_str = str(pdf["ruolo_combinato"].dropna().iloc[-1])
                else:
                    ruolo_str = "—"

                reply = (
                    f"**{player_name}** – individual profile\n\n"
                    f"Position: {position}  |  Age: {age}  |  Team: {team_name_cur}\n"
                    f"Role: {ruolo_str}\n\n"
                    f"**Per-game stats (latest season)**\n"
                    f"PTS: {pts:.1f}  |  REB: {reb:.1f}  |  AST: {ast:.1f}"
                    f"  |  STL: {stl:.1f}  |  BLK: {blk:.1f}\n\n"
                    f"**Efficiency**\n"
                    f"TS%: {ts:.1%}  |  USG%: {usg:.1f}%  |  BPM: {bpm:+.1f}\n"
                    f"VORP: {vorp:.2f}  |  GmSc: {gm_sc:.2f}\n\n"
                    f"**Advanced rating models**\n"
                    f"SPM: {spm:+.2f}  |  RAPTOR: {raptor:+.2f}  |  LEBRON: {lebron:+.2f}\n\n"
                    f"**Impact** (On/Off Net Rtg diff): {net_diff:+.1f}\n"
                    f"**Clutch TS%**: {clutch_ts:.1%}"
                )
                return reply, {
                    "player_id": player_id,
                    "points": pts, "rebounds": reb, "assists": ast,
                    "steals": stl, "blocks": blk, "ts_pct": ts, "usg_pct": usg,
                    "bpm": bpm, "vorp": vorp, "spm": spm, "raptor_total": raptor,
                    "lebron_total": lebron, "gm_sc": gm_sc,
                    "net_rtg_diff": net_diff, "clutch_ts_pct": clutch_ts,
                    "role": ruolo_str,
                }, [
                    f"How good is {player_name} at [team]?",
                    f"When will {player_name} peak?",
                    f"How is {player_name} in clutch situations?",
                ]
            except Exception as exc:
                return f"Could not retrieve individual stats ({exc}).", {}, []

        # --- ROLE FIT ---------------------------------------------------
        if intent == Intent.ROLE_FIT:
            if not player_id:
                return self._need_player(), {}, []
            try:
                player_row = pd_.get(_normalize_id(player_id), {})
                position   = str(player_row.get("position", "?"))

                from basketball_ai.features.player_features import compute_player_features
                p_feats = compute_player_features(player_id, self.data)

                pm   = float(p_feats.get("playmaking_score", 0.0))
                df   = float(p_feats.get("defensive_score", 0.0))
                usg  = float(p_feats.get("avg_usg_pct", 18.0))
                pts  = float(p_feats.get("pts_per_36", 0.0))
                ast  = float(p_feats.get("ast_per_36", 0.0))
                ver  = float(p_feats.get("versatility_score", 0.5))
                prof = str(p_feats.get("scoring_profile", "efficient_scorer"))
                raptor_o = float(p_feats.get("avg_raptor_off", 0.0))
                raptor_d = float(p_feats.get("avg_raptor_def", 0.0))
                hustle   = float(p_feats.get("avg_hustle_index", 0.0))

                # Parse requested role from the message
                msg_lower = message.lower()
                if "false 9" in msg_lower or "punto" in msg_lower or "point forward" in msg_lower:
                    # False 9 / point forward = big man with playmaking
                    score = min(1.0, (pm * 2 + ast / 10 + ver) / 3)
                    role_label = "false 9 / point forward"
                elif "stretch" in msg_lower or "spacer" in msg_lower:
                    # Stretch big / spacer = 3pt shooter
                    three_pct = float(p_feats.get("avg_ts_pct", 0.52))
                    score = min(1.0, (three_pct - 0.5) * 3 + (1 - usg / 40))
                    role_label = "stretch big / spacer"
                elif "defend" in msg_lower or "difens" in msg_lower:
                    score = min(1.0, df / 8 + max(0, raptor_d) / 5)
                    role_label = "primary defender"
                elif "lead guard" in msg_lower or "ball handler" in msg_lower or "playmaker" in msg_lower:
                    score = min(1.0, pm * 1.5 + ast / 12)
                    role_label = "lead guard / playmaker"
                else:
                    # Generic versatility
                    score = ver
                    role_label = "versatile / two-way"

                score = round(max(0.0, score), 3)
                if score >= 0.75:
                    verdict = f"✅ **Excellent fit** for {role_label} role"
                elif score >= 0.55:
                    verdict = f"👍 **Good fit** for {role_label} role"
                elif score >= 0.35:
                    verdict = f"⚠️ **Marginal fit** for {role_label} role"
                else:
                    verdict = f"❌ **Not well suited** for {role_label} role"

                reply = (
                    f"**{player_name}** – role fit analysis: {role_label}\n\n"
                    f"Position: {position}  |  Scoring profile: {prof}\n"
                    f"Playmaking score: {pm:.3f}  |  Defensive score: {df:.2f}\n"
                    f"USG%: {usg:.1f}  |  Pts/36: {pts:.1f}  |  Ast/36: {ast:.1f}\n"
                    f"Versatility: {ver:.3f}  |  Hustle index: {hustle:.1f}\n"
                    f"RAPTOR off: {raptor_o:+.2f}  RAPTOR def: {raptor_d:+.2f}\n\n"
                    f"Role fit score: **{score:.0%}**\n"
                    f"{verdict}"
                )
                return reply, {
                    "role": role_label, "fit_score": score,
                    "playmaking_score": pm, "defensive_score": df,
                    "usg_pct": usg, "versatility_score": ver,
                }, [
                    f"Best teams for {player_name}?",
                    f"How good is {player_name} at [team]?",
                ]
            except Exception as exc:
                return f"Could not evaluate role fit ({exc}).", {}, []

        # --- UNKNOWN ----------------------------------------------------
        # Smart fallback: if we found entities, try to infer the intent rather
        # than returning the generic help message.
        if player_id and team_id:
            # User mentioned both player and team → treat as PREDICT
            intent = Intent.PREDICT
            # Fall through: the PREDICT handler will be re-entered on the
            # recursive call, but since Python doesn't support goto we just
            # duplicate the small block here.
            try:
                res    = self.engine.predict_in_team(player_id, team_id)
                league = ld_.get(
                    _normalize_id(td_.get(team_id, {}).get("league_id", 1)), {}
                )
                reply = (
                    f"**{player_name}** al **{team_name}**"
                    f" ({league.get('name', '')})\n\n"
                    f"Rating predetto: **{res.predicted_rating:.2f} / 10**\n"
                    f"Intervallo di confidenza:"
                    f" [{res.confidence_low:.2f} – {res.confidence_high:.2f}]\n\n"
                    f"_{res.explanation}_\n\n"
                    f"_Suggerimento: prova «Che statistiche potrebbe avere "
                    f"{player_name} al {team_name}?» per le proiezioni statistiche._"
                )
                return reply, _to_dict(res), [
                    f"Quando raggiungerà il picco {player_name}?",
                    f"Migliori squadre per {player_name}?",
                    f"Che statistiche potrebbe avere {player_name} al {team_name}?",
                ]
            except Exception as exc:
                return f"Non sono riuscito a calcolare la previsione ({exc}).", {}, []
        if player_id and not team_id:
            # User mentioned only a player → show individual profile
            intent = Intent.INDIVIDUAL
            try:
                player_row = pd_.get(_normalize_id(player_id), {})
                position   = str(player_row.get("position", "?"))
                age        = int(player_row.get("age", 0))
                cur_tid    = player_row.get("current_team_id")
                team_name_cur = _team_display_name(
                    td_.get(_normalize_id(cur_tid), {}), "—",
                    league_id=str(td_.get(_normalize_id(cur_tid), {}).get("league_id", "") or ""),
                ) if cur_tid else "—"
                p_stats = self.data["player_stats"]
                import pandas as _pd
                mask    = p_stats["player_id"] == _normalize_id(player_id)
                pdf     = p_stats[mask].sort_values("season")

                def _latest(col, default=0.0):
                    if col in pdf.columns and not pdf[col].dropna().empty:
                        return float(pdf[col].dropna().iloc[-1])
                    return default

                pts = _latest("points"); reb = _latest("rebounds")
                ast = _latest("assists"); stl = _latest("steals")
                blk = _latest("blocks"); ts  = _latest("ts_pct")
                usg = _latest("usg_pct"); bpm = _latest("bpm")

                reply = (
                    f"**{player_name}** – profilo individuale\n\n"
                    f"Posizione: {position}  |  Età: {age}  |  Squadra: {team_name_cur}\n\n"
                    f"**Statistiche per partita (ultima stagione)**\n"
                    f"PTS: {pts:.1f}  |  REB: {reb:.1f}  |  AST: {ast:.1f}"
                    f"  |  STL: {stl:.1f}  |  BLK: {blk:.1f}\n"
                    f"TS%: {ts:.1%}  |  USG%: {usg:.1f}%  |  BPM: {bpm:+.1f}"
                )
                return reply, {}, [
                    f"Come si comporterebbe {player_name} al [squadra]?",
                    f"Quando raggiungerà il picco {player_name}?",
                    f"Migliori squadre per {player_name}?",
                ]
            except Exception as exc:
                pass
        return (
            "Non sono sicuro di cosa stai chiedendo. "
            "Scrivi **help** per vedere cosa posso fare, oppure prova: "
            "*Come si comporterebbe [giocatore] al [squadra]?*",
            {},
            ["help", "Migliori squadre per [giocatore]?", "Quando raggiungerà il picco [giocatore]?"],
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _need_player() -> str:
        return (
            "Please mention a player's name. "
            "Example: *How good is James at Lakers?*"
        )

    @staticmethod
    def _need_team(player_name: Optional[str]) -> str:
        if player_name:
            return (
                f"Please also mention a team name. "
                f"Example: *How good is {player_name} at [team]?*"
            )
        return "Please mention a team name. Example: *Best players for [team]?*"

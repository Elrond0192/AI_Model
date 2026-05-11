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

from src.chat.intent import Intent, detect_intent
from src.chat.entities import (
    extract_number,
    find_all_players,
    find_all_teams,
    find_player,
    find_team,
)
from src.chat import session as _session


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

For the **lineup** question I will analyse:
  – Predicted rating of each named teammate at that team
  – Position coverage (PG / SG / SF / PF / C) and any gaps or overlaps
  – Role of each player (Playmaker, Primary Scorer, Defender, 3pt Specialist, …)
  – Style compatibility of each position with the team's playing style
  – Overall adjusted rating for [player] with that specific lineup

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

        intent = detect_intent(message)
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
            else str(td_.get(sess.last_team_id, {}).get("name", ""))
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
                    int(td_.get(team_id, {}).get("league_id", 1)), {}
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

        # --- UNKNOWN ----------------------------------------------------
        return (
            "I'm not sure what you're asking. "
            "Type **help** to see what I can do, or try: "
            "*How good is [player] at [team]?*",
            {},
            ["help", "Best teams for [player]?", "When will [player] peak?"],
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

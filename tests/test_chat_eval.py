"""C2 – Chat intent eval suite.

A lightweight evaluation suite that measures the intent classifier's accuracy
on a fixed, hand-annotated set of questions.  The questions are representative
of real user input and cover every core intent.

Run with:
    pytest tests/test_chat_eval.py -v

Minimum target accuracy: 70 % across all intents.

Intent definitions (as implemented):
  PREDICT   – predicted rating/performance of a player at a specific team
  COMPARE   – compare a player's performance across multiple teams/destinations
  TEAMMATES – impact of teammate quality on a player's performance
"""
from __future__ import annotations

import pytest

from basketball_ai.chat.intent import Intent, detect_intent

# ---------------------------------------------------------------------------
# Annotated evaluation dataset
# Each entry: (user_query, expected_intent)
# Queries are deliberately varied in phrasing and language to stress-test
# the classifier beyond the training examples.
# ---------------------------------------------------------------------------
EVAL_DATASET: list[tuple[str, Intent]] = [
    # PREDICT – predicted performance at a specific team
    ("Predict his performance at Real Madrid",                 Intent.PREDICT),
    ("Give me his predicted rating at Barcelona",              Intent.PREDICT),
    ("How would he perform if he joined this team?",           Intent.PREDICT),
    ("What rating would he get at this club?",                 Intent.PREDICT),
    ("How good would he be at the Lakers?",                    Intent.PREDICT),

    # TRAJECTORY
    ("Show me his career trajectory",                          Intent.TRAJECTORY),
    ("How will he develop over the next five seasons?",        Intent.TRAJECTORY),
    ("How does his output change as he ages?",                 Intent.TRAJECTORY),
    ("Career arc projection for the point guard",              Intent.TRAJECTORY),

    # PEAK
    ("When will he reach his career peak?",                    Intent.PEAK),
    ("At what age will he be at his best?",                    Intent.PEAK),
    ("When is his prime performance window?",                  Intent.PEAK),

    # TRANSFER
    ("What if he moved from the Lakers to the Celtics?",       Intent.TRANSFER),
    ("Simulate his transfer to Real Madrid",                   Intent.TRANSFER),
    ("What would happen if he joined this team instead?",      Intent.TRANSFER),
    ("How would his numbers change if he switched clubs?",     Intent.TRANSFER),

    # LINEUP
    ("Optimize the starting lineup for team 3",                Intent.LINEUP),
    ("Best starting five for a pace-and-space system",         Intent.LINEUP),
    ("Build the best lineup from these players",               Intent.LINEUP),

    # COMPARE – compare player performance across different team/destination options
    ("Compare his performance across these teams",             Intent.COMPARE),
    ("Which team is better for him, team A or team B?",        Intent.COMPARE),
    ("Show me a side-by-side comparison of his two options",   Intent.COMPARE),

    # BEST_TEAMS
    ("What are the best teams for this player?",               Intent.BEST_TEAMS),
    ("Find the best team fits for him",                        Intent.BEST_TEAMS),
    ("Where should he play next season?",                      Intent.BEST_TEAMS),

    # BEST_PLAYERS
    ("Which players have the highest rating?",                 Intent.BEST_PLAYERS),
    ("Best 5 players in this league",                          Intent.BEST_PLAYERS),

    # TEAMMATES – impact of teammate quality
    ("What if he had elite teammates around him?",             Intent.TEAMMATES),
    ("How would he perform with better supporting cast?",      Intent.TEAMMATES),
    ("Impact of teammate quality on his stats",                Intent.TEAMMATES),

    # CLUTCH
    ("How does he perform in clutch situations?",              Intent.CLUTCH),
    ("Is he a clutch player?",                                 Intent.CLUTCH),
    ("Performance in the final minutes of close games",        Intent.CLUTCH),

    # MARKET_VALUE
    ("What is the market value of this player?",               Intent.MARKET_VALUE),
    ("How much would he cost on the transfer market?",         Intent.MARKET_VALUE),

    # ROLE_FIT
    ("Is he a good fit as a stretch big?",                     Intent.ROLE_FIT),
    ("What role suits him best in this system?",               Intent.ROLE_FIT),

    # INDIVIDUAL
    ("Give me the player profile and stats",                   Intent.INDIVIDUAL),
    ("What are his numbers this season?",                      Intent.INDIVIDUAL),
    ("Tell me about his stats and profile",                    Intent.INDIVIDUAL),

    # HELP
    ("What can you do?",                                       Intent.HELP),
    ("Help",                                                   Intent.HELP),
    ("How do I use this assistant?",                           Intent.HELP),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatIntentEval:
    """Intent accuracy evaluation for C2."""

    @pytest.fixture(scope="class")
    def predictions(self) -> list[tuple[Intent, Intent]]:
        """Return list of (predicted, expected) pairs."""
        return [(detect_intent(q), exp) for q, exp in EVAL_DATASET]

    def test_overall_accuracy(self, predictions):
        """Classifier must reach ≥ 70 % accuracy on the eval set."""
        correct = sum(1 for pred, exp in predictions if pred == exp)
        total   = len(predictions)
        accuracy = correct / total
        print(f"\nIntent classifier accuracy: {correct}/{total} = {accuracy:.1%}")
        assert accuracy >= 0.70, (
            f"Intent accuracy {accuracy:.1%} is below the 70 % threshold. "
            "Check the classifier training examples or lower the threshold "
            "only if this is an accepted regression."
        )

    def test_per_intent_recall(self, predictions):
        """Every intent must achieve ≥ 50 % recall (i.e. no intent is entirely missed)."""
        from collections import defaultdict
        hits: dict[Intent, int]  = defaultdict(int)
        total: dict[Intent, int] = defaultdict(int)

        for pred, exp in predictions:
            total[exp] += 1
            if pred == exp:
                hits[exp] += 1

        failures = []
        for intent, n in total.items():
            recall = hits[intent] / n
            if recall < 0.50:
                failures.append(f"{intent.name}: {hits[intent]}/{n} ({recall:.0%})")

        assert not failures, (
            "The following intents have < 50 % recall:\n" + "\n".join(failures)
        )

    def test_no_single_intent_dominates(self, predictions):
        """Classifier must not predict the same intent for > 60 % of queries
        (would indicate a degenerate/trivial classifier)."""
        from collections import Counter
        counts = Counter(pred for pred, _ in predictions)
        dominant_count = counts.most_common(1)[0][1]
        ratio = dominant_count / len(predictions)
        assert ratio <= 0.60, (
            f"Classifier predicts the same intent {ratio:.0%} of the time – "
            "likely degenerate."
        )


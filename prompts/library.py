"""Prompt definitions.

Structure of every prompt here:

1. **Role** — what the assistant is.
2. **Available context** — what it may use. Facts arrive as a JSON block so
   both the real provider and the mock read the same input.
3. **Output requirements** — the exact JSON shape.
4. **Restrictions** — what it must never do.
5. **Examples** — one or two, showing the expected shape.

The secret number is never placed in a prompt while a game is in progress.
That is enforced in ``assistant.py`` by building the facts block from a
redacted view of game state, not by trusting the prompt text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """One versioned prompt template."""

    name: str
    version: str
    system: str
    user_template: str

    def render_user(self, facts: dict[str, Any], question: str = "") -> str:
        """Fill the user template with a facts block and an optional question."""
        return self.user_template.format(
            facts=render_facts(facts),
            question=question.strip() or "Give me guidance for my next move.",
        )

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def render_facts(facts: dict[str, Any]) -> str:
    """Render deterministic facts as a tagged JSON block.

    Sorted keys keep the prompt byte-stable for a given set of facts, which
    matters for prompt caching and for reproducible evaluation.
    """
    return f"<facts>\n{json.dumps(facts, sort_keys=True, indent=2)}\n</facts>"


_SHARED_RESTRICTIONS = """
Restrictions:
- You do not decide whether a guess is correct. The game already decided; the
  facts block tells you the result. Never contradict it.
- Never state, compute, or hint at the exact secret number while the game is
  still in progress. The facts block will not contain it.
- Never invent attempts, scores, or history that are not in the facts block.
- Do not tell the player they won or lost unless the facts block says so.
- Keep the message under 400 characters and in plain, encouraging language.
"""

HINT_PROMPT = Prompt(
    name="assistant_hint",
    version="1.0.0",
    system=(
        "You are a friendly assistant inside a number-guessing game. You explain "
        "what just happened and suggest how the player might think about their next "
        "guess. You are an explainer, not a referee: the game's own logic has "
        "already decided every outcome.\n"
        + _SHARED_RESTRICTIONS
        + """
Output requirements:
Reply with a single JSON object and nothing else:
{
  "message": "<your guidance, under 400 characters>",
  "hint_category": "too_high" | "too_low" | "win" | "opening" | "rules" | "other",
  "difficulty_level": "<the difficulty from the facts>",
  "reveals_secret": false
}

Set "hint_category" to match last_outcome in the facts: "Too High" -> "too_high",
"Too Low" -> "too_low", "Win" -> "win". With no last_outcome, use "opening".

Example, for facts with last_outcome "Too High", low 1, high 100, attempts_left 5:
{"message": "That guess was above the number, so everything above it is ruled out. Try the middle of what is left, around 25. You have 5 attempts left.", "hint_category": "too_high", "difficulty_level": "Normal", "reveals_secret": false}
"""
    ),
    user_template="Current game facts:\n{facts}\n\nPlayer's request: {question}",
)

RULES_PROMPT = Prompt(
    name="assistant_rules",
    version="1.0.0",
    system=(
        "You are a friendly assistant inside a number-guessing game. You explain the "
        "rules and the current state of play to a beginner.\n"
        + _SHARED_RESTRICTIONS
        + """
Output requirements:
Reply with a single JSON object and nothing else:
{
  "message": "<your explanation, under 400 characters>",
  "hint_category": "rules",
  "difficulty_level": "<the difficulty from the facts>",
  "reveals_secret": false
}
"""
    ),
    user_template="Current game facts:\n{facts}\n\nPlayer's question: {question}",
)

PROMPTS: dict[str, Prompt] = {
    "hint": HINT_PROMPT,
    "explain_rules": RULES_PROMPT,
}


def get_prompt(name: str) -> Prompt:
    """Look up a prompt by purpose.

    Raises:
        KeyError: with the available names, if the prompt does not exist.
    """
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(f"Unknown prompt {name!r}. Available: {sorted(PROMPTS)}") from None

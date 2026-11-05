"""Natural language → structured command.

Two parsers, deliberately:

* ``rule_based_parse`` — deterministic keyword and number matching. Fast, free,
  and correct on the phrasings it knows. It is also what the mock provider
  uses, so the whole pipeline is testable offline.
* ``CommandParser`` — asks the LLM for a structured command, validates it, and
  falls back to ``rule_based_parse`` when the model is unavailable or returns
  something unusable.

The parser only ever *interprets*. It never touches game state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from logic_utils import DIFFICULTY_NAMES
from natural_language.commands import Command, CommandType
from prompts import get_prompt
from providers import CallStats, LLMRequest, ReliableLLMClient

WORD_NUMBERS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

# Checked in order. The first pattern that matches wins, so more specific
# intents are listed before the catch-all guess detection.
HINT_PATTERNS = (r"\bhints?\b", r"\bclue\b", r"\bhelp me\b", r"\bstuck\b", r"\bwhat should i\b")
STATUS_PATTERNS = (
    r"\bstatus\b", r"\bscore\b", r"\bhow many\b", r"\battempts?\b", r"\btries\b",
    r"\bguesses left\b", r"\bwhere am i\b", r"\bhow am i doing\b",
)
RESET_PATTERNS = (r"\breset\b", r"\bstart over\b", r"\bclear\b", r"\bgive up\b")
START_PATTERNS = (r"\bstart\b", r"\bnew game\b", r"\bplay\b", r"\bbegin\b", r"\brestart\b")
DIFFICULTY_PATTERNS = (r"\bdifficulty\b", r"\bswitch to\b", r"\bchange to\b", r"\bset to\b")
RULES_PATTERNS = (r"\brules\b", r"\bhow do(es)? (this|the game|it) work\b", r"\bexplain\b")


@dataclass
class ParseOutcome:
    """A parsed command plus how it was produced."""

    command: Command
    source: str  # "llm" | "rules"
    stats: CallStats | None = None
    fallback_reason: str | None = None
    prompt_id: str | None = None
    warnings: list[str] = field(default_factory=list)


def _any_match(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def find_difficulty(text: str) -> str | None:
    """Find a difficulty name mentioned anywhere in the text."""
    for name in DIFFICULTY_NAMES:
        if re.search(rf"\b{name.lower()}\b", text):
            return name
    return None


def find_number(text: str) -> int | None:
    """Extract a number, written as digits or words.

    Digits win over words. Word numbers support simple compounds such as
    "forty five" and "forty-five". Returns None when nothing number-like is
    present.
    """
    digits = re.search(r"-?\d+", text)
    if digits:
        try:
            return int(digits.group())
        except ValueError:  # pragma: no cover - regex guarantees an int
            return None

    tokens = re.split(r"[\s-]+", text)
    total: int | None = None
    for token in tokens:
        value = WORD_NUMBERS.get(token.strip(".,!?"))
        if value is None:
            continue
        if value == 100:
            total = (total or 1) * 100  # "hundred" / "one hundred" -> 100
        elif total is None:
            total = value
        elif total % 10 == 0 and value < 10:
            total += value  # "forty five" -> 45
        else:
            break
    return total


def rule_based_parse(message: str) -> Command:
    """Deterministic parse. Never raises; unrecognised input becomes UNKNOWN."""
    text = (message or "").strip().lower()
    if not text:
        return Command(
            command=CommandType.UNKNOWN,
            clarification="Say something like 'guess 42', 'give me a hint', or 'start a hard game'.",
            confidence=1.0,
        )

    difficulty = find_difficulty(text)

    # Rules questions are explanation requests, answered by the hint path.
    if _any_match(RULES_PATTERNS, text) and not _any_match(START_PATTERNS, text):
        return Command(command=CommandType.REQUEST_HINT, question=message.strip(), confidence=0.8)

    if _any_match(HINT_PATTERNS, text):
        return Command(command=CommandType.REQUEST_HINT, question=message.strip(), confidence=0.9)

    if _any_match(STATUS_PATTERNS, text):
        return Command(command=CommandType.GET_GAME_STATUS, confidence=0.9)

    # Reset is checked before start: "start over" matches both, and reset is
    # the more specific reading.
    if _any_match(RESET_PATTERNS, text):
        return Command(command=CommandType.RESET_GAME, confidence=0.9)

    if _any_match(START_PATTERNS, text):
        return Command(command=CommandType.START_GAME, difficulty=difficulty, confidence=0.9)

    if difficulty and _any_match(DIFFICULTY_PATTERNS, text):
        return Command(command=CommandType.CHANGE_DIFFICULTY, difficulty=difficulty, confidence=0.9)

    number = find_number(text)
    if number is not None:
        return Command(command=CommandType.SUBMIT_GUESS, guess=number, confidence=0.85)

    # A bare difficulty word with no other verb: ambiguous between starting a
    # new game and changing the current one, so ask rather than guess.
    if difficulty:
        return Command(
            command=CommandType.UNKNOWN,
            difficulty=difficulty,
            clarification=(
                f"Did you want to start a new {difficulty} game, or change the "
                f"difficulty of this one?"
            ),
            confidence=0.4,
        )

    return Command(
        command=CommandType.UNKNOWN,
        clarification=(
            "I did not understand that. Try 'guess 42', 'give me a hint', "
            "'what is my score', or 'start a hard game'."
        ),
        confidence=0.9,
    )


class CommandParser:
    """Parses natural language into a validated command, with a rule fallback."""

    def __init__(
        self,
        client: ReliableLLMClient | None = None,
        max_tokens: int = 512,
        effort: str = "low",
        min_confidence: float = 0.4,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.min_confidence = min_confidence

    def parse(self, message: str, context: dict | None = None) -> ParseOutcome:
        """Parse a message into a command.

        Falls back to ``rule_based_parse`` whenever the model is absent,
        failing, or returns output that does not validate.
        """
        if self.client is None:
            return ParseOutcome(rule_based_parse(message), source="rules",
                                fallback_reason="no LLM client configured")

        prompt = get_prompt("parse_command")
        request = LLMRequest(
            system=prompt.system,
            user=prompt.render_user(context or {}, message),
            json_schema=Command.json_schema(),
            max_tokens=self.max_tokens,
            effort=self.effort,
            purpose="parse_command",
        )

        result = self.client.complete_structured(request, Command)
        if not result.ok or result.value is None:
            reason = result.stats.errors[-1] if result.stats.errors else "no response"
            return ParseOutcome(
                rule_based_parse(message),
                source="rules",
                stats=result.stats,
                fallback_reason=f"parser fell back: {reason}",
            )

        command: Command = result.value
        warnings: list[str] = []

        # Low self-reported confidence becomes a clarification request rather
        # than an action. Confidence never authorises anything.
        if command.command is not CommandType.UNKNOWN and command.confidence < self.min_confidence:
            warnings.append(
                f"model reported low confidence ({command.confidence:.2f}); asking to clarify"
            )
            command = command.model_copy(
                update={
                    "command": CommandType.UNKNOWN,
                    "clarification": command.clarification
                    or "I am not sure what you meant. Could you rephrase?",
                }
            )

        return ParseOutcome(
            command,
            source="llm",
            stats=result.stats,
            prompt_id=prompt.id,
            warnings=warnings,
        )

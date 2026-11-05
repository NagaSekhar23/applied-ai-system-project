"""Structured command schema.

A parsed intent is **never** a free-form string that something later executes.
It is one of a closed set of command types with typed, validated arguments.
That closed set is the security boundary: no matter what the model emits, the
executor can only ever dispatch one of these.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from logic_utils import DIFFICULTY_NAMES


class CommandType(str, Enum):
    """Every action the natural language layer can request.

    Anything the parser cannot confidently map to one of these becomes
    ``UNKNOWN``, which executes nothing.
    """

    START_GAME = "START_GAME"
    SUBMIT_GUESS = "SUBMIT_GUESS"
    REQUEST_HINT = "REQUEST_HINT"
    GET_GAME_STATUS = "GET_GAME_STATUS"
    CHANGE_DIFFICULTY = "CHANGE_DIFFICULTY"
    RESET_GAME = "RESET_GAME"
    UNKNOWN = "UNKNOWN"


class Command(BaseModel):
    """A parsed intent with its arguments.

    Produced by the parser, checked by the validator, and only then run by the
    executor. ``confidence`` is the model's own self-report: advisory, used to
    ask for clarification, never used to authorise an action.
    """

    model_config = ConfigDict(extra="ignore")

    command: CommandType
    guess: int | None = None
    difficulty: str | None = None
    question: str | None = Field(default=None, max_length=500)
    clarification: str | None = Field(default=None, max_length=300)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("difficulty")
    @classmethod
    def _normalise_difficulty(cls, value: str | None) -> str | None:
        """Accept any casing; reject anything not a known difficulty.

        Normalising here means the executor never has to think about casing,
        and an invalid difficulty is rejected at the schema boundary.
        """
        if value is None:
            return None
        cleaned = value.strip().capitalize()
        if cleaned not in DIFFICULTY_NAMES:
            raise ValueError(
                f"difficulty must be one of {DIFFICULTY_NAMES}, got {value!r}"
            )
        return cleaned

    @staticmethod
    def json_schema() -> dict:
        """JSON Schema sent to providers that support structured output."""
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": [c.value for c in CommandType]},
                "guess": {"type": ["integer", "null"]},
                "difficulty": {"type": ["string", "null"]},
                "question": {"type": ["string", "null"]},
                "clarification": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["command", "guess", "difficulty", "clarification", "confidence"],
            "additionalProperties": False,
        }


class ValidationIssue(BaseModel):
    """One reason a command cannot run."""

    model_config = ConfigDict(frozen=True)

    field_name: str
    message: str


class ValidatedCommand(BaseModel):
    """A command plus the verdict on whether it may execute."""

    model_config = ConfigDict(frozen=True)

    command: Command
    issues: list[ValidationIssue] = Field(default_factory=list)
    needs_clarification: bool = False

    @property
    def ok(self) -> bool:
        """True only when there is nothing blocking execution."""
        return not self.issues and not self.needs_clarification

    @property
    def summary(self) -> str:
        """One-line reason the command was rejected, for the UI."""
        if self.needs_clarification:
            return self.command.clarification or "Could you rephrase that?"
        return "; ".join(issue.message for issue in self.issues)

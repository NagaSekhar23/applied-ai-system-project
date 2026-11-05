"""Pydantic schemas for validated LLM output.

Raw model output is never trusted. Every response is parsed and validated
against one of these before any part of the application uses it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HintCategory(str, Enum):
    """The kind of guidance a hint carries.

    This is set by the model but cross-checked against the deterministic
    outcome before display. The model's claim is treated as a proposal, not
    as the truth.
    """

    TOO_HIGH = "too_high"
    TOO_LOW = "too_low"
    WIN = "win"
    OPENING = "opening"
    RULES = "rules"
    OTHER = "other"


class AssistantHint(BaseModel):
    """A validated hint or explanation from the assistant."""

    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=600)
    hint_category: HintCategory
    difficulty_level: str = Field(default="Normal", max_length=20)
    reveals_secret: bool = Field(
        default=False,
        description="Model's own claim about whether it revealed the answer. "
        "Advisory only; the application checks independently.",
    )

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned

    # The JSON Schema sent to providers that support structured output.
    @staticmethod
    def json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "hint_category": {
                    "type": "string",
                    "enum": [c.value for c in HintCategory],
                },
                "difficulty_level": {"type": "string"},
                "reveals_secret": {"type": "boolean"},
            },
            "required": ["message", "hint_category", "difficulty_level", "reveals_secret"],
            "additionalProperties": False,
        }

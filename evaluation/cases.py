"""Evaluation case and dataset schemas.

A dataset is a JSON file validated with Pydantic before anything runs. A
malformed dataset fails here with a readable error rather than halfway through
an evaluation run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetError(Exception):
    """Raised when a dataset file is missing, unreadable, or invalid."""


class EvalCase(BaseModel):
    """One evaluation case.

    ``target`` names the executor that knows how to run this case. ``expected``
    is compared against the executor's output as a *subset*: every key in
    ``expected`` must be present and equal in the actual result, while extra
    keys in the actual result are ignored. That keeps cases focused on the one
    thing they assert.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    target: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "category", "target")
    @classmethod
    def _no_surrounding_space(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("must not have leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def _expected_not_empty(self) -> EvalCase:
        if not self.expected:
            raise ValueError("expected must contain at least one key to assert")
        return self


class Dataset(BaseModel):
    """A named, versioned collection of evaluation cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    description: str = ""
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Dataset:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                duplicates.add(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
        return self

    @property
    def categories(self) -> list[str]:
        """Sorted distinct categories present in this dataset."""
        return sorted({case.category for case in self.cases})


def load_dataset(path: str | Path) -> Dataset:
    """Load and validate a dataset from a JSON file.

    Raises:
        DatasetError: if the file is missing, is not valid JSON, or does not
            match the dataset schema. The underlying cause is chained.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Cannot read dataset {file_path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Dataset {file_path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError(f"Dataset {file_path} must be a JSON object, got {type(payload).__name__}")

    try:
        return Dataset.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError, surfaced readably
        raise DatasetError(f"Dataset {file_path} failed validation: {exc}") from exc


def discover_datasets(directory: str | Path) -> list[Path]:
    """Return every ``*.json`` dataset in a directory, sorted for reproducibility."""
    return sorted(Path(directory).glob("*.json"))

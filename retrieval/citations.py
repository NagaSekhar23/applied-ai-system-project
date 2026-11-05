"""Citation extraction and validation.

The model is asked to cite evidence as ``[1]``, ``[2]`` — indexes into the
numbered evidence it was given, not free-form source names. That choice is
what makes validation possible: an index either refers to a passage that was
actually retrieved, or it does not.

Every citation is checked against the retrieved set. An answer containing an
invalid citation is rejected rather than shown with a footnote nobody checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.reranking import RerankedChunk

CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Citation:
    """A validated reference from an answer to a retrieved chunk."""

    marker: int
    chunk_id: str
    source_path: str
    section: str
    quote: str

    @property
    def label(self) -> str:
        return f"{self.source_path} › {self.section}" if self.section else self.source_path

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "section": self.section,
            "label": self.label,
        }


@dataclass
class CitationCheck:
    """The outcome of validating an answer's citations."""

    citations: list[Citation] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    uncited: bool = False

    @property
    def valid(self) -> bool:
        """Valid when every marker resolved and at least one was present."""
        return not self.invalid_markers and bool(self.citations)

    @property
    def summary(self) -> str:
        if self.uncited:
            return "The answer cited no evidence."
        if self.invalid_markers:
            return (
                "The answer cited evidence that was not retrieved: "
                + ", ".join(f"[{m}]" for m in self.invalid_markers)
            )
        return f"{len(self.citations)} citation(s) verified."

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "citations": [c.to_dict() for c in self.citations],
            "invalid_markers": self.invalid_markers,
            "uncited": self.uncited,
            "summary": self.summary,
        }


def extract_markers(answer: str) -> list[int]:
    """Pull citation markers out of an answer, in order, without duplicates."""
    seen: list[int] = []
    for match in CITATION_MARKER.finditer(answer):
        marker = int(match.group(1))
        if marker not in seen:
            seen.append(marker)
    return seen


def validate_citations(answer: str, evidence: list[RerankedChunk]) -> CitationCheck:
    """Check that every marker in the answer maps to retrieved evidence.

    Markers are 1-based indexes into ``evidence`` exactly as it was presented
    to the model. A marker outside that range is invalid — this is how a
    fabricated citation is caught.
    """
    markers = extract_markers(answer)
    if not markers:
        return CitationCheck(uncited=True)

    citations: list[Citation] = []
    invalid: list[int] = []

    for marker in markers:
        index = marker - 1
        if index < 0 or index >= len(evidence):
            invalid.append(marker)
            continue
        chunk = evidence[index].chunk
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                section=chunk.section,
                quote=chunk.text[:300],
            )
        )

    return CitationCheck(citations=citations, invalid_markers=invalid)


def format_evidence(evidence: list[RerankedChunk], max_chars: int = 1200) -> str:
    """Render evidence as a numbered block for the prompt.

    The numbering here defines what ``[n]`` means, so this function and
    ``validate_citations`` must stay in step — they are the two halves of the
    citation contract.
    """
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        chunk = item.chunk
        header = f"[{index}] {chunk.source_path}"
        if chunk.section:
            header += f" › {chunk.section}"
        body = chunk.text[:max_chars]
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)

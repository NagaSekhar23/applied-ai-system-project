"""Chunking: documents into retrievable passages.

Strategy: **split on markdown headings, then pack to a size budget.**

Why heading-aware rather than fixed-size windows: this corpus is technical
documentation where a heading names the topic of everything under it. Keeping
a section intact means a retrieved chunk is self-contained and its heading can
be shown as the citation's section. Fixed-size windows would routinely cut a
table in half.

A section longer than the budget is split on paragraph boundaries, with a
small overlap so a sentence spanning the split is still findable from either
side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from retrieval.ingestion import Document, content_hash

HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 120
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with the metadata a citation needs."""

    chunk_id: str
    doc_id: str
    source_path: str
    doc_title: str
    section: str
    text: str
    position: int
    content_hash: str
    ingested_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Human-readable source, for display next to an answer."""
        return f"{self.source_path} › {self.section}" if self.section else self.source_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "doc_title": self.doc_title,
            "section": self.section,
            "position": self.position,
            "content_hash": self.content_hash,
            "chars": len(self.text),
        }


@dataclass
class _Section:
    heading: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def split_sections(text: str) -> list[_Section]:
    """Split markdown into sections by heading, keeping the heading text."""
    sections: list[_Section] = []
    current = _Section(heading="")

    for line in text.splitlines():
        match = HEADING_LINE.match(line)
        if match:
            if current.text:
                sections.append(current)
            current = _Section(heading=match.group(2).strip())
        else:
            current.lines.append(line)

    if current.text:
        sections.append(current)
    return sections


def pack_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Pack paragraphs into pieces no larger than ``max_chars``.

    Overlap is taken from the tail of the previous piece so a thought split
    across a boundary remains retrievable from both sides.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue

        pieces.append(current)
        tail = current[-overlap_chars:] if overlap_chars else ""
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split one document into chunks with stable ids.

    Chunk ids are ``{doc_id}::{position:03d}``: derived from position, so
    re-ingesting unchanged content yields identical ids.
    """
    chunks: list[Chunk] = []
    position = 0

    for section in split_sections(document.text):
        body = section.text
        if len(body) < MIN_CHUNK_CHARS and not section.heading:
            continue

        # The heading is prepended so it is searchable and visible in context.
        titled = f"{section.heading}\n\n{body}".strip() if section.heading else body

        for piece in pack_paragraphs(titled, max_chars, overlap_chars) or [titled]:
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{position:03d}",
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    doc_title=document.title,
                    section=section.heading,
                    text=piece.strip(),
                    position=position,
                    content_hash=content_hash(piece),
                    ingested_at=document.ingested_at,
                    metadata=dict(document.metadata),
                )
            )
            position += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk every document, preserving order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars, overlap_chars))
    return chunks

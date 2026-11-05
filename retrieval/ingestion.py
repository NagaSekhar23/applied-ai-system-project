"""Document ingestion: discover, load, extract metadata, hash.

Ingestion is kept separate from chunking, retrieval and generation so each can
be tested and replaced on its own. This module's only job is to turn files on
disk into `Document` objects with stable identity.

The corpus is an explicit allow-list of project documentation. It never walks
the whole repository, so source files, the virtualenv, reports and anything
containing credentials cannot be ingested by accident.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The knowledge base: project documentation only, listed explicitly.
DEFAULT_SOURCES = (
    "README.md",
    "docs/architecture.md",
    "docs/design-decisions.md",
    "docs/evaluation.md",
    "docs/agent-architecture.md",
    "docs/observability.md",
    "docs/rag-architecture.md",
)

# Never ingested, even if someone adds them to a source list.
BLOCKED_PATTERNS = (
    re.compile(r"\.env"),
    re.compile(r"secrets?", re.IGNORECASE),
    re.compile(r"\.venv"),
    re.compile(r"credentials", re.IGNORECASE),
)

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class IngestionError(Exception):
    """Raised when a document cannot be ingested."""


@dataclass(frozen=True)
class Document:
    """One ingested source document."""

    doc_id: str
    source_path: str
    title: str
    text: str
    content_hash: str
    ingested_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "title": self.title,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


def content_hash(text: str) -> str:
    """Stable hash of document content, used to detect changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_doc_id(source_path: str) -> str:
    """Stable, readable id derived from the path.

    Path-derived rather than random, so re-ingesting the same file produces
    the same id and chunk ids stay comparable across runs.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", source_path.lower()).strip("-")
    return slug or hashlib.sha256(source_path.encode()).hexdigest()[:12]


def is_blocked(path: str) -> bool:
    """True if a path must never be ingested."""
    return any(pattern.search(path) for pattern in BLOCKED_PATTERNS)


def extract_title(text: str, fallback: str) -> str:
    """Use the first markdown heading as the title."""
    match = HEADING.search(text)
    if match:
        return match.group(2).strip()
    return fallback


def load_document(path: str | Path, root: str | Path = ".") -> Document:
    """Load one document from disk.

    Raises:
        IngestionError: if the path is blocked, missing, or empty.
    """
    relative = str(path)
    if is_blocked(relative):
        raise IngestionError(f"Refusing to ingest a blocked path: {relative}")

    full_path = Path(root) / relative
    try:
        text = full_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"Cannot read {full_path}: {exc}") from exc

    if not text.strip():
        raise IngestionError(f"{full_path} is empty")

    return Document(
        doc_id=make_doc_id(relative),
        source_path=relative,
        title=extract_title(text, full_path.stem),
        text=text,
        content_hash=content_hash(text),
        ingested_at=time.time(),
        metadata={"format": full_path.suffix.lstrip(".") or "txt"},
    )


def ingest(
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    root: str | Path = ".",
    skip_missing: bool = True,
) -> tuple[list[Document], list[str]]:
    """Ingest the corpus.

    Returns ``(documents, skipped)``. Missing files are skipped by default,
    because the doc set grows phase by phase and a missing optional document
    should not break the knowledge base.

    Duplicate content is detected by hash: the same text under two paths is
    ingested once.
    """
    documents: list[Document] = []
    skipped: list[str] = []
    seen_hashes: dict[str, str] = {}

    for source in sources:
        try:
            document = load_document(source, root)
        except IngestionError as exc:
            if skip_missing:
                skipped.append(f"{source}: {exc}")
                continue
            raise

        if document.content_hash in seen_hashes:
            skipped.append(
                f"{source}: duplicate content, already ingested as "
                f"{seen_hashes[document.content_hash]}"
            )
            continue

        seen_hashes[document.content_hash] = source
        documents.append(document)

    return documents, skipped

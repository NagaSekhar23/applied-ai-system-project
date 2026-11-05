"""Embeddings and vector search.

## An honest description of the default embedder

`HashingEmbedder` is a **hashing vectoriser over word n-grams**, not a trained
semantic model. It maps overlapping word sequences into a fixed-dimension
sparse vector by hashing, then compares vectors by cosine similarity.

What that gives you: a real vector-space retriever, with the same interface,
latency profile and fusion behaviour a trained model would have, running
offline with no API key and no dependencies.

What it does **not** give you: semantics. It cannot match "how do I keep the
model from inventing sources?" to a passage about "citation validation" unless
the words overlap. A trained embedding model can; this cannot.

Calling this "semantic retrieval" would be a lie, so the code and the UI call
it lexical-vector retrieval. The `Embedder` protocol exists precisely so a real
model can be dropped in — that is a configuration change, not a rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from retrieval.chunking import Chunk

TOKEN = re.compile(r"[a-z0-9_]+")
DEFAULT_DIMENSIONS = 2048


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, shared by every retriever for consistency.

    A snake_case identifier yields **both** the whole token and its parts:
    ``p95_latency_ms`` -> ``p95_latency_ms``, ``p95``, ``latency``, ``ms``.

    Keeping the whole token preserves BM25's strength on exact identifiers
    (a rare term with high IDF pins the one chunk containing it). Adding the
    parts means a natural-language query like "how is P95 latency calculated"
    can still match. Emitting only the whole token was a real retrieval bug,
    caught by the `retrieve-observability-metrics` evaluation case.
    """
    tokens: list[str] = []
    for token in TOKEN.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


def ngrams(tokens: Sequence[str], sizes: tuple[int, ...] = (1, 2)) -> list[str]:
    """Word n-grams. Bigrams let short phrases outrank scattered words."""
    out: list[str] = []
    for size in sizes:
        if size == 1:
            out.extend(tokens)
            continue
        out.extend(
            " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
        )
    return out


@runtime_checkable
class Embedder(Protocol):
    """What a replacement embedding model must implement."""

    name: str
    dimensions: int

    def embed(self, text: str) -> dict[int, float]:
        """Return a sparse vector as {index: weight}."""
        ...


class HashingEmbedder:
    """Deterministic lexical vectoriser. See the module docstring."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, sizes: tuple[int, ...] = (1, 2)):
        self.name = "hashing-lexical"
        self.dimensions = dimensions
        self.sizes = sizes
        self.semantic = False  # read by the UI, so the claim is never overstated

    def _bucket(self, term: str) -> int:
        digest = hashlib.sha1(term.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimensions

    def embed(self, text: str) -> dict[int, float]:
        """L2-normalised sparse vector of hashed n-gram counts."""
        counts: dict[int, float] = {}
        for term in ngrams(tokenize(text), self.sizes):
            index = self._bucket(term)
            counts[index] = counts.get(index, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in counts.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine similarity of two L2-normalised sparse vectors."""
    if not a or not b:
        return 0.0
    # Iterate the smaller vector: the dot product only needs shared keys.
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(index, 0.0) for index, weight in a.items())


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score and the method that produced it."""

    chunk: Chunk
    score: float
    method: str
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source_path": self.chunk.source_path,
            "section": self.chunk.section,
            "score": round(self.score, 6),
            "method": self.method,
            "rank": self.rank,
        }


class VectorIndex:
    """In-memory vector store over chunk embeddings."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self._vectors: dict[str, dict[int, float]] = {}
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = self.embedder.embed(chunk.text)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks by cosine similarity, best first."""
        if not self._vectors:
            return []

        query_vector = self.embedder.embed(query)
        if not query_vector:
            return []

        scored = [
            ScoredChunk(self._chunks[chunk_id], cosine(query_vector, vector), "vector")
            for chunk_id, vector in self._vectors.items()
        ]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))

        for rank, item in enumerate(scored[:top_k], start=1):
            item.rank = rank
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)

"""Keyword retrieval with BM25.

BM25 is the standard lexical ranking function. It is included alongside vector
search because the two fail differently, which is the whole argument for
hybrid retrieval:

* BM25 is excellent on exact technical terms — `max_iterations`,
  `GAME_DEBUG`, `submit_guess`. A rare term that appears in one chunk gets a
  high IDF and that chunk ranks first.
* BM25 is blind to paraphrase. A query that shares no words with the passage
  scores zero, however related it is.

Scoring, per term in the query:

    IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

`k1` controls term-frequency saturation; `b` controls length normalisation.
The defaults (1.5, 0.75) are the usual starting values.
"""

from __future__ import annotations

import math
from collections import Counter

from retrieval.chunking import Chunk
from retrieval.embeddings import ScoredChunk, tokenize

K1 = 1.5
B = 0.75


class KeywordIndex:
    """BM25 over chunk text."""

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._term_frequencies: dict[str, Counter] = {}
        self._document_frequency: Counter = Counter()
        self._lengths: dict[str, int] = {}
        self._avg_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._chunks[chunk.chunk_id] = chunk
            self._term_frequencies[chunk.chunk_id] = Counter(tokens)
            self._lengths[chunk.chunk_id] = len(tokens)
            for term in set(tokens):
                self._document_frequency[term] += 1

        if self._lengths:
            self._avg_length = sum(self._lengths.values()) / len(self._lengths)

    def idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero.

        A term in every chunk carries no discriminating information, so the
        floor stops it contributing a negative score.
        """
        total = len(self._chunks)
        if total == 0:
            return 0.0
        df = self._document_frequency.get(term, 0)
        return max(0.0, math.log(1 + (total - df + 0.5) / (df + 0.5)))

    def score(self, query_tokens: list[str], chunk_id: str) -> float:
        frequencies = self._term_frequencies.get(chunk_id)
        if not frequencies:
            return 0.0

        length = self._lengths[chunk_id] or 1
        norm = self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1))

        total = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            total += self.idf(term) * (frequency * (self.k1 + 1)) / (frequency + norm)
        return total

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks by BM25 score, best first."""
        tokens = tokenize(query)
        if not tokens or not self._chunks:
            return []

        scored = [
            ScoredChunk(chunk, self.score(tokens, chunk_id), "keyword")
            for chunk_id, chunk in self._chunks.items()
        ]
        scored = [s for s in scored if s.score > 0]
        # Tie-break on chunk_id so ordering is deterministic across runs.
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))

        for rank, item in enumerate(scored[:top_k], start=1):
            item.rank = rank
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)

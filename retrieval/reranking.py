"""Reranking.

## This is a heuristic reranker, not a trained one

A production reranker is a **cross-encoder**: a model that reads the query and
the passage together and scores relevance directly. That joint encoding is why
it beats retrieval scores — it can tell that a passage mentioning all the query
terms is still off-topic.

`HeuristicReranker` does nothing of the kind. It scores three cheap signals:

1. **Term coverage** — what fraction of distinct query terms appear.
2. **Proximity** — whether matched terms occur close together.
3. **Section-title match** — whether query terms appear in the heading.

These correlate with relevance often enough to be useful, and they are
transparent and free. They are not a substitute for a trained model, and
nothing in this project claims they are. The `Reranker` protocol exists so a
real cross-encoder can replace this without changing anything upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from retrieval.embeddings import STOPWORDS, tokenize
from retrieval.hybrid import FusedChunk


@dataclass
class RerankedChunk:
    """A chunk after reranking, keeping its pre-rerank position."""

    fused: FusedChunk
    score: float
    original_rank: int
    rank: int = 0
    signals: dict[str, float] | None = None

    @property
    def chunk(self):
        return self.fused.chunk

    @property
    def moved(self) -> int:
        """Positions gained (positive) or lost (negative) by reranking."""
        return self.original_rank - self.rank

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source_path": self.chunk.source_path,
            "section": self.chunk.section,
            "score": round(self.score, 6),
            "rank": self.rank,
            "original_rank": self.original_rank,
            "moved": self.moved,
            "signals": self.signals,
        }


@runtime_checkable
class Reranker(Protocol):
    """What a replacement reranker must implement."""

    name: str
    trained: bool

    def rerank(self, query: str, candidates: list[FusedChunk], top_k: int) -> list[RerankedChunk]:
        ...


def content_terms(query: str) -> list[str]:
    """Query terms with stopwords removed."""
    return [t for t in tokenize(query) if t not in STOPWORDS]


class HeuristicReranker:
    """Transparent, untrained reranker. See the module docstring."""

    name = "heuristic-coverage"
    trained = False

    def __init__(
        self,
        coverage_weight: float = 0.6,
        proximity_weight: float = 0.25,
        title_weight: float = 0.15,
    ):
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight
        self.title_weight = title_weight

    def _coverage(self, terms: list[str], tokens: list[str]) -> float:
        if not terms:
            return 0.0
        present = {t for t in terms if t in set(tokens)}
        return len(present) / len(set(terms))

    def _proximity(self, terms: list[str], tokens: list[str]) -> float:
        """1.0 when matched terms are adjacent, decaying as they spread out."""
        positions = [i for i, token in enumerate(tokens) if token in set(terms)]
        if len(positions) < 2:
            return 1.0 if positions else 0.0
        span = positions[-1] - positions[0] + 1
        return min(1.0, len(positions) / span)

    def _title_match(self, terms: list[str], section: str) -> float:
        if not section or not terms:
            return 0.0
        section_tokens = set(tokenize(section))
        hits = sum(1 for t in set(terms) if t in section_tokens)
        return hits / len(set(terms))

    def rerank(
        self, query: str, candidates: list[FusedChunk], top_k: int = 5
    ) -> list[RerankedChunk]:
        """Score and reorder candidates."""
        terms = content_terms(query)
        reranked: list[RerankedChunk] = []

        for original_rank, candidate in enumerate(candidates, start=1):
            tokens = tokenize(candidate.chunk.text)
            coverage = self._coverage(terms, tokens)
            proximity = self._proximity(terms, tokens)
            title = self._title_match(terms, candidate.chunk.section)

            score = (
                self.coverage_weight * coverage
                + self.proximity_weight * proximity
                + self.title_weight * title
            )
            reranked.append(
                RerankedChunk(
                    fused=candidate,
                    score=score,
                    original_rank=original_rank,
                    signals={
                        "coverage": round(coverage, 4),
                        "proximity": round(proximity, 4),
                        "title_match": round(title, 4),
                    },
                )
            )

        # Fused score breaks ties, so reranking never discards retrieval's
        # opinion where its own signals are indifferent.
        reranked.sort(key=lambda r: (-r.score, -r.fused.rrf_score, r.chunk.chunk_id))
        for rank, item in enumerate(reranked[:top_k], start=1):
            item.rank = rank
        return reranked[:top_k]


class PassthroughReranker:
    """Keeps retrieval order. Used to measure whether reranking helps at all."""

    name = "passthrough"
    trained = False

    def rerank(
        self, query: str, candidates: list[FusedChunk], top_k: int = 5
    ) -> list[RerankedChunk]:
        out = [
            RerankedChunk(fused=c, score=c.rrf_score, original_rank=i, rank=i)
            for i, c in enumerate(candidates[:top_k], start=1)
        ]
        return out

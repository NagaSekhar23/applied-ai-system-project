"""Retrieval metrics: Recall@K, Precision@K, MRR.

Each is defined here once, with what it measures and what it cannot.

**Recall@K** = |relevant ∩ retrieved@K| / |relevant|
    Did the evidence come back at all? This is the metric that matters most
    for RAG, because generation can only use what retrieval returned.
    It says nothing about ordering.

**Precision@K** = |relevant ∩ retrieved@K| / K
    How much of what came back was useful. Low precision means the generator
    reads past noise, which costs tokens and invites distraction.
    With K larger than the number of relevant chunks, perfect precision is
    arithmetically impossible — so precision is read alongside K, never alone.

**MRR** = mean over queries of 1 / (rank of the first relevant result)
    Is the best evidence near the top? Matters because only top-k is passed
    to the generator and earlier passages carry more weight.
    It ignores every relevant result after the first.

**Relevance labels in this project are document-level.** A chunk counts as
relevant if it came from a document the case marks relevant. That is coarser
than passage-level ground truth: a chunk from the right document about the
wrong topic still counts. The numbers are therefore an upper bound on true
passage relevance, and are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(relevant_set & set(retrieved[:k]))
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top k that are relevant.

    Divides by the number actually returned, not by k, so a query that
    returned 2 results is not penalised for k=5.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(relevant) & set(top)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant result. 0.0 if none is relevant."""
    relevant_set = set(relevant)
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


@dataclass(frozen=True)
class RetrievalScores:
    """Per-query retrieval scores."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    k: int
    retrieved_count: int
    relevant_count: int

    def to_dict(self) -> dict:
        return {
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "k": self.k,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
        }


def score_retrieval(
    retrieved: Sequence[str], relevant: Sequence[str], k: int = 4
) -> RetrievalScores:
    """Compute all three retrieval metrics for one query."""
    return RetrievalScores(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        k=k,
        retrieved_count=len(retrieved),
        relevant_count=len(set(relevant)),
    )


def mean_reciprocal_rank(all_ranks: Sequence[float]) -> float:
    """Mean of per-query reciprocal ranks."""
    return sum(all_ranks) / len(all_ranks) if all_ranks else 0.0

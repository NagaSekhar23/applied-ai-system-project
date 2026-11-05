"""Hybrid retrieval with Reciprocal Rank Fusion.

## Why fuse on rank rather than score

BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded
in [0, 1]. Adding or averaging them compares incomparable units, and whichever
scale happens to be larger silently dominates. Normalising first helps but is
sensitive to outliers in a small result set.

Reciprocal Rank Fusion sidesteps this by discarding the magnitudes and using
only the ordering each retriever produced:

    RRF(d) = Σ over retrievers  1 / (k + rank_r(d))

with `k = 60`, the value from the original paper. Properties that matter here:

* **Scale-free.** Only ranks are compared, so no normalisation is needed.
* **Agreement wins.** A chunk ranked by both retrievers accumulates two terms
  and beats a chunk ranked first by one and missed by the other.
* **Diminishing returns.** The gap between rank 1 and 2 is larger than between
  9 and 10, which matches how much rank position actually means.

`k` damps the top ranks: a larger `k` flattens the curve and gives more weight
to broad agreement, a smaller `k` sharpens the advantage of being first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.embeddings import ScoredChunk

DEFAULT_K = 60


@dataclass
class FusedChunk:
    """A chunk with its fused score and where it came from."""

    scored: ScoredChunk
    rrf_score: float
    contributions: dict[str, int] = field(default_factory=dict)  # method -> rank

    @property
    def chunk(self):
        return self.scored.chunk

    @property
    def methods(self) -> list[str]:
        return sorted(self.contributions)

    @property
    def found_by_both(self) -> bool:
        return len(self.contributions) > 1

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source_path": self.chunk.source_path,
            "section": self.chunk.section,
            "rrf_score": round(self.rrf_score, 6),
            "contributions": self.contributions,
            "found_by_both": self.found_by_both,
        }


def reciprocal_rank_fusion(
    result_sets: dict[str, list[ScoredChunk]],
    k: int = DEFAULT_K,
    top_k: int = 5,
) -> list[FusedChunk]:
    """Fuse ranked result sets.

    Args:
        result_sets: method name -> that method's ranked results.
        k: RRF damping constant.
        top_k: how many fused results to return.
    """
    fused: dict[str, FusedChunk] = {}

    for method, results in sorted(result_sets.items()):
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            contribution = 1.0 / (k + rank)

            if chunk_id in fused:
                fused[chunk_id].rrf_score += contribution
                fused[chunk_id].contributions[method] = rank
            else:
                fused[chunk_id] = FusedChunk(
                    scored=ScoredChunk(scored.chunk, scored.score, "hybrid"),
                    rrf_score=contribution,
                    contributions={method: rank},
                )

    ordered = sorted(fused.values(), key=lambda f: (-f.rrf_score, f.chunk.chunk_id))
    for rank, item in enumerate(ordered[:top_k], start=1):
        item.scored.rank = rank
    return ordered[:top_k]


class HybridRetriever:
    """Runs both retrievers and fuses their rankings."""

    def __init__(self, vector_index, keyword_index, k: int = DEFAULT_K):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.k = k

    def search(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[FusedChunk]:
        """Retrieve and fuse.

        Each retriever returns more candidates than the final `top_k`, so
        fusion has room to promote a chunk that both ranked moderately well.
        """
        candidates = candidate_k or max(top_k * 3, 10)
        return reciprocal_rank_fusion(
            {
                "vector": self.vector_index.search(query, candidates),
                "keyword": self.keyword_index.search(query, candidates),
            },
            k=self.k,
            top_k=top_k,
        )

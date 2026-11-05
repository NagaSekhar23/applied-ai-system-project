"""Retrieval-augmented generation over this project's own documentation.

Pipeline: ingest -> chunk -> index (vector + keyword) -> fuse (RRF) ->
rerank -> generate with validated citations, or abstain.

Each stage lives in its own module with its own tests.
"""

from retrieval.answering import (
    ABSTAIN_MESSAGE,
    AnswerGenerator,
    GroundedAnswer,
    RagAnswer,
)
from retrieval.chunking import Chunk, chunk_document, chunk_documents, split_sections
from retrieval.citations import (
    Citation,
    CitationCheck,
    extract_markers,
    format_evidence,
    validate_citations,
)
from retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    ScoredChunk,
    VectorIndex,
    cosine,
    tokenize,
)
from retrieval.hybrid import FusedChunk, HybridRetriever, reciprocal_rank_fusion
from retrieval.ingestion import (
    DEFAULT_SOURCES,
    Document,
    IngestionError,
    ingest,
    load_document,
)
from retrieval.keyword_search import KeywordIndex
from retrieval.pipeline import IndexStats, RagPipeline, RetrievalResult
from retrieval.reranking import (
    HeuristicReranker,
    PassthroughReranker,
    Reranker,
    RerankedChunk,
)

__all__ = [
    "ABSTAIN_MESSAGE",
    "AnswerGenerator",
    "Chunk",
    "Citation",
    "CitationCheck",
    "DEFAULT_SOURCES",
    "Document",
    "Embedder",
    "FusedChunk",
    "GroundedAnswer",
    "HashingEmbedder",
    "HeuristicReranker",
    "HybridRetriever",
    "IndexStats",
    "IngestionError",
    "KeywordIndex",
    "PassthroughReranker",
    "RagAnswer",
    "RagPipeline",
    "RerankedChunk",
    "Reranker",
    "RetrievalResult",
    "ScoredChunk",
    "VectorIndex",
    "chunk_document",
    "chunk_documents",
    "cosine",
    "extract_markers",
    "format_evidence",
    "ingest",
    "load_document",
    "reciprocal_rank_fusion",
    "split_sections",
    "tokenize",
    "validate_citations",
]

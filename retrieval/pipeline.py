"""The RAG pipeline: ingest → chunk → index → retrieve → rerank → answer.

Each stage is a separate module with its own tests. This file only wires them
together and times them, so the composition is readable in one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from retrieval.answering import AnswerGenerator, RagAnswer
from retrieval.chunking import Chunk, chunk_documents
from retrieval.embeddings import Embedder, HashingEmbedder, VectorIndex
from retrieval.hybrid import FusedChunk, HybridRetriever
from retrieval.ingestion import DEFAULT_SOURCES, Document, ingest
from retrieval.keyword_search import KeywordIndex
from retrieval.reranking import HeuristicReranker, Reranker, RerankedChunk


@dataclass
class IndexStats:
    """What ingestion produced. Shown in the UI so the corpus is inspectable."""

    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    ingestion_ms: float = 0.0
    embedder: str = ""
    embedder_is_semantic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "sources": self.sources,
            "ingestion_ms": round(self.ingestion_ms, 2),
            "embedder": self.embedder,
            "embedder_is_semantic": self.embedder_is_semantic,
        }


@dataclass
class RetrievalResult:
    """Everything one retrieval produced, with per-stage timings."""

    query: str
    candidates: list[FusedChunk]
    evidence: list[RerankedChunk]
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidate_count": len(self.candidates),
            "evidence_count": len(self.evidence),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "evidence": [e.to_dict() for e in self.evidence],
        }


class RagPipeline:
    """Ingests a corpus once, then answers questions against it."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.reranker = reranker or HeuristicReranker()
        self.generator = generator or AnswerGenerator()

        self.vector_index = VectorIndex(self.embedder)
        self.keyword_index = KeywordIndex()
        self.hybrid = HybridRetriever(self.vector_index, self.keyword_index)

        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.stats = IndexStats(
            embedder=self.embedder.name,
            embedder_is_semantic=getattr(self.embedder, "semantic", False),
        )

    def build(
        self,
        sources: tuple[str, ...] = DEFAULT_SOURCES,
        root: str = ".",
        max_chars: int = 1200,
    ) -> IndexStats:
        """Ingest, chunk and index the corpus."""
        started = time.perf_counter()

        documents, skipped = ingest(sources, root)
        chunks = chunk_documents(documents, max_chars=max_chars)

        self.vector_index.add(chunks)
        self.keyword_index.add(chunks)

        self.documents = documents
        self.chunks = chunks
        self.stats = IndexStats(
            documents=len(documents),
            chunks=len(chunks),
            skipped=skipped,
            sources=[d.source_path for d in documents],
            ingestion_ms=(time.perf_counter() - started) * 1000,
            embedder=self.embedder.name,
            embedder_is_semantic=getattr(self.embedder, "semantic", False),
        )
        return self.stats

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Hybrid retrieval followed by reranking."""
        started = time.perf_counter()
        candidates = self.hybrid.search(query, top_k=max(top_k * 2, 8))
        retrieval_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        evidence = self.reranker.rerank(query, candidates, top_k=top_k)
        rerank_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(
            query=query,
            candidates=candidates,
            evidence=evidence,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
        )

    def ask(self, question: str, top_k: int = 4) -> tuple[RagAnswer, RetrievalResult]:
        """Retrieve, then generate a grounded answer or abstain."""
        retrieval = self.retrieve(question, top_k=top_k)
        answer = self.generator.answer(question, retrieval.evidence)
        answer.retrieval_ms = retrieval.total_ms
        return answer, retrieval

    @property
    def is_built(self) -> bool:
        return bool(self.chunks)

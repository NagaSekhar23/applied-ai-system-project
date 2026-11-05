"""Executors for the RAG layer.

Retrieval metrics and generation metrics are deliberately separate targets,
reported under different categories, because they answer different questions
and can fail independently.

The pipeline is built once and cached: ingesting the corpus for every case
would dominate the run time and measure nothing useful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.cases import EvalCase
from evaluation.executors import Executor, _capture
from evaluation.rag_metrics import score_retrieval
from providers import MockProvider, ReliableLLMClient, ScriptedProvider
from retrieval import AnswerGenerator, PassthroughReranker, RagPipeline

_PIPELINE_CACHE: dict[tuple[str, str], RagPipeline] = {}

# A frozen corpus, used for any case that asserts on *rank*.
#
# Retrieval rank is a property of the whole corpus: adding a document that
# mentions the same terms shifts every ranking. The live corpus is this
# project's own documentation, which changes on most commits, so rank
# assertions against it break for reasons unrelated to retrieval quality —
# which happened twice while this project was being built.
#
# Ground truth needs a stable substrate. Rank-sensitive cases therefore run
# against these fixtures, while cases that assert only "the right document
# was found" run against the live corpus, where that property is robust.
FIXTURE_SOURCES = (
    "search.md",
    "caching.md",
    "deployment.md",
    "testing.md",
)
FIXTURE_ROOT = str(Path(__file__).parent / "fixtures" / "corpus")


def _no_sleep(_seconds: float) -> None:
    pass


def get_pipeline(reranker: str = "heuristic", corpus: str = "live") -> RagPipeline:
    """Build and cache a pipeline. Ingestion is expensive and deterministic.

    Args:
        reranker: ``heuristic`` or ``passthrough``.
        corpus: ``live`` for the project's own docs, ``fixture`` for the
            frozen evaluation corpus.
    """
    key = (reranker, corpus)
    if key not in _PIPELINE_CACHE:
        client = ReliableLLMClient(MockProvider(), sleep=_no_sleep)
        pipeline = RagPipeline(
            reranker=PassthroughReranker() if reranker == "passthrough" else None,
            generator=AnswerGenerator(client),
        )
        if corpus == "fixture":
            pipeline.build(sources=FIXTURE_SOURCES, root=FIXTURE_ROOT)
        else:
            pipeline.build()
        _PIPELINE_CACHE[key] = pipeline
    return _PIPELINE_CACHE[key]


def execute_rag_retrieval(case: EvalCase) -> dict[str, Any]:
    """Retrieve for a query and score the ranking against labelled sources."""

    def run() -> dict[str, Any]:
        k = case.input.get("k", 4)
        pipeline = get_pipeline(
            case.input.get("reranker", "heuristic"),
            case.input.get("corpus", "live"),
        )
        result = pipeline.retrieve(case.input["query"], top_k=k)

        retrieved_sources = [item.chunk.source_path for item in result.evidence]
        relevant_sources = case.input.get("relevant_sources", [])

        # Deduplicate while preserving rank order: labels are document-level,
        # so several chunks from one document are one hit, not several.
        seen: list[str] = []
        for source in retrieved_sources:
            if source not in seen:
                seen.append(source)

        scores = score_retrieval(seen, relevant_sources, k=len(seen) or k)
        return {
            "retrieved_sources": seen,
            "top_source": seen[0] if seen else None,
            "evidence_count": len(result.evidence),
            "recall_at_k": scores.recall_at_k,
            "precision_at_k": scores.precision_at_k,
            "reciprocal_rank": scores.reciprocal_rank,
            "found_any_relevant": scores.recall_at_k > 0,
            "top_is_relevant": bool(seen) and seen[0] in set(relevant_sources),
            "fused_by_both": sum(1 for c in result.candidates if c.found_by_both),
        }

    return _capture(run)


def execute_rag_answer(case: EvalCase) -> dict[str, Any]:
    """Ask a question and report grounding, citations and abstention."""

    def run() -> dict[str, Any]:
        pipeline = get_pipeline()

        if case.input.get("provider") == "scripted":
            pipeline = RagPipeline(
                generator=AnswerGenerator(
                    ReliableLLMClient(
                        ScriptedProvider(list(case.input.get("script", []))),
                        max_retries=case.input.get("max_retries", 0),
                        sleep=_no_sleep,
                    )
                )
            )
            pipeline.build()
        elif case.input.get("provider") == "none":
            pipeline = RagPipeline(generator=AnswerGenerator(client=None))
            pipeline.build()

        answer, retrieval = pipeline.ask(
            case.input["question"], top_k=case.input.get("k", 4)
        )
        check = answer.citation_check

        return {
            "abstained": answer.abstained,
            "grounded": answer.grounded,
            "citation_count": len(answer.citations),
            "citations_valid": check.valid if check else False,
            "invalid_citation_count": len(check.invalid_markers) if check else 0,
            "has_reason": answer.reason is not None,
            "answer_is_non_empty": bool(answer.answer.strip()),
            "cited_sources": sorted({c.source_path for c in answer.citations}),
            "evidence_count": len(answer.evidence),
            "source": answer.source,
        }

    return _capture(run)


def execute_citation_validation(case: EvalCase) -> dict[str, Any]:
    """Validate a supplied answer string against a synthetic evidence list."""

    def run() -> dict[str, Any]:
        from retrieval.chunking import Chunk
        from retrieval.citations import validate_citations
        from retrieval.hybrid import FusedChunk
        from retrieval.embeddings import ScoredChunk
        from retrieval.reranking import RerankedChunk

        evidence: list[RerankedChunk] = []
        for index in range(case.input.get("evidence_count", 2)):
            chunk = Chunk(
                chunk_id=f"doc::{index:03d}",
                doc_id="doc",
                source_path=f"docs/file{index}.md",
                doc_title="Doc",
                section=f"Section {index}",
                text="evidence text",
                position=index,
                content_hash="hash",
                ingested_at=0.0,
            )
            evidence.append(
                RerankedChunk(
                    fused=FusedChunk(ScoredChunk(chunk, 1.0, "hybrid"), 1.0),
                    score=1.0,
                    original_rank=index + 1,
                    rank=index + 1,
                )
            )

        check = validate_citations(case.input["answer"], evidence)
        return {
            "valid": check.valid,
            "citation_count": len(check.citations),
            "invalid_markers": check.invalid_markers,
            "uncited": check.uncited,
        }

    return _capture(run)


def execute_ingestion(case: EvalCase) -> dict[str, Any]:
    """Check ingestion and chunk metadata."""

    def run() -> dict[str, Any]:
        pipeline = get_pipeline()
        chunks = pipeline.chunks
        return {
            "document_count": pipeline.stats.documents,
            "chunk_count_positive": len(chunks) > 0,
            "all_have_source": all(c.source_path for c in chunks),
            "all_have_chunk_id": all(c.chunk_id for c in chunks),
            "all_have_hash": all(c.content_hash for c in chunks),
            "all_have_timestamp": all(c.ingested_at > 0 for c in chunks),
            "unique_chunk_ids": len({c.chunk_id for c in chunks}) == len(chunks),
            "no_blocked_sources": all(
                ".env" not in c.source_path and ".venv" not in c.source_path
                for c in chunks
            ),
            "embedder_is_semantic": pipeline.stats.embedder_is_semantic,
        }

    return _capture(run)


RAG_EXECUTORS: dict[str, Executor] = {
    "rag_retrieval": execute_rag_retrieval,
    "rag_answer": execute_rag_answer,
    "citation_validation": execute_citation_validation,
    "rag_ingestion": execute_ingestion,
}

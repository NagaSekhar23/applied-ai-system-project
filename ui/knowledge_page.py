"""The Knowledge Assistant page.

Shows the answer, its verified citations, and the evidence those citations
point at. When the system abstains it says why, rather than producing
something plausible.
"""

from __future__ import annotations

import streamlit as st

from config import Settings
from observability import Component, Status, get_tracker
from observability.events import Event
from providers import build_client
from retrieval import AnswerGenerator, RagPipeline

EXAMPLES = [
    "What does reciprocal rank fusion do?",
    "How is the agent loop bounded?",
    "How are citations validated?",
    "What chunking strategy is used?",
]


def get_pipeline(settings: Settings) -> RagPipeline:
    """Build and index once per session, keyed by configuration."""
    key = (settings.provider, settings.model)
    if st.session_state.get("rag_key") != key:
        with st.spinner("Indexing project documentation…"):
            pipeline = RagPipeline(
                generator=AnswerGenerator(
                    build_client(settings),
                    max_tokens=settings.max_tokens,
                    effort=settings.effort,
                )
            )
            stats = pipeline.build()
        st.session_state.rag_pipeline = pipeline
        st.session_state.rag_key = key

        get_tracker().record(
            Event(
                operation="retrieval.ingest",
                component=Component.RETRIEVAL,
                duration_ms=stats.ingestion_ms,
                attributes={
                    "documents": stats.documents,
                    "chunks": stats.chunks,
                    "embedder": stats.embedder,
                },
            )
        )
    return st.session_state.rag_pipeline


def render_corpus(pipeline: RagPipeline) -> None:
    stats = pipeline.stats
    with st.expander(f"Knowledge base: {stats.documents} documents, {stats.chunks} chunks"):
        for source in stats.sources:
            st.markdown(f"- `{source}`")
        if stats.skipped:
            st.caption("Skipped:")
            for item in stats.skipped:
                st.caption(f"  • {item}")
        if not stats.embedder_is_semantic:
            st.info(
                f"Embedder `{stats.embedder}` is a **lexical** hashing vectoriser, not a "
                "trained semantic model. It matches on word overlap, so a paraphrase "
                "with no shared words will miss. The interface is replaceable."
            )


def render_answer(answer, retrieval) -> None:
    if answer.abstained:
        st.warning(f"**Not enough evidence.** {answer.answer}")
        st.caption(f"Reason: {answer.reason}")
    else:
        st.markdown(answer.answer)
        st.caption(
            f"Grounded in {len(answer.citations)} verified citation(s) · "
            f"prompt `{answer.prompt_id}`"
        )

    if answer.citations:
        st.markdown("##### Citations")
        for citation in answer.citations:
            with st.expander(f"[{citation.marker}] {citation.label}"):
                st.caption(f"chunk `{citation.chunk_id}`")
                st.markdown(f"> {citation.quote}")

    with st.expander(f"Retrieved evidence ({len(retrieval.evidence)} passages)"):
        st.caption(
            f"Retrieval {retrieval.retrieval_ms:.1f} ms · "
            f"rerank {retrieval.rerank_ms:.1f} ms"
        )
        for item in retrieval.evidence:
            both = "both retrievers" if item.fused.found_by_both else ", ".join(item.fused.methods)
            st.markdown(
                f"**[{item.rank}] {item.chunk.citation_label}** — "
                f"score {item.score:.3f}, found by {both}"
            )
            if item.signals:
                st.caption(f"signals: {item.signals}")
            st.text(item.chunk.text[:400] + ("…" if len(item.chunk.text) > 400 else ""))
            st.divider()


def render(settings: Settings) -> None:
    """Render the Knowledge Assistant page."""
    st.subheader("📚 Knowledge Assistant")
    st.caption(
        "Answers questions about this project from its own documentation. Every "
        "claim is cited, every citation is checked against what was actually "
        "retrieved, and the assistant abstains when the evidence is thin."
    )

    pipeline = get_pipeline(settings)
    render_corpus(pipeline)

    if settings.is_mock:
        st.info(
            "**Mock mode.** Answers are composed deterministically from the top "
            "retrieved passage, with a real citation. Retrieval, reranking and "
            "citation validation are genuine; the phrasing is not model-generated."
        )

    st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))

    question = st.chat_input("Ask about this project…")
    if question:
        tracker = get_tracker()
        with st.spinner("Searching the documentation…"), tracker.request("rag.question"):
            answer, retrieval = pipeline.ask(question)
            tracker.record(
                Event(
                    operation="retrieval.answer",
                    component=Component.RETRIEVAL,
                    status=Status.DEGRADED if answer.abstained else Status.OK,
                    duration_ms=retrieval.total_ms,
                    attributes={
                        "abstained": answer.abstained,
                        "grounded": answer.grounded,
                        "citations": len(answer.citations),
                        "evidence": len(answer.evidence),
                        "retrieval_ms": round(retrieval.retrieval_ms, 2),
                        "rerank_ms": round(retrieval.rerank_ms, 2),
                    },
                )
            )
            if answer.stats is not None:
                tracker.record_llm_call("retrieval.generate", answer.stats)

        st.session_state.setdefault("rag_history", []).append((question, answer, retrieval))
        st.rerun()

    for question, answer, retrieval in reversed(st.session_state.get("rag_history", [])):
        st.divider()
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_answer(answer, retrieval)

    if st.session_state.get("rag_history") and st.button("Clear questions"):
        st.session_state.rag_history = []
        st.rerun()

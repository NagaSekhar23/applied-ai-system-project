"""Phase 7 tests: ingestion, chunking, retrieval, fusion, reranking,
citations, abstention and the retrieval metrics."""

import json

import pytest

from evaluation.rag_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)
from providers import MockProvider, ReliableLLMClient, ScriptedProvider
from providers.base import LLMTimeoutError
from retrieval import (
    ABSTAIN_MESSAGE,
    AnswerGenerator,
    Chunk,
    HashingEmbedder,
    HeuristicReranker,
    HybridRetriever,
    IngestionError,
    KeywordIndex,
    PassthroughReranker,
    RagPipeline,
    ScoredChunk,
    VectorIndex,
    chunk_document,
    chunk_documents,
    cosine,
    extract_markers,
    format_evidence,
    ingest,
    load_document,
    reciprocal_rank_fusion,
    split_sections,
    tokenize,
    validate_citations,
)
from retrieval.hybrid import FusedChunk
from retrieval.ingestion import Document, content_hash, is_blocked, make_doc_id
from retrieval.reranking import RerankedChunk


def no_sleep(_seconds):
    pass


def make_chunk(chunk_id: str, text: str, source: str = "docs/a.md", section: str = "S") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        source_path=source,
        doc_title="Doc",
        section=section,
        text=text,
        position=0,
        content_hash="h",
        ingested_at=1.0,
    )


def make_evidence(count: int = 3) -> list[RerankedChunk]:
    out = []
    for index in range(count):
        chunk = make_chunk(f"doc::{index:03d}", "evidence text", f"docs/file{index}.md")
        out.append(
            RerankedChunk(
                fused=FusedChunk(ScoredChunk(chunk, 1.0, "hybrid"), 1.0),
                score=1.0,
                original_rank=index + 1,
                rank=index + 1,
            )
        )
    return out


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    """One indexed pipeline for the whole module; ingestion is deterministic."""
    client = ReliableLLMClient(MockProvider(), sleep=no_sleep)
    instance = RagPipeline(generator=AnswerGenerator(client))
    instance.build()
    return instance


# --- Ingestion ---------------------------------------------------------------


def test_ingests_the_documentation_corpus(pipeline):
    assert pipeline.stats.documents >= 6
    assert pipeline.stats.chunks > 20
    assert "README.md" in pipeline.stats.sources


def test_document_ids_are_stable_for_a_path():
    assert make_doc_id("docs/architecture.md") == make_doc_id("docs/architecture.md")
    assert make_doc_id("docs/a.md") != make_doc_id("docs/b.md")


def test_content_hash_detects_change():
    assert content_hash("a") == content_hash("a")
    assert content_hash("a") != content_hash("b")


@pytest.mark.parametrize(
    "path", [".env", ".env.local", "config/secrets.yaml", ".venv/lib/x.py", "my_credentials.txt"]
)
def test_sensitive_paths_are_blocked(path):
    assert is_blocked(path)
    with pytest.raises(IngestionError, match="blocked"):
        load_document(path)


def test_missing_file_raises_or_is_skipped(tmp_path):
    with pytest.raises(IngestionError, match="Cannot read"):
        load_document("does-not-exist.md", root=tmp_path)

    documents, skipped = ingest(("does-not-exist.md",), root=tmp_path)
    assert documents == []
    assert len(skipped) == 1


def test_empty_file_is_rejected(tmp_path):
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    with pytest.raises(IngestionError, match="empty"):
        load_document("empty.md", root=tmp_path)


def test_duplicate_content_is_ingested_once(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nSame body.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nSame body.", encoding="utf-8")
    documents, skipped = ingest(("a.md", "b.md"), root=tmp_path)
    assert len(documents) == 1
    assert any("duplicate" in s for s in skipped)


def test_title_comes_from_the_first_heading(tmp_path):
    (tmp_path / "d.md").write_text("# The Title\n\nBody.", encoding="utf-8")
    assert load_document("d.md", root=tmp_path).title == "The Title"


def test_ingested_documents_carry_metadata(pipeline):
    document = pipeline.documents[0]
    assert document.content_hash
    assert document.ingested_at > 0
    assert document.source_path


# --- Chunking ----------------------------------------------------------------


def test_sections_split_on_headings():
    sections = split_sections("# A\n\nalpha\n\n## B\n\nbeta")
    assert [s.heading for s in sections] == ["A", "B"]


def test_chunks_have_stable_unique_ids(pipeline):
    ids = [c.chunk_id for c in pipeline.chunks]
    assert len(ids) == len(set(ids))
    assert all("::" in cid for cid in ids)


def test_chunk_ids_are_reproducible():
    document = Document(
        doc_id="d", source_path="d.md", title="D",
        text="# A\n\nalpha\n\n# B\n\nbeta", content_hash="h", ingested_at=1.0,
    )
    first = [c.chunk_id for c in chunk_document(document)]
    second = [c.chunk_id for c in chunk_document(document)]
    assert first == second


def test_chunks_preserve_source_metadata(pipeline):
    for chunk in pipeline.chunks:
        assert chunk.source_path
        assert chunk.doc_id
        assert chunk.content_hash
        assert chunk.ingested_at > 0


def test_long_sections_are_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(8))
    document = Document(
        doc_id="d", source_path="d.md", title="D",
        text=f"# Big\n\n{body}", content_hash="h", ingested_at=1.0,
    )
    chunks = chunk_document(document, max_chars=600, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 900 for c in chunks)


def test_chunk_citation_label_includes_section():
    assert "docs/a.md › S" == make_chunk("c", "t").citation_label


def test_chunk_documents_handles_an_empty_list():
    assert chunk_documents([]) == []


# --- Embeddings and vector search --------------------------------------------


def test_tokenizer_splits_snake_case_and_keeps_the_whole_token():
    tokens = tokenize("p95_latency_ms")
    assert "p95_latency_ms" in tokens
    assert "p95" in tokens and "latency" in tokens


def test_embedding_is_deterministic():
    embedder = HashingEmbedder()
    assert embedder.embed("hello world") == embedder.embed("hello world")


def test_embedding_is_normalised():
    vector = HashingEmbedder().embed("some text here")
    assert sum(v * v for v in vector.values()) == pytest.approx(1.0, abs=1e-9)


def test_embedding_of_empty_text_is_empty():
    assert HashingEmbedder().embed("") == {}


def test_embedder_declares_it_is_not_semantic():
    assert HashingEmbedder().semantic is False


def test_cosine_of_identical_vectors_is_one():
    vector = HashingEmbedder().embed("identical text")
    assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-9)


def test_cosine_of_disjoint_vectors_is_zero():
    embedder = HashingEmbedder()
    assert cosine(embedder.embed("alpha"), embedder.embed("")) == 0.0


def test_vector_search_ranks_the_matching_chunk_first():
    index = VectorIndex()
    index.add([
        make_chunk("a", "reciprocal rank fusion combines rankings"),
        make_chunk("b", "the game has three difficulty levels"),
    ])
    results = index.search("reciprocal rank fusion", top_k=2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].rank == 1


def test_vector_search_on_an_empty_index():
    assert VectorIndex().search("anything") == []


# --- Keyword search ----------------------------------------------------------


def test_bm25_ranks_the_exact_term_first():
    index = KeywordIndex()
    index.add([
        make_chunk("a", "the GAME_DEBUG variable shows the panel"),
        make_chunk("b", "difficulty settings and attempt limits"),
    ])
    assert index.search("GAME_DEBUG", top_k=2)[0].chunk.chunk_id == "a"


def test_bm25_returns_nothing_for_an_unseen_term():
    index = KeywordIndex()
    index.add([make_chunk("a", "alpha beta")])
    assert index.search("kubernetes") == []


def test_bm25_idf_is_never_negative():
    index = KeywordIndex()
    index.add([make_chunk(str(i), "common word") for i in range(5)])
    assert index.idf("common") >= 0.0


def test_bm25_on_an_empty_index():
    assert KeywordIndex().search("anything") == []


# --- Hybrid fusion -----------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    shared = make_chunk("shared", "x")
    only_vector = make_chunk("vector-only", "y")
    only_keyword = make_chunk("keyword-only", "z")

    fused = reciprocal_rank_fusion(
        {
            "vector": [ScoredChunk(only_vector, 1.0, "vector"), ScoredChunk(shared, 0.5, "vector")],
            "keyword": [ScoredChunk(only_keyword, 9.0, "keyword"), ScoredChunk(shared, 4.0, "keyword")],
        },
        top_k=3,
    )
    # "shared" is second in both lists but beats items ranked first in only one.
    assert fused[0].chunk.chunk_id == "shared"
    assert fused[0].found_by_both


def test_rrf_is_scale_free():
    """A huge BM25 score must not dominate a bounded cosine score."""
    a, b = make_chunk("a", "x"), make_chunk("b", "y")
    fused = reciprocal_rank_fusion(
        {
            "vector": [ScoredChunk(a, 0.01, "vector")],
            "keyword": [ScoredChunk(b, 9999.0, "keyword")],
        },
        top_k=2,
    )
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


def test_rrf_handles_an_empty_result_set():
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []


def test_hybrid_retriever_combines_both_indexes():
    chunks = [
        make_chunk("a", "reciprocal rank fusion combines ranked lists"),
        make_chunk("b", "GAME_DEBUG shows the developer panel"),
    ]
    vector, keyword = VectorIndex(), KeywordIndex()
    vector.add(chunks)
    keyword.add(chunks)
    results = HybridRetriever(vector, keyword).search("reciprocal rank fusion", top_k=2)
    assert results[0].chunk.chunk_id == "a"


# --- Reranking ---------------------------------------------------------------


def test_reranker_prefers_full_term_coverage():
    partial = FusedChunk(ScoredChunk(make_chunk("partial", "citation only"), 1.0, "h"), 0.02)
    full = FusedChunk(
        ScoredChunk(make_chunk("full", "citation validation checks every marker"), 1.0, "h"), 0.01
    )
    ranked = HeuristicReranker().rerank("citation validation", [partial, full], top_k=2)
    assert ranked[0].chunk.chunk_id == "full"


def test_reranker_reports_its_signals():
    candidate = FusedChunk(ScoredChunk(make_chunk("a", "citation validation"), 1.0, "h"), 0.01)
    ranked = HeuristicReranker().rerank("citation validation", [candidate], top_k=1)
    assert set(ranked[0].signals) == {"coverage", "proximity", "title_match"}


def test_reranker_declares_it_is_not_trained():
    assert HeuristicReranker().trained is False


def test_reranker_tracks_movement():
    a = FusedChunk(ScoredChunk(make_chunk("a", "unrelated words"), 1.0, "h"), 0.02)
    b = FusedChunk(ScoredChunk(make_chunk("b", "citation validation"), 1.0, "h"), 0.01)
    ranked = HeuristicReranker().rerank("citation validation", [a, b], top_k=2)
    assert ranked[0].moved == 1  # promoted from rank 2 to rank 1


def test_passthrough_reranker_preserves_order():
    candidates = [
        FusedChunk(ScoredChunk(make_chunk(c, "text"), 1.0, "h"), 0.01) for c in ("a", "b", "c")
    ]
    ranked = PassthroughReranker().rerank("anything", candidates, top_k=3)
    assert [r.chunk.chunk_id for r in ranked] == ["a", "b", "c"]


# --- Citations ---------------------------------------------------------------


def test_extract_markers_deduplicates_and_preserves_order():
    assert extract_markers("See [2] and [1], also [2].") == [2, 1]


def test_valid_citations_resolve_to_evidence():
    check = validate_citations("Bounded [1] and timed [2].", make_evidence(3))
    assert check.valid
    assert len(check.citations) == 2
    assert check.citations[0].source_path == "docs/file0.md"


def test_out_of_range_citation_is_invalid():
    check = validate_citations("As shown [9].", make_evidence(3))
    assert not check.valid
    assert check.invalid_markers == [9]


def test_answer_without_citations_is_flagged_as_uncited():
    check = validate_citations("No markers here.", make_evidence(3))
    assert check.uncited
    assert not check.valid


def test_one_bad_marker_invalidates_the_whole_answer():
    check = validate_citations("Good [1] and bad [9].", make_evidence(2))
    assert not check.valid
    assert len(check.citations) == 1


def test_format_evidence_numbers_from_one():
    rendered = format_evidence(make_evidence(2))
    assert rendered.startswith("[1] docs/file0.md")
    assert "[2] docs/file1.md" in rendered


def test_evidence_numbering_matches_validation():
    """format_evidence and validate_citations are two halves of one contract."""
    evidence = make_evidence(3)
    rendered = format_evidence(evidence)
    assert f"[2] {evidence[1].chunk.source_path}" in rendered
    check = validate_citations("See [2].", evidence)
    assert check.citations[0].chunk_id == evidence[1].chunk.chunk_id


# --- Answer generation and abstention ----------------------------------------


def test_answers_a_documented_question(pipeline):
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert not answer.abstained
    assert answer.grounded
    assert answer.citations


def test_abstains_on_an_off_topic_question(pipeline):
    answer, _ = pipeline.ask("What is the capital of France?")
    assert answer.abstained
    assert not answer.grounded
    assert answer.reason
    assert answer.answer == ABSTAIN_MESSAGE


def test_abstains_without_a_generator():
    pipeline = RagPipeline(generator=AnswerGenerator(client=None))
    pipeline.build()
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert answer.abstained
    assert "no LLM client" in answer.reason


def test_abstains_when_generation_fails():
    pipeline = RagPipeline(
        generator=AnswerGenerator(
            ReliableLLMClient(ScriptedProvider([LLMTimeoutError("down")] * 3), max_retries=1, sleep=no_sleep)
        )
    )
    pipeline.build()
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert answer.abstained
    assert "failed" in answer.reason


def test_rejects_a_fabricated_citation():
    pipeline = RagPipeline(
        generator=AnswerGenerator(
            ReliableLLMClient(
                ScriptedProvider(['{"answer": "It fuses [9].", "sufficient_evidence": true}']),
                max_retries=0, sleep=no_sleep,
            )
        )
    )
    pipeline.build()
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert answer.abstained
    assert "citation validation failed" in answer.reason


def test_rejects_an_uncited_answer():
    pipeline = RagPipeline(
        generator=AnswerGenerator(
            ReliableLLMClient(
                ScriptedProvider(['{"answer": "It just works.", "sufficient_evidence": true}']),
                max_retries=0, sleep=no_sleep,
            )
        )
    )
    pipeline.build()
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert answer.abstained


def test_honours_the_models_insufficiency_judgement():
    pipeline = RagPipeline(
        generator=AnswerGenerator(
            ReliableLLMClient(
                ScriptedProvider(['{"answer": "Cannot tell.", "sufficient_evidence": false}']),
                max_retries=0, sleep=no_sleep,
            )
        )
    )
    pipeline.build()
    answer, _ = pipeline.ask("What does reciprocal rank fusion do?")
    assert answer.abstained
    assert "insufficient" in answer.reason


def test_answer_serialises_to_json(pipeline):
    answer, _ = pipeline.ask("How are citations validated?")
    json.dumps(answer.to_dict())


def test_retrieval_records_stage_timings(pipeline):
    result = pipeline.retrieve("citation validation", top_k=3)
    assert result.retrieval_ms >= 0
    assert result.rerank_ms >= 0
    assert result.total_ms == result.retrieval_ms + result.rerank_ms


# --- Retrieval metrics -------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a", "c"], ["a", "c"], k=2) == 1.0
    assert recall_at_k([], ["a"], k=2) == 0.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(["a"], [], k=1) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b"], ["a"], k=2) == 0.5
    assert precision_at_k(["a"], ["a"], k=5) == 1.0  # divides by what was returned
    assert precision_at_k([], ["a"], k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5]) == 0.75
    assert mean_reciprocal_rank([]) == 0.0


def test_score_retrieval_bundles_all_three():
    scores = score_retrieval(["a", "b", "c"], ["b"], k=3)
    assert scores.recall_at_k == 1.0
    assert scores.precision_at_k == pytest.approx(1 / 3)
    assert scores.reciprocal_rank == 0.5
    json.dumps(scores.to_dict())

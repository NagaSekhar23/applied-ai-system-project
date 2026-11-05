"""AppTest coverage for the Knowledge Assistant page."""

import pytest
from streamlit.testing.v1 import AppTest

from observability import Tracker
from observability.tracker import set_tracker

APP = "app.py"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "GAME_DEBUG"]:
        monkeypatch.delenv(name, raising=False)
    set_tracker(Tracker())
    yield
    set_tracker(None)


def goto_knowledge(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Knowledge Assistant")
    return at.run()


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


def last(at: AppTest):
    return at.session_state["rag_history"][-1]


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [i.value for i in at.info]
    parts += [c.value for c in at.caption] + [w.value for w in at.warning]
    parts += [h.value for h in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_page_is_reachable_and_indexes_the_corpus():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    assert not at.exception
    assert at.session_state["rag_pipeline"].is_built


def test_corpus_contents_are_listed():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    text = all_text(at)
    assert "README.md" in text
    assert "docs/architecture.md" in text


def test_page_states_the_embedder_is_not_semantic():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    assert "lexical" in all_text(at).lower()


def test_documented_question_is_answered_with_citations():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    ask(at, "What does reciprocal rank fusion do?")

    _question, answer, _retrieval = last(at)
    assert not answer.abstained
    assert answer.grounded
    assert answer.citations


def test_off_topic_question_abstains_visibly():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    ask(at, "What is the capital of France?")

    _question, answer, _retrieval = last(at)
    assert answer.abstained
    assert "Not enough evidence" in all_text(at)


def test_citations_point_at_real_sources():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    ask(at, "How are citations validated?")

    _question, answer, _retrieval = last(at)
    corpus = set(at.session_state["rag_pipeline"].stats.sources)
    for citation in answer.citations:
        assert citation.source_path in corpus


def test_history_accumulates():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    ask(at, "What chunking strategy is used?")
    ask(at, "How is the agent loop bounded?")
    assert len(at.session_state["rag_history"]) == 2


def test_questions_are_recorded_for_observability():
    at = goto_knowledge(AppTest.from_file(APP, default_timeout=30).run())
    ask(at, "What does reciprocal rank fusion do?")

    at.sidebar.radio[0].set_value("Observability")
    at.run()
    assert "No activity recorded yet" not in all_text(at)

"""Tests for DoclingRAGAgent.run_with_trace and the run() wrapper equivalence."""

import pytest
from docling_core.types.doc.document import DoclingDocument

from docling_agent import RAGIteration, RAGResult, RAGTrace
from docling_agent.agent.rag import DoclingRAGAgent

from .test_utils import MockBackend


def _make_doc(name: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_title(text=f"Title of {name}", parent=doc.body)
    doc.add_text(label="text", text=f"Body content of {name}.", parent=doc.body)
    return doc


def _make_result(answer: str) -> RAGResult:
    return RAGResult(
        answer=answer,
        iterations=[
            RAGIteration(
                iteration=1,
                section_ref="#/texts/0",
                reason="picked first section",
                section_text_length=42,
                can_answer=True,
                response=answer,
            )
        ],
        converged=True,
    )


@pytest.fixture
def rag_agent() -> DoclingRAGAgent:
    return DoclingRAGAgent(backend=MockBackend(), tools=[])


def test_run_with_trace_single_doc_returns_typed_trace(monkeypatch, rag_agent):
    doc = _make_doc("doc_a")
    monkeypatch.setattr(
        DoclingRAGAgent,
        "_rag_loop",
        lambda self, *, query, doc: _make_result("answer for " + doc.name),
    )

    trace = rag_agent.run_with_trace("what is X?", sources=[doc])

    assert isinstance(trace, RAGTrace)
    assert trace.query == "what is X?"
    assert len(trace.per_document) == 1
    assert trace.per_document[0].answer == "answer for doc_a"
    assert trace.per_document[0].iterations  # non-empty
    assert trace.per_document[0].converged is True
    # Single-doc path: _merge_answers returns the lone answer unchanged
    assert trace.final_answer == "answer for doc_a"


def test_run_with_trace_multi_doc_preserves_order_and_merges(monkeypatch, rag_agent):
    doc_a = _make_doc("doc_a")
    doc_b = _make_doc("doc_b")

    monkeypatch.setattr(
        DoclingRAGAgent,
        "_rag_loop",
        lambda self, *, query, doc: _make_result("answer for " + doc.name),
    )
    monkeypatch.setattr(
        DoclingRAGAgent,
        "_merge_answers",
        lambda self, *, query, answers: "MERGED:" + "|".join(answers),
    )

    trace = rag_agent.run_with_trace("q", sources=[doc_a, doc_b])

    assert len(trace.per_document) == 2
    assert trace.per_document[0].answer == "answer for doc_a"
    assert trace.per_document[1].answer == "answer for doc_b"
    assert trace.final_answer == "MERGED:answer for doc_a|answer for doc_b"


def test_run_wraps_final_answer_from_trace(monkeypatch, rag_agent):
    doc = _make_doc("doc_a")
    monkeypatch.setattr(
        DoclingRAGAgent,
        "_rag_loop",
        lambda self, *, query, doc: _make_result("the answer"),
    )

    answer_doc = rag_agent.run("q", sources=[doc])
    trace = rag_agent.run_with_trace("q", sources=[doc])

    # run() must wrap the same final_answer that run_with_trace exposes
    assert answer_doc.name == "rag_answer"
    body_texts = [item.text for item, _ in answer_doc.iterate_items() if hasattr(item, "text")]
    assert trace.final_answer in body_texts


def test_run_with_trace_accepts_legacy_document_kwarg(monkeypatch, rag_agent):
    doc = _make_doc("legacy")
    monkeypatch.setattr(
        DoclingRAGAgent,
        "_rag_loop",
        lambda self, *, query, doc: _make_result("legacy answer"),
    )

    trace = rag_agent.run_with_trace("q", document=doc)

    assert len(trace.per_document) == 1
    assert trace.final_answer == "legacy answer"


def test_run_with_trace_raises_on_empty_sources(rag_agent):
    with pytest.raises(ValueError, match="at least one DoclingDocument"):
        rag_agent.run_with_trace("q")

"""Tests for the exploration history DoclingRAGAgent feeds back into section selection."""

import pytest
from docling_core.types.doc.document import DoclingDocument

from docling_agent.agent.rag import DoclingRAGAgent, _shorten
from docling_agent.agent.rag_models import RAGIteration
from docling_agent.backends.base import BaseSession

from .test_utils import MockBackend

REFS = {"#/texts/1", "#/texts/2", "#/texts/3", "#/texts/4"}


class RecordingSession(BaseSession):
    """Session that records every prompt and replies with scripted responses, last one repeated."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def instruct(self, prompt: str, *, requirements=None, retry_budget: int = 1) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _selection(ref: str) -> str:
    return f'```json\n{{"reason": "picked", "section_ref": "{ref}"}}\n```'


def _iteration(n: int, ref: str, reason: str, can_answer: bool = False) -> RAGIteration:
    return RAGIteration(
        iteration=n,
        section_ref=ref,
        reason=reason,
        section_text_length=10,
        can_answer=can_answer,
        response="still missing the numbers",
    )


@pytest.fixture
def rag_agent() -> DoclingRAGAgent:
    return DoclingRAGAgent(backend=MockBackend(), tools=[])


def _select(
    agent: DoclingRAGAgent,
    session: RecordingSession,
    iterations: list[RAGIteration],
    valid_refs: set[str] = REFS,
):
    return agent._select_section(
        m=session,
        query="what is X?",
        outline_text="outline",
        valid_refs=valid_refs,
        iterations=iterations,
    )


def test_history_block_lists_prior_iterations(rag_agent):
    session = RecordingSession(_selection("#/texts/3"))
    iterations = [
        _iteration(1, "#/texts/1", "the summary mentions X"),
        _iteration(2, "#/texts/2", "looked like a comparison table", can_answer=True),
    ]

    selection = _select(rag_agent, session, iterations)

    prompt = session.prompts[0]
    assert "Exploration history (most recent last):" in prompt
    assert "  - #/texts/1: the summary mentions X -> not helpful" in prompt
    assert "  - #/texts/2: looked like a comparison table -> helpful" in prompt
    assert "Already consulted section refs" not in prompt
    assert selection.section_ref == "#/texts/3"


def test_empty_history_renders_none(rag_agent):
    session = RecordingSession(_selection("#/texts/1"))

    selection = _select(rag_agent, session, [])

    assert "Exploration history (most recent last):\n  none\n" in session.prompts[0]
    assert selection.section_ref == "#/texts/1"
    assert selection.reason == "picked"


def test_visited_refs_are_derived_from_iterations(rag_agent):
    session = RecordingSession(_selection("#/texts/4"))
    iterations = [_iteration(1, "#/texts/1", "r1"), _iteration(2, "#/texts/3", "r3")]

    _select(rag_agent, session, iterations)

    assert "Unvisited section refs to choose from: ['#/texts/2', '#/texts/4']" in session.prompts[0]


def test_history_is_windowed_and_reasons_truncated(rag_agent):
    session = RecordingSession(_selection("#/texts/9"))
    long_reason = "word " * 100
    iterations = [_iteration(n, f"#/texts/{n}", f"reason number {n}") for n in range(1, 8)]
    iterations.append(_iteration(8, "#/texts/8", long_reason))

    _select(rag_agent, session, iterations, valid_refs={f"#/texts/{n}" for n in range(1, 10)})

    prompt = session.prompts[0]
    assert "(3 earlier iteration(s) omitted)" in prompt
    for n in (1, 2, 3):
        assert f"  - #/texts/{n}:" not in prompt
    for n in (4, 5, 6, 7):
        assert f"  - #/texts/{n}: reason number {n} -> not helpful" in prompt
    history_line = next(line for line in prompt.splitlines() if line.startswith("  - #/texts/8:"))
    assert history_line.endswith("… -> not helpful")
    assert len(history_line) < len(long_reason)


def test_fallback_when_session_returns_junk(rag_agent):
    session = RecordingSession("no json here")
    iterations = [_iteration(1, "#/texts/1", "r1")]

    selection = _select(rag_agent, session, iterations)

    assert selection.reason == "fallback"
    assert selection.section_ref == "#/texts/2"


def test_rag_loop_feeds_history_into_next_selection(monkeypatch, rag_agent):
    doc = DoclingDocument(name="doc")
    doc.add_title(text="Title", parent=doc.body)
    doc.add_heading(text="Section one", level=1, parent=doc.body)
    doc.add_text(label="text", text="Content of section one.", parent=doc.body)
    doc.add_heading(text="Section two", level=1, parent=doc.body)
    doc.add_text(label="text", text="Content of section two.", parent=doc.body)

    session = RecordingSession(
        '```json\n{"reason": "section one looks relevant", "section_ref": "#/texts/1"}\n```',
        '```json\n{"can_answer": false, "response": "missing the figures"}\n```',
        '```json\n{"reason": "section two then", "section_ref": "#/texts/3"}\n```',
        '```json\n{"can_answer": true, "response": "X is 42"}\n```',
    )
    monkeypatch.setattr(DoclingRAGAgent, "_create_reasoning_session", lambda self, **kwargs: session)

    result = rag_agent._rag_loop(query="what is X?", doc=doc)

    assert result.converged is True
    assert [it.section_ref for it in result.iterations] == ["#/texts/1", "#/texts/3"]
    second_selection_prompt = session.prompts[2]
    assert "  - #/texts/1: section one looks relevant -> not helpful" in second_selection_prompt
    assert "missing the figures" not in second_selection_prompt
    assert "Unvisited section refs to choose from: ['#/texts/0', '#/texts/3']" in second_selection_prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("short", "short"),
        ("a\n  b\tc", "a b c"),
        ("alpha beta gamma", "alpha beta…"),
        ("abcdefghijklmnop", "abcdefghijk…"),
    ],
)
def test_shorten(text, expected):
    assert _shorten(text, 12) == expected
    assert len(_shorten(text, 12)) <= 12

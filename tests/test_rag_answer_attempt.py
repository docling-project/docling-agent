"""Tests for DoclingRAGAgent._attempt_answer parsing of malformed model output."""

import pytest
from docling_core.types.doc.document import DoclingDocument

from docling_agent.agent.rag import DoclingRAGAgent
from docling_agent.agent.rag_models import AnswerAttempt
from docling_agent.backends.base import BaseSession

from .test_utils import MockBackend


class CannedSession(BaseSession):
    """Session that always answers with the same canned string, whatever the requirements."""

    def __init__(self, response: str) -> None:
        self.response = response

    def instruct(self, prompt: str, *, requirements=None, retry_budget: int = 1) -> str:
        return self.response


@pytest.fixture
def rag_agent() -> DoclingRAGAgent:
    return DoclingRAGAgent(backend=MockBackend(), tools=[])


def _attempt(agent: DoclingRAGAgent, response: str) -> AnswerAttempt:
    return agent._attempt_answer(
        m=CannedSession(response),
        query="what is X?",
        section_ref="#/texts/0",
        section_text="some section content",
    )


def test_attempt_answer_without_json_block_falls_back(rag_agent):
    attempt = _attempt(rag_agent, "I am sorry, I cannot comply with that format.")

    assert attempt.can_answer is False
    assert attempt.response


def test_attempt_answer_with_missing_key_falls_back(rag_agent):
    attempt = _attempt(rag_agent, '```json\n{"response": "here it is"}\n```')

    assert attempt.can_answer is False


def test_attempt_answer_with_wrong_types_falls_back(rag_agent):
    attempt = _attempt(rag_agent, '```json\n{"can_answer": "yes", "response": 42}\n```')

    assert attempt.can_answer is False


def test_attempt_answer_with_valid_json_is_unchanged(rag_agent):
    attempt = _attempt(rag_agent, '```json\n{"can_answer": true, "response": "X is 42"}\n```')

    assert attempt.can_answer is True
    assert attempt.response == "X is 42"


def test_rag_loop_survives_a_backend_that_never_returns_json(monkeypatch, rag_agent):
    doc = DoclingDocument(name="doc")
    doc.add_title(text="Title", parent=doc.body)
    doc.add_heading(text="Section one", level=1, parent=doc.body)
    doc.add_text(label="text", text="Content of section one.", parent=doc.body)

    session = CannedSession("no json here at all")
    monkeypatch.setattr(DoclingRAGAgent, "_create_reasoning_session", lambda self, **kwargs: session)

    result = rag_agent._rag_loop(query="what is X?", doc=doc)

    assert result.converged is False
    assert result.iterations
    assert all(it.can_answer is False for it in result.iterations)

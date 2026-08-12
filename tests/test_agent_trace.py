"""Tests for the generic AgentTrace: base default, RAG specialization, orchestrator tree, export."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from docling_core.types.doc.document import DoclingDocument

from docling_agent import AgentStep, AgentTrace
from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.orchestrator import _COLLECTOR, DoclingOrchestratorAgent, _record_child
from docling_agent.agent.rag import DoclingRAGAgent
from docling_agent.agent.rag_models import RAGIteration, RAGResult, RAGTrace
from docling_agent.task_model import AgentTask

from .test_utils import MockBackend


def _make_doc(name: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_title(text=f"Title of {name}", parent=doc.body)
    doc.add_text(label="text", text=f"Body of {name}.", parent=doc.body)
    return doc


class _DummyAgent(BaseDoclingAgent):
    """Minimal concrete agent that returns a fixed document, exercising the base default."""

    def __init__(self) -> None:
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_WRITER,
            backend=MockBackend(),
            tools=[],
        )

    def run(self, task, document=None, sources=[], **kwargs) -> DoclingDocument:
        return _make_doc("dummy_result")


def test_base_run_with_trace_default_wraps_run():
    """Any agent gets an AgentTrace from the base default, carrying the produced doc."""
    agent = _DummyAgent()
    trace = agent.run_with_trace("do a thing")

    assert isinstance(trace, AgentTrace)
    assert trace.agent_type == "writer"
    assert trace.task == "do a thing"
    assert trace.model_id == "mock-model"
    assert trace.result_name == "dummy_result"
    assert trace.output is not None and trace.output.name == "dummy_result"
    assert trace.duration_ms >= 0
    assert trace.steps == []


def test_run_and_run_with_trace_agree():
    """run() and run_with_trace().output produce the same document."""
    agent = _DummyAgent()
    assert agent.run("x").name == agent.run_with_trace("x").output.name


def _make_result(answer: str) -> RAGResult:
    return RAGResult(
        answer=answer,
        iterations=[
            RAGIteration(
                iteration=1,
                section_ref="#/texts/0",
                reason="picked first section",
                section_text_length=10,
                can_answer=True,
                response=answer,
            )
        ],
        converged=True,
    )


def test_rag_trace_is_agent_trace(monkeypatch):
    """RAGTrace is an AgentTrace; run_with_trace fills generic + RAG-specific fields."""
    monkeypatch.setattr(
        DoclingRAGAgent,
        "_rag_loop",
        lambda self, *, query, doc: _make_result("answer for " + doc.name),
    )
    agent = DoclingRAGAgent(backend=MockBackend(), tools=[])
    doc = _make_doc("doc_a")

    trace = agent.run_with_trace("what is X?", sources=[doc])

    assert isinstance(trace, RAGTrace)
    assert isinstance(trace, AgentTrace)
    assert trace.agent_type == "rag"
    assert trace.task == "what is X?"
    assert trace.query == "what is X?"
    assert trace.final_answer == "answer for doc_a"
    assert trace.output is not None and trace.output.name == "rag_answer"
    # run() returns the same answer document built inside run_with_trace.
    assert agent.run("what is X?", sources=[doc]).name == "rag_answer"


def test_rag_trace_reports_its_duration(monkeypatch):
    """RAGTrace times its own run, so the rag node of a tree is not a flat 0 ms."""

    def slow_loop(self, *, query, doc):
        time.sleep(0.002)
        return _make_result("answer")

    monkeypatch.setattr(DoclingRAGAgent, "_rag_loop", slow_loop)
    agent = DoclingRAGAgent(backend=MockBackend(), tools=[])

    trace = agent.run_with_trace("what is X?", sources=[_make_doc("doc_a")])

    assert trace.duration_ms >= 1


def test_output_excluded_from_serialization():
    """The produced document lives on `output` but is never serialized; result_name is."""
    trace = AgentTrace(agent_type="writer", task="t", result_name="r", output=_make_doc("r"))
    data = json.loads(trace.to_json())
    assert "output" not in data
    assert data["result_name"] == "r"
    assert data["agent_type"] == "writer"


def test_save_roundtrip(tmp_path):
    """save() writes JSON that re-parses into an equal trace (output is not persisted)."""
    child = AgentTrace(agent_type="rag", task="q", steps=[AgentStep(name="iter 1", detail="d")])
    root = AgentTrace(agent_type="orchestrator", task="q", children=[child], output=_make_doc("ignored"))
    out = tmp_path / "nested" / "trace.json"
    root.save(out)

    assert out.exists()
    reparsed = AgentTrace.model_validate_json(out.read_text(encoding="utf-8"))
    assert reparsed.children[0].agent_type == "rag"
    assert reparsed.children[0].steps[0].name == "iter 1"
    # output was excluded, so the reparsed tree (output=None) equals the dumped form.
    assert reparsed == AgentTrace.model_validate(json.loads(root.to_json()))


def test_nested_ragtrace_subclass_fields_are_serialized():
    """A RAGTrace nested under children keeps its RAG-specific fields in the JSON.

    Without SerializeAsAny, pydantic would serialize the child using the declared
    AgentTrace type and silently drop query/per_document/final_answer.
    """
    rag = RAGTrace(
        agent_type="rag",
        task="q",
        query="q",
        per_document=[_make_result("a")],
        final_answer="a",
    )
    root = AgentTrace(agent_type="orchestrator", task="q", children=[rag])
    child = json.loads(root.to_json())["children"][0]

    assert child["final_answer"] == "a"
    assert child["query"] == "q"
    assert len(child["per_document"]) == 1


def test_orchestrator_builds_tree(monkeypatch):
    """run_task_with_trace nests the children recorded during dispatch into one tree."""
    orch = DoclingOrchestratorAgent(backend=MockBackend(), tools=[])
    produced = _make_doc("final")

    def fake_run_task(task):
        # Simulate dispatch recording two sub-agent traces.
        _record_child(AgentTrace(agent_type="enricher", task=task.query))
        _record_child(AgentTrace(agent_type="rag", task=task.query))
        return produced

    monkeypatch.setattr(orch, "run_task", fake_run_task)

    tree = orch.run_task_with_trace(AgentTask(query="explain the doc"))

    assert tree.agent_type == "orchestrator"
    assert tree.task == "explain the doc"
    assert [c.agent_type for c in tree.children] == ["enricher", "rag"]
    assert tree.output is produced
    assert tree.result_name == "final"


def test_record_child_noop_without_active_tree(monkeypatch):
    """Outside run_task_with_trace, recording a child is a no-op (run_task unaffected)."""
    orch = DoclingOrchestratorAgent(backend=MockBackend(), tools=[])
    monkeypatch.setattr(orch, "run_task", lambda task: _make_doc("final"))

    # Assert the collector itself, not a symptom: run_task_with_trace always installs a
    # fresh list, so a leak is only observable on _COLLECTOR.
    assert _COLLECTOR.get() is None
    _record_child(AgentTrace(agent_type="rag", task="q"))  # must not raise
    assert _COLLECTOR.get() is None

    assert orch.run_task_with_trace(AgentTask(query="q")).children == []
    assert _COLLECTOR.get() is None  # the run reset it on the way out


def test_concurrent_traced_runs_do_not_share_children(monkeypatch):
    """Two concurrent runs on the same orchestrator collect into separate trees.

    The collector is scoped to the execution context (contextvars), not to the
    orchestrator instance, so overlapping runs never cross-contaminate.

    Both barriers matter: the runs must be open *before* either records, otherwise a
    single shared collector saved and restored around each run unwinds LIFO and passes
    this test by accident.
    """
    orch = DoclingOrchestratorAgent(backend=MockBackend(), tools=[])
    both_open = threading.Barrier(2, timeout=10)
    both_recorded = threading.Barrier(2, timeout=10)

    def fake_run_task(task):
        both_open.wait()  # both traced runs have installed their collector
        _record_child(AgentTrace(agent_type=task.query, task=task.query))
        both_recorded.wait()  # keep both open until each has recorded into its own
        return _make_doc(task.query)

    monkeypatch.setattr(orch, "run_task", fake_run_task)

    with ThreadPoolExecutor(max_workers=2) as pool:
        trees = list(pool.map(lambda q: orch.run_task_with_trace(AgentTask(query=q)), ["a", "b"]))

    assert [[child.agent_type for child in tree.children] for tree in trees] == [["a"], ["b"]]

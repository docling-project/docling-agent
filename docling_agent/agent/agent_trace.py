"""Generic agent trace: the cross-agent generalization of the RAG reasoning trace.

``AgentStep`` and ``AgentTrace`` lift the pattern introduced for RAG
(``RAGIteration`` / ``RAGResult`` / ``RAGTrace``) to every agent: a run produces an
``AgentTrace`` carrying its timing, model and result, and the orchestrator nests the
traces of the sub-agents it dispatched to under ``children``, forming a tree that can
be exported to a single file for debugging. ``steps`` is the extension point for
per-step detail; no agent populates it yet (RAG carries its own in ``per_document``).

The trace is a value object: it is built and returned, never stored on the agent. The
orchestrator collects its children in a context-scoped variable local to its own module
(``orchestrator._COLLECTOR``), not in shared global state.
``RAGTrace`` is a subclass of ``AgentTrace`` (see ``rag_models``), so the existing
``DoclingRAGAgent.run_with_trace()`` is just the richest specialization of the same
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, Field, SerializeAsAny


class AgentStep(BaseModel):
    """One unit of work inside an agent run (a stage, operation, or iteration)."""

    name: str = Field(description="Short identifier of the step (e.g. 'summarize', 'outline', 'iteration 1').")
    detail: str = Field(default="", description="Human-readable description of what the step did or why.")
    output: str = Field(default="", description="Short textual result/outcome of the step.")
    duration_ms: int = Field(default=0, description="Wall-clock duration of the step in milliseconds.")
    model_id: str = Field(default="", description="Model that produced this step, if any.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra structured detail for the step. Values must be JSON-serializable.",
    )


class AgentTrace(BaseModel):
    """Trace of a single agent run, with optional nested sub-agent traces.

    This is the generic counterpart of ``RAGResult``/``RAGTrace``. The produced
    document is kept on ``output`` for in-memory callers but is excluded from
    serialization (it is the run's payload, not part of the trace record); use
    ``result_name`` for a lightweight pointer in the exported file.
    """

    agent_type: str = Field(default="", description="Type of the agent that produced this trace.")
    task: str = Field(default="", description="The task/query this run answered.")
    steps: list[AgentStep] = Field(default_factory=list, description="Ordered steps performed during the run.")
    children: list[SerializeAsAny[AgentTrace]] = Field(
        default_factory=list,
        description="Traces of sub-agents dispatched by this agent (e.g. orchestrator -> rag/enricher). "
        "SerializeAsAny preserves subclass fields (e.g. RAGTrace) when a child is serialized.",
    )
    duration_ms: int = Field(default=0, description="Wall-clock duration of the whole run in milliseconds.")
    model_id: str = Field(default="", description="Primary model used by this run.")
    result_name: str | None = Field(default=None, description="Name of the produced DoclingDocument, if any.")
    attachments: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific structured payloads attached to the run. Values must be JSON-serializable.",
    )
    output: DoclingDocument | None = Field(
        default=None,
        exclude=True,
        description="The produced document (in-memory only; excluded from serialization).",
    )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the trace tree to JSON (the ``output`` document is excluded)."""
        return self.model_dump_json(indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the trace tree as JSON to ``path``, creating parent directories. Returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

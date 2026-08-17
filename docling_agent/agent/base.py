from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

# from smolagents import MCPClient, Tool, ToolCollection
from docling_agent.agent.agent_trace import AgentTrace

# from smolagents.models import ChatMessage, MessageRole, Model
from docling_agent.backends import BaseBackend, create_backend
from docling_agent.backends.base import BaseSession
from docling_agent.task_model import BackendConfig

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

# Use shared logger from docling_agent.agents


class DoclingAgentType(Enum):
    """Enumeration of supported agent types."""

    # Core agent types
    DOCLING_DOCUMENT_WRITER = "writer"
    DOCLING_DOCUMENT_EDITOR = "editor"
    DOCLING_DOCUMENT_EXTRACTOR = "extractor"
    DOCLING_DOCUMENT_ENRICHER = "enricher"
    DOCLING_DOCUMENT_RAG = "rag"
    DOCLING_DOCUMENT_ORCHESTRATOR = "orchestrator"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> DoclingAgentType:
        """Create AgentType from string value."""
        for agent_type in cls:
            if agent_type.value == value:
                return agent_type
        raise ValueError(f"Invalid agent type: {value}. Valid types: {[t.value for t in cls]}")

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all available agent type strings."""
        return [agent_type.value for agent_type in cls]


class BaseDoclingAgent(ABC):
    """Abstract base class for all Docling agents.

    Agents are behavioral objects that execute natural language tasks
    on documents using LLM backends.

    Attributes:
        agent_type: The type of agent (writer, editor, enricher, etc.)
        backend: The LLM backend used for model interactions
        tools: List of tools available to the agent
        max_iteration: Maximum number of iterations for agent operations
    """

    def __init__(
        self,
        *,
        agent_type: DoclingAgentType,
        backend: BaseBackend,
        tools: list,
        max_iteration: int = 16,
    ):
        """Initialize the base agent.

        Args:
            agent_type: The type of agent being created
            backend: The LLM backend to use
            tools: List of tools available to the agent
            max_iteration: Maximum iterations for agent operations (default: 16)
        """
        self.agent_type = agent_type
        self.backend = backend
        self.tools = tools
        self.max_iteration = max_iteration

    @staticmethod
    def default_backend() -> BaseBackend:
        """Build the default backend used by existing agent constructors."""
        return create_backend(BackendConfig(type="mellea"))

    def get_reasoning_model_id(self) -> str:
        """Return the backend-scoped reasoning model id."""
        return self.backend.models.reasoning

    def get_writing_model_id(self) -> str:
        """Return the backend-scoped writing model id."""
        return self.backend.models.writing

    def get_extraction_model_id(self) -> str:
        """Return the backend-scoped extraction model id."""
        return cast(str, self.backend.models.extraction)

    def _create_reasoning_session(self, *, system_prompt: str | None = None) -> BaseSession:
        """Create a reasoning session with the backend.

        Args:
            system_prompt: Optional system prompt to initialize the session.

        Returns:
            A backend session configured for reasoning tasks.
        """
        return self.backend.create_session(
            model=self.get_reasoning_model_id(),
            system_prompt=system_prompt,
        )

    def _create_writing_session(self, *, system_prompt: str | None = None) -> BaseSession:
        """Create a writing session with the backend.

        Args:
            system_prompt: Optional system prompt to initialize the session.

        Returns:
            A backend session configured for writing tasks.
        """
        return self.backend.create_session(
            model=self.get_writing_model_id(),
            system_prompt=system_prompt,
        )

    def _create_extraction_session(self, *, system_prompt: str | None = None) -> BaseSession:
        """Create an extraction session with the backend.

        Args:
            system_prompt: Optional system prompt to initialize the session.

        Returns:
            A backend session configured for extraction tasks.
        """
        return self.backend.create_session(
            model=self.get_extraction_model_id(),
            system_prompt=system_prompt,
        )

    @abstractmethod
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        """Execute the agent for a task and return a document."""
        raise NotImplementedError

    def run_with_trace(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> AgentTrace:
        """Execute the agent and return a generic ``AgentTrace``.

        Default implementation: time ``run()`` and wrap its result into a single
        opaque trace, so every agent exposes a trace without bespoke code. Agents
        with internal structure override this to return a richer ``AgentTrace``
        subclass: ``DoclingRAGAgent`` returns a ``RAGTrace`` carrying its
        per-document iterations.

        ``run()`` remains the source of truth for the produced document; the document
        is carried on ``AgentTrace.output`` (excluded from serialization).

        Args:
            task: The natural language task to execute.
            document: Optional document the task operates on.
            sources: Optional source documents or paths for the task.
            **kwargs: Additional agent-specific arguments forwarded to ``run()``.

        Returns:
            The trace of the run, carrying the produced document on ``output``.
        """
        start = time.perf_counter()
        result = self.run(task, document=document, sources=sources, **kwargs)
        duration_ms = int((time.perf_counter() - start) * 1000)
        return AgentTrace(
            agent_type=str(self.agent_type),
            task=task,
            duration_ms=duration_ms,
            model_id=self.get_reasoning_model_id(),
            result_name=getattr(result, "name", None),
            output=result,
        )

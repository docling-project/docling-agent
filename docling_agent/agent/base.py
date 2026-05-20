from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from pydantic import BaseModel, ConfigDict

from docling_agent.backends import BaseBackend, create_backend
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


class BaseDoclingAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_type: DoclingAgentType
    backend: BaseBackend
    tools: list

    max_iteration: int = 16

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

    def _create_reasoning_session(self, *, system_prompt: str | None = None):
        return self.backend.create_session(
            model=self.get_reasoning_model_id(),
            system_prompt=system_prompt,
        )

    def _create_writing_session(self, *, system_prompt: str | None = None):
        return self.backend.create_session(
            model=self.get_writing_model_id(),
            system_prompt=system_prompt,
        )

    def _create_extraction_session(self, *, system_prompt: str | None = None):
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

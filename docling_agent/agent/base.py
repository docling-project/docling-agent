from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from mellea.backends.model_ids import ModelIdentifier
from pydantic import BaseModel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

# Use shared logger from docling_agent.agents


class DoclingAgentType(Enum):
    """Enumeration of supported agent types."""

    # Core agent types
    DOCLING_DOCUMENT_WRITER = "writer"
    DOCLING_DOCUMENT_EDITOR = "editor"
    DOCLING_DOCUMENT_EXTRACTOR = "extractor"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "DoclingAgentType":
        """Create AgentType from string value."""
        for agent_type in cls:
            if agent_type.value == value:
                return agent_type
        raise ValueError(
            f"Invalid agent type: {value}. Valid types: {[t.value for t in cls]}"
        )

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all available agent type strings."""
        return [agent_type.value for agent_type in cls]


class BaseDoclingAgent(BaseModel):
    agent_type: DoclingAgentType
    model_id: ModelIdentifier
    tools: list

    # model needed for reasoning/instruction following
    reasoning_model_id: ModelIdentifier | None = None

    # model needed for writing, summarizing, etc
    writing_model_id: ModelIdentifier | None = None

    max_iteration: int = 16

    def get_reasoning_model_id(self) -> ModelIdentifier:
        """Return the reasoning model id, falling back to the primary model."""
        return self.reasoning_model_id or self.model_id

    def get_writing_model_id(self) -> ModelIdentifier:
        """Return the writing model id, falling back to the primary model."""
        return self.writing_model_id or self.model_id

    class Config:
        arbitrary_types_allowed = True  # Needed for complex types like Model

    @abstractmethod
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> "DoclingDocument":
        """Execute the agent for a task and return a document."""
        raise NotImplementedError

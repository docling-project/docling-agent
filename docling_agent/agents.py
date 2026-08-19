# Public re-exports for convenience imports in examples
from docling_agent.agent.agent_trace import AgentStep, AgentTrace
from docling_agent.agent.editor import DoclingEditingAgent
from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.extractor import DoclingExtractingAgent
from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.agent.rag import DoclingRAGAgent, ReasoningBasedPageSelector, TreeGuidedPageSelector
from docling_agent.agent.rag_models import RAGIteration, RAGResult, RAGTrace
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.backends import (
    BaseBackend,
    LiteLLMBackend,
    LMStudioBackend,
    MelleaBackend,
    OllamaBackend,
    create_backend,
)
from docling_agent.logging import logger
from docling_agent.task_model import (
    AddTask,
    AgentTask,
    BackendConfig,
    EnrichTask,
    ExtractTask,
    ListTask,
    ModelConfig,
    OutputConfig,
    RAGTask,
    ViewTask,
    WriteTask,
    load_task,
)

__all__ = [
    "AddTask",
    "AgentStep",
    "AgentTask",
    "AgentTrace",
    "BackendConfig",
    "BaseBackend",
    "DoclingEditingAgent",
    "DoclingEnrichingAgent",
    "DoclingExtractingAgent",
    "DoclingOrchestratorAgent",
    "DoclingRAGAgent",
    "DoclingWritingAgent",
    "EnrichTask",
    "ExtractTask",
    "LMStudioBackend",
    "ListTask",
    "LiteLLMBackend",
    "MelleaBackend",
    "ModelConfig",
    "OllamaBackend",
    "OutputConfig",
    "RAGIteration",
    "RAGResult",
    "RAGTask",
    "RAGTrace",
    "ReasoningBasedPageSelector",
    "TreeGuidedPageSelector",
    "ViewTask",
    "WriteTask",
    "create_backend",
    "load_task",
    "logger",
]

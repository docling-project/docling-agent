# Public re-exports for convenience imports in examples
from docling_agent.agent.editor import DoclingEditingAgent
from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.extractor import DoclingExtractingAgent
from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.agent.rag import DoclingRAGAgent
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
    AgentTask,
    BackendConfig,
    EnrichTask,
    ExtractTask,
    ModelConfig,
    OutputConfig,
    RAGTask,
    WriteTask,
    load_task,
)

__all__ = [
    "AgentTask",
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
    "LiteLLMBackend",
    "MelleaBackend",
    "ModelConfig",
    "OllamaBackend",
    "OutputConfig",
    "RAGTask",
    "WriteTask",
    "create_backend",
    "load_task",
    "logger",
]

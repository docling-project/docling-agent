import logging

# Public re-exports for convenience imports in examples
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.agent.editor import DoclingEditingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

__all__ = [
    "DoclingWritingAgent",
    "DoclingEditingAgent",
    "logger",
]

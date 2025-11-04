import logging

from docling_agent.agent.editor import DoclingEditingAgent

# Public re-exports for convenience imports in examples
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.agent.extraction import DoclingExtractingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

__all__ = [
    "DoclingEditingAgent",
    "DoclingWritingAgent",
    "DoclingExtractingAgent",
    "logger",
]

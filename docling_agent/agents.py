import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from docling_agent.agent.base import DoclingAgentType, BaseDoclingAgent
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.agent.editor import DoclingEditingAgent

import copy
import logging
from typing import ClassVar

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    ListItem,
    NodeItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    convert_html_to_docling_document,
    convert_markdown_to_docling_document,
    find_outline_v2,
    has_html_code_block,
    has_markdown_code_block,
    validate_html_to_docling_document,
    validate_markdown_to_docling_document,
    validate_outline_format,
)
from docling_agent.agent_models import setup_local_session

# from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from docling_agent.resources.prompts import (
    SYSTEM_PROMPT_EXPERT_TABLE_WRITER,
    SYSTEM_PROMPT_EXPERT_WRITER,
    SYSTEM_PROMPT_FOR_OUTLINE,
    SYSTEM_PROMPT_FOR_TASK_ANALYSIS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DoclingExtractionAgent(BaseDoclingAgent):

    def __init__(self, *, model_id: ModelIdentifier, tools: list):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EXTRACTOR,
            model_id=model_id,
            tools=tools,
        )

    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument] = [],
        **kwargs,
    ) -> DoclingDocument:
        # TBD

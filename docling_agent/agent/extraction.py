import json
import logging
from pathlib import Path
from typing import ClassVar

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import Field

# from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor
from docling_core.types.doc import (
    CodeLanguageLabel,
    DoclingDocument,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent_models import setup_local_session

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DoclingExtractingAgent(BaseDoclingAgent):
    system_prompt_schema_extraction: ClassVar[str] = (
        "You are a precise data engineer. Given a task, output only a JSON schema (no backticks) describing the fields to extract with simple value types like string, integer, float, date, or arrays/objects of those."
    )

    # Optional during validation; initialized in __init__
    extractor: DocumentExtractor | None = None

    # Stores the last extraction results as {path_str: [json_obj, ...]}
    last_results: dict[str, list[dict]] = Field(default_factory=dict)

    def __init__(self, *, model_id: ModelIdentifier, tools: list):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EXTRACTOR,
            model_id=model_id,
            tools=tools,
        )
        self.extractor = DocumentExtractor(
            allowed_formats=[InputFormat.IMAGE, InputFormat.PDF]
        )

    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[Path] | None = None,
        **kwargs,
    ) -> DoclingDocument:
        # If the task already includes a JSON schema, use it; otherwise generate one.
        schema = self._extract_schema_from_task(task=task)
        logger.info("Schema generated from task.")

        if sources is None:
            sources = []

        total = len(sources)
        successes = 0
        failures = 0
        total_items = 0
        self.last_results = {}

        logger.info(f"Starting extraction for {total} source(s) for schema: {schema}")
        for idx, source in enumerate(sources, start=1):
            logger.info(f"[{idx}/{total}] Extracting from: {source}")
            if isinstance(source, Path):
                try:
                    result = self.extractor.extract(source=source, template=schema)
                    # Ensure list-of-json for values
                    items: list
                    if isinstance(result, list):
                        items = result
                    else:
                        items = [result]
                    self.last_results[source] = items  # key by path string
                    successes += 1
                    total_items += len(items)
                    logger.info(
                        f"Completed {source} with {len(items)} item(s) extracted."
                    )
                except Exception as e:
                    failures += 1
                    logger.error(f"Failed to extract from {source}: {e}")
            else:
                failures += 1
                logger.error(f"source {source} is not the right type (expected Path)")

        logger.info(
            f"Extraction summary: files={total}, successes={successes}, failures={failures}, items={total_items}"
        )

        # Build a document with headings per source and fenced JSON blocks per item.
        doc = DoclingDocument(name="extraction results")
        doc.add_title(text="Extraction Results")
        for path, items in self.last_results.items():
            doc.add_heading(text=f"filename: {path.name}", level=1)
            for item in items:
                for page in item.pages:
                    payload = json.dumps(page.extracted_data, indent=2)
                    doc.add_code(
                        code_language=CodeLanguageLabel.JSON,
                        text=f"```json\n{payload}\n```",
                    )

        return doc

    def _extract_schema_from_task(self, task: str, loop_budget: int = 5) -> dict:
        try:
            schema = json.loads(task)
            return schema
        except Exception:
            logger.info("not a direct json")

        def validate_json_str(text: str) -> bool:
            try:
                json.loads(text)
                return True
            except Exception:
                return False

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_schema_extraction,
        )

        prompt = f"{task}"
        logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Return only a valid JSON object without code fences.",
                    validation_fn=simple_validate(validate_json_str),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        return json.loads(answer.value)

from pathlib import Path
from typing import Any, ClassVar

from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import Field

from docling_core.types.doc.document import (
    DoclingDocument,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import find_json_dicts
from docling_agent.agent_models import setup_local_session, view_linear_context
from docling_agent.logging import logger


class DoclingEnrichingAgent(BaseDoclingAgent):
    """Agent for enriching a document with metadata like summaries, keywords,
    entities, and classifications.

    This scaffold routes the task to one of several enrichment operations using
    a small reasoning step that returns a JSON instruction containing an
    `operation` field. Each operation function currently iterates items and is
    left for concrete implementation.
    """

    # Simple system prompt to route enrichment tasks
    system_prompt_for_enrichment_routing: ClassVar[str] = (
        """
You are a precise document enrichment router. Given a natural language task description, select exactly one operation to run and return only one JSON object in a ```json ...``` block, with the following schema:

{
  "operation": "summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items",
  "args": { }
}

Return no extra commentary. If multiple seem plausible, choose the single best fit.
        """
    )

    # Store last chosen operation for introspection/debugging (optional)
    last_operation: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, *, model_id: ModelIdentifier, tools: list):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_ENRICHER,
            model_id=model_id,
            tools=tools,
        )

    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        if document is None:
            raise ValueError("Document must not be None")

        op = self._choose_operation(task=task)
        self.last_operation = op

        operation = op.get("operation")
        args = op.get("args", {})

        logger.info(f"Chosen enrichment operation: {operation}")

        if operation == "summarize_items":
            self._summarize_items(document=document, **args)
        elif operation == "find_search_keywords":
            self._find_search_keywords(document=document, **args)
        elif operation == "detect_key_entities":
            self._detect_key_entities(document=document, **args)
        elif operation == "classify_items":
            self._classify_items(document=document, **args)
        else:
            raise ValueError(
                f"Unknown enrichment operation: {operation}. Op payload: {op}"
            )

        return document

    def _choose_operation(self, *, task: str, loop_budget: int = 5) -> dict[str, Any]:
        logger.info(f"task: {task}")

        m = setup_local_session(
            model_id=self.get_reasoning_model_id(),
            system_prompt=self.system_prompt_for_enrichment_routing,
        )

        answer = m.instruct(
            task,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        view_linear_context(m)

        ops = find_json_dicts(text=answer.value)
        if not ops:
            raise ValueError("No routing operation detected in model response")
        if "operation" not in ops[0]:
            raise ValueError(f"`operation` not found in routing result: {ops[0]}")
        return ops[0]

    # --- Enrichment operations (scaffolds) ---
    def _summarize_items(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_summarize_items: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)  # placeholder to avoid unused warnings
            # TODO: implement summarization per item
            # e.g., update item.meta.summary
            pass

    def _find_search_keywords(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_find_search_keywords: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement keyword extraction per item
            pass

    def _detect_key_entities(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_detect_key_entities: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement entity detection per item
            pass

    def _classify_items(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_classify_items: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement classification per item (language, function, etc.)
            pass


from pathlib import Path
from typing import Any, ClassVar

from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import Field

from docling_core.types.doc.document import (
    DoclingDocument,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownTextSerializer,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    find_json_dicts,
    has_json_dicts,
    find_markdown_code_block,
    has_markdown_code_block,
)
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
    system_prompt_for_enrichment_routing: ClassVar[str] = """
You are a precise document enrichment router. Given a natural language task description, select exactly one operation to run and return only one JSON object in a ```json ...``` block, with the following schema:

{
  "operation": "summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items",
  "args": { }
}

Return no extra commentary. If multiple seem plausible, choose the single best fit.
        """

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
            requirements=[
                Requirement(
                    description="Put the resulting function calls in the format ```json <insert-content>```",
                    validation_fn=simple_validate(has_json_dicts),
                ),
                Requirement(
                    description="""Given a natural language task description, select exactly one operation to run and return only one JSON object in a ```json ...``` block, with the following schema:

{
  "operation": "summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items",
  "args": { }
}""",
                    validation_fn=simple_validate(
                        DoclingEnrichingAgent._validate_json_format
                    ),
                ),
            ],
        )

        view_linear_context(m)

        ops = find_json_dicts(text=answer.value)
        if not ops:
            raise ValueError("No routing operation detected in model response")
        if "operation" not in ops[0]:
            raise ValueError(f"`operation` not found in routing result: {ops[0]}")
        return ops[0]

    @staticmethod
    def _validate_json_format(content: str) -> bool:
        """Validate that content contains a json with valid format.

        Rules:
        - The json must be inside a ```json ... ``` code block.
        - The operations must be of type "summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items".
        """

        ops = find_json_dicts(text=content)

        # Must contain exactly one JSON object in a code block
        if len(ops) != 1:
            return False

        # Operation must be one of the allowed values
        allowed_ops = {
            "summarize_items",
            "find_search_keywords",
            "detect_key_entities",
            "classify_items",
        }

        return ops[0].get("operation") in allowed_ops

    # --- Enrichment operations (scaffolds) ---
    def _summarize_items(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_summarize_items: iterating over document items")

        md_serializer = MarkdownDocSerializer(doc=document)
        html_serializer = HTMLDocSerializer(doc=document)

        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)  # placeholder to avoid unused warnings

            # check if item already has a summary
            if (item.meta is not None) and (item.meta.summary is not None):
                continue

            if isinstance(item, TitleItem):
                pass
            elif isinstance(item, SectionHeaderItem):
                pass
            elif isinstance(item, ListItem):
                pass
            elif isinstance(item, TextItem):
                ser = MarkdownTextSerializer()
                res = ser.serialize(item=item, doc=document, doc_serializer=md_serializer)

                ctask = f"""Summarize the following text snippet in two or three succinct sentences:\n\n```{task}\n\n```."""
                logger.info(f"task: {ctask}")
                
                answer = m.instruct(
                    f"{ctask}",
                    requirements=[
                        Requirement(
                            description="Put the resulting markdown outline in the format ```markdown <insert-content>```",
                            validation_fn=simple_validate(has_markdown_code_block),
                        ),
                    ],
                    strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
                )

                summary = find_markdown_code_block(text=answer.value)

                if summary is None:
                    raise ValueError("Failed to generate a valid summary.")

                item.meta.summary = SummaryMetaField(text=summary)

            elif isinstance(item, TableItem):
                ser = HTMLTableSerializer()
                res = ser.serialize(item=item, doc=document, doc_serializer=html_serializer)

                ctask = f"""Summarize the following html table in two or three succinct sentences:\n\n```{task}\n\n```."""
                logger.info(f"task: {ctask}")
                
                answer = m.instruct(
                    f"{ctask}",
                    requirements=[
                        Requirement(
                            description="Put the resulting markdown outline in the format ```markdown <insert-content>```",
                            validation_fn=simple_validate(has_markdown_code_block),
                        ),
                    ],
                    # user_variables={"name": name, "notes": notes},
                    strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
                )

                summary = find_markdown_code_block(text=answer.value)

                if summary is None:
                    raise ValueError("Failed to generate a valid summary.")

                item.meta.summary = SummaryMetaField(text=summary)
            elif isinstance(item, PictureItem):
                pass
            else:
                logger.warning(f"Can not summerarize item of label ({item.label})")

    def _find_search_keywords(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_find_search_keywords: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement keyword extraction per item

    def _detect_key_entities(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_detect_key_entities: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement entity detection per item

    def _classify_items(self, *, document: DoclingDocument, **kwargs) -> None:
        logger.info("_classify_items: iterating over document items")
        for item, level in document.iterate_items(with_groups=True):
            _ = (item, level)
            # TODO: implement classification per item (language, function, etc.)

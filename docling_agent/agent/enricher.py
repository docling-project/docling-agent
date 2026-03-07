import json
import re
from pathlib import Path
from typing import Any, ClassVar

from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import Field

from docling_core.types.doc.document import (
    BaseMeta,
    DoclingDocument,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TextItem,
    TitleItem,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    collect_subtree_text,
    find_json_dicts,
    has_json_dicts,
    make_hierarchical_document,
    serialize_table_to_html,
)
from docling_agent.agent_models import setup_local_session, view_linear_context
from docling_agent.logging import logger

# Mapping from routing names and short names to method names
_OP_ALIASES: dict[str, str] = {
    "summarize_items": "_summarize_items",
    "summarize": "_summarize_items",
    "find_search_keywords": "_find_search_keywords",
    "keywords": "_find_search_keywords",
    "detect_key_entities": "_detect_key_entities",
    "entities": "_detect_key_entities",
    "classify_items": "_classify_items",
}

_ROUTING_OPS = frozenset(
    ["summarize_items", "find_search_keywords", "detect_key_entities", "classify_items"]
)


class DoclingEnrichingAgent(BaseDoclingAgent):
    """Agent for enriching a document with metadata like summaries, keywords,
    entities, and classifications."""

    system_prompt_for_enrichment_routing: ClassVar[str] = """
You are a precise document enrichment router. Given a natural language task description, select exactly one operation to run and return only one JSON object in a ```json ...``` block, with the following schema:

{
  "operation": "summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items",
  "args": { }
}

Return no extra commentary. If multiple seem plausible, choose the single best fit.
    """

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

        # Explicit operations list bypasses LLM routing entirely
        operations: list[str] | None = kwargs.get("operations")
        if operations is not None:
            result = document
            for op_name in operations:
                method_name = _OP_ALIASES.get(op_name)
                if method_name is None:
                    raise ValueError(f"Unknown operation: {op_name!r}")
                method = getattr(self, method_name)
                result = method(document=result) or result
            return result

        op = self._choose_operation(task=task)
        self.last_operation = op

        operation = op.get("operation")
        logger.info(f"Chosen enrichment operation: {operation}")

        method_name = _OP_ALIASES.get(operation or "")
        if method_name is None:
            raise ValueError(
                f"Unknown enrichment operation: {operation}. Op payload: {op}"
            )
        method = getattr(self, method_name)
        return method(document=document) or document

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
                    description=(
                        "Select exactly one operation and return only one JSON object "
                        "in a ```json ...``` block with 'operation' set to one of: "
                        + ", ".join(sorted(_ROUTING_OPS))
                    ),
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
        ops = find_json_dicts(text=content)
        if len(ops) != 1:
            return False
        return ops[0].get("operation") in _ROUTING_OPS

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_items(
        self,
        *,
        document: DoclingDocument,
        fix_heading_levels: bool = True,
        min_text_length: int = 100,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_summarize_items: starting")

        if fix_heading_levels:
            self._fix_heading_levels(document=document)

        hier_doc = make_hierarchical_document(document)

        m = setup_local_session(model_id=self.get_reasoning_model_id())

        self._walk_and_summarize(
            node=hier_doc.body,
            doc=hier_doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
        )
        self._summarize_leaf_items(
            m=m,
            document=hier_doc,
            loop_budget=loop_budget,
        )

        return hier_doc

    def _fix_heading_levels(self, *, document: DoclingDocument) -> None:
        from docling_agent.agent.editor import DoclingEditingAgent

        editor = DoclingEditingAgent(model_id=self.get_reasoning_model_id(), tools=[])
        editor.run(
            task=(
                "Fix the section heading levels so the document hierarchy is correct. "
                "Level 1 is a top-level section, level 2 a subsection, etc."
            ),
            document=document,
        )

    def _walk_and_summarize(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: Any,
        loop_budget: int,
        min_text_length: int,
    ) -> None:
        if isinstance(node, (TitleItem, SectionHeaderItem)):
            if not (node.meta and node.meta.summary):
                text = collect_subtree_text(node, doc)
                if len(text) >= min_text_length:
                    summary = self._generate_summary(
                        m=m, text=text, loop_budget=loop_budget
                    )
                    if summary:
                        if node.meta is None:
                            node.meta = BaseMeta()
                        node.meta.summary = SummaryMetaField(text=summary)

        for child_ref in node.children or []:
            try:
                child = child_ref.resolve(doc)
                self._walk_and_summarize(
                    node=child,
                    doc=doc,
                    m=m,
                    loop_budget=loop_budget,
                    min_text_length=min_text_length,
                )
            except Exception as exc:
                logger.warning(f"Could not resolve child {child_ref}: {exc}")

    def _summarize_leaf_items(
        self,
        *,
        m: Any,
        document: DoclingDocument,
        loop_budget: int,
    ) -> None:
        for item, _ in document.iterate_items():
            if item.meta and item.meta.summary:
                continue
            if isinstance(item, TableItem):
                html = serialize_table_to_html(table=item, doc=document)
                summary = self._generate_summary(
                    m=m,
                    text=f"HTML table:\n{html}",
                    loop_budget=loop_budget,
                )
            elif isinstance(item, PictureItem):
                captions = [
                    c.resolve(document).text
                    for c in item.captions
                    if hasattr(c.resolve(document), "text")
                ]
                text = " ".join(captions)
                summary = (
                    self._generate_summary(m=m, text=text, loop_budget=loop_budget)
                    if text
                    else None
                )
            else:
                continue
            if summary:
                if item.meta is None:
                    item.meta = BaseMeta()
                item.meta.summary = SummaryMetaField(text=summary)

    def _generate_summary(
        self,
        *,
        m: Any,
        text: str,
        loop_budget: int = 5,
    ) -> str | None:
        ctask = (
            "Summarize the following content in two or three succinct sentences. "
            "Return only plain text with no markdown formatting.\n\n"
            f"{text}"
        )

        def _validate_summary(content: str) -> bool:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            return 1 <= len(sentences) <= 5

        try:
            answer = m.instruct(
                ctask,
                strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
                requirements=[
                    Requirement(
                        description="Write 2-3 succinct sentences summarizing the content. Return plain text only.",
                        validation_fn=simple_validate(_validate_summary),
                    ),
                ],
            )
            return answer.value.strip() or None
        except Exception as exc:
            logger.warning(f"Summary generation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Keywords
    # ------------------------------------------------------------------

    def _find_search_keywords(
        self,
        *,
        document: DoclingDocument,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_find_search_keywords: iterating over document items")

        m = setup_local_session(model_id=self.get_reasoning_model_id())

        for item, _ in document.iterate_items():
            if not isinstance(item, (TitleItem, SectionHeaderItem, TextItem)):
                continue
            if not hasattr(item, "text") or not item.text:
                continue

            ctask = (
                "Extract 3 to 7 search keywords from the following text. "
                "Return them as a JSON array of strings in a ```json ...``` block.\n\n"
                f"{item.text}"
            )

            def _validate_keywords(content: str) -> bool:
                match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                if not match:
                    return False
                try:
                    val = json.loads(match.group(1))
                    return isinstance(val, list) and len(val) >= 1
                except Exception:
                    return False

            try:
                answer = m.instruct(
                    ctask,
                    strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
                    requirements=[
                        Requirement(
                            description="Return 3-7 keywords as a JSON array in a ```json ...``` block.",
                            validation_fn=simple_validate(_validate_keywords),
                        ),
                    ],
                )
                match = re.search(r"```json\s*(.*?)\s*```", answer.value, re.DOTALL)
                if match:
                    keywords = json.loads(match.group(1))
                    if item.meta is None:
                        item.meta = BaseMeta()
                    item.meta.keywords = keywords  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning(f"Keyword extraction failed for {item.self_ref}: {exc}")

        return document

    # ------------------------------------------------------------------
    # Stubs
    # ------------------------------------------------------------------

    def _detect_key_entities(
        self, *, document: DoclingDocument, **kwargs
    ) -> DoclingDocument:
        logger.info("_detect_key_entities: not yet implemented")
        return document

    def _classify_items(
        self, *, document: DoclingDocument, **kwargs
    ) -> DoclingDocument:
        logger.info("_classify_items: not yet implemented")
        return document

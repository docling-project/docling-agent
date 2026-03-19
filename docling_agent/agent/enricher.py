import json
import re
from pathlib import Path
from typing import Any, ClassVar, Protocol

from docling_core.types.doc.document import (
    BaseMeta,
    DoclingDocument,
    FloatingMeta,
    NodeItem,
    PictureItem,
    PictureMeta,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TitleItem,
)
from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import Field

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    collect_subtree_text,
    find_json_dicts,
    has_json_dicts,
    make_hierarchical_document,
    serialize_table_to_html,
)
from docling_agent.agent_models import _LoggingSession, setup_local_session, view_linear_context
from docling_agent.logging import logger


class _GenerateFn(Protocol):
    """Protocol for content generation functions."""

    def __call__(
        self, *, m: _LoggingSession, text: str, loop_budget: int
    ) -> str | list[str] | list[dict[str, str]] | None: ...


class _SetMetaFn(Protocol):
    """Protocol for metadata setter functions."""

    def __call__(self, meta: BaseMeta, result: Any) -> None: ...


class _ValidationFn(Protocol):
    """Protocol for validation functions."""

    def __call__(self, content: str) -> bool: ...


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

_ROUTING_OPS = frozenset(["summarize_items", "find_search_keywords", "detect_key_entities", "classify_items"])


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
            raise ValueError(f"Unknown enrichment operation: {operation}. Op payload: {op}")
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
                        "in a ```json ...``` block with 'operation' set to one of: " + ", ".join(sorted(_ROUTING_OPS))
                    ),
                    validation_fn=simple_validate(DoclingEnrichingAgent._validate_json_format),
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
                "Level 1 is a top-level section, level 2 a subsection, level 3 a sub-subsection, etc. "
                "Use the update_section_heading_level operation with the exact field name 'to_level' "
                "(not 'level' or 'levels'). Return the operation in the exact format specified in the system prompt."
            ),
            document=document,
        )

    def _walk_and_enrich(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: _LoggingSession,
        loop_budget: int,
        min_text_length: int,
        meta_attr: str,
        generate_fn: _GenerateFn,
        set_meta_fn: _SetMetaFn,
    ) -> None:
        """Generic method to walk document tree and enrich nodes with metadata."""
        if isinstance(node, TitleItem | SectionHeaderItem):
            if not (node.meta and hasattr(node.meta, meta_attr) and getattr(node.meta, meta_attr)):
                text = collect_subtree_text(node, doc)
                if len(text) >= min_text_length:
                    result = generate_fn(m=m, text=text, loop_budget=loop_budget)
                    if result:
                        if node.meta is None:
                            node.meta = BaseMeta()
                        set_meta_fn(node.meta, result)

        for child_ref in node.children or []:
            try:
                child = child_ref.resolve(doc)
                self._walk_and_enrich(
                    node=child,
                    doc=doc,
                    m=m,
                    loop_budget=loop_budget,
                    min_text_length=min_text_length,
                    meta_attr=meta_attr,
                    generate_fn=generate_fn,
                    set_meta_fn=set_meta_fn,
                )
            except Exception as exc:
                logger.warning(f"Could not resolve child {child_ref}: {exc}")

    def _enrich_leaf_items(
        self,
        *,
        m: _LoggingSession,
        document: DoclingDocument,
        loop_budget: int,
        meta_attr: str,
        generate_fn: _GenerateFn,
        set_meta_fn: _SetMetaFn,
    ) -> None:
        """Generic method to enrich leaf items (tables, pictures) with metadata."""
        for item, _ in document.iterate_items():
            if item.meta and hasattr(item.meta, meta_attr) and getattr(item.meta, meta_attr):
                continue
            if isinstance(item, TableItem):
                html = serialize_table_to_html(table=item, doc=document)
                result = generate_fn(
                    m=m,
                    text=f"HTML table:\n{html}",
                    loop_budget=loop_budget,
                )
                if result:
                    if item.meta is None:
                        item.meta = FloatingMeta()
                    set_meta_fn(item.meta, result)
            elif isinstance(item, PictureItem):
                captions = [c.resolve(document).text for c in item.captions if hasattr(c.resolve(document), "text")]
                text = " ".join(captions)
                result = generate_fn(m=m, text=text, loop_budget=loop_budget) if text else None
                if result:
                    if item.meta is None:
                        item.meta = PictureMeta()
                    set_meta_fn(item.meta, result)

    def _walk_and_summarize(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: _LoggingSession,
        loop_budget: int,
        min_text_length: int,
    ) -> None:
        """Walk document tree and add summaries to sections."""

        def set_summary(meta: BaseMeta, result: Any) -> None:
            meta.summary = SummaryMetaField(text=result)

        self._walk_and_enrich(
            node=node,
            doc=doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
            meta_attr="summary",
            generate_fn=self._generate_summary,
            set_meta_fn=set_summary,
        )

    def _summarize_leaf_items(
        self,
        *,
        m: _LoggingSession,
        document: DoclingDocument,
        loop_budget: int,
    ) -> None:
        """Add summaries to leaf items (tables, pictures)."""

        def set_summary(meta: BaseMeta, result: Any) -> None:
            meta.summary = SummaryMetaField(text=result)

        self._enrich_leaf_items(
            m=m,
            document=document,
            loop_budget=loop_budget,
            meta_attr="summary",
            generate_fn=self._generate_summary,
            set_meta_fn=set_summary,
        )

    def _generate_content(
        self,
        *,
        m: _LoggingSession,
        text: str,
        task_prompt: str,
        requirement_description: str,
        validation_fn: _ValidationFn,
        loop_budget: int = 5,
    ) -> str | None:
        """Generic method to generate content (summaries, keywords, etc.) from text."""
        ctask = f"{task_prompt}\n\n{text}"

        try:
            answer = m.instruct(
                ctask,
                strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
                requirements=[
                    Requirement(
                        description=requirement_description,
                        validation_fn=simple_validate(validation_fn),
                    ),
                ],
            )
            return answer.value.strip() or None
        except Exception as exc:
            logger.warning(f"Content generation failed: {exc}")
            return None

    def _generate_summary(
        self,
        *,
        m: _LoggingSession,
        text: str,
        loop_budget: int = 5,
    ) -> str | None:
        def _validate_summary(content: str) -> bool:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            return 1 <= len(sentences) <= 5

        return self._generate_content(
            m=m,
            text=text,
            task_prompt=(
                "Summarize the following content in two or three succinct sentences. "
                "Return only plain text with no markdown formatting."
            ),
            requirement_description="Write 2-3 succinct sentences summarizing the content. Return plain text only.",
            validation_fn=_validate_summary,
            loop_budget=loop_budget,
        )

    def _generate_keywords(
        self,
        *,
        m: _LoggingSession,
        text: str,
        loop_budget: int = 5,
    ) -> list[str] | None:
        def _validate_keywords(content: str) -> bool:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if not match:
                return False
            try:
                val = json.loads(match.group(1))
                return isinstance(val, list) and 3 <= len(val) <= 7
            except Exception:
                return False

        result = self._generate_content(
            m=m,
            text=text,
            task_prompt=(
                "Extract 3 to 7 compelling and specific search keywords from the following content. "
                "Focus on key concepts, technical terms, and important topics that would help someone find this content. "
                "Return them as a JSON array of strings in a ```json ...``` block."
            ),
            requirement_description="Return 3-7 keywords as a JSON array in a ```json ...``` block.",
            validation_fn=_validate_keywords,
            loop_budget=loop_budget,
        )

        if result:
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception as exc:
                    logger.warning(f"Failed to parse keywords JSON: {exc}")
        return None

    # ------------------------------------------------------------------
    # Keywords
    # ------------------------------------------------------------------

    def _find_search_keywords(
        self,
        *,
        document: DoclingDocument,
        fix_heading_levels: bool = True,
        min_text_length: int = 100,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_find_search_keywords: starting")

        if fix_heading_levels:
            self._fix_heading_levels(document=document)

        hier_doc = make_hierarchical_document(document)

        m = setup_local_session(model_id=self.get_reasoning_model_id())

        self._walk_and_extract_keywords(
            node=hier_doc.body,
            doc=hier_doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
        )
        self._extract_keywords_from_leaf_items(
            m=m,
            document=hier_doc,
            loop_budget=loop_budget,
        )

        return hier_doc

    def _walk_and_extract_keywords(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: _LoggingSession,
        loop_budget: int,
        min_text_length: int,
    ) -> None:
        """Walk document tree and add keywords to sections."""

        def set_keywords(meta: BaseMeta, result: Any) -> None:
            meta.docling_agent__keywords = result

        self._walk_and_enrich(
            node=node,
            doc=doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
            meta_attr="docling_agent__keywords",
            generate_fn=self._generate_keywords,
            set_meta_fn=set_keywords,
        )

    def _extract_keywords_from_leaf_items(
        self,
        *,
        m: _LoggingSession,
        document: DoclingDocument,
        loop_budget: int,
    ) -> None:
        """Add keywords to leaf items (tables, pictures)."""

        def set_keywords(meta: BaseMeta, result: Any) -> None:
            meta.docling_agent__keywords = result

        self._enrich_leaf_items(
            m=m,
            document=document,
            loop_budget=loop_budget,
            meta_attr="docling_agent__keywords",
            generate_fn=self._generate_keywords,
            set_meta_fn=set_keywords,
        )

    # ------------------------------------------------------------------
    # Entity Detection
    # ------------------------------------------------------------------

    def _detect_key_entities(
        self,
        *,
        document: DoclingDocument,
        fix_heading_levels: bool = True,
        min_text_length: int = 100,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_detect_key_entities: starting")

        if fix_heading_levels:
            self._fix_heading_levels(document=document)

        hier_doc = make_hierarchical_document(document)

        m = setup_local_session(model_id=self.get_reasoning_model_id())

        self._walk_and_extract_entities(
            node=hier_doc.body,
            doc=hier_doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
        )
        self._extract_entities_from_leaf_items(
            m=m,
            document=hier_doc,
            loop_budget=loop_budget,
        )

        return hier_doc

    def _walk_and_extract_entities(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: _LoggingSession,
        loop_budget: int,
        min_text_length: int,
    ) -> None:
        """Walk document tree and add entities to sections."""

        def set_entities(meta: BaseMeta, result: Any) -> None:
            meta.docling_agent__entities = result

        self._walk_and_enrich(
            node=node,
            doc=doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
            meta_attr="docling_agent__entities",
            generate_fn=self._generate_entities,
            set_meta_fn=set_entities,
        )

    def _extract_entities_from_leaf_items(
        self,
        *,
        m: _LoggingSession,
        document: DoclingDocument,
        loop_budget: int,
    ) -> None:
        """Add entities to leaf items (tables, pictures)."""

        def set_entities(meta: BaseMeta, result: Any) -> None:
            meta.docling_agent__entities = result

        self._enrich_leaf_items(
            m=m,
            document=document,
            loop_budget=loop_budget,
            meta_attr="docling_agent__entities",
            generate_fn=self._generate_entities,
            set_meta_fn=set_entities,
        )

    def _generate_entities(
        self,
        *,
        m: _LoggingSession,
        text: str,
        loop_budget: int = 5,
    ) -> list[dict[str, str]] | None:
        def _validate_entities(content: str) -> bool:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if not match:
                return False
            try:
                val = json.loads(match.group(1))
                if not isinstance(val, list):
                    return False
                # Check that each item has entity_type and mention
                return all(isinstance(item, dict) and "entity_type" in item and "mention" in item for item in val)
            except Exception:
                return False

        result = self._generate_content(
            m=m,
            text=text,
            task_prompt=(
                "Extract named entities from the following content. "
                "For each entity, identify its type (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, etc.) and the exact mention text. "
                "Return them as a JSON array of objects with 'entity_type' and 'mention' fields in a ```json ...``` block. "
                "Only include significant entities, avoid generic terms. "
                "Do not repeat the same entity even if it appears multiple times."
            ),
            requirement_description="Return entities as a JSON array of objects with 'entity_type' and 'mention' fields in a ```json ...``` block.",
            validation_fn=_validate_entities,
            loop_budget=loop_budget,
        )

        if result:
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception as exc:
                    logger.warning(f"Failed to parse entities JSON: {exc}")
        return None

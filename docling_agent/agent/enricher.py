import json
import re
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

from docling_core.types.doc.document import (
    BaseMeta,
    CodeLanguageLabel,
    CodeMetaField,
    DocItemLabel,
    DoclingDocument,
    EntitiesMetaField,
    EntityMention,
    FloatingMeta,
    GroupItem,
    NodeItem,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
    SectionHeaderItem,
    SummaryMetaField,
    TableData,
    TableItem,
    TabularChartMetaField,
    TextItem,
    TitleItem,
)
from mellea.stdlib.requirements import Requirement, simple_validate
from pydantic import Field

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    collect_subtree_text,
    find_json_dicts,
    has_json_dicts,
    make_hierarchical_document,
    serialize_table_to_html,
)
from docling_agent.agent_models import view_linear_context
from docling_agent.backends.base import BaseSession
from docling_agent.logging import logger


class _GenerateFn(Protocol):
    """Protocol for content generation functions."""

    def __call__(
        self, *, m: BaseSession, text: str, loop_budget: int
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
    "classify": "_classify_items",
}

_ROUTING_OPS = (
    "summarize_items",
    "find_search_keywords",
    "detect_key_entities",
    "classify_items",
)


class DoclingEnrichingAgent(BaseDoclingAgent):
    """Agent for enriching a document with metadata like summaries, keywords,
    entities, and classifications."""

    system_prompt_for_enrichment_routing: ClassVar[str] = """
You are a precise document enrichment router. Given a natural language task description, select one or more enrichment operations to run and return only one JSON object in a ```json ...``` block, with the following schema:

{
  "operations": ["summarize_items" | "find_search_keywords" | "detect_key_entities" | "classify_items", ...],
  "reason": "short explanation"
}

Return no extra commentary. Include all operations that are materially requested by the task. Preserve execution order.
    """
    _PICTURE_CLASS_NAMES: ClassVar[tuple[str, ...]] = (
        "bar-chart",
        "line-chart",
        "pie-chart",
        "scatter-chart",
        "spider-chart",
        "flowchart",
        "diagram",
        "schematic",
        "illustration",
        "photo",
        "table-image",
        "other",
    )
    _SUPPORTED_CHART_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"bar-chart", "line-chart", "pie-chart", "scatter-chart", "spider-chart"}
    )

    last_operation: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
    ):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_ENRICHER,
            backend=backend or self.default_backend(),
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
            return self._run_operations(task=task, document=document, operations=operations)

        plan = self._choose_operations(task=task)
        self.last_operation = plan
        inferred_operations = plan.get("operations", [])
        logger.info(f"Chosen enrichment operations: {inferred_operations}")
        return self._run_operations(
            task=task,
            document=document,
            operations=inferred_operations,
        )

    def _run_operations(
        self,
        *,
        task: str,
        document: DoclingDocument,
        operations: list[str],
    ) -> DoclingDocument:
        result = document
        for op_name in operations:
            method_name = _OP_ALIASES.get(op_name)
            if method_name is None:
                raise ValueError(f"Unknown operation: {op_name!r}")
            method = getattr(self, method_name)
            method_kwargs: dict[str, Any] = {"document": result}
            if method_name == "_detect_key_entities":
                method_kwargs["task"] = task
            result = method(**method_kwargs) or result
        return result

    def _choose_operations(self, *, task: str, loop_budget: int = 5) -> dict[str, Any]:
        logger.info(f"task: {task}")

        m = self._create_reasoning_session(system_prompt=self.system_prompt_for_enrichment_routing)

        answer = m.instruct(
            task,
            requirements=[
                Requirement(
                    description="Put the resulting function calls in the format ```json <insert-content>```",
                    validation_fn=simple_validate(has_json_dicts),
                ),
                Requirement(
                    description=(
                        "Select one or more operations and return only one JSON object "
                        "in a ```json ...``` block with 'operations' as a list containing only: "
                        + ", ".join(sorted(_ROUTING_OPS))
                    ),
                    validation_fn=simple_validate(DoclingEnrichingAgent._validate_json_plan),
                ),
            ],
            retry_budget=loop_budget,
        )

        view_linear_context(m)

        ops = find_json_dicts(text=answer)
        if not ops:
            raise ValueError("No routing operation detected in model response")
        if "operations" not in ops[0]:
            raise ValueError(f"`operations` not found in routing result: {ops[0]}")
        return ops[0]

    @staticmethod
    def _validate_json_plan(content: str) -> bool:
        ops = find_json_dicts(text=content)
        if len(ops) != 1:
            return False
        planned = ops[0].get("operations")
        return (
            isinstance(planned, list)
            and len(planned) >= 1
            and all(isinstance(op, str) and op in _ROUTING_OPS for op in planned)
        )

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_items(
        self,
        *,
        document: DoclingDocument,
        fix_heading_levels: bool = True,
        min_text_length: int = 40,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        if fix_heading_levels:
            logger.info("_summarize_items: fix heading levels")
            self._fix_heading_levels(document=document)

        logger.info("_summarize_items: make hierarchical document")
        hier_doc = make_hierarchical_document(document)

        m = self._create_extraction_session()

        logger.info("_summarize_items: summarize the document")
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

        editor = DoclingEditingAgent(
            backend=self.backend,
            tools=[],
        )
        editor.run(
            task=("Ensure the section headings have the correct level."),
            document=document,
        )

    def _walk_and_enrich(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: BaseSession,
        loop_budget: int,
        min_text_length: int,
        meta_attr: str,
        generate_fn: _GenerateFn,
        set_meta_fn: _SetMetaFn,
    ) -> None:
        """Generic method to walk document tree and enrich nodes with metadata."""
        should_enrich = False
        threshold = min_text_length

        if isinstance(node, TitleItem | SectionHeaderItem):
            should_enrich = True
        elif isinstance(node, TextItem):
            should_enrich = node.label != DocItemLabel.CAPTION
            threshold = min(min_text_length, 40)
        elif isinstance(node, GroupItem):
            should_enrich = node.self_ref != "#/body"
            threshold = min(min_text_length, 40)

        if should_enrich:
            if not (node.meta and hasattr(node.meta, meta_attr) and getattr(node.meta, meta_attr)):
                text = collect_subtree_text(node, doc)
                if len(text) >= threshold:
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
        m: BaseSession,
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
        m: BaseSession,
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
        m: BaseSession,
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
        m: BaseSession,
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
                requirements=[
                    Requirement(
                        description=requirement_description,
                        validation_fn=simple_validate(validation_fn),
                    ),
                ],
                retry_budget=loop_budget,
            )
            return answer.strip() or None
        except Exception as exc:
            logger.warning(f"Content generation failed: {exc}")
            return None

    def _generate_summary(
        self,
        *,
        m: BaseSession,
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
        m: BaseSession,
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

        m = self._create_reasoning_session()

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
        m: BaseSession,
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
        m: BaseSession,
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
        task: str | None = None,
        document: DoclingDocument,
        fix_heading_levels: bool = False,
        min_text_length: int = 100,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_detect_key_entities: starting")

        if fix_heading_levels:
            self._fix_heading_levels(document=document)

        hier_doc = make_hierarchical_document(document)

        m = self._create_reasoning_session()
        entity_targets = self._infer_entity_targets(
            m=m,
            task=task,
            loop_budget=loop_budget,
        )

        self._walk_and_extract_entities(
            node=hier_doc.body,
            doc=hier_doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
            entity_targets=entity_targets,
        )
        self._extract_entities_from_leaf_items(
            m=m,
            document=hier_doc,
            loop_budget=loop_budget,
            entity_targets=entity_targets,
        )

        return hier_doc

    def _infer_entity_targets(
        self,
        *,
        m: BaseSession,
        task: str | None,
        loop_budget: int,
    ) -> dict[str, Any] | None:
        if not task:
            return None

        def _validate_entity_target_spec(content: str) -> bool:
            matches = find_json_dicts(text=content)
            if len(matches) != 1:
                return False
            spec = matches[0]
            labels = spec.get("labels", [])
            focus_terms = spec.get("focus_terms", [])
            generic = spec.get("generic", False)
            return (
                isinstance(generic, bool)
                and isinstance(labels, list)
                and all(isinstance(label, str) for label in labels)
                and isinstance(focus_terms, list)
                and all(isinstance(term, str) for term in focus_terms)
            )

        answer = self._generate_content(
            m=m,
            text=task,
            task_prompt=(
                "You are interpreting a document enrichment request. "
                "Identify which entities the user actually wants extracted. "
                "Return one JSON object in a ```json ...``` block with this schema: "
                '{"generic": boolean, "labels": [string, ...], "focus_terms": [string, ...], "instructions": string}. '
                "Set generic=true only if the request does not narrow the entity scope beyond generic named entities."
            ),
            requirement_description=(
                "Return one JSON object in a ```json ...``` block with keys "
                "generic, labels, focus_terms, and instructions."
            ),
            validation_fn=_validate_entity_target_spec,
            loop_budget=loop_budget,
        )

        if not answer:
            return None

        specs = find_json_dicts(text=answer)
        return specs[0] if specs else None

    def _walk_and_extract_entities(
        self,
        *,
        node: NodeItem,
        doc: DoclingDocument,
        m: BaseSession,
        loop_budget: int,
        min_text_length: int,
        entity_targets: dict[str, Any] | None,
    ) -> None:
        """Walk document tree and add entities to sections."""

        def set_entities(meta: BaseMeta, result: Any) -> None:
            meta.entities = result

        def generate_entities(*, m: BaseSession, text: str, loop_budget: int):
            return self._generate_entities(
                m=m,
                text=text,
                loop_budget=loop_budget,
                entity_targets=entity_targets,
            )

        self._walk_and_enrich(
            node=node,
            doc=doc,
            m=m,
            loop_budget=loop_budget,
            min_text_length=min_text_length,
            meta_attr="entities",
            generate_fn=generate_entities,
            set_meta_fn=set_entities,
        )

    def _extract_entities_from_leaf_items(
        self,
        *,
        m: BaseSession,
        document: DoclingDocument,
        loop_budget: int,
        entity_targets: dict[str, Any] | None,
    ) -> None:
        """Add entities to leaf items (tables, pictures)."""

        def set_entities(meta: BaseMeta, result: Any) -> None:
            meta.entities = result

        def generate_entities(*, m: BaseSession, text: str, loop_budget: int):
            return self._generate_entities(
                m=m,
                text=text,
                loop_budget=loop_budget,
                entity_targets=entity_targets,
            )

        self._enrich_leaf_items(
            m=m,
            document=document,
            loop_budget=loop_budget,
            meta_attr="entities",
            generate_fn=generate_entities,
            set_meta_fn=set_entities,
        )

    def _generate_entities(
        self,
        *,
        m: BaseSession,
        text: str,
        entity_targets: dict[str, Any] | None = None,
        loop_budget: int = 5,
    ) -> EntitiesMetaField | None:
        def _validate_entities(content: str) -> bool:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if not match:
                return False
            try:
                val = json.loads(match.group(1))
                if not isinstance(val, list):
                    return False
                return all(isinstance(item, dict) and "text" in item for item in val)
            except Exception:
                return False

        target_clause = ""
        if entity_targets:
            labels = entity_targets.get("labels", [])
            focus_terms = entity_targets.get("focus_terms", [])
            instructions = entity_targets.get("instructions", "")
            generic = entity_targets.get("generic", False)
            if not generic or labels or focus_terms or instructions:
                target_clause = (
                    "\nFocus only on entities relevant to this query-derived extraction brief:\n"
                    f"- labels: {labels}\n"
                    f"- focus_terms: {focus_terms}\n"
                    f"- instructions: {instructions}\n"
                    "If an entity is not relevant to that brief, omit it."
                )

        result = self._generate_content(
            m=m,
            text=text,
            task_prompt=(
                "Extract named entities from the following content. "
                "For each entity, identify the exact mention text and its label/category. "
                "Return them as a JSON array of objects with keys 'text', 'label', and optional 'original' in a ```json ...``` block. "
                "Only include significant entities, avoid generic terms. "
                "Do not repeat the same entity even if it appears multiple times."
                f"{target_clause}"
            ),
            requirement_description="Return entities as a JSON array of objects with keys 'text', 'label', and optional 'original' in a ```json ...``` block.",
            validation_fn=_validate_entities,
            loop_budget=loop_budget,
        )

        if result:
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    mentions = [
                        EntityMention(
                            text=str(item["text"]).strip(),
                            original=str(item["original"]).strip() if item.get("original") else None,
                            label=str(item["label"]).strip() if item.get("label") else None,
                        )
                        for item in payload
                        if isinstance(item, dict) and str(item.get("text", "")).strip()
                    ]
                    if mentions:
                        return EntitiesMetaField(mentions=mentions)
                except Exception as exc:
                    logger.warning(f"Failed to parse entities JSON: {exc}")
        return None

    # ------------------------------------------------------------------
    # Picture Classification
    # ------------------------------------------------------------------

    def _classify_items(
        self,
        *,
        document: DoclingDocument,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        logger.info("_classify_items: starting")

        for item, _ in document.iterate_items():
            if not isinstance(item, PictureItem):
                continue

            text = self._picture_context(item=item, document=document)
            if not text:
                logger.info("Skipping picture classification without textual context")
                continue

            meta = self._ensure_picture_meta(item)

            classification = self._classify_picture_context(text=text, loop_budget=loop_budget)
            if classification is not None:
                meta.classification = classification

            chart_type = self._primary_chart_type(classification)
            chart_meta: TabularChartMetaField | None = None
            if chart_type is not None:
                chart_meta = self._extract_tabular_chart(text=text, chart_type=chart_type, loop_budget=loop_budget)
                if chart_meta is not None:
                    meta.tabular_chart = chart_meta

            code_meta = self._generate_picture_code(
                text=text,
                classification=classification,
                chart_meta=chart_meta,
                loop_budget=loop_budget,
            )
            if code_meta is not None:
                meta.code = code_meta

        return document

    def _metadata_origin(self) -> str:
        return f"docling-agent:{self.backend.backend_type}:{self.get_reasoning_model_id()}"

    def _ensure_picture_meta(self, item: PictureItem) -> PictureMeta:
        if isinstance(item.meta, PictureMeta):
            return item.meta

        summary = item.meta.summary if item.meta and getattr(item.meta, "summary", None) else None
        item.meta = PictureMeta(summary=summary)
        return item.meta

    def _picture_context(self, *, item: PictureItem, document: DoclingDocument) -> str:
        parts: list[str] = []

        if item.meta and item.meta.summary and item.meta.summary.text:
            parts.append(f"Summary: {item.meta.summary.text}")

        captions = [c.resolve(document).text for c in item.captions if hasattr(c.resolve(document), "text")]
        if captions:
            parts.append("Captions:\n" + "\n".join(f"- {caption}" for caption in captions))

        return "\n\n".join(part for part in parts if part).strip()

    def _classify_picture_context(
        self,
        *,
        text: str,
        loop_budget: int,
    ) -> PictureClassificationMetaField | None:
        def _validate(content: str) -> bool:
            payload = self._extract_single_json_dict(content)
            if payload is None:
                return False
            predictions = payload.get("predictions")
            return isinstance(predictions, list) and all(
                isinstance(label, str) and label in self._PICTURE_CLASS_NAMES for label in predictions
            )

        m = self._create_reasoning_session()
        answer = m.instruct(
            (
                "Classify the following picture description.\n\n"
                f"{text}\n\n"
                "Return a JSON object in a ```json``` block with the form:\n"
                '{"predictions": ["<class-name>", "..."]}\n'
                "Choose one to three labels from this closed set only:\n" + ", ".join(self._PICTURE_CLASS_NAMES)
            ),
            requirements=[
                Requirement(
                    description="Return a valid picture-classification JSON object in a ```json``` block.",
                    validation_fn=simple_validate(_validate),
                ),
            ],
            retry_budget=loop_budget,
        )

        payload = self._extract_single_json_dict(answer)
        if payload is None:
            return None

        labels = payload.get("predictions")
        if not isinstance(labels, list):
            return None

        deduped = [label for label in dict.fromkeys(labels) if isinstance(label, str)]
        if not deduped:
            return None

        return PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(class_name=label, created_by=self._metadata_origin())
                for label in deduped
            ]
        )

    def _primary_chart_type(self, classification: PictureClassificationMetaField | None) -> str | None:
        if classification is None:
            return None

        for prediction in classification.predictions:
            if prediction.class_name in self._SUPPORTED_CHART_TYPES:
                return prediction.class_name
        return None

    def _extract_tabular_chart(
        self,
        *,
        text: str,
        chart_type: str,
        loop_budget: int,
    ) -> TabularChartMetaField | None:
        def _validate(content: str) -> bool:
            payload = self._extract_single_json_dict(content)
            return self._is_valid_chart_payload(payload)

        m = self._create_reasoning_session()
        answer = m.instruct(
            (
                f"The following picture is classified as a {chart_type}.\n\n"
                f"{text}\n\n"
                "Extract its chart data and return a JSON object in a ```json``` block with the form:\n"
                "{"
                '"title": "<chart title or null>", '
                '"columns": ["<column-1>", "<column-2>", "..."], '
                '"rows": [["<value>", "<value>", "..."], ["<value>", "<value>", "..."]]'
                "}"
            ),
            requirements=[
                Requirement(
                    description="Return valid chart-data JSON with title, columns, and rows.",
                    validation_fn=simple_validate(_validate),
                ),
            ],
            retry_budget=loop_budget,
        )

        payload = self._extract_single_json_dict(answer)
        if not self._is_valid_chart_payload(payload):
            return None

        payload_dict = cast(dict[str, Any], payload)
        columns = cast(list[str], payload_dict["columns"])
        rows = cast(list[list[str | int | float]], payload_dict["rows"])
        title = payload_dict.get("title")

        return TabularChartMetaField(
            title=title if isinstance(title, str) else None,
            chart_data=self._table_data_from_rows(columns=columns, rows=rows),
            created_by=self._metadata_origin(),
        )

    def _generate_picture_code(
        self,
        *,
        text: str,
        classification: PictureClassificationMetaField | None,
        chart_meta: TabularChartMetaField | None,
        loop_budget: int,
    ) -> CodeMetaField | None:
        def _validate(content: str) -> bool:
            return self._extract_python_code_block(content) is not None

        labels = ", ".join(pred.class_name for pred in classification.predictions) if classification else "unknown"
        chart_payload = ""
        if chart_meta is not None:
            chart_payload = f"\n\nChart data:\n```json\n{chart_meta.chart_data.model_dump_json(indent=2)}\n```"

        m = self._create_reasoning_session()
        answer = m.instruct(
            (
                "Write Python code that recreates the described picture as closely as possible. "
                "Prefer matplotlib. Return only a ```python``` code block.\n\n"
                f"Picture classes: {labels}\n\n"
                f"Picture description:\n{text}"
                f"{chart_payload}"
            ),
            requirements=[
                Requirement(
                    description="Return only a Python code block fenced with ```python```.",
                    validation_fn=simple_validate(_validate),
                ),
            ],
            retry_budget=loop_budget,
        )

        code = self._extract_python_code_block(answer)
        if code is None:
            return None

        return CodeMetaField(
            text=code,
            language=CodeLanguageLabel("Python"),
            created_by=self._metadata_origin(),
        )

    def _extract_single_json_dict(self, text: str) -> dict[str, Any] | None:
        payloads = find_json_dicts(text=text)
        if len(payloads) != 1:
            return None
        payload = payloads[0]
        return payload if isinstance(payload, dict) else None

    def _extract_python_code_block(self, text: str) -> str | None:
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            return None
        code = match.group(1).strip()
        return code or None

    def _is_valid_chart_payload(self, payload: dict[str, Any] | None) -> bool:
        if payload is None:
            return False
        columns = payload.get("columns")
        rows = payload.get("rows")
        if not isinstance(columns, list) or len(columns) < 2 or not all(isinstance(col, str) for col in columns):
            return False
        if not isinstance(rows, list) or not rows:
            return False
        return all(
            isinstance(row, list)
            and len(row) == len(columns)
            and all(isinstance(value, str | int | float) for value in row)
            for row in rows
        )

    def _table_data_from_rows(self, *, columns: list[str], rows: list[list[str | int | float]]) -> TableData:
        table_cells: list[dict[str, Any]] = []

        for col_idx, value in enumerate(columns):
            table_cells.append(
                {
                    "start_row_offset_idx": 0,
                    "end_row_offset_idx": 1,
                    "start_col_offset_idx": col_idx,
                    "end_col_offset_idx": col_idx + 1,
                    "text": value,
                    "column_header": True,
                }
            )

        for row_idx, row in enumerate(rows, start=1):
            for col_idx, cell_value in enumerate(row):
                table_cells.append(
                    {
                        "start_row_offset_idx": row_idx,
                        "end_row_offset_idx": row_idx + 1,
                        "start_col_offset_idx": col_idx,
                        "end_col_offset_idx": col_idx + 1,
                        "text": str(cell_value),
                    }
                )

        return TableData.model_validate(
            {
                "table_cells": table_cells,
                "num_rows": len(rows) + 1,
                "num_cols": len(columns),
            }
        )

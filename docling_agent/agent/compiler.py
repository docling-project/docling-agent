from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from docling_core.experimental.serializer.outline import OutlineFormat, OutlineMode
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import DocItemLabel, DoclingDocument
from mellea.stdlib.requirements import Requirement, simple_validate
from pydantic import BaseModel
from typing_extensions import override

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import create_document_outline
from docling_agent.agent.library import (
    DocCompileArtifact,
    DocCompileEntityRow,
    DocCompileRelationRow,
)
from docling_agent.logging import log_info, log_warning

CompileSubtask = Literal["summarize", "outline", "topics", "entities"]

TERM_REVIEW_CATEGORIES = {
    "concept",
    "person",
    "organization",
    "location",
    "material",
    "unit",
    "product",
    "date",
    "method",
    "property",
    "metric",
    "formula",
    "dataset",
    "model",
    "regulation",
    "generic",
    "unknown",
}


class _NoOpProgress:
    def __enter__(self) -> _NoOpProgress:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def add_task(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class CompileContext(BaseModel):
    """Library context needed to create stable compile rows."""

    doc_id: str
    project_id: str


class DeepSearchGLMProvider(Protocol):
    """Minimal interface for fast NLP providers used by compile mode."""

    source_model: str

    def apply_on_document(self, document: DoclingDocument) -> dict[str, Any]: ...


class LazyDeepSearchGLMProvider:
    """Lazy adapter around deepsearch-glm."""

    def __init__(self, *, model_names: str, force_download: bool = False) -> None:
        self.model_names = model_names
        self.source_model = model_names
        self.force_download = force_download
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from deepsearch_glm.nlp_utils import init_nlp_model  # type: ignore[import-not-found,import-untyped]
            from deepsearch_glm.utils.load_pretrained_models import (  # type: ignore[import-not-found,import-untyped]
                load_pretrained_nlp_models,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Compile NLP provider 'deepsearch-glm' is not installed. "
                "Install with docling-agent[compile]."
            ) from exc

        load_pretrained_nlp_models(force=self.force_download, verbose=False)
        self._model = init_nlp_model(model_names=self.model_names)
        return self._model

    def apply_on_document(self, document: DoclingDocument) -> dict[str, Any]:
        model = self._load_model()
        if hasattr(model, "apply_on_doc"):
            doc_dict = docling_document_to_glm_document(document)
            result = model.apply_on_doc(doc_dict)
            if isinstance(result, dict) and result.get("model-application", {}).get("success") is False:
                text = MarkdownDocSerializer(doc=document).serialize().text
                result = model.apply_on_text(text)
        else:
            text = MarkdownDocSerializer(doc=document).serialize().text
            result = model.apply_on_text(text)
        return result if isinstance(result, dict) else {"result": result}


def docling_document_to_glm_document(document: DoclingDocument) -> dict[str, Any]:
    """Convert a DoclingDocument to the GLM JSON shape expected by deepsearch-glm."""
    document_hash = _stable_document_hash(document)
    main_text: list[dict[str, Any]] = []
    page_dimensions = _page_dimensions(document)

    for item, _ in document.iterate_items():
        text = getattr(item, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        name, type_ = _glm_labels_for_item(item)
        prov = _glm_prov_for_item(item, text)
        main_text.append(
            {
                "text": text,
                "name": name,
                "type": type_,
                "prov": [prov],
                "docling_ref": str(item.self_ref),
            }
        )

    return {
        "file-info": {
            "document-hash": document_hash,
            "filename": document.name,
        },
        "page-dimensions": page_dimensions,
        "main-text": main_text,
    }


def _stable_document_hash(document: DoclingDocument) -> str:
    payload = f"{document.name}:{len(document.texts)}:{document.version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _page_dimensions(document: DoclingDocument) -> list[dict[str, float | int]]:
    dimensions: list[dict[str, float | int]] = []
    pages = getattr(document, "pages", None)
    if isinstance(pages, dict):
        for page_no, page in sorted(pages.items()):
            size = getattr(page, "size", None)
            width = float(getattr(size, "width", 1.0) or 1.0)
            height = float(getattr(size, "height", 1.0) or 1.0)
            dimensions.append({"page": int(page_no), "width": width, "height": height})
    return dimensions or [{"page": 1, "width": 1.0, "height": 1.0}]


def _glm_labels_for_item(item: Any) -> tuple[str, str]:
    label = getattr(item, "label", None)
    value = getattr(label, "value", label)
    mapping = {
        DocItemLabel.TITLE.value: ("title", "title"),
        DocItemLabel.SECTION_HEADER.value: ("section-header", "subtitle-level-1"),
        DocItemLabel.TEXT.value: ("paragraph", "paragraph"),
        DocItemLabel.PARAGRAPH.value: ("paragraph", "paragraph"),
        DocItemLabel.CAPTION.value: ("caption", "caption"),
        DocItemLabel.LIST_ITEM.value: ("list-item", "paragraph"),
        DocItemLabel.PAGE_HEADER.value: ("page-header", "page-header"),
        DocItemLabel.PAGE_FOOTER.value: ("page-footer", "page-footer"),
        DocItemLabel.FOOTNOTE.value: ("footnote", "footnote"),
        DocItemLabel.CODE.value: ("code", "paragraph"),
        DocItemLabel.FORMULA.value: ("formula", "equation"),
    }
    return mapping.get(str(value), ("paragraph", "paragraph"))


def _glm_prov_for_item(item: Any, text: str) -> dict[str, Any]:
    provenance = getattr(item, "prov", None) or []
    if provenance:
        first = provenance[0]
        bbox = getattr(first, "bbox", None)
        return {
            "page": int(getattr(first, "page_no", 1) or 1),
            "span": [0, len(text)],
            "bbox": [
                float(getattr(bbox, "l", 0.0) or 0.0),
                float(getattr(bbox, "t", 0.0) or 0.0),
                float(getattr(bbox, "r", 1.0) or 1.0),
                float(getattr(bbox, "b", 1.0) or 1.0),
            ],
        }
    return {"page": 1, "span": [0, len(text)], "bbox": [0.0, 0.0, 1.0, 1.0]}


class DoclingCompilerAgent(BaseDoclingAgent):
    """Compile source documents into summary, outline, entities, and relations."""

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
        nlp_provider: DeepSearchGLMProvider | None = None,
        nlp_model_names: str = "language;term",
    ) -> None:
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_COMPILER,
            backend=backend or self.default_backend(),
            tools=tools,
        )
        self.nlp_provider = nlp_provider or LazyDeepSearchGLMProvider(model_names=nlp_model_names)
        self.nlp_model_names = nlp_model_names

    @override
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs: Any,
    ) -> DoclingDocument:
        raise NotImplementedError("Use compile_document() instead.")

    def compile_document(
        self,
        *,
        document: DoclingDocument,
        context: CompileContext,
        subtasks: list[CompileSubtask],
        llm_review_terms: bool = False,
        llm_review_batch_size: int = 80,
    ) -> DocCompileArtifact:
        log_info(f"Compiling {document.name!r}", subtasks=subtasks)
        requested = set(subtasks)
        raw_nlp: dict[str, Any] | None = None
        entity_rows: list[DocCompileEntityRow] = []
        relation_rows: list[DocCompileRelationRow] = []

        if requested & {"entities", "topics"}:
            raw_nlp = self.nlp_provider.apply_on_document(document)
            document_term_rows = self._entity_rows_from_nlp(
                document=document,
                context=context,
                raw_nlp=raw_nlp,
                source_model=self.nlp_provider.source_model,
            )
            if llm_review_terms:
                document_term_rows = self._review_term_rows_with_llm(
                    document=document,
                    rows=document_term_rows,
                    batch_size=llm_review_batch_size,
                )
            term_rows = self._canonical_term_rows(context=context, document_term_rows=document_term_rows)
            entity_rows = [*document_term_rows, *term_rows]
            relation_rows = self._relation_rows_from_entities(context=context, entity_rows=entity_rows)

        concepts = self._concepts_from_entities(entity_rows)
        topics = self._topics_from_entities(entity_rows) if "topics" in requested else []
        summary = self._summarize_document(document) if "summarize" in requested else None
        outline = (
            create_document_outline(
                doc=document,
                mode=OutlineMode.OUTLINE,
                format=OutlineFormat.JSON,
            )
            if "outline" in requested
            else None
        )

        return DocCompileArtifact(
            summary=summary,
            outline=outline,
            topics=topics,
            concepts=concepts,
            entities=entity_rows if "entities" in requested else [],
            relations=relation_rows if "entities" in requested else [],
            raw_nlp=raw_nlp,
        )

    def _summarize_document(self, document: DoclingDocument, *, max_chars: int = 12000) -> str | None:
        text = MarkdownDocSerializer(doc=document).serialize().text[:max_chars]
        if not text.strip():
            return None

        session = self._create_writing_session()

        def _valid(content: str) -> bool:
            return bool(content.strip())

        try:
            return session.instruct(
                (
                    "Write a concise 2-3 sentence summary of the document. "
                    "Return plain text only.\n\n"
                    f"{text}"
                ),
                requirements=[
                    Requirement(
                        description="Return a concise plain-text document summary.",
                        validation_fn=simple_validate(_valid),
                    )
                ],
                retry_budget=3,
            ).strip()
        except Exception as exc:
            log_warning("Compile summary generation failed", exception=exc)
            return None

    def _review_term_rows_with_llm(
        self,
        *,
        document: DoclingDocument,
        rows: list[DocCompileEntityRow],
        batch_size: int,
    ) -> list[DocCompileEntityRow]:
        term_rows = [row for row in rows if row.kind in {"document-term", "concept"}]
        if not term_rows:
            return rows

        grouped: dict[str, list[DocCompileEntityRow]] = {}
        for row in term_rows:
            normalized = self._normalize_term(row.normalized_text)
            if normalized:
                grouped.setdefault(normalized, []).append(row)

        if not grouped:
            return rows

        examples_by_term = self._term_examples(document=document, terms=list(grouped))
        review_by_term: dict[str, dict[str, str]] = {}
        terms = sorted(grouped, key=lambda term: (-len(grouped[term]), term))
        batch_size = max(1, batch_size)
        batch_count = (len(terms) + batch_size - 1) // batch_size

        progress = self._term_review_progress(total=len(terms), document_name=document.name)
        with progress:
            task_id = progress.add_task("Reviewing terms", total=len(terms))
            for offset in range(0, len(terms), batch_size):
                batch_index = offset // batch_size + 1
                batch_terms = terms[offset : offset + batch_size]
                payload = [
                    {
                        "id": f"t{index}",
                        "term": term,
                        "count": len(grouped[term]),
                        "source_labels": sorted({row.label for row in grouped[term] if row.label}),
                        "examples": examples_by_term.get(term, [])[:2],
                    }
                    for index, term in enumerate(batch_terms)
                ]

                session_started = perf_counter()
                session = self._create_extraction_session()
                session_elapsed = perf_counter() - session_started

                query_started = perf_counter()
                answer = session.instruct(
                    self._term_review_prompt(payload),
                    requirements=[
                        Requirement(
                            description=(
                                "Return a JSON object with key 'terms' containing review objects "
                                "with id, decision, canonical, category, and optional importance."
                            ),
                            validation_fn=simple_validate(self._is_valid_term_review_response),
                        )
                    ],
                    retry_budget=3,
                )
                query_elapsed = perf_counter() - query_started
                log_info(
                    "Compile LLM term review batch "
                    f"{batch_index}/{batch_count}: terms={len(batch_terms)}, "
                    f"session_create_s={session_elapsed:.3f}, query_s={query_elapsed:.3f}"
                )
                progress.update(task_id, advance=len(batch_terms))
                parsed = self._parse_term_review_response(answer)
                if parsed is None:
                    log_warning("Compile term review failed; keeping original terms for this batch")
                    continue
                for item in parsed:
                    term_id = item.get("id")
                    if not isinstance(term_id, str) or not term_id.startswith("t"):
                        continue
                    try:
                        term = batch_terms[int(term_id[1:])]
                    except (IndexError, ValueError):
                        continue
                    review_by_term[term] = {
                        "decision": str(item.get("decision") or "keep").strip().casefold(),
                        "canonical": str(item.get("canonical") or grouped[term][0].normalized_text).strip(),
                        "category": self._normalize_term_review_category(item.get("category")),
                        "importance": str(item.get("importance") or "").strip(),
                    }

        reviewed_rows: list[DocCompileEntityRow] = []
        for row in rows:
            if row.kind not in {"document-term", "concept"}:
                reviewed_rows.append(row)
                continue
            term_key = self._normalize_term(row.normalized_text)
            review = review_by_term.get(term_key)
            if review is None:
                reviewed_rows.append(row)
                continue
            if review["decision"] == "drop":
                continue
            row.normalized_text = review["canonical"] or row.normalized_text
            row.label = review["category"]
            row.source_model = self._reviewed_source_model(row.source_model)
            reviewed_rows.append(row)
        return reviewed_rows

    def _term_review_prompt(self, candidates: list[dict[str, Any]]) -> str:
        result = (
            "Review candidate document terms extracted from a source document.\n"
            "For each candidate, decide whether it is a meaningful entity, term, or concept.\n"
            "Drop wrong OCR fragments, boilerplate, overly generic words, and non-concepts.\n"
            "Keep domain concepts, named entities, materials, methods, units, properties, products, "
            "companies, people, locations, dates, models, datasets, formulas, metrics, and regulations.\n"
            f"Use exactly one category from this list: {sorted(TERM_REVIEW_CATEGORIES)}.\n"
            "Return only JSON with this shape: "
            '{"terms":[{"id":"t0","decision":"keep|drop","canonical":"...","category":"concept",'
            '"importance":"core|supporting|incidental"}]}.\n\n'
            f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
        )
        # print(result)

        return result

    def _term_review_progress(self, *, total: int, document_name: str):
        try:
            from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn, TimeElapsedColumn
            from rich.text import Text
        except ImportError:
            return _NoOpProgress()

        class TermsPerSecondColumn(ProgressColumn):
            def render(self, task: Task) -> Text:
                speed = task.speed
                return Text("-- terms/s" if speed is None else f"{speed:.2f} terms/s")

        return Progress(
            TextColumn(f"[bold]LLM term review[/bold] {document_name}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} terms"),
            TermsPerSecondColumn(),
            TimeElapsedColumn(),
            transient=True,
            disable=total <= 0,
        )

    def _is_valid_term_review_response(self, content: str) -> bool:
        return self._parse_term_review_response(content) is not None

    def _parse_term_review_response(self, content: str) -> list[dict[str, Any]] | None:
        payload = self._extract_json_object(content)
        if not isinstance(payload, dict):
            return None
        terms = payload.get("terms")
        if not isinstance(terms, list):
            return None
        for item in terms:
            if not isinstance(item, dict):
                return None
            if not isinstance(item.get("id"), str):
                return None
            decision = str(item.get("decision") or "").casefold()
            if decision not in {"keep", "drop"}:
                return None
        return terms

    def _extract_json_object(self, content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            marker = "```json"
            if text.startswith(marker):
                text = text[len(marker) :]
            else:
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    def _normalize_term_review_category(self, value: Any) -> str:
        category = str(value or "unknown").strip().casefold().replace(" ", "_")
        return category if category in TERM_REVIEW_CATEGORIES else "unknown"

    def _reviewed_source_model(self, source_model: str | None) -> str:
        reviewer = f"llm-review:{self.backend.backend_type}:{self.get_extraction_model_id()}"
        return f"{source_model}+{reviewer}" if source_model else reviewer

    def _term_examples(self, *, document: DoclingDocument, terms: list[str], max_examples: int = 2) -> dict[str, list[str]]:
        examples: dict[str, list[str]] = {term: [] for term in terms}
        pending = set(terms)
        for _, text in self._text_item_refs(document):
            lowered = text.casefold()
            for term in list(pending):
                if term not in lowered:
                    continue
                examples[term].append(self._trim_example(text, term))
                if len(examples[term]) >= max_examples:
                    pending.discard(term)
            if not pending:
                break
        return examples

    def _trim_example(self, text: str, term: str, *, radius: int = 120) -> str:
        lowered = text.casefold()
        start = lowered.find(term)
        if start < 0:
            return text[: radius * 2].strip()
        left = max(0, start - radius)
        right = min(len(text), start + len(term) + radius)
        return text[left:right].strip()

    def _entity_rows_from_nlp(
        self,
        *,
        document: DoclingDocument,
        context: CompileContext,
        raw_nlp: dict[str, Any],
        source_model: str,
    ) -> list[DocCompileEntityRow]:
        rows: list[DocCompileEntityRow] = []
        for instance in self._instances_from_table(raw_nlp):
            row = self._entity_row_from_instance(
                context=context,
                instance=instance,
                source_model=source_model,
                page_by_path=self._page_by_subj_path(raw_nlp),
            )
            if row is not None:
                rows.append(row)

        if rows:
            deduped: dict[int, DocCompileEntityRow] = {}
            for row in rows:
                deduped.setdefault(row.entity_hash, row)
            return list(deduped.values())

        item_refs = self._text_item_refs(document)
        default_xpath = item_refs[0][0] if item_refs else "/document"

        for mention in self._iter_mentions(raw_nlp):
            entity_text = self._first_str(mention, ("text", "entity_text", "term", "word", "name", "mention"))
            if not entity_text:
                continue
            normalized_text = self._first_str(mention, ("normalized_text", "normalized", "canonical", "lemma"))
            label = self._first_str(mention, ("label", "type", "entity_type", "tag"))
            kind = self._kind_for_label(label=label, mention=mention)
            xpath = self._first_str(mention, ("xpath", "path", "self_ref")) or self._xpath_for_text(
                entity_text,
                item_refs,
                default_xpath,
            )
            char_start = self._first_int(mention, ("char_start", "start", "start_char", "begin"))
            char_end = self._first_int(mention, ("char_end", "end", "end_char"))
            page_no = self._first_int(mention, ("page_no", "page", "page_number"))
            confidence = self._first_float(mention, ("confidence", "score", "probability"))

            normalized = normalized_text or entity_text
            row = DocCompileEntityRow(
                doc_id=context.doc_id,
                project_id=context.project_id,
                xpath=xpath,
                entity_hash=self._entity_hash(
                    doc_id=context.doc_id,
                    xpath=xpath,
                    normalized_text=normalized,
                    label=label,
                    kind=kind,
                    char_start=char_start,
                    char_end=char_end,
                ),
                entity_text=entity_text,
                normalized_text=normalized,
                label=label,
                kind=kind,
                source_model=source_model,
                confidence=confidence,
                page_no=page_no,
                char_start=char_start,
                char_end=char_end,
            )
            rows.append(row)

        deduped = {}
        for row in rows:
            deduped.setdefault(row.entity_hash, row)
        return list(deduped.values())

    def _instances_from_table(self, raw_nlp: dict[str, Any]) -> list[dict[str, Any]]:
        instances = raw_nlp.get("instances")
        if not isinstance(instances, dict):
            return []
        headers = instances.get("headers")
        data = instances.get("data")
        if not isinstance(headers, list) or not isinstance(data, list):
            return []
        rows: list[dict[str, Any]] = []
        for values in data:
            if isinstance(values, list):
                rows.append(dict(zip(headers, values, strict=False)))
        return rows

    def _entity_row_from_instance(
        self,
        *,
        context: CompileContext,
        instance: dict[str, Any],
        source_model: str,
        page_by_path: dict[str, int],
    ) -> DocCompileEntityRow | None:
        instance_type = self._string_value(instance.get("type"))
        if not instance_type or instance_type == "sentence":
            return None
        entity_text = self._string_value(instance.get("name")) or self._string_value(instance.get("original"))
        if not entity_text:
            return None
        normalized = self._string_value(instance.get("original")) or entity_text
        xpath = self._self_ref_to_xpath(self._string_value(instance.get("subj_path")) or "#")
        char_start = self._int_value(instance.get("char_i"))
        char_end = self._int_value(instance.get("char_j"))
        confidence = self._float_value(instance.get("conf"))
        label = self._string_value(instance.get("subtype"))
        kind = self._kind_for_label(label=instance_type, mention={"kind": instance_type})
        return DocCompileEntityRow(
            doc_id=context.doc_id,
            project_id=context.project_id,
            xpath=xpath,
            entity_hash=self._entity_hash(
                doc_id=context.doc_id,
                xpath=xpath,
                normalized_text=normalized,
                label=label,
                kind=kind,
                char_start=char_start,
                char_end=char_end,
            ),
            entity_text=entity_text,
            normalized_text=normalized,
            label=label,
            kind=kind,
            source_model=source_model,
            confidence=confidence,
            page_no=page_by_path.get(self._string_value(instance.get("subj_path")) or ""),
            char_start=char_start,
            char_end=char_end,
        )

    def _page_by_subj_path(self, raw_nlp: dict[str, Any]) -> dict[str, int]:
        page_by_element_ref: dict[str, int] = {}
        page_elements = raw_nlp.get("page-elements")
        if isinstance(page_elements, list):
            for index, element in enumerate(page_elements):
                if not isinstance(element, dict):
                    continue
                page = self._int_value(element.get("page"))
                if page is not None:
                    page_by_element_ref[f"#/page-elements/{index}"] = page

        result: dict[str, int] = {}
        texts = raw_nlp.get("texts")
        if isinstance(texts, list):
            for text in texts:
                if not isinstance(text, dict):
                    continue
                sref = self._string_value(text.get("sref"))
                prov = text.get("prov")
                if not sref or not isinstance(prov, list) or not prov:
                    continue
                first = prov[0]
                if not isinstance(first, dict):
                    continue
                page_ref = self._string_value(first.get("$ref"))
                if page_ref and page_ref in page_by_element_ref:
                    result[sref] = page_by_element_ref[page_ref]
        return result

    def _relation_rows_from_entities(
        self,
        *,
        context: CompileContext,
        entity_rows: list[DocCompileEntityRow],
    ) -> list[DocCompileRelationRow]:
        counts: Counter[tuple[int, int, str]] = Counter()

        terms_by_normalized = {
            self._normalize_term(row.normalized_text): row
            for row in entity_rows
            if row.kind == "term" and self._normalize_term(row.normalized_text)
        }
        for child_key, child in terms_by_normalized.items():
            for parent_key in self._term_parent_candidates(child_key):
                parent = terms_by_normalized.get(parent_key)
                if parent is None:
                    continue
                counts[(parent.entity_hash, child.entity_hash, "sub-term")] += 1
                counts[(child.entity_hash, parent.entity_hash, "super-term")] += 1
                break

        return [
            DocCompileRelationRow(
                doc_id=context.doc_id,
                project_id=context.project_id,
                entity_hash_i=left,
                entity_hash_j=right,
                relation_k=relation,
                count=count,
            )
            for (left, right, relation), count in sorted(counts.items())
        ]

    def _iter_mentions(self, value: Any) -> list[dict[str, Any]]:
        mentions: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if any(key in value for key in ("text", "entity_text", "term", "word", "name", "mention")):
                mentions.append(value)
            for child in value.values():
                mentions.extend(self._iter_mentions(child))
        elif isinstance(value, list):
            for child in value:
                mentions.extend(self._iter_mentions(child))
        return mentions

    def _text_item_refs(self, document: DoclingDocument) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        for item, _ in document.iterate_items():
            text = getattr(item, "text", None)
            if not isinstance(text, str) or not text:
                continue
            refs.append((self._self_ref_to_xpath(str(item.self_ref)), text))
        return refs

    def _xpath_for_text(
        self,
        entity_text: str,
        item_refs: list[tuple[str, str]],
        default_xpath: str,
    ) -> str:
        needle = entity_text.lower()
        for xpath, text in item_refs:
            if needle in text.lower():
                return xpath
        return default_xpath

    def _concepts_from_entities(self, rows: list[DocCompileEntityRow]) -> list[str]:
        terms = [row for row in rows if row.kind == "term" and row.normalized_text.strip()]
        terms.sort(key=lambda row: (-row.count, row.normalized_text.casefold()))
        return [row.normalized_text for row in terms[:25]]

    def _topics_from_entities(self, rows: list[DocCompileEntityRow]) -> list[str]:
        values = [row.normalized_text for row in rows if row.normalized_text.strip()]
        return [value for value, _ in Counter(values).most_common(10)]

    def _kind_for_label(self, *, label: str | None, mention: dict[str, Any]) -> str:
        explicit = self._first_str(mention, ("kind", "category"))
        if explicit:
            normalized_explicit = explicit.lower()
            if normalized_explicit in {"term", "keyterm", "nounphrase", "noun_phrase"}:
                return "document-term"
            return normalized_explicit
        if not label:
            return "unknown"
        normalized = label.lower()
        if normalized in {"term", "keyterm", "nounphrase", "noun_phrase", "concept"}:
            return "concept" if normalized == "concept" else "document-term"
        if normalized in {"topic"}:
            return "topic"
        return "entity"

    def _canonical_term_rows(
        self,
        *,
        context: CompileContext,
        document_term_rows: list[DocCompileEntityRow],
    ) -> list[DocCompileEntityRow]:
        grouped: dict[str, list[DocCompileEntityRow]] = {}
        for row in document_term_rows:
            if row.kind not in {"document-term", "concept"}:
                continue
            normalized = self._normalize_term(row.normalized_text)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(row)

        rows: list[DocCompileEntityRow] = []
        for normalized, mentions in sorted(grouped.items()):
            display_text = Counter(row.normalized_text for row in mentions).most_common(1)[0][0]
            rows.append(
                DocCompileEntityRow(
                    doc_id=context.doc_id,
                    project_id=context.project_id,
                    xpath=None,
                    entity_hash=self._term_hash(doc_id=context.doc_id, normalized_text=normalized),
                    entity_text=display_text,
                    normalized_text=display_text,
                    count=len(mentions),
                    label="canonical",
                    kind="term",
                    source_model=mentions[0].source_model,
                    confidence=None,
                    page_no=None,
                    char_start=None,
                    char_end=None,
                )
            )
        return rows

    def _term_parent_candidates(self, normalized_term: str) -> list[str]:
        tokens = normalized_term.split()
        candidates = [" ".join(tokens[index:]) for index in range(1, len(tokens))]
        return [candidate for candidate in candidates if candidate]

    def _normalize_term(self, term: str) -> str:
        return " ".join(term.casefold().split())

    def _term_hash(self, *, doc_id: str, normalized_text: str) -> int:
        payload = json.dumps(
            {
                "doc_id": doc_id,
                "normalized_text": normalized_text,
                "kind": "term",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

    def _entity_hash(
        self,
        *,
        doc_id: str,
        xpath: str,
        normalized_text: str,
        label: str | None,
        kind: str,
        char_start: int | None,
        char_end: int | None,
    ) -> int:
        payload = json.dumps(
            {
                "doc_id": doc_id,
                "xpath": xpath,
                "normalized_text": normalized_text,
                "label": label,
                "kind": kind,
                "char_start": char_start,
                "char_end": char_end,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

    def _self_ref_to_xpath(self, self_ref: str) -> str:
        if self_ref.startswith("#/"):
            return "/" + self_ref[2:]
        if self_ref == "#":
            return "/"
        return self_ref

    def _first_str(self, data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _first_int(self, data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    def _first_float(self, data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None

    def _string_value(self, value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _int_value(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _float_value(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

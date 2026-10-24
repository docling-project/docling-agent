from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import (
    BaseMeta,
    DocItemLabel,
    DoclingDocument,
    EntitiesMetaField,
    EntityMention,
    KeywordsMetaField,
    PictureItem,
    TableItem,
    TextItem,
)
from typing_extensions import override

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import serialize_table_to_html
from docling_agent.logging import log_stage_start

_GLM_OP_ALIASES: dict[str, str] = {
    "find_search_keywords": "_find_search_keywords",
    "keywords": "_find_search_keywords",
    "detect_key_entities": "_detect_key_entities",
    "entities": "_detect_key_entities",
}


_InitNlpModel = Callable[..., Any]


class GLMAdapter:
    """Thin adapter around GLM NLP model application."""

    def __init__(
        self,
        *,
        model_names: str = "language;semantic;sentence;term;verb;conn;geoloc;reference",
        filters: list[str] | None = None,
        loglevel: str = "WARNING",
    ) -> None:
        try:
            init_nlp_model = self._load_init_nlp_model()
        except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatch
            raise ImportError(
                "GLM is required to use DoclingGLMEnricherAgent. Install the optional dependency first."
            ) from exc

        self._model = init_nlp_model(model_names=model_names, filters=filters or [], loglevel=loglevel)

    @staticmethod
    def _load_init_nlp_model() -> _InitNlpModel:
        module = import_module("deepsearch_glm.nlp_utils")
        return getattr(module, "init_nlp_model")

    def apply_on_text(self, text: str) -> dict[str, Any]:
        """Apply the configured GLM model to text."""
        return self._model.apply_on_text(text)


class DoclingGLMEnricherAgent(BaseDoclingAgent):
    """Enrich Docling documents using GLM as a non-LLM alternative."""

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
        adapter: GLMAdapter | None = None,
        model_names: str = "language;semantic;sentence;term;verb;conn;geoloc;reference",
        filters: list[str] | None = None,
        loglevel: str = "WARNING",
    ) -> None:
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_ENRICHER,
            backend=backend or self.default_backend(),
            tools=tools,
        )
        self._adapter = adapter or GLMAdapter(
            model_names=model_names,
            filters=filters,
            loglevel=loglevel,
        )

    @override
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        if document is None:
            raise ValueError("Document must not be None")

        operations: list[str] = kwargs.get("operations") or ["detect_key_entities", "find_search_keywords"]
        result = document
        for op_name in operations:
            method_name = _GLM_OP_ALIASES.get(op_name)
            if method_name is None:
                raise ValueError(
                    f"Unsupported GLM enrichment operation: {op_name!r}. "
                    "Supported operations are detect_key_entities/entities and find_search_keywords/keywords."
                )
            result = getattr(self, method_name)(document=result)
        return result

    def _find_search_keywords(self, *, document: DoclingDocument) -> DoclingDocument:
        log_stage_start("Finding search keywords with GLM")
        self._enrich_leaf_items(document=document, include_entities=False, include_keywords=True)
        return document

    def _detect_key_entities(self, *, document: DoclingDocument) -> DoclingDocument:
        log_stage_start("Detecting key entities with GLM")
        self._enrich_leaf_items(document=document, include_entities=True, include_keywords=False)
        return document

    def _enrich_leaf_items(
        self,
        *,
        document: DoclingDocument,
        include_entities: bool,
        include_keywords: bool,
    ) -> None:
        for item, _ in document.iterate_items():
            text = self._text_for_item(item=item, document=document)
            if text is None or not text.strip():
                continue

            result = self._adapter.apply_on_text(text)
            if not isinstance(result, dict):
                continue

            entities = self._extract_entities(result=result, source_text=text) if include_entities else None
            keywords = self._extract_keywords(result=result) if include_keywords else None
            if entities is None and keywords is None:
                continue

            if item.meta is None:
                item.meta = BaseMeta()
            if entities is not None:
                item.meta.entities = entities
            if keywords is not None:
                item.meta.keywords = keywords

    @staticmethod
    def _text_for_item(*, item: Any, document: DoclingDocument) -> str | None:
        if isinstance(item, TextItem):
            if item.label == DocItemLabel.CAPTION:
                return None
            return item.text
        if isinstance(item, TableItem):
            return serialize_table_to_html(table=item, doc=document)
        if isinstance(item, PictureItem):
            captions = [c.resolve(document).text for c in item.captions if hasattr(c.resolve(document), "text")]
            return " ".join(captions)
        return None

    @staticmethod
    def _extract_entities(*, result: dict[str, Any], source_text: str) -> EntitiesMetaField | None:
        entity_rows = DoclingGLMEnricherAgent._table_rows(result, "entities")
        if not entity_rows:
            entity_rows = [
                row
                for row in DoclingGLMEnricherAgent._table_rows(result, "instances")
                if str(row.get("type", "")).strip() not in {"", "sentence"}
            ]

        mentions: list[EntityMention] = []
        seen: set[tuple[str, str | None, tuple[int, int] | None]] = set()
        for row in entity_rows:
            text = str(row.get("original") or row.get("text") or "").strip()
            if not text:
                continue
            label = str(row.get("label") or row.get("type") or row.get("subtype") or "entity").strip() or "entity"
            char_i = row.get("char_i")
            char_j = row.get("char_j")
            charspan = (int(char_i), int(char_j)) if isinstance(char_i, int) and isinstance(char_j, int) else None
            key = (text, label, charspan)
            if key in seen:
                continue
            seen.add(key)
            mention_kwargs: dict[str, Any] = {"text": text, "label": label}
            if charspan is not None:
                mention_kwargs["charspan"] = charspan
            mentions.append(EntityMention.model_construct(**mention_kwargs))

        if mentions:
            return EntitiesMetaField(mentions=mentions)
        return None

    @staticmethod
    def _extract_keywords(*, result: dict[str, Any]) -> KeywordsMetaField | None:
        rows = DoclingGLMEnricherAgent._table_rows(result, "instances")
        keywords = DoclingGLMEnricherAgent._unique_values(
            row.get("original") or row.get("text") for row in rows if str(row.get("type", "")).strip() == "term"
        )
        if keywords:
            return KeywordsMetaField(values=keywords)
        return None

    @staticmethod
    def _table_rows(result: dict[str, Any], key: str) -> list[dict[str, Any]]:
        table = result.get(key)
        if not isinstance(table, dict):
            return []
        headers = table.get("headers")
        data = table.get("data")
        if not isinstance(headers, list) or not isinstance(data, list):
            return []
        rows: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, list) or len(row) != len(headers):
                continue
            rows.append(dict(zip(headers, row, strict=False)))
        return rows

    @staticmethod
    def _unique_values(values: Iterable[Any]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

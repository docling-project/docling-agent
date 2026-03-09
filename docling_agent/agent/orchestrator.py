"""Orchestrator agent: top-level entry point for the docling-agent CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mellea.backends.model_ids import ModelIdentifier

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.library import DoclingLibrary
from docling_agent.logging import logger

if TYPE_CHECKING:
    from docling_agent.task_model import (
        AgentTask,
        EnrichTask,
        ExtractTask,
        RAGTask,
        WriteTask,
    )


# Internal type alias: a resolved document paired with its library id.
_SourcePair = tuple[DoclingDocument, str]


class DoclingOrchestratorAgent(BaseDoclingAgent):
    """Top-level orchestrator.

    Receives an ``AgentTask``, resolves source files (converting them via
    Docling and caching results in a local library), applies lazy enrichment,
    and dispatches to the appropriate sub-agent.
    """

    library_path: Path = Path.home() / ".docling_agent" / "library"

    def __init__(
        self,
        *,
        model_id: ModelIdentifier,
        writing_model_id: ModelIdentifier | None = None,
        tools: list,
        library_path: Path | None = None,
    ) -> None:
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_ORCHESTRATOR,
            model_id=model_id,
            tools=tools,
        )
        if writing_model_id is not None:
            self.writing_model_id = writing_model_id
        if library_path is not None:
            self.library_path = library_path

    # BaseDoclingAgent abstract method — not used directly
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        raise NotImplementedError("Use run_task(AgentTask) instead.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_task(self, task: "AgentTask") -> DoclingDocument:
        """Convert sources, enrich lazily, and dispatch to the right sub-agent."""
        library = DoclingLibrary(path=self.library_path)
        source_pairs = self._resolve_sources(task, library)

        mode = task.mode
        logger.info(f"Orchestrator: mode={mode}, sources={len(source_pairs)}")

        if mode == "rag":
            return self._run_rag(task=task, source_pairs=source_pairs, library=library)
        elif mode == "extract":
            return self._run_extract(task=task, source_pairs=source_pairs)
        elif mode == "write":
            return self._run_write(task=task, source_pairs=source_pairs)
        elif mode == "enrich":
            return self._run_enrich(
                task=task, source_pairs=source_pairs, library=library
            )
        else:
            raise ValueError(f"Unknown task mode: {mode!r}")

    # ------------------------------------------------------------------
    # Step 1: Source resolution
    # ------------------------------------------------------------------

    def _resolve_sources(
        self, task: "AgentTask", library: DoclingLibrary
    ) -> list[_SourcePair]:
        """Expand paths/globs, load from library cache or convert, return (doc, doc_id) pairs."""
        from docling.datamodel.base_models import ConversionStatus
        from docling.document_converter import DocumentConverter

        raw_paths = self._expand_paths(task)
        if not raw_paths and not task.sources:
            return []

        results: list[_SourcePair] = []
        raw_to_convert: list[Path] = []

        for p in raw_paths:
            source_key = str(p.resolve())

            if p.suffix.lower() == ".json":
                # Try loading as a pre-serialised DoclingDocument
                try:
                    doc = DoclingDocument.model_validate_json(
                        p.read_text(encoding="utf-8")
                    )
                    entry = library.lookup_by_source(source_key)
                    if entry is None:
                        entry = library.store(doc, source_key)
                    else:
                        # Refresh stored document in case file changed
                        library.store(doc, source_key)
                    results.append((doc, entry.doc_id))
                    logger.info(f"Loaded pre-converted document: {p.name}")
                    continue
                except Exception as exc:
                    logger.warning(f"Could not load {p} as DoclingDocument: {exc}")

            # Check library cache for already-converted files
            entry = library.lookup_by_source(source_key)
            if entry is not None:
                doc = library.load_doc(entry.doc_id)
                if doc is not None:
                    logger.info(f"Library cache hit: {p.name} → {entry.doc_id}")
                    results.append((doc, entry.doc_id))
                    continue
                logger.warning(
                    f"Library entry exists but document missing; reconverting {p.name}"
                )

            raw_to_convert.append(p)

        # Batch-convert all uncached files
        if raw_to_convert:
            converter = DocumentConverter()
            for p in raw_to_convert:
                source_key = str(p.resolve())
                try:
                    conv = converter.convert(p)
                    if conv.status == ConversionStatus.SUCCESS:
                        doc = conv.document
                        entry = library.store(doc, source_key)
                        results.append((doc, entry.doc_id))
                        logger.info(f"Converted and cached: {p.name} → {entry.doc_id}")
                    else:
                        logger.warning(f"Conversion failed for {p.name}: {conv.status}")
                except Exception as exc:
                    logger.error(f"Error converting {p}: {exc}")

        logger.info(f"Resolved {len(results)} document(s)")
        return results

    def _expand_paths(self, task: "AgentTask") -> list[Path]:
        """Expand task.sources (with optional glob for directories)."""
        glob_pattern: str = getattr(task, "glob", None) or "**/*"
        raw_paths: list[Path] = []
        for src in task.sources:
            p = Path(src)
            if p.is_dir():
                raw_paths.extend(q for q in p.rglob(glob_pattern) if q.is_file())
            elif p.is_file():
                raw_paths.append(p)
            else:
                logger.warning(f"Source not found, skipping: {src}")
        return raw_paths

    # ------------------------------------------------------------------
    # Step 2: Lazy enrichment helper
    # ------------------------------------------------------------------

    def _ensure_enriched(
        self,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
        operations: list[str],
    ) -> list[_SourcePair]:
        """Run enrichment on documents that are missing the requested enrichments.

        Returns updated (doc, doc_id) pairs where each doc is the enriched
        version (``_summarize_items`` returns a hierarchical document).
        """
        from docling_agent.agent.enricher import DoclingEnrichingAgent

        enricher = DoclingEnrichingAgent(
            model_id=self.get_writing_model_id(),
            tools=[],
        )

        updated: list[_SourcePair] = []
        for doc, doc_id in source_pairs:
            entry = library.get_entry(doc_id)
            needed = list(operations)  # copy

            if "summarize" in needed and entry and entry.status.has_summaries:
                needed.remove("summarize")
            if "keywords" in needed and entry and entry.status.has_keywords:
                needed.remove("keywords")

            if needed:
                logger.info(f"Enriching {doc.name!r} with operations={needed}")
                enriched_doc = enricher.run(task="", document=doc, operations=needed)
                # Persist enriched document back to library
                library.store(enriched_doc, entry.source_path if entry else "in-memory")
                # Update status flags
                status_updates: dict[str, bool] = {}
                if "summarize" in needed:
                    status_updates["has_summaries"] = True
                    status_updates["is_hierarchical"] = True
                if "keywords" in needed:
                    status_updates["has_keywords"] = True
                library.update_status(doc_id, **status_updates)
                # Extract top-level summary and keywords for the library index
                self._update_library_meta(doc_id, enriched_doc, library)
                updated.append((enriched_doc, doc_id))
            else:
                logger.info(f"Skipping enrichment for {doc.name!r} (already done)")
                updated.append((doc, doc_id))

        return updated

    def _update_library_meta(
        self, doc_id: str, doc: DoclingDocument, library: DoclingLibrary
    ) -> None:
        """Extract document-level summary and keywords from enriched doc and persist."""
        summary: str | None = None
        keywords: list[str] = []

        for item, _ in doc.iterate_items():
            if isinstance(item, TitleItem) and item.meta and item.meta.summary:
                summary = item.meta.summary.text
                break
            if isinstance(item, SectionHeaderItem) and item.meta and item.meta.summary:
                if summary is None:
                    summary = item.meta.summary.text

        library.update_meta(doc_id, summary=summary, keywords=keywords)

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    def _run_rag(
        self,
        *,
        task: "RAGTask",
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        from docling_agent.agent.rag import DoclingRAGAgent

        if task.enrich_before_rag:
            source_pairs = self._ensure_enriched(
                source_pairs, library, operations=["summarize"]
            )

        docs = [doc for doc, _ in source_pairs]
        rag_agent = DoclingRAGAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
            max_iterations=task.max_iterations,
        )
        return rag_agent.run(task=task.query, sources=docs)

    def _run_extract(
        self,
        *,
        task: "ExtractTask",
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        from docling_agent.agent.extractor import DoclingExtractingAgent

        extractor = DoclingExtractingAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        return extractor.run(task=task.query, sources=sources)

    def _run_write(
        self,
        *,
        task: "WriteTask",
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        from docling_agent.agent.writer import DoclingWritingAgent

        writer = DoclingWritingAgent(
            model_id=self._model_id_for("reasoning", task),
            writing_model_id=self._model_id_for("writing", task),
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        return writer.run(task=task.query, sources=sources)

    def _run_enrich(
        self,
        *,
        task: "EnrichTask",
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        ops: list[str] = list(task.operations)
        enriched_pairs = self._ensure_enriched(source_pairs, library, operations=ops)

        # Return: single doc → return it directly; multiple → a composite summary doc
        if len(enriched_pairs) == 1:
            return enriched_pairs[0][0]

        result_doc = DoclingDocument(name="enriched_collection")
        for doc, doc_id in enriched_pairs:
            entry = library.get_entry(doc_id)
            heading = doc.name
            result_doc.add_heading(text=heading, level=1, parent=result_doc.body)
            if entry and entry.summary:
                result_doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=entry.summary,
                    parent=result_doc.body,
                )
        return result_doc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_id_for(self, role: str, task: "AgentTask") -> ModelIdentifier:
        from mellea.backends import model_ids

        name = task.models.reasoning if role == "reasoning" else task.models.writing
        resolved = getattr(model_ids, name, None)
        if resolved is None:
            logger.warning(
                f"Unknown model id {name!r}; falling back to OPENAI_GPT_OSS_20B"
            )
            return model_ids.OPENAI_GPT_OSS_20B
        return resolved

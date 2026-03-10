"""Orchestrator agent: top-level entry point for the docling-agent CLI."""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)
from mellea.backends import model_ids
from mellea.backends.model_ids import ModelIdentifier
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import find_json_dicts
from docling_agent.agent.editor import DoclingEditingAgent
from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.extractor import DoclingExtractingAgent
from docling_agent.agent.library import DoclingLibrary
from docling_agent.agent.rag import DoclingRAGAgent
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.agent_models import setup_local_session
from docling_agent.logging import logger  # type: ignore[import-untyped]
from docling_agent.task_model import (
    AgentTask,
    EditingTask,
    EnrichTask,
    ExtractTask,
    RAGTask,
    WriteTask,
)

# Internal type alias: a resolved document paired with its library id.
_SourcePair = tuple[DoclingDocument, str]


class _SourcePairs(list):
    """List of ``_SourcePair`` with a compact repr to avoid polluting rich tracebacks."""

    def __repr__(self) -> str:
        logger.info("_SourcePairs.__repr__")
        entries = ", ".join(
            f"(DoclingDocument(name={doc.name!r}, version={doc.version!r}, body=[]), {did!r})" for doc, did in self
        )
        return f"[{entries}]"


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
        logger.info(f"DoclingOrchestratorAgent.__init__: model_id={model_id!r}")
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
        logger.info("DoclingOrchestratorAgent.run")
        raise NotImplementedError("Use run_task(AgentTask) instead.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_task(self, task: AgentTask) -> DoclingDocument:
        """Convert sources, enrich lazily, and dispatch to the right sub-agent."""
        logger.info(f"DoclingOrchestratorAgent.run_task: mode={task.mode!r}")
        return self._dispatch(task, DoclingLibrary(path=self.library_path))

    def _dispatch(self, task: AgentTask, library: DoclingLibrary) -> DoclingDocument:
        logger.info(f"_dispatch: mode={task.mode!r}")
        source_pairs = self._resolve_sources(task, library)

        logger.info(f"Orchestrator: mode={task.mode}, sources={len(source_pairs)}")

        if task.mode is None:
            return self._run_plan(task=task, source_pairs=source_pairs, library=library)
        elif isinstance(task, RAGTask):
            return self._run_rag(task=task, source_pairs=source_pairs, library=library)
        elif isinstance(task, ExtractTask):
            return self._run_extract(task=task, source_pairs=source_pairs)
        elif isinstance(task, WriteTask):
            return self._run_write(task=task, source_pairs=source_pairs)
        elif isinstance(task, EditingTask):
            return self._run_edit(task=task, source_pairs=source_pairs)
        elif isinstance(task, EnrichTask):
            return self._run_enrich(task=task, source_pairs=source_pairs, library=library)
        else:
            raise ValueError(f"Unknown task mode: {task.mode!r}")

    # ------------------------------------------------------------------
    # Step 1: Source resolution
    # ------------------------------------------------------------------

    def _resolve_sources(self, task: AgentTask, library: DoclingLibrary) -> list[_SourcePair]:
        """Expand paths/globs, load from library cache or convert, return (doc, doc_id) pairs."""
        logger.info(f"_resolve_sources: sources={task.sources}")
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
                    doc = DoclingDocument.model_validate_json(p.read_text(encoding="utf-8"))
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
                cached_doc: DoclingDocument | None = library.load_doc(entry.doc_id)
                if cached_doc is not None:
                    logger.info(f"Library cache hit: {p.name} → {entry.doc_id}")
                    results.append((cached_doc, entry.doc_id))
                    continue
                logger.warning(f"Library entry exists but document missing; reconverting {p.name}")

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

    def _expand_paths(self, task: AgentTask) -> list[Path]:
        """Expand task.sources (with optional glob for directories)."""
        logger.info("_expand_paths")
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
        logger.info(f"_ensure_enriched: operations={operations}, docs={len(source_pairs)}")
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

    def _update_library_meta(self, doc_id: str, doc: DoclingDocument, library: DoclingLibrary) -> None:
        """Extract document-level summary and keywords from enriched doc and persist."""
        logger.info(f"_update_library_meta: doc_id={doc_id!r}")
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
        task: RAGTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        logger.info(f"_run_rag: query={task.query!r}, docs={len(source_pairs)}")
        if task.enrich_before_rag:
            source_pairs = self._ensure_enriched(source_pairs, library, operations=["summarize"])

        docs: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        rag_agent = DoclingRAGAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
            max_iterations=task.max_iterations,
        )
        return rag_agent.run(task=task.query, sources=docs)

    def _run_extract(
        self,
        *,
        task: ExtractTask,
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        logger.info(f"_run_extract: query={task.query!r}, docs={len(source_pairs)}")
        extractor = DoclingExtractingAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        return extractor.run(task=task.query, sources=sources)

    def _run_write(
        self,
        *,
        task: WriteTask,
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        logger.info(f"_run_write: query={task.query!r}, docs={len(source_pairs)}")
        writer = DoclingWritingAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        return writer.run(task=task.query, sources=sources)

    def _run_edit(
        self,
        *,
        task: EditingTask,
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        logger.info(f"_run_edit: query={task.query!r}, docs={len(source_pairs)}")
        editor = DoclingEditingAgent(
            model_id=self._model_id_for("reasoning", task),
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        return editor.run(task=task.query, sources=sources)

    def _run_enrich(
        self,
        *,
        task: EnrichTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        logger.info(f"_run_enrich: docs={len(source_pairs)}")
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

    _PLANNER_SYSTEM_PROMPT: str = (
        "You are a task planning agent for a document intelligence system. "
        "Given a user query and an optional list of available documents, decide "
        "which agent mode(s) best handle the request and formulate concrete sub-tasks.\n\n"
        "Available modes:\n"
        "  rag      - answer questions by querying document content\n"
        "  extract  - extract structured data from documents\n"
        "  write    - write or generate a new document\n"
        "  edit     - edit an existing document\n"
        "  enrich   - summarize and annotate document content\n\n"
        'Output ONLY a JSON object in a markdown code block with key "tasks" '
        'containing a list of task objects. Each task must have: "mode" (required), '
        '"query" (specific instruction for that sub-task), and "sources" '
        "(list of document names from the provided list; empty list is fine for write tasks)."
    )

    def _run_plan(
        self,
        *,
        task: AgentTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        """Use an LLM to decide which mode(s) best serve the query, then execute them."""
        logger.info(f"_run_plan: query={task.query!r}, docs={len(source_pairs)}")
        source_names = [doc.name for doc, _ in source_pairs]
        sources_text = "\n".join(f"  - {n}" for n in source_names) if source_names else "  (none)"

        prompt = (
            f"Query: {task.query}\n\n"
            f"Available sources:\n{sources_text}\n\n"
            "Decide the best task(s). Most queries need a single task.\n"
            "Output your plan as a JSON code block."
        )

        m = setup_local_session(
            model_id=self.get_reasoning_model_id(),
            system_prompt=self._PLANNER_SYSTEM_PROMPT,
        )

        raw: ModelOutputThunk = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=3),
            requirements=[
                Requirement(
                    description="Output must contain a JSON object with a 'tasks' list",
                    validation_fn=simple_validate(
                        lambda r: bool(find_json_dicts(r)) and "tasks" in (find_json_dicts(r) or [{}])[0]
                    ),
                )
            ],
        )

        dicts = find_json_dicts(str(raw))
        if not dicts or "tasks" not in dicts[0]:
            raise ValueError(f"Planner did not return a valid plan; got: {raw!r}")

        planned_tasks = dicts[0]["tasks"]
        logger.info(f"Planner produced {len(planned_tasks)} sub-task(s)")

        name_to_pair: dict[str, _SourcePair] = {doc.name: (doc, did) for doc, did in source_pairs}
        results: list[DoclingDocument] = []

        for plan in planned_tasks:
            mode = plan.get("mode")
            query = plan.get("query", task.query)
            planned_sources: list[str] = plan.get("sources", source_names)
            resolved = [name_to_pair[n] for n in planned_sources if n in name_to_pair] or source_pairs

            logger.info(f"  sub-task: mode={mode!r}, query={query!r}")

            if mode == "rag":
                results.append(
                    self._run_rag(
                        task=RAGTask(
                            query=query,
                            sources=task.sources,
                            models=task.models,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                        library=library,
                    )
                )
            elif mode == "extract":
                results.append(
                    self._run_extract(
                        task=ExtractTask(
                            query=query,
                            sources=task.sources,
                            models=task.models,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                    )
                )
            elif mode == "write":
                results.append(
                    self._run_write(
                        task=WriteTask(
                            query=query,
                            sources=task.sources,
                            models=task.models,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                    )
                )
            elif mode == "edit":
                results.append(
                    self._run_edit(
                        task=EditingTask(
                            query=query,
                            sources=task.sources,
                            models=task.models,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                    )
                )
            elif mode == "enrich":
                results.append(
                    self._run_enrich(
                        task=EnrichTask(
                            query=query,
                            sources=task.sources,
                            models=task.models,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                        library=library,
                    )
                )
            else:
                logger.warning(f"Planner produced unknown mode {mode!r}, skipping")

        if not results:
            raise ValueError("Planner produced no executable sub-tasks")
        if len(results) == 1:
            return results[0]

        combined = DoclingDocument(name="plan_results")
        for i, res in enumerate(results):
            combined.add_heading(text=f"Result {i + 1}", level=1, parent=combined.body)
            text = MarkdownDocSerializer(doc=res).serialize().text
            combined.add_text(label=DocItemLabel.TEXT, text=text, parent=combined.body)
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_id_for(self, role: str, task: AgentTask) -> ModelIdentifier:
        logger.info(f"_model_id_for: role={role!r}")
        name = task.models.reasoning if role == "reasoning" else task.models.writing
        resolved = getattr(model_ids, name, None)
        if resolved is None:
            logger.warning(f"Unknown model id {name!r}; falling back to OPENAI_GPT_OSS_20B")
            return model_ids.OPENAI_GPT_OSS_20B
        return resolved

"""Orchestrator agent: top-level entry point for the docling-agent CLI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, cast

from docling.datamodel.base_models import ConversionStatus, FormatToMimeType, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)
from mellea.stdlib.requirements import Requirement, simple_validate
from tabulate import tabulate
from typing_extensions import override

from docling_agent.agent.agent_trace import AgentTrace
from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import find_json_dicts
from docling_agent.agent.editor import DoclingEditingAgent
from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.extractor import DoclingExtractingAgent
from docling_agent.agent.library import DocLibraryEntry, DoclingLibrary
from docling_agent.agent.rag import DoclingRAGAgent
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.logging import log_error, log_info, log_warning
from docling_agent.task_model import (
    AddTask,
    AgentTask,
    ClearTask,
    EditingTask,
    EnrichTask,
    ExtractTask,
    ListTask,
    RAGTask,
    ViewTask,
    WriteTask,
)

# Internal type alias: a resolved document paired with its library id.
_SourcePair = tuple[DoclingDocument, str]


class _SourcePairs(list):
    """List of ``_SourcePair`` with a compact repr to avoid polluting rich tracebacks."""

    def __repr__(self) -> str:
        log_info("_SourcePairs.__repr__")
        entries = ", ".join(
            f"(DoclingDocument(name={doc.name!r}, version={doc.version!r}, body=[]), {did!r})" for doc, did in self
        )
        return f"[{entries}]"


class DoclingOrchestratorAgent(BaseDoclingAgent):
    """Top-level orchestrator agent for coordinating document operations.

    This agent:
    1. Receives an AgentTask specifying the operation mode and sources
    2. Resolves source files (converting via Docling, caching in library)
    3. Applies lazy enrichment as needed
    4. Dispatches to the appropriate specialized sub-agent (RAG, Writer, Editor, etc.)

    The orchestrator manages a document library for caching converted documents
    and coordinates the workflow across different agent types.

    Attributes:
        library_path: Path to the document library for caching conversions.
    """

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
        library_path: Path | None = None,
    ) -> None:
        """Initialize the DoclingOrchestratorAgent.

        Args:
            tools: List of tools available to the agent.
            backend: Optional backend for LLM interactions. If not provided,
                uses the default backend.
            library_path: Optional path to document library. If not provided,
                uses ~/.docling_agent/library.
        """
        log_info("DoclingOrchestratorAgent.__init__")
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_ORCHESTRATOR,
            backend=backend or self.default_backend(),
            tools=tools,
        )
        self.library_path: Path = library_path or (Path.home() / ".docling_agent" / "library")
        # When set (by run_task_with_trace), sub-agent traces are collected into this list
        # to compose the orchestrator trace tree. None means tracing is off (no overhead).
        self._child_traces: list[AgentTrace] | None = None

    def _record_child(self, trace: AgentTrace) -> None:
        """Record a sub-agent trace into the current tree, if tracing is active."""
        if self._child_traces is not None:
            self._child_traces.append(trace)

    # BaseDoclingAgent abstract method — not used directly
    @override
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        log_info("DoclingOrchestratorAgent.run")
        raise NotImplementedError("Use run_task(AgentTask) instead.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_task(self, task: AgentTask) -> DoclingDocument:
        """Convert sources, enrich lazily, and dispatch to the right sub-agent."""
        log_info(f"DoclingOrchestratorAgent.run_task: mode={task.mode!r}")
        return self._dispatch(task, DoclingLibrary(path=self.library_path, project_id=task.project_id))

    def run_task_with_trace(self, task: AgentTask) -> AgentTrace:
        """Run the task and return the orchestrator trace tree.

        The returned :class:`AgentTrace` nests one child trace per sub-agent the
        orchestrator dispatched to (RAG, enricher, writer ...), so the whole session
        is a single tree that can be exported to a file. ``run_task()`` is unchanged
        and incurs no tracing overhead when called directly.
        """
        previous = self._child_traces
        self._child_traces = []
        start = time.perf_counter()
        try:
            doc = self.run_task(task)
            children = self._child_traces
        finally:
            self._child_traces = previous
        duration_ms = int((time.perf_counter() - start) * 1000)
        return AgentTrace(
            agent_type=str(self.agent_type),
            task=task.query,
            children=children,
            duration_ms=duration_ms,
            model_id=self.get_reasoning_model_id(),
            result_name=getattr(doc, "name", None),
            output=doc,
        )

    def _dispatch(self, task: AgentTask, library: DoclingLibrary) -> DoclingDocument:
        log_info(f"_dispatch: mode={task.mode!r}")
        source_pairs = self._resolve_sources(task, library)

        log_info(f"Orchestrator: mode={task.mode}, sources={len(source_pairs)}")

        if task.mode is None:
            return self._run_plan(task=task, source_pairs=source_pairs, library=library)
        elif isinstance(task, AddTask):
            return self._run_add(task=task, source_pairs=source_pairs, library=library)
        elif isinstance(task, ListTask):
            return self._run_list(task=task, library=library)
        elif isinstance(task, ViewTask):
            return self._run_view(task=task, library=library)
        elif isinstance(task, ClearTask):
            return self._run_clear(task=task, library=library)
        elif isinstance(task, RAGTask):
            return self._run_rag(task=task, source_pairs=source_pairs, library=library)
        elif isinstance(task, ExtractTask):
            return self._run_extract(task=task, source_pairs=source_pairs)
        elif isinstance(task, WriteTask):
            return self._run_write(task=task, source_pairs=source_pairs, library=library)
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
        log_info(f"_resolve_sources: sources={task.sources}")
        raw_paths = self._expand_paths(task)
        conversion = self._source_conversion_preset(task)
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
                        entry = library.store(
                            doc,
                            source_key,
                            project_id=task.project_id,
                            original_mimetype="application/json",
                            conversion_pipeline="preconverted-json",
                        )
                    else:
                        # Refresh stored document in case file changed
                        library.store(
                            doc,
                            source_key,
                            project_id=task.project_id,
                            original_mimetype=entry.original_mimetype,
                        )
                    results.append((doc, entry.doc_id))
                    log_info(f"Loaded pre-converted document: {p.name}")
                    continue
                except Exception as exc:
                    log_warning(f"Could not load {p} as DoclingDocument: {exc}")

            # Check library cache for already-converted files
            entry = library.lookup_by_source(source_key)
            if entry is not None:
                cached_doc: DoclingDocument | None = (
                    library.load_doc(entry.doc_id)
                    if self._cache_entry_matches_conversion(entry, conversion=conversion)
                    else None
                )
                if cached_doc is not None:
                    log_info(f"Library cache hit: {p.name} → {entry.doc_id}")
                    results.append((cached_doc, entry.doc_id))
                    continue
                log_warning(f"Library entry exists but document missing; reconverting {p.name}")

            raw_to_convert.append(p)

        # Batch-convert all uncached files
        if raw_to_convert:
            converter = self._build_source_converter(conversion)
            for p in raw_to_convert:
                source_key = str(p.resolve())
                try:
                    conv = converter.convert(p)
                    if conv.status == ConversionStatus.SUCCESS:
                        doc = conv.document
                        entry = library.store(
                            doc,
                            source_key,
                            project_id=task.project_id,
                            original_mimetype=self._mimetype_for_input_format(conv.input.format),
                            conversion_pipeline=self._pipeline_name_for_input_format(
                                converter=converter,
                                input_format=conv.input.format,
                                conversion=conversion,
                            ),
                        )
                        results.append((doc, entry.doc_id))
                        log_info(f"Converted and cached: {p.name} → {entry.doc_id}")
                    else:
                        log_warning(f"Conversion failed for {p.name}: {conv.status}")
                except Exception as exc:
                    log_error(f"Error converting {p}: {exc}")

        log_info(f"Resolved {len(results)} document(s)")
        return results

    def _expand_paths(self, task: AgentTask) -> list[Path]:
        """Expand task.sources (with optional glob for directories)."""
        log_info("_expand_paths")
        glob_pattern: str = getattr(task, "glob", None) or "**/*"
        raw_paths: list[Path] = []
        for src in task.sources:
            p = Path(src)
            if p.is_dir():
                raw_paths.extend(q for q in p.rglob(glob_pattern) if q.is_file())
            elif p.is_file():
                raw_paths.append(p)
            else:
                log_warning(f"Source not found, skipping: {src}")
        return raw_paths

    # ------------------------------------------------------------------
    # Step 2: Lazy enrichment helper
    # ------------------------------------------------------------------

    def _ensure_enriched(
        self,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
        operations: list[str],
        task: str = "",
    ) -> list[_SourcePair]:
        """Run enrichment on documents that are missing the requested enrichments.

        Returns updated (doc, doc_id) pairs where each doc is the enriched
        version (``_summarize_items`` returns a hierarchical document).
        """
        log_info(f"_ensure_enriched: operations={operations}, docs={len(source_pairs)}")
        enricher = DoclingEnrichingAgent(
            backend=self.backend,
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
                log_info(f"Enriching {doc.name!r} with operations={needed}")
                etrace = enricher.run_with_trace(task=task, document=doc, operations=needed)
                self._record_child(etrace)
                enriched_doc = cast(DoclingDocument, etrace.output)
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
                library.record_enrichments(
                    doc_id,
                    self._normalize_enrichment_operations(needed),
                    task=task or None,
                )
                # Extract top-level summary and keywords for the library index
                self._update_library_meta(doc_id, enriched_doc, library)
                updated.append((enriched_doc, doc_id))
            else:
                log_info(f"Skipping enrichment for {doc.name!r} (already done)")
                updated.append((doc, doc_id))

        return updated

    def _update_library_meta(self, doc_id: str, doc: DoclingDocument, library: DoclingLibrary) -> None:
        """Extract document-level summary and keywords from enriched doc and persist."""
        log_info(f"_update_library_meta: doc_id={doc_id!r}")
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

    def _run_add(
        self,
        *,
        task: AddTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        log_info(f"_run_add: docs={len(source_pairs)}")
        entries = [library.get_entry(doc_id) for _, doc_id in source_pairs]
        return self._entries_to_doc(
            name="library_add_result",
            title=f"Added {len(source_pairs)} document(s)",
            entries=[entry for entry in entries if entry is not None],
            detailed=False,
        )

    def _run_list(self, *, task: ListTask, library: DoclingLibrary) -> DoclingDocument:
        log_info(f"_run_list: postgres_filter={task.postgres_filter!r}")
        if task.postgres_filter:
            entries = library.query_entries_by_postgres_filter(task.postgres_filter, limit=task.limit)
        else:
            entries = sorted(library.all_entries(), key=lambda entry: entry.updated_at, reverse=True)[: task.limit]
        return self._entries_to_doc(
            name="library_list",
            title=f"Library documents ({len(entries)})",
            entries=entries,
            detailed=False,
        )

    def _run_view(self, *, task: ViewTask, library: DoclingLibrary) -> DoclingDocument:
        log_info(f"_run_view: postgres_filter={task.postgres_filter!r}")
        entries = library.query_entries_by_postgres_filter(task.postgres_filter, limit=task.limit)
        return self._entries_to_doc(
            name="library_view",
            title=f"Library document state ({len(entries)})",
            entries=entries,
            detailed=True,
        )

    def _run_clear(self, *, task: ClearTask, library: DoclingLibrary) -> DoclingDocument:
        log_info(f"_run_clear: project_id={task.project_id!r}, all_projects={task.all_projects}")
        removed = library.clear(project_id=task.project_id, all_projects=task.all_projects)
        scope = "all projects" if task.all_projects else f"project {task.project_id!r}"
        doc = DoclingDocument(name="library_clear_result")
        doc.add_heading(text="Library cleared", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text=f"Removed {removed} document(s) from {scope}.", parent=doc.body)
        return doc

    def _run_rag(
        self,
        *,
        task: RAGTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        log_info(f"_run_rag: query={task.query!r}, docs={len(source_pairs)}")
        if task.enrich_before_rag:
            source_pairs = self._ensure_enriched(source_pairs, library, operations=["summarize"], task=task.query)

        docs: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        rag_agent = DoclingRAGAgent(
            backend=self.backend,
            tools=[],
            max_iterations=task.max_iterations,
        )
        trace = rag_agent.run_with_trace(task=task.query, sources=docs)
        self._record_child(trace)
        return cast(DoclingDocument, trace.output)

    def _run_extract(
        self,
        *,
        task: ExtractTask,
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        log_info(f"_run_extract: query={task.query!r}, docs={len(source_pairs)}")
        extractor = DoclingExtractingAgent(
            backend=self.backend,
            tools=[],
        )
        # For extraction, pass the original source paths (not converted DoclingDocuments)
        # because DocumentExtractor needs raw files to perform extraction
        raw_paths = self._expand_paths(task)
        sources: list[DoclingDocument | Path] = cast(list[DoclingDocument | Path], raw_paths)
        trace = extractor.run_with_trace(task=task.query, sources=sources)
        self._record_child(trace)
        return cast(DoclingDocument, trace.output)

    def _run_write(
        self,
        *,
        task: WriteTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        log_info(f"_run_write: query={task.query!r}, docs={len(source_pairs)}")
        writer = DoclingWritingAgent(
            backend=self.backend,
            tools=[],
        )
        sources: list[DoclingDocument | Path] = [doc for doc, _ in source_pairs]
        trace = writer.run_with_trace(task=task.query, sources=sources)
        self._record_child(trace)
        doc = cast(DoclingDocument, trace.output)
        entry = library.store_in_memory(doc, project_id=task.project_id, document_origin="written")
        log_info(f"Stored written document in library: {doc.name!r} → {entry.doc_id}")
        return doc

    def _run_edit(
        self,
        *,
        task: EditingTask,
        source_pairs: list[_SourcePair],
    ) -> DoclingDocument:
        log_info(f"_run_edit: query={task.query!r}, docs={len(source_pairs)}")
        editor = DoclingEditingAgent(
            backend=self.backend,
            tools=[],
        )
        if not source_pairs:
            raise ValueError("Edit tasks require at least one source document")
        document = source_pairs[0][0]
        trace = editor.run_with_trace(task=task.query, document=document)
        self._record_child(trace)
        return cast(DoclingDocument, trace.output)

    def _run_enrich(
        self,
        *,
        task: EnrichTask,
        source_pairs: list[_SourcePair],
        library: DoclingLibrary,
    ) -> DoclingDocument:
        log_info(f"_run_enrich: docs={len(source_pairs)}")
        if task.operations is None:
            enriched_pairs = []
            for doc, doc_id in source_pairs:
                enricher = DoclingEnrichingAgent(backend=self.backend, tools=[])
                log_info(f"Enriching {doc.name!r} by inferred operations from query")
                etrace = enricher.run_with_trace(task=task.query, document=doc)
                self._record_child(etrace)
                enriched_doc = cast(DoclingDocument, etrace.output)
                entry = library.get_entry(doc_id)
                library.store(enriched_doc, entry.source_path if entry else "in-memory")
                inferred_ops = enricher.get_last_operation().get("operations", [])
                status_updates: dict[str, bool] = {}
                if "summarize_items" in inferred_ops:
                    status_updates["has_summaries"] = True
                    status_updates["is_hierarchical"] = True
                if "find_search_keywords" in inferred_ops:
                    status_updates["has_keywords"] = True
                if status_updates:
                    library.update_status(doc_id, **status_updates)
                library.record_enrichments(
                    doc_id,
                    self._normalize_enrichment_operations(inferred_ops),
                    task=task.query or None,
                )
                self._update_library_meta(doc_id, enriched_doc, library)
                enriched_pairs.append((enriched_doc, doc_id))
        else:
            ops: list[str] = list(task.operations)
            enriched_pairs = self._ensure_enriched(source_pairs, library, operations=ops, task=task.query)

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
        "  add      - add new source documents to the document library\n"
        "  list     - list documents in the document library\n"
        "  view     - view detailed document library state\n"
        "  clear    - clear documents from a project or the entire library\n"
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
        log_info(f"_run_plan: query={task.query!r}, docs={len(source_pairs)}")
        source_names = [doc.name for doc, _ in source_pairs]
        sources_text = "\n".join(f"  - {n}" for n in source_names) if source_names else "  (none)"

        prompt = (
            f"Query: {task.query}\n\n"
            f"Available sources:\n{sources_text}\n\n"
            "Decide the best task(s). Most queries need a single task.\n"
            "Output your plan as a JSON code block."
        )

        m = self._create_reasoning_session(system_prompt=self._PLANNER_SYSTEM_PROMPT)

        raw = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Output must contain a JSON object with a 'tasks' list",
                    validation_fn=simple_validate(
                        lambda r: bool(find_json_dicts(r)) and "tasks" in (find_json_dicts(r) or [{}])[0]
                    ),
                )
            ],
            retry_budget=3,
        )

        dicts = find_json_dicts(raw)
        if not dicts or "tasks" not in dicts[0]:
            raise ValueError(f"Planner did not return a valid plan; got: {raw!r}")

        planned_tasks = dicts[0]["tasks"]
        log_info(f"Planner produced {len(planned_tasks)} sub-task(s)")

        name_to_pair: dict[str, _SourcePair] = {doc.name: (doc, did) for doc, did in source_pairs}
        results: list[DoclingDocument] = []

        for plan in planned_tasks:
            mode = plan.get("mode")
            query = plan.get("query", task.query)
            planned_sources: list[str] = plan.get("sources", source_names)
            resolved = [name_to_pair[n] for n in planned_sources if n in name_to_pair] or source_pairs

            log_info(f"  sub-task: mode={mode!r}, query={query!r}")

            if mode == "add":
                results.append(
                    self._run_add(
                        task=AddTask(
                            query=query,
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                        library=library,
                    )
                )
            elif mode == "list":
                results.append(
                    self._run_list(
                        task=ListTask(
                            query=query,
                            project_id=task.project_id,
                            backend=task.backend,
                            logging=task.logging,
                        ),
                        library=library,
                    )
                )
            elif mode == "view":
                log_warning("Planner produced mode 'view' without a PostgreSQL filter, skipping")
            elif mode == "clear":
                log_warning(
                    "Planner produced mode 'clear', skipping because clearing requires explicit CLI confirmation"
                )
            elif mode == "rag":
                results.append(
                    self._run_rag(
                        task=RAGTask(
                            query=query,
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
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
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
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
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                        library=library,
                    )
                )
            elif mode == "edit":
                results.append(
                    self._run_edit(
                        task=EditingTask(
                            query=query,
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
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
                            project_id=task.project_id,
                            sources=task.sources,
                            backend=task.backend,
                            logging=task.logging,
                        ),
                        source_pairs=resolved,
                        library=library,
                    )
                )
            else:
                log_warning(f"Planner produced unknown mode {mode!r}, skipping")

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

    def _mimetype_for_input_format(self, input_format: InputFormat) -> str | None:
        mimetypes = FormatToMimeType.get(input_format, [])
        return mimetypes[0] if mimetypes else None

    def _pipeline_name_for_input_format(
        self,
        *,
        converter: DocumentConverter,
        input_format: InputFormat,
        conversion: Literal["fast", "standard", "expensive"],
    ) -> str:
        format_option = converter.format_to_options.get(input_format)
        if format_option is None:
            return f"unknown:{conversion}"
        return f"{format_option.pipeline_cls.__name__}:{conversion}"

    def _source_conversion_preset(self, task: AgentTask) -> Literal["fast", "standard", "expensive"]:
        if isinstance(task, AddTask):
            return task.conversion
        return "standard"

    def _build_source_converter(self, conversion: Literal["fast", "standard", "expensive"]) -> DocumentConverter:
        converter = DocumentConverter()
        for format_option in converter.format_to_options.values():
            pipeline_options = format_option.pipeline_options
            if pipeline_options is None:
                continue
            if hasattr(pipeline_options, "do_ocr"):
                pipeline_options.do_ocr = conversion != "fast"
            if hasattr(pipeline_options, "do_table_structure"):
                pipeline_options.do_table_structure = conversion != "fast"
            if hasattr(pipeline_options, "do_picture_classification"):
                pipeline_options.do_picture_classification = True
            if hasattr(pipeline_options, "do_chart_extraction"):
                pipeline_options.do_chart_extraction = conversion == "expensive"
            if hasattr(pipeline_options, "generate_page_images"):
                pipeline_options.generate_page_images = True
        return converter

    def _cache_entry_matches_conversion(
        self,
        entry: DocLibraryEntry,
        *,
        conversion: Literal["fast", "standard", "expensive"],
    ) -> bool:
        for run in entry.status.pipelines:
            if run.name.endswith(f":{conversion}"):
                return True
            if conversion == "standard" and run.name.rsplit(":", 1)[-1] not in {"fast", "standard", "expensive"}:
                return True
        return False

    def _normalize_enrichment_operations(self, operations: list[str]) -> list[str]:
        aliases = {
            "summarize_items": "summarize",
            "find_search_keywords": "keywords",
            "extract_entities": "entities",
            "classify_pictures": "classify",
        }
        normalized = [aliases.get(operation, operation) for operation in operations]
        return list(dict.fromkeys(normalized))

    def _entries_to_doc(
        self,
        *,
        name: str,
        title: str,
        entries: list[DocLibraryEntry],
        detailed: bool,
    ) -> DoclingDocument:
        doc = DoclingDocument(name=name)
        doc.add_heading(text=title, level=1, parent=doc.body)
        if not entries:
            doc.add_text(label=DocItemLabel.TEXT, text="No documents matched.", parent=doc.body)
            return doc

        if detailed:
            for entry in entries:
                doc.add_heading(text=f"{entry.name} ({entry.doc_id})", level=2, parent=doc.body)
                doc.add_text(label=DocItemLabel.TEXT, text=self._format_entry_detail(entry), parent=doc.body)
            return doc

        rows = [
            [
                entry.doc_id,
                entry.project_id,
                entry.document_origin,
                entry.name,
                entry.original_mimetype or "",
                "" if entry.stats.page_count is None else entry.stats.page_count,
                entry.stats.table_count,
                entry.stats.picture_count,
                entry.stats.text_count,
                entry.stats.xml_char_count,
                entry.updated_at,
                entry.source_path,
            ]
            for entry in entries
        ]
        table = tabulate(
            rows,
            headers=[
                "doc_id",
                "project",
                "origin",
                "name",
                "mimetype",
                "pages",
                "tables",
                "pictures",
                "texts",
                "xml_chars",
                "updated_at",
                "source",
            ],
            tablefmt="plain",
        )
        doc.add_code(text=table, parent=doc.body)
        return doc

    def _format_entry_detail(self, entry: DocLibraryEntry) -> str:
        pipelines = ", ".join(f"{run.name} @ {run.ran_at}" for run in entry.status.pipelines) or "(none)"
        enrichments = (
            ", ".join(
                f"{run.name} @ {run.ran_at}" + (f" ({run.task})" if run.task else "")
                for run in entry.status.enrichments
            )
            or "(none)"
        )
        keywords = ", ".join(entry.keywords) or "(none)"
        return "\n".join(
            [
                f"- doc_id: {entry.doc_id}",
                f"- project_id: {entry.project_id}",
                f"- source_path: {entry.source_path}",
                f"- document_origin: {entry.document_origin}",
                f"- original_mimetype: {entry.original_mimetype or ''}",
                f"- doc_path: {entry.doc_path}",
                f"- doc_format: {entry.doc_format}",
                f"- created_at: {entry.created_at}",
                f"- updated_at: {entry.updated_at}",
                f"- is_hierarchical: {entry.status.is_hierarchical}",
                f"- has_summaries: {entry.status.has_summaries}",
                f"- has_keywords: {entry.status.has_keywords}",
                f"- page_count: {entry.stats.page_count}",
                f"- table_count: {entry.stats.table_count}",
                f"- picture_count: {entry.stats.picture_count}",
                f"- text_count: {entry.stats.text_count}",
                f"- xml_char_count: {entry.stats.xml_char_count}",
                f"- pipelines: {pipelines}",
                f"- enrichments: {enrichments}",
                f"- summary: {entry.summary or ''}",
                f"- keywords: {keywords}",
            ]
        )

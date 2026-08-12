"""Chunkless RAG agent using DoclingDocument tree structure and per-node summaries."""

import json
import re
import time
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from docling_core.experimental.serializer.outline import (
    OutlineFormat,
)
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    ImageRefMode,
    NodeItem,
    RefItem,
    SectionHeaderItem,
    TitleItem,
)
from mellea.stdlib.requirements import Requirement, simple_validate
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from typing_extensions import override

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    collect_subtree_text,
    create_document_outline,
    find_json_dicts,
    get_item_by_ref,
)
from docling_agent.agent.rag_models import (
    AnswerAttempt,
    RAGIteration,
    RAGResult,
    RAGTrace,
    SectionSelection,
)
from docling_agent.agent_models import view_linear_context
from docling_agent.backends.base import BaseBackend
from docling_agent.logging import log_debug, log_info, log_warning


class DoclingRAGAgent(BaseDoclingAgent):
    """Chunkless RAG agent using document structure and per-node summaries.

    Builds a compact document outline with per-node summaries, lets the LLM
    iteratively select the most relevant section, reads only that section's
    content, and attempts to answer the query without loading the full
    document into the context window.

    Attributes:
        max_iterations: Maximum number of RAG iterations to perform before stopping
        verbose: Enable verbose output with rich formatting for debugging
        enable_document_selection: Enable document filtering before RAG when multiple documents are provided
        use_page_level: Use pages as retrieval units instead of sections
        use_batch_selection: Use batch-based selection instead of iterative selection (experimental)
        batch_size: Number of pages or sections to evaluate per batch when using batch selection
        top_k: Maximum number of pages or sections to select when using batch selection
    """

    _RAG_SYSTEM_PROMPT: ClassVar[str] = (
        "You are a precise research assistant. You are given a query and a document outline "
        "with per-section summaries or keyphrases. Your job is to iteratively select the most relevant sections "
        "and build an answer from their content. "
        "Always ground your answer in the document content. "
        "Do not hallucinate or add information not present in the retrieved sections."
    )

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
        max_iterations: int = 5,
        verbose: bool = False,
        enable_document_selection: bool = False,
        use_page_level: bool = False,
        use_batch_selection: bool = False,
        batch_size: int = 30,
        top_k: int = 10,
    ):
        """Initialize the RAG agent.

        Args:
            tools: List of tools available to the agent
            backend: LLM backend to use (default: mellea)
            max_iterations: Maximum number of RAG iterations (default: 5)
            verbose: Enable verbose output (default: False)
            enable_document_selection: Enable document filtering (default: False)
            use_page_level: Use pages instead of sections (default: False)
            use_batch_selection: Use batch-based selection (default: False)
            batch_size: Pages/sections per batch (default: 30)
            top_k: Maximum pages/sections to select (default: 10)
        """
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_RAG,
            backend=backend or self.default_backend(),
            tools=tools,
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.enable_document_selection = enable_document_selection
        self.use_page_level = use_page_level
        self.use_batch_selection = use_batch_selection
        self.batch_size = batch_size
        self.top_k = top_k
        self._console = Console(highlight=False) if verbose else None

    def _rprint(self, renderable: Any) -> None:
        """Print to the rich console only when verbose mode is enabled."""
        if self._console is not None:
            self._console.print(renderable)

    @override
    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        trace = self.run_with_trace(task, document=document, sources=sources, **kwargs)
        # run_with_trace always builds and attaches the answer document on `output`.
        return cast(DoclingDocument, trace.output)

    def run_with_trace(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> RAGTrace:
        """Run the RAG loop and return the full reasoning trace.

        Same orchestration as run(), but returns a typed RAGTrace exposing the
        per-document RAGResult (selections, reasons, convergence) and the merged
        final_answer. run() is a thin wrapper around this method.
        """
        start = time.perf_counter()
        docs = [s for s in sources if isinstance(s, DoclingDocument)]
        if not docs and document is not None:
            docs = [document]
        if not docs:
            raise ValueError("DoclingRAGAgent requires at least one DoclingDocument.")

        # Optional document selection
        if self.enable_document_selection and len(docs) > 1:
            self._rprint(Rule(f"[bold cyan]Selecting relevant documents from {len(docs)} candidates[/bold cyan]"))

            # Build document summaries
            documents_dict = {doc.name: doc for doc in docs}
            doc_summaries = {doc.name: self._extract_document_summary(doc) for doc in docs}

            # Select relevant documents
            selected_doc_names = self._select_relevant_documents(
                query=task,
                documents=documents_dict,
                doc_summaries=doc_summaries,
            )

            # Filter docs to only selected ones
            docs = [documents_dict[name] for name in selected_doc_names if name in documents_dict]
            self._rprint(
                Panel(
                    f"Selected {len(docs)} document(s): {[d.name for d in docs]}",
                    title="[cyan]Document Selection[/cyan]",
                    border_style="cyan",
                )
            )

        per_document: list[RAGResult] = []
        for doc in docs:
            result = self._rag_loop(query=task, doc=doc)
            per_document.append(result)
            log_info(f"RAG loop finished: converged={result.converged}, iterations={len(result.iterations)}")

        if len(docs) > 1:
            self._rprint(Rule(f"[bold cyan]Merging answers from {len(docs)} documents[/bold cyan]"))

        final_answer = self._merge_answers(
            query=task,
            answers=[r.answer for r in per_document],
        )

        answer_doc = DoclingDocument(name="rag_answer")
        answer_doc.add_title(text="Answer", parent=answer_doc.body)
        answer_doc.add_text(label=DocItemLabel.TEXT, text=final_answer, parent=answer_doc.body)

        return RAGTrace(
            agent_type=str(self.agent_type),
            task=task,
            duration_ms=int((time.perf_counter() - start) * 1000),
            model_id=self.get_reasoning_model_id(),
            result_name=answer_doc.name,
            output=answer_doc,
            query=task,
            per_document=per_document,
            final_answer=final_answer,
        )

    # ------------------------------------------------------------------
    # RAG loop
    # ------------------------------------------------------------------

    def _rag_loop(self, *, query: str, doc: DoclingDocument) -> RAGResult:
        m = self._create_reasoning_session(system_prompt=self._RAG_SYSTEM_PROMPT)

        visited: set[str] = set()
        iterations: list[RAGIteration] = []

        outline_text = create_document_outline(doc, format=OutlineFormat.MARKDOWN)
        log_debug(f"[RAG OUTLINE — {doc.name!r}]\n{outline_text}")
        valid_refs = self._extract_section_refs(doc)

        self._rprint(Rule(f"[bold cyan]RAG loop — {doc.name!r}[/bold cyan]"))
        self._rprint(
            Panel(
                f"[bold]Query:[/bold] {query}\n\n"
                f"[bold]Sections available:[/bold] {len(valid_refs)}  "
                f"[bold]Max iterations:[/bold] {self.max_iterations}",
                title="[cyan]Setup[/cyan]",
                border_style="cyan",
            )
        )

        # Fallback: no section headers → return full doc text
        if not valid_refs:
            log_warning("No section headers found; falling back to full document text.")
            self._rprint(
                Text(
                    "⚠ No section headers found — returning full document.",
                    style="yellow",
                )
            )

            full_text = MarkdownDocSerializer(doc=doc).serialize().text
            return RAGResult(answer=full_text, iterations=[], converged=True)

        for i in range(self.max_iterations):
            unvisited = valid_refs - visited
            if not unvisited:
                log_info("All sections visited; stopping early.")
                self._rprint(Text("All sections visited — stopping early.", style="yellow"))
                break

            self._rprint(Rule(f"[bold]Iteration {i + 1} / {self.max_iterations}[/bold]"))

            selection = self._select_section(
                m=m,
                query=query,
                outline_text=outline_text,
                valid_refs=valid_refs,
                visited=visited,
            )
            visited.add(selection.section_ref)

            self._rprint(
                Panel(
                    f"[bold]Selected:[/bold]  {selection.section_ref}\n[bold]Reason:[/bold]    {selection.reason}",
                    title="[cyan]Section Selection[/cyan]",
                    border_style="blue",
                )
            )

            section_text = self._get_section_content(doc, selection.section_ref)
            preview = section_text[:300].replace("\n", " ") + (" …" if len(section_text) > 300 else "")
            self._rprint(
                Panel(
                    f"[dim]{preview}[/dim]\n\n[bold]Length:[/bold] {len(section_text)} chars",
                    title="[cyan]Section Content[/cyan]",
                    border_style="dim",
                )
            )

            attempt = self._attempt_answer(
                m=m,
                query=query,
                section_ref=selection.section_ref,
                section_text=section_text,
            )

            status_color = "green" if attempt.can_answer else "yellow"
            status_label = "✓ Can answer" if attempt.can_answer else "✗ Need more context"
            self._rprint(
                Panel(
                    f"[bold]Status:[/bold]   [{status_color}]{status_label}[/{status_color}]\n"
                    f"[bold]Response:[/bold] {attempt.response[:400]}",
                    title="[cyan]Answer Attempt[/cyan]",
                    border_style=status_color,
                )
            )

            iterations.append(
                RAGIteration(
                    iteration=i + 1,
                    section_ref=selection.section_ref,
                    reason=selection.reason,
                    section_text_length=len(section_text),
                    can_answer=attempt.can_answer,
                    response=attempt.response,
                )
            )

            if attempt.can_answer:
                self._rprint(
                    Panel(
                        attempt.response,
                        title=f"[bold green]Final Answer (converged in {i + 1} iteration(s))[/bold green]",
                        border_style="green",
                    )
                )
                return RAGResult(
                    answer=attempt.response,
                    iterations=iterations,
                    converged=True,
                )

        last = (
            iterations[-1]
            if iterations
            else RAGIteration(
                iteration=0,
                section_ref="",
                reason="",
                section_text_length=0,
                can_answer=False,
                response="No content could be retrieved.",
            )
        )
        self._rprint(
            Panel(
                last.response,
                title="[bold yellow]Partial Answer (max iterations reached)[/bold yellow]",
                border_style="yellow",
            )
        )
        return RAGResult(
            answer=(f"[Partial answer after {len(iterations)} iteration(s)]\n\n{last.response}"),
            iterations=iterations,
            converged=False,
        )

    def _extract_section_refs(self, doc: DoclingDocument) -> set[str]:
        """Extract section or page references based on mode.

        Returns:
            Set of section refs (e.g., "#/body/0") or page refs (e.g., "#/pages/0")
        """
        if self.use_page_level:
            # Page-level mode: return page references using JSON pointer format
            refs: set[str] = set()
            num_pages = len(doc.pages) if doc.pages else 0
            for i in range(num_pages):
                refs.add(f"#/pages/{i}")  # Use JSON pointer format
            return refs
        else:
            # Section-level mode: return section header references
            refs = set()
            for item, _ in doc.iterate_items():
                if isinstance(item, TitleItem | SectionHeaderItem):
                    refs.add(item.self_ref)
            return refs

    def _extract_page_summaries(self, doc: DoclingDocument) -> dict[int, str]:
        """Extract page-level summaries from enriched document.

        For page-level enrichment, the summary is stored in the meta field of the
        first document item on each page.

        TODO: Consider storing page summaries in a dedicated field in the future.
        TODO: Check the case of a summary in an item with multiple page provenances.

        Args:
            doc: The enriched DoclingDocument

        Returns:
            Dictionary mapping page number (1-indexed) to summary text
        """
        page_summaries: dict[int, str] = {}

        for item, _ in doc.iterate_items():
            if not isinstance(item, DocItem) or not item.prov:
                continue
            page_num = item.prov[0].page_no
            # Only process if we haven't seen this page yet
            if page_num not in page_summaries:
                # Check if item has meta with summary
                if item.meta and item.meta.summary and item.meta.summary.text:
                    page_summaries[page_num] = item.meta.summary.text
                    break  # Only need a summary per page
        return page_summaries

    def _extract_page_keyphrases(self, doc: DoclingDocument) -> dict[int, list[str]]:
        """Extract page-level keyphrases from enriched document.

        For page-level enrichment, keyphrases are stored in the meta field of the
        first document item on each page.

        TODO: Consider storing page summaries in a dedicated field in the future.
        TODO: Check the case of keyphrases in an item with multiple page provenances.

        Args:
            doc: The enriched DoclingDocument

        Returns:
            Dictionary mapping page number (1-indexed) to keyphrases
        """
        page_keyphrases: dict[int, list[str]] = {}

        for item, _ in doc.iterate_items():
            if not isinstance(item, DocItem) or not item.prov:
                continue
            page_num = item.prov[0].page_no
            if page_num not in page_keyphrases:
                if item.meta and item.meta.keywords and item.meta.keywords.values:
                    page_keyphrases[page_num] = item.meta.keywords.values
                    break
        return page_keyphrases

    # ------------------------------------------------------------------
    # Section selection
    # ------------------------------------------------------------------

    def _select_section(
        self,
        *,
        m: Any,
        query: str,
        outline_text: str,
        valid_refs: set[str],
        visited: set[str],
    ) -> SectionSelection:
        unvisited = sorted(valid_refs - visited)

        prompt = (
            f"Query: {query}\n\n"
            f"Document outline (with summaries):\n{outline_text}\n\n"
            f"Already consulted section refs: {sorted(visited) or 'none'}\n\n"
            f"Unvisited section refs to choose from: {unvisited}\n\n"
            "Select the single most relevant UNVISITED section ref to consult next. "
            "Return a JSON object in a ```json``` block with exactly two keys:\n"
            '  "reason": your chain-of-thought for why this section is relevant (string)\n'
            '  "section_ref": the exact ref string from the unvisited list above (string)'
        )

        def _validate(content: str) -> bool:
            dicts = find_json_dicts(content)
            if len(dicts) != 1:
                return False
            d = dicts[0]
            return (
                isinstance(d.get("reason"), str)
                and isinstance(d.get("section_ref"), str)
                and d["section_ref"] in unvisited
            )

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description=(
                        f"Return one JSON object with 'reason' (string) and 'section_ref' (one of: {unvisited})"
                    ),
                    validation_fn=simple_validate(_validate),
                ),
            ],
            retry_budget=3,
        )

        view_linear_context(m)

        dicts = find_json_dicts(answer)
        d = dicts[0] if dicts else {}
        if not isinstance(d.get("reason"), str) or d.get("section_ref") not in unvisited:
            # Rejection sampling exhausted without a valid response; pick first unvisited
            return SectionSelection(reason="fallback", section_ref=unvisited[0])
        return SectionSelection(reason=d["reason"], section_ref=d["section_ref"])

    # ------------------------------------------------------------------
    # Section content
    # ------------------------------------------------------------------

    def _get_section_content(self, doc: DoclingDocument, section_ref: str) -> str:
        """Return all text belonging to the given section node or page."""
        # Handle page-level mode
        if self.use_page_level and section_ref.startswith("#/pages/"):
            return self._get_page_content(doc, section_ref)

        node = get_item_by_ref(doc, section_ref)
        if node is None:
            log_warning(f"Could not resolve section ref {section_ref!r}")
            return ""

        # For hierarchical docs, collect_subtree_text gathers all nested content.
        # For flat docs, only the header text itself will be returned; we supplement
        # with level-based sibling scanning below.
        subtree = collect_subtree_text(node, doc)

        if len(node.children or []) == 0 and isinstance(node, TitleItem | SectionHeaderItem):
            # Flat document: scan forward until next same-or-higher section
            subtree = self._collect_flat_section_text(doc, section_ref)

        return subtree

    def _get_page_content(self, doc: DoclingDocument, page_ref: str) -> str:
        """Extract all text content from a specific page.

        Serializes a document page into text using the markdown format.
        Tables are serialized using the HTML format in order to capture
        nested rich content.

        TODO: Sync with _summarize_pages in enricher.py
        TODO: replace page_ref parameter by page number (1-indexed)
        TODO: reuse the serializer if this method is called multiple times
        """
        page_no: int = int(page_ref.split("/")[-1])

        # Markdown serialization parameters
        md_params = MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="",
            escape_underscores=False,
            escape_html=False,
            compact_tables=True,
            traverse_pictures=True,
        )

        serializer = MarkdownDocSerializer(doc=doc, table_serializer=HTMLTableSerializer(), params=md_params)
        page_text = serializer.serialize(pages={page_no}).text

        return page_text

    def _collect_flat_section_text(self, doc: DoclingDocument, section_ref: str) -> str:
        """Scan iterate_items for the section and collect siblings until next section."""
        texts: list[str] = []
        in_section = False
        section_level: int | None = None

        for item, depth in doc.iterate_items():
            if item.self_ref == section_ref:
                in_section = True
                section_level = depth
                if hasattr(item, "text") and item.text:
                    texts.append(item.text)
                continue

            if in_section:
                if (
                    isinstance(item, TitleItem | SectionHeaderItem)
                    and section_level is not None
                    and depth <= section_level
                ):
                    break
                if hasattr(item, "text") and item.text:
                    texts.append(item.text)

        return "\n\n".join(texts)

    # ------------------------------------------------------------------
    # Answer attempt
    # ------------------------------------------------------------------

    def _attempt_answer(
        self,
        *,
        m: Any,
        query: str,
        section_ref: str,
        section_text: str,
    ) -> AnswerAttempt:
        prompt = (
            f"Content of section '{section_ref}':\n\n{section_text}\n\n"
            f"Based on all context provided so far, can you answer: '{query}'?\n\n"
            "Return a JSON object in a ```json``` block with exactly two keys:\n"
            '  "can_answer": true if you have enough information to answer, false otherwise\n'
            '  "response": the full answer if can_answer is true, or what is still missing if false'
        )

        def _validate(content: str) -> bool:
            dicts = find_json_dicts(content)
            if len(dicts) != 1:
                return False
            d = dicts[0]
            return isinstance(d.get("can_answer"), bool) and isinstance(d.get("response"), str)

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Return one JSON object with 'can_answer' (boolean) and 'response' (string)",
                    validation_fn=simple_validate(_validate),
                ),
            ],
            retry_budget=3,
        )

        view_linear_context(m)

        d = find_json_dicts(answer)[0]
        return AnswerAttempt(can_answer=d["can_answer"], response=d["response"])

    # ------------------------------------------------------------------
    # Multi-document merging
    # ------------------------------------------------------------------

    def _merge_answers(self, *, query: str, answers: list[str]) -> str:
        if len(answers) == 1:
            return answers[0]

        m = self._create_writing_session(
            system_prompt=(
                "You are a precise scientific writer. "
                "Synthesize the provided partial answers into a single coherent response."
            )
        )
        formatted = "\n\n".join(f"[Source {i + 1}]\n{a}" for i, a in enumerate(answers))
        answer = m.instruct(
            f"Query: {query}\n\nPartial answers:\n{formatted}\n\nSynthesize a final answer.",
            retry_budget=3,
        )

        view_linear_context(m)

        return answer.strip()

    # ------------------------------------------------------------------
    # Document selection
    # ------------------------------------------------------------------

    def _extract_document_summary(self, doc: DoclingDocument) -> str:
        """Extract document-level summary from enriched document.

        The summary is stored in the meta field of the root body item.
        If no summary is found, return a fallback string with the document name.
        """

        item: NodeItem | None = get_item_by_ref(doc, "#/body")
        if item and item.meta and item.meta.summary and item.meta.summary.text:
            return item.meta.summary.text
        else:
            log_warning(f"No document summary found for {doc.name}")
            return f"Document: {doc.name}"

    def _select_relevant_documents(
        self,
        *,
        query: str,
        documents: dict[str, DoclingDocument],
        doc_summaries: dict[str, str],
    ) -> list[str]:
        """Select which documents are relevant for answering the query.

        This removes evaluation bias by not assuming we know which document
        contains the answer. The model must decide based on document summaries.

        Args:
            query: The query to answer
            documents: Dictionary mapping doc_id to DoclingDocument
            doc_summaries: Dictionary mapping doc_id to document summary

        Returns:
            List of relevant doc_ids
        """
        log_info(f"Selecting relevant documents from {len(documents)} candidates")

        # Build prompt with all document summaries
        doc_list = "\n".join([f"Document '{doc_id}':\n{summary}" for doc_id, summary in doc_summaries.items()])

        prompt = f"""You are analyzing a collection of documents to find which ones are relevant for answering a query.

QUERY:
{query}

AVAILABLE DOCUMENTS:
{doc_list}

TASK:
Identify the MOST relevant document(s) for answering the query. Be selective and precise.

IMPORTANT GUIDELINES:
- Prefer selecting few documents that are highly relevant
- Only select additional documents if they provide essential complementary information
- Do NOT select documents just because they might be tangentially related
- Quality over quantity: it's better to select fewer, highly relevant documents than many loosely related ones
- If the query is specific to one company/topic, typically only 1 document is needed
- If the query compares multiple entities, select only the documents for those specific entities

Format your response as:
Document 'doc_id': [reason]

Only include documents that are actually relevant to the query.
"""

        try:
            # Get reasoning model
            m = self._create_reasoning_session(system_prompt=self._RAG_SYSTEM_PROMPT)
            response = m.instruct(prompt, retry_budget=3)

            view_linear_context(m)

            # Parse response to extract doc_ids. Use word-boundary matching so that
            # a short id is not spuriously matched inside a longer one.
            selected_docs = [
                doc_id for doc_id in documents.keys() if re.search(rf"(?<!\w){re.escape(doc_id)}(?!\w)", response)
            ]

            if not selected_docs:
                log_warning("No documents selected by model, using all documents as fallback")
                selected_docs = list(documents.keys())

            log_info(f"Selected {len(selected_docs)} relevant document(s): {selected_docs}")
            return selected_docs

        except Exception as e:
            log_warning(f"Error selecting documents: {e}")
            # Fallback to all documents
            return list(documents.keys())


# ---------------------------------------------------------------------------
# Page-level selectors for RAG evaluation
# ---------------------------------------------------------------------------


class ReasoningBasedPageSelector:
    """Selects top-K pages using a reasoning model and per-page enrichment metadata.

    Implements an iterative batch-based approach: pages are evaluated in sliding
    windows with current candidates re-scored in each new batch so that all scores
    are comparable.  Works with both page-level and element-level step-3 enrichment.

    Args:
        backend: LLM backend for reasoning calls.
        k: Maximum number of pages to return.
        batch_size: Pages evaluated per reasoning iteration (candidates + new pages).
        early_stopping_threshold: Reserved for future use.
        summarization_style: ``"sentences"`` reads ``meta.summary``;
            ``"keyphrases"`` reads ``meta.keywords``.
    """

    def __init__(
        self,
        backend: BaseBackend,
        k: int = 10,
        batch_size: int = 30,
        early_stopping_threshold: float = 0.95,
        summarization_style: Literal["sentences", "keyphrases"] = "sentences",
    ) -> None:
        self.backend = backend
        self.k = k
        self.batch_size = batch_size
        self.early_stopping_threshold = early_stopping_threshold
        self.summarization_style: Literal["sentences", "keyphrases"] = summarization_style

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_relevant_documents(
        self,
        query: str,
        documents: dict[str, DoclingDocument],
        doc_summaries: dict[str, str],
    ) -> list[str]:
        """Select which documents are relevant for answering the query.

        The model decides based on document-level summaries without peeking at
        ground-truth labels (no evaluation bias).

        Args:
            query: The query to answer.
            documents: Mapping of doc_id → DoclingDocument.
            doc_summaries: Mapping of doc_id → document context string.

        Returns:
            List of relevant doc_ids.
        """
        log_info(f"[ReasoningBased] Selecting relevant documents from {len(documents)} candidates")

        doc_list = "\n".join(f"Document '{doc_id}':\n{summary}" for doc_id, summary in doc_summaries.items())

        prompt = f"""You are analyzing a collection of documents to find which ones are relevant for answering a query.

QUERY:
{query}

AVAILABLE DOCUMENTS:
{doc_list}

TASK:
Identify the MOST relevant document(s) for answering the query. Be selective and precise.

IMPORTANT GUIDELINES:
- Prefer selecting 1-2 documents that are highly relevant
- Only select additional documents if they provide essential complementary information
- Do NOT select documents just because they might be tangentially related
- Quality over quantity: it's better to select fewer, highly relevant documents than many loosely related ones
- If the query is specific to one company/topic, typically only 1 document is needed
- If the query compares multiple entities, select only the documents for those specific entities

Format your response as:
Document 'doc_id': [reason]

Only include documents that are actually relevant to the query.
"""
        try:
            model_name = self.backend.config.models.reasoning if self.backend.config.models else "default"
            session = self.backend.create_session(model=model_name)
            response = session.instruct(prompt=prompt)

            selected_docs = [
                doc_id for doc_id in documents if re.search(rf"(?<!\w){re.escape(doc_id)}(?!\w)", response)
            ]

            if not selected_docs:
                log_warning("[ReasoningBased] No documents selected by model, using all as fallback")
                selected_docs = list(documents.keys())

            log_info(f"[ReasoningBased] Selected {len(selected_docs)} document(s): {selected_docs}")
            return selected_docs

        except Exception as e:
            log_warning(f"[ReasoningBased] Error selecting documents: {e}")
            return list(documents.keys())

    def select_pages(
        self,
        query: str,
        document: DoclingDocument,
        doc_summary: str,
    ) -> list[tuple[int, float]]:
        """Iteratively select top-K pages using the reasoning model.

        Pages are evaluated in batches; the current candidate set is re-evaluated
        together with each new batch so all scores remain comparable.

        Args:
            query: The query to answer.
            document: The enriched DoclingDocument.
            doc_summary: Document-level context string.

        Returns:
            List of ``(page_number, relevance_score)`` tuples (1-indexed),
            sorted by descending relevance.
        """
        page_enrichment = self._extract_page_enrichment(document)

        if not page_enrichment:
            log_warning("[ReasoningBased] No page enrichment data found in document")
            return []

        log_info(f"[ReasoningBased] Selecting top {self.k} pages from {len(page_enrichment)} pages")

        remaining_pages = list(page_enrichment.keys())
        candidate_pages: dict[int, float] = {}
        batch_num = 0

        while remaining_pages:
            batch_num += 1
            available_slots = self.batch_size - len(candidate_pages)

            if available_slots <= 0:
                log_info(f"[ReasoningBased] Reached {len(candidate_pages)} candidates, finalizing")
                break

            new_pages = remaining_pages[:available_slots]
            remaining_pages = remaining_pages[available_slots:]
            batch_pages = list(candidate_pages.keys()) + new_pages

            log_info(
                f"[ReasoningBased] Batch {batch_num}: {len(candidate_pages)} candidates + "
                f"{len(new_pages)} new = {len(batch_pages)} total"
            )

            selected = self._evaluate_batch(
                query=query,
                doc_summary=doc_summary,
                batch_pages=batch_pages,
                page_summaries=page_enrichment,
                current_candidates=candidate_pages,
            )

            candidate_pages = dict(selected)

            if len(candidate_pages) > self.k:
                sorted_candidates = sorted(candidate_pages.items(), key=lambda x: x[1], reverse=True)
                candidate_pages = dict(sorted_candidates[: self.k])

            log_info(
                f"[ReasoningBased] Top candidates: "
                f"{sorted(candidate_pages.items(), key=lambda x: x[1], reverse=True)[:5]}"
            )

        sorted_pages = sorted(candidate_pages.items(), key=lambda x: x[1], reverse=True)[: self.k]
        log_info(f"[ReasoningBased] Final selection: {len(sorted_pages)} pages")
        return sorted_pages

    def rerank_across_documents(
        self,
        query: str,
        all_selected_pages: dict[str, list[tuple[int, float]]],
        doc_summaries: dict[str, str],
        page_summaries_by_doc: dict[str, dict[int, str]],
    ) -> list[tuple[str, int, float]]:
        """Re-rank top pages from multiple documents in a single shared context.

        Page scores from different documents are not directly comparable because
        they were evaluated in separate contexts.  This method issues one joint
        re-ranking request so all scores become comparable.

        Args:
            query: The query to answer.
            all_selected_pages: Mapping doc_id → ``[(page_num, score), …]``.
            doc_summaries: Mapping doc_id → document context string.
            page_summaries_by_doc: Mapping doc_id → ``{page_num: enrichment_text}``.

        Returns:
            List of ``(doc_id, page_num, score)`` tuples sorted by descending score.
        """
        if len(all_selected_pages) <= 1:
            result = [
                (doc_id, page_num, score) for doc_id, pages in all_selected_pages.items() for page_num, score in pages
            ]
            return sorted(result, key=lambda x: x[2], reverse=True)[: self.k]

        log_info(f"[ReasoningBased] Re-ranking pages across {len(all_selected_pages)} documents")

        if self.summarization_style == "keyphrases":
            context_label = "PAGE KEYWORDS"
            no_data_placeholder = "No keywords available"
            task_hint = (
                "Each page is described by a set of keyphrases. "
                "Use these keyphrases to judge how well the page content matches the query."
            )
        else:
            context_label = "PAGE SUMMARIES"
            no_data_placeholder = "No summary available"
            task_hint = (
                "Each page is described by a short prose summary. "
                "Use these summaries to judge how well the page content matches the query."
            )

        pages_by_doc = []
        for doc_id in sorted(all_selected_pages.keys()):
            pages = all_selected_pages[doc_id]
            page_enrichment = page_summaries_by_doc[doc_id]
            doc_summary = doc_summaries[doc_id]
            doc_pages_text = f"\nDOCUMENT: {doc_id}\nDOCUMENT CONTEXT: {doc_summary}\n{context_label}:\n"
            for page_num, _ in pages:
                entry = page_enrichment.get(page_num, no_data_placeholder)
                doc_pages_text += f"  Page {page_num}: {entry}\n"
            pages_by_doc.append(doc_pages_text)

        prompt = f"""You are analyzing pages from multiple documents to find the most relevant pages for answering a query.

{task_hint}

QUERY:
{query}

CANDIDATE PAGES FROM MULTIPLE DOCUMENTS:
{"".join(pages_by_doc)}

TASK:
Re-evaluate ALL pages listed above and select the top {self.k} most relevant pages for answering the query.
Compare pages across ALL documents to determine which are most relevant.

For each relevant page, provide:
1. Document ID
2. Page number
3. Relevance score (0.0 to 1.0, where 1.0 is highly relevant)
4. Brief reason

IMPORTANT:
- Evaluate pages from all documents in the same context
- Ensure scores are comparable across documents
- Only return pages that are actually relevant to the query
- You may return fewer than {self.k} pages if not enough are relevant

Format your response as:
Document [doc_id], Page X: [score] - [reason]

If no pages are relevant, respond with "No relevant pages."
"""
        try:
            model_name = self.backend.config.models.reasoning if self.backend.config.models else "default"
            session = self.backend.create_session(model=model_name)
            response = session.instruct(prompt=prompt)

            pattern = r"Document\s+([^,]+),\s*Page\s+(\d+):\s*\[?([0-9.]+)\]?"
            matches = re.findall(pattern, response, re.IGNORECASE)

            reranked_pages = []
            for doc_id_raw, page_str, score_str in matches:
                try:
                    doc_id = doc_id_raw.strip()
                    page_num = int(page_str)
                    score = float(score_str)
                    if doc_id in all_selected_pages:
                        original_pages = [p[0] for p in all_selected_pages[doc_id]]
                        if page_num in original_pages:
                            score = max(0.0, min(1.0, score))
                            reranked_pages.append((doc_id, page_num, score))
                            log_debug(f"[ReasoningBased] Re-ranked: {doc_id}, Page {page_num} score={score}")
                except (ValueError, TypeError):
                    continue

            if not reranked_pages:
                log_warning("[ReasoningBased] No pages parsed from re-ranking response, using original scores")
                result = [
                    (doc_id, page_num, score)
                    for doc_id, pages in all_selected_pages.items()
                    for page_num, score in pages
                ]
                return sorted(result, key=lambda x: x[2], reverse=True)[: self.k]

            reranked_pages = sorted(reranked_pages, key=lambda x: x[2], reverse=True)[: self.k]
            log_info(f"[ReasoningBased] Re-ranked {len(reranked_pages)} pages across documents")
            return reranked_pages

        except Exception as e:
            log_warning(f"[ReasoningBased] Error re-ranking pages: {e}")
            result = [
                (doc_id, page_num, score) for doc_id, pages in all_selected_pages.items() for page_num, score in pages
            ]
            return sorted(result, key=lambda x: x[2], reverse=True)[: self.k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_page_enrichment(self, document: DoclingDocument) -> dict[int, str]:
        """Extract per-page enrichment text from an enriched document.

        Reads ``meta.summary`` (style ``"sentences"``) or ``meta.keywords``
        (style ``"keyphrases"``) from the first matching item found on each page.
        Works for both page-level and element-level step-3 enrichment.

        Args:
            document: The enriched DoclingDocument.

        Returns:
            Mapping page_number (1-indexed) → enrichment string.
        """
        page_data: dict[int, str] = {}

        for item, _ in document.iterate_items():
            if not (hasattr(item, "prov") and item.prov):
                continue
            page_no = item.prov[0].page_no
            if page_no in page_data:
                continue
            if not (hasattr(item, "meta") and item.meta):
                continue

            text: str | None = None

            if self.summarization_style == "keyphrases":
                if isinstance(item.meta, dict):
                    kw_data = item.meta.get("keywords", {})
                    if isinstance(kw_data, dict):
                        values = kw_data.get("values", [])
                        if values:
                            text = "; ".join(str(v) for v in values)
                elif hasattr(item.meta, "keywords") and item.meta.keywords:
                    kw_values = item.meta.keywords.values
                    if kw_values:
                        text = "; ".join(str(v) for v in kw_values)
            else:
                if isinstance(item.meta, dict):
                    summary_data = item.meta.get("summary", {})
                    if isinstance(summary_data, dict):
                        text = summary_data.get("text") or None
                elif hasattr(item.meta, "summary") and item.meta.summary:
                    summary_obj = item.meta.summary
                    if hasattr(summary_obj, "text"):
                        text = summary_obj.text or None

            if text:
                page_data[page_no] = text

        return page_data

    def _evaluate_batch(
        self,
        query: str,
        doc_summary: str,
        batch_pages: list[int],
        page_summaries: dict[int, str],
        current_candidates: dict[int, float],
    ) -> list[tuple[int, float]]:
        """Call the reasoning model to score a batch of pages.

        Args:
            query: The query to answer.
            doc_summary: Document-level context string.
            batch_pages: Page numbers in this batch (candidates + new pages).
            page_summaries: All per-page enrichment strings.
            current_candidates: Current top-candidate scores (unused by model, kept for future).

        Returns:
            List of ``(page_number, score)`` tuples for relevant pages.
        """
        prompt = self._build_evaluation_prompt(
            query=query,
            doc_summary=doc_summary,
            batch_pages=batch_pages,
            page_summaries=page_summaries,
            k=self.k,
        )
        try:
            model_name = self.backend.config.models.reasoning if self.backend.config.models else "default"
            session = self.backend.create_session(model=model_name)
            response = session.instruct(prompt=prompt)
            return self._parse_model_response(response, batch_pages)
        except Exception as e:
            log_warning(f"[ReasoningBased] Error evaluating batch: {e}")
            return []

    def _build_evaluation_prompt(
        self,
        query: str,
        doc_summary: str,
        batch_pages: list[int],
        page_summaries: dict[int, str],
        k: int,
    ) -> str:
        """Build the per-batch evaluation prompt.

        The prompt is adapted to the enrichment style: ``"keyphrases"`` shows
        keyword lists; ``"sentences"`` shows prose summaries.
        """
        if self.summarization_style == "keyphrases":
            section_label = "PAGE KEYWORDS"
            task_hint = (
                "Each page is described by a set of keyphrases extracted from its content. "
                "Use these keyphrases to judge how well the page content matches the query."
            )
        else:
            section_label = "PAGE SUMMARIES"
            task_hint = (
                "Each page is described by a short prose summary of its content. "
                "Use these summaries to judge how well the page content matches the query."
            )

        page_entries = "\n".join(f"Page {p}: {page_summaries[p]}" for p in batch_pages)

        return f"""You are analyzing a document to find the most relevant pages for answering a query.

{task_hint}

DOCUMENT CONTEXT:
{doc_summary}

QUERY:
{query}

{section_label}:
{page_entries}

TASK:
Evaluate ALL pages listed above and identify the top {k} most relevant pages for answering the query.
Compare all pages against each other to determine relative relevance.

For each relevant page, provide:
1. Page number
2. Relevance score (0.0 to 1.0, where 1.0 is highly relevant)
3. Brief reason

IMPORTANT:
- Evaluate all pages in the same context to ensure comparable scores
- Only return pages that are actually relevant to the query
- Rank pages by relevance, with higher scores for more relevant pages
- You may return fewer than {k} pages if not enough are relevant

Format your response as:
Page X: [score] - [reason]

If no pages are relevant, respond with "No relevant pages."
"""

    def _parse_model_response(
        self,
        response: str,
        batch_pages: list[int],
    ) -> list[tuple[int, float]]:
        """Parse ``Page X: [score]`` entries from the model response.

        Args:
            response: Raw model response text.
            batch_pages: Valid page numbers for this batch (others are discarded).

        Returns:
            List of ``(page_number, score)`` tuples.
        """
        selected_pages: list[tuple[int, float]] = []
        pattern = r"Page\s+(\d+):\s*\[?([0-9.]+)\]?"
        for page_str, score_str in re.findall(pattern, response, re.IGNORECASE):
            try:
                page_num = int(page_str)
                score = float(score_str)
                if page_num in batch_pages:
                    selected_pages.append((page_num, max(0.0, min(1.0, score))))
                    log_debug(f"[ReasoningBased] Parsed: Page {page_num} score={score}")
            except (ValueError, TypeError):
                continue
        if not selected_pages:
            log_debug(f"[ReasoningBased] No pages parsed. Response preview: {response[:200]}")
        return selected_pages


class TreeGuidedPageSelector:
    """Selects top-K pages by tree-guided traversal of the document heading structure.

    Unlike :class:`ReasoningBasedPageSelector` which evaluates pages in flat batches,
    this selector navigates the document's hierarchical heading tree:

    1. **Document selection** — the model picks relevant document(s) from the corpus.
    2. **Top-level scan** — the model sees L1 section headings with their enrichment
       text and selects the most promising subset.
    3. **Drill-down loop** — for each selected heading the model decides:

       - ``"stop"``     → the heading contains enough evidence.
       - ``"drill"``    → explore the children of the selected headings.
       - ``"siblings"`` → explore neighbouring headings at the same level.

    4. **Convergence** — the loop ends when the model is confident or the iteration
       budget is exhausted.
    5. **Page extraction** — page numbers of all visited nodes are collected and
       returned as up to K ``(page, score)`` tuples.

    Requires that step 3 was run with **element-level** enrichment so that each
    :class:`~docling_core.types.doc.SectionHeaderItem` (and
    :class:`~docling_core.types.doc.TitleItem`) has ``meta.summary`` or
    ``meta.keywords`` populated.

    Args:
        backend: LLM backend for reasoning calls.
        k: Maximum number of pages to return.
        max_iterations: Maximum drill-down iterations per document.
        summarization_style: ``"sentences"`` reads ``meta.summary``;
            ``"keyphrases"`` reads ``meta.keywords``.
    """

    _MAX_NODES_PER_PROMPT: int = 40

    def __init__(
        self,
        backend: BaseBackend,
        k: int = 10,
        max_iterations: int = 8,
        summarization_style: Literal["sentences", "keyphrases"] = "sentences",
    ) -> None:
        self.backend = backend
        self.k = k
        self.max_iterations = max_iterations
        self.summarization_style: Literal["sentences", "keyphrases"] = summarization_style

    # ------------------------------------------------------------------
    # Public interface (mirrors ReasoningBasedPageSelector)
    # ------------------------------------------------------------------

    def select_relevant_documents(
        self,
        query: str,
        documents: dict[str, DoclingDocument],
        doc_summaries: dict[str, str],
    ) -> list[str]:
        """Select relevant documents — same logic as :class:`ReasoningBasedPageSelector`."""
        log_info(f"[TreeGuided] Selecting relevant documents from {len(documents)} candidates")

        doc_list = "\n".join(f"Document '{doc_id}':\n{summary}" for doc_id, summary in doc_summaries.items())
        prompt = f"""You are analyzing a collection of documents to find which ones are relevant for answering a query.

QUERY:
{query}

AVAILABLE DOCUMENTS:
{doc_list}

TASK:
Identify the MOST relevant document(s) for answering the query. Be selective and precise.

IMPORTANT GUIDELINES:
- Prefer selecting 1-2 documents that are highly relevant
- Only select additional documents if they provide essential complementary information
- Do NOT select documents just because they might be tangentially related
- If the query is specific to one company/topic, typically only 1 document is needed
- If the query compares multiple entities, select only the documents for those specific entities

Format your response as:
Document 'doc_id': [reason]

Only include documents that are actually relevant to the query.
"""
        try:
            model_name = self.backend.config.models.reasoning if self.backend.config.models else "default"
            session = self.backend.create_session(model=model_name)
            response = session.instruct(prompt=prompt)
            selected_docs = [
                doc_id for doc_id in documents if re.search(rf"(?<!\w){re.escape(doc_id)}(?!\w)", response)
            ]
            if not selected_docs:
                log_warning("[TreeGuided] No documents selected by model, using all as fallback")
                selected_docs = list(documents.keys())
            log_info(f"[TreeGuided] Selected {len(selected_docs)} document(s): {selected_docs}")
            return selected_docs
        except Exception as e:
            log_warning(f"[TreeGuided] Error selecting documents: {e}")
            return list(documents.keys())

    def select_pages(
        self,
        query: str,
        document: DoclingDocument,
        doc_summary: str,
    ) -> list[tuple[int, float]]:
        """Run tree-guided traversal and return up to K ``(page_number, score)`` tuples.

        Args:
            query: The query to answer.
            document: The enriched DoclingDocument (element-level enrichment expected).
            doc_summary: Document-level context string.

        Returns:
            List of ``(page_number, score)`` tuples sorted by descending score, 1-indexed.
        """
        l1_nodes = self._get_heading_nodes(document, parent_ref=None)
        if not l1_nodes:
            log_warning("[TreeGuided] No heading nodes with enrichment found")
            return []

        log_info(f"[TreeGuided] Starting tree traversal: {len(l1_nodes)} top-level headings")

        visited_refs: dict[str, tuple[int, float]] = {}
        frontier = l1_nodes[: self._MAX_NODES_PER_PROMPT]

        for iteration in range(self.max_iterations):
            if not frontier:
                log_info("[TreeGuided] Empty frontier, stopping")
                break

            log_info(f"[TreeGuided] Iteration {iteration + 1}: frontier size={len(frontier)}")

            decision = self._ask_model(
                query=query,
                doc_summary=doc_summary,
                frontier=frontier,
                visited_refs=set(visited_refs.keys()),
                iteration=iteration,
            )

            if decision is None:
                log_warning("[TreeGuided] Model returned unparseable decision, stopping")
                break

            selected_refs: list[str] = decision.get("selected_refs", [])
            action: str = decision.get("action", "stop")
            confident: bool = decision.get("confident", False)

            frontier_refs = {n["ref"] for n in frontier}
            selected_refs = [r for r in selected_refs if r in frontier_refs]

            if not selected_refs:
                log_info("[TreeGuided] Model selected no refs from frontier, stopping")
                break

            base_score = 1.0 - (iteration * 0.05)
            for rank, ref in enumerate(selected_refs):
                if ref not in visited_refs:
                    node = next((n for n in frontier if n["ref"] == ref), None)
                    page = node["page"] if node else None
                    if page is not None:
                        score = max(0.1, base_score - rank * 0.03)
                        visited_refs[ref] = (page, score)

            log_info(
                f"[TreeGuided] Selected {len(selected_refs)} ref(s): {selected_refs[:5]} | "
                f"action={action!r} confident={confident}"
            )

            if confident or action == "stop":
                log_info("[TreeGuided] Model is confident, stopping traversal")
                break

            if action == "drill":
                new_frontier: list[dict] = []
                for ref in selected_refs:
                    children = self._get_heading_nodes(document, parent_ref=ref)
                    if children:
                        new_frontier.extend(children)
                if not new_frontier:
                    action = "siblings"
                else:
                    frontier = new_frontier[: self._MAX_NODES_PER_PROMPT]
                    continue

            if action == "siblings":
                new_frontier = []
                for ref in selected_refs:
                    siblings = self._get_sibling_nodes(document, ref, l1_nodes)
                    new_frontier.extend(s for s in siblings if s["ref"] not in visited_refs)
                seen: set[str] = set()
                deduped: list[dict] = []
                for n in new_frontier:
                    if n["ref"] not in seen:
                        seen.add(n["ref"])
                        deduped.append(n)
                frontier = deduped[: self._MAX_NODES_PER_PROMPT]
                if not frontier:
                    log_info("[TreeGuided] No unvisited siblings found, stopping")
                    break
                continue

            break  # action == "stop" or unknown

        page_scores: dict[int, float] = {}
        for _ref, (page, score) in visited_refs.items():
            if page not in page_scores or score > page_scores[page]:
                page_scores[page] = score

        result = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[: self.k]
        log_info(f"[TreeGuided] Final pages: {result}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_enrichment_text(self, item: NodeItem) -> str | None:
        """Return the enrichment string for a heading item, or ``None`` if absent."""
        if not (hasattr(item, "meta") and item.meta):
            return None
        if self.summarization_style == "keyphrases":
            if isinstance(item.meta, dict):
                kw_data = item.meta.get("keywords", {})
                if isinstance(kw_data, dict):
                    values = kw_data.get("values", [])
                    if values:
                        return "; ".join(str(v) for v in values)
            elif hasattr(item.meta, "keywords") and item.meta.keywords:
                kw_values = item.meta.keywords.values
                if kw_values:
                    return "; ".join(str(v) for v in kw_values)
        else:
            if isinstance(item.meta, dict):
                summary_data = item.meta.get("summary", {})
                if isinstance(summary_data, dict):
                    return summary_data.get("text") or None
            elif hasattr(item.meta, "summary") and item.meta.summary:
                text = getattr(item.meta.summary, "text", None)
                if text:
                    return text
        return None

    def _get_heading_nodes(
        self,
        document: DoclingDocument,
        parent_ref: str | None,
    ) -> list[dict]:
        """Return enriched heading nodes that are direct children of *parent_ref*.

        If *parent_ref* is ``None``, the direct children of ``document.body`` are
        used (i.e. the L1 headings).

        Each returned dict has the keys: ``ref``, ``text``, ``page``,
        ``enrichment``, ``n_children``.  Headings without enrichment data are
        excluded.
        """
        if parent_ref is None:
            parent_node: NodeItem = document.body
        else:
            try:
                parent_node = RefItem(cref=parent_ref).resolve(document)
            except Exception:
                return []

        nodes: list[dict] = []
        for child_ref in parent_node.children or []:
            try:
                child = child_ref.resolve(document)
            except Exception:
                continue
            if not isinstance(child, (SectionHeaderItem, TitleItem)):
                continue

            enrichment = self._get_enrichment_text(child)
            if not enrichment:
                continue

            page = child.prov[0].page_no if hasattr(child, "prov") and child.prov else None
            if page is None:
                continue

            nodes.append(
                {
                    "ref": child.self_ref,
                    "text": child.text if hasattr(child, "text") and child.text else "(untitled)",
                    "page": page,
                    "enrichment": enrichment,
                    "n_children": len(child.children or []),
                }
            )
        return nodes

    def _get_sibling_nodes(
        self,
        document: DoclingDocument,
        ref: str,
        l1_nodes: list[dict],
    ) -> list[dict]:
        """Return siblings of *ref* (other nodes at the same level under the same parent).

        For L1 headings the siblings are all other entries in *l1_nodes*.
        For deeper nodes the parent is located by scanning body children.
        """
        if any(n["ref"] == ref for n in l1_nodes):
            return [n for n in l1_nodes if n["ref"] != ref]

        for candidate_ref_item in document.body.children or []:
            try:
                candidate = candidate_ref_item.resolve(document)
            except Exception:
                continue
            if not isinstance(candidate, (SectionHeaderItem, TitleItem)):
                continue
            child_refs = [cr.cref for cr in (candidate.children or [])]
            if ref in child_refs:
                return self._get_heading_nodes(document, parent_ref=candidate.self_ref)

        return []

    def _ask_model(
        self,
        *,
        query: str,
        doc_summary: str,
        frontier: list[dict],
        visited_refs: set[str],
        iteration: int,
    ) -> dict | None:
        """Ask the reasoning model to select nodes from *frontier* and decide the next action.

        Returns a dict with:

        * ``selected_refs`` — list of ref strings chosen from the frontier.
        * ``action`` — one of ``"drill"``, ``"siblings"``, ``"stop"``.
        * ``confident`` — ``True`` if the model believes it found the answer.

        Returns ``None`` if the response cannot be parsed.
        """
        enrich_label = "KEYPHRASES" if self.summarization_style == "keyphrases" else "SUMMARY"
        enrich_hint = (
            "keyphrases extracted from their content"
            if self.summarization_style == "keyphrases"
            else "short prose summaries of their content"
        )

        frontier_lines = "\n".join(
            f"  ref={n['ref']!r}  page={n['page']}  children={n['n_children']}  "
            f"heading={n['text'][:60]!r}  {enrich_label}={n['enrichment'][:120]!r}"
            for n in frontier
        )
        visited_note = (
            f"\nAlready-visited refs (do not select these again): {sorted(visited_refs)}" if visited_refs else ""
        )

        prompt = f"""You are performing a tree-guided search through a document to find pages that answer a query.

DOCUMENT CONTEXT:
{doc_summary}

QUERY:
{query}

CURRENT CANDIDATE HEADINGS (each shown with {enrich_hint}):
{frontier_lines}{visited_note}

TASK (iteration {iteration + 1}):
1. Select the heading refs from the list above that are MOST LIKELY to contain information relevant to the query.
   You may select multiple refs if several are relevant.
2. Decide what to do next:
   - "drill"    → you want to explore the children of the selected headings for more detail
   - "siblings" → the selected headings are not quite right; explore their neighbours instead
   - "stop"     → the selected headings contain enough evidence to answer the query
3. Set "confident" to true if you believe the selected headings are sufficient to answer the query.

Return ONLY a JSON object in a ```json``` block with these keys:
  "selected_refs": list of ref strings chosen from the frontier above (must be exact matches)
  "action": one of "drill", "siblings", "stop"
  "confident": true or false
  "reason": brief explanation of your choice (string)
"""
        try:
            model_name = self.backend.config.models.reasoning if self.backend.config.models else "default"
            session = self.backend.create_session(model=model_name)
            response = session.instruct(prompt=prompt)

            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if not match:
                match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                log_warning("[TreeGuided] No JSON found in model response")
                return None

            data = json.loads(match.group(1) if "```" in response else match.group(0))

            if not isinstance(data.get("selected_refs"), list):
                data["selected_refs"] = []
            if data.get("action") not in ("drill", "siblings", "stop"):
                data["action"] = "stop"
            data["confident"] = bool(data.get("confident", False))
            return data

        except Exception as e:
            log_warning(f"[TreeGuided] Error calling model: {e}")
            return None

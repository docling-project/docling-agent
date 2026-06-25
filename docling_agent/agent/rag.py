"""Chunkless RAG agent using DoclingDocument tree structure and per-node summaries."""

from pathlib import Path
from typing import Any, ClassVar

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

        answer_doc = DoclingDocument(name="rag_answer")
        answer_doc.add_title(text="Answer", parent=answer_doc.body)
        answer_doc.add_text(label=DocItemLabel.TEXT, text=trace.final_answer, parent=answer_doc.body)
        return answer_doc

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
        return RAGTrace(query=task, per_document=per_document, final_answer=final_answer)

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

            # Parse response to extract doc_ids
            selected_docs = []
            for doc_id in documents.keys():
                # Look for the doc_id in the response
                if f"'{doc_id}'" in response or f'"{doc_id}"' in response or doc_id in response:
                    selected_docs.append(doc_id)

            if not selected_docs:
                log_warning("No documents selected by model, using all documents as fallback")
                selected_docs = list(documents.keys())

            log_info(f"Selected {len(selected_docs)} relevant document(s): {selected_docs}")
            return selected_docs

        except Exception as e:
            log_warning(f"Error selecting documents: {e}")
            # Fallback to all documents
            return list(documents.keys())

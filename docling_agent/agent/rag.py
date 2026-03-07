"""Chunkless RAG agent using DoclingDocument tree structure and per-node summaries."""

from pathlib import Path
from typing import Any, ClassVar

from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    collect_subtree_text,
    find_json_dicts,
    get_item_by_ref,
)
from docling_agent.agent.rag_models import (
    AnswerAttempt,
    RAGIteration,
    RAGResult,
    SectionSelection,
)
from docling_agent.agent_models import setup_local_session
from docling_agent.logging import logger


class DoclingRAGAgent(BaseDoclingAgent):
    """Chunkless RAG agent.

    Builds a compact document outline (with per-node summaries), lets the LLM
    iteratively select the most relevant section, reads only that section's
    content, and attempts to answer the query — without ever loading the full
    document into the context window.
    """

    _RAG_SYSTEM_PROMPT: ClassVar[str] = (
        "You are a precise research assistant. You are given a query and a document outline "
        "with per-section summaries. Your job is to iteratively select the most relevant sections "
        "and build an answer from their content. "
        "Always ground your answer in the document content. "
        "Do not hallucinate or add information not present in the retrieved sections."
    )

    max_iterations: int = 5
    verbose: bool = False

    def __init__(
        self,
        *,
        model_id: ModelIdentifier,
        tools: list,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_RAG,
            model_id=model_id,
            tools=tools,
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._console = Console(highlight=False) if verbose else None

    def _rprint(self, renderable: Any) -> None:
        """Print to the rich console only when verbose mode is enabled."""
        if self._console is not None:
            self._console.print(renderable)

    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] = [],
        **kwargs,
    ) -> DoclingDocument:
        docs = [s for s in sources if isinstance(s, DoclingDocument)]
        if not docs and document is not None:
            docs = [document]
        if not docs:
            raise ValueError("DoclingRAGAgent requires at least one DoclingDocument.")

        per_doc_answers: list[str] = []
        all_iterations: list[RAGIteration] = []

        for doc in docs:
            result = self._rag_loop(query=task, doc=doc)
            per_doc_answers.append(result.answer)
            all_iterations.extend(result.iterations)
            logger.info(
                f"RAG loop finished: converged={result.converged}, "
                f"iterations={len(result.iterations)}"
            )

        if len(docs) > 1:
            self._rprint(
                Rule(f"[bold cyan]Merging answers from {len(docs)} documents[/bold cyan]")
            )

        final_answer = self._merge_answers(query=task, answers=per_doc_answers)

        answer_doc = DoclingDocument(name="rag_answer")
        answer_doc.add_title(text="Answer", parent=answer_doc.body)
        answer_doc.add_text(
            label=DocItemLabel.TEXT, text=final_answer, parent=answer_doc.body
        )
        return answer_doc

    # ------------------------------------------------------------------
    # RAG loop
    # ------------------------------------------------------------------

    def _rag_loop(self, *, query: str, doc: DoclingDocument) -> RAGResult:
        m = setup_local_session(
            model_id=self.get_reasoning_model_id(),
            system_prompt=self._RAG_SYSTEM_PROMPT,
        )

        visited: set[str] = set()
        iterations: list[RAGIteration] = []

        outline_text = self._build_outline(doc)
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
            logger.warning("No section headers found; falling back to full document text.")
            self._rprint(
                Text("⚠ No section headers found — returning full document.", style="yellow")
            )
            from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
            full_text = MarkdownDocSerializer(doc=doc).serialize().text
            return RAGResult(answer=full_text, iterations=[], converged=True)

        for i in range(self.max_iterations):
            unvisited = valid_refs - visited
            if not unvisited:
                logger.info("All sections visited; stopping early.")
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
                    f"[bold]Selected:[/bold]  {selection.section_ref}\n"
                    f"[bold]Reason:[/bold]    {selection.reason}",
                    title=f"[cyan]Section Selection[/cyan]",
                    border_style="blue",
                )
            )

            section_text = self._get_section_content(doc, selection.section_ref)
            preview = section_text[:300].replace("\n", " ") + (
                " …" if len(section_text) > 300 else ""
            )
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

        last = iterations[-1] if iterations else RAGIteration(
            iteration=0, section_ref="", reason="", section_text_length=0,
            can_answer=False, response="No content could be retrieved.",
        )
        self._rprint(
            Panel(
                last.response,
                title=f"[bold yellow]Partial Answer (max iterations reached)[/bold yellow]",
                border_style="yellow",
            )
        )
        return RAGResult(
            answer=(
                f"[Partial answer after {len(iterations)} iteration(s)]\n\n"
                f"{last.response}"
            ),
            iterations=iterations,
            converged=False,
        )

    # ------------------------------------------------------------------
    # Outline
    # ------------------------------------------------------------------

    def _build_outline(self, doc: DoclingDocument) -> str:
        from docling_core.experimental.serializer.outline import (
            OutlineDocSerializer,
            OutlineMode,
            OutlineParams,
        )

        serializer = OutlineDocSerializer(
            doc=doc,
            params=OutlineParams(mode=OutlineMode.OUTLINE),
        )
        return serializer.serialize().text

    def _extract_section_refs(self, doc: DoclingDocument) -> set[str]:
        refs: set[str] = set()
        for item, _ in doc.iterate_items():
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                refs.add(item.self_ref)
        return refs

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
                        f"Return one JSON object with 'reason' (string) and "
                        f"'section_ref' (one of: {unvisited})"
                    ),
                    validation_fn=simple_validate(_validate),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )

        d = find_json_dicts(answer.value)[0]
        return SectionSelection(reason=d["reason"], section_ref=d["section_ref"])

    # ------------------------------------------------------------------
    # Section content
    # ------------------------------------------------------------------

    def _get_section_content(self, doc: DoclingDocument, section_ref: str) -> str:
        """Return all text belonging to the given section node."""
        node = get_item_by_ref(doc, section_ref)
        if node is None:
            logger.warning(f"Could not resolve section ref {section_ref!r}")
            return ""

        # For hierarchical docs, collect_subtree_text gathers all nested content.
        # For flat docs, only the header text itself will be returned; we supplement
        # with level-based sibling scanning below.
        subtree = collect_subtree_text(node, doc)

        if len(node.children or []) == 0 and isinstance(
            node, (TitleItem, SectionHeaderItem)
        ):
            # Flat document: scan forward until next same-or-higher section
            subtree = self._collect_flat_section_text(doc, section_ref)

        return subtree

    def _collect_flat_section_text(
        self, doc: DoclingDocument, section_ref: str
    ) -> str:
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
                if isinstance(item, (TitleItem, SectionHeaderItem)) and depth <= section_level:
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
            return isinstance(d.get("can_answer"), bool) and isinstance(
                d.get("response"), str
            )

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Return one JSON object with 'can_answer' (boolean) and 'response' (string)",
                    validation_fn=simple_validate(_validate),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )

        d = find_json_dicts(answer.value)[0]
        return AnswerAttempt(can_answer=d["can_answer"], response=d["response"])

    # ------------------------------------------------------------------
    # Multi-document merging
    # ------------------------------------------------------------------

    def _merge_answers(self, *, query: str, answers: list[str]) -> str:
        if len(answers) == 1:
            return answers[0]

        m = setup_local_session(
            model_id=self.get_writing_model_id(),
            system_prompt=(
                "You are a precise scientific writer. "
                "Synthesize the provided partial answers into a single coherent response."
            ),
        )
        formatted = "\n\n".join(
            f"[Source {i + 1}]\n{a}" for i, a in enumerate(answers)
        )
        answer = m.instruct(
            f"Query: {query}\n\nPartial answers:\n{formatted}\n\nSynthesize a final answer.",
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )
        return answer.value.strip()

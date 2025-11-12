import re
from pathlib import Path
from typing import ClassVar

from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    BaseMeta,
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    convert_html_to_docling_document,
    convert_markdown_to_docling_document,
    find_markdown_code_block,
    has_html_code_block,
    has_markdown_code_block,
    validate_html_to_docling_document,
    validate_markdown_to_docling_document,
)
from docling_agent.agent_models import setup_local_session
from docling_agent.logging import logger

# from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from docling_agent.resources.prompts import (
    SYSTEM_PROMPT_EXPERT_TABLE_WRITER,
    SYSTEM_PROMPT_EXPERT_WRITER,
    SYSTEM_PROMPT_FOR_OUTLINE,
    SYSTEM_PROMPT_FOR_TASK_ANALYSIS,
)


class DoclingWritingAgent(BaseDoclingAgent):
    task_analysis: DoclingDocument = DoclingDocument(name="report")

    system_prompt_for_task_analysis: ClassVar[str] = SYSTEM_PROMPT_FOR_TASK_ANALYSIS

    system_prompt_for_outline: ClassVar[str] = SYSTEM_PROMPT_FOR_OUTLINE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    system_prompt_expert_table_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_TABLE_WRITER

    _outline_descriptions: ClassVar[list[str]] = [
        "paragraph:",
        "list:",
        "table:",
        "figure:",
        "picture:",
    ]

    def __init__(self, *, model_id: ModelIdentifier, tools: list):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_WRITER,
            model_id=model_id,
            tools=tools,
        )

    def run(
        self,
        task: str,
        document: DoclingDocument | None = None,
        sources: list[DoclingDocument | Path] | None = None,
        **kwargs,
    ) -> DoclingDocument:
        # Avoid mutable default list for sources
        sources = sources or []
        # self._analyse_task_for_topics_and_followup_questions(task=task)

        # self._analyse_task_for_final_destination(task=task)

        # Plan an outline for the document
        outline: DoclingDocument = self._make_outline_for_writing(task=task)

        # Write the actual document item by item
        result_document: DoclingDocument = self._populate_document_with_content(
            task=task, outline=outline
        )

        return result_document

    def _analyse_task_for_final_destination(self, *, task: str):
        return

    def _make_outline_for_writing(
        self, *, task: str, loop_budget: int = 5
    ) -> DoclingDocument:
        m = setup_local_session(
            model_id=self.model_id, system_prompt=self.system_prompt_for_outline
        )

        answer = m.instruct(
            f"{task}",
            requirements=[
                Requirement(
                    description="Put the resulting markdown outline in the format ```markdown <insert-content>```",
                    validation_fn=simple_validate(has_markdown_code_block),
                ),
                Requirement(
                    description="The resulting outline should be in markdown format. If not a title or subheading, start each line with `paragraph: `, `table: `, `picture: ` or `list: ` followed by a single sentence summary.",
                    # validation_fn=simple_validate(validate_outline_format),
                    validation_fn=simple_validate(
                        DoclingWritingAgent._validate_outline_format
                    ),
                ),
            ],
            # user_variables={"name": name, "notes": notes},
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        outline = self._find_outline(text=answer.value, task=task)

        if outline is None:
            raise ValueError("Failed to generate a valid outline document.")

        return outline

    @staticmethod
    def _validate_outline_format(content: str) -> bool:
        """Validate that content contains a markdown outline with valid lines.

        Rules:
        - The outline must be inside a ```md or ```markdown code block.
        - Non-heading lines must start with one of:
          "paragraph:", "list:", "table:", "figure:", or "picture:",
          followed by a single sentence summary ending with a period.
        """

        # Extract markdown block
        md = find_markdown_code_block(content)
        if not md:
            return False

        # Parse markdown to DoclingDocument
        try:
            converter = DocumentConverter(allowed_formats=[InputFormat.MD])
            conv: ConversionResult = converter.convert_string(
                content=md, format=InputFormat.MD, name="outline.md"
            )
        except Exception:
            return False

        starts = ["paragraph", "list", "table", "figure", "picture"]
        pattern = re.compile(rf"^({'|'.join(starts)}):\s(.*)\.$")

        invalid_lines: list[str] = []

        # Validate content lines: only TextItem lines are checked
        for item, _ in conv.document.iterate_items(with_groups=True):
            if isinstance(item, TextItem):
                if not pattern.match(item.text):
                    invalid_lines.append(item.text)

        return len(invalid_lines) == 0

    def _find_outline(self, text: str, task: str) -> DoclingDocument | None:
        starts = ["paragraph", "list", "table", "figure", "picture"]

        md = find_markdown_code_block(text)

        if not md:
            return None

        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        conv: ConversionResult = converter.convert_string(
            content=md, format=InputFormat.MD, name=f"outline for {task}"
        )

        # Build a fresh outline document rather than deep-copying content
        outline = DoclingDocument(name=f"outline for: {conv.document.name}")

        invalid_lines: list[str] = []

        for item, level in conv.document.iterate_items(with_groups=True):
            if isinstance(item, TitleItem):
                outline.add_title(text=item.text)

            elif isinstance(item, SectionHeaderItem):
                outline.add_heading(text=item.text, level=item.level)

            elif isinstance(item, TextItem):
                pattern = rf"^({'|'.join(starts)}):\s(.*)\.$"
                match = re.match(pattern, item.text)

                if not match:
                    logger.warning(
                        f"line `{item.text}` does not match pattern `{pattern}`"
                    )
                    invalid_lines.append(item.text)
                    continue

                label = match[1]
                summary = match[2]

                meta = BaseMeta(summary=SummaryMetaField(text=summary))

                if label == "paragraph":
                    _ = outline.add_text(label=DocItemLabel.TEXT, text=item.text)
                    _.meta = meta

                elif label == "table":
                    # Create an empty placeholder table with summary in meta
                    caption = outline.add_text(label=DocItemLabel.CAPTION, text="")
                    data = TableData(table_cells=[], num_rows=0, num_cols=0)

                    _ = outline.add_table(
                        label=DocItemLabel.TABLE, data=data, caption=caption
                    )
                    _.meta = meta

                elif label in ["figure", "picture"]:
                    # Add a picture with a caption derived from the summary
                    caption = outline.add_text(label=DocItemLabel.CAPTION, text="")

                    _ = outline.add_picture(caption=caption)
                    _.meta = meta

                elif label == "list":
                    _ = outline.add_group(
                        name="list",
                        label=GroupLabel.LIST,  # , parent=None
                    )
                    _.meta = meta

                else:
                    logger.warning(f"label {label} is not supported")
            else:
                logger.warning(f"could not classify item: {item}")
                continue

        if len(invalid_lines) > 0:
            message = (
                "Every content line should start with one of: "
                f"{starts}. The following lines need to be updated: "
                + "\n".join(invalid_lines)
            )
            logger.error(message)
            return None

        # Debug serializer block removed

        return outline

    def _populate_document_with_content(
        self, *, task: str, outline: DoclingDocument, loop_budget: int = 5
    ) -> DoclingDocument:
        headers: dict[int, str] = {}

        document = DoclingDocument(name=f"report on task: {task}")

        for item, _ in outline.iterate_items(with_groups=True):
            headers = self._process_outline_item(
                document=document,
                headers=headers,
                item=item,
                loop_budget=loop_budget,
            )

        return document

    def _ordered_hierarchy(self, headers: dict[int, str]) -> dict[int, str]:
        return {lvl: headers[lvl] for lvl in sorted(headers.keys())}

    def _summary_for(self, *, node: NodeItem) -> str | None:
        if node.meta and node.meta.summary and node.meta.summary.text:
            return node.meta.summary.text
        logger.error(f"No summary found for {node}")
        return None

    def _write_and_merge(
        self,
        *,
        kind: str,
        summary: str,
        headers: dict[int, str],
        document: DoclingDocument,
        loop_budget: int,
    ) -> None:
        h = self._ordered_hierarchy(headers)
        if kind == "paragraph":
            content = self._write_paragraph(
                summary=summary, hierarchy=h, loop_budget=loop_budget
            )
        elif kind == "table":
            content = self._write_table(
                summary=summary, hierarchy=h, loop_budget=loop_budget
            )
        elif kind == "list":
            content = self._write_list(
                summary=summary, hierarchy=h, loop_budget=loop_budget
            )
        else:
            logger.warning(f"Unsupported content kind: {kind}")
            return

        self._update_document_with_content(document=document, content=content)

    def _update_headers(
        self, *, item: SectionHeaderItem, headers: dict[int, str]
    ) -> dict[int, str]:
        for k in [k for k in list(headers.keys()) if k > item.level]:
            del headers[k]
        headers[item.level] = item.text
        return headers

    def _process_outline_item(
        self,
        *,
        document: DoclingDocument,
        headers: dict[int, str],
        item: NodeItem,
        loop_budget: int,
    ) -> dict[int, str]:
        if isinstance(item, TitleItem):
            headers[0] = item.text
            document.add_title(text=item.text)

        elif isinstance(item, SectionHeaderItem):
            logger.info(f"starting in {item.text}")
            headers = self._update_headers(item=item, headers=headers)
            document.add_heading(text=item.text, level=item.level)

        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.CAPTION:
                return headers
            summary = self._summary_for(node=item)
            if summary:
                logger.info("writing a paragraph")
                self._write_and_merge(
                    kind="paragraph",
                    summary=summary,
                    headers=headers,
                    document=document,
                    loop_budget=loop_budget,
                )
            else:
                logger.warning(f"Skipping paragraph without summary: {item.text!r}")

        elif isinstance(item, TableItem):
            summary = self._summary_for(node=item)
            if summary:
                logger.info("writing a table")
                self._write_and_merge(
                    kind="table",
                    summary=summary,
                    headers=headers,
                    document=document,
                    loop_budget=loop_budget,
                )
            else:
                logger.warning("Skipping table without summary")

        elif isinstance(item, PictureItem):
            summary = self._summary_for(node=item)
            if summary:
                logger.info("writing a picture")
                caption = document.add_text(label=DocItemLabel.CAPTION, text=summary)
                document.add_picture(caption=caption)
            else:
                logger.warning("Skipping picture without summary")

        elif isinstance(item, GroupItem) and item.label == GroupLabel.LIST:
            summary = self._summary_for(node=item)
            if summary:
                logger.info("writing a list")
                self._write_and_merge(
                    kind="list",
                    summary=summary,
                    headers=headers,
                    document=document,
                    loop_budget=loop_budget,
                )
            else:
                logger.warning("Skipping list without summary")

        else:
            logger.info(f"Unhandled item: {item}")

        return headers

    def _update_document_with_content(
        self, *, document: DoclingDocument, content: DoclingDocument
    ) -> DoclingDocument:
        to_item: dict[str, NodeItem] = {}

        for item, level in content.iterate_items(with_groups=True):
            if isinstance(item, GroupItem) and item.self_ref == "#/body":
                to_item[item.self_ref] = document.body
            elif isinstance(item, GroupItem):
                if item.parent and item.parent.cref in to_item:
                    g = document.add_group(
                        name=item.name,
                        label=item.label,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = g
                else:
                    g = document.add_group(
                        name=item.name, label=item.label, parent=None
                    )
                    to_item[item.self_ref] = g

            elif isinstance(item, ListItem):
                if item.parent and item.parent.cref in to_item:
                    li = document.add_list_item(
                        text=item.text,
                        formatting=item.formatting,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = li
                else:
                    logger.debug(f"Skipping ListItem without known parent: {item}")

            elif isinstance(item, TextItem):
                if item.parent and item.parent.cref in to_item:
                    te = document.add_text(
                        text=item.text,
                        label=item.label,
                        formatting=item.formatting,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = te
                else:
                    logger.debug(f"Skipping TextItem without known parent: {item}")

            elif isinstance(item, TableItem):
                if item.parent and item.parent.cref in to_item:
                    if len(item.captions) > 0:
                        # Create an empty caption placeholder to avoid deref issues
                        caption = document.add_text(
                            label=DocItemLabel.CAPTION,
                            text="",
                            parent=to_item[item.parent.cref],
                        )
                        te = document.add_table(
                            data=item.data,
                            caption=caption,
                            parent=to_item[item.parent.cref],
                        )
                        to_item[item.self_ref] = te
                    else:
                        te = document.add_table(
                            data=item.data,
                            parent=to_item[item.parent.cref],
                        )
                        to_item[item.self_ref] = te
                else:
                    # print("skipping: ", item)
                    continue
            else:
                # print("skipping: ", item)
                continue

        return document

    def _write_paragraph(
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] | None = None,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        hierarchy = hierarchy or {}
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        if len(context) > 0:
            context = rf"Given the current context in the document:\n\n```markdown\n{context}```\n\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"{context}Write me a single paragraph that expands the following summary: {summary}"
        # logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="The resulting markdown paragraph should use latex notation for superscript, subscript or inline equations. This means that every superscript, subscript and inline equation in must start and end with a $ sign.",
                    validation_fn=simple_validate(
                        validate_markdown_to_docling_document
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        result = convert_markdown_to_docling_document(text=answer.value)
        return result if result is not None else DoclingDocument(name="content")

    def _write_table(
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] | None = None,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        hierarchy = hierarchy or {}
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"Given the current context in the document:\n\n```{context}```\n\nwrite me a single HTML table that expands the following summary: {summary}"
        # logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Put the resulting HTML table in the format ```html <insert-content>```",
                    validation_fn=simple_validate(has_html_code_block),
                ),
                Requirement(
                    description="The HTML table should have a valid formatting.",
                    validation_fn=simple_validate(validate_html_to_docling_document),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        result = convert_html_to_docling_document(text=answer.value)

        return result if result is not None else DoclingDocument(name="content")

    def _write_list(
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] | None = None,
        loop_budget: int = 5,
    ) -> DoclingDocument:
        hierarchy = hierarchy or {}
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"Given the current context in the document:\n\n```{context}```\n\nwrite me a list (can be nested) in markdown that expands the following summary: {summary}"
        # logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="The resulting markdown list should use latex notation for superscript, subscript or inline equations. This means that every superscript, subscript and inline equation in must start and end with a $ sign.",
                    validation_fn=simple_validate(
                        validate_markdown_to_docling_document
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        result = convert_markdown_to_docling_document(text=answer.value)

        return result if result is not None else DoclingDocument(name="content")

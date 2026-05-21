import re
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from docling_core.experimental.serializer.outline import OutlineFormat
from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from mellea.stdlib.requirements import Requirement, simple_validate
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from docling_agent.agent.base import BaseDoclingAgent, DoclingAgentType
from docling_agent.agent.base_functions import (
    convert_html_to_docling_table,
    convert_markdown_to_docling_document,
    create_document_outline,
    find_json_dicts,
    has_html_code_block,
    insert_document,
    serialize_item_to_markdown,
    serialize_table_to_html,
    validate_html_to_docling_table,
)
from docling_agent.agent_models import view_linear_context
from docling_agent.logging import log_debug, log_error, log_info, log_warning

# from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from docling_agent.resources.prompts import (
    SYSTEM_PROMPT_EXPERT_WRITER,
    SYSTEM_PROMPT_FOR_EDITING_DOCUMENT,
    SYSTEM_PROMPT_FOR_EDITING_TABLE,
)


# Pydantic models for operation validation
class UpdateContentOperation(BaseModel):
    """Operation to update content of a single document item."""

    operation: Literal["update_content"]
    ref: Annotated[str, Field(pattern=_JSON_POINTER_REGEX)]


class RewriteContentOperation(BaseModel):
    """Operation to rewrite content of multiple document items."""

    operation: Literal["rewrite_content"]
    refs: list[Annotated[str, Field(pattern=_JSON_POINTER_REGEX)]]


class SectionHeadingLevelChange(BaseModel):
    """A single section heading level change."""

    ref: Annotated[str, Field(pattern=_JSON_POINTER_REGEX)]
    to_level: Annotated[int, Field(ge=-1)]


class MissingSectionHeadingInsertion(BaseModel):
    """Insert a missing section heading inferred between two existing headings."""

    previous_ref: Annotated[str, Field(pattern=_JSON_POINTER_REGEX)]
    next_ref: Annotated[str, Field(pattern=_JSON_POINTER_REGEX)]
    regex: str
    level: Annotated[int, Field(ge=1)]


class UpdateSectionHeadingLevelOperation(BaseModel):
    """Operation to update section heading levels."""

    operation: Literal["update_section_heading_level"]
    changes: list[SectionHeadingLevelChange]
    insertions: list[MissingSectionHeadingInsertion] = Field(default_factory=list)


DocumentOperation = Annotated[
    UpdateContentOperation | RewriteContentOperation | UpdateSectionHeadingLevelOperation,
    Field(discriminator="operation"),
]


class DoclingEditingAgent(BaseDoclingAgent):
    system_prompt_for_editing_document: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_DOCUMENT
    system_prompt_for_editing_table: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_TABLE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    def __init__(
        self,
        *,
        tools: list,
        backend=None,
    ):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EDITOR,
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

        op = self._identify_document_items(task=task, document=document)

        # Type-safe dispatch using Pydantic models
        if isinstance(op, UpdateContentOperation):
            self._update_content(task=task, document=document, sref=op.ref)
        elif isinstance(op, RewriteContentOperation):
            self._rewrite_content(
                task=task,
                document=document,
                refs=op.refs,
            )
        elif isinstance(op, UpdateSectionHeadingLevelOperation):
            self._update_section_heading_level(
                task=task,
                document=document,
                changes=op.changes,
                insertions=op.insertions,
            )
        else:
            message = f"Could not execute operation: {op}"
            log_info(message)
            raise ValueError(message)

        return document

    def _identify_document_items(
        self,
        task: str,
        document: DoclingDocument,
        loop_budget: int = 3,
    ) -> DocumentOperation:
        log_info(f"task: {task}")

        # TODO: check best format for describing the outline (MARKDOWN vs JSON) for short and long documents
        outline = create_document_outline(doc=document, format=OutlineFormat.MARKDOWN)
        log_debug(f"outline: {outline}")

        context = rf"""Given the current outline of the document:

```
{outline}
```

"""

        identification = rf"""To accomplish the following task:

{task}

Analyze the document outline above and:
    1. Identify all document items that are relevant to this task
    2. Determine the specific operation needed (update_content, rewrite_content, or update_section_heading_level)
    3. Plan the exact changes required

For heading level tasks specifically:
    - Examine the hierarchical structure of all section_header items in the outline
    - Determine the correct level for each section based on its position in the document hierarchy
    - Remember: sibling sections should have the same level, child sections should be one level deeper than their parent
    - Include ALL section_header items that need level adjustments in your changes list
    - Use `to_level = -1` if a section_header is actually body text and should become a TextItem
    - Use `to_level = 0` if a section_header is actually the document title and should become a TitleItem
    - There can be at most one TitleItem in the final document
    - If a section heading is missing between 2 existing section headers, provide an `insertions` entry with
      `previous_ref`, `next_ref`, `regex`, and `level`

IMPORTANT: You must return EXACTLY ONE operation in a ```json...``` block with 1 JSON object containing the fields:
    - "operation" (not "action") set to one of: update_content, rewrite_content, update_section_heading_level
    - "ref", "refs", or "changes", depending on the operation
    - For update_section_heading_level: provide a complete "changes" list and, when needed, an "insertions" list

Now, provide me with the operation to execute the task using the exact field names specified above!
"""

        prompt = f"{context}{identification}"
        log_info(f"prompt: {prompt}")

        m = self._create_reasoning_session(system_prompt=self.system_prompt_for_editing_document)

        def _validate_operation_format(content: str) -> bool:
            """Validate that the response contains a valid operation JSON with correct field names."""
            ops: list[dict] = find_json_dicts(text=content)
            if len(ops) == 0:
                log_debug("No JSON objects found in response")
                return False

            op: dict = ops[0]

            try:
                TypeAdapter(DocumentOperation).validate_python(op)
                log_debug(f"Successfully validated operation: {op.get('operation')}")
                return True

            except ValidationError as e:
                log_debug(f"Validation error for operation: {e}")
                return False
            except Exception as e:
                log_debug(f"Unexpected error during validation: {e}")
                return False

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description='Return exactly one JSON object in ```json...``` format with an "operation" field',
                    validation_fn=simple_validate(_validate_operation_format),
                ),
            ],
            retry_budget=loop_budget,
        )
        log_info(f"answer: {answer}")

        view_linear_context(m)

        ops: list[dict] = find_json_dicts(text=answer)

        if len(ops) == 0:
            raise ValueError("No operation is detected")

        op: dict = ops[0]

        try:
            return TypeAdapter(DocumentOperation).validate_python(op)
        except ValidationError as e:
            raise ValueError(f"Operation validation failed: {e}") from e

    def _update_content(self, task: str, document: DoclingDocument, sref: str):
        log_info("_update_content_of_document_items")

        ref = RefItem(cref=sref)
        item = ref.resolve(document)

        if isinstance(item, TableItem):
            self._update_content_of_table(task=task, document=document, table=item)

        elif isinstance(item, TextItem):
            self._update_content_of_textitem(task=task, document=document, item=item)

        else:
            log_warning(f"Dont know how to update the item (of label={item.label}) for task: {task}")

    def _update_content_of_table(
        self,
        task: str,
        document: DoclingDocument,
        table: TableItem,
        loop_budget: int = 3,
    ):
        log_info("_update_content_of_table")

        html_table = serialize_table_to_html(table=table, doc=document)

        prompt = f"""Given the following HTML table,

```html
{html_table}
```

Execute the following task: {task}
"""
        log_info(f"prompt: {prompt}")

        m = self._create_reasoning_session(system_prompt=self.system_prompt_for_editing_table)

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Put the resulting HTML table in the format ```html <insert-content>```",
                    validation_fn=simple_validate(has_html_code_block),
                ),
                Requirement(
                    description="The HTML table should have a valid formatting.",
                    validation_fn=simple_validate(validate_html_to_docling_table),
                ),
            ],
            retry_budget=loop_budget,
        )

        log_info(f"response: {answer}")

        new_tables = convert_html_to_docling_table(text=answer)

        if new_tables and len(new_tables) == 1:
            table.data = new_tables[0].data
        elif new_tables and len(new_tables) > 1:
            log_error("too many tables returned ...")
            table.data = new_tables[0].data

    def _update_content_of_textitem(
        self,
        task: str,
        document: DoclingDocument,
        item: TextItem,
        loop_budget: int = 3,
    ):
        log_info("_update_content_of_text")

        text = serialize_item_to_markdown(item=item, doc=document)

        prompt = f"""Given the following {item.label},

```md
{text}
```

Execute the following task: {task}
"""
        log_info(f"prompt: {prompt}")

        m = self._create_reasoning_session(system_prompt=self.system_prompt_for_editing_table)

        answer = m.instruct(
            prompt,
            retry_budget=loop_budget,
        )
        log_info(f"response: {answer}")

        updated_doc = convert_markdown_to_docling_document(text=answer)
        if updated_doc is None:
            log_warning("No valid document produced for updated content.")
            return

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)

    def _update_section_heading_level(
        self,
        task: str,
        document: DoclingDocument,
        changes: list[SectionHeadingLevelChange],
        insertions: list[MissingSectionHeadingInsertion],
    ):
        for change in changes:
            self._apply_section_heading_change(document=document, change=change)

        for insertion in insertions:
            self._insert_missing_section_header(document=document, insertion=insertion)

    def _apply_section_heading_change(
        self,
        *,
        document: DoclingDocument,
        change: SectionHeadingLevelChange,
    ) -> None:
        item = RefItem(cref=change.ref).resolve(document)

        if not isinstance(item, SectionHeaderItem):
            log_warning(f"{change.ref} is not SectionHeaderItem (got {type(item).__name__})")
            return

        if change.to_level == -1:
            self._convert_section_header_to_text_item(document=document, item=item)
            return

        if change.to_level == 0:
            self._convert_section_header_to_title_item(document=document, item=item)
            return

        item.level = change.to_level

    def _convert_section_header_to_text_item(
        self,
        *,
        document: DoclingDocument,
        item: SectionHeaderItem,
    ) -> None:
        replacement = TextItem(
            self_ref=item.self_ref,
            parent=item.parent,
            children=list(item.children),
            content_layer=item.content_layer,
            meta=item.meta,
            label=DocItemLabel.TEXT,
            prov=list(item.prov),
            source=list(item.source),
            comments=list(item.comments),
            orig=item.orig,
            text=item.text,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        self._replace_text_item_in_place(document=document, old_item=item, new_item=replacement)

    def _convert_section_header_to_title_item(
        self,
        *,
        document: DoclingDocument,
        item: SectionHeaderItem,
    ) -> None:
        existing_titles = [
            candidate
            for candidate in document.texts
            if isinstance(candidate, TitleItem) and candidate.self_ref != item.self_ref
        ]
        if existing_titles:
            raise ValueError(
                f"Cannot convert {item.self_ref} to TitleItem because the document already contains "
                f"{existing_titles[0].self_ref}."
            )

        replacement = TitleItem(
            self_ref=item.self_ref,
            parent=item.parent,
            children=list(item.children),
            content_layer=item.content_layer,
            meta=item.meta,
            prov=list(item.prov),
            source=list(item.source),
            comments=list(item.comments),
            orig=item.orig,
            text=item.text,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        self._replace_text_item_in_place(document=document, old_item=item, new_item=replacement)

    def _replace_text_item_in_place(
        self,
        *,
        document: DoclingDocument,
        old_item: TextItem,
        new_item: TextItem,
    ) -> None:
        try:
            index = int(old_item.self_ref.split("/")[-1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot determine text index from ref: {old_item.self_ref}") from exc

        if index < 0 or index >= len(document.texts):
            raise ValueError(f"Text index out of bounds for ref: {old_item.self_ref}")

        document.texts[index] = new_item

    def _insert_missing_section_header(
        self,
        *,
        document: DoclingDocument,
        insertion: MissingSectionHeadingInsertion,
    ) -> None:
        candidate = self._find_matching_text_item_between_refs(document=document, insertion=insertion)
        if candidate is None:
            log_warning(
                f"No matching TextItem found between {insertion.previous_ref} and {insertion.next_ref} "
                f"for regex {insertion.regex!r}"
            )
            return

        replacement = SectionHeaderItem(
            self_ref=candidate.self_ref,
            parent=candidate.parent,
            children=list(candidate.children),
            content_layer=candidate.content_layer,
            meta=candidate.meta,
            prov=list(candidate.prov),
            source=list(candidate.source),
            comments=list(candidate.comments),
            orig=candidate.orig,
            text=candidate.text,
            formatting=candidate.formatting,
            hyperlink=candidate.hyperlink,
            level=insertion.level,
        )
        self._replace_text_item_in_place(document=document, old_item=candidate, new_item=replacement)

    def _find_matching_text_item_between_refs(
        self,
        *,
        document: DoclingDocument,
        insertion: MissingSectionHeadingInsertion,
    ) -> TextItem | None:
        ordered_items = [item for item, _ in document.iterate_items(with_groups=True)]
        refs = [item.self_ref for item in ordered_items]

        try:
            previous_index = refs.index(insertion.previous_ref)
            next_index = refs.index(insertion.next_ref)
        except ValueError:
            log_warning(f"Could not locate insertion boundaries: {insertion.previous_ref} .. {insertion.next_ref}")
            return None

        if previous_index >= next_index:
            log_warning(f"Invalid insertion boundaries: {insertion.previous_ref} is not before {insertion.next_ref}")
            return None

        pattern = re.compile(insertion.regex)
        matches = [
            item
            for item in ordered_items[previous_index + 1 : next_index]
            if isinstance(item, TextItem)
            and not isinstance(item, SectionHeaderItem | TitleItem)
            and pattern.search(item.text)
        ]

        if len(matches) > 1:
            log_warning(
                f"Found multiple matching TextItems for regex {insertion.regex!r} between "
                f"{insertion.previous_ref} and {insertion.next_ref}; using {matches[0].self_ref}"
            )

        return matches[0] if matches else None

    def _rewrite_content(
        self,
        task: str,
        document: DoclingDocument,
        refs: list[str],
        loop_budget: int = 3,
    ):
        log_info("_update_content_of_text")

        texts = []
        for sref in refs:
            ref = RefItem(cref=sref)
            item = ref.resolve(document)

            texts.append(serialize_item_to_markdown(item=item, doc=document))

        text = "\n\n".join(texts)

        prompt = f"""Given the following text section in markdown,

```md
{text}
```

Execute the following task: {task}
"""
        log_info(f"prompt: {prompt}")

        m = self._create_reasoning_session(system_prompt=self.system_prompt_expert_writer)

        answer = m.instruct(
            prompt,
            retry_budget=loop_budget,
        )
        log_info(f"response: {answer}")

        updated_doc = convert_markdown_to_docling_document(text=answer)
        if updated_doc is None:
            log_warning("No valid document produced for rewrite.")
            return

        ref = RefItem(cref=refs[0])
        item = ref.resolve(document)

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)

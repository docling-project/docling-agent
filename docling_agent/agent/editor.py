from pathlib import Path
from typing import Annotated, ClassVar, Literal

from docling_core.experimental.serializer.outline import OutlineFormat
from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc.document import (
    DoclingDocument,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
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
from docling_agent.agent_models import setup_local_session, view_linear_context
from docling_agent.logging import logger

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
    to_level: Annotated[int, Field(ge=0)]


class UpdateSectionHeadingLevelOperation(BaseModel):
    """Operation to update section heading levels."""

    operation: Literal["update_section_heading_level"]
    changes: list[SectionHeadingLevelChange]


DocumentOperation = Annotated[
    UpdateContentOperation | RewriteContentOperation | UpdateSectionHeadingLevelOperation,
    Field(discriminator="operation"),
]


class DoclingEditingAgent(BaseDoclingAgent):
    system_prompt_for_editing_document: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_DOCUMENT
    system_prompt_for_editing_table: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_TABLE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    def __init__(self, *, model_id: ModelIdentifier, tools: list):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EDITOR,
            model_id=model_id,
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
            self._update_section_heading_level(task=task, document=document, changes=op.changes)
        else:
            message = f"Could not execute operation: {op}"
            logger.info(message)
            raise ValueError(message)

        return document

    def _identify_document_items(
        self,
        task: str,
        document: DoclingDocument,
        loop_budget: int = 5,
    ) -> DocumentOperation:
        logger.info(f"task: {task}")

        outline = create_document_outline(doc=document, format=OutlineFormat.MARKDOWN)
        logger.debug(f"outline: {outline}")

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

IMPORTANT: You must return EXACTLY ONE operation in a ```json...``` block with 1 JSON object containing the fields:
    - "operation" (not "action") set to one of: update_content, rewrite_content, update_section_heading_level
    - "ref", "refs", or "changes", depending on the operation
    - For update_section_heading_level: provide a complete "changes" list with all sections that need adjustment.

Now, provide me with the operation to execute the task using the exact field names specified above!
"""

        prompt = f"{context}{identification}"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_document,
        )

        def _validate_operation_format(content: str) -> bool:
            """Validate that the response contains a valid operation JSON with correct field names."""
            ops: list[dict] = find_json_dicts(text=content)
            if len(ops) == 0:
                logger.debug("No JSON objects found in response")
                return False

            op: dict = ops[0]

            try:
                TypeAdapter(DocumentOperation).validate_python(op)
                logger.debug(f"Successfully validated operation: {op.get('operation')}")
                return True

            except ValidationError as e:
                logger.debug(f"Validation error for operation: {e}")
                return False
            except Exception as e:
                logger.debug(f"Unexpected error during validation: {e}")
                return False

        answer = m.instruct(
            prompt,
            # strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
            strategy=RejectionSamplingStrategy(loop_budget=1),
            requirements=[
                Requirement(
                    description='Return exactly one JSON object in ```json...``` format with an "operation" field',
                    validation_fn=simple_validate(_validate_operation_format),
                ),
            ],
        )
        logger.info(f"answer: {answer.value}")

        view_linear_context(m)

        ops: list[dict] = find_json_dicts(text=answer.value)

        if len(ops) == 0:
            raise ValueError("No operation is detected")

        op: dict = ops[0]

        try:
            return TypeAdapter(DocumentOperation).validate_python(op)
        except ValidationError as e:
            raise ValueError(f"Operation validation failed: {e}") from e

    def _update_content(self, task: str, document: DoclingDocument, sref: str):
        logger.info("_update_content_of_document_items")

        ref = RefItem(cref=sref)
        item = ref.resolve(document)

        if isinstance(item, TableItem):
            self._update_content_of_table(task=task, document=document, table=item)

        elif isinstance(item, TextItem):
            self._update_content_of_textitem(task=task, document=document, item=item)

        else:
            logger.warning(f"Dont know how to update the item (of label={item.label}) for task: {task}")

    def _update_content_of_table(
        self,
        task: str,
        document: DoclingDocument,
        table: TableItem,
        loop_budget: int = 5,
    ):
        logger.info("_update_content_of_table")

        html_table = serialize_table_to_html(table=table, doc=document)

        prompt = f"""Given the following HTML table,

```html
{html_table}
```

Execute the following task: {task}
"""
        # logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_table,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
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
        )

        logger.info(f"response: {answer.value}")

        new_tables = convert_html_to_docling_table(text=answer.value)

        if new_tables and len(new_tables) == 1:
            table.data = new_tables[0].data
        elif new_tables and len(new_tables) > 1:
            logger.error("too many tables returned ...")
            table.data = new_tables[0].data

    def _update_content_of_textitem(
        self,
        task: str,
        document: DoclingDocument,
        item: TextItem,
        loop_budget: int = 5,
    ):
        logger.info("_update_content_of_text")

        text = serialize_item_to_markdown(item=item, doc=document)

        prompt = f"""Given the following {item.label},

```md
{text}
```

Execute the following task: {task}
"""
        # logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_table,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        # logger.info(f"response: {answer.value}")

        updated_doc = convert_markdown_to_docling_document(text=answer.value)
        if updated_doc is None:
            logger.warning("No valid document produced for updated content.")
            return

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)

    def _update_section_heading_level(
        self, task: str, document: DoclingDocument, changes: list[SectionHeadingLevelChange]
    ):
        for change in changes:
            ref = RefItem(cref=change.ref)
            item = ref.resolve(document)

            if isinstance(item, SectionHeaderItem):
                item.level = change.to_level
            else:
                logger.warning(f"{change.ref} is not SectionHeaderItem (got {type(item).__name__})")

    def _rewrite_content(
        self,
        task: str,
        document: DoclingDocument,
        refs: list[str],
        loop_budget: int = 5,
    ):
        logger.info("_update_content_of_text")

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
        logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        logger.info(f"response: {answer.value}")

        updated_doc = convert_markdown_to_docling_document(text=answer.value)
        if updated_doc is None:
            logger.warning("No valid document produced for rewrite.")
            return

        ref = RefItem(cref=refs[0])
        item = ref.resolve(document)

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)

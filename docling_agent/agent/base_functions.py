import json
import re
from io import BytesIO

# from smolagents import MCPClient, Tool, ToolCollection
# from smolagents.models import ChatMessage, MessageRole, Model
from docling.datamodel.base_models import ConversionStatus, InputFormat
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
    RefItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.io import DocumentStream
from docling_agent.agents import logger

# Use shared logger from docling_agent.agents


def find_json_dicts(text: str) -> list[dict]:
    """
    Extract JSON dictionaries from ```json code blocks
    """
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    for i, json_content in enumerate(matches):
        try:
            # print(f"call {i}: {json_content}")
            calls.append(json.loads(json_content))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON in match {i}: {e}")

    return calls


def find_crefs(text: str) -> list[RefItem]:
    """
    Check if a string matches the pattern ```markdown(.*)?```
    """
    labels: str = "|".join(e.value for e in DocItemLabel)
    pattern = rf"#/({labels})/\d+"

    refs: list[RefItem] = []
    for m in re.finditer(pattern, text, re.DOTALL):
        refs.append(RefItem(cref=m.group(0)))

    return refs


def has_crefs(text: str) -> bool:
    return len(find_crefs(text)) > 0


def create_document_outline(doc: DoclingDocument) -> str:
    label_counter: dict[DocItemLabel, int] = {
        DocItemLabel.TABLE: 0,
        DocItemLabel.PICTURE: 0,
        DocItemLabel.TEXT: 0,
    }

    lines: list[str] = []
    for item, level in doc.iterate_items(with_groups=True):
        if isinstance(item, TitleItem):
            lines.append(f"title (reference={item.self_ref}): {item.text}")

        elif isinstance(item, SectionHeaderItem):
            lines.append(
                f"section-header (level={item.level}, reference={item.self_ref}): {item.text}"
            )

        elif isinstance(item, ListItem):
            continue

        elif isinstance(item, TextItem):
            lines.append(f"{item.label} (reference={item.self_ref})")

        elif isinstance(item, TableItem):
            label_counter[item.label] += 1
            lines.append(
                f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
            )

        elif isinstance(item, PictureItem):
            label_counter[item.label] += 1
            lines.append(
                f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
            )

    outline = "\n\n".join(lines)

    return outline


def find_outline_v1(text: str) -> DoclingDocument | None:
    starts = ["paragraph", "list", "table", "figure", "picture"]

    md = find_markdown_code_block(text)

    if md:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        buff = BytesIO(md.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.md", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        lines: list[str] = []
        for item, level in conv.document.iterate_items(with_groups=True):
            if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
                continue
            elif isinstance(item, TextItem):
                pattern = rf"^({'|'.join(starts)}):\s(.*)\.$"
                match = bool(re.match(pattern, text, re.DOTALL))
                if match is None:
                    lines.append(item.text)
            else:
                continue

        if len(lines) > 0:
            message = f"Every content line should start with one out of the following choices: {starts}. The following lines need to be updated: {'\n'.join(lines)}"
            logger.error(message)

            return None
        else:
            return conv.document
    else:
        return None


def find_outline_v2(text: str) -> DoclingDocument | None:
    starts = ["paragraph", "list", "table", "figure", "picture"]

    md = find_markdown_code_block(text)

    if not md:
        return None

    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    buff = BytesIO(md.encode("utf-8"))
    doc_stream = DocumentStream(name="tmp.md", stream=buff)

    conv: ConversionResult = converter.convert(doc_stream)

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
            match = re.match(pattern, item.text, re.DOTALL)

            if not match:
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
                # Add a group placeholder for a list; attach summary via meta
                try:
                    _ = outline.add_group(
                        name="list", label=GroupLabel.UNSPECIFIED, parent=None
                    )
                except TypeError:
                    # Fallback for API variants that don't require explicit parent
                    _ = outline.add_group(name="list", label=GroupLabel.UNSPECIFIED)
                _.meta = meta

            else:
                logger.warning(f"NOT SUPPORTED: {label}")
        else:
            continue

    if len(invalid_lines) > 0:
        message = (
            "Every content line should start with one of: "
            f"{starts}. The following lines need to be updated: "
            + "\n".join(invalid_lines)
        )
        logger.error(message)
        return None

    # print(outline.export_to_markdown())

    return outline


def validate_outline_format(text: str) -> bool:
    logger.info(f"testing validate_outline_format for {text[0:64]}")
    return find_outline_v2(text) is not None


def serialize_item_to_markdown(item: TextItem, doc: DoclingDocument) -> str:
    """Serialize a text item to markdown format using existing serializer."""
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )

    serializer = MarkdownDocSerializer(doc=doc, params=MarkdownParams())

    result = serializer.serialize(item=item)
    return result.text


def serialize_table_to_html(table: TableItem, doc: DoclingDocument) -> str:
    from docling_core.transforms.serializer.html import (
        HTMLDocSerializer,
        HTMLTableSerializer,
    )

    # Create the table serializer
    table_serializer = HTMLTableSerializer()

    # Create a document serializer (needed as dependency)
    doc_serializer = HTMLDocSerializer(doc=doc)

    # Serialize the table
    result = table_serializer.serialize(
        item=table, doc_serializer=doc_serializer, doc=doc
    )

    return result.text


def find_html_code_block(text: str) -> str | None:
    """
    Check if a string matches the pattern ```html(.*)?```
    """
    pattern = r"```html(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def has_html_code_block(text: str) -> bool:
    """
    Check if a string contains a html code block pattern anywhere in the text
    """
    logger.info(f"testing has_html_code_block for {text[0:64]}")
    return find_html_code_block(text) is not None


def find_markdown_code_block(text: str) -> str | None:
    """
    Check if a string matches the pattern ```(md|markdown)(.*)?```
    """
    pattern = r"```(md|markdown)(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2) if match else None


def has_markdown_code_block(text: str) -> bool:
    """
    Check if a string contains a markdown code block pattern anywhere in the text
    """
    logger.info(f"testing has_markdown_code_block for {text[0:64]}")
    return find_markdown_code_block(text) is not None


def convert_html_to_docling_table(text: str) -> list[TableItem] | None:
    text_ = find_html_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

        buff = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.html", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document.tables

    except Exception as exc:
        logger.error(exc)
        return None

    return None


def validate_html_to_docling_table(text: str) -> bool:
    logger.info(f"validate_html_to_docling_table for {text[0:64]}")
    return convert_html_to_docling_table(text) is not None


def convert_markdown_to_docling_document(text: str) -> DoclingDocument | None:
    text_ = find_markdown_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        buff = BytesIO(text_.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.md", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document
    except Exception:
        return None

    return None


def validate_markdown_to_docling_document(text: str) -> bool:
    logger.info(f"testing validate_markdown_docling_document for {text[0:64]}")
    return convert_markdown_to_docling_document(text) is not None


def convert_html_to_docling_document(text: str) -> DoclingDocument | None:
    text_ = find_html_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

        buff = BytesIO(text_.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.html", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document
    except Exception as exc:
        logger.error(f"error: {exc}")
        return None

    return None


def validate_html_to_docling_document(text: str) -> bool:
    logger.info(f"testing validate_html_docling_document for {text[0:64]}")
    return convert_html_to_docling_document(text) is not None


def insert_document(
    *, item: NodeItem, doc: DoclingDocument, updated_doc: DoclingDocument
) -> DoclingDocument:
    logger.info(f"inserting new document at item {item.self_ref}")

    group_item = GroupItem(
        label=GroupLabel.UNSPECIFIED,
        name="inserted-group",
        self_ref="#",  # temporary placeholder
    )

    if isinstance(item, ListItem):
        # we should delete all the children of the list-item and put the text to ""
        raise ValueError("ListItem insertion is not yet supported!")

    doc.replace_item(
        old_item=item, new_item=group_item
    )  # group_item is being updated here ...

    to_item: dict[str, NodeItem] = {}
    for _item, level in updated_doc.iterate_items(with_groups=True):
        if isinstance(_item, GroupItem) and _item.self_ref == "#/body":
            to_item[_item.self_ref] = group_item

        elif _item.parent is None:
            logger.error(f"Item with null parent: {_item}")

        elif _item.parent.cref not in to_item:
            logger.error(f"Item with unknown parent: {_item}")

        elif isinstance(_item, GroupItem):
            gr = doc.add_group(
                name=_item.name,
                label=_item.label,
                parent=to_item[_item.parent.cref],
            )
            to_item[_item.self_ref] = gr

        elif isinstance(_item, ListItem):
            li = doc.add_list_item(
                text=_item.text,
                formatting=_item.formatting,
                parent=to_item[_item.parent.cref],
            )
            to_item[_item.self_ref] = li

        elif isinstance(_item, TextItem):
            te = doc.add_text(
                text=_item.text,
                label=_item.label,
                formatting=_item.formatting,
                parent=to_item[_item.parent.cref],
            )
            to_item[_item.self_ref] = te

        elif isinstance(_item, TableItem):
            if len(_item.captions) > 0:
                # Caption entries may be references; create an empty caption text item
                caption = doc.add_text(label=DocItemLabel.CAPTION, text="")
                te = doc.add_table(
                    data=_item.data,
                    caption=caption,
                )
                to_item[_item.self_ref] = te
            else:
                te = doc.add_table(
                    data=_item.data,
                )
                to_item[_item.self_ref] = te

        else:
            logger.warning(f"No support to insert items of type: {type(item).__name__}")

    return doc

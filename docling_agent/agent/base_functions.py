import json
import re
from io import BytesIO

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListGroup,
    ListItem,
    NodeItem,
    PictureItem,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.io import DocumentStream

from docling_agent.logging import logger


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


def has_json_dicts(text: str) -> bool:
    """
    Extract JSON dictionaries from ```json code blocks
    """
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    for i, json_content in enumerate(matches):
        try:
            calls.append(json.loads(json_content))
        except:
            return False

    return len(calls) > 0


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
    # logger.info(f"testing has_html_code_block for {text[0:64]}")
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
    # logger.info(f"testing has_markdown_code_block for {text[0:64]}")
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
    # logger.info(f"validate_html_to_docling_table for {text[0:64]}")
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
    # logger.info(f"testing validate_markdown_docling_document for {text[0:64]}")
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
    # logger.info(f"testing validate_html_docling_document for {text[0:64]}")
    return convert_html_to_docling_document(text) is not None


def insert_document(
    *, item: NodeItem, doc: DoclingDocument, updated_doc: DoclingDocument
) -> DoclingDocument:
    # logger.info(f"inserting new document at item {item.self_ref}")

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


# ---------------------------------------------------------------------------
# Document tree utilities
# ---------------------------------------------------------------------------


def get_item_by_ref(doc: DoclingDocument, ref: str) -> NodeItem | None:
    """Resolve a self_ref string to a NodeItem. Returns None on failure."""
    try:
        return RefItem(cref=ref).resolve(doc)
    except Exception:
        return None


def collect_subtree_text(node: NodeItem, doc: DoclingDocument) -> str:
    """Recursively collect all text from a node and its descendants.

    Resolves each child RefItem and concatenates text from TextItem instances
    (which includes TitleItem, SectionHeaderItem, ListItem, TextItem proper).
    Non-text nodes (TableItem, PictureItem, GroupItem) are traversed for their
    children but do not contribute text directly.
    """
    parts: list[str] = []
    if hasattr(node, "text") and node.text:
        parts.append(node.text)
    for child_ref in node.children or []:
        try:
            child = child_ref.resolve(doc)
            subtree = collect_subtree_text(child, doc)
            if subtree:
                parts.append(subtree)
        except Exception:
            pass
    return "\n".join(parts)


def _copy_list_group(
    source: ListGroup,
    source_doc: DoclingDocument,
    target_doc: DoclingDocument,
    parent: NodeItem,
) -> ListGroup:
    new_group = target_doc.add_list_group(parent=parent)
    new_group.meta = source.meta
    for child_ref in source.children or []:
        try:
            child = child_ref.resolve(source_doc)
            if isinstance(child, ListItem):
                new_item = target_doc.add_list_item(
                    text=child.text,
                    enumerated=child.enumerated,
                    parent=new_group,
                )
                new_item.meta = child.meta
                # Recursively copy nested list groups
                for nested_ref in child.children or []:
                    try:
                        nested = nested_ref.resolve(source_doc)
                        if isinstance(nested, ListGroup):
                            _copy_list_group(nested, source_doc, target_doc, new_item)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f"Could not copy list child: {exc}")
    return new_group


def _copy_table(
    source: TableItem,
    source_doc: DoclingDocument,
    target_doc: DoclingDocument,
    parent: NodeItem,
) -> TableItem:
    new_table = target_doc.add_table(data=source.data, parent=parent)
    new_table.meta = source.meta
    for cap_ref in source.captions:
        try:
            cap = cap_ref.resolve(source_doc)
            if hasattr(cap, "text"):
                new_cap = target_doc.add_text(
                    label=cap.label, text=cap.text, parent=new_table
                )
                new_cap.meta = cap.meta
                new_table.captions.append(new_cap.get_ref())
        except Exception as exc:
            logger.warning(f"Could not copy table caption: {exc}")
    return new_table


def _copy_picture(
    source: PictureItem,
    source_doc: DoclingDocument,
    target_doc: DoclingDocument,
    parent: NodeItem,
) -> PictureItem:
    new_pic = target_doc.add_picture(image=source.image, parent=parent)
    new_pic.meta = source.meta
    for cap_ref in source.captions:
        try:
            cap = cap_ref.resolve(source_doc)
            if hasattr(cap, "text"):
                new_cap = target_doc.add_text(
                    label=cap.label, text=cap.text, parent=new_pic
                )
                new_cap.meta = cap.meta
                new_pic.captions.append(new_cap.get_ref())
        except Exception as exc:
            logger.warning(f"Could not copy picture caption: {exc}")
    return new_pic


def _flatten_into(
    node: NodeItem,
    source_doc: DoclingDocument,
    target_doc: DoclingDocument,
    target_parent: NodeItem,
) -> None:
    """Recursively add node's children to target_parent, preserving atomic units."""
    for child_ref in node.children or []:
        try:
            child = child_ref.resolve(source_doc)
        except Exception as exc:
            logger.warning(f"Could not resolve child {child_ref}: {exc}")
            continue

        if isinstance(child, ListGroup):
            _copy_list_group(child, source_doc, target_doc, target_parent)
        elif isinstance(child, TableItem):
            _copy_table(child, source_doc, target_doc, target_parent)
        elif isinstance(child, PictureItem):
            _copy_picture(child, source_doc, target_doc, target_parent)
        elif isinstance(child, TitleItem):
            new_item = target_doc.add_title(text=child.text, parent=target_parent)
            new_item.meta = child.meta
            _flatten_into(child, source_doc, target_doc, target_parent)
        elif isinstance(child, SectionHeaderItem):
            new_item = target_doc.add_heading(
                text=child.text, level=child.level, parent=target_parent
            )
            new_item.meta = child.meta
            _flatten_into(child, source_doc, target_doc, target_parent)
        elif isinstance(child, ListItem):
            logger.warning(
                f"ListItem {child.self_ref} found outside a ListGroup; skipping"
            )
        elif isinstance(child, GroupItem):
            # Dissolve other groups (recurse into children without adding the group)
            _flatten_into(child, source_doc, target_doc, target_parent)
        elif hasattr(child, "text"):
            new_item = target_doc.add_text(
                label=child.label, text=child.text, parent=target_parent
            )
            new_item.meta = child.meta


def make_flat_document(doc: DoclingDocument) -> DoclingDocument:
    """Return a new document where every item is a direct child of body.

    Iterates ``doc`` in document order and appends each item to the new body,
    preserving:
    - SectionHeaderItem.level  (needed for make_hierarchical_document to invert)
    - List internal structure  (ListGroup → ListItem nesting is kept)
    - Table / picture caption children
    All other parent-child links (section → text) are dissolved.
    """
    new_doc = DoclingDocument(name=doc.name)
    _flatten_into(doc.body, doc, new_doc, new_doc.body)
    return new_doc


def make_hierarchical_document(doc: DoclingDocument) -> DoclingDocument:
    """Return a new document with maximal section nesting.

    Iterates ``doc`` in document order (after flattening first).  Maintains a
    stack of open section headers keyed by their level.  Each non-header item
    (text, table, picture, list) is appended as a child of the most recently
    opened section header (or of body if no header has been seen yet).
    A section header at level N is appended as a child of the nearest ancestor
    whose level is strictly less than N.

    Lists, table-caption pairs and picture-caption pairs are treated as atomic
    units and are not split across parent boundaries.
    """
    flat = make_flat_document(doc)
    new_doc = DoclingDocument(name=doc.name)

    # open_sections maps level -> SectionHeaderItem (only section headers, not title).
    open_sections: dict[int, NodeItem] = {}
    # title_node is the most recently seen TitleItem; text before any section header
    # becomes a child of the title rather than of body.
    title_node: NodeItem | None = None

    def _current_parent() -> NodeItem:
        if open_sections:
            return open_sections[max(open_sections)]
        if title_node is not None:
            return title_node
        return new_doc.body

    def _parent_for_level(level: int) -> NodeItem:
        # Section headers nest only under other section headers, never under the title.
        candidates = [lv for lv in open_sections if lv < level]
        if not candidates:
            return new_doc.body
        return open_sections[max(candidates)]

    for child_ref in flat.body.children or []:
        try:
            child = child_ref.resolve(flat)
        except Exception as exc:
            logger.warning(f"Could not resolve body child {child_ref}: {exc}")
            continue

        if isinstance(child, TitleItem):
            new_item = new_doc.add_title(text=child.text, parent=new_doc.body)
            new_item.meta = child.meta
            title_node = new_item
            open_sections = {}

        elif isinstance(child, SectionHeaderItem):
            level = child.level
            parent = _parent_for_level(level)
            new_item = new_doc.add_heading(text=child.text, level=level, parent=parent)
            new_item.meta = child.meta
            # Close all open sections at >= this level
            open_sections = {lv: n for lv, n in open_sections.items() if lv < level}
            open_sections[level] = new_item

        elif isinstance(child, ListGroup):
            _copy_list_group(child, flat, new_doc, _current_parent())

        elif isinstance(child, TableItem):
            _copy_table(child, flat, new_doc, _current_parent())

        elif isinstance(child, PictureItem):
            _copy_picture(child, flat, new_doc, _current_parent())

        elif hasattr(child, "text"):
            new_item = new_doc.add_text(
                label=child.label, text=child.text, parent=_current_parent()
            )
            new_item.meta = child.meta

        else:
            logger.warning(
                f"Unhandled item type {type(child).__name__} in make_hierarchical_document"
            )

    return new_doc

"""Unit tests for make_flat_document and make_hierarchical_document."""

import pytest
from docling_core.types.doc.document import (
    BaseMeta,
    DocItemLabel,
    DoclingDocument,
    ListGroup,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableCell,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)

from docling_agent.agent.base_functions import (
    collect_subtree_text,
    make_flat_document,
    make_hierarchical_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def body_texts(doc: DoclingDocument) -> list[str]:
    """Return the text of every direct child of body (in order)."""
    result = []
    for ref in doc.body.children:
        item = ref.resolve(doc)
        result.append(getattr(item, "text", type(item).__name__))
    return result


def child_texts(node, doc: DoclingDocument) -> list[str]:
    """Return the text of every direct child of node."""
    return [getattr(ref.resolve(doc), "text", "?") for ref in node.children or []]


def all_texts_in_order(doc: DoclingDocument) -> list[str]:
    """DFS ordering of all item texts (groups skipped)."""
    result = []
    for item, _ in doc.iterate_items():
        if hasattr(item, "text") and item.text:
            result.append(item.text)
    return result


# ---------------------------------------------------------------------------
# make_flat_document
# ---------------------------------------------------------------------------


class TestMakeFlatDocument:
    def test_already_flat_document_is_unchanged(self):
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.TEXT, text="Alpha", parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Beta", parent=doc.body)

        flat = make_flat_document(doc)

        assert body_texts(flat) == ["Alpha", "Beta"]

    def test_dissolves_section_header_children(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Intro", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Para 1", parent=s1)
        doc.add_text(label=DocItemLabel.TEXT, text="Para 2", parent=s1)

        flat = make_flat_document(doc)

        assert body_texts(flat) == ["Intro", "Para 1", "Para 2"]

    def test_title_children_dissolved(self):
        doc = DoclingDocument(name="test")
        title = doc.add_title(text="My Paper", parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Abstract", parent=title)
        s1 = doc.add_heading(text="Section 1", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Body", parent=s1)

        flat = make_flat_document(doc)

        assert body_texts(flat) == ["My Paper", "Abstract", "Section 1", "Body"]

    def test_preserves_section_header_level(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Top", level=1, parent=doc.body)
        s2 = doc.add_heading(text="Sub", level=2, parent=s1)
        doc.add_text(label=DocItemLabel.TEXT, text="Content", parent=s2)

        flat = make_flat_document(doc)

        items = {ref.resolve(flat).text: ref.resolve(flat) for ref in flat.body.children}
        assert isinstance(items["Top"], SectionHeaderItem)
        assert items["Top"].level == 1
        assert isinstance(items["Sub"], SectionHeaderItem)
        assert items["Sub"].level == 2

    def test_preserves_list_internal_structure(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Section", level=1, parent=doc.body)
        lg = doc.add_list_group(parent=s1)
        doc.add_list_item(text="Item A", parent=lg)
        doc.add_list_item(text="Item B", parent=lg)

        flat = make_flat_document(doc)

        # body should have: Section, ListGroup
        flat_children = [ref.resolve(flat) for ref in flat.body.children]
        assert flat_children[0].text == "Section"
        list_group = flat_children[1]
        assert isinstance(list_group, ListGroup)
        assert child_texts(list_group, flat) == ["Item A", "Item B"]

    def test_preserves_table_caption(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Results", level=1, parent=doc.body)
        table = doc.add_table(data=TableData(num_rows=1, num_cols=1), parent=s1)
        cap = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1", parent=table)
        table.captions.append(cap.get_ref())

        flat = make_flat_document(doc)

        flat_children = [ref.resolve(flat) for ref in flat.body.children]
        assert flat_children[0].text == "Results"
        new_table = flat_children[1]
        assert isinstance(new_table, TableItem)
        assert len(new_table.captions) == 1
        cap_text = new_table.captions[0].resolve(flat)
        assert cap_text.text == "Table 1"

    def test_preserves_picture_caption(self):
        doc = DoclingDocument(name="test")
        pic = doc.add_picture(parent=doc.body)
        cap = doc.add_text(label=DocItemLabel.CAPTION, text="Figure 1", parent=pic)
        pic.captions.append(cap.get_ref())

        flat = make_flat_document(doc)

        flat_children = [ref.resolve(flat) for ref in flat.body.children]
        new_pic = flat_children[0]
        assert isinstance(new_pic, PictureItem)
        assert len(new_pic.captions) == 1
        assert new_pic.captions[0].resolve(flat).text == "Figure 1"

    def test_preserves_meta_summary(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Section", level=1, parent=doc.body)
        s1.meta = BaseMeta(summary=SummaryMetaField(text="A summary."))
        doc.add_text(label=DocItemLabel.TEXT, text="Content", parent=s1)

        flat = make_flat_document(doc)

        section = next(
            ref.resolve(flat)
            for ref in flat.body.children
            if getattr(ref.resolve(flat), "text", "") == "Section"
        )
        assert section.meta is not None
        assert section.meta.summary.text == "A summary."

    def test_empty_document(self):
        doc = DoclingDocument(name="empty")
        flat = make_flat_document(doc)
        assert flat.body.children == []

    def test_document_order_preserved(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="A", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="a1", parent=s1)
        doc.add_text(label=DocItemLabel.TEXT, text="a2", parent=s1)
        s2 = doc.add_heading(text="B", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="b1", parent=s2)

        flat = make_flat_document(doc)
        assert body_texts(flat) == ["A", "a1", "a2", "B", "b1"]


# ---------------------------------------------------------------------------
# make_hierarchical_document
# ---------------------------------------------------------------------------


class TestMakeHierarchicalDocument:
    def test_flat_text_under_section_header(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="Intro", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Para", parent=doc.body)

        hier = make_hierarchical_document(doc)

        intro = hier.body.children[0].resolve(hier)
        assert intro.text == "Intro"
        assert child_texts(intro, hier) == ["Para"]

    def test_subsection_nested_under_parent_section(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="Top", level=1, parent=doc.body)
        doc.add_heading(text="Sub", level=2, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Detail", parent=doc.body)

        hier = make_hierarchical_document(doc)

        top = hier.body.children[0].resolve(hier)
        assert top.text == "Top"
        sub = top.children[0].resolve(hier)
        assert sub.text == "Sub"
        assert child_texts(sub, hier) == ["Detail"]

    def test_title_is_sibling_of_section_headers(self):
        doc = DoclingDocument(name="test")
        doc.add_title(text="My Paper", parent=doc.body)
        doc.add_heading(text="Introduction", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Body", parent=doc.body)

        hier = make_hierarchical_document(doc)

        body_children = [ref.resolve(hier) for ref in hier.body.children]
        assert body_children[0].text == "My Paper"
        assert body_children[1].text == "Introduction"
        # section header is NOT nested inside the title
        assert not any(
            getattr(ref.resolve(hier), "text", "") == "Introduction"
            for ref in body_children[0].children or []
        )

    def test_text_before_first_section_becomes_title_child(self):
        doc = DoclingDocument(name="test")
        doc.add_title(text="Paper", parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Abstract", parent=doc.body)
        doc.add_heading(text="Intro", level=1, parent=doc.body)

        hier = make_hierarchical_document(doc)

        title = hier.body.children[0].resolve(hier)
        assert title.text == "Paper"
        assert child_texts(title, hier) == ["Abstract"]

    def test_same_level_sections_are_siblings(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="A", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="a", parent=doc.body)
        doc.add_heading(text="B", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="b", parent=doc.body)

        hier = make_hierarchical_document(doc)

        assert body_texts(hier) == ["A", "B"]
        sec_a = hier.body.children[0].resolve(hier)
        sec_b = hier.body.children[1].resolve(hier)
        assert child_texts(sec_a, hier) == ["a"]
        assert child_texts(sec_b, hier) == ["b"]

    def test_higher_level_section_closes_lower(self):
        # After a level-2 section, a level-1 section should go to body, not inside level-2
        doc = DoclingDocument(name="test")
        doc.add_heading(text="Methods", level=1, parent=doc.body)
        doc.add_heading(text="Setup", level=2, parent=doc.body)
        doc.add_heading(text="Results", level=1, parent=doc.body)

        hier = make_hierarchical_document(doc)

        assert body_texts(hier) == ["Methods", "Results"]
        methods = hier.body.children[0].resolve(hier)
        assert child_texts(methods, hier) == ["Setup"]

    def test_three_level_nesting(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="L1", level=1, parent=doc.body)
        doc.add_heading(text="L2", level=2, parent=doc.body)
        doc.add_heading(text="L3", level=3, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="leaf", parent=doc.body)

        hier = make_hierarchical_document(doc)

        l1 = hier.body.children[0].resolve(hier)
        l2 = l1.children[0].resolve(hier)
        l3 = l2.children[0].resolve(hier)
        assert l1.text == "L1"
        assert l2.text == "L2"
        assert l3.text == "L3"
        assert child_texts(l3, hier) == ["leaf"]

    def test_list_group_placed_under_current_section(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="Section", level=1, parent=doc.body)
        lg = doc.add_list_group(parent=doc.body)
        doc.add_list_item(text="X", parent=lg)
        doc.add_list_item(text="Y", parent=lg)

        hier = make_hierarchical_document(doc)

        section = hier.body.children[0].resolve(hier)
        assert section.text == "Section"
        list_group = section.children[0].resolve(hier)
        assert isinstance(list_group, ListGroup)
        assert child_texts(list_group, hier) == ["X", "Y"]

    def test_preserves_meta(self):
        doc = DoclingDocument(name="test")
        s = doc.add_heading(text="Section", level=1, parent=doc.body)
        s.meta = BaseMeta(summary=SummaryMetaField(text="My summary."))

        hier = make_hierarchical_document(doc)

        section = hier.body.children[0].resolve(hier)
        assert section.meta is not None
        assert section.meta.summary.text == "My summary."

    def test_empty_document(self):
        doc = DoclingDocument(name="empty")
        hier = make_hierarchical_document(doc)
        assert hier.body.children == []


# ---------------------------------------------------------------------------
# Round-trip invariants
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _build_complex_doc(self) -> DoclingDocument:
        """Build a hierarchical document with title, nested sections, list, table."""
        doc = DoclingDocument(name="complex")
        title = doc.add_title(text="Paper Title", parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Abstract text.", parent=title)

        s1 = doc.add_heading(text="Introduction", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Intro body.", parent=s1)
        s2 = doc.add_heading(text="Background", level=2, parent=s1)
        doc.add_text(label=DocItemLabel.TEXT, text="Background detail.", parent=s2)

        s3 = doc.add_heading(text="Methods", level=1, parent=doc.body)
        lg = doc.add_list_group(parent=s3)
        doc.add_list_item(text="Step A", parent=lg)
        doc.add_list_item(text="Step B", parent=lg)

        table = doc.add_table(data=TableData(num_rows=1, num_cols=1), parent=s3)
        cap = doc.add_text(label=DocItemLabel.CAPTION, text="Table caption", parent=table)
        table.captions.append(cap.get_ref())

        return doc

    def test_flat_then_hier_preserves_text_order(self):
        doc = self._build_complex_doc()
        original_texts = all_texts_in_order(doc)
        recovered_texts = all_texts_in_order(make_hierarchical_document(make_flat_document(doc)))
        assert recovered_texts == original_texts

    def test_hier_then_flat_preserves_text_order(self):
        doc = self._build_complex_doc()
        original_texts = all_texts_in_order(doc)
        recovered_texts = all_texts_in_order(make_flat_document(make_hierarchical_document(doc)))
        assert recovered_texts == original_texts

    def test_flat_is_idempotent(self):
        doc = self._build_complex_doc()
        flat1 = make_flat_document(doc)
        flat2 = make_flat_document(flat1)
        assert body_texts(flat1) == body_texts(flat2)

    def test_hierarchical_is_idempotent(self):
        doc = self._build_complex_doc()
        hier1 = make_hierarchical_document(doc)
        hier2 = make_hierarchical_document(hier1)
        assert all_texts_in_order(hier1) == all_texts_in_order(hier2)

    def test_section_levels_preserved_through_round_trip(self):
        doc = DoclingDocument(name="test")
        doc.add_heading(text="L1", level=1, parent=doc.body)
        doc.add_heading(text="L2", level=2, parent=doc.body)
        doc.add_heading(text="L3", level=3, parent=doc.body)

        flat = make_flat_document(make_hierarchical_document(doc))
        levels = {
            ref.resolve(flat).text: ref.resolve(flat).level
            for ref in flat.body.children
            if isinstance(ref.resolve(flat), SectionHeaderItem)
        }
        assert levels == {"L1": 1, "L2": 2, "L3": 3}


# ---------------------------------------------------------------------------
# collect_subtree_text
# ---------------------------------------------------------------------------


class TestCollectSubtreeText:
    def test_leaf_text_item(self):
        doc = DoclingDocument(name="test")
        item = doc.add_text(label=DocItemLabel.TEXT, text="Hello world", parent=doc.body)
        assert collect_subtree_text(item, doc) == "Hello world"

    def test_section_with_children(self):
        doc = DoclingDocument(name="test")
        s = doc.add_heading(text="Section", level=1, parent=doc.body)
        doc.add_text(label=DocItemLabel.TEXT, text="Child text", parent=s)

        text = collect_subtree_text(s, doc)
        assert "Section" in text
        assert "Child text" in text

    def test_skips_empty_text(self):
        doc = DoclingDocument(name="test")
        s = doc.add_heading(text="Header", level=1, parent=doc.body)
        result = collect_subtree_text(s, doc)
        assert result == "Header"

    def test_nested_sections(self):
        doc = DoclingDocument(name="test")
        s1 = doc.add_heading(text="Top", level=1, parent=doc.body)
        s2 = doc.add_heading(text="Sub", level=2, parent=s1)
        doc.add_text(label=DocItemLabel.TEXT, text="Deep", parent=s2)

        text = collect_subtree_text(s1, doc)
        assert "Top" in text
        assert "Sub" in text
        assert "Deep" in text

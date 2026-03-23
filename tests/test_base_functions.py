import json
from pathlib import Path

from docling_core.experimental.serializer.outline import OutlineMode
from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem
from test_data_gen_flag import GEN_TEST_DATA

from docling_agent.agent.base_functions import create_document_outline, find_json_dicts, make_hierarchical_document


def test_find_json_dicts_with_list_input():
    """Test that find_json_dicts correctly handles JSON arrays."""
    text = """```json
[
  {
    "operation": "update_section_heading_level",
    "reference": "#/texts/1020",
    "new_level": 2
  },
  {
    "operation": "update_section_heading_level",
    "reference": "#/texts/1344",
    "new_level": 2
  }
]
```"""

    result = find_json_dicts(text)

    # Expected: a flat list of dicts, not a nested list
    expected = [
        {
            "operation": "update_section_heading_level",
            "reference": "#/texts/1020",
            "new_level": 2,
        },
        {
            "operation": "update_section_heading_level",
            "reference": "#/texts/1344",
            "new_level": 2,
        },
    ]

    assert result == expected
    assert len(result) == 2
    assert isinstance(result, list)
    assert all(isinstance(item, dict) for item in result)


def test_find_json_dicts_with_single_dict():
    """Test that find_json_dicts correctly handles a single JSON object."""
    text = """```json
{
  "operation": "update_section_heading_level",
  "reference": "#/texts/1020",
  "new_level": 2
}
```"""

    result = find_json_dicts(text)

    expected = [
        {
            "operation": "update_section_heading_level",
            "reference": "#/texts/1020",
            "new_level": 2,
        }
    ]

    assert result == expected
    assert len(result) == 1
    assert isinstance(result[0], dict)


def test_find_json_dicts_with_multiple_code_blocks():
    """Test that find_json_dicts handles multiple JSON code blocks."""
    text = """
First block:
```json
{"key1": "value1"}
```

Second block:
```json
[
  {"key2": "value2"},
  {"key3": "value3"}
]
```
"""

    result = find_json_dicts(text)

    expected = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]

    assert result == expected
    assert len(result) == 3


def test_find_json_dicts_with_no_json():
    """Test that find_json_dicts returns empty list when no JSON blocks found."""
    text = "This is just plain text with no JSON code blocks."

    result = find_json_dicts(text)

    assert result == []
    assert len(result) == 0


def test_find_json_dicts_with_invalid_json():
    """Test that find_json_dicts handles invalid JSON gracefully."""
    text = """```json
{invalid json}
```"""

    result = find_json_dicts(text)

    # Should return empty list and log a warning
    assert result == []


def test_make_hierarchical_document():
    """Test that make_hierarchical_document creates proper hierarchical structure from flat document."""
    # Load the JSON file
    json_path = Path(__file__).parent / "data" / "2408.09869v5.json"
    with open(json_path) as f:
        doc_dict = json.load(f)

    # Instantiate as DoclingDocument
    original_doc = DoclingDocument.model_validate(doc_dict)

    # Call make_hierarchical_document
    hierarchical_doc = make_hierarchical_document(original_doc)

    # Verify the result is a DoclingDocument
    assert isinstance(hierarchical_doc, DoclingDocument)
    assert hierarchical_doc.name == original_doc.name

    # Verify hierarchical structure was created
    # The function should nest items under section headers based on their levels

    # Count section headers and verify they have children in hierarchical doc
    section_headers_with_children = 0
    total_section_headers = 0

    for item, level in hierarchical_doc.iterate_items(with_groups=True):
        if isinstance(item, SectionHeaderItem):
            total_section_headers += 1
            # In a hierarchical document, section headers should typically have children
            # (unless they're at the very end of the document)
            if item.children and len(item.children) > 0:
                section_headers_with_children += 1

    # Verify that we have section headers in the document
    assert total_section_headers > 0, "Document should contain section headers"

    # Verify that at least some section headers have children (hierarchical structure)
    # In a properly hierarchical document, most section headers should have content nested under them
    assert section_headers_with_children > 0, "Hierarchical document should have section headers with children"

    # Verify that the hierarchical document has proper nesting
    # Check that body's direct children are reduced compared to original (items moved under sections)
    original_body_children_count = len(original_doc.body.children) if original_doc.body.children else 0
    hierarchical_body_children_count = len(hierarchical_doc.body.children) if hierarchical_doc.body.children else 0

    # The hierarchical document should have fewer direct children in body
    # because items are nested under section headers
    assert hierarchical_body_children_count <= original_body_children_count, (
        "Hierarchical document should have fewer or equal direct body children due to nesting"
    )

    # Verify section header levels are preserved
    original_levels = []
    hierarchical_levels = []

    for item, _ in original_doc.iterate_items(with_groups=True):
        if isinstance(item, SectionHeaderItem):
            original_levels.append(item.level)

    for item, _ in hierarchical_doc.iterate_items(with_groups=True):
        if isinstance(item, SectionHeaderItem):
            hierarchical_levels.append(item.level)

    # The levels should be the same (order and values)
    assert original_levels == hierarchical_levels, "Section header levels should be preserved in hierarchical document"

    # Verify that section headers are properly nested by level
    # A section header at level N should be a child of a section header at level < N
    # or a direct child of body if it's a top-level section
    for item, _ in hierarchical_doc.iterate_items(with_groups=True):
        if isinstance(item, SectionHeaderItem) and item.parent:
            parent = item.parent.resolve(hierarchical_doc)
            if isinstance(parent, SectionHeaderItem):
                # Parent section should have a lower level than child
                assert parent.level < item.level, (
                    f"Parent section (level {parent.level}) should have lower level than child (level {item.level})"
                )


def test_create_document_outline():
    """Test that create_document_outline generates the expected outline format using OutlineDocSerializer."""

    json_path = Path(__file__).parent / "data" / "2408.09869v5.json"
    with open(json_path) as f:
        doc_dict = json.load(f)
    doc = DoclingDocument.model_validate(doc_dict)

    # Generate the outline
    outline_json = create_document_outline(doc, mode=OutlineMode.OUTLINE)

    # Path to ground truth file
    ground_truth_path = Path(__file__).parent / "data" / "2408.09869v5_outline.json"

    if GEN_TEST_DATA:
        # Regenerate ground truth
        with open(ground_truth_path, "w") as f:
            f.write(outline_json)
        print(f"Generated ground truth file: {ground_truth_path}")
    else:
        # Compare against ground truth
        assert ground_truth_path.exists(), f"Ground truth file not found: {ground_truth_path}"

        with open(ground_truth_path) as f:
            expected_outline_json = f.read()

        # Parse both JSON strings to compare as data structures (order-independent)
        outline_data = json.loads(outline_json)
        expected_data = json.loads(expected_outline_json)

        assert outline_data == expected_data, "Generated outline does not match ground truth"

    # Additional assertions to verify outline structure
    assert isinstance(outline_json, str), "Outline should be a JSON string"
    assert len(outline_json) > 0, "Outline should not be empty"

    # Parse and verify JSON structure
    outline_data = json.loads(outline_json)
    assert isinstance(outline_data, list), "Outline should be a JSON array"
    assert len(outline_data) > 0, "Outline should contain items"

    # Verify each item has required fields
    for item in outline_data:
        assert "ref" in item, "Each item should have a 'ref' field"
        assert "item" in item, "Each item should have an 'item' field (item type)"
        assert item["ref"].startswith("#/"), "Reference should be a JSON pointer"

    # Verify we have different item types
    item_types = {item["item"] for item in outline_data}
    assert len(item_types) > 1, "Outline should contain multiple item types"

    # Verify section headers have title and level
    section_headers = [item for item in outline_data if item.get("item") == "section_header"]
    assert len(section_headers) > 0, "Outline should contain section headers"
    for header in section_headers:
        assert "title" in header, "Section headers should have a title"
        assert "level" in header, "Section headers should have a level"

    # Verify the document contains tables and pictures (as we know 2408.09869v5.json has them)
    tables = [item for item in outline_data if item.get("item") == "table"]
    pictures = [item for item in outline_data if item.get("item") == "picture"]
    assert len(tables) > 0, "Outline should contain at least one table"
    assert len(pictures) > 0, "Outline should contain at least one picture"

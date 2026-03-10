from docling_agent.agent.base_functions import find_json_dicts


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

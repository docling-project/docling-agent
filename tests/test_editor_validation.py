"""Tests for the DoclingEditingAgent."""

import pytest
from pydantic import ValidationError

from docling_agent.agent.editor import (
    RewriteContentOperation,
    UpdateContentOperation,
    UpdateSectionHeadingLevelOperation,
)


class TestUpdateContentOperation:
    """Test UpdateContentOperation validation."""

    def test_valid_update_content_operation(self):
        """Test valid update_content operation."""
        op = UpdateContentOperation(operation="update_content", ref="#/texts/1")
        assert op.ref == "#/texts/1"

    def test_valid_update_content_with_nested_ref(self):
        """Test valid update_content with nested reference."""
        op = UpdateContentOperation(operation="update_content", ref="#/section-header/5")
        assert op.ref == "#/section-header/5"

    def test_invalid_ref_pattern(self):
        """Test that invalid ref pattern raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UpdateContentOperation(operation="update_content", ref="invalid-ref")
        # Pydantic's built-in pattern validation error message
        assert "String should match pattern" in str(exc_info.value)

    def test_missing_ref_field(self):
        """Test that missing ref field raises ValidationError."""
        with pytest.raises(ValidationError):
            UpdateContentOperation(operation="update_content")


class TestRewriteContentOperation:
    """Test RewriteContentOperation validation."""

    def test_valid_rewrite_content_operation(self):
        """Test valid rewrite_content operation."""
        op = RewriteContentOperation(operation="rewrite_content", refs=["#/texts/1", "#/texts/2"])
        assert len(op.refs) == 2

    def test_invalid_ref_in_list(self):
        """Test that invalid ref in list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RewriteContentOperation(operation="rewrite_content", refs=["#/texts/1", "invalid-ref"])
        # Pydantic's built-in pattern validation error message
        assert "String should match pattern" in str(exc_info.value)


class TestUpdateSectionHeadingLevelOperation:
    """Test UpdateSectionHeadingLevelOperation validation."""

    def test_valid_update_section_heading_level_operation(self):
        """Test valid update_section_heading_level operation."""
        op = UpdateSectionHeadingLevelOperation(
            operation="update_section_heading_level",
            changes=[{"ref": "#/section-header/1", "to_level": 2}, {"ref": "#/section-header/2", "to_level": 3}],
        )
        assert len(op.changes) == 2
        assert op.changes[0].ref == "#/section-header/1"
        assert op.changes[0].to_level == 2
        assert op.changes[1].ref == "#/section-header/2"
        assert op.changes[1].to_level == 3

    def test_empty_changes_list(self):
        """Test that empty changes list is valid."""
        op = UpdateSectionHeadingLevelOperation(operation="update_section_heading_level", changes=[])
        assert isinstance(op, UpdateSectionHeadingLevelOperation)
        assert op.changes == []

    def test_invalid_ref_in_changes(self):
        """Test that invalid ref in changes raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UpdateSectionHeadingLevelOperation(
                operation="update_section_heading_level", changes=[{"ref": "invalid-ref", "to_level": 2}]
            )
        # Pydantic's built-in pattern validation error message
        assert "String should match pattern" in str(exc_info.value)

    def test_negative_level_value(self):
        """Test that negative level value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UpdateSectionHeadingLevelOperation(
                operation="update_section_heading_level", changes=[{"ref": "#/section-header/1", "to_level": -1}]
            )
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_valid_zero_level(self):
        """Test that zero level is valid."""
        op = UpdateSectionHeadingLevelOperation(
            operation="update_section_heading_level", changes=[{"ref": "#/section-header/1", "to_level": 0}]
        )
        assert op.changes[0].to_level == 0


class TestJsonPointerPatternValidation:
    """Test JSON pointer pattern validation across all operations."""

    @pytest.mark.parametrize(
        "valid_ref",
        [
            "#",
            "#/texts/0",
            "#/section-header/1",
            "#/main-text/42",
            "#/body",
            "#/pictures/5",
            "#/tables/10",
        ],
    )
    def test_valid_json_pointer_patterns(self, valid_ref):
        """Test that valid JSON pointer patterns are accepted."""
        op = UpdateContentOperation(operation="update_content", ref=valid_ref)
        assert op.ref == valid_ref

    @pytest.mark.parametrize(
        "invalid_ref",
        [
            "texts/1",  # Missing #
            "#texts/1",  # Missing /
            "#/texts",  # Missing index (should be valid actually based on regex)
            "#/texts/abc",  # Non-numeric index
            "#/texts/1/extra",  # Too many parts
            "invalid",  # Completely invalid
            "",  # Empty string
        ],
    )
    def test_invalid_json_pointer_patterns(self, invalid_ref):
        """Test that invalid JSON pointer patterns are rejected."""
        # Note: Some of these might actually be valid based on the regex pattern
        # The regex is: r"^#(?:/([\w-]+)(?:/(\d+))?)?$"
        # This allows: #, #/name, #/name/123
        if invalid_ref in ["#/texts"]:  # This is actually valid
            op = UpdateContentOperation(operation="update_content", ref=invalid_ref)
            assert op.ref == invalid_ref
        else:
            with pytest.raises(ValidationError):
                UpdateContentOperation(operation="update_content", ref=invalid_ref)

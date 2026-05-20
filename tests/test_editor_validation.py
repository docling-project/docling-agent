"""Tests for the DoclingEditingAgent."""

import pytest
from docling_core.types.doc.document import DocItemLabel, DoclingDocument, SectionHeaderItem, TextItem, TitleItem
from pydantic import ValidationError

from docling_agent.agent.base import DoclingAgentType
from docling_agent.agent.editor import (
    DoclingEditingAgent,
    MissingSectionHeadingInsertion,
    RewriteContentOperation,
    SectionHeadingLevelChange,
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

    def test_valid_minus_one_level(self):
        """Test that -1 level is valid for converting to TextItem."""
        op = UpdateSectionHeadingLevelOperation(
            operation="update_section_heading_level", changes=[{"ref": "#/section-header/1", "to_level": -1}]
        )
        assert op.changes[0].to_level == -1

    def test_level_below_minus_one_is_invalid(self):
        """Test that values below -1 still raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UpdateSectionHeadingLevelOperation(
                operation="update_section_heading_level", changes=[{"ref": "#/section-header/1", "to_level": -2}]
            )
        assert "greater than or equal to -1" in str(exc_info.value).lower()

    def test_valid_zero_level(self):
        """Test that zero level is valid."""
        op = UpdateSectionHeadingLevelOperation(
            operation="update_section_heading_level", changes=[{"ref": "#/section-header/1", "to_level": 0}]
        )
        assert op.changes[0].to_level == 0

    def test_valid_insertions_payload(self):
        """Test that insertions payload is accepted."""
        op = UpdateSectionHeadingLevelOperation(
            operation="update_section_heading_level",
            changes=[{"ref": "#/section-header/1", "to_level": 2}],
            insertions=[
                {
                    "previous_ref": "#/section-header/9",
                    "next_ref": "#/section-header/11",
                    "regex": r"^9\.3\s+.+$",
                    "level": 2,
                }
            ],
        )
        assert len(op.insertions) == 1
        assert op.insertions[0].previous_ref == "#/section-header/9"
        assert op.insertions[0].next_ref == "#/section-header/11"
        assert op.insertions[0].regex == r"^9\.3\s+.+$"
        assert op.insertions[0].level == 2


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


class TestSectionHeadingFixHelpers:
    @staticmethod
    def _agent() -> DoclingEditingAgent:
        return DoclingEditingAgent.model_construct(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EDITOR,
            backend=None,
            tools=[],
        )

    def test_converts_section_header_to_text_item(self):
        doc = DoclingDocument(name="test")
        header = doc.add_heading(text="Appendix note", level=1, parent=doc.body)

        self._agent()._update_section_heading_level(
            task="",
            document=doc,
            changes=[SectionHeadingLevelChange(ref=header.self_ref, to_level=-1)],
            insertions=[],
        )

        updated = doc.texts[int(header.self_ref.split("/")[-1])]
        assert isinstance(updated, TextItem)
        assert not isinstance(updated, SectionHeaderItem)
        assert updated.label == DocItemLabel.TEXT
        assert updated.text == "Appendix note"

    def test_converts_section_header_to_title_item(self):
        doc = DoclingDocument(name="test")
        header = doc.add_heading(text="My document", level=1, parent=doc.body)

        self._agent()._update_section_heading_level(
            task="",
            document=doc,
            changes=[SectionHeadingLevelChange(ref=header.self_ref, to_level=0)],
            insertions=[],
        )

        updated = doc.texts[int(header.self_ref.split("/")[-1])]
        assert isinstance(updated, TitleItem)
        assert updated.text == "My document"

    def test_rejects_second_title_item(self):
        doc = DoclingDocument(name="test")
        doc.add_title(text="Existing title", parent=doc.body)
        header = doc.add_heading(text="Another title", level=1, parent=doc.body)

        with pytest.raises(ValueError, match="already contains"):
            self._agent()._update_section_heading_level(
                task="",
                document=doc,
                changes=[SectionHeadingLevelChange(ref=header.self_ref, to_level=0)],
                insertions=[],
            )

    def test_inserts_missing_section_header_by_regex(self):
        doc = DoclingDocument(name="test")
        previous = doc.add_heading(text="9.1 Alpha", level=2, parent=doc.body)
        candidate = doc.add_text(label=DocItemLabel.TEXT, text="9.3 Gamma", parent=doc.body)
        next_header = doc.add_heading(text="9.4 Delta", level=2, parent=doc.body)

        self._agent()._update_section_heading_level(
            task="",
            document=doc,
            changes=[],
            insertions=[
                MissingSectionHeadingInsertion(
                    previous_ref=previous.self_ref,
                    next_ref=next_header.self_ref,
                    regex=r"^9\.3\s+.+$",
                    level=2,
                )
            ],
        )

        updated = doc.texts[int(candidate.self_ref.split("/")[-1])]
        assert isinstance(updated, SectionHeaderItem)
        assert updated.level == 2
        assert updated.text == "9.3 Gamma"

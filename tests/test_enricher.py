import json
import re
from pathlib import Path

import pytest
from docling_core.transforms.serializer.markdown import (
    ImageRefMode,
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import DoclingDocument, EntityMention

from docling_agent.agent.editor import DoclingEditingAgent
from docling_agent.agent.enricher import DoclingEnrichingAgent

from .test_utils import MockBackend


@pytest.fixture(scope="module")
def mock_backend():
    """Fixture providing a mocked backend instance."""
    return MockBackend()


@pytest.fixture(scope="module")
def test_document():
    """Fixture providing the test document loaded from JSON."""
    json_path = Path("tests/data/2408.09869v5.json")
    with open(json_path) as f:
        doc_dict = json.load(f)
    return DoclingDocument.model_validate(doc_dict)


@pytest.fixture(scope="module")
def enricher(mock_backend):
    """Fixture providing a DoclingEnrichingAgent instance."""
    return DoclingEnrichingAgent(backend=mock_backend, tools=[])


@pytest.fixture(scope="module")
def markdown_serializer(test_document):
    """Fixture providing a configured MarkdownDocSerializer."""
    md_params = MarkdownParams(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        escape_underscores=False,
        escape_html=False,
        compact_tables=True,
        traverse_pictures=True,
    )
    return MarkdownDocSerializer(
        doc=test_document,
        table_serializer=MarkdownTableSerializer(),
        params=md_params,
    )


def test_heading_levels_problem_exists(test_document):
    """Demonstrate that the document has incorrect heading levels.

    The document has:
    - "3 Processing pipeline" with level 1
    - "3.4 Extensibility" with level 1

    This is incorrect because "3.4 Extensibility" should have level > 1
    since it's a subsection of "3 Processing pipeline".
    """
    document = test_document

    # Find the section headers
    processing_pipeline = None
    extensibility = None
    subsections = []

    for item, _ in document.iterate_items():
        if hasattr(item, "text") and hasattr(item, "level"):
            if item.text == "3 Processing pipeline":
                processing_pipeline = item
                print(f"Found '3 Processing pipeline': level={item.level}")
            elif item.text and item.text.startswith("3.") and item.text != "3 Processing pipeline":
                subsections.append((item.text, item.level))
                if item.text == "3.4 Extensibility":
                    extensibility = item

    print("\nAll subsections of section 3:")
    for text, level in subsections:
        print(f"  {text}: level={level}")

    assert processing_pipeline is not None, "Could not find '3 Processing pipeline' section"
    assert extensibility is not None, "Could not find '3.4 Extensibility' section"

    # Both have level 1 (the problem)
    assert processing_pipeline.level == 1
    assert extensibility.level == 1

    print("\n❌ PROBLEM CONFIRMED:")
    print(f"   '3 Processing pipeline' has level {processing_pipeline.level}")
    print(f"   '3.4 Extensibility' has level {extensibility.level}")
    print("   But '3.4 Extensibility' should have level > 1 since it's a subsection!")


def test_fix_heading_levels(monkeypatch, test_document, enricher):
    """Test that _fix_heading_levels correctly adjusts section header levels.

    This test shows the before/after state of all section headers in the document.
    The main issue is that subsections (e.g., "3.1", "3.2", "3.4") have the same
    level as their parent section ("3 Processing pipeline").

    The editor call is stubbed so the test stays deterministic and does not
    depend on a live backend.
    """

    def _fake_editor_run(self, task: str, document: DoclingDocument | None = None, **kwargs):
        assert document is not None

        for item, _ in document.iterate_items():
            if not (hasattr(item, "text") and hasattr(item, "level") and item.label == "section_header"):
                continue

            text = item.text or ""
            if re.match(r"^\d+\.\d+(?:\.\d+)*\s", text):
                item.level = 2
            elif re.match(r"^\d+\s", text):
                item.level = 1

        return document

    monkeypatch.setattr(DoclingEditingAgent, "run", _fake_editor_run)

    # Expected section headers BEFORE fixing (all at level 1 - incorrect!)
    expected_before = [
        ("Docling Technical Report", 1),
        ("Version 1.0", 1),
        ("Abstract", 1),
        ("1 Introduction", 1),
        ("2 Getting Started", 1),
        ("3 Processing pipeline", 1),
        ("3.1 PDF backends", 1),
        ("3.2 AI models", 1),
        ("Layout Analysis Model", 1),
        ("Table Structure Recognition", 1),
        ("OCR", 1),
        ("3.3 Assembly", 1),
        ("3.4 Extensibility", 1),
        ("4 Performance", 1),
        ("5 Applications", 1),
        ("6 Future work and contributions", 1),
        ("References", 1),
        ("Appendix", 1),
        ("5 EXPERIMENTS", 1),
        ("Baselines for Object Detection", 1),
    ]

    document = test_document

    # Extract section headers BEFORE fixing
    headers_before = []
    for item, _ in document.iterate_items():
        if hasattr(item, "text") and hasattr(item, "level") and item.label == "section_header":
            headers_before.append((item.text, item.level))

    print("\n" + "=" * 70)
    print("BEFORE _fix_heading_levels:")
    print("=" * 70)
    for text, level in headers_before:
        print(f"  {text:50s} level={level}")

    # Verify the before state matches expectations
    assert headers_before == expected_before, "Before state doesn't match expected"

    # Call _fix_heading_levels
    print("\n" + "=" * 70)
    print("Calling _fix_heading_levels with stubbed editor...")
    print("=" * 70)
    enricher._fix_heading_levels(document=document)

    # Extract section headers AFTER fixing
    headers_after = []
    for item, _ in document.iterate_items():
        if hasattr(item, "text") and hasattr(item, "level") and item.label == "section_header":
            headers_after.append((item.text, item.level))

    print("\n" + "=" * 70)
    print("AFTER _fix_heading_levels:")
    print("=" * 70)
    for text, level in headers_after:
        print(f"  {text:50s} level={level}")

    # Verify that levels have changed
    assert headers_after != headers_before, "Heading levels were not modified"

    # Key assertions: subsections should have higher levels than their parent sections
    headers_dict = dict(headers_after)

    # "3 Processing pipeline" should have a lower level than its subsections
    pipeline_level = headers_dict.get("3 Processing pipeline")
    assert pipeline_level is not None, "Could not find '3 Processing pipeline'"

    for subsection in ["3.1 PDF backends", "3.2 AI models", "3.3 Assembly", "3.4 Extensibility"]:
        subsection_level = headers_dict.get(subsection)
        assert subsection_level is not None, f"Could not find '{subsection}'"
        assert subsection_level > pipeline_level, (
            f"'{subsection}' (level {subsection_level}) should have higher level than '3 Processing pipeline' (level {pipeline_level})"
        )

    print("\n" + "=" * 70)
    print("✓ Test passed: heading levels were correctly fixed!")
    print("=" * 70)


def test_make_entity_mention(enricher):
    """Regression test for _make_entity_mention function."""
    source_text = "International Business Machines is a technology company based in Armonk."

    # Basic entity mention
    item = {"text": "International Business Machines"}
    mention = enricher._make_entity_mention(item=item, source_text=source_text)
    assert isinstance(mention, EntityMention)
    assert mention.text == "International Business Machines"
    assert mention.charspan == (0, 31)

    # Entity with original and label
    item = {"text": "IBM", "original": "International Business Machines", "label": "ORGANIZATION"}
    mention = enricher._make_entity_mention(item=item, source_text=source_text)
    assert mention.text == "IBM"
    assert mention.orig == "International Business Machines"
    assert mention.label == "ORGANIZATION"
    assert mention.charspan == (0, 31)

    # Case-insensitive matching
    item = {"text": "international business machines"}
    mention = enricher._make_entity_mention(item=item, source_text=source_text)
    assert mention.charspan == (0, 31)

    # Entity not found
    item = {"text": "Red Hat"}
    mention = enricher._make_entity_mention(item=item, source_text=source_text)
    assert mention.charspan is None

    # Original takes precedence in search
    source_abbrev = "The CEO of IBM announced products."
    item = {"text": "International Business Machines", "original": "IBM"}
    mention = enricher._make_entity_mention(item=item, source_text=source_abbrev)
    assert mention.charspan == (11, 14)


def test_generate_summary(monkeypatch, enricher):
    """Regression test for _generate_summary function with different styles and scopes."""

    def _fake_generate_content(self, *, m, text, task_prompt, requirement_description, validation_fn, loop_budget=5):
        # Return different outputs based on the requirement_description to simulate different styles
        if "key phrases" in requirement_description.lower():
            return "concept A; concept B; concept C"
        else:
            return "This is a test summary. It has multiple sentences."

    monkeypatch.setattr(DoclingEnrichingAgent, "_generate_content", _fake_generate_content)

    m = enricher._create_extraction_session()

    # Test sentences style with section scope (default)
    summary = enricher._generate_summary(m=m, text="Sample text", style="sentences", scope="section")
    assert summary == "This is a test summary. It has multiple sentences."

    # Test keyphrases style with section scope
    summary = enricher._generate_summary(m=m, text="Sample text", style="keyphrases", scope="section")
    assert summary == "concept A; concept B; concept C"

    # Test sentences style with document scope
    summary = enricher._generate_summary(m=m, text="Sample text", style="sentences", scope="document")
    assert summary == "This is a test summary. It has multiple sentences."

    # Test keyphrases style with document scope
    summary = enricher._generate_summary(m=m, text="Sample text", style="keyphrases", scope="document")
    assert summary == "concept A; concept B; concept C"


def test_generate_document_level_summary(monkeypatch, test_document, enricher, markdown_serializer):
    """Regression test for _generate_document_level_summary function."""

    def _fake_generate_summary(self, *, m, text, loop_budget=5, style="sentences", scope="section"):
        return "Document-level summary from first pages."

    monkeypatch.setattr(DoclingEnrichingAgent, "_generate_summary", _fake_generate_summary)

    document = test_document
    serializer = markdown_serializer

    # Test with default parameters
    summary = enricher._generate_document_level_summary(
        document=document, serializer=serializer, num_pages=3, style="sentences"
    )
    assert summary == "Document-level summary from first pages."

    # Test with keyphrases style
    summary = enricher._generate_document_level_summary(
        document=document, serializer=serializer, num_pages=2, style="keyphrases"
    )
    assert summary == "Document-level summary from first pages."


def test_summarize_pages(monkeypatch, test_document, enricher):
    """Regression test for _summarize_pages function."""

    def _fake_generate_summary(self, *, m, text, loop_budget=5, style="sentences", scope="section"):
        if scope == "document":
            return "Overall document summary."
        return f"Page summary in {style} style."

    monkeypatch.setattr(DoclingEnrichingAgent, "_generate_summary", _fake_generate_summary)

    document = test_document

    # Test with keyphrases style (default)
    result_doc = enricher._summarize_pages(document=document, style="keyphrases")

    # Verify document-level summary was added to body
    assert result_doc.body.meta is not None
    assert result_doc.body.meta.summary is not None
    assert result_doc.body.meta.summary.text == "Overall document summary."

    # Verify page-level summaries were added
    pages_with_summaries = 0
    for page_no in document.pages.keys():
        try:
            first_item, _ = next(iter(document.iterate_items(traverse_pictures=True, page_no=page_no)))
            if hasattr(first_item, "meta") and first_item.meta and first_item.meta.summary:
                pages_with_summaries += 1
                assert first_item.meta.summary.text == "Page summary in keyphrases style."
        except StopIteration:
            continue

    # At least some pages should have summaries
    assert pages_with_summaries > 0


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Demonstrating the heading levels problem")
    print("=" * 70)
    test_heading_levels_problem_exists()

    print("\n" + "=" * 70)
    print("TEST 2: Testing _fix_heading_levels function")
    print("=" * 70)
    test_fix_heading_levels()

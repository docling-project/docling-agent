import json
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from mellea.backends import model_ids

from docling_agent.agent.enricher import DoclingEnrichingAgent


def test_heading_levels_problem_exists():
    """Demonstrate that the document has incorrect heading levels.

    The document has:
    - "3 Processing pipeline" with level 1
    - "3.4 Extensibility" with level 1

    This is incorrect because "3.4 Extensibility" should have level > 1
    since it's a subsection of "3 Processing pipeline".
    """
    # Load the test document
    json_path = Path("tests/data/2408.09869v5.json")
    with open(json_path) as f:
        doc_dict = json.load(f)

    # Fix version compatibility issue
    doc_dict["version"] = "1.8.0"

    document = DoclingDocument.model_validate(doc_dict)

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


@pytest.mark.slow
def test_fix_heading_levels():
    """Test that _fix_heading_levels correctly adjusts section header levels.

    This test shows the before/after state of all section headers in the document.
    The main issue is that subsections (e.g., "3.1", "3.2", "3.4") have the same
    level as their parent section ("3 Processing pipeline").

    Note: This test is marked as 'slow' because it makes LLM calls.
    Skip with: pytest -m "not slow"
    """
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

    # Load the test document
    json_path = Path("tests/data/2408.09869v5.json")
    with open(json_path) as f:
        doc_dict = json.load(f)

    # Fix version compatibility issue
    doc_dict["version"] = "1.8.0"

    document = DoclingDocument.model_validate(doc_dict)

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
    enricher = DoclingEnrichingAgent(model_id=model_ids.OPENAI_GPT_OSS_20B, tools=[])

    print("\n" + "=" * 70)
    print("Calling _fix_heading_levels...")
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


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Demonstrating the heading levels problem")
    print("=" * 70)
    test_heading_levels_problem_exists()

    print("\n" + "=" * 70)
    print("TEST 2: Testing _fix_heading_levels function")
    print("=" * 70)
    test_fix_heading_levels()

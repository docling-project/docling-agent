"""Tests for DoclingOrchestratorAgent.

This module includes regression tests for the entity constraint bug fix,
specifically testing task propagation from orchestrator to enricher.
"""

import tempfile
from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import DoclingDocument

from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.library import DocLibraryEntry, DoclingLibrary
from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.agent.writer import DoclingWritingAgent
from docling_agent.task_model import AddTask, ClearTask, ListTask, ViewTask, WriteTask


@pytest.fixture
def enricher(mock_backend) -> DoclingEnrichingAgent:
    """Fixture providing a DoclingEnrichingAgent instance."""
    return DoclingEnrichingAgent(backend=mock_backend, tools=[])


@pytest.fixture
def orchestrator(mock_backend) -> DoclingOrchestratorAgent:
    """Fixture providing a DoclingOrchestratorAgent instance."""
    return DoclingOrchestratorAgent(backend=mock_backend, tools=[])


def test_ensure_enriched_task_propagation(
    monkeypatch: pytest.MonkeyPatch, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    """Test that _ensure_enriched propagates task parameter to enricher.run().

    This is a regression test for the bug fix where task was not being passed
    from orchestrator to enricher, causing entity constraints to be ignored.
    """
    captured_task = []

    def _fake_enricher_run(self, task, document=None, sources=None, operations=None, **kwargs):
        captured_task.append(task)
        return document

    # Mock the enricher's run method
    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    # Create a mock library with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        entry = library.store(test_document, "test.json")

        # Call _ensure_enriched with a task
        source_pairs = [(test_document, entry.doc_id)]
        task_query = "Extract person names and email addresses"

        orchestrator._ensure_enriched(
            source_pairs=source_pairs, library=library, operations=["entities"], task=task_query
        )

        # Verify task was passed to enricher
        assert len(captured_task) == 1
        assert captured_task[0] == task_query


def test_ensure_enriched_empty_task_default(
    monkeypatch: pytest.MonkeyPatch, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    """Test that _ensure_enriched uses empty string as default task."""
    captured_task = []

    def _fake_enricher_run(self, task, document=None, sources=None, operations=None, **kwargs):
        captured_task.append(task)
        return document

    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        entry = library.store(test_document, "test.json")

        source_pairs = [(test_document, entry.doc_id)]

        # Call without task parameter (should default to "")
        orchestrator._ensure_enriched(source_pairs=source_pairs, library=library, operations=["entities"])

        assert len(captured_task) == 1
        assert captured_task[0] == ""


def test_ensure_enriched_multiple_operations(
    monkeypatch: pytest.MonkeyPatch, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    """Test that _ensure_enriched handles multiple operations correctly."""
    enricher_calls = []

    def _fake_enricher_run(self, task, document=None, sources=None, operations=None, **kwargs):
        enricher_calls.append((task, operations))
        return document

    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        entry = library.store(test_document, "test.json")

        source_pairs = [(test_document, entry.doc_id)]
        task_query = "Enrich with summaries and entities"

        # Request multiple operations
        orchestrator._ensure_enriched(
            source_pairs=source_pairs,
            library=library,
            operations=["summarize", "entities"],
            task=task_query,
        )

        # Enricher should be called once with both operations
        assert len(enricher_calls) == 1
        assert enricher_calls[0][0] == task_query
        assert set(enricher_calls[0][1]) == {"summarize", "entities"}


def test_add_mode_adds_sources_to_library(
    tmp_path: Path, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    source_path = tmp_path / "document.json"
    source_path.write_text(test_document.model_dump_json(indent=2), encoding="utf-8")
    library_path = tmp_path / "library"
    orchestrator.library_path = library_path

    result = orchestrator.run_task(AddTask(project_id="alpha", sources=[str(source_path)]))
    library = DoclingLibrary(path=library_path, project_id="alpha")
    entries = library.all_entries()
    markdown = MarkdownDocSerializer(doc=result).serialize().text

    assert len(entries) == 1
    assert entries[0].project_id == "alpha"
    assert "Added 1 document" in markdown
    assert entries[0].doc_id in markdown


def test_build_source_converter_applies_named_presets(orchestrator: DoclingOrchestratorAgent) -> None:
    fast = orchestrator._build_source_converter("fast")
    standard = orchestrator._build_source_converter("standard")
    expensive = orchestrator._build_source_converter("expensive")

    fast_pdf = fast.format_to_options[InputFormat.PDF].pipeline_options
    standard_pdf = standard.format_to_options[InputFormat.PDF].pipeline_options
    expensive_pdf = expensive.format_to_options[InputFormat.PDF].pipeline_options

    assert fast_pdf.do_ocr is False
    assert fast_pdf.do_table_structure is False
    assert fast_pdf.do_picture_classification is True
    assert fast_pdf.do_chart_extraction is False
    assert fast_pdf.generate_page_images is True

    assert standard_pdf.do_ocr is True
    assert standard_pdf.do_table_structure is True
    assert standard_pdf.do_picture_classification is True
    assert standard_pdf.do_chart_extraction is False
    assert standard_pdf.generate_page_images is True

    assert expensive_pdf.do_ocr is True
    assert expensive_pdf.do_table_structure is True
    assert expensive_pdf.do_picture_classification is True
    assert expensive_pdf.do_chart_extraction is True
    assert expensive_pdf.generate_page_images is True


def test_cache_entry_must_match_conversion_preset(
    tmp_path: Path, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    library = DoclingLibrary(path=tmp_path)
    entry = library.store(test_document, "/tmp/document.pdf", conversion_pipeline="StandardPdfPipeline:fast")
    legacy_entry = library.store(test_document, "/tmp/legacy.pdf", conversion_pipeline="StandardPdfPipeline")

    assert orchestrator._cache_entry_matches_conversion(entry, conversion="fast") is True
    assert orchestrator._cache_entry_matches_conversion(entry, conversion="standard") is False
    assert orchestrator._cache_entry_matches_conversion(legacy_entry, conversion="standard") is True
    assert orchestrator._cache_entry_matches_conversion(legacy_entry, conversion="expensive") is False


def test_list_mode_lists_all_projects(
    tmp_path: Path, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    library = DoclingLibrary(path=tmp_path)
    library.store(test_document, "/tmp/default.pdf", project_id="default")
    library.store(test_document, "/tmp/alpha.pdf", project_id="alpha")

    result = orchestrator._dispatch(ListTask(limit=10), library)
    markdown = MarkdownDocSerializer(doc=result).serialize().text

    assert "default" in markdown
    assert "alpha" in markdown
    assert "/tmp/default.pdf" in markdown
    assert "/tmp/alpha.pdf" in markdown


def test_view_mode_uses_postgres_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    orchestrator: DoclingOrchestratorAgent,
    test_document: DoclingDocument,
) -> None:
    library = DoclingLibrary(path=tmp_path)
    entry = library.store(
        test_document,
        "/tmp/default.pdf",
        project_id="default",
        original_mimetype="application/pdf",
    )
    calls = []

    def _fake_query(self: DoclingLibrary, postgres_filter: str, *, limit: int = 100) -> list[DocLibraryEntry]:
        calls.append((postgres_filter, limit))
        return [entry]

    monkeypatch.setattr(DoclingLibrary, "query_entries_by_postgres_filter", _fake_query)

    result = orchestrator._dispatch(ViewTask(postgres_filter="project_id = 'default'", limit=7), library)
    markdown = MarkdownDocSerializer(doc=result).serialize().text

    assert calls == [("project_id = 'default'", 7)]
    assert entry.doc_id in markdown
    assert "application/pdf" in markdown
    assert "has\\_summaries: False" in markdown


def test_clear_mode_removes_project_entries(
    tmp_path: Path, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    library = DoclingLibrary(path=tmp_path)
    removed_entry = library.store(test_document, "/tmp/alpha.pdf", project_id="alpha")
    kept_entry = library.store(test_document, "/tmp/beta.pdf", project_id="beta")

    result = orchestrator._dispatch(ClearTask(project_id="alpha"), library)
    markdown = MarkdownDocSerializer(doc=result).serialize().text

    assert "Removed 1 document" in markdown
    assert library.get_entry(removed_entry.doc_id) is None
    assert library.get_entry(kept_entry.doc_id) is not None


def test_write_mode_stores_written_document_in_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orchestrator: DoclingOrchestratorAgent
) -> None:
    library = DoclingLibrary(path=tmp_path, project_id="reports")
    written_doc = DoclingDocument(name="generated_report")

    def _fake_writer_run(self, task, document=None, sources=None, **kwargs):
        return written_doc

    monkeypatch.setattr(DoclingWritingAgent, "run", _fake_writer_run)

    result = orchestrator._dispatch(WriteTask(project_id="reports", query="write report"), library)
    entries = library.query_entries(project_id="reports", document_origin="written")

    assert result is written_doc
    assert len(entries) == 1
    assert entries[0].name == "generated_report"
    assert entries[0].project_id == "reports"
    assert entries[0].document_origin == "written"
    assert entries[0].source_path == f"written:{entries[0].doc_id}"
    assert Path(entries[0].doc_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

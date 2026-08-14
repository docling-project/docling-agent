from pathlib import Path

import pytest
from docling_core.types.doc.document import DocItemLabel, DoclingDocument, TableData

from docling_agent.agent.library import DoclingLibrary


def test_store_uses_dclx_payload(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    doc.add_text(label=DocItemLabel.TEXT, text="hello", parent=doc.body)

    library = DoclingLibrary(path=tmp_path)
    entry = library.store(
        doc,
        "/tmp/sample.pdf",
        original_mimetype="application/pdf",
        conversion_pipeline="StandardPdfPipeline",
    )

    doc_path = Path(entry.doc_path)
    assert entry.doc_format == "dclx"
    assert entry.document_origin == "converted"
    assert entry.original_mimetype == "application/pdf"
    assert doc_path.name == f"document_{entry.doc_id}.dclx"
    assert doc_path.exists()
    assert not (tmp_path / entry.doc_id / "document.json").exists()
    assert [run.name for run in entry.status.pipelines] == ["StandardPdfPipeline"]
    assert entry.stats.page_count is None
    assert entry.stats.table_count == 0
    assert entry.stats.picture_count == 0
    assert entry.stats.text_count == 1
    assert entry.stats.xml_char_count > 0

    loaded = library.load_doc(entry.doc_id)
    assert loaded is not None
    assert loaded.name == "sample"


def test_store_records_document_stats(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    doc.add_text(label=DocItemLabel.TEXT, text="hello", parent=doc.body)
    doc.add_table(data=TableData(num_rows=1, num_cols=1), parent=doc.body)

    library = DoclingLibrary(path=tmp_path)
    entry = library.store(doc, "/tmp/sample.pdf")

    assert entry.stats.page_count is None
    assert entry.stats.table_count == 1
    assert entry.stats.picture_count == 0
    assert entry.stats.text_count == 1
    assert entry.stats.xml_char_count > 0


def test_project_id_defaults_and_isolates_sources(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    default_library = DoclingLibrary(path=tmp_path)
    other_library = DoclingLibrary(path=tmp_path, project_id="other")

    default_entry = default_library.store(doc, "/tmp/sample.pdf")
    other_entry = other_library.store(doc, "/tmp/sample.pdf")

    assert default_entry.project_id == "default"
    assert other_entry.project_id == "other"
    assert default_entry.doc_id != other_entry.doc_id
    assert default_library.lookup_by_source("/tmp/sample.pdf") == default_entry
    assert other_library.lookup_by_source("/tmp/sample.pdf") == other_entry


def test_query_entries_filters_local_index(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path, project_id="alpha")
    first = library.store(
        DoclingDocument(name="invoice"),
        "/tmp/invoice.pdf",
        original_mimetype="application/pdf",
        conversion_pipeline="StandardPdfPipeline",
    )
    second = library.store(
        DoclingDocument(name="notes"),
        "/tmp/notes.md",
        original_mimetype="text/markdown",
        conversion_pipeline="SimplePipeline",
    )
    library.record_enrichments(first.doc_id, ["summarize", "keywords"], task="summarize invoice")
    library.update_meta(first.doc_id, summary="Quarterly billing statement", keywords=["billing"])

    assert library.query_entries(mimetype="application/pdf") == [library.get_entry(first.doc_id)]
    assert library.query_entries(pipeline="SimplePipeline") == [second]
    assert [entry.doc_id for entry in library.query_entries(enrichments=["summarize"])] == [first.doc_id]
    assert [entry.doc_id for entry in library.query_entries(text="billing")] == [first.doc_id]


def test_store_in_memory_uses_origin_specific_source_key(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path, project_id="alpha")

    written = library.store_in_memory(DoclingDocument(name="written"), document_origin="written")
    in_memory = library.store_in_memory(DoclingDocument(name="memory"))

    assert written.project_id == "alpha"
    assert written.document_origin == "written"
    assert written.source_path == f"written:{written.doc_id}"
    assert in_memory.document_origin == "in_memory"
    assert in_memory.source_path == f"in_memory:{in_memory.doc_id}"
    assert written.source_path != in_memory.source_path
    assert library.query_entries(document_origin="written") == [written]
    assert library.query_entries(document_origin="in_memory") == [in_memory]


def test_clear_project_removes_entries_and_document_dirs(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    library = DoclingLibrary(path=tmp_path)
    alpha = library.store(doc, "/tmp/alpha.pdf", project_id="alpha")
    beta = library.store(doc, "/tmp/beta.pdf", project_id="beta")

    removed = library.clear(project_id="alpha")

    assert removed == 1
    assert library.get_entry(alpha.doc_id) is None
    assert library.get_entry(beta.doc_id) is not None
    assert not Path(alpha.doc_path).exists()
    assert Path(beta.doc_path).exists()


def test_clear_all_projects_removes_everything(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    library = DoclingLibrary(path=tmp_path)
    first = library.store(doc, "/tmp/first.pdf", project_id="alpha")
    second = library.store(doc, "/tmp/second.pdf", project_id="beta")

    removed = library.clear(all_projects=True)

    assert removed == 2
    assert library.all_entries() == []
    assert not Path(first.doc_path).exists()
    assert not Path(second.doc_path).exists()


def test_postgres_filter_requires_database_url(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path)

    with pytest.raises(RuntimeError, match="PostgreSQL filtering requires"):
        library.query_entries_by_postgres_filter("project_id = 'alpha'")


def test_postgres_filter_rejects_multi_statement_tokens(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path)

    with pytest.raises(ValueError, match="single WHERE predicate"):
        library._validate_postgres_filter("project_id = 'alpha'; DROP TABLE docling_library_entries")

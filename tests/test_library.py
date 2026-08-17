import csv
from pathlib import Path

import pytest
from docling_core.types.doc.document import DocItemLabel, DoclingDocument, TableData

from docling_agent.agent.library import (
    DocCompileArtifact,
    DocCompileEntityRow,
    DocCompileRelationRow,
    DoclingLibrary,
)


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


def test_load_doc_can_skip_archive_extraction(tmp_path: Path) -> None:
    doc = DoclingDocument(name="sample")
    doc.add_text(label=DocItemLabel.TEXT, text="hello", parent=doc.body)

    library = DoclingLibrary(path=tmp_path)
    entry = library.store(doc, "/tmp/sample.pdf")
    artifacts_dir = Path(entry.doc_path).with_name(f"{Path(entry.doc_path).stem}_artifacts")

    loaded = library.load_doc(entry.doc_id, extract_archive=False)

    assert loaded is not None
    assert loaded.name == "sample"
    assert not artifacts_dir.exists()


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


def test_project_concepts_csv_scores_terms_by_documents_mentions_and_children(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path, project_id="alpha")
    model_doc1 = DocCompileEntityRow(
        doc_id="doc1",
        project_id="alpha",
        entity_hash=1,
        entity_text="model",
        normalized_text="model",
        kind="term",
        count=2,
    )
    hubbard_doc1 = DocCompileEntityRow(
        doc_id="doc1",
        project_id="alpha",
        entity_hash=2,
        entity_text="Hubbard model",
        normalized_text="Hubbard model",
        kind="term",
        count=5,
    )
    three_d_doc1 = DocCompileEntityRow(
        doc_id="doc1",
        project_id="alpha",
        entity_hash=3,
        entity_text="3D Hubbard model",
        normalized_text="3D Hubbard model",
        kind="term",
        count=1,
    )
    model_doc2 = DocCompileEntityRow(
        doc_id="doc2",
        project_id="alpha",
        entity_hash=4,
        entity_text="model",
        normalized_text="model",
        kind="term",
        count=3,
    )
    hubbard_doc2 = DocCompileEntityRow(
        doc_id="doc2",
        project_id="alpha",
        entity_hash=5,
        entity_text="Hubbard model",
        normalized_text="Hubbard model",
        kind="term",
        count=4,
    )
    two_d_doc2 = DocCompileEntityRow(
        doc_id="doc2",
        project_id="alpha",
        entity_hash=6,
        entity_text="2D Hubbard model",
        normalized_text="2D Hubbard model",
        kind="term",
        count=1,
    )
    path = tmp_path / "concepts.csv"

    library._write_project_concepts_csv(
        path,
        project_id="alpha",
        results=[
            (
                None,
                DocCompileArtifact(
                    entities=[model_doc1, hubbard_doc1, three_d_doc1],
                    relations=[
                        DocCompileRelationRow(
                            doc_id="doc1",
                            project_id="alpha",
                            entity_hash_i=model_doc1.entity_hash,
                            entity_hash_j=hubbard_doc1.entity_hash,
                            relation_k="sub-term",
                            count=1,
                        ),
                        DocCompileRelationRow(
                            doc_id="doc1",
                            project_id="alpha",
                            entity_hash_i=hubbard_doc1.entity_hash,
                            entity_hash_j=three_d_doc1.entity_hash,
                            relation_k="sub-term",
                            count=1,
                        ),
                    ],
                ),
                False,
            ),
            (
                None,
                DocCompileArtifact(
                    entities=[model_doc2, hubbard_doc2, two_d_doc2],
                    relations=[
                        DocCompileRelationRow(
                            doc_id="doc2",
                            project_id="alpha",
                            entity_hash_i=hubbard_doc2.entity_hash,
                            entity_hash_j=model_doc2.entity_hash,
                            relation_k="super-term",
                            count=1,
                        ),
                        DocCompileRelationRow(
                            doc_id="doc2",
                            project_id="alpha",
                            entity_hash_i=hubbard_doc2.entity_hash,
                            entity_hash_j=two_d_doc2.entity_hash,
                            relation_k="sub-term",
                            count=1,
                        )
                    ],
                ),
                False,
            ),
        ],
    )

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows[0] == {
        "project_id": "alpha",
        "concept": "Hubbard model",
        "document_count": "2",
        "total_mentions": "9",
        "child_count": "2",
        "score": "54",
    }
    assert rows[1]["concept"] == "model"
    assert rows[1]["child_count"] == "1"


def test_postgres_compile_entities_are_keyed_per_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[str] = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql: str, params=None):
            executed.append(sql)

    library = DoclingLibrary(path=tmp_path)
    library.database_url = "postgresql://example"
    monkeypatch.setattr(library, "_connect_pg", lambda: FakeConnection())

    library._ensure_pg_table()

    sql = "\n".join(executed)
    assert "PRIMARY KEY (doc_id, entity_hash)" in sql
    assert "DROP CONSTRAINT IF EXISTS docling_compile_entities_pkey" in sql

import csv
import json
from pathlib import Path

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import DocItemLabel, DoclingDocument

from docling_agent.agent.compiler import CompileContext, DoclingCompilerAgent, docling_document_to_glm_document
from docling_agent.agent.library import DocCompileEntityRow, DocCompileRun, DoclingLibrary
from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.task_model import CompileTask, load_task


class FakeNLPProvider:
    source_model = "fake-term"

    def apply_on_document(self, document: DoclingDocument):
        return {
            "terms": [
                {"text": "Docling", "label": "TERM", "confidence": 0.9},
                {"text": "IBM", "label": "ORG", "confidence": 0.8},
            ]
        }


class FakeTermTreeProvider:
    source_model = "fake-term-tree"

    def apply_on_document(self, document: DoclingDocument):
        return {
            "terms": [
                {"text": "model", "label": "TERM", "confidence": 0.9},
                {"text": "Hubbard model", "label": "TERM", "confidence": 0.9},
                {"text": "2D Hubbard model", "label": "TERM", "confidence": 0.9},
            ]
        }


class FakeGLMTableProvider:
    source_model = "fake-glm"

    def apply_on_document(self, document: DoclingDocument):
        return {
            "model-application": {"success": True},
            "texts": [
                {
                    "sref": "#/texts/0",
                    "prov": [{"$ref": "#/page-elements/0"}],
                    "text": "Docling is developed by IBM.",
                }
            ],
            "page-elements": [{"page": 1}],
            "instances": {
                "headers": [
                    "type",
                    "subtype",
                    "subj_hash",
                    "subj_name",
                    "subj_path",
                    "conf",
                    "hash",
                    "ihash",
                    "coor_i",
                    "coor_j",
                    "char_i",
                    "char_j",
                    "ctok_i",
                    "ctok_j",
                    "wtok_i",
                    "wtok_j",
                    "wtok-match",
                    "name",
                    "original",
                ],
                "data": [
                    [
                        "sentence",
                        "proper",
                        1,
                        "TEXT",
                        "#/texts/0",
                        1.0,
                        2,
                        3,
                        None,
                        None,
                        0,
                        27,
                        0,
                        27,
                        0,
                        5,
                        True,
                        "Docling is developed by IBM.",
                        "Docling is developed by IBM.",
                    ],
                    [
                        "term",
                        "single-term",
                        1,
                        "TEXT",
                        "#/texts/0",
                        0.9,
                        4,
                        5,
                        None,
                        None,
                        0,
                        7,
                        0,
                        7,
                        0,
                        1,
                        True,
                        "Docling",
                        "Docling",
                    ],
                ],
            },
        }


def _compile_doc() -> DoclingDocument:
    doc = DoclingDocument(name="compile-sample")
    doc.add_text(label=DocItemLabel.TEXT, text="Docling is developed by IBM.", parent=doc.body)
    return doc


def test_docling_document_to_glm_document_has_required_shape() -> None:
    glm_doc = docling_document_to_glm_document(_compile_doc())

    assert "document-hash" in glm_doc["file-info"]
    assert glm_doc["page-dimensions"] == [{"page": 1, "width": 1.0, "height": 1.0}]
    assert glm_doc["main-text"][0]["text"] == "Docling is developed by IBM."
    assert glm_doc["main-text"][0]["prov"][0]["page"] == 1
    assert glm_doc["main-text"][0]["docling_ref"] == "#/texts/0"


def test_compile_task_loads_from_yaml(tmp_path: Path) -> None:
    task_path = tmp_path / "compile.yaml"
    task_path.write_text(
        """
mode: compile
project_id: alpha
sources:
  - ./docs
glob: "*.pdf"
subtasks: [entities, topics]
nlp_provider: deepsearch-glm
nlp_models: "language;term"
force: true
llm_review_terms: true
llm_review_batch_size: 12
conversion: fast
""",
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, CompileTask)
    assert task.project_id == "alpha"
    assert task.subtasks == ["entities", "topics"]
    assert task.glob == "*.pdf"
    assert task.force is True
    assert task.llm_review_terms is True
    assert task.llm_review_batch_size == 12
    assert task.conversion == "fast"


def test_compile_task_loads_without_sources(tmp_path: Path) -> None:
    task_path = tmp_path / "compile-library.yaml"
    task_path.write_text(
        """
mode: compile
project_id: alpha
subtasks: [entities]
limit: 5
""",
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, CompileTask)
    assert task.sources == []
    assert task.project_id == "alpha"
    assert task.subtasks == ["entities"]
    assert task.limit == 5


def test_compiler_normalizes_entities_and_relations(mock_backend) -> None:
    agent = DoclingCompilerAgent(backend=mock_backend, tools=[], nlp_provider=FakeNLPProvider())
    artifact = agent.compile_document(
        document=_compile_doc(),
        context=CompileContext(doc_id="doc1", project_id="alpha"),
        subtasks=["entities", "topics"],
    )

    assert [row.entity_text for row in artifact.entities] == ["Docling", "IBM", "Docling"]
    assert {row.kind for row in artifact.entities} == {"document-term", "term", "entity"}
    assert artifact.topics == ["Docling", "IBM"]
    assert artifact.relations == []


def test_compiler_derives_canonical_terms_and_term_tree_relations(mock_backend) -> None:
    agent = DoclingCompilerAgent(backend=mock_backend, tools=[], nlp_provider=FakeTermTreeProvider())
    artifact = agent.compile_document(
        document=_compile_doc(),
        context=CompileContext(doc_id="doc1", project_id="alpha"),
        subtasks=["entities"],
    )

    terms = {row.normalized_text: row for row in artifact.entities if row.kind == "term"}
    document_terms = [row for row in artifact.entities if row.kind == "document-term"]

    assert set(terms) == {"model", "Hubbard model", "2D Hubbard model"}
    assert len(document_terms) == 3
    assert all(row.xpath is None for row in terms.values())
    assert all(row.count == 1 for row in terms.values())

    relation_pairs = {
        (terms_by_hash[relation.entity_hash_i].normalized_text, relation.relation_k, terms_by_hash[relation.entity_hash_j].normalized_text)
        for relation in artifact.relations
        if relation.relation_k in {"sub-term", "super-term"}
        for terms_by_hash in [{row.entity_hash: row for row in terms.values()}]
    }
    assert ("model", "sub-term", "Hubbard model") in relation_pairs
    assert ("Hubbard model", "super-term", "model") in relation_pairs
    assert ("Hubbard model", "sub-term", "2D Hubbard model") in relation_pairs
    assert ("2D Hubbard model", "super-term", "Hubbard model") in relation_pairs


class FakeNoisyTermsProvider:
    source_model = "fake-noisy"

    def apply_on_document(self, document: DoclingDocument):
        return {
            "terms": [
                {"text": "Tc", "label": "TERM", "confidence": 0.9},
                {"text": "critical temperature", "label": "TERM", "confidence": 0.9},
                {"text": "the", "label": "TERM", "confidence": 0.2},
                {"text": "IBM", "label": "ORG", "confidence": 0.8},
            ]
        }


class ReviewSession:
    def __init__(self, response: str, prompts: list[str]) -> None:
        self.response = response
        self.prompts = prompts

    def instruct(self, prompt: str, *, requirements=None, retry_budget: int = 1) -> str:
        self.prompts.append(prompt)
        return self.response


class ReviewBackend:
    backend_type = "mock-review"

    def __init__(self, response: str) -> None:
        from unittest.mock import MagicMock

        from docling_agent.task_model import ModelConfig

        self.config = MagicMock()
        self.config.models = ModelConfig(reasoning="mock-model", writing="mock-model", extraction="mock-extract")
        self.prompts: list[str] = []
        self.response = response

    @property
    def models(self):
        return self.config.models

    def create_session(self, *, model: str, system_prompt: str | None = None):
        return ReviewSession(self.response, self.prompts)


def test_compiler_llm_review_filters_canonicalizes_and_categorizes_terms() -> None:
    response = json.dumps(
        {
            "terms": [
                {
                    "id": "t0",
                    "decision": "keep",
                    "canonical": "critical temperature",
                    "category": "property",
                    "importance": "core",
                },
                {
                    "id": "t1",
                    "decision": "keep",
                    "canonical": "critical temperature",
                    "category": "property",
                    "importance": "core",
                },
                {
                    "id": "t2",
                    "decision": "drop",
                    "canonical": None,
                    "category": "generic",
                    "importance": "incidental",
                },
            ]
        }
    )
    backend = ReviewBackend(response)
    agent = DoclingCompilerAgent(backend=backend, tools=[], nlp_provider=FakeNoisyTermsProvider())
    artifact = agent.compile_document(
        document=_compile_doc(),
        context=CompileContext(doc_id="doc1", project_id="alpha"),
        subtasks=["entities"],
        llm_review_terms=True,
    )

    canonical_terms = [row for row in artifact.entities if row.kind == "term"]
    document_terms = [row for row in artifact.entities if row.kind == "document-term"]
    entity_rows = [row for row in artifact.entities if row.kind == "entity"]

    assert [row.normalized_text for row in document_terms] == ["critical temperature", "critical temperature"]
    assert {row.label for row in document_terms} == {"property"}
    assert len(canonical_terms) == 1
    assert canonical_terms[0].normalized_text == "critical temperature"
    assert canonical_terms[0].count == 2
    assert entity_rows[0].entity_text == "IBM"
    assert backend.prompts


class ManyTermsProvider:
    source_model = "fake-many"

    def apply_on_document(self, document: DoclingDocument):
        return {"terms": [{"text": f"term {index}", "label": "TERM"} for index in range(5)]}


def test_compiler_llm_review_batches_terms() -> None:
    response = json.dumps(
        {
            "terms": [
                {"id": "t0", "decision": "keep", "canonical": "kept", "category": "concept"},
                {"id": "t1", "decision": "keep", "canonical": "kept", "category": "concept"},
            ]
        }
    )
    backend = ReviewBackend(response)
    agent = DoclingCompilerAgent(backend=backend, tools=[], nlp_provider=ManyTermsProvider())

    agent.compile_document(
        document=_compile_doc(),
        context=CompileContext(doc_id="doc1", project_id="alpha"),
        subtasks=["entities"],
        llm_review_terms=True,
        llm_review_batch_size=2,
    )

    assert len(backend.prompts) == 3


def test_compiler_concepts_use_canonical_term_counts(mock_backend) -> None:
    agent = DoclingCompilerAgent(backend=mock_backend, tools=[], nlp_provider=FakeNLPProvider())
    concepts = agent._concepts_from_entities(
        [
            DocCompileEntityRow(
                doc_id="doc1",
                project_id="alpha",
                xpath="/texts/0",
                entity_hash=1,
                entity_text="Noisy mention",
                normalized_text="Noisy mention",
                kind="document-term",
            ),
            DocCompileEntityRow(
                doc_id="doc1",
                project_id="alpha",
                entity_hash=2,
                entity_text="Hubbard model",
                normalized_text="Hubbard model",
                kind="term",
                count=4,
            ),
            DocCompileEntityRow(
                doc_id="doc1",
                project_id="alpha",
                entity_hash=3,
                entity_text="ground state",
                normalized_text="ground state",
                kind="term",
                count=2,
            ),
            DocCompileEntityRow(
                doc_id="doc1",
                project_id="alpha",
                entity_hash=4,
                entity_text="Figure",
                normalized_text="Figure",
                kind="term",
                count=1,
            ),
        ]
    )

    assert concepts == ["Hubbard model", "ground state", "Figure"]


def test_compiler_parses_deepsearch_glm_instances_table(mock_backend) -> None:
    agent = DoclingCompilerAgent(backend=mock_backend, tools=[], nlp_provider=FakeGLMTableProvider())
    artifact = agent.compile_document(
        document=_compile_doc(),
        context=CompileContext(doc_id="doc1", project_id="alpha"),
        subtasks=["entities"],
    )

    assert len(artifact.entities) == 2
    row = next(row for row in artifact.entities if row.kind == "document-term")
    assert row.entity_text == "Docling"
    assert row.normalized_text == "Docling"
    assert row.label == "single-term"
    assert row.kind == "document-term"
    assert row.xpath == "/texts/0"
    assert row.char_start == 0
    assert row.char_end == 7
    assert row.confidence == 0.9
    assert row.page_no == 1
    term = next(row for row in artifact.entities if row.kind == "term")
    assert term.normalized_text == "Docling"
    assert term.xpath is None
    assert term.count == 1


def test_library_stores_compile_csv_files(tmp_path: Path) -> None:
    library = DoclingLibrary(path=tmp_path, project_id="alpha")
    doc = _compile_doc()
    entry = library.store(doc, "/tmp/compile-sample.pdf", project_id="alpha")
    agent = DoclingCompilerAgent(backend=None, tools=[], nlp_provider=FakeNLPProvider())
    artifact = agent.compile_document(
        document=doc,
        context=CompileContext(doc_id=entry.doc_id, project_id="alpha"),
        subtasks=["entities", "topics"],
    )

    library.store_compile_result(
        entry.doc_id,
        artifact=artifact,
        run=DocCompileRun(name="compile", provider="deepsearch-glm", model_names="language;term"),
    )
    updated = library.get_entry(entry.doc_id)
    assert updated is not None

    entities_path = Path(updated.compile.entities_path or "")
    relations_path = Path(updated.compile.relations_path or "")
    compile_path = Path(updated.compile.compile_path or "")

    assert entities_path.exists()
    assert relations_path.exists()
    assert compile_path.exists()
    with open(entities_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == [
            "doc_id",
            "project_id",
            "xpath",
            "entity_hash",
            "entity_text",
            "normalized_text",
            "count",
            "label",
            "kind",
            "source_model",
            "confidence",
            "page_no",
            "char_start",
            "char_end",
            "created_at",
        ]

    with open(relations_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == [
            "doc_id",
            "project_id",
            "entity_hash_i",
            "entity_hash_j",
            "relation_k",
            "count",
            "created_at",
        ]


def test_orchestrator_compile_dispatch_persists_artifacts(
    tmp_path: Path,
    monkeypatch,
    mock_backend,
) -> None:
    doc = _compile_doc()
    source_path = tmp_path / "document.json"
    source_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    library_path = tmp_path / "library"

    def _fake_init(self, *, tools, backend=None, nlp_provider=None, nlp_model_names="language;term"):
        self.backend = backend
        self.tools = tools
        self.nlp_provider = FakeNLPProvider()
        self.nlp_model_names = nlp_model_names

    monkeypatch.setattr(DoclingCompilerAgent, "__init__", _fake_init)

    orchestrator = DoclingOrchestratorAgent(backend=mock_backend, tools=[], library_path=library_path)
    result = orchestrator.run_task(
        CompileTask(
            project_id="alpha",
            sources=[str(source_path)],
            subtasks=["entities", "topics"],
        )
    )
    markdown = MarkdownDocSerializer(doc=result).serialize().text
    library = DoclingLibrary(path=library_path, project_id="alpha")
    entries = library.all_entries()

    assert len(entries) == 1
    assert Path(entries[0].compile.entities_path or "").exists()
    assert Path(entries[0].compile.relations_path or "").exists()
    project_path = library_path.parent / "projects" / "alpha"
    assert (project_path / "entities.csv").exists()
    assert (project_path / "relations.csv").exists()
    assert (project_path / "terms.csv").exists()
    assert (project_path / "concepts.csv").exists()
    assert (project_path / "summaries.md").exists()
    assert (project_path / "wiki" / "summaries").is_dir()
    assert (project_path / "wiki" / "concepts").is_dir()
    assert (project_path / "wiki" / "entities").is_dir()
    assert (project_path / "wiki" / "queries").is_dir()
    manifest = json.loads((project_path / "compile.json").read_text(encoding="utf-8"))
    assert manifest["project_id"] == "alpha"
    assert manifest["documents"][0]["doc_id"] == entries[0].doc_id
    assert manifest["artifacts"]["terms_path"] == str(project_path / "terms.csv")
    assert "entities: 3" in markdown
    assert "relations: 0" in markdown
    assert "#### Concepts" not in markdown


def test_orchestrator_compile_selects_project_entries(
    tmp_path: Path,
    monkeypatch,
    mock_backend,
) -> None:
    library_path = tmp_path / "library"
    library = DoclingLibrary(path=library_path, project_id="alpha")
    kept = library.store(_compile_doc(), "/tmp/alpha.pdf", project_id="alpha")
    library.store(_compile_doc(), "/tmp/beta.pdf", project_id="beta")

    def _fake_init(self, *, tools, backend=None, nlp_provider=None, nlp_model_names="language;term"):
        self.backend = backend
        self.tools = tools
        self.nlp_provider = FakeNLPProvider()
        self.nlp_model_names = nlp_model_names

    monkeypatch.setattr(DoclingCompilerAgent, "__init__", _fake_init)

    orchestrator = DoclingOrchestratorAgent(backend=mock_backend, tools=[], library_path=library_path)
    result = orchestrator.run_task(CompileTask(project_id="alpha", sources=[], subtasks=["entities"]))
    markdown = MarkdownDocSerializer(doc=result).serialize().text
    updated = DoclingLibrary(path=library_path, project_id="alpha").get_entry(kept.doc_id)

    assert updated is not None
    assert Path(updated.compile.entities_path or "").exists()
    assert "Compiled 1 document" in markdown


def test_orchestrator_compile_uses_postgres_filter_selection(
    tmp_path: Path,
    monkeypatch,
    mock_backend,
) -> None:
    library_path = tmp_path / "library"
    library = DoclingLibrary(path=library_path, project_id="alpha")
    entry = library.store(_compile_doc(), "/tmp/alpha.pdf", project_id="alpha")
    calls = []

    def _fake_query(self: DoclingLibrary, postgres_filter: str, *, limit: int = 100):
        calls.append((postgres_filter, limit))
        return [entry]

    def _fake_init(self, *, tools, backend=None, nlp_provider=None, nlp_model_names="language;term"):
        self.backend = backend
        self.tools = tools
        self.nlp_provider = FakeNLPProvider()
        self.nlp_model_names = nlp_model_names

    monkeypatch.setattr(DoclingLibrary, "query_entries_by_postgres_filter", _fake_query)
    monkeypatch.setattr(DoclingCompilerAgent, "__init__", _fake_init)

    orchestrator = DoclingOrchestratorAgent(backend=mock_backend, tools=[], library_path=library_path)
    orchestrator.run_task(
        CompileTask(
            project_id="default",
            sources=[],
            subtasks=["entities"],
            postgres_filter="project_id = 'alpha'",
            limit=7,
        )
    )

    assert calls == [("project_id = 'alpha'", 7)]


def test_orchestrator_compile_uses_task_conversion_preset(mock_backend) -> None:
    orchestrator = DoclingOrchestratorAgent(backend=mock_backend, tools=[])

    task = CompileTask(project_id="alpha", sources=["/tmp/docs"], conversion="fast")

    assert orchestrator._source_conversion_preset(task) == "fast"

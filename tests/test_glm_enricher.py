from docling_core.types.doc.document import DocItemLabel, DoclingDocument

from docling_agent.agent.glm_enricher import DoclingGLMEnricherAgent


class FakeGLMAdapter:
    """Simple adapter stub for GLM enricher tests."""

    def __init__(self, responses: dict[str, dict]):
        self._responses = responses
        self.calls: list[str] = []

    def apply_on_text(self, text: str) -> dict:
        self.calls.append(text)
        return self._responses.get(text, {})


def _make_instances(*rows: tuple[str, str, int, int, str]) -> dict:
    return {
        "headers": ["type", "subtype", "char_i", "char_j", "original"],
        "data": [list(row) for row in rows],
    }


def test_detect_key_entities_enriches_leaf_text_items() -> None:
    document = DoclingDocument(name="source")
    paragraph = document.add_text(label=DocItemLabel.PARAGRAPH, text="Alice founded IBM", parent=document.body)
    document.add_text(label=DocItemLabel.CAPTION, text="caption", parent=document.body)

    adapter = FakeGLMAdapter(
        {
            "Alice founded IBM": {
                "entities": {
                    "headers": ["type", "char_i", "char_j", "original"],
                    "data": [
                        ["person", 0, 5, "Alice"],
                        ["organization", 14, 17, "IBM"],
                    ],
                }
            }
        }
    )
    agent = DoclingGLMEnricherAgent(tools=[], adapter=adapter)

    result = agent.run(task="extract entities", document=document, operations=["detect_key_entities"])

    assert result is document
    assert adapter.calls == ["Alice founded IBM"]
    assert paragraph.meta is not None
    assert paragraph.meta.entities is not None
    assert [(m.text, m.label, m.charspan) for m in paragraph.meta.entities.mentions] == [
        ("Alice", "person", (0, 5)),
        ("IBM", "organization", (14, 17)),
    ]


def test_find_search_keywords_uses_term_instances() -> None:
    document = DoclingDocument(name="source")
    paragraph = document.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Quantum computing uses superconducting qubits",
        parent=document.body,
    )

    adapter = FakeGLMAdapter(
        {
            "Quantum computing uses superconducting qubits": {
                "instances": _make_instances(
                    ("term", "single-term", 0, 17, "Quantum computing"),
                    ("term", "single-term", 23, 46, "superconducting qubits"),
                    ("term", "single-term", 23, 46, "superconducting qubits"),
                )
            }
        }
    )
    agent = DoclingGLMEnricherAgent(tools=[], adapter=adapter)

    agent.run(task="extract keywords", document=document, operations=["find_search_keywords"])

    assert paragraph.meta is not None
    assert paragraph.meta.keywords is not None
    assert paragraph.meta.keywords.values == ["Quantum computing", "superconducting qubits"]


def test_instances_fallback_populates_entities_when_entities_table_missing() -> None:
    document = DoclingDocument(name="source")
    paragraph = document.add_text(label=DocItemLabel.PARAGRAPH, text="Paris is in France", parent=document.body)

    adapter = FakeGLMAdapter(
        {
            "Paris is in France": {
                "instances": _make_instances(
                    ("sentence", "", 0, 18, "Paris is in France"),
                    ("geoloc", "city", 0, 5, "Paris"),
                    ("geoloc", "country", 12, 18, "France"),
                )
            }
        }
    )
    agent = DoclingGLMEnricherAgent(tools=[], adapter=adapter)

    agent.run(task="extract entities", document=document, operations=["entities"])

    assert paragraph.meta is not None
    assert paragraph.meta.entities is not None
    assert [(m.text, m.label) for m in paragraph.meta.entities.mentions] == [
        ("Paris", "geoloc"),
        ("France", "geoloc"),
    ]


def test_unsupported_operation_raises_value_error() -> None:
    document = DoclingDocument(name="source")
    adapter = FakeGLMAdapter({})
    agent = DoclingGLMEnricherAgent(tools=[], adapter=adapter)

    try:
        agent.run(task="summarize", document=document, operations=["summarize_items"])
    except ValueError as exc:
        assert "Unsupported GLM enrichment operation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")

"""Tests for DoclingOrchestratorAgent.

This module includes regression tests for the entity constraint bug fix,
specifically testing task propagation from orchestrator to enricher.
"""

import tempfile
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument

from docling_agent.agent.enricher import DoclingEnrichingAgent
from docling_agent.agent.library import DoclingLibrary
from docling_agent.agent.orchestrator import DoclingOrchestratorAgent


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

    def _fake_enricher_run(self, task, document, operations):
        captured_task.append(task)
        return document

    # Mock the enricher's run method
    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    # Create a mock library with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        library.store(test_document, "test.json")

        # Call _ensure_enriched with a task
        source_pairs = [(test_document, "test-doc-id")]
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

    def _fake_enricher_run(self, task, document, operations):
        captured_task.append(task)
        return document

    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        library.store(test_document, "test.json")

        source_pairs = [(test_document, "test-doc-id")]

        # Call without task parameter (should default to "")
        orchestrator._ensure_enriched(source_pairs=source_pairs, library=library, operations=["entities"])

        assert len(captured_task) == 1
        assert captured_task[0] == ""


def test_ensure_enriched_multiple_operations(
    monkeypatch: pytest.MonkeyPatch, orchestrator: DoclingOrchestratorAgent, test_document: DoclingDocument
) -> None:
    """Test that _ensure_enriched handles multiple operations correctly."""
    enricher_calls = []

    def _fake_enricher_run(self, task, document, operations):
        enricher_calls.append((task, operations))
        return document

    monkeypatch.setattr(DoclingEnrichingAgent, "run", _fake_enricher_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        library = DoclingLibrary(path=Path(tmpdir))
        library.store(test_document, "test.json")

        source_pairs = [(test_document, "test-doc-id")]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

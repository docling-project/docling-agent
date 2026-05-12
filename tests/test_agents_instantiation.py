import pytest

from docling_agent.backends import create_backend
from docling_agent.task_model import BackendConfig, ModelConfig


def _backend():
    return create_backend(
        BackendConfig(
            type="mellea",
            models=ModelConfig(
                reasoning="OPENAI_GPT_OSS_20B",
                writing="OPENAI_GPT_OSS_20B",
            ),
        )
    )


def test_instantiate_docling_writing_agent():
    from docling_agent.agent.writer import DoclingWritingAgent

    try:
        _ = DoclingWritingAgent(backend=_backend(), tools=[])
    except Exception as e:
        pytest.fail(f"DoclingWritingAgent instantiation raised: {e}")


def test_instantiate_docling_editing_agent():
    from docling_agent.agent.editor import DoclingEditingAgent

    try:
        _ = DoclingEditingAgent(backend=_backend(), tools=[])
    except Exception as e:
        pytest.fail(f"DoclingEditingAgent instantiation raised: {e}")


def test_instantiate_docling_extracting_agent():
    from docling_agent.agent.extractor import DoclingExtractingAgent

    try:
        _ = DoclingExtractingAgent(backend=_backend(), tools=[])
    except Exception as e:
        pytest.fail(f"DoclingExtractingAgent instantiation raised: {e}")


def test_instantiate_docling_enriching_agent():
    from docling_agent.agent.enricher import DoclingEnrichingAgent

    try:
        _ = DoclingEnrichingAgent(backend=_backend(), tools=[])
    except Exception as e:
        pytest.fail(f"DoclingEnrichingAgent instantiation raised: {e}")

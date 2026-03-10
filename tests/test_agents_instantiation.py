import pytest


def test_instantiate_docling_writing_agent():
    from mellea.backends import model_ids

    from docling_agent.agent.writer import DoclingWritingAgent

    try:
        _ = DoclingWritingAgent(model_id=model_ids.OPENAI_GPT_OSS_20B, tools=[])
    except Exception as e:
        pytest.fail(f"DoclingWritingAgent instantiation raised: {e}")


def test_instantiate_docling_editing_agent():
    from mellea.backends import model_ids

    from docling_agent.agent.editor import DoclingEditingAgent

    try:
        _ = DoclingEditingAgent(model_id=model_ids.OPENAI_GPT_OSS_20B, tools=[])
    except Exception as e:
        pytest.fail(f"DoclingEditingAgent instantiation raised: {e}")


def test_instantiate_docling_extracting_agent():
    from mellea.backends import model_ids

    from docling_agent.agent.extractor import DoclingExtractingAgent

    try:
        _ = DoclingExtractingAgent(model_id=model_ids.OPENAI_GPT_OSS_20B, tools=[])
    except Exception as e:
        pytest.fail(f"DoclingExtractingAgent instantiation raised: {e}")

from docling_agent.backends import (
    LiteLLMBackend,
    LlamaServerBackend,
    LMStudioBackend,
    MelleaBackend,
    OllamaBackend,
    create_backend,
)
from docling_agent.task_model import BackendConfig


def test_create_mellea_backend_from_config():
    backend = create_backend(BackendConfig(type="mellea"))
    assert isinstance(backend, MelleaBackend)
    assert backend.backend_type == "mellea"


def test_create_ollama_backend_from_config():
    backend = create_backend(BackendConfig(type="ollama", base_url="http://localhost:11434"))
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://localhost:11434"


def test_create_lmstudio_backend_from_config():
    backend = create_backend(BackendConfig(type="lmstudio", base_url="http://localhost:1234/v1"))
    assert isinstance(backend, LMStudioBackend)
    assert backend.base_url == "http://localhost:1234/v1"


def test_create_litellm_backend_from_config():
    backend = create_backend(BackendConfig(type="litellm", api_key_env="LITELLM_API_KEY"))
    assert isinstance(backend, LiteLLMBackend)
    assert backend.api_key_env == "LITELLM_API_KEY"


def test_create_llama_server_backend_from_config():
    backend = create_backend(BackendConfig(type="llama-server"))
    assert isinstance(backend, LlamaServerBackend)
    assert backend.backend_type == "llama-server"
    assert backend.base_url == "http://localhost:8080/v1"


def test_create_llama_server_backend_with_custom_base_url():
    backend = create_backend(BackendConfig(type="llama-server", base_url="http://example.com:9000/v1"))
    assert isinstance(backend, LlamaServerBackend)
    assert backend.base_url == "http://example.com:9000/v1"

from __future__ import annotations

from typing_extensions import Self

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend
from docling_agent.task_model import BackendConfig


class LlamaServerBackend(OpenAICompatibleBackend):
    """Backend for llama.cpp's llama-server OpenAI-compatible API.

    llama-server ships with llama.cpp and exposes an OpenAI-compatible HTTP API
    for running GGUF models locally. This backend connects to llama-server's
    default endpoint.

    Default connection: http://localhost:8080/v1
    """

    backend_type = "llama-server"

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct a llama-server backend from configuration.

        Sets the default base URL to llama-server's standard endpoint if not specified.

        Args:
            config: Backend configuration.

        Returns:
            Initialized llama-server backend instance.
        """
        config = config.model_copy(
            update={
                "base_url": config.base_url or "http://localhost:8080/v1",
            }
        )
        return cls(config=config)

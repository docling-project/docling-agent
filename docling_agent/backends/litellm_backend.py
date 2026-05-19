from __future__ import annotations

from typing_extensions import Self

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend
from docling_agent.task_model import BackendConfig


class LiteLLMBackend(OpenAICompatibleBackend):
    """Backend for LiteLLM proxy server.

    LiteLLM provides a unified interface to 100+ LLM providers through
    an OpenAI-compatible API.

    Default connection: http://localhost:4000/v1
    """

    backend_type = "litellm"

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct a LiteLLM backend from configuration.

        Sets the default base URL to LiteLLM's standard endpoint if not specified.

        Args:
            config: Backend configuration.

        Returns:
            Initialized LiteLLM backend instance.
        """
        config = config.model_copy(
            update={
                "base_url": config.base_url or "http://localhost:4000/v1",
            }
        )
        return cls(config=config)

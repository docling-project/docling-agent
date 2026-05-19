from __future__ import annotations

from typing_extensions import Self

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend
from docling_agent.task_model import BackendConfig


class LMStudioBackend(OpenAICompatibleBackend):
    """Backend for LM Studio's local OpenAI-compatible API.

    LM Studio provides a local server for running LLMs with an OpenAI-compatible
    API. This backend connects to LM Studio's default endpoint.

    Default connection: http://localhost:1234/v1
    """

    backend_type = "lmstudio"

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct an LM Studio backend from configuration.

        Sets the default base URL to LM Studio's standard endpoint if not specified.

        Args:
            config: Backend configuration.

        Returns:
            Initialized LM Studio backend instance.
        """
        config = config.model_copy(
            update={
                "base_url": config.base_url or "http://localhost:1234/v1",
            }
        )
        return cls(config=config)

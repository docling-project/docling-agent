from __future__ import annotations

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend
from docling_agent.task_model import BackendConfig


class LiteLLMBackend(OpenAICompatibleBackend):
    """Direct LiteLLM backend."""

    backend_type = "litellm"

    @classmethod
    def from_config(cls, config: BackendConfig) -> LiteLLMBackend:
        config = config.model_copy(
            update={
                "base_url": config.base_url or "http://localhost:4000/v1",
            }
        )
        return cls(config=config)

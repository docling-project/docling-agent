from __future__ import annotations

from typing_extensions import Self

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend
from docling_agent.task_model import BackendConfig


class LMStudioBackend(OpenAICompatibleBackend):
    """Direct LM Studio backend."""

    backend_type = "lmstudio"

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        config = config.model_copy(
            update={
                "base_url": config.base_url or "http://localhost:1234/v1",
            }
        )
        return cls(config=config)

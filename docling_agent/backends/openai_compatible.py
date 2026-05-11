from __future__ import annotations

from docling_agent.backends.base import BaseBackend
from docling_agent.task_model import BackendConfig


class OpenAICompatibleBackend(BaseBackend):
    """Shared config holder for OpenAI-compatible direct backends."""

    backend_type = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int | None = None,
        api_key_env: str | None = None,
        options: dict | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.options = options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> OpenAICompatibleBackend:
        return cls(
            base_url=config.base_url,
            timeout=config.timeout,
            api_key_env=config.api_key_env,
            options=config.options,
        )

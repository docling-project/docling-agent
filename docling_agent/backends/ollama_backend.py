from __future__ import annotations

from mellea.stdlib.requirements import Requirement

from docling_agent.backends.base import BaseBackend
from docling_agent.task_model import BackendConfig


class OllamaBackend(BaseBackend):
    """Direct Ollama backend placeholder for the Phase 1 abstraction."""

    backend_type = "ollama"

    def __init__(self, *, base_url: str | None = None, timeout: int | None = None, options: dict | None = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.options = options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> OllamaBackend:
        return cls(base_url=config.base_url, timeout=config.timeout, options=config.options)

    def instruct(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        raise NotImplementedError("OllamaBackend direct execution is not implemented in Phase 1.")

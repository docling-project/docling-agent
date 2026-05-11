from __future__ import annotations

from mellea.stdlib.requirements import Requirement

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend


class LiteLLMBackend(OpenAICompatibleBackend):
    """Direct LiteLLM backend placeholder for the Phase 1 abstraction."""

    backend_type = "litellm"

    def instruct(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        raise NotImplementedError("LiteLLMBackend direct execution is not implemented in Phase 1.")

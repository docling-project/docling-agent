from __future__ import annotations

from mellea.stdlib.requirements import Requirement

from docling_agent.backends.openai_compatible import OpenAICompatibleBackend


class LMStudioBackend(OpenAICompatibleBackend):
    """Direct LM Studio backend placeholder for the Phase 1 abstraction."""

    backend_type = "lmstudio"

    def instruct(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        raise NotImplementedError("LMStudioBackend direct execution is not implemented in Phase 1.")

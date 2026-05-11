from __future__ import annotations

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling_agent.backends.base import BaseBackend
from docling_agent.task_model import BackendConfig


class MelleaBackend(BaseBackend):
    """Compatibility backend that preserves current Mellea-based execution."""

    backend_type = "mellea"

    def __init__(self, *, base_url: str | None = None, timeout: int | None = None, options: dict | None = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.options = options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> MelleaBackend:
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
        model_id = getattr(model_ids, model, model)

        ctx = ChatContext()
        if system_prompt:
            ctx = ctx.add(Message(role="system", content=system_prompt))

        session = MelleaSession(
            ctx=ctx,
            backend=OllamaModelBackend(model_id=model_id),
        )
        result = session.instruct(
            prompt,
            requirements=requirements or [],
            strategy=RejectionSamplingStrategy(loop_budget=retry_budget),
        )
        return result.value

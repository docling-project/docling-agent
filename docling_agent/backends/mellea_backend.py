from __future__ import annotations

from typing import cast

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling_agent.agent_models import should_log_llm_io
from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.logging import logger
from docling_agent.task_model import BackendConfig


class MelleaSessionAdapter(BaseSession):
    """Stateful Mellea session wrapper used by ``MelleaBackend``."""

    def __init__(self, session: MelleaSession) -> None:
        self._session = session

    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        if should_log_llm_io():
            logger.debug(f"[LLM REQUEST]\n{prompt}")
        result = self._session.instruct(
            prompt,
            requirements=cast(list[Requirement | str], requirements or []),
            strategy=RejectionSamplingStrategy(loop_budget=retry_budget),
        )
        value = result.value
        if value is None:
            raise ValueError("Mellea returned no response text.")
        if should_log_llm_io():
            logger.debug(f"[LLM RESPONSE]\n{value}")
        return value

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        rows: list[tuple[int, str, str]] = []
        context_components = self._session.ctx.view_for_generation()
        if not context_components:
            return rows

        for idx, comp in enumerate(context_components):
            if isinstance(comp, Message):
                content = comp.content
                if len(content) > 64:
                    content = f"{content[0:32]} ... {content[-32:]}"
                rows.append((idx, comp.role, content))
            else:
                rows.append((idx, "<unknown>", str(comp)[0:64]))
        return rows


class MelleaBackend(BaseBackend):
    """Compatibility backend that preserves current Mellea-based execution."""

    backend_type = "mellea"

    def __init__(self, *, config: BackendConfig) -> None:
        self.config = config
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.options = config.options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> MelleaBackend:
        return cls(config=config)

    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        model_id = getattr(model_ids, model, model)

        ctx = ChatContext()
        if system_prompt:
            ctx = ctx.add(Message(role="system", content=system_prompt))

        session = MelleaSession(
            ctx=ctx,
            backend=OllamaModelBackend(model_id=model_id),
        )
        return MelleaSessionAdapter(session)

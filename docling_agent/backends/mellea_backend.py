from __future__ import annotations

import logging
from typing import cast

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy
from typing_extensions import Self

from docling_agent.agent_models import should_log_llm_io
from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.logging import log_llm_request, log_llm_response
from docling_agent.task_model import BackendConfig

# Suppress Mellea's template warnings about unused Message attributes
logging.getLogger("mellea").setLevel(logging.ERROR)


class MelleaSessionAdapter(BaseSession):
    """Adapter wrapping a Mellea session to conform to BaseSession interface.

    Provides compatibility between Mellea's native session API and the
    standardized backend session interface.
    """

    def __init__(self, session: MelleaSession) -> None:
        """Initialize the adapter with a Mellea session.

        Args:
            session: The underlying Mellea session to wrap.
        """
        self._session = session

    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        """Execute an instruction using Mellea's rejection sampling strategy.

        Args:
            prompt: The instruction or query to send to the LLM.
            requirements: Optional structured output requirements for validation.
            retry_budget: Maximum number of retry attempts on validation failure.

        Returns:
            The generated text response from the LLM.

        Raises:
            ValueError: If Mellea returns no response text.
        """
        if should_log_llm_io():
            log_llm_request(
                prompt,
                model=self._session.backend.model_id if hasattr(self._session.backend, "model_id") else None,
                retry_budget=retry_budget,
                has_requirements=requirements is not None,
            )

        result = self._session.instruct(
            prompt,
            requirements=cast(list[Requirement | str], requirements or []),
            strategy=RejectionSamplingStrategy(loop_budget=retry_budget),
        )

        value = result.value
        if value is None:
            raise ValueError("Mellea returned no response text.")

        if should_log_llm_io():
            log_llm_response(
                value,
                model=self._session.backend.model_id if hasattr(self._session.backend, "model_id") else None,
            )

        return value

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        """Extract conversation context from Mellea session for debugging.

        Returns:
            List of tuples (index, role, content_preview) representing the
            conversation history, with content truncated for readability.
        """
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
    """Backend implementation using the Mellea framework.

    Mellea provides structured output validation, rejection sampling,
    and advanced prompting capabilities. This backend uses Ollama as
    the underlying model provider.
    """

    backend_type = "mellea"

    def __init__(self, *, config: BackendConfig) -> None:
        """Initialize the Mellea backend with configuration.

        Args:
            config: Backend configuration including connection settings.
        """
        self.config = config
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.options = config.options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct a Mellea backend from configuration.

        Args:
            config: Backend configuration.

        Returns:
            Initialized Mellea backend instance.
        """
        return cls(config=config)

    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        """Create a new Mellea session with Ollama backend.

        Args:
            model: Model identifier (resolved from model_ids or used directly).
            system_prompt: Optional system-level instructions.

        Returns:
            A new session wrapped in MelleaSessionAdapter.
        """
        model_id = getattr(model_ids, model, model)

        ctx = ChatContext()
        if system_prompt:
            ctx = ctx.add(Message(role="system", content=system_prompt))

        session = MelleaSession(
            ctx=ctx,
            backend=OllamaModelBackend(model_id=model_id),
        )
        return MelleaSessionAdapter(session)

from __future__ import annotations

from abc import ABC, abstractmethod

from mellea.stdlib.requirements import Requirement
from typing_extensions import Self

from docling_agent.task_model import BackendConfig, ModelConfig


class BaseSession(ABC):
    """Shared contract for stateful backend sessions."""

    @abstractmethod
    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        """Run one instruction call and return the generated text."""
        raise NotImplementedError

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        """Return rows suitable for chat-context logging when supported."""
        return None


class BaseBackend(ABC):
    """Shared contract for all runtime backends."""

    backend_type: str
    config: BackendConfig

    @classmethod
    @abstractmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Build a backend instance from task configuration."""
        raise NotImplementedError

    @property
    def models(self) -> ModelConfig:
        """Return the backend-scoped role model configuration."""
        return self.config.models

    @abstractmethod
    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        """Create a stateful backend session."""
        raise NotImplementedError

    def instruct(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        """Run one instruction call and return the generated text."""
        session = self.create_session(model=model, system_prompt=system_prompt)
        return session.instruct(
            prompt,
            requirements=requirements,
            retry_budget=retry_budget,
        )

from __future__ import annotations

from abc import ABC, abstractmethod

from mellea.stdlib.requirements import Requirement


class BaseBackend(ABC):
    """Shared contract for all runtime backends."""

    backend_type: str

    @classmethod
    @abstractmethod
    def from_config(cls, config) -> BaseBackend:
        """Build a backend instance from task configuration."""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

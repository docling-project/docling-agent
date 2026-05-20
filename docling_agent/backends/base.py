from __future__ import annotations

from abc import ABC, abstractmethod

from mellea.stdlib.requirements import Requirement
from typing_extensions import Self

from docling_agent.task_model import BackendConfig, ModelConfig


class BaseSession(ABC):
    """Abstract base class for stateful backend sessions.

    A session maintains conversation context and handles LLM interactions
    with retry logic and optional structured output requirements.
    """

    @abstractmethod
    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        """Execute an instruction and return the generated text.

        Args:
            prompt: The instruction or query to send to the LLM.
            requirements: Optional structured output requirements for validation.
            retry_budget: Maximum number of retry attempts on validation failure.

        Returns:
            The generated text response from the LLM.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        """Provide chat context for debugging and logging.

        Returns:
            List of tuples (index, role, content) representing the conversation,
            or None if context tracking is not supported by this backend.
        """
        return None


class BaseBackend(ABC):
    """Abstract base class for LLM backend implementations.

    Defines the interface for connecting to different LLM providers
    (Mellea, Ollama, LM Studio, LiteLLM, etc.) and managing sessions.

    Attributes:
        backend_type: Identifier for the backend type (e.g., "mellea", "ollama").
        config: Configuration settings for this backend instance.
    """

    backend_type: str
    config: BackendConfig

    @classmethod
    @abstractmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct a backend instance from configuration.

        Args:
            config: Backend configuration including type, URL, credentials, etc.

        Returns:
            Initialized backend instance.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def models(self) -> ModelConfig:
        """Access the model configuration for different agent roles.

        Returns:
            Model identifiers for reasoning, writing, and extraction tasks.
        """
        return self.config.models

    @abstractmethod
    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        """Create a new stateful session for multi-turn interactions.

        Args:
            model: Model identifier to use for this session.
            system_prompt: Optional system-level instructions for the LLM.

        Returns:
            A new session instance maintaining conversation context.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
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
        """Execute a single instruction without maintaining session state.

        Convenience method that creates a session, runs one instruction,
        and discards the session. For multi-turn conversations, use
        create_session() instead.

        Args:
            prompt: The instruction or query to send to the LLM.
            model: Model identifier to use.
            system_prompt: Optional system-level instructions.
            requirements: Optional structured output requirements.
            retry_budget: Maximum retry attempts on validation failure.

        Returns:
            The generated text response from the LLM.
        """
        session = self.create_session(model=model, system_prompt=system_prompt)
        return session.instruct(
            prompt,
            requirements=requirements,
            retry_budget=retry_budget,
        )

"""Test utilities for docling-agent tests.

This module provides mock implementations and helper functions
that can be reused across different test modules.
"""

from unittest.mock import MagicMock

from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.task_model import ModelConfig


class MockSession(BaseSession):
    """Mock session for testing that doesn't require external services."""

    def instruct(self, prompt: str, *, requirements=None, retry_budget: int = 1) -> str:
        """Return a mock response without calling any LLM."""
        return "mock response"


class MockBackend(BaseBackend):
    """Mock backend for testing that doesn't require external services.

    This backend can be used in tests to avoid dependencies on external
    services like Ollama, Mellea, LM Studio, etc. It's particularly useful
    for CI/CD environments where these services may not be available.

    Example:
        >>> backend = MockBackend()
        >>> enricher = DoclingEnrichingAgent(backend=backend, tools=[])
    """

    backend_type = "mellea"

    def __init__(self):
        # Use MagicMock for config to avoid validation issues
        self.config = MagicMock()
        self.config.models = ModelConfig(
            reasoning="mock-model",
            writing="mock-model",
        )

    @classmethod
    def from_config(cls, config):
        """Create a MockBackend from config (ignores the config)."""
        return cls()

    def create_session(self, *, model: str, system_prompt: str | None = None) -> BaseSession:
        """Create a mock session."""
        return MockSession()

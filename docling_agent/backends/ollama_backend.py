from __future__ import annotations

from typing import Any

import httpx
from mellea.stdlib.requirements import Requirement
from typing_extensions import Self

from docling_agent.agent_models import should_log_llm_io
from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.logging import logger
from docling_agent.task_model import BackendConfig


class OllamaSession(BaseSession):
    """Direct HTTP session for Ollama API communication.

    Maintains conversation history and handles retries for Ollama's
    chat completion endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        base_url: str,
        timeout: float,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an Ollama session.

        Args:
            model: Ollama model identifier (e.g., "granite4.1:3b", "qwen3.5:latest").
            system_prompt: Optional system-level instructions.
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
            options: Additional Ollama-specific options (temperature, top_p, etc.).
        """
        self.model = model
        self.options = options or {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        """Send an instruction to Ollama and return the response.

        Args:
            prompt: The instruction or query to send.
            requirements: Not used by Ollama backend (no structured output validation).
            retry_budget: Maximum number of retry attempts on failure.

        Returns:
            The generated text response from Ollama.

        Raises:
            ValueError: If Ollama returns an empty response or request fails.
            httpx.HTTPStatusError: If the HTTP request fails.
        """
        _ = requirements
        attempts = max(1, retry_budget)
        user_message = {"role": "user", "content": prompt}
        self._messages.append(user_message)

        if should_log_llm_io():
            logger.debug(f"[LLM REQUEST]\n{prompt}")

        last_error: Exception | None = None
        try:
            for _attempt in range(attempts):
                try:
                    response = self._client.post(
                        "/api/chat",
                        json={
                            "model": self.model,
                            "messages": [message.copy() for message in self._messages],
                            "stream": False,
                            **self.options,
                        },
                    )
                    response.raise_for_status()
                    text = self._extract_text(response.json())
                    if not text.strip():
                        raise ValueError("Ollama returned an empty response.")
                    self._messages.append({"role": "assistant", "content": text})
                    if should_log_llm_io():
                        logger.debug(f"[LLM RESPONSE]\n{text}")
                    return text
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError("Ollama request failed without an explicit error.")
        except Exception:
            self._messages.pop()
            raise

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        """Extract conversation history for debugging.

        Returns:
            List of tuples (index, role, content_preview) with truncated content.
        """
        rows: list[tuple[int, str, str]] = []
        for idx, message in enumerate(self._messages):
            content = message["content"]
            if len(content) > 64:
                content = f"{content[0:32]} ... {content[-32:]}"
            rows.append((idx, message["role"], content))
        return rows

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        """Extract text content from Ollama API response.

        Args:
            payload: JSON response from Ollama API.

        Returns:
            The message content string.

        Raises:
            ValueError: If response structure is invalid or missing content.
        """
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response is missing the 'message' object.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response is missing string message content.")
        return content


class OllamaBackend(BaseBackend):
    """Backend for direct communication with Ollama API.

    Connects to a local or remote Ollama instance without additional
    abstraction layers. Suitable for running open-source models locally.

    Default connection: http://localhost:11434
    """

    backend_type = "ollama"

    def __init__(self, *, config: BackendConfig) -> None:
        """Initialize the Ollama backend with configuration.

        Args:
            config: Backend configuration including base URL and options.
        """
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.timeout = config.timeout or 120
        self.options = config.options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> Self:
        """Construct an Ollama backend from configuration.

        Args:
            config: Backend configuration.

        Returns:
            Initialized Ollama backend instance.
        """
        return cls(config=config)

    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        """Create a new Ollama session.

        Args:
            model: Ollama model identifier (e.g., "granite4.1:3b", "qwen3.5:latest").
            system_prompt: Optional system-level instructions.

        Returns:
            A new OllamaSession instance.
        """
        return OllamaSession(
            model=model,
            system_prompt=system_prompt,
            base_url=self.base_url,
            timeout=float(self.timeout),
            options=self.options,
        )
